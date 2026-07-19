"""Durable, fenced execution leases for persistent ``/goal`` loops."""

from __future__ import annotations

import time
import uuid
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_state import SessionDB
from hermes_cli._goal_execution_registry import SUPERVISORS, SUPERVISORS_LOCK


logger = logging.getLogger(__name__)


def get_goal_recovery_supervisor(key: str, factory):
    """Return one process-owned supervisor, surviving this module's reload."""
    with SUPERVISORS_LOCK:
        supervisor = SUPERVISORS.get(key)
        if supervisor is None:
            supervisor = factory()
            SUPERVISORS[key] = supervisor
        return supervisor


def clear_goal_recovery_supervisor(key: str, supervisor: object) -> bool:
    with SUPERVISORS_LOCK:
        if SUPERVISORS.get(key) is not supervisor:
            return False
        SUPERVISORS.pop(key, None)
        return True


def discover_goal_profile_homes(
    launch_home: Path | str, *, profiles_root: Path | str | None = None
) -> list[Path]:
    """Return the launch home plus existing canonical profile state homes.

    Discovery is deliberately restricted to direct children of the
    HOME-anchored Hermes profiles directory.  A child is relevant only when it
    already owns ``state.db``; no unrelated trees or arbitrary files are
    inspected.
    """
    launch = Path(launch_home).expanduser().resolve()
    homes = [launch]
    if profiles_root is None:
        import hermes_constants

        root = hermes_constants.get_default_hermes_root() / "profiles"
    else:
        root = Path(profiles_root)
    try:
        root = root.expanduser().resolve()
    except OSError:
        return homes
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        children = []
    for child in children:
        try:
            if child.is_symlink():
                continue
            resolved = child.resolve()
            resolved.relative_to(root)
            state_db = child / "state.db"
            state_db.resolve().relative_to(root)
            if child.is_dir() and state_db.is_file() and resolved != launch:
                homes.append(resolved)
        except (OSError, ValueError):
            continue
    return homes


class GoalRecoverySupervisor:
    """Owned singleton-style lifecycle for multi-profile goal recovery."""

    def __init__(
        self,
        launch_home: Path | str,
        *,
        profiles_root: Path | str | None = None,
        coordinator_factory=None,
        dispatch=None,
        recover_callback=None,
        clock: Callable[[], float] = time.time,
        stop_event: threading.Event | None = None,
        thread_factory=threading.Thread,
        scan_seconds: float = 1.0,
        error_backoff_seconds: float = 5.0,
        recovery_workers: int = 4,
        executor_factory=ThreadPoolExecutor,
    ) -> None:
        self.launch_home = Path(launch_home).expanduser().resolve()
        self.profiles_root = profiles_root
        self.coordinator_factory = coordinator_factory or GoalExecutionCoordinator
        self.dispatch = dispatch or (lambda *_args: None)
        self.recover_callback = recover_callback
        self.clock = clock
        self.stop_event = stop_event or threading.Event()
        self.thread_factory = thread_factory
        self.scan_seconds = max(0.01, float(scan_seconds))
        self.error_backoff_seconds = max(self.scan_seconds, float(error_backoff_seconds))
        self.recovery_workers = max(2, int(recovery_workers))
        self.executor_factory = executor_factory
        self.thread = None
        self._executor = None
        self._lock = threading.Lock()
        self._coordinators: dict[Path, object] = {}
        self._bootstrapped: set[Path] = set()
        self._retry_after: dict[Path, float] = {}
        self.profile_errors: dict[Path, BaseException] = {}
        self.closed_coordinators: list[object] = []
        self._profile_futures: dict[Path, Future] = {}

    def _finish_stopped_thread_locked(self, thread) -> None:
        """Detach one exited owned thread and close its coordinators once."""
        if self.thread is not thread or thread.is_alive():
            return
        self.thread = None
        coordinators = list(self._coordinators.values())
        self._coordinators.clear()
        self._bootstrapped.clear()
        self._retry_after.clear()
        self.closed_coordinators.extend(coordinators)
        for coordinator in coordinators:
            try:
                coordinator.close()
            except Exception:
                logger.debug("goal coordinator close failed", exc_info=True)

    def start(self) -> bool:
        self._collect_completed(now=float(self.clock()))
        with self._lock:
            if self.thread is not None:
                if self.thread.is_alive():
                    return False
                if self._profile_futures:
                    return False
                self._finish_stopped_thread_locked(self.thread)
            self.stop_event.clear()
            self._ensure_executor_locked()
            self.thread = self.thread_factory(
                target=self.run, name="goal-recovery", daemon=True
            )
            self.thread.start()
            return True

    def run(self) -> None:
        while not self.stop_event.wait(self.scan_seconds):
            self.scan_once(wait=False)

    def _ensure_executor_locked(self):
        if self._executor is None:
            self._executor = self.executor_factory(
                max_workers=self.recovery_workers, thread_name_prefix="goal-profile"
            )
        return self._executor

    def _recover_profile(self, home: Path) -> int:
        with self._lock:
            coordinator = self._coordinators.get(home)
        if coordinator is None:
            created = self.coordinator_factory(home)
            with self._lock:
                coordinator = self._coordinators.setdefault(home, created)
            if coordinator is not created:
                created.close()
        with self._lock:
            bootstrapped = home in self._bootstrapped
        if not bootstrapped:
            page = getattr(coordinator, "bootstrap_reconcile_page", None)
            if page is None:
                coordinator.bootstrap_reconcile()
                complete = True
            else:
                _seeded, complete = page()
            if complete:
                with self._lock:
                    self._bootstrapped.add(home)
        if self.recover_callback is None:
            return int(coordinator.recover_once(self.dispatch))
        return int(self.recover_callback(coordinator))

    def _collect_completed(self, *, now: float) -> int:
        recovered = 0
        with self._lock:
            completed = [
                (home, future)
                for home, future in self._profile_futures.items()
                if future.done()
            ]
            for home, _future in completed:
                self._profile_futures.pop(home, None)
        for home, future in completed:
            try:
                recovered += int(future.result())
                self.profile_errors.pop(home, None)
                self._retry_after.pop(home, None)
            except Exception as exc:
                self.profile_errors[home] = exc
                self._retry_after[home] = now + self.error_backoff_seconds
                logger.warning("goal recovery failed for profile %s: %s", home, exc)
        return recovered

    def scan_once(self, *, wait: bool = True) -> int:
        now = float(self.clock())
        recovered = self._collect_completed(now=now)
        submitted = []
        for home in discover_goal_profile_homes(
            self.launch_home, profiles_root=self.profiles_root
        ):
            if self._retry_after.get(home, 0.0) > now:
                continue
            with self._lock:
                if home in self._profile_futures:
                    continue
                future = self._ensure_executor_locked().submit(self._recover_profile, home)
                self._profile_futures[home] = future
                submitted.append(future)
        if wait and submitted:
            wait_futures(submitted)
            recovered += self._collect_completed(now=now)
        return recovered

    def stop(self, *, timeout: float | None = 5.0) -> bool:
        with self._lock:
            thread = self.thread
            if thread is None:
                return False
            self.stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        if not self.close(timeout=timeout):
            return False
        with self._lock:
            self._finish_stopped_thread_locked(thread)
        return True

    def close(self, *, timeout: float | None = 5.0) -> bool:
        with self._lock:
            futures = list(self._profile_futures.values())
        if futures:
            _done, pending = wait_futures(futures, timeout=timeout)
            if pending:
                return False
            self._collect_completed(now=float(self.clock()))
        with self._lock:
            coordinators = list(self._coordinators.values())
            self._coordinators.clear()
            self._bootstrapped.clear()
            executor = self._executor
            self._executor = None
        for coordinator in coordinators:
            try:
                coordinator.close()
            except Exception:
                logger.debug("goal coordinator close failed", exc_info=True)
        self.closed_coordinators.extend(coordinators)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        return True


@dataclass(frozen=True)
class GoalExecutionRecord:
    session_id: str
    generation: int
    owner_id: Optional[str]
    claim_token: Optional[str]
    lease_expires_at: float
    heartbeat_at: float
    next_run_at: float
    attempt: int
    last_error: Optional[str]


class GoalExecutionCoordinator:
    """Profile-scoped lease coordinator over canonical :class:`SessionDB` APIs."""

    def __init__(
        self,
        home: Path | str,
        *,
        owner_id: str | None = None,
        clock: Callable[[], float] = time.time,
        lease_seconds: float = 30.0,
        wait_poll_seconds: float = 1.0,
        max_recovery_attempts: int = 5,
        recovery_batch_limit: int = 100,
    ) -> None:
        self.home = Path(home)
        self.owner_id = owner_id or f"backend-{uuid.uuid4().hex}"
        self.clock = clock
        self.lease_seconds = max(1.0, float(lease_seconds))
        self.wait_poll_seconds = max(0.1, float(wait_poll_seconds))
        self.max_recovery_attempts = max(1, int(max_recovery_attempts))
        if isinstance(recovery_batch_limit, bool):
            raise ValueError("recovery_batch_limit must be a positive integer")
        try:
            self.recovery_batch_limit = int(recovery_batch_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("recovery_batch_limit must be a positive integer") from exc
        if self.recovery_batch_limit < 1:
            raise ValueError("recovery_batch_limit must be a positive integer")
        self.db = SessionDB(self.home / "state.db")

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _record(row) -> GoalExecutionRecord:
        return GoalExecutionRecord(
            session_id=str(row["session_id"]),
            generation=int(row["generation"]),
            owner_id=row["owner_id"],
            claim_token=row["claim_token"],
            lease_expires_at=float(row["lease_expires_at"] or 0),
            heartbeat_at=float(row["heartbeat_at"] or 0),
            next_run_at=float(row["next_run_at"] or 0),
            attempt=int(row["attempt"] or 0),
            last_error=row["last_error"],
        )

    def get(self, session_id: str) -> GoalExecutionRecord | None:
        row = self.db.get_goal_execution_lease(session_id)
        return self._record(row) if row else None

    def claim(
        self, session_id: str, generation: int, *, lease_seconds: float | None = None
    ) -> GoalExecutionRecord | None:
        now = float(self.clock())
        expires = now + max(1.0, float(lease_seconds or self.lease_seconds))
        row = self.db.claim_goal_execution_lease(
            session_id,
            int(generation),
            self.owner_id,
            uuid.uuid4().hex,
            now=now,
            expires_at=expires,
        )
        return self._record(row) if row else None

    def heartbeat(self, session_id: str, generation: int, claim_token: str) -> bool:
        now = float(self.clock())
        return self.db.heartbeat_goal_execution_lease(
            session_id,
            int(generation),
            self.owner_id,
            claim_token,
            now=now,
            expires_at=now + self.lease_seconds,
        )

    def schedule(
        self,
        session_id: str,
        generation: int,
        claim_token: str,
        *,
        next_run_at: float,
        reason: str = "",
        increment_attempt: bool = False,
    ) -> bool:
        return self.db.schedule_goal_execution_lease(
            session_id,
            int(generation),
            self.owner_id,
            claim_token,
            next_run_at=float(next_run_at),
            reason=reason,
            increment_attempt=increment_attempt,
            now=float(self.clock()),
        )

    def release(self, session_id: str, generation: int, claim_token: str) -> bool:
        return self.schedule(session_id, generation, claim_token, next_run_at=0.0)

    def invalidate(self, session_id: str, generation: int) -> bool:
        return self.db.invalidate_goal_execution_lease(
            session_id, int(generation), now=float(self.clock())
        )

    def due_records(self, *, now: float | None = None) -> list[GoalExecutionRecord]:
        current = float(self.clock() if now is None else now)
        return [
            self._record(row)
            for row in self.db.due_goal_execution_leases(
                now=current, limit=self.recovery_batch_limit
            )
        ]

    def bootstrap_reconcile(self, *, limit: int = 1000) -> int:
        """Seed missing leases from one bounded, SQLite-filtered active scan."""
        seeded, _complete = self.bootstrap_reconcile_page(limit=limit)
        return seeded

    def bootstrap_reconcile_page(self, *, limit: int = 1000) -> tuple[int, bool]:
        """Reconcile one bounded page; already-reconciled rows are SQL-filtered.

        This makes a restart between pages safe: the next process begins at the
        first still-unreconciled active goal rather than replaying page one.
        """
        from hermes_cli.goals import GoalState

        now = float(self.clock())
        seeded = 0
        rows = self.db.unreconciled_active_goal_state_rows(limit=limit)
        for row in rows:
            try:
                state = GoalState.from_json(row["value"])
            except Exception:
                continue
            session_id = str(row["key"])[len("goal:") :]
            next_run_at = float(state.waiting_until or now)
            if self.db.seed_goal_execution_lease(
                session_id,
                int(state.generation),
                next_run_at=next_run_at,
                now=now,
            ):
                seeded += 1
        return seeded, len(rows) < limit

    def pause_after_failure(
        self, session_id: str, generation: int, claim_token: str, *, reason: str
    ) -> bool:
        return self.db.pause_goal_execution_after_failure(
            session_id,
            int(generation),
            self.owner_id,
            claim_token,
            reason=reason,
            now=float(self.clock()),
        )

    def recover_once(self, dispatch: Callable[[GoalExecutionRecord, object], None]) -> int:
        """Claim and dispatch indexed due goals; never scan historical state."""
        from hermes_cli.goals import GoalState, _pid_alive, _session_waiting

        now = float(self.clock())
        claimed = 0
        for due in self.due_records(now=now):
            try:
                raw = self.db.get_meta(f"goal:{due.session_id}")
                state = GoalState.from_json(raw) if raw else None
            except Exception:
                self.invalidate(due.session_id, due.generation)
                continue
            if state is None or state.status != "active":
                self.invalidate(due.session_id, due.generation)
                continue
            session_id = due.session_id
            generation = int(state.generation)
            if generation != due.generation:
                # Fence cleanup to the generation represented by the stale row.
                # Passing the freshly-read state generation could revoke an
                # equal-generation claim established after this scan snapshot.
                self.invalidate(session_id, due.generation)
                continue
            record = self.claim(session_id, generation)
            if record is None:
                continue
            waiting_until = float(state.waiting_until or 0.0)
            waiting_live = bool(
                (state.waiting_on_pid and _pid_alive(state.waiting_on_pid))
                or (state.waiting_on_session and _session_waiting(state.waiting_on_session))
            )
            if waiting_until > now or waiting_live:
                self.schedule(
                    session_id,
                    generation,
                    record.claim_token or "",
                    next_run_at=(
                        waiting_until if waiting_until > now else now + self.wait_poll_seconds
                    ),
                    reason=str(state.waiting_reason or "goal wait"),
                )
                continue
            try:
                dispatch(record, state)
                claimed += 1
            except Exception as exc:
                attempt = record.attempt + 1
                failure = f"continuation dispatch failed: {type(exc).__name__}: {exc}"
                if attempt >= self.max_recovery_attempts:
                    self.pause_after_failure(
                        session_id,
                        generation,
                        record.claim_token or "",
                        reason=f"recoverable blocker after {attempt} attempts — {failure}",
                    )
                else:
                    delay = min(60.0, float(2 ** min(attempt, 6)))
                    self.schedule(
                        session_id,
                        generation,
                        record.claim_token or "",
                        next_run_at=now + delay,
                        reason=failure,
                        increment_attempt=True,
                    )
        return claimed
