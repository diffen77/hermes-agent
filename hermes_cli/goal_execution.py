"""Durable, fenced execution leases for persistent ``/goal`` loops."""

from __future__ import annotations

import time
import uuid
import logging
import os
import stat
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_state import SessionDB
from hermes_cli._goal_execution_registry import SUPERVISORS, SUPERVISORS_LOCK


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveryTurnAdmission:
    """Dispatch acknowledgement; acceptance is not turn completion."""

    accepted: bool
    completed: bool = False
    retry_after_seconds: float | None = None
    reason: str = ""
    completion: Future | None = None


@dataclass(frozen=True)
class RecoveryScanResult:
    """Truthful accounting for one durable recovery scan."""

    claimed: int = 0
    accepted: int = 0
    completed: int = 0
    failed: int = 0
    completions: tuple[Future, ...] = ()

    def __int__(self) -> int:
        # Compatibility for callers whose historical integer meant a
        # synchronously completed dispatch, never merely a claimed lease.
        return self.completed

    def __eq__(self, other):
        if isinstance(other, int):
            return self.completed == other
        if isinstance(other, RecoveryScanResult):
            return (
                self.claimed,
                self.accepted,
                self.completed,
                self.failed,
                self.completions,
            ) == (
                other.claimed,
                other.accepted,
                other.completed,
                other.failed,
                other.completions,
            )
        return NotImplemented


@dataclass(frozen=True)
class GoalProfileIdentity:
    """Canonical path and filesystem objects trusted by one recovery owner."""

    home: Path
    directory_device: int
    directory_inode: int
    database_device: int | None
    database_inode: int | None


def _capture_profile_identity(
    home: Path | str, *, allow_missing_db: bool = False
) -> GoalProfileIdentity:
    """Capture a canonical, non-aliased profile and optional state database."""

    profile = Path(home).expanduser().absolute()
    profile_lstat = profile.lstat()
    if stat.S_ISLNK(profile_lstat.st_mode) or not stat.S_ISDIR(profile_lstat.st_mode):
        raise RuntimeError(f"unsafe goal profile directory: {profile}")
    if profile.resolve() != profile:
        raise RuntimeError(f"non-canonical goal profile directory: {profile}")
    state_db = profile / "state.db"
    try:
        db_lstat = state_db.lstat()
    except FileNotFoundError:
        if allow_missing_db:
            return GoalProfileIdentity(
                profile,
                int(profile_lstat.st_dev),
                int(profile_lstat.st_ino),
                None,
                None,
            )
        raise
    if stat.S_ISLNK(db_lstat.st_mode) or not stat.S_ISREG(db_lstat.st_mode):
        raise RuntimeError(f"unsafe goal profile state.db: {state_db}")
    if int(db_lstat.st_nlink) != 1:
        raise RuntimeError(f"unsafe hardlinked goal profile state.db: {state_db}")
    if state_db.resolve().parent != profile:
        raise RuntimeError(f"goal profile state.db escapes exact profile: {state_db}")
    return GoalProfileIdentity(
        profile,
        int(profile_lstat.st_dev),
        int(profile_lstat.st_ino),
        int(db_lstat.st_dev),
        int(db_lstat.st_ino),
    )


def _exact_profile_db_identity(home: Path) -> tuple[int, int]:
    """Compatibility helper returning the exact safe database identity."""

    identity = _capture_profile_identity(home)
    assert identity.database_device is not None and identity.database_inode is not None
    return identity.database_device, identity.database_inode


def _verify_database_open_identity(identity: GoalProfileIdentity) -> None:
    """Minimize validation/open TOCTOU with a no-follow descriptor when available."""

    if identity.database_inode is None:
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(identity.home / "state.db", flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_dev) != identity.database_device
            or int(opened.st_ino) != identity.database_inode
        ):
            raise RuntimeError(
                f"goal profile state.db identity changed while opening: {identity.home}"
            )
    finally:
        os.close(descriptor)


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
    launch_identity = _capture_profile_identity(launch_home, allow_missing_db=True)
    launch = launch_identity.home
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
            child = child.absolute()
            resolved = child.resolve()
            resolved.relative_to(root)
            _capture_profile_identity(child)
            if resolved != launch:
                homes.append(resolved)
        except (OSError, RuntimeError, ValueError):
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
        self._launch_identity = _capture_profile_identity(
            launch_home, allow_missing_db=True
        )
        self.launch_home = self._launch_identity.home
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
        self._lifecycle = threading.Condition(self._lock)
        self._submissions_inflight = 0
        self._submitting_profiles: set[Path] = set()
        self._stopping = False
        self._closed = False
        self._coordinators: dict[Path, object] = {}
        self._bootstrapped: set[Path] = set()
        self._retry_after: dict[Path, float] = {}
        self.profile_errors: dict[Path, BaseException] = {}
        self.closed_coordinators: list[object] = []
        self._profile_futures: dict[Path, Future] = {}
        self._profile_identities: dict[Path, GoalProfileIdentity] = {
            self.launch_home: self._launch_identity
        }
        self._completion_futures: set[Future] = set()
        self._accounting_epoch = 0
        # Ordering must survive a successful outcome.  Terminal callbacks can
        # arrive out of order, so clearing this on success would let an older
        # failure restore stale error/backoff state.
        self._profile_terminal_epochs: dict[Path, int] = {}
        self.accepted = 0
        self.completed = 0
        self.failed = 0

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
            if self._closed:
                return False
            if self.thread is not None:
                if self.thread.is_alive():
                    return False
                if self._profile_futures:
                    return False
                self._finish_stopped_thread_locked(self.thread)
            self._stopping = False
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

    def _recover_profile(self, home: Path):
        with self._lock:
            expected_identity = self._profile_identities.get(home)
        if expected_identity is None:
            raise RuntimeError(f"missing expected goal profile identity: {home}")
        current_identity = _capture_profile_identity(
            home, allow_missing_db=(home == self.launch_home and expected_identity.database_inode is None)
        )
        if current_identity != expected_identity:
            raise RuntimeError(f"goal profile identity changed before recovery: {home}")
        with self._lock:
            coordinator = self._coordinators.get(home)
        if coordinator is None:
            created = self.coordinator_factory(home)
            try:
                opened_identity = _capture_profile_identity(
                    home, allow_missing_db=(expected_identity.database_inode is None)
                )
                if (
                    opened_identity.home != expected_identity.home
                    or opened_identity.directory_device != expected_identity.directory_device
                    or opened_identity.directory_inode != expected_identity.directory_inode
                    or (
                        expected_identity.database_inode is not None
                        and opened_identity != expected_identity
                    )
                ):
                    raise RuntimeError(f"goal profile identity changed during open: {home}")
            except BaseException:
                try:
                    created.close()
                finally:
                    raise
            with self._lock:
                self._profile_identities[home] = opened_identity
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
        expected_identity = self._profile_identities[home]
        if _capture_profile_identity(
            home, allow_missing_db=(expected_identity.database_inode is None)
        ) != expected_identity:
            raise RuntimeError(f"goal profile identity changed before recovery dispatch: {home}")
        if self.recover_callback is None:
            return coordinator.recover_once(self.dispatch)
        return self.recover_callback(coordinator)

    def _next_accounting_epoch(self) -> int:
        with self._lock:
            self._accounting_epoch += 1
            return self._accounting_epoch

    def _record_profile_failure(self, home: Path, epoch: int, exc: BaseException, now: float) -> None:
        with self._lock:
            if epoch < self._profile_terminal_epochs.get(home, -1):
                return
            self._profile_terminal_epochs[home] = epoch
            self.profile_errors[home] = exc
            self._retry_after[home] = now + self.error_backoff_seconds

    def _record_profile_success(self, home: Path, epoch: int) -> None:
        with self._lock:
            if self._profile_terminal_epochs.get(home, -1) > epoch:
                return
            self._profile_terminal_epochs[home] = epoch
            self.profile_errors.pop(home, None)
            self._retry_after.pop(home, None)

    def _track_terminal_completion(self, home: Path, future: Future, epoch: int) -> None:
        with self._lock:
            self._completion_futures.add(future)

        def account(done: Future) -> None:
            now = float(self.clock())
            try:
                done.result()
            except BaseException as exc:
                with self._lock:
                    self.failed += 1
                self._record_profile_failure(home, epoch, exc, now)
            else:
                with self._lock:
                    self.completed += 1
                self._record_profile_success(home, epoch)
            finally:
                with self._lock:
                    self._completion_futures.discard(done)

        future.add_done_callback(account)

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
            epoch = self._next_accounting_epoch()
            try:
                result = future.result()
                if isinstance(result, RecoveryScanResult):
                    recovered += result.completed
                    with self._lock:
                        self.accepted += result.accepted
                        self.completed += result.completed
                        self.failed += result.failed
                    if result.failed:
                        self._record_profile_failure(
                            home,
                            epoch,
                            RuntimeError(f"{result.failed} recovery turn dispatch(es) failed"),
                            now,
                        )
                    elif result.completed or not result.accepted:
                        self._record_profile_success(home, epoch)
                    for completion in result.completions:
                        completion_epoch = self._next_accounting_epoch()
                        self._track_terminal_completion(home, completion, completion_epoch)
                else:
                    recovered += int(result)
                    self._record_profile_success(home, epoch)
            except Exception as exc:
                self._record_profile_failure(home, epoch, exc, now)
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
            try:
                discovered_identity = _capture_profile_identity(
                    home, allow_missing_db=(home == self.launch_home)
                )
            except BaseException as exc:
                epoch = self._next_accounting_epoch()
                self._record_profile_failure(home, epoch, exc, now)
                continue
            with self._lock:
                pinned_identity = self._profile_identities.get(home)
                renamed_from = next(
                    (
                        pinned_home
                        for pinned_home, identity in self._profile_identities.items()
                        if pinned_home != home
                        and pinned_home in self._coordinators
                        and (
                            identity.directory_device,
                            identity.directory_inode,
                            identity.database_device,
                            identity.database_inode,
                        )
                        == (
                            discovered_identity.directory_device,
                            discovered_identity.directory_inode,
                            discovered_identity.database_device,
                            discovered_identity.database_inode,
                        )
                    ),
                    None,
                )
                if renamed_from is not None:
                    self._accounting_epoch += 1
                    epoch = self._accounting_epoch
                    exc = RuntimeError(
                        f"goal profile identity renamed from {renamed_from} to {home}"
                    )
                    for unsafe_home in (renamed_from, home):
                        self._profile_terminal_epochs[unsafe_home] = epoch
                        self.profile_errors[unsafe_home] = exc
                        self._retry_after[unsafe_home] = now + self.error_backoff_seconds
                    continue
                if (
                    home in self._coordinators
                    and pinned_identity is not None
                    and discovered_identity != pinned_identity
                ):
                    self._accounting_epoch += 1
                    epoch = self._accounting_epoch
                    exc = RuntimeError(
                        f"goal profile identity changed after coordinator open: {home}"
                    )
                    self._profile_terminal_epochs[home] = epoch
                    self.profile_errors[home] = exc
                    self._retry_after[home] = now + self.error_backoff_seconds
                    continue
                if (
                    self._closed
                    or self._stopping
                    or home in self._profile_futures
                    or home in self._submitting_profiles
                ):
                    continue
                executor = self._ensure_executor_locked()
                self._profile_identities[home] = discovered_identity
                self._submitting_profiles.add(home)
                self._submissions_inflight += 1
            future = None
            try:
                # Calling a custom executor under ``_lock`` deadlocks when it
                # runs callbacks inline, since recovery itself needs the lock.
                # The admission count prevents shutdown from detaching the
                # executor while this submission is outside the lock.
                future = executor.submit(self._recover_profile, home)
            except RuntimeError as exc:
                # Contain an unexpected executor lifecycle race instead of
                # crashing the scheduler thread with submit-after-shutdown.
                self.profile_errors[home] = exc
            finally:
                with self._lifecycle:
                    self._submitting_profiles.discard(home)
                    self._submissions_inflight -= 1
                    if future is not None:
                        self._profile_futures[home] = future
                    self._lifecycle.notify_all()
            if future is not None:
                submitted.append(future)
        if wait and submitted:
            wait_futures(submitted)
            recovered += self._collect_completed(now=now)
        return recovered

    def stop(self, *, timeout: float | None = 5.0) -> bool:
        if not self.stop_workers(timeout=timeout):
            return False
        with self._lock:
            coordinators = list(self._coordinators.values())
            self._coordinators.clear()
            self._bootstrapped.clear()
            self._retry_after.clear()
        for coordinator in coordinators:
            try:
                coordinator.close()
            except Exception:
                logger.debug("goal coordinator close failed", exc_info=True)
        self.closed_coordinators.extend(coordinators)
        return True

    def stop_workers(self, *, timeout: float | None = 5.0) -> bool:
        """Stop and join scheduler/profile workers without closing profile DBs."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._lifecycle:
            thread = self.thread
            if thread is None:
                return False
            self._stopping = True
            self.stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return False
        with self._lifecycle:
            while self._submissions_inflight:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            futures = list(self._profile_futures.values())
        if futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            _done, pending = wait_futures(futures, timeout=remaining)
            if pending:
                return False
            self._collect_completed(now=float(self.clock()))
        with self._lock:
            executor = self._executor
            self._executor = None
            self.thread = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        return True

    def detach_coordinators(self) -> list[object]:
        """Drop coordinator references after an external owner closed them."""
        with self._lock:
            coordinators = list(self._coordinators.values())
            self._coordinators.clear()
            self._bootstrapped.clear()
            self._retry_after.clear()
        return coordinators

    def _legacy_stop(self, *, timeout: float | None = 5.0) -> bool:
        """Compatibility shim retained for reload-stable instances."""
        if not self.stop(timeout=timeout):
            return False
        return True

    def close(self, *, timeout: float | None = 5.0) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._lifecycle:
            self._closed = True
            self._stopping = True
            self.stop_event.set()
            thread = self.thread
        if thread is not None:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
            if thread.is_alive():
                return False
        with self._lifecycle:
            while self._submissions_inflight:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._lifecycle.wait(remaining)
            futures = list(self._profile_futures.values())
        if futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            _done, pending = wait_futures(futures, timeout=remaining)
            if pending:
                return False
            self._collect_completed(now=float(self.clock()))
        with self._lock:
            coordinators = list(self._coordinators.values())
            self._coordinators.clear()
            self._bootstrapped.clear()
            executor = self._executor
            self._executor = None
            self.thread = None
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
        before_open = _capture_profile_identity(home, allow_missing_db=True)
        self.home = before_open.home
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
        _verify_database_open_identity(before_open)
        self.db = SessionDB(self.home / "state.db")
        try:
            self.profile_identity = _capture_profile_identity(self.home)
            if (
                self.profile_identity.home != before_open.home
                or self.profile_identity.directory_device != before_open.directory_device
                or self.profile_identity.directory_inode != before_open.directory_inode
                or (
                    before_open.database_inode is not None
                    and self.profile_identity != before_open
                )
            ):
                raise RuntimeError(f"goal profile identity changed during database open: {self.home}")
        except BaseException:
            self.db.close()
            raise

    def validate_profile_identity(self) -> None:
        if _capture_profile_identity(self.home) != self.profile_identity:
            raise RuntimeError(f"goal profile identity changed: {self.home}")

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

    def recover_once(
        self, dispatch: Callable[[GoalExecutionRecord, object], object]
    ) -> RecoveryScanResult:
        """Claim and dispatch indexed due goals; never scan historical state."""
        from hermes_cli.goals import GoalState, _pid_alive, _session_waiting

        now = float(self.clock())
        claimed = accepted = completed = failed = 0
        completions = []
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
            self.validate_profile_identity()
            record = self.claim(session_id, generation)
            if record is None:
                continue
            claimed += 1
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
                try:
                    self.validate_profile_identity()
                except RuntimeError as exc:
                    self.schedule(
                        session_id,
                        generation,
                        record.claim_token or "",
                        next_run_at=now + self.wait_poll_seconds,
                        reason=f"profile security error: {exc}",
                    )
                    raise
                admission = dispatch(record, state)
                if isinstance(admission, RecoveryTurnAdmission):
                    if admission.accepted:
                        accepted += 1
                    if admission.completed:
                        completed += 1
                    if admission.completion is not None:
                        completions.append(admission.completion)
                    if not admission.accepted and admission.retry_after_seconds is not None:
                        self.schedule(
                            session_id,
                            generation,
                            record.claim_token or "",
                            next_run_at=now
                            + max(0.01, float(admission.retry_after_seconds)),
                            reason=admission.reason or "recovery turn admission deferred",
                        )
                else:
                    # Existing synchronous dispatch callbacks complete when
                    # they return. This preserves their truthful old contract.
                    accepted += 1
                    completed += 1
            except Exception as exc:
                if "profile identity changed" in str(exc):
                    raise
                failed += 1
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
        return RecoveryScanResult(
            claimed=claimed,
            accepted=accepted,
            completed=completed,
            failed=failed,
            completions=tuple(completions),
        )
