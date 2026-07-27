#!/usr/bin/env python3
"""One parent-owned, lease-fenced fixture-quarantine recovery tick.

This is intentionally a narrow cron boundary around
``hermes_cli.kanban_fleet.apply_safe_recovery``.  It is not a worker reaper,
service, or general recovery controller.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, cast

# Direct ``python scripts/...`` execution puts scripts/, not the repository, on
# sys.path.  Resolve the checked-out source without consulting the shell.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli.kanban_fleet import apply_safe_recovery

_SHA_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_GENERATION_RE = re.compile(r"[0-9a-zA-Z_.:-]{1,128}\Z")
_STATE_VERSION = 1
_STATE_MAX_BYTES = 4096
_CONTROL_DIRECTORY = "factory-recovery-controller"
_STATE_FILE = "heartbeat.json"
_LEASE_FILE = "lease.lock"

ApplyFn = Callable[..., dict[str, object]]


class ControllerStateError(RuntimeError):
    """The durable controller state is malformed or unsafe to trust."""


def _is_delegated_child() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return is_delegated_child_process_context()
    except Exception:
        return bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))


def _source_sha(expected: str | None, *, repo_root: Path | None = None) -> str:
    source_root = _REPO_ROOT if repo_root is None else repo_root
    if expected is not None and not _SHA_RE.fullmatch(expected):
        raise ValueError("invalid_expected_source_sha")
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=source_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    candidate = completed.stdout.strip()
    if completed.returncode != 0 or not _SHA_RE.fullmatch(candidate):
        raise ValueError("source_sha_unavailable")
    actual = candidate.lower()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if dirty.returncode != 0:
        raise ValueError("source_status_unavailable")
    if dirty.stdout:
        raise ValueError("source_tree_dirty")
    if expected is not None and expected.lower() != actual:
        raise ValueError("expected_source_sha_mismatch")
    return actual


def _control_paths(root: Path) -> tuple[Path, Path, Path]:
    directory = root / "kanban" / _CONTROL_DIRECTORY
    return directory, directory / _LEASE_FILE, directory / _STATE_FILE


def _open_lease(directory: Path, lease_path: Path) -> int:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lease_path, flags, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ControllerStateError("unsafe_lease_file")
        # Contenders must not even chmod the shared inode: acquire first so a
        # BUSY tick has no filesystem mutation beyond opening the descriptor.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(fd, 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _bounded_generation(value: object) -> str | None:
    if value is None:
        return None
    candidate = str(value)
    return candidate if _GENERATION_RE.fullmatch(candidate) else None


def _validate_state(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ControllerStateError("malformed_state")
    required = {
        "version",
        "candidate_source_sha",
        "started_at",
        "completed_at",
        "outcome",
        "observation_generation",
        "mutation_count",
        "consecutive_failures",
        "failure_fingerprint",
        "next_eligible_at",
    }
    if set(raw) != required:
        raise ControllerStateError("malformed_state")
    if raw["version"] != _STATE_VERSION:
        raise ControllerStateError("malformed_state")
    if not isinstance(raw["candidate_source_sha"], str) or not _SHA_RE.fullmatch(
        cast(str, raw["candidate_source_sha"])
    ):
        raise ControllerStateError("malformed_state")
    for key in ("started_at", "mutation_count", "consecutive_failures", "next_eligible_at"):
        if type(raw[key]) is not int or cast(int, raw[key]) < 0:
            raise ControllerStateError("malformed_state")
    if raw["completed_at"] is not None and (
        type(raw["completed_at"]) is not int or cast(int, raw["completed_at"]) < 0
    ):
        raise ControllerStateError("malformed_state")
    if raw["outcome"] not in {
        "RUNNING",
        "APPLIED",
        "NO_OP",
        "NOT_VERIFIED",
        "ERROR",
    }:
        raise ControllerStateError("malformed_state")
    generation = raw["observation_generation"]
    if generation is not None and (
        not isinstance(generation, str) or not _GENERATION_RE.fullmatch(generation)
    ):
        raise ControllerStateError("malformed_state")
    if cast(int, raw["mutation_count"]) > 1:
        raise ControllerStateError("malformed_state")
    if raw["failure_fingerprint"] not in {None, "malformed_state", "safe_recovery_failed"}:
        raise ControllerStateError("malformed_state")
    return cast(dict[str, object], raw)


def _read_state(path: Path) -> dict[str, object] | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ControllerStateError("state_read_failed") from exc
    if len(data) > _STATE_MAX_BYTES:
        raise ControllerStateError("malformed_state")
    try:
        return _validate_state(json.loads(data))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerStateError("malformed_state") from exc


def _write_state(path: Path, state: Mapping[str, object]) -> None:
    safe = _validate_state(dict(state))
    data = (json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(data) > _STATE_MAX_BYTES:
        raise ControllerStateError("state_too_large")
    fd, temporary = tempfile.mkstemp(prefix=".heartbeat.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _state(
    *,
    source_sha: str,
    started_at: int,
    completed_at: int | None,
    outcome: str,
    generation: str | None,
    mutations: int,
    failures: int,
    failure_fingerprint: str | None,
    next_eligible_at: int,
) -> dict[str, object]:
    return {
        "version": _STATE_VERSION,
        "candidate_source_sha": source_sha,
        "started_at": started_at,
        "completed_at": completed_at,
        "outcome": outcome,
        "observation_generation": generation,
        "mutation_count": mutations,
        "consecutive_failures": failures,
        "failure_fingerprint": failure_fingerprint,
        "next_eligible_at": next_eligible_at,
    }


def _failure_delay(failures: int, base: int, maximum: int) -> int:
    return min(maximum, base * (2 ** min(max(0, failures - 1), 30)))


def run_tick(
    *,
    root: Path,
    expected_source_sha: str | None = None,
    stale_after_seconds: int = 3600,
    limit: int = 20,
    backoff_base_seconds: int = 60,
    backoff_max_seconds: int = 3600,
    force: bool = False,
    now: int | None = None,
    apply_fn: ApplyFn = apply_safe_recovery,
    source_repo_root: Path | None = None,
) -> dict[str, object]:
    """Run one lease-fenced recovery tick and return a bounded public result."""
    # This check must precede root normalization, mkdir, git lookup, and lock open.
    if _is_delegated_child():
        return {"outcome": "DENIED", "reason": "delegated_child_context", "mutations": 0}
    if backoff_base_seconds < 1 or backoff_max_seconds < backoff_base_seconds:
        return {"outcome": "ERROR", "reason": "invalid_backoff", "mutations": 0}
    try:
        source_sha = _source_sha(expected_source_sha, repo_root=source_repo_root)
    except ValueError as exc:
        return {"outcome": "ERROR", "reason": str(exc), "mutations": 0}

    root = Path(os.path.abspath(root.expanduser()))
    directory, lease_path, state_path = _control_paths(root)
    try:
        lease_fd = _open_lease(directory, lease_path)
    except BlockingIOError:
        return {"outcome": "BUSY", "mutations": 0}
    except (OSError, ControllerStateError):
        return {"outcome": "ERROR", "reason": "lease_unavailable", "mutations": 0}

    started = int(time.time()) if now is None else int(now)

    def completed_time() -> int:
        return int(time.time()) if now is None else started

    try:
        try:
            previous = _read_state(state_path)
        except ControllerStateError:
            replacement = _state(
                source_sha=source_sha,
                started_at=started,
                completed_at=completed_time(),
                outcome="ERROR",
                generation=None,
                mutations=0,
                failures=1,
                failure_fingerprint="malformed_state",
                next_eligible_at=0,
            )
            try:
                _write_state(state_path, replacement)
            except (OSError, ControllerStateError):
                pass
            return {"outcome": "ERROR", "reason": "malformed_state", "mutations": 0}

        if (
            previous is not None
            and not force
            and previous["outcome"] == "ERROR"
            and previous["failure_fingerprint"] is not None
            and previous["candidate_source_sha"] == source_sha
            and cast(int, previous["next_eligible_at"]) > started
        ):
            return {
                "outcome": "BACKOFF",
                "mutations": 0,
                "retry_at": cast(int, previous["next_eligible_at"]),
            }

        prior_failures = 0 if previous is None else cast(int, previous["consecutive_failures"])
        _write_state(
            state_path,
            _state(
                source_sha=source_sha,
                started_at=started,
                completed_at=None,
                outcome="RUNNING",
                generation=None,
                mutations=0,
                failures=prior_failures,
                failure_fingerprint=None,
                next_eligible_at=0,
            ),
        )
        try:
            applied = apply_fn(
                root=root,
                now=started,
                stale_after_seconds=stale_after_seconds,
                limit=limit,
            )
            verdict = str(applied.get("verdict", "NOT_VERIFIED"))
            mutations = applied.get("mutations", 0)
            if type(mutations) is not int or not 0 <= cast(int, mutations) <= 1:
                raise ControllerStateError("invalid_apply_result")
            mutation_count = cast(int, mutations)
            generation = _bounded_generation(applied.get("observation_generation"))
            boards = applied.get("boards", [])
            receipt_recovered = isinstance(boards, list) and any(
                isinstance(board, dict) and board.get("outcome") == "receipt_recovered"
                for board in boards
            )
            if verdict == "APPLIED":
                outcome = "APPLIED" if mutation_count or receipt_recovered else "NO_OP"
                failures = 0
                failure_fingerprint = None
                next_eligible = 0
            elif verdict == "NOT_VERIFIED":
                outcome = "NOT_VERIFIED"
                failures = 0
                failure_fingerprint = None
                next_eligible = 0
            else:
                raise ControllerStateError("invalid_apply_result")
            _write_state(
                state_path,
                _state(
                    source_sha=source_sha,
                    started_at=started,
                    completed_at=completed_time(),
                    outcome=outcome,
                    generation=generation,
                    mutations=mutation_count,
                    failures=failures,
                    failure_fingerprint=failure_fingerprint,
                    next_eligible_at=next_eligible,
                ),
            )
            public: dict[str, object] = {"outcome": outcome, "mutations": mutation_count}
            if generation is not None:
                public["observation_generation"] = generation
            if receipt_recovered:
                public["receipt_recovered"] = True
            if outcome == "NOT_VERIFIED":
                public["reason"] = "safe_recovery_not_verified"
            return public
        except Exception:
            failure_fingerprint = "safe_recovery_failed"
            repeated = (
                previous is not None
                and previous["outcome"] == "ERROR"
                and previous["failure_fingerprint"] == failure_fingerprint
                and previous["candidate_source_sha"] == source_sha
            )
            failures = prior_failures + 1 if repeated else 1
            next_eligible = (
                started + _failure_delay(failures - 1, backoff_base_seconds, backoff_max_seconds)
                if repeated
                else 0
            )
            try:
                _write_state(
                    state_path,
                    _state(
                        source_sha=source_sha,
                        started_at=started,
                        completed_at=completed_time(),
                        outcome="ERROR",
                        generation=None,
                        mutations=0,
                        failures=failures,
                        failure_fingerprint=failure_fingerprint,
                        next_eligible_at=next_eligible,
                    ),
                )
            except (OSError, ControllerStateError):
                pass
            return {"outcome": "ERROR", "reason": "safe_recovery_failed", "mutations": 0}
    except (OSError, ControllerStateError):
        return {"outcome": "ERROR", "reason": "state_persistence_failed", "mutations": 0}
    finally:
        os.close(lease_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one parent-owned, lease-fenced safe fixture-quarantine recovery tick."
    )
    parser.add_argument("--root", type=Path, help="Selected HERMES_HOME/Kanban root")
    parser.add_argument("--expected-source-sha", help="Validated candidate source SHA to record")
    parser.add_argument("--stale-after-seconds", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--backoff-base-seconds", type=int, default=60)
    parser.add_argument("--backoff-max-seconds", type=int, default=3600)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Parent diagnostic only: ignore failure backoff (never child authorization)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.root is None:
        from hermes_cli.kanban_db import kanban_home

        root = kanban_home()
    else:
        root = args.root
    result = run_tick(
        root=root,
        expected_source_sha=args.expected_source_sha,
        stale_after_seconds=args.stale_after_seconds,
        limit=args.limit,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_max_seconds=args.backoff_max_seconds,
        force=args.force,
    )
    outcome = str(result["outcome"])
    should_emit = outcome in {"DENIED", "BUSY", "NOT_VERIFIED", "ERROR", "APPLIED"}
    if should_emit:
        sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 1 if outcome in {"DENIED", "NOT_VERIFIED", "ERROR"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
