"""Read-only owner ledger across every on-disk Kanban board.

The normal Kanban connection deliberately initializes and migrates a board.
Fleet visibility must have no such side effect: this module discovers existing
SQLite files and opens them with ``mode=ro`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping, cast
from urllib.parse import quote

from agent.redact import redact_sensitive_text
from hermes_cli import kanban_db as kb

_KNOWN_FIXTURE_ASSIGNEES = frozenset(
    {"builder", "reviewer", "closer", "grok-creative"}
)
_DELIVERY_STEPS = frozenset({"implementation", "verifier", "closure"})
_MAX_DISCOVERY_ERRORS = 20
_SAFE_FIXTURE_STATUSES = frozenset(
    {"triage", "todo", "scheduled", "ready", "done"}
)
_TASK_SELECT = (
    "SELECT id, title, assignee, status, created_by, created_at, "
    "project_id, workspace_path, workflow_template_id, current_step_key, "
    "current_run_id, worker_pid, last_heartbeat_at, started_at "
    "FROM tasks ORDER BY id"
)
_RUN_SELECT = (
    "SELECT id, task_id, worker_pid, last_heartbeat_at, started_at "
    "FROM task_runs WHERE ended_at IS NULL ORDER BY id"
)


class _BoundedErrors:
    def __init__(self) -> None:
        self.total = 0
        self.items: list[dict[str, str]] = []

    def add(self, scope: str, code: str) -> None:
        self.total += 1
        if len(self.items) < _MAX_DISCOVERY_ERRORS:
            self.items.append({"scope": scope, "code": code})


def _readonly_connect(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    wal = Path(f"{resolved}-wal")
    # A clean, closed database can be opened immutable, which prevents even
    # SQLite's transient empty -wal/-shm creation. An active WAL must remain
    # visible, so use ordinary read-only mode when it contains frames.
    immutable = not wal.exists() or wal.stat().st_size == 0
    uri = f"file:{quote(str(resolved))}?mode=ro" + ("&immutable=1" if immutable else "")
    conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _record_error(errors: _BoundedErrors, scope: str, code: str) -> None:
    """Record a bounded, non-sensitive error category."""
    errors.add(scope, code)


def _board_paths(
    root: Path, errors: _BoundedErrors | None = None
) -> tuple[list[tuple[str, Path]], bool]:
    errors = errors if errors is not None else _BoundedErrors()
    found: list[tuple[str, Path]] = []
    complete = True
    identities: set[tuple[int, int]] = set()

    def add(slug: str, path: Path) -> None:
        nonlocal complete
        try:
            info = path.stat()
        except FileNotFoundError:
            return
        except OSError:
            complete = False
            _record_error(errors, "boards", "board_stat_failed")
            return
        if not stat.S_ISREG(info.st_mode):
            complete = False
            _record_error(errors, "boards", "board_not_regular")
            return
        identity = (info.st_dev, info.st_ino)
        if identity in identities:
            return
        identities.add(identity)
        found.append((slug, path))

    default = root / "kanban.db"
    add(kb.DEFAULT_BOARD, default)
    boards = root / "kanban" / "boards"
    try:
        boards_mode = boards.stat().st_mode
    except FileNotFoundError:
        return found, complete
    except OSError:
        _record_error(errors, "boards", "boards_directory_stat_failed")
        return found, False
    if not stat.S_ISDIR(boards_mode):
        _record_error(errors, "boards", "boards_path_not_directory")
        return found, False
    try:
        children = sorted(boards.iterdir(), key=lambda item: item.name)
    except OSError:
        _record_error(errors, "boards", "boards_directory_read_failed")
        return found, False
    for child in children:
        try:
            is_directory = stat.S_ISDIR(child.stat().st_mode)
        except OSError:
            complete = False
            _record_error(errors, "boards", "board_directory_stat_failed")
            continue
        if is_directory:
            add(child.name, child / "kanban.db")
    return found, complete


def _profile_directories(root: Path) -> tuple[list[Path], bool]:
    profiles = root / "profiles"
    try:
        mode = profiles.stat().st_mode
    except FileNotFoundError:
        return [], True
    except OSError:
        return [], False
    if not stat.S_ISDIR(mode):
        return [], False
    try:
        entries = sorted(profiles.iterdir(), key=lambda item: item.name)
    except OSError:
        return [], False

    directories: list[Path] = []
    complete = True
    for entry in entries:
        try:
            if stat.S_ISDIR(entry.stat().st_mode):
                directories.append(entry)
        except OSError:
            complete = False
    return directories, complete


def _optional_regular_file(path: Path) -> tuple[bool, bool]:
    """Return ``(exists_as_file, discovery_complete)`` for an optional file."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False, True
    except OSError:
        return False, False
    is_file = stat.S_ISREG(mode)
    return is_file, is_file


def _known_profile_names(
    root: Path, errors: _BoundedErrors | None = None
) -> tuple[set[str], bool]:
    errors = errors if errors is not None else _BoundedErrors()
    names: set[str] = set()
    complete = True
    try:
        root_mode = root.stat().st_mode
    except FileNotFoundError:
        root_mode = None
    except OSError:
        root_mode = None
        complete = False
        _record_error(errors, "profiles", "root_stat_failed")
    if root_mode is not None:
        if stat.S_ISDIR(root_mode):
            names.add("default")
        else:
            complete = False
            _record_error(errors, "profiles", "root_not_directory")

    profiles, profiles_complete = _profile_directories(root)
    if not profiles_complete:
        _record_error(errors, "profiles", "profile_discovery_failed")
    complete = complete and profiles_complete
    for profile in profiles:
        is_config, config_complete = _optional_regular_file(profile / "config.yaml")
        complete = complete and config_complete
        if not config_complete:
            _record_error(errors, "profiles", "profile_config_stat_failed")
        if is_config:
            names.add(profile.name)
    return names, complete


def _known_project_ids(
    root: Path, errors: _BoundedErrors | None = None
) -> tuple[set[str], bool]:
    errors = errors if errors is not None else _BoundedErrors()
    ids: set[str] = set()
    complete = True
    profiles, profiles_complete = _profile_directories(root)
    if not profiles_complete:
        _record_error(errors, "projects", "profile_discovery_failed")
    complete = complete and profiles_complete
    paths = [root / "projects.db", *(profile / "projects.db" for profile in profiles)]
    for path in paths:
        is_registry, registry_complete = _optional_regular_file(path)
        complete = complete and registry_complete
        if not registry_complete:
            _record_error(errors, "projects", "project_registry_stat_failed")
        if not is_registry:
            continue
        try:
            with closing(_readonly_connect(path)) as conn:
                table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projects'"
                ).fetchone()
                if not table:
                    complete = False
                    _record_error(errors, "projects", "project_registry_schema_missing")
                    continue
                ids.update(
                    str(row[0])
                    for row in conn.execute("SELECT id FROM projects")
                    if row[0]
                )
        except (OSError, sqlite3.Error):
            complete = False
            _record_error(errors, "projects", "project_registry_read_failed")
    return ids, complete


def _workspace_family(path_value: object) -> tuple[bool, str | None]:
    if not path_value:
        return False, None
    path = Path(str(path_value)).expanduser()
    try:
        resolved = path.resolve(strict=False)
        temp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        is_temp = resolved == temp_root or resolved.is_relative_to(temp_root)
    except (OSError, RuntimeError):
        return False, None
    if not is_temp:
        return False, None
    parts = resolved.parts
    try:
        marker = parts.index(".worktrees")
    except ValueError:
        return True, str(resolved.parent)
    return True, str(Path(*parts[:marker]))


def classify_fixture_candidates(
    tasks: Iterable[Mapping[str, object]],
    links: Iterable[tuple[str, str]],
    *,
    known_profiles: set[str],
    known_project_ids: set[str],
    profiles_complete: bool = True,
    projects_complete: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Return strong fixture families keyed by task id.

    No single name, path, project id, or status is sufficient. A row must have
    an absent test-role profile, an unknown project, a temporary workspace,
    and belong to a multi-task family with delivery dependency evidence or the
    known Grok review-fixture topology.
    """
    # Absence can only be proven against complete, positive registry evidence.
    # Partial discovery is not evidence that an omitted profile/project is
    # absent, so fail closed before examining any task rows.
    if (
        not profiles_complete
        or not projects_complete
        or not known_profiles
        or not known_project_ids
    ):
        return {}
    rows = [dict(row) for row in tasks]
    eligible: dict[str, dict[str, object]] = {}
    families: dict[tuple[object, object, str | None], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("id") or "")
        assignee = str(row.get("assignee") or "")
        project_id = str(row.get("project_id") or "")
        temporary, family_root = _workspace_family(row.get("workspace_path"))
        if (
            not task_id
            or assignee not in _KNOWN_FIXTURE_ASSIGNEES
            or assignee in known_profiles
            or not project_id
            or project_id in known_project_ids
            or not temporary
        ):
            continue
        eligible[task_id] = row
        families[(project_id, row.get("created_at"), family_root)].append(row)

    linked = {(str(parent), str(child)) for parent, child in links}
    result: dict[str, tuple[str, ...]] = {}
    for family in families.values():
        family_ids = {str(row["id"]) for row in family}
        steps = {str(row.get("current_step_key") or "") for row in family}
        workflow_ids = {str(row.get("workflow_template_id") or "") for row in family}
        family_links = {
            edge for edge in linked if edge[0] in family_ids and edge[1] in family_ids
        }
        delivery_family = (
            "delivery_v1" in workflow_ids
            and _DELIVERY_STEPS.issubset(steps)
            and len(family_links) >= 2
        )
        grok_family = (
            len(family) >= 3
            and {str(row.get("assignee") or "") for row in family}
            == {"grok-creative"}
            and len({str(row.get("workspace_path") or "") for row in family}) >= 3
        )
        if not (delivery_family or grok_family):
            continue
        reason = (
            "delivery_v1_temp_unknown_project_family"
            if delivery_family
            else "grok_temp_unknown_project_family"
        )
        for task_id in sorted(family_ids & eligible.keys()):
            result[task_id] = (
                "nonexistent_test_role_profile",
                "unknown_project_id",
                "temporary_workspace_family",
                reason,
            )
    return result


def _read_process_identity(pid: int) -> dict[str, object]:
    """Return bounded identity evidence without exposing cmdline or environment."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return {"state": "not_found"}
    except PermissionError:
        return {"state": "permission_denied"}
    except OSError:
        return {"state": "unknown"}
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        start_ticks = int(raw.rsplit(")", 1)[1].split()[19])
        boot_line = next(
            line
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
            if line.startswith("btime ")
        )
        birth_epoch = int(boot_line.split()[1]) + start_ticks / os.sysconf("SC_CLK_TCK")
        return {"state": "present", "birth_epoch": int(birth_epoch)}
    except PermissionError:
        return {"state": "permission_denied"}
    except (FileNotFoundError, ProcessLookupError):
        return {"state": "birth_ambiguous"}
    except (OSError, ValueError, IndexError, StopIteration):
        return {"state": "birth_ambiguous"}


def _classify_process_identity(
    pid_value: object, *, expected_started_at: int | None
) -> dict[str, str]:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return {
            "classification": "pid_missing_or_invalid",
            "verdict": "NOT_VERIFIED",
            "reclaim_authority": "NOT_AUTHORIZED",
        }
    if pid <= 0:
        return {
            "classification": "pid_missing_or_invalid",
            "verdict": "NOT_VERIFIED",
            "reclaim_authority": "NOT_AUTHORIZED",
        }
    observed = _read_process_identity(pid)
    state = str(observed.get("state") or "unknown")
    if state != "present":
        return {
            "classification": state,
            "verdict": "NOT_VERIFIED",
            "reclaim_authority": "NOT_AUTHORIZED",
        }
    birth = observed.get("birth_epoch")
    if birth is None or expected_started_at is None:
        classification = "birth_ambiguous"
    elif int(birth) > int(expected_started_at) + 2:
        classification = "pid_reused_or_birth_mismatch"
    else:
        # This corroborates the row but cannot authorize reclaim: the exact
        # process birth identity was not persisted when the run was claimed.
        classification = "present_birth_consistent"
    return {
        "classification": classification,
        "verdict": "NOT_VERIFIED",
        "reclaim_authority": "NOT_AUTHORIZED",
    }


def _after_tasks_snapshot(slug: str) -> None:
    """Test seam after the first table read in the coherent transaction."""


def _path_identity(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


class _ObservationDriftError(RuntimeError):
    pass


def _snapshot_generation(
    identity: tuple[int, int],
    schema_version: int,
    tasks: list[dict[str, object]],
    links: list[tuple[object, object]],
    run_rows: dict[int, dict[str, object]],
) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                identity[0],
                identity[1],
                schema_version,
                tasks,
                links,
                [run_rows[key] for key in sorted(run_rows)],
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]


def _bounded(ids: Iterable[str], limit: int) -> dict[str, object]:
    ordered = sorted(set(ids))
    shown = ordered[:limit]
    return {"count": len(ordered), "task_ids": shown, "omitted": len(ordered) - len(shown)}


def _safe_text(value: str, limit: int = 180) -> str:
    """Force-redact and deterministically bound a string at the ledger boundary."""
    safe = redact_sensitive_text(
        value,
        force=True,
        redact_url_credentials=True,
    )
    if len(safe) <= limit:
        return safe
    suffix = f"…#{hashlib.sha256(safe.encode()).hexdigest()[:12]}"
    return safe[: limit - len(suffix)] + suffix


def _safe_projection(value: object) -> object:
    """Apply the egress safety boundary to every projected string/key."""
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, list):
        return [_safe_projection(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_projection(item) for item in value)
    if isinstance(value, dict):
        return {
            _safe_text(key) if isinstance(key, str) else key: _safe_projection(item)
            for key, item in value.items()
        }
    return value


def _board_ledger(
    slug: str,
    path: Path,
    *,
    known_profiles: set[str],
    known_project_ids: set[str],
    profiles_complete: bool,
    projects_complete: bool,
    now: int,
    stale_after_seconds: int,
    limit: int,
) -> dict[str, object]:
    identity_before = _path_identity(path)
    with closing(_readonly_connect(path)) as conn:
        conn.execute("BEGIN")
        schema_before = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        data_before = int(conn.execute("PRAGMA data_version").fetchone()[0])
        tasks = [dict(row) for row in conn.execute(_TASK_SELECT)]
        _after_tasks_snapshot(slug)
        links = [tuple(row) for row in conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        )]
        run_rows = {
            int(row["id"]): dict(row)
            for row in conn.execute(_RUN_SELECT)
        }
        schema_after = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        data_after = int(conn.execute("PRAGMA data_version").fetchone()[0])
        identity_after = _path_identity(path)
        conn.rollback()
    if (
        identity_before != identity_after
        or schema_before != schema_after
        or data_before != data_after
    ):
        raise _ObservationDriftError("board identity or generation drifted")
    snapshot_generation = _snapshot_generation(
        identity_before, schema_before, tasks, links, run_rows
    )

    candidates = classify_fixture_candidates(
        tasks,
        links,
        known_profiles=known_profiles,
        known_project_ids=known_project_ids,
        profiles_complete=profiles_complete,
        projects_complete=projects_complete,
    )
    counts = dict(sorted(Counter(str(row["status"]) for row in tasks).items()))
    running: list[dict[str, object]] = []
    stale_ids: list[str] = []
    for row in tasks:
        if row["status"] != "running":
            continue
        run_id = int(row["current_run_id"]) if row.get("current_run_id") else None
        run = run_rows.get(run_id or -1, {})
        pid = run.get("worker_pid") or row.get("worker_pid")
        heartbeat = run.get("last_heartbeat_at") or row.get("last_heartbeat_at")
        started = run.get("started_at") or row.get("started_at")
        age = max(0, now - int(heartbeat)) if heartbeat is not None else None
        fresh = age is not None and age <= stale_after_seconds
        process_identity = _classify_process_identity(
            pid,
            expected_started_at=int(started) if started is not None else None,
        )
        if process_identity["classification"] in {
            "not_found",
            "pid_reused_or_birth_mismatch",
        } or (not fresh and started is not None and now - int(started) > stale_after_seconds):
            stale_ids.append(str(row["id"]))
        running.append(
            {
                "task_id": str(row["id"]),
                "run_id": run_id,
                "assignee": row.get("assignee"),
                "project_id": row.get("project_id"),
                "workspace": row.get("workspace_path"),
                "pid": int(pid) if pid is not None else None,
                "process_identity": process_identity,
                "last_heartbeat_at": int(heartbeat) if heartbeat is not None else None,
                "heartbeat_age_seconds": age,
                "heartbeat_fresh": fresh,
            }
        )
    running.sort(key=lambda item: str(item["task_id"]))

    blocker_ids = [str(row["id"]) for row in tasks if row["status"] == "blocked"]
    stranded_ids = [
        str(row["id"])
        for row in tasks
        if row["status"] == "ready"
        and str(row["id"]) not in candidates
        and (not row.get("assignee") or str(row["assignee"]) not in known_profiles)
    ]
    active_fixture_ids = [
        task_id
        for task_id in candidates
        if next(row for row in tasks if str(row["id"]) == task_id)["status"] != "archived"
    ]
    quarantined_fixture_ids = [task_id for task_id in candidates if task_id not in active_fixture_ids]
    fixture_ids = sorted(candidates)[:limit]
    return {
        "slug": slug,
        "snapshot": {"generation": snapshot_generation, "verdict": "VERIFIED"},
        "counts": counts,
        "running": running[:limit],
        "running_omitted": max(0, len(running) - limit),
        "stale_running": _bounded(stale_ids, limit),
        "blockers": _bounded(blocker_ids, limit),
        "stranded_ready": _bounded(stranded_ids, limit),
        "fixture_candidates": {
            "active": len(active_fixture_ids),
            "quarantined": len(quarantined_fixture_ids),
            "task_ids": fixture_ids,
            "omitted": max(0, len(candidates) - len(fixture_ids)),
        },
    }


def build_fleet_ledger(
    *,
    root: Path | None = None,
    known_profiles: set[str] | None = None,
    known_project_ids: set[str] | None = None,
    now: int | None = None,
    stale_after_seconds: int = 3600,
    limit: int = 20,
) -> dict[str, object]:
    """Build a bounded, deterministic, read-only fleet projection."""
    root_value = (root or kb.kanban_home()).expanduser()
    root = Path(os.path.abspath(root_value))
    errors = _BoundedErrors()
    if known_profiles is None:
        known_profiles, profiles_complete = _known_profile_names(root, errors)
    else:
        profiles_complete = True
    if known_project_ids is None:
        known_project_ids, projects_complete = _known_project_ids(root, errors)
    else:
        projects_complete = True
    now = int(time.time()) if now is None else int(now)
    limit = max(1, min(int(limit), 100))
    board_paths, boards_complete = _board_paths(root, errors)
    boards: list[dict[str, object]] = []
    for slug, path in board_paths:
        try:
            boards.append(
                _board_ledger(
                    slug,
                    path,
                    known_profiles=set(known_profiles),
                    known_project_ids=set(known_project_ids),
                    profiles_complete=profiles_complete,
                    projects_complete=projects_complete,
                    now=now,
                    stale_after_seconds=max(1, int(stale_after_seconds)),
                    limit=limit,
                )
            )
        except _ObservationDriftError:
            boards_complete = False
            _record_error(errors, "boards", "board_snapshot_drift")
        except (OSError, sqlite3.Error):
            boards_complete = False
            _record_error(errors, "boards", "board_snapshot_read_failed")
    completeness = {
        "profiles": profiles_complete,
        "projects": projects_complete,
        "boards": boards_complete,
    }
    verified = all(completeness.values())
    verdict = "VERIFIED" if verified else "NOT_VERIFIED"
    totals = {
        "boards": len(boards),
        "running": sum(len(board["running"]) + int(board["running_omitted"]) for board in boards),
        "stale_running": sum(int(board["stale_running"]["count"]) for board in boards),
        "blockers": sum(int(board["blockers"]["count"]) for board in boards),
        "stranded_ready": sum(int(board["stranded_ready"]["count"]) for board in boards),
        "fixture_candidates_active": sum(int(board["fixture_candidates"]["active"]) for board in boards),
        "fixture_candidates_quarantined": sum(int(board["fixture_candidates"]["quarantined"]) for board in boards),
    }
    generation_payload = {
        "generated_at": now,
        "completeness": completeness,
        "boards": [
            [board["slug"], board["snapshot"]["generation"]] for board in boards
        ],
    }
    observation_generation = hashlib.sha256(
        json.dumps(generation_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    error_limit = min(limit, _MAX_DISCOVERY_ERRORS)
    payload = {
        "generated_at": now,
        "observation_generation": observation_generation,
        "verdict": verdict,
        "reclaim_authority": "NOT_AUTHORIZED",
        "completeness": completeness,
        "errors": {
            "count": errors.total,
            "items": errors.items[:error_limit],
            "omitted": max(0, errors.total - min(error_limit, len(errors.items))),
        },
        "root": str(root),
        "totals": totals,
        "boards": boards,
    }
    return cast(dict[str, object], _safe_projection(payload))


def _recovery_receipt_id(slug: str, task_id: str) -> str:
    return hashlib.sha256(
        f"fixture-quarantine-v1\0{slug}\0{task_id}".encode()
    ).hexdigest()


def _receipt_path(root: Path, receipt_id: str) -> Path:
    return root / "kanban" / "reconciler-receipts" / f"{receipt_id}.json"


def _write_recovery_receipt(root: Path, payload: dict[str, object]) -> Path:
    """Atomically persist one bounded, redacted idempotency receipt."""
    receipt_id = str(payload["receipt_id"])
    target = _receipt_path(root, receipt_id)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    safe = cast(dict[str, object], _safe_projection(payload))
    data = (json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{receipt_id}.", suffix=".tmp", dir=target.parent
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def _after_recovery_commit(slug: str, task_id: str) -> None:
    """Failure-injection seam after the durable DB commit, before the receipt."""


def _apply_board_fixture_quarantine(
    *,
    root: Path,
    slug: str,
    path: Path,
    expected_generation: str,
    known_profiles: set[str],
    known_project_ids: set[str],
    now: int,
) -> dict[str, object]:
    """CAS one proven, non-running fixture leaf to archived."""
    identity_before = _path_identity(path)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        tasks = [dict(row) for row in conn.execute(_TASK_SELECT)]
        links = [
            tuple(row)
            for row in conn.execute(
                "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
            )
        ]
        run_rows = {
            int(row["id"]): dict(row) for row in conn.execute(_RUN_SELECT)
        }
        identity_locked = _path_identity(path)
        actual_generation = _snapshot_generation(
            identity_locked,
            schema_version,
            tasks,
            links,
            run_rows,
        )
        if identity_before != identity_locked or actual_generation != expected_generation:
            conn.rollback()
            return {"board": slug, "outcome": "cas_mismatch", "mutations": 0}

        candidates = classify_fixture_candidates(
            tasks,
            links,
            known_profiles=known_profiles,
            known_project_ids=known_project_ids,
            profiles_complete=True,
            projects_complete=True,
        )
        rows = {str(row["id"]): row for row in tasks}
        active_children = {
            str(parent)
            for parent, child in links
            if str(child) in rows and rows[str(child)].get("status") != "archived"
        }
        for task_id in sorted(candidates):
            row = rows[task_id]
            receipt_id = _recovery_receipt_id(slug, task_id)
            receipt = _receipt_path(root, receipt_id)
            if row.get("status") != "archived" or receipt.is_file():
                continue
            conn.rollback()
            _write_recovery_receipt(
                root,
                {
                    "receipt_id": receipt_id,
                    "action": "fixture_quarantine",
                    "board": slug,
                    "task_id": task_id,
                    "outcome": "already_archived",
                    "recorded_at": now,
                },
            )
            return {
                "board": slug,
                "outcome": "receipt_recovered",
                "mutations": 0,
                "receipt_id": receipt_id,
            }

        for task_id in sorted(candidates):
            row = rows[task_id]
            receipt_id = _recovery_receipt_id(slug, task_id)
            receipt = _receipt_path(root, receipt_id)
            if receipt.is_file():
                continue
            if (
                row.get("status") not in _SAFE_FIXTURE_STATUSES
                or row.get("current_run_id") is not None
                or row.get("worker_pid") is not None
                or task_id in active_children
            ):
                continue
            updated = conn.execute(
                "UPDATE tasks SET status='archived', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL "
                "WHERE id=? AND status=? AND current_run_id IS NULL "
                "AND worker_pid IS NULL",
                (task_id, row["status"]),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return {"board": slug, "outcome": "cas_mismatch", "mutations": 0}
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, 'archived', ?, ?)",
                (
                    task_id,
                    json.dumps(
                        {
                            "source": "fleet_safe_recovery",
                            "class": "fixture_quarantine",
                            "receipt_id": receipt_id,
                        },
                        sort_keys=True,
                    ),
                    now,
                ),
            )
            conn.commit()
            _after_recovery_commit(slug, task_id)
            _write_recovery_receipt(
                root,
                {
                    "receipt_id": receipt_id,
                    "action": "fixture_quarantine",
                    "board": slug,
                    "task_id": task_id,
                    "outcome": "applied",
                    "recorded_at": now,
                    "observation_generation": expected_generation,
                },
            )
            return {
                "board": slug,
                "outcome": "applied",
                "mutations": 1,
                "receipt_id": receipt_id,
            }
        conn.rollback()
        return {"board": slug, "outcome": "no_safe_candidate", "mutations": 0}
    finally:
        conn.close()


def apply_safe_recovery(
    *,
    root: Path | None = None,
    now: int | None = None,
    stale_after_seconds: int = 3600,
    limit: int = 20,
) -> dict[str, object]:
    """Apply only reversible deterministic fixture quarantine recoveries.

    Human decisions, credentials, costs, product/runtime state, blocked tasks,
    stranded work, and ambiguous worker ownership are deliberately excluded.
    """
    try:
        from agent.delegation_context import is_delegated_child_process_context

        delegated = is_delegated_child_process_context()
    except Exception:
        delegated = bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))
    if delegated:
        return {
            "verdict": "DENIED",
            "reason": "delegated_child_context",
            "mutations": 0,
            "boards": [],
        }
    root_value = (root or kb.kanban_home()).expanduser()
    root = Path(os.path.abspath(root_value))
    errors = _BoundedErrors()
    known_profiles, profiles_complete = _known_profile_names(root, errors)
    known_project_ids, projects_complete = _known_project_ids(root, errors)
    if (
        not profiles_complete
        or not projects_complete
        or not known_profiles
        or not known_project_ids
    ):
        return {
            "verdict": "NOT_VERIFIED",
            "reason": "authority_discovery_incomplete",
            "mutations": 0,
            "boards": [],
        }
    now = int(time.time()) if now is None else int(now)
    ledger = build_fleet_ledger(
        root=root,
        known_profiles=known_profiles,
        known_project_ids=known_project_ids,
        now=now,
        stale_after_seconds=stale_after_seconds,
        limit=limit,
    )
    if ledger["verdict"] != "VERIFIED":
        return {
            "verdict": "NOT_VERIFIED",
            "reason": "fleet_observation_incomplete",
            "observation_generation": ledger["observation_generation"],
            "mutations": 0,
            "boards": [],
        }
    board_paths, paths_complete = _board_paths(root)
    if not paths_complete:
        return {
            "verdict": "NOT_VERIFIED",
            "reason": "board_rediscovery_incomplete",
            "observation_generation": ledger["observation_generation"],
            "mutations": 0,
            "boards": [],
        }
    paths_by_slug = {slug: path for slug, path in board_paths}
    results: list[dict[str, object]] = []
    ledger_boards = cast(list[dict[str, object]], ledger["boards"])
    for board in ledger_boards:
        slug = str(board["slug"])
        path = paths_by_slug.get(slug)
        if path is None:
            results.append({"board": slug, "outcome": "cas_mismatch", "mutations": 0})
            continue
        # Profile/project registries are separate authorities from each board.
        # Re-read them immediately before the board transaction; absence may
        # never become proof merely because the earlier observation is stale.
        fresh_profiles, fresh_profiles_complete = _known_profile_names(root)
        fresh_projects, fresh_projects_complete = _known_project_ids(root)
        if (
            not fresh_profiles_complete
            or not fresh_projects_complete
            or fresh_profiles != known_profiles
            or fresh_projects != known_project_ids
        ):
            results.append(
                {"board": slug, "outcome": "authority_cas_mismatch", "mutations": 0}
            )
            continue
        results.append(
            _apply_board_fixture_quarantine(
                root=root,
                slug=slug,
                path=path,
                expected_generation=str(
                    cast(dict[str, object], board["snapshot"])["generation"]
                ),
                known_profiles=fresh_profiles,
                known_project_ids=fresh_projects,
                now=now,
            )
        )
    rejected = any(str(result["outcome"]).endswith("cas_mismatch") for result in results)
    payload = {
        "verdict": "NOT_VERIFIED" if rejected else "APPLIED",
        "reason": "fresh_authority_cas_rejected" if rejected else None,
        "observation_generation": ledger["observation_generation"],
        "mutations": sum(cast(int, result["mutations"]) for result in results),
        "boards": results,
    }
    return cast(dict[str, object], _safe_projection(payload))
