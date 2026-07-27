"""Read-only owner ledger across every on-disk Kanban board.

The normal Kanban connection deliberately initializes and migrates a board.
Fleet visibility must have no such side effect: this module discovers existing
SQLite files and opens them with ``mode=ro`` only.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote

from hermes_cli import kanban_db as kb

_KNOWN_FIXTURE_ASSIGNEES = frozenset(
    {"builder", "reviewer", "closer", "grok-creative"}
)
_DELIVERY_STEPS = frozenset({"implementation", "verifier", "closure"})


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


def _board_paths(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    default = root / "kanban.db"
    if default.is_file():
        found.append((kb.DEFAULT_BOARD, default))
    boards = root / "kanban" / "boards"
    if boards.is_dir():
        for child in sorted(boards.iterdir(), key=lambda item: item.name):
            db = child / "kanban.db"
            if child.is_dir() and db.is_file():
                found.append((child.name, db))
    return found


def _known_project_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    paths = [root / "projects.db"]
    profiles = root / "profiles"
    if profiles.is_dir():
        paths.extend(sorted(profiles.glob("*/projects.db")))
    for path in paths:
        if not path.is_file():
            continue
        try:
            with closing(_readonly_connect(path)) as conn:
                table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projects'"
                ).fetchone()
                if table:
                    ids.update(
                        str(row[0])
                        for row in conn.execute("SELECT id FROM projects")
                        if row[0]
                    )
        except sqlite3.Error:
            continue
    return ids


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
) -> dict[str, tuple[str, ...]]:
    """Return strong fixture families keyed by task id.

    No single name, path, project id, or status is sufficient. A row must have
    an absent test-role profile, an unknown project, a temporary workspace,
    and belong to a multi-task family with delivery dependency evidence or the
    known Grok review-fixture topology.
    """
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


def _pid_alive(pid_value: object) -> bool:
    try:
        pid = int(pid_value)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def _bounded(ids: Iterable[str], limit: int) -> dict[str, object]:
    ordered = sorted(set(ids))
    shown = ordered[:limit]
    return {"count": len(ordered), "task_ids": shown, "omitted": len(ordered) - len(shown)}


def _board_ledger(
    slug: str,
    path: Path,
    *,
    known_profiles: set[str],
    known_project_ids: set[str],
    now: int,
    stale_after_seconds: int,
    limit: int,
) -> dict[str, object]:
    with closing(_readonly_connect(path)) as conn:
        tasks = [dict(row) for row in conn.execute(
            "SELECT id, title, assignee, status, created_by, created_at, "
            "project_id, workspace_path, workflow_template_id, current_step_key, "
            "current_run_id, worker_pid, last_heartbeat_at, started_at "
            "FROM tasks ORDER BY id"
        )]
        links = [tuple(row) for row in conn.execute(
            "SELECT parent_id, child_id FROM task_links ORDER BY parent_id, child_id"
        )]
        run_rows = {
            int(row["id"]): dict(row)
            for row in conn.execute(
                "SELECT id, task_id, worker_pid, last_heartbeat_at, started_at "
                "FROM task_runs WHERE ended_at IS NULL"
            )
        }

    candidates = classify_fixture_candidates(
        tasks,
        links,
        known_profiles=known_profiles,
        known_project_ids=known_project_ids,
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
        alive = _pid_alive(pid)
        if not alive or (not fresh and started is not None and now - int(started) > stale_after_seconds):
            stale_ids.append(str(row["id"]))
        running.append(
            {
                "task_id": str(row["id"]),
                "run_id": run_id,
                "assignee": row.get("assignee"),
                "project_id": row.get("project_id"),
                "workspace": row.get("workspace_path"),
                "pid": int(pid) if pid is not None else None,
                "pid_alive": alive,
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
    root = (root or kb.kanban_home()).expanduser().resolve()
    if known_profiles is None:
        known_profiles = set(kb.list_profiles_on_disk())
    if known_project_ids is None:
        known_project_ids = _known_project_ids(root)
    now = int(time.time()) if now is None else int(now)
    limit = max(1, min(int(limit), 100))
    boards = [
        _board_ledger(
            slug,
            path,
            known_profiles=set(known_profiles),
            known_project_ids=set(known_project_ids),
            now=now,
            stale_after_seconds=max(1, int(stale_after_seconds)),
            limit=limit,
        )
        for slug, path in _board_paths(root)
    ]
    totals = {
        "boards": len(boards),
        "running": sum(len(board["running"]) + int(board["running_omitted"]) for board in boards),
        "stale_running": sum(int(board["stale_running"]["count"]) for board in boards),
        "blockers": sum(int(board["blockers"]["count"]) for board in boards),
        "stranded_ready": sum(int(board["stranded_ready"]["count"]) for board in boards),
        "fixture_candidates_active": sum(int(board["fixture_candidates"]["active"]) for board in boards),
        "fixture_candidates_quarantined": sum(int(board["fixture_candidates"]["quarantined"]) for board in boards),
    }
    return {"generated_at": now, "root": str(root), "totals": totals, "boards": boards}
