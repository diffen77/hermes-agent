from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _tree_fingerprint(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def fleet_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.create_board("alpha", name="Alpha")
    kb.create_board("beta", name="Beta")
    return home


def test_fleet_ledger_is_read_only_and_surfaces_owner_diagnostics(
    fleet_home: Path, tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli.kanban_fleet import build_fleet_ledger

    now = 10_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    repo = tmp_path / "real-repo"
    repo.mkdir()
    with kb.connect_closing(board="alpha") as conn:
        running = kb.create_task(
            conn,
            title="real running work",
            assignee="john",
            workspace_kind="dir",
            workspace_path=str(repo),
        )
        conn.execute("UPDATE tasks SET project_id = ? WHERE id = ?", ("p_real", running))
        claimed = kb.claim_task(conn, running, claimer="john")
        assert claimed is not None
        kb._set_worker_pid(conn, running, os.getpid())
        assert kb.heartbeat_worker(conn, running)
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?",
            (now - 10, running),
        )
        conn.execute(
            "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
            (now - 10, claimed.current_run_id),
        )

        blocked = kb.create_task(conn, title="blocked", assignee="raffe")
        assert kb.claim_task(conn, blocked, claimer="raffe")
        assert kb.block_task(
            conn, blocked, reason="needs_input: decision", kind="needs_input"
        )

        stranded = kb.create_task(conn, title="stranded", assignee="missing-profile")

        fixture_root = tmp_path / "delivery-review-fixture"
        fixture_root.mkdir()
        fixture_ids = []
        parent = None
        for role, assignee in (
            ("implementation", "builder"),
            ("verifier", "reviewer"),
            ("closure", "closer"),
        ):
            task_id = kb.create_task(
                conn,
                title=f"Fixture {role}",
                assignee=assignee,
                created_by="orchestrator",
                parents=(parent,) if parent else (),
                session_id="fixture-session",
                workspace_kind="worktree",
                workspace_path=str(fixture_root / ".worktrees" / f"{role}-review"),
            )
            conn.execute(
                "UPDATE tasks SET project_id = ?, workflow_template_id = ?, "
                "current_step_key = ? WHERE id = ?",
                ("p_fixture", "delivery_v1", role, task_id),
            )
            fixture_ids.append(task_id)
            parent = task_id

        # An isolated signal is not enough: this legitimate task happens to use
        # a role-like name but has a real project and non-temporary workspace.
        legitimate_builder = kb.create_task(
            conn,
            title="legitimate builder task",
            assignee="builder",
            workspace_kind="dir",
            workspace_path=str(repo),
        )
        conn.execute(
            "UPDATE tasks SET project_id = ? WHERE title = ?",
            ("p_real", "legitimate builder task"),
        )

    with kb.connect_closing(board="beta") as conn:
        kb.create_task(conn, title="beta todo")

    before = _tree_fingerprint(fleet_home)
    ledger = build_fleet_ledger(
        root=fleet_home,
        known_profiles={"john", "raffe"},
        known_project_ids={"p_real"},
        now=now,
        stale_after_seconds=60,
    )
    after = _tree_fingerprint(fleet_home)

    assert after == before
    assert [board["slug"] for board in ledger["boards"]] == ["alpha", "beta"]
    alpha = ledger["boards"][0]
    assert alpha["counts"]["running"] == 1
    assert alpha["running"] == [
        {
            "task_id": running,
            "run_id": claimed.current_run_id,
            "assignee": "john",
            "project_id": "p_real",
            "workspace": str(repo),
            "pid": os.getpid(),
            "pid_alive": True,
            "last_heartbeat_at": now - 10,
            "heartbeat_age_seconds": 10,
            "heartbeat_fresh": True,
        }
    ]
    assert alpha["blockers"] == {"count": 1, "task_ids": [blocked], "omitted": 0}
    assert alpha["stranded_ready"] == {
        "count": 2,
        "task_ids": sorted([stranded, legitimate_builder]),
        "omitted": 0,
    }
    assert alpha["fixture_candidates"] == {
        "active": len(fixture_ids),
        "quarantined": 0,
        "task_ids": sorted(fixture_ids),
        "omitted": 0,
    }
    assert alpha["stale_running"] == {"count": 0, "task_ids": [], "omitted": 0}
    assert ledger["totals"]["fixture_candidates_active"] == 3


def test_fleet_cli_json_does_not_initialize_missing_default_board(
    fleet_home: Path,
) -> None:
    before = _tree_fingerprint(fleet_home)

    payload = json.loads(kc.run_slash("fleet --json"))

    assert _tree_fingerprint(fleet_home) == before
    assert [board["slug"] for board in payload["boards"]] == ["alpha", "beta"]
    assert not (fleet_home / "kanban.db").exists()


def _delivery_fixture_rows(tmp_path: Path, project_id: str):
    rows = []
    links = []
    parent = None
    for index, (step, assignee) in enumerate(
        (("implementation", "builder"), ("verifier", "reviewer"), ("closure", "closer"))
    ):
        task_id = f"t_delivery_{index}"
        rows.append(
            {
                "id": task_id,
                "assignee": assignee,
                "project_id": project_id,
                "workspace_path": str(tmp_path / ".worktrees" / task_id),
                "created_at": 123,
                "workflow_template_id": "delivery_v1",
                "current_step_key": step,
            }
        )
        if parent is not None:
            links.append((parent, task_id))
        parent = task_id
    return rows, links


def test_fixture_classifier_preserves_known_p102_delivery_rows(tmp_path: Path) -> None:
    from hermes_cli.kanban_fleet import classify_fixture_candidates

    rows, links = _delivery_fixture_rows(tmp_path, "p_102c8528")

    assert classify_fixture_candidates(
        rows,
        links,
        known_profiles={"john", "raffe", "janne"},
        known_project_ids={"p_102c8528"},
    ) == {}


@pytest.mark.parametrize(
    ("known_profiles", "known_project_ids"),
    [
        ({"john", "raffe", "janne"}, set()),
        (set(), {"p_unrelated"}),
    ],
)
def test_fixture_classifier_fails_closed_without_registry_or_profile_evidence(
    tmp_path: Path,
    known_profiles: set[str],
    known_project_ids: set[str],
) -> None:
    from hermes_cli.kanban_fleet import classify_fixture_candidates

    rows, links = _delivery_fixture_rows(tmp_path, "p_102c8528")

    assert classify_fixture_candidates(
        rows,
        links,
        known_profiles=known_profiles,
        known_project_ids=known_project_ids,
    ) == {}


def test_fleet_outputs_redact_and_bound_running_identity(
    fleet_home: Path, tmp_path: Path
) -> None:
    secret = "ghp_" + "A" * 24
    workspace = tmp_path / secret / ("nested-" + "x" * 300)
    with kb.connect_closing(board="alpha") as conn:
        task_id = kb.create_task(
            conn,
            title="safe",
            assignee="john",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        conn.execute("UPDATE tasks SET project_id=? WHERE id=?", ("p_safe", task_id))
        assert kb.claim_task(conn, task_id, claimer="john") is not None

    json_output = kc.run_slash("fleet --json")
    human_output = kc.run_slash("fleet")
    payload = json.loads(json_output)
    running = payload["boards"][0]["running"][0]

    assert secret not in json_output
    assert secret not in human_output
    assert running["project_id"] == "p_safe"
    assert len(running["workspace"]) <= 180
    assert "project=p_safe" in human_output
    assert "workspace=" in human_output
