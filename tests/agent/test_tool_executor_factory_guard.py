"""Run-scoped Factory mutation middleware at the shared tool seam."""

from __future__ import annotations

from pathlib import Path

from agent.tool_executor import _factory_mutation_block
from hermes_cli import kanban_db as kb


def _binding(workspace: str) -> dict:
    return {
        "desktop_project_id": "p_102c8528",
        "board_slug": "factory",
        "repo_root": str(Path(workspace).parent),
        "git_common_dir": str(Path(workspace).parent / ".git"),
        "remote_identity": "NousResearch/hermes-agent",
        "workspace_root": workspace,
        "target_ref": "refs/heads/factory/task",
        "criterion_set_digest": "criteria-sha256",
        "writer_profile": "john",
        "reviewer_profile": "raffe",
        "closer_profile": "janne",
    }


def test_mutation_middleware_allows_only_exact_active_writer_tuple(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "repo" / "worktree"
    workspace.mkdir(parents=True)
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="guard",
            assignee="john",
            project_id="p_102c8528",
            enforcement_version="factory-v1",
            project_binding=_binding(str(workspace)),
            board="factory",
        )
        task = kb.claim_task(conn, task_id, claimer="writer")
        auth_id = conn.execute(
            "SELECT id FROM factory_authorizations WHERE task_id=?", (task_id,)
        ).fetchone()[0]

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_FACTORY_EXECUTION_GENERATION", str(task.execution_generation))
    monkeypatch.setenv("HERMES_FACTORY_AUTHORIZATION_ID", auth_id)
    monkeypatch.setenv("HERMES_FACTORY_BINDING_DIGEST", task.project_binding_digest)

    assert _factory_mutation_block(
        "patch", {"path": str(workspace / "module.py")}, task_id
    ) is None

    monkeypatch.setenv(
        "HERMES_FACTORY_EXECUTION_GENERATION", str(task.execution_generation - 1)
    )
    denial = _factory_mutation_block(
        "patch", {"path": str(workspace / "module.py")}, task_id
    )
    assert denial is not None
    assert "factory-v1 mutation denied" in denial


def test_mutation_middleware_preserves_legacy_tasks(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="john")
        kb.claim_task(conn, task_id, claimer="legacy")

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    assert _factory_mutation_block("write_file", {"path": "anything.py"}, task_id) is None


def test_mutation_middleware_denies_task_scope_mismatch_and_process_side_effects(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "repo" / "worktree"
    workspace.mkdir(parents=True)
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="guard mismatch",
            assignee="john",
            project_id="p_102c8528",
            enforcement_version="factory-v1",
            project_binding=_binding(str(workspace)),
            board="factory",
        )
        task = kb.claim_task(conn, task_id, claimer="writer")
        auth_id = conn.execute(
            "SELECT id FROM factory_authorizations WHERE task_id=?", (task_id,)
        ).fetchone()[0]

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_FACTORY_EXECUTION_GENERATION", str(task.execution_generation))
    monkeypatch.setenv("HERMES_FACTORY_AUTHORIZATION_ID", auth_id)
    monkeypatch.setenv("HERMES_FACTORY_BINDING_DIGEST", task.project_binding_digest)

    mismatch = _factory_mutation_block(
        "patch", {"path": str(workspace / "module.py")}, "another-task"
    )
    assert mismatch is not None
    assert "task scope mismatch" in mismatch

    process_block = _factory_mutation_block(
        "process", {"action": "kill", "session_id": "bg-1"}, task_id
    )
    assert process_block is not None


def test_factory_terminal_workdir_is_pinned_to_authorized_workspace(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "repo" / "worktree"
    workspace.mkdir(parents=True)
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="terminal guard",
            assignee="john",
            project_id="p_102c8528",
            enforcement_version="factory-v1",
            project_binding=_binding(str(workspace)),
            board="factory",
        )
        task = kb.claim_task(conn, task_id, claimer="writer")
        auth_id = conn.execute(
            "SELECT id FROM factory_authorizations WHERE task_id=?", (task_id,)
        ).fetchone()[0]

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_FACTORY_EXECUTION_GENERATION", str(task.execution_generation))
    monkeypatch.setenv("HERMES_FACTORY_AUTHORIZATION_ID", auth_id)
    monkeypatch.setenv("HERMES_FACTORY_BINDING_DIGEST", task.project_binding_digest)

    args = {"command": "git status --short"}
    assert _factory_mutation_block("terminal", args, task_id) is None
    assert args["workdir"] == str(workspace.resolve())

    denial = _factory_mutation_block(
        "terminal",
        {"command": "git status --short", "workdir": str(tmp_path.parent)},
        task_id,
    )
    assert denial is not None
    assert "outside bound workspace" in denial
