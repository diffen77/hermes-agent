"""Identity, role, authorization, and lease behavior for FEC-1.0 tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _binding(workspace: str = "/repo/.worktrees/task") -> dict:
    return {
        "desktop_project_id": "p_102c8528",
        "board_slug": "factory",
        "repo_root": "/repo",
        "git_common_dir": "/repo/.git",
        "remote_identity": "NousResearch/hermes-agent",
        "workspace_root": workspace,
        "target_ref": "refs/heads/factory/task",
        "criterion_set_digest": "criteria-sha256",
        "writer_profile": "john",
        "reviewer_profile": "raffe",
        "closer_profile": "janne",
    }


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def test_factory_task_creation_binds_exact_identity_and_legacy_fallback_is_unchanged(conn):
    legacy = kb.create_task(conn, title="legacy", project_id="missing")
    assert kb.get_task(conn, legacy).project_id is None

    with pytest.raises(kb.FactoryEnforcementError, match="explicit project binding"):
        kb.create_task(
            conn,
            title="invalid factory task",
            assignee="john",
            project_id="p_102c8528",
            enforcement_version="factory-v1",
        )

    task_id = kb.create_task(
        conn,
        title="factory task",
        assignee="john",
        project_id="p_102c8528",
        enforcement_version="factory-v1",
        project_binding=_binding(),
        board="factory",
    )
    task = kb.get_task(conn, task_id)
    binding = conn.execute(
        "SELECT * FROM factory_task_bindings WHERE task_id=?", (task_id,)
    ).fetchone()

    assert task.enforcement_version == "factory-v1"
    assert task.project_id == "p_102c8528"
    assert task.project_binding_digest == binding["binding_digest"]
    canonical = json.loads(binding["binding_json"])
    assert canonical["task_id"] == task_id
    assert canonical["desktop_project_id"] == task.project_id
    assert canonical["workspace_root"] == "/repo/.worktrees/task"
    assert kb.list_events(conn, task_id)[-1].kind == "factory_identity_bound"


def test_factory_claim_issues_run_scoped_writer_authority_and_two_resource_leases(conn):
    task_id = kb.create_task(
        conn,
        title="factory task",
        assignee="john",
        project_id="p_102c8528",
        enforcement_version="factory-v1",
        project_binding=_binding(),
        board="factory",
    )

    claimed = kb.claim_task(conn, task_id, claimer="host:1", ttl_seconds=60)
    assert claimed is not None
    assert claimed.execution_generation == 1

    auth = conn.execute(
        "SELECT * FROM factory_authorizations WHERE task_id=?", (task_id,)
    ).fetchone()
    assert auth["run_id"] == claimed.current_run_id
    assert auth["profile"] == "john"
    assert auth["role"] == "implementation"
    assert json.loads(auth["allowed_effects_json"]) == [
        "artifact_publish",
        "commit",
        "push",
        "source_write",
    ]
    leases = conn.execute(
        "SELECT * FROM factory_resource_leases WHERE task_id=? ORDER BY resource_key",
        (task_id,),
    ).fetchall()
    assert len(leases) == 2
    assert {row["execution_generation"] for row in leases} == {1}
    assert {row["authorization_id"] for row in leases} == {auth["id"]}


def test_factory_claim_denies_wrong_role_before_transition(conn):
    task_id = kb.create_task(
        conn,
        title="wrong writer",
        assignee="raffe",
        project_id="p_102c8528",
        enforcement_version="factory-v1",
        project_binding=_binding(),
        board="factory",
    )

    with pytest.raises(kb.FactoryEnforcementError, match="role authority denied"):
        kb.claim_task(conn, task_id, claimer="host:2")

    assert kb.get_task(conn, task_id).status == "ready"
    events = kb.list_events(conn, task_id)
    assert events[-1].kind == "factory_authority_denied"
