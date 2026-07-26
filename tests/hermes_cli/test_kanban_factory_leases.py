"""Concurrency and recovery fences for FEC-1.0 claims."""

from __future__ import annotations

import time
from pathlib import Path

from hermes_cli import kanban_db as kb


def _binding(workspace: str = "/repo/.worktrees/shared") -> dict:
    return {
        "desktop_project_id": "p_102c8528",
        "board_slug": "factory",
        "repo_root": "/repo",
        "git_common_dir": "/repo/.git",
        "remote_identity": "NousResearch/hermes-agent",
        "workspace_root": workspace,
        "target_ref": "refs/heads/factory/shared",
        "criterion_set_digest": "criteria-sha256",
        "writer_profile": "john",
        "reviewer_profile": "raffe",
        "closer_profile": "janne",
    }


def _task(conn, title: str) -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee="john",
        project_id="p_102c8528",
        enforcement_version="factory-v1",
        project_binding=_binding(),
        board="factory",
    )


def test_second_writer_for_same_resources_is_rejected_without_claiming(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        winner_id = _task(conn, "winner")
        loser_id = _task(conn, "loser")

        winner = kb.claim_task(conn, winner_id, claimer="host:1", ttl_seconds=60)
        loser = kb.claim_task(conn, loser_id, claimer="host:2", ttl_seconds=60)

        assert winner is not None
        assert loser is None
        assert kb.get_task(conn, loser_id).status == "ready"
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_authorizations WHERE task_id=?", (loser_id,)
        ).fetchone()[0] == 0
        assert kb.list_events(conn, loser_id)[-1].kind == "factory_lease_conflict"


def test_heartbeat_requires_exact_current_factory_tuple_and_renews_owned_leases(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id = _task(conn, "heartbeat")
        task = kb.claim_task(conn, task_id, claimer="host:1", ttl_seconds=60)
        auth = conn.execute(
            "SELECT id FROM factory_authorizations WHERE task_id=?", (task_id,)
        ).fetchone()
        before = conn.execute(
            "SELECT MIN(expires_at) FROM factory_resource_leases WHERE task_id=?", (task_id,)
        ).fetchone()[0]

        assert kb.heartbeat_claim(
            conn,
            task_id,
            claimer="host:1",
            ttl_seconds=120,
            expected_run_id=task.current_run_id,
            expected_generation=task.execution_generation,
            authorization_id=auth["id"],
        )
        after = conn.execute(
            "SELECT MIN(expires_at) FROM factory_resource_leases WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        assert after > before

        assert not kb.heartbeat_claim(
            conn,
            task_id,
            claimer="host:1",
            expected_run_id=task.current_run_id,
            expected_generation=task.execution_generation - 1,
            authorization_id=auth["id"],
        )
        assert kb.list_events(conn, task_id)[-1].kind == "factory_stale_callback_denied"


def test_reclaim_revokes_old_authority_and_same_task_reclaims_at_next_generation(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id = _task(conn, "recover")
        first = kb.claim_task(conn, task_id, claimer="host:1", ttl_seconds=60)
        first_run = first.current_run_id
        first_generation = first.execution_generation

        assert kb.reclaim_task(conn, task_id, reason="forced restart")
        old_auth = conn.execute(
            "SELECT revoked_at, revoke_reason FROM factory_authorizations WHERE run_id=?",
            (first_run,),
        ).fetchone()
        released = conn.execute(
            "SELECT COUNT(*) FROM factory_resource_leases "
            "WHERE run_id=? AND released_at IS NOT NULL",
            (first_run,),
        ).fetchone()[0]
        assert old_auth["revoked_at"] is not None
        assert old_auth["revoke_reason"] == "reclaimed"
        assert released == 2

        second = kb.claim_task(conn, task_id, claimer="host:2", ttl_seconds=60)
        assert second is not None
        assert second.id == task_id
        assert second.current_run_id != first_run
        assert second.execution_generation == first_generation + 1
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE id=?", (task_id,)).fetchone()[0] == 1
