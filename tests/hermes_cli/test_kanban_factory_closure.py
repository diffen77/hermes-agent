"""Immutable FEC artifacts, evidence, review, and closure gates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _binding() -> dict:
    return {
        "desktop_project_id": "p_102c8528",
        "board_slug": "factory",
        "repo_root": "/repo",
        "git_common_dir": "/repo/.git",
        "remote_identity": "NousResearch/hermes-agent",
        "workspace_root": "/repo/.worktrees/task",
        "target_ref": "refs/heads/factory/task",
        "criterion_set_digest": "criteria-sha256",
        "writer_profile": "john",
        "reviewer_profile": "raffe",
        "closer_profile": "janne",
    }


def _claimed_factory(conn):
    task_id = kb.create_task(
        conn,
        title="deliver factory contract",
        assignee="john",
        project_id="p_102c8528",
        enforcement_version="factory-v1",
        project_binding=_binding(),
        board="factory",
    )
    task = kb.claim_task(conn, task_id, claimer="writer")
    auth = conn.execute(
        "SELECT id FROM factory_authorizations WHERE task_id=? AND revoked_at IS NULL",
        (task_id,),
    ).fetchone()[0]
    return task_id, task, auth


def test_factory_completion_fails_closed_without_frozen_artifacts_review_and_closure_auth(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id, task, _auth = _claimed_factory(conn)

        with pytest.raises(kb.FactoryEnforcementError, match="closure contract incomplete"):
            kb.complete_task(conn, task_id, summary="not actually verified", expected_run_id=task.current_run_id)

        assert kb.get_task(conn, task_id).status == "running"
        assert kb.list_events(conn, task_id)[-1].kind == "factory_closure_denied"


def test_factory_artifact_rejects_remote_readback_that_differs_from_commit(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id, task, writer_auth = _claimed_factory(conn)

        with pytest.raises(kb.FactoryEnforcementError, match="remote readback"):
            kb.publish_factory_artifact(
                conn,
                task_id,
                run_id=task.current_run_id,
                generation=task.execution_generation,
                authorization_id=writer_auth,
                commit_sha="a" * 40,
                tree_sha="b" * 40,
                diff_base_sha="c" * 40,
                diff_or_patch_sha256="d" * 64,
                remote_readback_sha="e" * 40,
                classified_files=["hermes_cli/kanban_db.py"],
                excluded_files_by_class={},
            )


def test_factory_delivery_requires_immutable_artifact_evidence_review_and_closer_chain(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id, writer_task, writer_auth = _claimed_factory(conn)
        artifact_id = kb.publish_factory_artifact(
            conn,
            task_id,
            run_id=writer_task.current_run_id,
            generation=writer_task.execution_generation,
            authorization_id=writer_auth,
            commit_sha="a" * 40,
            tree_sha="b" * 40,
            diff_base_sha="c" * 40,
            diff_or_patch_sha256=_digest("patch"),
            remote_readback_sha="a" * 40,
            classified_files=["hermes_cli/kanban_db.py"],
            excluded_files_by_class={"generated": ["coverage.xml"]},
        )
        artifact_set_digest = kb.freeze_factory_artifacts(
            conn,
            task_id,
            run_id=writer_task.current_run_id,
            generation=writer_task.execution_generation,
            authorization_id=writer_auth,
        )
        for criterion_id in kb.FACTORY_CRITERIA:
            kb.submit_factory_evidence(
                conn,
                task_id,
                run_id=writer_task.current_run_id,
                generation=writer_task.execution_generation,
                authorization_id=writer_auth,
                criterion_id=criterion_id,
                evidence_type="test_report",
                source_uri=f"test://{criterion_id}",
                digest=_digest(criterion_id),
            )

        assert kb.block_task(conn, task_id, reason="review-required")
        review_auth = kb.claim_factory_role(conn, task_id, profile="raffe", role="review")
        review_id = kb.submit_factory_review(
            conn,
            task_id,
            run_id=review_auth["run_id"],
            generation=review_auth["execution_generation"],
            authorization_id=review_auth["authorization_id"],
            reviewer_profile="raffe",
            decision="accepted",
            notes="all 14 criteria independently checked",
        )
        closure_auth = kb.claim_factory_role(conn, task_id, profile="janne", role="closure")

        assert kb.complete_task(
            conn,
            task_id,
            summary="FEC-1.0 verified and preserved",
            expected_run_id=closure_auth["run_id"],
            expected_generation=closure_auth["execution_generation"],
            authorization_id=closure_auth["authorization_id"],
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "done"
        assert task.artifact_set_digest == artifact_set_digest
        assert task.accepted_review_id == review_id
        assert task.closure_authorization_id is not None

        artifact = conn.execute(
            "SELECT * FROM factory_artifacts WHERE id=?", (artifact_id,)
        ).fetchone()
        review = conn.execute(
            "SELECT * FROM factory_reviews WHERE id=?", (review_id,)
        ).fetchone()
        assert artifact["binding_digest"] == task.project_binding_digest
        assert review["artifact_manifest_sha256"] == artifact_set_digest
        assert review["reviewer_profile"] == "raffe"
        assert conn.execute(
            "SELECT COUNT(*) FROM factory_evidence WHERE task_id=?", (task_id,)
        ).fetchone()[0] == 14


def test_factory_artifact_rejects_incomplete_or_non_readback_manifest(tmp_path: Path):
    with kb.connect(tmp_path / "kanban.db") as conn:
        task_id, writer_task, writer_auth = _claimed_factory(conn)

        with pytest.raises((TypeError, ValueError, kb.FactoryEnforcementError)):
            kb.publish_factory_artifact(
                conn,
                task_id,
                run_id=writer_task.current_run_id,
                generation=writer_task.execution_generation,
                authorization_id=writer_auth,
                commit_sha="a" * 40,
                tree_sha="b" * 40,
                diff_base_sha="c" * 40,
                diff_or_patch_sha256=_digest("patch"),
                remote_readback_sha="d" * 40,
                classified_files=["hermes_cli/kanban_db.py"],
                excluded_files_by_class={},
            )

        assert conn.execute(
            "SELECT COUNT(*) FROM factory_artifacts WHERE task_id=?", (task_id,)
        ).fetchone()[0] == 0
