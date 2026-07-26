"""Behavior tests for the opt-in FEC-1.0 Kanban persistence contract."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def factory_db(tmp_path: Path):
    path = tmp_path / "kanban.db"
    conn = kb.connect(path)
    try:
        yield conn, path
    finally:
        conn.close()


def test_factory_schema_is_additive_and_legacy_tasks_remain_unenforced(factory_db):
    conn, _ = factory_db

    task_id = kb.create_task(conn, title="legacy")
    task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.enforcement_version is None
    assert task.project_binding_digest is None
    assert task.execution_generation == 0
    assert task.current_artifact_generation == 0

    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "factory_task_bindings",
        "factory_authorizations",
        "factory_resource_leases",
        "factory_artifacts",
        "factory_evidence",
        "factory_reviews",
    } <= tables


def test_factory_schema_migrates_old_tasks_idempotently(tmp_path: Path):
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    raw.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES ('legacy', 'legacy', 'ready', 1)"
    )
    raw.commit()
    raw.close()

    kb.init_db(path)
    kb.init_db(path)

    with kb.connect(path) as conn:
        row = conn.execute(
            "SELECT enforcement_version, project_binding_digest, "
            "execution_generation, current_artifact_generation FROM tasks WHERE id='legacy'"
        ).fetchone()
        assert tuple(row) == (None, None, 0, 0)


def test_factory_history_tables_reject_update_and_delete(factory_db):
    conn, _ = factory_db
    task_id = kb.create_task(conn, title="bound")
    conn.execute(
        """
        INSERT INTO factory_task_bindings (
            task_id, version, desktop_project_id, board_slug, repo_root,
            git_common_dir, remote_identity, workspace_root, target_ref,
            binding_json, binding_digest, criterion_set_digest,
            writer_profile, reviewer_profile, closer_profile, created_at
        ) VALUES (?, 'factory-v1', 'p_1', 'default', '/repo', '/repo/.git',
                  'NousResearch/hermes-agent', '/repo/.worktrees/t', 'refs/heads/fec',
                  '{}', 'binding-digest', 'criteria-digest', 'john', 'raffe', 'janne', 1)
        """,
        (task_id,),
    )

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE factory_task_bindings SET target_ref='refs/heads/other' WHERE task_id=?",
            (task_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM factory_task_bindings WHERE task_id=?", (task_id,))
