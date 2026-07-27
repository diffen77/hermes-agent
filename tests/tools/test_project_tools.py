"""Behavior tests for session-scoped project agent tools."""

from __future__ import annotations

import json

import pytest

from hermes_cli import projects_db as pdb
from tools import project_tools


@pytest.fixture(autouse=True)
def _reset_project_callbacks(monkeypatch):
    monkeypatch.setattr(project_tools, "_workspace_callback", None)
    monkeypatch.setattr(project_tools, "_active_project_callback", None, raising=False)


def _create(name: str, path: str | None = None) -> str:
    with pdb.connect_closing() as conn:
        return pdb.create_project(
            conn,
            name=name,
            folders=[path] if path else [],
            primary_path=path,
        )


def _global_active() -> str | None:
    with pdb.connect_closing() as conn:
        return pdb.get_active_id(conn)


def test_task_switches_are_isolated_without_global_marker_drift(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()
    project_a = _create("A", str(path_a))
    project_b = _create("B", str(path_b))
    with pdb.connect_closing() as conn:
        pdb.set_active(conn, project_a)

    task_paths = {"task-a": str(path_a), "task-b": str(path_a)}
    project_tools.set_project_workspace_callback(
        lambda task_id, path, _name: task_paths.__setitem__(task_id, path)
    )

    def active_for_task(task_id: str) -> str | None:
        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, task_paths[task_id])
            return project.id if project else None

    project_tools.set_project_active_callback(active_for_task)

    assert (
        json.loads(project_tools.project_switch("B", task_id="task-a"))["success"]
        is True
    )
    assert _global_active() == project_a
    assert json.loads(project_tools.project_list(task_id="task-a"))["active_id"] == project_b
    assert json.loads(project_tools.project_list(task_id="task-b"))["active_id"] == project_a

    assert (
        json.loads(project_tools.project_switch("B", task_id="task-b"))["success"]
        is True
    )
    assert _global_active() == project_a
    assert json.loads(project_tools.project_list(task_id="task-a"))["active_id"] == project_b
    assert json.loads(project_tools.project_list(task_id="task-b"))["active_id"] == project_b


def test_task_create_reanchors_session_without_changing_global_active(tmp_path):
    original_path = tmp_path / "original"
    created_path = tmp_path / "created"
    original_path.mkdir()
    created_path.mkdir()
    original = _create("Original", str(original_path))
    with pdb.connect_closing() as conn:
        pdb.set_active(conn, original)

    moved = {}
    project_tools.set_project_workspace_callback(
        lambda task_id, path, _name: moved.__setitem__(task_id, path)
    )

    result = json.loads(
        project_tools.project_create("Created", str(created_path), task_id="task-a")
    )

    assert result["success"] is True
    assert moved == {"task-a": str(created_path)}
    assert _global_active() == original


def test_task_list_fails_closed_when_session_project_is_unresolved(tmp_path):
    project = _create("Global", str(tmp_path / "global"))
    with pdb.connect_closing() as conn:
        pdb.set_active(conn, project)
    project_tools.set_project_active_callback(lambda _task_id: None)

    listing = json.loads(project_tools.project_list(task_id="missing-task"))

    assert listing["active_id"] is None
    assert all(item["active"] is False for item in listing["projects"])


def test_legacy_calls_without_task_id_keep_global_active_semantics(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()
    project_a = _create("A", str(path_a))
    project_b = _create("B", str(path_b))

    assert json.loads(project_tools.project_switch("A"))["success"] is True
    assert _global_active() == project_a
    assert json.loads(project_tools.project_list())["active_id"] == project_a

    assert json.loads(project_tools.project_switch("B"))["success"] is True
    assert _global_active() == project_b

    created = json.loads(project_tools.project_create("C"))
    assert created["success"] is True
    assert _global_active() == created["id"]
