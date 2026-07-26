from __future__ import annotations

import json

from hermes_cli import projects_db as pdb
from tools import project_tools


def _projects(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with pdb.connect_closing() as conn:
        first = pdb.create_project(conn, name="First", folders=[str(tmp_path / "first")])
        second = pdb.create_project(conn, name="Second", folders=[str(tmp_path / "second")])
        pdb.set_active(conn, first)
    return first, second


def test_task_scoped_project_switch_preserves_global_active_id(tmp_path, monkeypatch):
    first, second = _projects(tmp_path, monkeypatch)
    moved = []
    project_tools.set_project_workspace_callback(lambda *args: moved.append(args))
    try:
        result = json.loads(project_tools.project_switch(second, task_id="session-1"))
    finally:
        project_tools.set_project_workspace_callback(None)
    assert result["success"] is True
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == first
    assert moved and moved[0][0] == "session-1"


def test_legacy_project_switch_without_task_id_updates_global_active_id(tmp_path, monkeypatch):
    _first, second = _projects(tmp_path, monkeypatch)
    result = json.loads(project_tools.project_switch(second))
    assert result["success"] is True
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == second


def test_task_scoped_project_create_preserves_global_active_id(tmp_path, monkeypatch):
    first, _second = _projects(tmp_path, monkeypatch)
    moved = []
    project_tools.set_project_workspace_callback(lambda *args: moved.append(args))
    try:
        result = json.loads(project_tools.project_create(
            "Third", str(tmp_path / "third"), task_id="session-1",
        ))
    finally:
        project_tools.set_project_workspace_callback(None)
    assert result["success"] is True
    with pdb.connect_closing() as conn:
        assert pdb.get_active_id(conn) == first
    assert moved and moved[0][0] == "session-1"