from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from urllib.parse import quote

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
            "process_identity": {
                "classification": "pid_reused_or_birth_mismatch",
                "verdict": "NOT_VERIFIED",
                "reclaim_authority": "NOT_AUTHORIZED",
            },
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
    assert alpha["stale_running"] == {
        "count": 1,
        "task_ids": [running],
        "omitted": 0,
    }
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


def test_mixed_project_registry_failure_cannot_establish_project_absence(
    tmp_path: Path,
) -> None:
    from hermes_cli.kanban_fleet import _known_project_ids, classify_fixture_candidates

    root = tmp_path / ".hermes"
    root.mkdir()
    with sqlite3.connect(root / "projects.db") as conn:
        conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO projects (id) VALUES ('p_other')")
    lost_registry = root / "profiles" / "lost" / "projects.db"
    lost_registry.parent.mkdir(parents=True)
    lost_registry.write_bytes(b"not a sqlite database")
    rows, links = _delivery_fixture_rows(tmp_path, "p_102c8528")

    project_ids, projects_complete = _known_project_ids(root)

    assert project_ids == {"p_other"}
    assert projects_complete is False
    assert classify_fixture_candidates(
        rows,
        links,
        known_profiles={"john", "raffe", "janne"},
        known_project_ids=project_ids,
        projects_complete=projects_complete,
    ) == {}


@pytest.mark.parametrize(
    "registry_relative",
    (Path("projects.db"), Path("profiles") / "worker" / "projects.db"),
)
def test_project_registry_observation_drift_is_bounded_and_not_verified(
    fleet_home: Path, monkeypatch, registry_relative: Path
) -> None:
    from hermes_cli import kanban_fleet as fleet

    registry = fleet_home / registry_relative
    registry.parent.mkdir(parents=True, exist_ok=True)
    writer = sqlite3.connect(registry)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")
    writer.execute("INSERT INTO projects (id) VALUES ('p_known')")
    writer.commit()
    secret = "registry-race-ghp_" + "A" * 24
    committed = False

    def commit_before_registry_snapshot(path: Path, immutable: bool) -> None:
        nonlocal committed
        if path == registry.resolve() and not committed:
            writer.execute("INSERT INTO projects (id) VALUES (?)", (secret,))
            writer.commit()
            committed = True

    monkeypatch.setattr(fleet, "_after_readonly_connect", commit_before_registry_snapshot)
    try:
        ledger = fleet.build_fleet_ledger(root=fleet_home, now=123)
    finally:
        writer.close()

    assert committed
    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["projects"] is False
    assert ledger["errors"] == {
        "count": 1,
        "items": [{"scope": "projects", "code": "project_registry_read_failed"}],
        "omitted": 0,
    }
    assert secret not in json.dumps(ledger)


def test_project_registry_programming_errors_propagate(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    registry = fleet_home / "projects.db"
    with sqlite3.connect(registry) as conn:
        conn.execute("CREATE TABLE projects (id TEXT PRIMARY KEY)")

    original_connect = fleet._readonly_connect

    def selective_programming_error(path: Path) -> sqlite3.Connection:
        if path == registry:
            raise RuntimeError("programming error")
        return original_connect(path)

    monkeypatch.setattr(fleet, "_readonly_connect", selective_programming_error)

    with pytest.raises(RuntimeError, match="programming error"):
        fleet.build_fleet_ledger(root=fleet_home, now=123)


def test_mixed_profile_discovery_failure_cannot_establish_profile_absence(
    tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli.kanban_fleet import _known_profile_names, classify_fixture_candidates

    root = tmp_path / ".hermes"
    healthy_config = root / "profiles" / "john" / "config.yaml"
    healthy_config.parent.mkdir(parents=True)
    healthy_config.write_text("model: {}\n")
    lost_config = root / "profiles" / "lost" / "config.yaml"
    lost_config.parent.mkdir(parents=True)
    lost_config.write_text("model: {}\n")
    original_stat = Path.stat

    def partial_stat(path: Path, *args, **kwargs):
        if path == lost_config:
            raise PermissionError("profile config is unreadable")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", partial_stat)
    rows, links = _delivery_fixture_rows(tmp_path, "p_102c8528")

    profile_names, profiles_complete = _known_profile_names(root)

    assert profile_names == {"default", "john"}
    assert profiles_complete is False
    assert classify_fixture_candidates(
        rows,
        links,
        known_profiles=profile_names,
        known_project_ids={"p_other"},
        profiles_complete=profiles_complete,
    ) == {}


def test_empty_project_registry_discovery_is_complete_but_fails_closed(
    tmp_path: Path,
) -> None:
    from hermes_cli.kanban_fleet import _known_project_ids, classify_fixture_candidates

    root = tmp_path / ".hermes"
    root.mkdir()
    rows, links = _delivery_fixture_rows(tmp_path, "p_102c8528")

    project_ids, projects_complete = _known_project_ids(root)

    assert project_ids == set()
    assert projects_complete is True
    assert classify_fixture_candidates(
        rows,
        links,
        known_profiles={"john", "raffe", "janne"},
        known_project_ids=project_ids,
        projects_complete=projects_complete,
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


def test_incomplete_discovery_prevents_false_green_verification(
    fleet_home: Path, monkeypatch, capsys
) -> None:
    from hermes_cli import kanban_fleet as fleet

    original_stat = Path.stat

    def denied(path: Path, *args, **kwargs):
        if path == fleet_home / "profiles":
            raise PermissionError("secret path must not escape")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    ledger = fleet.build_fleet_ledger(root=fleet_home, now=123)

    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["profiles"] is False
    assert ledger["observation_generation"]
    assert ledger["errors"]["count"] >= 1
    assert "secret path must not escape" not in json.dumps(ledger)

    args = type("Args", (), {"stale_after": "1h", "limit": 20, "json": True, "verify": True})()
    monkeypatch.setattr(fleet, "build_fleet_ledger", lambda **_: ledger)
    assert kc._cmd_fleet(args) != 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "NOT_VERIFIED"


def test_board_discovery_permission_error_is_bounded_and_not_verified(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    boards_dir = fleet_home / "kanban" / "boards"
    original_iterdir = Path.iterdir

    def denied(path: Path):
        if path == boards_dir:
            raise PermissionError("do not reveal this detail")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    ledger = fleet.build_fleet_ledger(root=fleet_home, now=123, limit=1)

    assert ledger["completeness"]["boards"] is False
    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["errors"]["count"] == 1
    assert len(ledger["errors"]["items"]) <= 1
    assert "do not reveal this detail" not in json.dumps(ledger)


def test_discovery_errors_remain_bounded(fleet_home: Path) -> None:
    from hermes_cli.kanban_fleet import build_fleet_ledger

    boards_dir = fleet_home / "kanban" / "boards"
    for index in range(30):
        (boards_dir / f"invalid-{index:02d}" / "kanban.db").mkdir(parents=True)

    ledger = build_fleet_ledger(
        root=fleet_home,
        known_profiles={"default"},
        known_project_ids={"p_known"},
        now=123,
        limit=100,
    )

    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["errors"]["count"] == 30
    assert len(ledger["errors"]["items"]) == 20
    assert ledger["errors"]["omitted"] == 10


def test_board_discovery_deduplicates_symlink_and_hardlink_aliases(
    fleet_home: Path,
) -> None:
    from hermes_cli.kanban_fleet import build_fleet_ledger

    alpha = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    alias_dir = fleet_home / "kanban" / "boards" / "alpha-alias"
    alias_dir.mkdir()
    try:
        os.link(alpha, alias_dir / "kanban.db")
    except OSError:
        (alias_dir / "kanban.db").symlink_to(alpha)

    ledger = build_fleet_ledger(
        root=fleet_home,
        known_profiles={"default"},
        known_project_ids={"p_known"},
        now=123,
    )

    assert [board["slug"] for board in ledger["boards"]] == ["alpha", "beta"]
    assert ledger["totals"]["boards"] == 2


def test_board_snapshot_is_coherent_under_concurrent_write(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    with kb.connect_closing(board="alpha") as conn:
        parent = kb.create_task(conn, title="parent")
    writer_errors: list[BaseException] = []

    def write_after_tasks(slug: str) -> None:
        if slug != "alpha":
            return

        def writer() -> None:
            try:
                with kb.connect_closing(board="alpha") as conn:
                    child = kb.create_task(conn, title="concurrent child", parents=(parent,))
                    assert child
            except BaseException as exc:  # pragma: no cover - asserted below
                writer_errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    monkeypatch.setattr(fleet, "_after_tasks_snapshot", write_after_tasks)
    ledger = fleet.build_fleet_ledger(
        root=fleet_home,
        known_profiles={"default"},
        known_project_ids={"p_known"},
        now=123,
    )

    assert writer_errors == []
    alpha = ledger["boards"][0]
    assert alpha["counts"] == {"ready": 1}
    assert alpha["snapshot"]["verdict"] == "VERIFIED"


def test_readonly_connect_never_verifies_state_stale_before_connect(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    alpha_path = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    writers: list[sqlite3.Connection] = []

    def commit_between_preflight_and_connect(path: Path, immutable: bool) -> None:
        if path != alpha_path.resolve() or not immutable or writers:
            return
        writer = kb.connect(board="alpha")
        writers.append(writer)
        task_id = kb.create_task(writer, title="committed before observer connect")
        assert task_id

    monkeypatch.setattr(
        fleet, "_before_readonly_connect", commit_between_preflight_and_connect, raising=False
    )
    try:
        ledger = fleet.build_fleet_ledger(
            root=fleet_home,
            known_profiles={"default"},
            known_project_ids={"p_known"},
            now=123,
        )
    finally:
        for writer in writers:
            writer.close()

    assert writers
    if ledger["verdict"] == "VERIFIED":
        alpha = next(board for board in ledger["boards"] if board["slug"] == "alpha")
        assert alpha["counts"] == {"ready": 1}
    else:
        assert ledger["completeness"]["boards"] is False
        assert {item["code"] for item in ledger["errors"]["items"]} & {
            "board_snapshot_drift",
            "board_snapshot_read_failed",
        }


def test_readonly_connect_pins_snapshot_before_returning(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    alpha_path = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    writers: list[sqlite3.Connection] = []

    def commit_after_connect_before_snapshot(path: Path, immutable: bool) -> None:
        if path != alpha_path.resolve() or writers:
            return
        writer = kb.connect(board="alpha")
        writers.append(writer)
        task_id = kb.create_task(writer, title="committed before snapshot pin")
        assert task_id

    monkeypatch.setattr(
        fleet,
        "_after_readonly_connect",
        commit_after_connect_before_snapshot,
        raising=False,
    )
    try:
        ledger = fleet.build_fleet_ledger(
            root=fleet_home,
            known_profiles={"default"},
            known_project_ids={"p_known"},
            now=123,
        )
    finally:
        for writer in writers:
            writer.close()

    assert writers
    if ledger["verdict"] == "VERIFIED":
        alpha = next(board for board in ledger["boards"] if board["slug"] == "alpha")
        assert alpha["counts"] == {"ready": 1}
    else:
        assert ledger["completeness"]["boards"] is False
        assert {item["code"] for item in ledger["errors"]["items"]} & {
            "board_snapshot_drift",
            "board_snapshot_read_failed",
        }


def test_readonly_connect_fails_closed_if_active_wal_disappears_before_open(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    alpha_path = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    writer = kb.connect(board="alpha")
    task_id = kb.create_task(writer, title="checkpointed during observer open")
    assert task_id
    assert Path(f"{alpha_path}-wal").exists()

    def close_writer_before_open(path: Path, immutable: bool) -> None:
        if path == alpha_path.resolve() and not immutable:
            writer.close()

    monkeypatch.setattr(fleet, "_before_readonly_connect", close_writer_before_open)
    try:
        ledger = fleet.build_fleet_ledger(
            root=fleet_home,
            known_profiles={"default"},
            known_project_ids={"p_known"},
            now=123,
        )
    finally:
        try:
            writer.close()
        except sqlite3.ProgrammingError:
            pass

    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["boards"] is False
    assert [board["slug"] for board in ledger["boards"]] == ["beta"]
    assert ledger["errors"]["items"] == [
        {"scope": "boards", "code": "board_snapshot_drift"}
    ]


def test_readonly_connect_fails_closed_if_wal_changes_inside_sqlite_open(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    alpha_path = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    real_connect = sqlite3.connect
    writers: list[sqlite3.Connection] = []

    def connect_with_commit(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        target = str(args[0]) if args else ""
        if quote(str(alpha_path.resolve())) in target and not writers:
            writer = real_connect(alpha_path)
            writer.execute(
                "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
                ("race", "race", "ready", 123),
            )
            writer.commit()
            writers.append(writer)
        return conn

    monkeypatch.setattr(fleet.sqlite3, "connect", connect_with_commit)
    try:
        ledger = fleet.build_fleet_ledger(
            root=fleet_home,
            known_profiles={"default"},
            known_project_ids={"p_known"},
            now=123,
        )
    finally:
        for writer in writers:
            writer.close()

    assert writers
    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["boards"] is False
    assert [board["slug"] for board in ledger["boards"]] == ["beta"]
    assert ledger["errors"]["items"] == [
        {"scope": "boards", "code": "board_snapshot_drift"}
    ]


def test_board_snapshot_rejects_identity_drift(
    fleet_home: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    original_identity = fleet._path_identity
    alpha = fleet_home / "kanban" / "boards" / "alpha" / "kanban.db"
    alpha_calls = 0

    def drifting_identity(path: Path) -> tuple[int, int]:
        nonlocal alpha_calls
        identity = original_identity(path)
        if path == alpha:
            alpha_calls += 1
            if alpha_calls > 1:
                return identity[0], identity[1] + 1
        return identity

    monkeypatch.setattr(fleet, "_path_identity", drifting_identity)
    ledger = fleet.build_fleet_ledger(
        root=fleet_home,
        known_profiles={"default"},
        known_project_ids={"p_known"},
        now=123,
    )

    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["boards"] is False
    assert [board["slug"] for board in ledger["boards"]] == ["beta"]
    assert ledger["errors"]["items"] == [
        {"scope": "boards", "code": "board_snapshot_drift"}
    ]


def test_malformed_dynamic_owner_metadata_omits_board_and_fails_closed(
    fleet_home: Path
) -> None:
    from hermes_cli import kanban_fleet as fleet

    secret = "not-a-pid-ghp_" + "A" * 24
    with kb.connect_closing(board="alpha") as conn:
        task_id = kb.create_task(conn, title="malformed owner")
        assert kb.claim_task(conn, task_id, claimer="default") is not None
        conn.execute("UPDATE tasks SET worker_pid = ? WHERE id = ?", (secret, task_id))
        conn.execute(
            "UPDATE task_runs SET worker_pid = NULL WHERE task_id = ? AND ended_at IS NULL",
            (task_id,),
        )

    ledger = fleet.build_fleet_ledger(
        root=fleet_home,
        known_profiles={"default"},
        known_project_ids={"p_known"},
        now=123,
    )

    assert ledger["verdict"] == "NOT_VERIFIED"
    assert ledger["completeness"]["boards"] is False
    assert [board["slug"] for board in ledger["boards"]] == ["beta"]
    assert ledger["errors"]["items"] == [
        {"scope": "boards", "code": "board_snapshot_read_failed"}
    ]
    assert secret not in json.dumps(ledger)


def test_pid_reuse_and_permission_ambiguity_are_never_dead(monkeypatch) -> None:
    from hermes_cli import kanban_fleet as fleet

    monkeypatch.setattr(
        fleet,
        "_read_process_identity",
        lambda pid: {"state": "present", "birth_epoch": 500},
    )
    reused = fleet._classify_process_identity(77, expected_started_at=100)
    monkeypatch.setattr(
        fleet,
        "_read_process_identity",
        lambda pid: {"state": "permission_denied"},
    )
    denied = fleet._classify_process_identity(77, expected_started_at=100)

    assert reused["classification"] == "pid_reused_or_birth_mismatch"
    assert reused["verdict"] == "NOT_VERIFIED"
    assert denied == {
        "classification": "permission_denied",
        "verdict": "NOT_VERIFIED",
        "reclaim_authority": "NOT_AUTHORIZED",
    }
    assert "dead" not in json.dumps([reused, denied]).lower()


def _create_recovery_fixture(
    fleet_home: Path, tmp_path: Path, *, board: str = "alpha"
) -> list[str]:
    with sqlite3.connect(fleet_home / "projects.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO projects (id) VALUES ('p_real')")
    root = tmp_path / "safe-recovery-fixture"
    task_ids: list[str] = []
    parent = None
    with kb.connect_closing(board=board) as conn:
        for step, assignee in (
            ("implementation", "builder"),
            ("verifier", "reviewer"),
            ("closure", "closer"),
        ):
            task_id = kb.create_task(
                conn,
                title=f"Recovery fixture {step}",
                assignee=assignee,
                created_by="orchestrator",
                parents=(parent,) if parent else (),
                workspace_kind="worktree",
                workspace_path=str(root / ".worktrees" / step),
            )
            conn.execute(
                "UPDATE tasks SET project_id='p_fixture', created_at=123, "
                "workflow_template_id='delivery_v1', current_step_key=? WHERE id=?",
                (step, task_id),
            )
            task_ids.append(task_id)
            parent = task_id
    return task_ids


def _task_statuses(board: str, task_ids: list[str]) -> dict[str, str]:
    with kb.connect_closing(board=board) as conn:
        return {
            str(row["id"]): str(row["status"])
            for row in conn.execute(
                f"SELECT id, status FROM tasks WHERE id IN ({','.join('?' for _ in task_ids)})",
                task_ids,
            )
        }


def test_safe_recovery_is_leaf_first_bounded_and_idempotent(
    fleet_home: Path, tmp_path: Path
) -> None:
    from hermes_cli.kanban_fleet import apply_safe_recovery

    task_ids = _create_recovery_fixture(fleet_home, tmp_path)
    implementation, verifier, closure = task_ids

    first = apply_safe_recovery(root=fleet_home, now=1_000)
    assert first["verdict"] == "APPLIED"
    assert first["mutations"] == 1, first
    assert max(board["mutations"] for board in first["boards"]) <= 1
    assert _task_statuses("alpha", task_ids) == {
        implementation: "ready",
        verifier: "todo",
        closure: "archived",
    }

    second = apply_safe_recovery(root=fleet_home, now=1_001)
    third = apply_safe_recovery(root=fleet_home, now=1_002)
    fourth = apply_safe_recovery(root=fleet_home, now=1_003)

    assert second["mutations"] == 1
    assert third["mutations"] == 1
    assert fourth["mutations"] == 0
    assert set(_task_statuses("alpha", task_ids).values()) == {"archived"}
    receipts = sorted((fleet_home / "kanban" / "reconciler-receipts").glob("*.json"))
    assert len(receipts) == 3
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in receipts)
    assert all(json.loads(path.read_text())["action"] == "fixture_quarantine" for path in receipts)
    with kb.connect_closing(board="alpha") as conn:
        events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='archived' "
            "AND json_extract(payload, '$.source')='fleet_safe_recovery'"
        ).fetchone()[0]
    assert events == 3


def test_safe_recovery_restart_after_commit_recovers_receipt_without_remutation(
    fleet_home: Path, tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    task_ids = _create_recovery_fixture(fleet_home, tmp_path)
    injected = False

    def crash_after_commit(slug: str, task_id: str) -> None:
        nonlocal injected
        injected = True
        raise RuntimeError("simulated process loss")

    monkeypatch.setattr(fleet, "_after_recovery_commit", crash_after_commit)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        fleet.apply_safe_recovery(root=fleet_home, now=2_000)
    assert injected is True
    assert list(_task_statuses("alpha", task_ids).values()).count("archived") == 1
    receipt_dir = fleet_home / "kanban" / "reconciler-receipts"
    assert not receipt_dir.exists() or list(receipt_dir.glob("*.json")) == []

    monkeypatch.setattr(fleet, "_after_recovery_commit", lambda slug, task_id: None)
    recovered = fleet.apply_safe_recovery(root=fleet_home, now=2_001)

    assert recovered["mutations"] == 0
    assert any(board["outcome"] == "receipt_recovered" for board in recovered["boards"])
    assert list(_task_statuses("alpha", task_ids).values()).count("archived") == 1
    assert len(list(receipt_dir.glob("*.json"))) == 1


def test_safe_recovery_denies_delegated_child_without_writes(
    fleet_home: Path, tmp_path: Path, monkeypatch
) -> None:
    from agent.delegation_context import delegated_child_context
    from hermes_cli.kanban_fleet import apply_safe_recovery

    _create_recovery_fixture(fleet_home, tmp_path)
    before = _tree_fingerprint(fleet_home)

    with delegated_child_context():
        result = apply_safe_recovery(root=fleet_home, now=3_000)

    assert result == {
        "verdict": "DENIED",
        "reason": "delegated_child_context",
        "mutations": 0,
        "boards": [],
    }
    assert _tree_fingerprint(fleet_home) == before

    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    subprocess_result = apply_safe_recovery(root=fleet_home, now=3_001)
    assert subprocess_result == result
    assert _tree_fingerprint(fleet_home) == before


@pytest.mark.parametrize("unsafe_status", ["running", "blocked", "review"])
def test_safe_recovery_never_mutates_live_or_human_gated_fixture(
    fleet_home: Path, tmp_path: Path, unsafe_status: str
) -> None:
    from hermes_cli.kanban_fleet import apply_safe_recovery

    task_ids = _create_recovery_fixture(fleet_home, tmp_path)
    closure = task_ids[-1]
    with kb.connect_closing(board="alpha") as conn:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (unsafe_status, closure))
    before = _tree_fingerprint(fleet_home)

    result = apply_safe_recovery(root=fleet_home, now=4_000)

    assert result["mutations"] == 0
    assert _tree_fingerprint(fleet_home) == before


def test_safe_recovery_fresh_generation_cas_rejects_racing_write(
    fleet_home: Path, tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    task_ids = _create_recovery_fixture(fleet_home, tmp_path)
    wrote = False

    def race_after_snapshot(slug: str) -> None:
        nonlocal wrote
        if slug != "alpha" or wrote:
            return
        wrote = True
        with kb.connect_closing(board="alpha") as conn:
            kb.create_task(conn, title="racing authoritative write")

    monkeypatch.setattr(fleet, "_after_tasks_snapshot", race_after_snapshot)
    result = fleet.apply_safe_recovery(root=fleet_home, now=5_000)

    assert wrote is True
    assert result["verdict"] == "NOT_VERIFIED"
    assert result["reason"] == "fresh_authority_cas_rejected"
    assert result["mutations"] == 0
    assert any(board["outcome"] == "cas_mismatch" for board in result["boards"])
    assert "archived" not in _task_statuses("alpha", task_ids).values()


def test_safe_recovery_rechecks_profile_and_project_authorities_before_apply(
    fleet_home: Path, tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli import kanban_fleet as fleet

    task_ids = _create_recovery_fixture(fleet_home, tmp_path)
    original = fleet._known_project_ids
    calls = 0

    def changing_projects(root: Path, errors=None):
        nonlocal calls
        calls += 1
        projects, complete = original(root, errors)
        if calls > 1:
            projects.add("p_fixture")
        return projects, complete

    monkeypatch.setattr(fleet, "_known_project_ids", changing_projects)
    result = fleet.apply_safe_recovery(root=fleet_home, now=5_100)

    assert result["verdict"] == "NOT_VERIFIED"
    assert result["reason"] == "fresh_authority_cas_rejected"
    assert result["mutations"] == 0
    boards = result["boards"]
    assert isinstance(boards, list)
    assert any(board["outcome"] == "authority_cas_mismatch" for board in boards)
    assert "archived" not in _task_statuses("alpha", task_ids).values()
