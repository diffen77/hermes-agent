from __future__ import annotations

import json
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "factory_recovery_controller.py"
SOURCE_SHA = "a" * 40


@pytest.fixture(autouse=True)
def clean_executing_source_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Run controller tests against an exact, clean checkout of the script."""
    global SCRIPT, SOURCE_SHA
    source_repo = tmp_path / "controller-source"
    copied_script = source_repo / "scripts" / "factory_recovery_controller.py"
    copied_script.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "scripts" / "factory_recovery_controller.py", copied_script)
    subprocess.run(["git", "init", "-q"], cwd=source_repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"], cwd=source_repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Controller Tests"], cwd=source_repo, check=True
    )
    subprocess.run(
        ["git", "add", "scripts/factory_recovery_controller.py"], cwd=source_repo, check=True
    )
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source_repo, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    SCRIPT = copied_script
    SOURCE_SHA = sha
    from scripts import factory_recovery_controller as controller

    monkeypatch.setattr(controller, "_REPO_ROOT", source_repo)
    return source_repo


@pytest.fixture
def fleet_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.create_board("alpha", name="Alpha")
    kb.create_board("beta", name="Beta")
    return home


def _fixture(home: Path, tmp_path: Path, *, board: str = "alpha") -> list[str]:
    with sqlite3.connect(home / "projects.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY)")
        conn.execute("INSERT OR IGNORE INTO projects (id) VALUES ('p_real')")
    root = tmp_path / f"fixture-{board}"
    parent = None
    ids: list[str] = []
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
            ids.append(task_id)
            parent = task_id
    return ids


def _statuses(board: str, ids: list[str]) -> list[str]:
    with kb.connect_closing(board=board) as conn:
        return [
            str(row[0])
            for row in conn.execute(
                f"SELECT status FROM tasks WHERE id IN ({','.join('?' for _ in ids)})",
                ids,
            )
        ]


def _run_cli(home: Path, *extra: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(home),
        "--expected-source-sha",
        SOURCE_SHA,
        *extra,
    ]
    process_env = dict(os.environ) if env is None else dict(env)
    process_env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=process_env,
        text=True,
        capture_output=True,
        timeout=10,
    )


def test_delegated_child_is_denied_before_any_filesystem_write(tmp_path: Path) -> None:
    root = tmp_path / "must-not-exist"
    env = dict(os.environ, HERMES_DELEGATED_CHILD_CONTEXT="1")

    completed = _run_cli(root, env=env)

    assert completed.returncode != 0
    assert json.loads(completed.stdout) == {
        "mutations": 0,
        "outcome": "DENIED",
        "reason": "delegated_child_context",
    }
    assert completed.stderr == ""
    assert not root.exists()


def test_nonblocking_lease_excludes_overlap_and_kill_releases_it(
    fleet_home: Path, tmp_path: Path, clean_executing_source_repo: Path
) -> None:
    ids = _fixture(fleet_home, tmp_path)
    entered = tmp_path / "entered"
    holder_code = f"""
import time
from pathlib import Path
from scripts.factory_recovery_controller import run_tick

def hold(**kwargs):
    Path({str(entered)!r}).write_text('ready')
    while True:
        time.sleep(1)

run_tick(root=Path({str(fleet_home)!r}), expected_source_sha={SOURCE_SHA!r}, apply_fn=hold, source_repo_root=Path({str(clean_executing_source_repo)!r}))
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_code],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert entered.exists(), holder.communicate(timeout=1)
        controller_dir = fleet_home / "kanban" / "factory-recovery-controller"
        before_busy = {
            path.name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
            for path in controller_dir.iterdir()
            if path.is_file()
        }

        busy = _run_cli(fleet_home)
        assert busy.returncode == 0
        assert json.loads(busy.stdout) == {"mutations": 0, "outcome": "BUSY"}
        after_busy = {
            path.name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
            for path in controller_dir.iterdir()
            if path.is_file()
        }
        assert after_busy == before_busy
        assert "archived" not in _statuses("alpha", ids)

        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=5)
        restarted = _run_cli(fleet_home)
        assert restarted.returncode == 0, restarted.stderr
        assert json.loads(restarted.stdout)["outcome"] == "APPLIED"
        assert _statuses("alpha", ids).count("archived") == 1
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


def test_crash_after_db_commit_next_tick_recovers_receipt_without_remutation(
    fleet_home: Path, tmp_path: Path, clean_executing_source_repo: Path
) -> None:
    alpha = _fixture(fleet_home, tmp_path, board="alpha")
    beta = _fixture(fleet_home, tmp_path, board="beta")
    crash_code = f"""
import os
from pathlib import Path
from hermes_cli import kanban_fleet
from scripts.factory_recovery_controller import run_tick
kanban_fleet._after_recovery_commit = lambda slug, task_id: os._exit(91)
run_tick(root=Path({str(fleet_home)!r}), expected_source_sha={SOURCE_SHA!r}, apply_fn=kanban_fleet.apply_safe_recovery, source_repo_root=Path({str(clean_executing_source_repo)!r}))
"""

    crashed = subprocess.run(
        [sys.executable, "-c", crash_code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert crashed.returncode == 91
    assert _statuses("alpha", alpha).count("archived") == 1
    receipt_dir = fleet_home / "kanban" / "reconciler-receipts"
    assert not receipt_dir.exists() or not list(receipt_dir.glob("*.json"))

    recovered = _run_cli(fleet_home)

    assert recovered.returncode == 0, recovered.stderr
    payload = json.loads(recovered.stdout)
    assert payload["outcome"] == "APPLIED"
    assert payload["mutations"] == 0
    assert payload["receipt_recovered"] is True
    assert _statuses("alpha", alpha).count("archived") == 1
    assert "archived" not in _statuses("beta", beta)
    assert len(list(receipt_dir.glob("*.json"))) == 1


def test_heartbeat_is_bounded_atomic_private_and_records_source_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    calls = 0

    def no_op(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "verdict": "APPLIED",
            "mutations": 0,
            "observation_generation": "generation-1",
            "boards": [],
        }

    result = controller.run_tick(
        root=tmp_path, expected_source_sha=SOURCE_SHA.upper(), now=100, apply_fn=no_op
    )

    assert result["outcome"] == "NO_OP"
    assert calls == 1
    state_path = tmp_path / "kanban" / "factory-recovery-controller" / "heartbeat.json"
    state = json.loads(state_path.read_text())
    assert state == {
        "candidate_source_sha": SOURCE_SHA,
        "completed_at": 100,
        "consecutive_failures": 0,
        "failure_fingerprint": None,
        "mutation_count": 0,
        "next_eligible_at": 0,
        "observation_generation": "generation-1",
        "outcome": "NO_OP",
        "started_at": 100,
        "version": 1,
    }
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert state_path.stat().st_size < 4096
    assert not list(state_path.parent.glob("*.tmp"))


def test_repeated_equivalent_error_gets_one_immediate_retry_before_bounded_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    calls = 0

    def fail(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("secret detail must not persist")

    first = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=100,
        backoff_base_seconds=10,
        backoff_max_seconds=15,
        apply_fn=fail,
    )
    immediate_retry = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=101,
        backoff_base_seconds=10,
        backoff_max_seconds=15,
        apply_fn=fail,
    )
    backed_off = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=102,
        backoff_base_seconds=10,
        backoff_max_seconds=15,
        apply_fn=fail,
    )

    assert first == {"outcome": "ERROR", "reason": "safe_recovery_failed", "mutations": 0}
    assert immediate_retry["outcome"] == "ERROR"
    assert backed_off == {"outcome": "BACKOFF", "mutations": 0, "retry_at": 111}
    assert calls == 2
    state_path = tmp_path / "kanban" / "factory-recovery-controller" / "heartbeat.json"
    persisted = state_path.read_text()
    assert "secret detail" not in persisted
    state = json.loads(persisted)
    assert state["consecutive_failures"] == 2
    assert state["next_eligible_at"] == 111

    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    before = state_path.read_bytes()
    denied = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=200,
        force=True,
        apply_fn=fail,
    )
    assert denied["outcome"] == "DENIED"
    assert state_path.read_bytes() == before
    assert calls == 2


def test_changed_source_generation_bypasses_existing_error_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_executing_source_repo: Path,
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    calls = 0

    def fail(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("internal")

    for now in (100, 101):
        assert controller.run_tick(
            root=tmp_path,
            expected_source_sha=SOURCE_SHA,
            now=now,
            backoff_base_seconds=60,
            backoff_max_seconds=60,
            apply_fn=fail,
        )["outcome"] == "ERROR"

    tracked = clean_executing_source_repo / "scripts" / "factory_recovery_controller.py"
    tracked.write_text(tracked.read_text() + "\n# next exact source\n")
    subprocess.run(["git", "add", "scripts/factory_recovery_controller.py"], cwd=clean_executing_source_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "next source"], cwd=clean_executing_source_repo, check=True)
    next_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clean_executing_source_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    retried = controller.run_tick(
        root=tmp_path,
        expected_source_sha=next_sha,
        now=102,
        backoff_base_seconds=60,
        backoff_max_seconds=60,
        apply_fn=fail,
    )

    assert retried["outcome"] == "ERROR"
    assert calls == 3


def test_not_verified_never_suppresses_immediately_actionable_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    calls = 0

    def changing_observation(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "verdict": "NOT_VERIFIED",
                "mutations": 0,
                "observation_generation": "incomplete-1",
                "boards": [],
            }
        return {
            "verdict": "APPLIED",
            "mutations": 1,
            "observation_generation": "actionable-2",
            "boards": [{"outcome": "archived_fixture"}],
        }

    first = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=100,
        backoff_base_seconds=3600,
        backoff_max_seconds=3600,
        apply_fn=changing_observation,
    )
    recovered = controller.run_tick(
        root=tmp_path,
        expected_source_sha=SOURCE_SHA,
        now=101,
        backoff_base_seconds=3600,
        backoff_max_seconds=3600,
        apply_fn=changing_observation,
    )

    assert first["outcome"] == "NOT_VERIFIED"
    assert recovered["outcome"] == "APPLIED"
    assert recovered["mutations"] == 1
    assert calls == 2


def test_malformed_state_fails_closed_without_calling_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    directory = tmp_path / "kanban" / "factory-recovery-controller"
    directory.mkdir(parents=True)
    (directory / "heartbeat.json").write_text('{"candidate_source_sha":"unterminated')
    calls = 0

    def forbidden(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"verdict": "APPLIED", "mutations": 1}

    result = controller.run_tick(
        root=tmp_path, expected_source_sha=SOURCE_SHA, now=300, apply_fn=forbidden
    )

    assert result == {"outcome": "ERROR", "reason": "malformed_state", "mutations": 0}
    assert calls == 0
    replacement = json.loads((directory / "heartbeat.json").read_text())
    assert replacement["outcome"] == "ERROR"
    assert replacement["mutation_count"] == 0


def test_source_sha_uses_argv_not_shell_and_healthy_noop_is_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from scripts import factory_recovery_controller as controller

    git_sha = controller._source_sha(None)
    assert len(git_sha) == 40
    assert all(character in "0123456789abcdef" for character in git_sha)
    git_root = tmp_path / "git-source"
    git_result = controller.run_tick(
        root=git_root,
        now=400,
        apply_fn=lambda **kwargs: {
            "verdict": "APPLIED",
            "mutations": 0,
            "boards": [],
        },
    )
    assert git_result["outcome"] == "NO_OP"
    git_state = json.loads(
        (git_root / "kanban" / "factory-recovery-controller" / "heartbeat.json").read_text()
    )
    assert git_state["candidate_source_sha"] == git_sha

    hostile_root = tmp_path / "hostile"
    rejected = _run_cli(hostile_root, "--expected-source-sha", "abc;touch-pwned")
    assert rejected.returncode != 0
    assert json.loads(rejected.stdout)["reason"] == "invalid_expected_source_sha"
    assert not hostile_root.exists()
    assert not (REPO_ROOT / "touch-pwned").exists()

    monkeypatch.setattr(
        controller,
        "run_tick",
        lambda **kwargs: {"outcome": "NO_OP", "mutations": 0},
    )
    assert controller.main(["--root", str(tmp_path), "--expected-source-sha", SOURCE_SHA]) == 0
    assert capsys.readouterr().out == ""


def test_source_sha_rejects_abbreviated_expected_sha() -> None:
    from scripts import factory_recovery_controller as controller

    with pytest.raises(ValueError, match="invalid_expected_source_sha"):
        controller._source_sha(SOURCE_SHA[:7])


def test_source_sha_rejects_mismatch_with_executing_checkout() -> None:
    from scripts import factory_recovery_controller as controller

    mismatch = ("0" if SOURCE_SHA[0] != "0" else "1") + SOURCE_SHA[1:]
    with pytest.raises(ValueError, match="expected_source_sha_mismatch"):
        controller._source_sha(mismatch)


def test_source_sha_rejects_dirty_executing_checkout(
    clean_executing_source_repo: Path,
) -> None:
    from scripts import factory_recovery_controller as controller

    tracked = clean_executing_source_repo / "scripts" / "factory_recovery_controller.py"
    tracked.write_text(tracked.read_text() + "\n# dirty\n")
    with pytest.raises(ValueError, match="source_tree_dirty"):
        controller._source_sha(SOURCE_SHA)


def test_source_sha_rejects_untracked_executing_checkout(
    clean_executing_source_repo: Path,
) -> None:
    from scripts import factory_recovery_controller as controller

    (clean_executing_source_repo / "controller-relevant.py").write_text("dirty = True\n")
    with pytest.raises(ValueError, match="source_tree_dirty"):
        controller._source_sha(SOURCE_SHA)


def test_source_sha_accepts_exact_clean_checkout_and_normalizes_case() -> None:
    from scripts import factory_recovery_controller as controller

    assert controller._source_sha(SOURCE_SHA.upper()) == SOURCE_SHA
