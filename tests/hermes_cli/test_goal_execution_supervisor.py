from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def close_test_supervisors(monkeypatch):
    """Every test deterministically drains executors before interpreter exit."""
    from hermes_cli import goal_execution

    supervisor_type = goal_execution.GoalRecoverySupervisor
    supervisors = []

    def tracked_supervisor(*args, **kwargs):
        supervisor = supervisor_type(*args, **kwargs)
        supervisors.append(supervisor)
        return supervisor

    monkeypatch.setattr(goal_execution, "GoalRecoverySupervisor", tracked_supervisor)
    yield
    closed = set()
    for supervisor in reversed(supervisors):
        identity = id(supervisor)
        if identity in closed:
            continue
        closed.add(identity)
        close = getattr(supervisor, "close", None)
        if not callable(close) or getattr(supervisor, "_closed", False):
            continue
        close(timeout=1.0)


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    for db in goals._DB_CACHE.values():
        db.close()
    goals._DB_CACHE.clear()


class _FakeThread:
    created = []

    def __init__(self, *, target, name, daemon):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined = False
        self.__class__.created.append(self)

    def start(self):
        self.started = True

    def join(self, timeout=None):
        self.joined = True

    def is_alive(self):
        return self.started and not self.joined


def _remove_goal_execution_leases_for_legacy_fixture(hermes_home):
    """Model pre-B1b3 active state_meta rows, which had no lease rows."""
    with sqlite3.connect(hermes_home / "state.db") as conn:
        conn.execute("DELETE FROM goal_execution_leases")


def test_profile_discovery_is_home_anchored_and_state_db_bounded(tmp_path):
    from hermes_cli.goal_execution import discover_goal_profile_homes

    launch = tmp_path / "launch-home"
    launch.mkdir()
    root = tmp_path / ".hermes"
    valid = root / "profiles" / "work"
    empty = root / "profiles" / "empty"
    unrelated = tmp_path / "other-product" / "profiles" / "foreign"
    valid.mkdir(parents=True)
    empty.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    (valid / "state.db").touch()
    (unrelated / "state.db").touch()

    assert discover_goal_profile_homes(launch, profiles_root=root / "profiles") == [
        launch.resolve(),
        valid.resolve(),
    ]


def test_profile_discovery_defaults_to_canonical_root_and_rejects_escapes(
    tmp_path, monkeypatch
):
    import hermes_constants
    from hermes_cli.goal_execution import discover_goal_profile_homes

    root = tmp_path / "custom-root"
    launch = root / "profiles" / "launch"
    valid = root / "profiles" / "valid"
    outside = tmp_path / "outside"
    for home in (launch, valid, outside):
        home.mkdir(parents=True)
        (home / "state.db").touch()
    (root / "profiles" / "escaped-profile").symlink_to(outside, target_is_directory=True)
    escaped_db = root / "profiles" / "escaped-db"
    escaped_db.mkdir()
    (escaped_db / "state.db").symlink_to(outside / "state.db")
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)

    assert discover_goal_profile_homes(launch) == [launch.resolve(), valid.resolve()]


def test_profile_discovery_rejects_state_db_symlink_to_sibling_profile(tmp_path):
    from hermes_cli.goal_execution import discover_goal_profile_homes

    launch = tmp_path / "launch"
    profiles = tmp_path / "profiles"
    victim = profiles / "victim"
    linked = profiles / "linked"
    launch.mkdir()
    victim.mkdir(parents=True)
    linked.mkdir()
    (victim / "state.db").touch()
    (linked / "state.db").symlink_to(victim / "state.db")

    assert discover_goal_profile_homes(launch, profiles_root=profiles) == [
        launch.resolve(),
        victim.resolve(),
    ]


def test_profile_discovery_rejects_hardlinked_databases_and_unsafe_launch_home(tmp_path):
    from hermes_cli.goal_execution import discover_goal_profile_homes

    launch = tmp_path / "launch"
    profiles = tmp_path / "profiles"
    victim = profiles / "victim"
    linked = profiles / "linked"
    launch.mkdir()
    victim.mkdir(parents=True)
    linked.mkdir()
    (victim / "state.db").touch()
    os.link(victim / "state.db", linked / "state.db")

    # Both names are unsafe once the inode has multiple links.
    assert discover_goal_profile_homes(launch, profiles_root=profiles) == [launch.resolve()]

    os.link(victim / "state.db", launch / "state.db")
    with pytest.raises(RuntimeError, match="hardlink"):
        discover_goal_profile_homes(launch, profiles_root=profiles)


def test_launch_profile_directory_symlink_is_rejected(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    real = tmp_path / "real"
    linked = tmp_path / "linked"
    real.mkdir()
    (real / "state.db").touch()
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(RuntimeError, match="profile directory"):
        GoalRecoverySupervisor(linked)


def test_profile_db_is_revalidated_after_coordinator_open(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    profiles = tmp_path / "profiles"
    victim = profiles / "victim"
    raced = profiles / "raced"
    launch.mkdir()
    victim.mkdir(parents=True)
    raced.mkdir()
    (victim / "state.db").touch()
    raced_db = raced / "state.db"
    raced_db.touch()
    recovered = []

    class Coordinator:
        def __init__(self, home):
            self.home = Path(home)
            self.closed = False
            if self.home == raced.resolve():
                raced_db.unlink()
                raced_db.symlink_to(victim / "state.db")
        def bootstrap_reconcile(self):
            recovered.append(self.home)
        def recover_once(self, _dispatch):
            recovered.append(self.home)
            return 1
        def close(self):
            self.closed = True

    supervisor = GoalRecoverySupervisor(
        launch,
        profiles_root=profiles,
        coordinator_factory=Coordinator,
    )

    assert supervisor.scan_once() == 2
    assert raced.resolve() not in recovered
    assert raced.resolve() in supervisor.profile_errors


def test_open_coordinator_identity_is_pinned_across_later_scans(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    profile = tmp_path / "profiles" / "work"
    launch.mkdir(parents=True)
    profile.mkdir(parents=True)
    (profile / "state.db").touch()
    calls = []

    class Coordinator:
        def __init__(self, home):
            self.home = Path(home)
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            calls.append(self.home)
            return 0
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(
        launch,
        profiles_root=tmp_path / "profiles",
        coordinator_factory=Coordinator,
    )
    assert supervisor.scan_once() == 0
    original_calls = list(calls)

    old_profile = profile.with_name("work-old")
    profile.rename(old_profile)
    profile.mkdir()
    (profile / "state.db").touch()

    assert supervisor.scan_once() == 0
    assert calls == original_calls + [launch.resolve()]
    assert profile.resolve() in supervisor.profile_errors
    assert "identity" in str(supervisor.profile_errors[profile.resolve()])


def test_recovery_releases_old_claim_when_database_is_replaced_before_dispatch(
    hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    state = GoalManager("identity-race").set("must not run in replacement")
    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="old-db", clock=lambda: 100.0
    )
    original_claim = coordinator.claim

    def claim_then_replace(*args, **kwargs):
        record = original_claim(*args, **kwargs)
        old_path = hermes_home / "state-old.db"
        (hermes_home / "state.db").rename(old_path)
        (hermes_home / "state.db").touch()
        return record

    monkeypatch.setattr(coordinator, "claim", claim_then_replace)
    dispatched = []

    with pytest.raises(RuntimeError, match="identity changed"):
        coordinator.recover_once(lambda *_args: dispatched.append(True))

    assert dispatched == []
    old_lease = coordinator.get("identity-race")
    assert old_lease is not None
    assert old_lease.generation == state.generation
    assert old_lease.owner_id is None
    assert old_lease.claim_token is None
    assert "identity" in str(old_lease.last_error).lower()


def test_recovery_releases_old_claim_when_profile_directory_is_replaced(
    hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    GoalManager("directory-race").set("must remain on the old inode")
    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="old-directory", clock=lambda: 100.0
    )
    original_claim = coordinator.claim
    old_home = hermes_home.with_name("old-hermes")

    def claim_then_replace(*args, **kwargs):
        record = original_claim(*args, **kwargs)
        hermes_home.rename(old_home)
        hermes_home.mkdir()
        (hermes_home / "state.db").touch()
        return record

    monkeypatch.setattr(coordinator, "claim", claim_then_replace)

    with pytest.raises(RuntimeError, match="identity changed"):
        coordinator.recover_once(lambda *_args: pytest.fail("replacement dispatched"))

    old_lease = coordinator.get("directory-race")
    assert old_lease is not None
    assert old_lease.owner_id is None
    assert "identity" in str(old_lease.last_error).lower()


def test_supervisor_is_owned_idempotent_and_closes_every_profile(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    _FakeThread.created = []
    launch = tmp_path / "launch"
    profile = tmp_path / "profiles" / "work"
    launch.mkdir(parents=True)
    profile.mkdir(parents=True)
    (profile / "state.db").touch()
    calls = []

    class Coordinator:
        def __init__(self, home):
            self.home = Path(home)
            self.closed = False
        def bootstrap_reconcile(self):
            calls.append(("bootstrap", self.home))
        def recover_once(self, dispatch):
            calls.append(("recover", self.home))
            return 0
        def close(self):
            self.closed = True

    supervisor = GoalRecoverySupervisor(
        launch,
        profiles_root=tmp_path / "profiles",
        coordinator_factory=Coordinator,
        dispatch=lambda *_args: None,
        thread_factory=_FakeThread,
    )

    assert supervisor.start() is True
    assert supervisor.start() is False
    assert len(_FakeThread.created) == 1
    supervisor.scan_once()
    assert len(calls) == 4
    assert {
        home: [operation for operation, called_home in calls if called_home == home]
        for home in (launch.resolve(), profile.resolve())
    } == {
        launch.resolve(): ["bootstrap", "recover"],
        profile.resolve(): ["bootstrap", "recover"],
    }
    assert supervisor.stop(timeout=0.1) is True
    assert _FakeThread.created[0].joined
    assert all(c.closed for c in supervisor.closed_coordinators)
    assert supervisor.stop(timeout=0.1) is False


def test_supervisor_stop_timeout_retains_thread_and_coordinators_until_exit(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    launch.mkdir()
    scan_started = threading.Event()
    release_scan = threading.Event()
    scan_finished = threading.Event()

    class Coordinator:
        def __init__(self, _home):
            self.closed = 0

        def bootstrap_reconcile(self):
            pass

        def recover_once(self, _dispatch):
            scan_started.set()
            assert release_scan.wait(2.0)
            scan_finished.set()
            return 0

        def close(self):
            self.closed += 1

    supervisor = GoalRecoverySupervisor(
        launch,
        coordinator_factory=Coordinator,
        scan_seconds=0.01,
    )
    assert supervisor.start() is True
    assert scan_started.wait(1.0)
    old_thread = supervisor.thread
    coordinator = supervisor._coordinators[launch.resolve()]

    assert supervisor.stop(timeout=0.01) is False
    assert supervisor.thread is old_thread
    assert supervisor._profile_futures
    assert coordinator.closed == 0
    assert supervisor.start() is False

    release_scan.set()
    assert scan_finished.wait(1.0)
    next(iter(supervisor._profile_futures.values())).result(timeout=1.0)
    old_thread.join(timeout=1.0)
    assert not old_thread.is_alive()
    assert supervisor.start() is True
    assert supervisor.thread is not old_thread
    assert coordinator.closed == 1
    assert supervisor.stop(timeout=1.0) is True
    assert coordinator.closed == 1


def test_shutdown_racing_scan_rejects_late_submit_without_thread_exception(
    tmp_path, monkeypatch
):
    from concurrent.futures import Future
    from hermes_cli import goal_execution
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    launch.mkdir()
    discovery_entered = threading.Event()
    release_discovery = threading.Event()
    thread_finished = threading.Event()
    captured_thread_exceptions = []
    discovery_calls = 0

    class Executor:
        def __init__(self, **_kwargs):
            self.shutdown_called = False
            self.submissions = 0

        def submit(self, fn, *args):
            if self.shutdown_called:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self.submissions += 1
            future = Future()
            future.set_result(fn(*args))
            return future

        def shutdown(self, **_kwargs):
            self.shutdown_called = True

    executor = Executor()

    def discover(*_args, **_kwargs):
        nonlocal discovery_calls
        discovery_calls += 1
        if discovery_calls == 2:
            discovery_entered.set()
            assert release_discovery.wait(timeout=5)
        return [launch.resolve()]

    class Coordinator:
        def bootstrap_reconcile(self):
            pass
        def recover_once(self, _dispatch):
            return 0
        def close(self):
            pass

    monkeypatch.setattr(goal_execution, "discover_goal_profile_homes", discover)
    monkeypatch.setattr(
        threading,
        "excepthook",
        lambda args: captured_thread_exceptions.append(args.exc_value),
    )
    supervisor = GoalRecoverySupervisor(
        launch,
        coordinator_factory=lambda _home: Coordinator(),
        executor_factory=lambda **_kwargs: executor,
    )
    assert supervisor.scan_once() == 0

    scanner = threading.Thread(
        target=lambda: (supervisor.scan_once(), thread_finished.set())
    )
    scanner.start()
    assert discovery_entered.wait(timeout=5)
    assert supervisor.close(timeout=1.0) is True
    release_discovery.set()
    scanner.join(timeout=5)

    assert not scanner.is_alive()
    assert thread_finished.is_set()
    assert captured_thread_exceptions == []
    assert executor.submissions == 1
    assert supervisor.scan_once() == 0


def test_supervisor_isolates_profile_errors_and_retries_later(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    bad = tmp_path / "profiles" / "bad"
    good = tmp_path / "profiles" / "good"
    for home in (launch, bad, good):
        home.mkdir(parents=True)
    (bad / "state.db").touch()
    (good / "state.db").touch()
    recovered = []

    class Coordinator:
        def __init__(self, home):
            self.home = Path(home)
            if self.home == bad.resolve():
                raise RuntimeError("locked")
        def bootstrap_reconcile(self):
            pass
        def recover_once(self, dispatch):
            recovered.append(self.home)
            return 1
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(
        launch,
        profiles_root=tmp_path / "profiles",
        coordinator_factory=Coordinator,
        dispatch=lambda *_args: None,
        clock=lambda: 10.0,
        error_backoff_seconds=5.0,
    )

    assert supervisor.scan_once() == 2
    assert len(recovered) == 2
    assert set(recovered) == {launch.resolve(), good.resolve()}
    assert bad.resolve() in supervisor.profile_errors


def test_restart_scan_recovers_launch_and_secondary_profile_without_live_sessions(
    hermes_home, tmp_path
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.goal_execution import GoalExecutionCoordinator, GoalRecoverySupervisor
    from hermes_cli.goals import GoalManager

    profiles_root = tmp_path / ".hermes-root" / "profiles"
    secondary = profiles_root / "work"
    secondary.mkdir(parents=True)
    GoalManager("launch-goal").set("launch objective")
    token = set_hermes_home_override(secondary)
    try:
        GoalManager("profile-goal").set("profile objective")
    finally:
        reset_hermes_home_override(token)
    recovered = []

    def coordinator_factory(home):
        return GoalExecutionCoordinator(home, owner_id="fresh-backend", clock=lambda: 100.0)

    def recover(coordinator):
        return coordinator.recover_once(
            lambda record, state: recovered.append(
                (coordinator.home.resolve(), record.session_id, state.goal)
            )
        )

    supervisor = GoalRecoverySupervisor(
        hermes_home,
        profiles_root=profiles_root,
        coordinator_factory=coordinator_factory,
        recover_callback=recover,
        clock=lambda: 100.0,
    )
    assert supervisor.scan_once() == 2
    assert len(recovered) == 2
    assert set(recovered) == {
        (hermes_home.resolve(), "launch-goal", "launch objective"),
        (secondary.resolve(), "profile-goal", "profile objective"),
    }


def test_stop_event_wakes_scheduler_without_real_sleep(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    launch.mkdir()
    waits = []

    class StopAfterOneWait(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            self.set()
            return True

    supervisor = GoalRecoverySupervisor(
        launch,
        coordinator_factory=lambda _home: None,
        dispatch=lambda *_args: None,
        stop_event=StopAfterOneWait(),
        scan_seconds=7.0,
    )
    supervisor.run()
    assert waits == [7.0]


def test_process_registry_reuses_supervisor_across_real_module_reload(tmp_path):
    """Exercise a real reload without replacing this pytest process's classes."""
    child_home = tmp_path / "reload-hermes-home"
    child_home.mkdir()
    script = textwrap.dedent(
        """
        import importlib
        import threading

        from hermes_cli import goal_execution

        key = "test-reload-singleton"
        created = []
        first = goal_execution.get_goal_recovery_supervisor(
            key, lambda: created.append(object()) or created[-1]
        )
        reloaded = importlib.reload(goal_execution)
        second = reloaded.get_goal_recovery_supervisor(key, lambda: object())
        assert first is second
        assert len(created) == 1
        assert reloaded.clear_goal_recovery_supervisor(key, first) is True
        assert reloaded.clear_goal_recovery_supervisor(key, first) is False
        assert threading.enumerate() == [threading.main_thread()]
        """
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(child_home)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_bootstrap_decodes_only_active_rows_and_steady_state_uses_due_index(
    hermes_home, monkeypatch
):
    from hermes_cli import goals
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    for index in range(200):
        manager = GoalManager(f"historical-{index}")
        manager.set("old")
        manager.pause("archived")
    active = GoalManager("active-session").set("resume this")
    _remove_goal_execution_leases_for_legacy_fixture(hermes_home)
    decoded = 0
    original = goals.GoalState.from_json

    def counted(raw):
        nonlocal decoded
        decoded += 1
        return original(raw)

    monkeypatch.setattr(goals.GoalState, "from_json", staticmethod(counted))
    coordinator = GoalExecutionCoordinator(hermes_home, owner_id="recovery", clock=lambda: 100.0)

    assert coordinator.bootstrap_reconcile(limit=50) == 1
    assert decoded == 1
    assert coordinator.get("active-session").generation == active.generation
    monkeypatch.setattr(
        coordinator.db,
        "active_goal_state_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("historical scan")),
    )
    dispatched = []
    assert coordinator.recover_once(lambda record, state: dispatched.append((record, state))) == 1
    assert [record.session_id for record, _state in dispatched] == ["active-session"]


def test_post_bootstrap_active_goal_is_immediately_visible_without_restart(
    hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator, GoalRecoverySupervisor
    from hermes_cli.goals import GoalManager

    recovered = []
    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="live-supervisor", clock=lambda: 10**12
    )
    supervisor = GoalRecoverySupervisor(
        hermes_home,
        coordinator_factory=lambda _home: coordinator,
        dispatch=lambda record, state: recovered.append((record.session_id, state.goal)),
        clock=lambda: 10**12,
    )

    assert supervisor.scan_once() == 0
    assert hermes_home.resolve() in supervisor._bootstrapped

    def unexpected_reconciliation(*_args, **_kwargs):
        pytest.fail("post-bootstrap insert must not require reconciliation")

    monkeypatch.setattr(coordinator, "bootstrap_reconcile", unexpected_reconciliation)
    monkeypatch.setattr(coordinator, "bootstrap_reconcile_page", unexpected_reconciliation)

    state = GoalManager("inserted-after-bootstrap").set("recover now")
    lease = coordinator.get("inserted-after-bootstrap")
    assert lease is not None
    assert lease.generation == state.generation
    assert lease.owner_id is None
    assert lease.next_run_at > 0
    assert supervisor.scan_once() == 1
    assert recovered == [("inserted-after-bootstrap", "recover now")]
    assert supervisor.close(timeout=1.0) is True


def test_steady_state_due_scan_is_bounded(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    for index in range(25):
        GoalManager(f"due-{index:02d}").set("work")
    coordinator = GoalExecutionCoordinator(
        hermes_home,
        owner_id="recovery",
        clock=lambda: 100.0,
        recovery_batch_limit=7,
    )
    coordinator.bootstrap_reconcile(limit=100)
    dispatched = []

    assert coordinator.recover_once(
        lambda record, _state: dispatched.append(record.session_id)
    ) == 7
    assert len(dispatched) == 7


def test_capacity_rejection_reschedules_exact_claim_without_burning_attempt(
    hermes_home,
):
    from hermes_cli.goal_execution import (
        GoalExecutionCoordinator,
        RecoveryTurnAdmission,
    )
    from hermes_cli.goals import GoalManager

    state = GoalManager("capacity-deferred").set("work later")
    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="recovery", clock=lambda: 100.0
    )

    result = coordinator.recover_once(
        lambda _record, _state: RecoveryTurnAdmission(
            accepted=False,
            retry_after_seconds=0.5,
            reason="recovery turn capacity exhausted",
        )
    )

    assert result.claimed == 1
    assert result.accepted == 0
    assert result.failed == 0
    lease = coordinator.get("capacity-deferred")
    assert lease is not None
    assert lease.generation == state.generation
    assert lease.owner_id is None
    assert lease.claim_token is None
    assert lease.next_run_at == pytest.approx(100.5)
    assert lease.attempt == 0
    assert lease.last_error == "recovery turn capacity exhausted"


def test_timed_wait_wakes_through_coordinator_after_deadline(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager, save_goal

    now = [10.0]
    state = GoalManager("timed-wait").set("wake later")
    state.waiting_until = 25.0
    state.waiting_reason = "cooldown"
    save_goal("timed-wait", state)
    coordinator = GoalExecutionCoordinator(hermes_home, owner_id="recovery", clock=lambda: now[0])
    coordinator.bootstrap_reconcile()
    dispatched = []

    assert coordinator.recover_once(lambda record, goal: dispatched.append(goal.goal)) == 0
    now[0] = 25.0
    assert coordinator.recover_once(lambda record, goal: dispatched.append(goal.goal)) == 1
    assert dispatched == ["wake later"]


def test_missing_pid_wait_is_ready_after_restart(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager, save_goal

    state = GoalManager("pid-wait").set("wake after process")
    state.waiting_on_pid = 999_999_999
    state.waiting_reason = "build"
    save_goal("pid-wait", state)
    coordinator = GoalExecutionCoordinator(hermes_home, owner_id="recovery", clock=lambda: 50.0)
    coordinator.bootstrap_reconcile()
    dispatched = []

    assert coordinator.recover_once(lambda record, goal: dispatched.append(goal.goal)) == 1
    assert dispatched == ["wake after process"]


def test_stale_due_generation_mismatch_cannot_revoke_newer_live_claim(
    hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    manager = GoalManager("generation-race")
    first = manager.set("old goal")
    stale = GoalExecutionCoordinator(hermes_home, owner_id="stale", clock=lambda: 10.0)
    stale.bootstrap_reconcile()
    old_due = stale.get("generation-race")
    assert old_due is not None and old_due.generation == first.generation

    newer_state = manager.set("new goal")
    assert stale.invalidate("generation-race", newer_state.generation)
    newer = GoalExecutionCoordinator(hermes_home, owner_id="newer", clock=lambda: 10.0)
    newer_claim = newer.claim("generation-race", newer_state.generation)
    assert newer_claim is not None
    monkeypatch.setattr(stale, "due_records", lambda **_kwargs: [old_due])

    assert stale.recover_once(lambda *_args: pytest.fail("stale row dispatched")) == 0
    surviving = newer.get("generation-race")
    assert surviving is not None
    assert surviving.generation == newer_state.generation
    assert surviving.owner_id == "newer"
    assert surviving.claim_token == newer_claim.claim_token


def test_synchronous_recovery_failures_pause_after_finite_attempts(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    now = [100.0]
    state = GoalManager("broken-recovery").set("cannot reconstruct")
    coordinator = GoalExecutionCoordinator(
        hermes_home,
        owner_id="recovery",
        clock=lambda: now[0],
        max_recovery_attempts=3,
    )
    coordinator.bootstrap_reconcile()

    def fail(_record, _state):
        raise RuntimeError("bad persisted session")

    for expected_attempt in (1, 2):
        assert coordinator.recover_once(fail) == 0
        record = coordinator.get("broken-recovery")
        assert record.attempt == expected_attempt
        now[0] = record.next_run_at
    assert coordinator.recover_once(fail) == 0

    paused = GoalManager("broken-recovery").state
    assert paused.status == "paused"
    assert "recoverable blocker after 3 attempts" in paused.paused_reason
    record = coordinator.get("broken-recovery")
    assert record.owner_id is None
    assert record.next_run_at == 0


def test_recovery_result_distinguishes_acceptance_completion_and_failure(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator, RecoveryTurnAdmission
    from hermes_cli.goals import GoalManager

    GoalManager("accepted-only").set("queued")
    GoalManager("completed-now").set("done")
    GoalManager("failed-now").set("retry")
    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="accounting", clock=lambda: 100.0
    )

    def dispatch(record, _state):
        if record.session_id == "accepted-only":
            return RecoveryTurnAdmission(accepted=True)
        if record.session_id == "failed-now":
            raise RuntimeError("scheduled failure")
        return None

    result = coordinator.recover_once(dispatch)

    assert result.accepted == 2
    assert result.completed == 1
    assert result.failed == 1
    assert int(result) == 1


def test_failed_recovery_pause_is_fenced_to_original_generation_and_token(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    manager = GoalManager("pause-race")
    first = manager.set("old work")
    stale = GoalExecutionCoordinator(
        hermes_home,
        owner_id="stale",
        clock=lambda: 100.0,
        max_recovery_attempts=1,
    )
    stale.bootstrap_reconcile()
    newer = GoalExecutionCoordinator(hermes_home, owner_id="newer", clock=lambda: 100.0)
    replacement = {}

    def replace_then_fail(_record, _state):
        replacement["state"] = manager.set("replacement work")
        assert stale.invalidate("pause-race", replacement["state"].generation)
        replacement["claim"] = newer.claim(
            "pause-race", replacement["state"].generation
        )
        assert replacement["claim"] is not None
        raise RuntimeError("old dispatch failed")

    assert stale.recover_once(replace_then_fail) == 0
    current = GoalManager("pause-race").state
    assert current is not None
    assert current.status == "active"
    assert current.generation == first.generation + 1
    assert current.goal == "replacement work"
    lease = newer.get("pause-race")
    assert lease is not None
    assert lease.owner_id == "newer"
    assert lease.claim_token == replacement["claim"].claim_token


def test_bootstrap_pages_are_bounded_complete_and_restart_safe(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    for index in range(1001):
        GoalManager(f"active-{index:04d}").set("work")
    _remove_goal_execution_leases_for_legacy_fixture(hermes_home)

    first = GoalExecutionCoordinator(hermes_home, owner_id="first", clock=lambda: 100.0)
    seeded, complete = first.bootstrap_reconcile_page(limit=1000)
    assert (seeded, complete) == (1000, False)
    first.close()

    resumed = GoalExecutionCoordinator(hermes_home, owner_id="resumed", clock=lambda: 100.0)
    seeded, complete = resumed.bootstrap_reconcile_page(limit=1000)
    assert (seeded, complete) == (1, True)
    assert resumed.get("active-1000") is not None


def test_due_scan_uses_two_index_bounded_queries_without_temp_sort(hermes_home):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    coordinator = GoalExecutionCoordinator(
        hermes_home, owner_id="recovery", clock=lambda: 100.0, recovery_batch_limit=3
    )
    for name in ("scheduled-c", "scheduled-a", "scheduled-b"):
        GoalManager(name).set("work")
    coordinator.bootstrap_reconcile()
    claimed = coordinator.claim("expired-a", 1, lease_seconds=1)
    assert claimed is not None

    plans = coordinator.db.explain_due_goal_execution_leases(now=102.0, limit=3)
    details = [str(row[3]) for plan in plans for row in plan]
    assert any("idx_goal_execution_scheduled" in detail for detail in details)
    assert any("idx_goal_execution_expired" in detail for detail in details)
    assert not any("TEMP B-TREE" in detail for detail in details)
    assert [
        row["session_id"]
        for row in coordinator.db.due_goal_execution_leases(now=102.0, limit=3)
    ] == ["scheduled-a", "scheduled-b", "scheduled-c"]


def test_scheduler_dispatches_profiles_in_parallel_single_flight(tmp_path):
    from hermes_cli.goal_execution import GoalRecoverySupervisor

    launch = tmp_path / "launch"
    other = tmp_path / "profiles" / "other"
    launch.mkdir(parents=True)
    other.mkdir(parents=True)
    (other / "state.db").touch()
    blocked = threading.Event()
    release = threading.Event()
    progressed = threading.Event()
    calls = []

    class Coordinator:
        def __init__(self, home):
            self.home = Path(home)
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            calls.append(self.home)
            if self.home == launch.resolve():
                blocked.set()
                assert release.wait(5)
            else:
                progressed.set()
            return 1
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(
        launch,
        profiles_root=tmp_path / "profiles",
        coordinator_factory=Coordinator,
        recovery_workers=2,
    )
    assert supervisor.scan_once(wait=False) == 0
    assert blocked.wait(1)
    assert progressed.wait(1)
    assert supervisor.scan_once(wait=False) == 1
    assert calls.count(launch.resolve()) == 1
    release.set()
    supervisor.close(timeout=2)


def test_async_recovery_success_is_accounted_only_on_terminal_completion(tmp_path):
    from concurrent.futures import Future
    from hermes_cli.goal_execution import GoalRecoverySupervisor, RecoveryScanResult

    launch = tmp_path / "launch"
    launch.mkdir()
    completion = Future()

    class Coordinator:
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            return RecoveryScanResult(accepted=1, completions=(completion,))
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(launch, coordinator_factory=lambda _home: Coordinator())
    supervisor.profile_errors[launch.resolve()] = RuntimeError("prior")

    assert supervisor.scan_once() == 0
    assert (supervisor.accepted, supervisor.completed, supervisor.failed) == (1, 0, 0)
    assert launch.resolve() in supervisor.profile_errors

    completion.set_result(None)
    assert (supervisor.accepted, supervisor.completed, supervisor.failed) == (1, 1, 0)
    assert launch.resolve() not in supervisor.profile_errors


def test_async_recovery_failure_is_accounted_after_durable_retry(tmp_path):
    from concurrent.futures import Future
    from hermes_cli.goal_execution import GoalRecoverySupervisor, RecoveryScanResult

    launch = tmp_path / "launch"
    launch.mkdir()
    completion = Future()

    class Coordinator:
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            return RecoveryScanResult(accepted=1, completions=(completion,))
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(launch, coordinator_factory=lambda _home: Coordinator())
    assert supervisor.scan_once() == 0
    completion.set_exception(RuntimeError("retry was scheduled"))

    assert (supervisor.accepted, supervisor.completed, supervisor.failed) == (1, 0, 1)
    assert "retry was scheduled" in str(supervisor.profile_errors[launch.resolve()])


def test_old_async_success_cannot_clear_newer_profile_failure(tmp_path):
    from concurrent.futures import Future
    from hermes_cli.goal_execution import GoalRecoverySupervisor, RecoveryScanResult

    launch = tmp_path / "launch"
    launch.mkdir()
    old = Future()
    newer = Future()
    completions = iter((old, newer))

    class Coordinator:
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            return RecoveryScanResult(accepted=1, completions=(next(completions),))
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(launch, coordinator_factory=lambda _home: Coordinator())
    assert supervisor.scan_once() == 0
    assert supervisor.scan_once() == 0

    newer.set_exception(RuntimeError("newer failure"))
    old.set_result(None)

    assert (supervisor.accepted, supervisor.completed, supervisor.failed) == (2, 1, 1)
    assert "newer failure" in str(supervisor.profile_errors[launch.resolve()])


def test_old_async_failure_cannot_restore_error_after_newer_profile_success(tmp_path):
    from concurrent.futures import Future
    from hermes_cli.goal_execution import GoalRecoverySupervisor, RecoveryScanResult

    launch = tmp_path / "launch"
    launch.mkdir()
    old = Future()
    newer = Future()
    completions = iter((old, newer))

    class Coordinator:
        def bootstrap_reconcile_page(self):
            return 0, True
        def recover_once(self, _dispatch):
            return RecoveryScanResult(accepted=1, completions=(next(completions),))
        def close(self):
            pass

    supervisor = GoalRecoverySupervisor(
        launch,
        coordinator_factory=lambda _home: Coordinator(),
        clock=lambda: 100.0,
        error_backoff_seconds=5.0,
    )
    assert supervisor.scan_once() == 0
    assert supervisor.scan_once() == 0

    newer.set_result(None)
    old.set_exception(RuntimeError("older failure"))

    assert (supervisor.accepted, supervisor.completed, supervisor.failed) == (2, 1, 1)
    assert launch.resolve() not in supervisor.profile_errors
    assert launch.resolve() not in supervisor._retry_after
