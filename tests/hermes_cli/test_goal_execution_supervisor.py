from __future__ import annotations

import importlib
import sqlite3
import threading
from pathlib import Path

import pytest


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


def test_process_registry_reuses_supervisor_across_real_module_reload():
    from hermes_cli import goal_execution

    key = "test-reload-singleton"
    created = []
    first = goal_execution.get_goal_recovery_supervisor(
        key, lambda: created.append(object()) or created[-1]
    )
    reloaded = importlib.reload(goal_execution)
    try:
        second = reloaded.get_goal_recovery_supervisor(key, lambda: object())
        assert first is second
        assert len(created) == 1
    finally:
        assert reloaded.clear_goal_recovery_supervisor(key, first) is True
        assert reloaded.clear_goal_recovery_supervisor(key, first) is False


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
