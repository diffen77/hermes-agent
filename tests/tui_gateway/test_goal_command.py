"""Tests for /goal handling in tui_gateway.

The TUI routes ``/goal`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_goal_command`` queues the kickoff message onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/goal`` directly in the server and return a
``{"type": "send", "notice": ..., "message": ...}`` payload the TUI client
uses to render a system line and fire the kickoff prompt.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module DB cache so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        # Reset module-level session state without re-importing. importlib.reload
        # would re-register the module's atexit hooks (ThreadPoolExecutor
        # shutdown, _shutdown_sessions); the duplicates race the stderr
        # buffer at interpreter shutdown and surface as Fatal Python error:
        # _enter_buffered_busy. Clearing the per-session dicts gives the
        # next test a clean slate; _methods is NOT cleared because it's
        # populated at module import time and re-registration only happens
        # via reload (which we don't do).
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-goal-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


def _accept_directive(
    server, session, directive, monkeypatch, *, forward_activation=True, on_run=None
):
    """Submit one directive and wait until its kickoff worker has exited."""
    sid, _, s = session
    delivered = threading.Event()
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)

    def run_prompt(_rid, _sid, _session, _text):
        try:
            if on_run is not None:
                on_run()
            delivered.set()
        finally:
            # Mirror _run_prompt_submit's ownership release. The command tests
            # need an accepted-and-finished turn, not a permanently busy seam.
            with _session["history_lock"]:
                _session["running"] = False
                server._clear_inflight_turn(_session)

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    params = {"session_id": sid, "text": directive["message"]}
    if forward_activation:
        params["goal_activation"] = directive.get("goal_activation")
    response = _call(server, "prompt.submit", **params)
    worker = s.get("_run_thread")
    assert worker is not None
    assert delivered.wait(timeout=5)
    worker.join(timeout=5)
    assert not worker.is_alive()
    return response


# ── command.dispatch /goal ────────────────────────────────────────────


def test_goal_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_whitespace_only_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="   ", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_status_alias_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="status", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_set_returns_send_with_notice_without_persisting(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="build a rocket", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "build a rocket"
    assert "notice" in result
    assert "Goal set" in result["notice"]
    assert "20-turn budget" in result["notice"]

    # Drafting a send directive is not acceptance. Persistence is deferred to
    # the authoritative prompt.submit claim.
    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state is None
    assert result["goal_activation"]["goal"] == "build a rocket"


def test_goal_draft_returns_transport_safe_activation_without_persisting(server, session):
    from hermes_cli.goals import GoalContract, GoalManager

    sid, session_key, _ = session
    contract = GoalContract(
        outcome="a verified staging slice is shipped",
        verification="focused staging tests pass",
        constraints="preserve production behavior",
        boundaries="only touch the staging path",
        stop_when="deployment credentials are required",
    )
    with patch("hermes_cli.goals.draft_contract", return_value=contract) as draft:
        r = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft Ship a verified staging slice",
            session_id=sid,
        )

    draft.assert_called_once_with("Ship a verified staging slice")
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "Ship a verified staging slice"
    assert "Completion contract:" in result["notice"]
    assert "- Verification: focused staging tests pass" in result["notice"]

    assert GoalManager(session_key).state is None
    assert result["goal_activation"]["goal"] == "Ship a verified staging slice"
    assert result["goal_activation"]["contract"] == contract.to_dict()
    assert result["goal_activation"]["session_key"] == session_key
    assert result["goal_activation"]["token"]


def test_goal_activation_is_persisted_atomically_with_accepted_kickoff(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalContract, GoalManager

    _, session_key, _ = session
    contract = GoalContract(verification="focused staging tests pass")
    with patch("hermes_cli.goals.draft_contract", return_value=contract):
        directive = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft Ship a verified staging slice",
            session_id=session[0],
        )["result"]

    assert GoalManager(session_key).state is None
    response = _accept_directive(server, session, directive, monkeypatch)

    assert response["result"]["status"] == "streaming"
    state = GoalManager(session_key).state
    assert state is not None
    assert state.goal == directive["message"]
    assert state.contract == contract


def test_goal_activation_is_rolled_back_when_kickoff_is_rejected(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    directive = _call(
        server,
        "command.dispatch",
        name="goal",
        arg="ship the verified slice",
        session_id=sid,
    )["result"]
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(
        server,
        "_wait_agent",
        lambda _session, _rid: {"error": {"message": "agent unavailable"}},
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: True)

    response = _call(
        server,
        "prompt.submit",
        session_id=sid,
        text=directive["message"],
        goal_activation=directive["goal_activation"],
    )
    worker = s["_run_thread"]
    worker.join(timeout=5)

    assert response["result"]["status"] == "streaming"
    assert not worker.is_alive()
    assert not s["running"]
    assert not GoalManager(session_key).has_goal()


def test_desktop_send_directive_activates_on_matching_prompt_claim_without_client_metadata(
    server, session, monkeypatch
):
    """Current Desktop/TUI clients submit only the directive message text."""
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    directive = _call(
        server,
        "slash.exec",
        command="goal ship the verified slice",
        session_id=sid,
    )["result"]

    assert GoalManager(session_key).state is None
    response = _accept_directive(
        server,
        session,
        directive,
        monkeypatch,
        forward_activation=False,
    )

    assert response["result"]["status"] == "streaming"
    state = GoalManager(session_key).state
    assert state is not None
    assert state.goal == "ship the verified slice"


def test_empty_successful_goal_turns_use_progressive_capped_backoff_before_budget_pause(
    server, session, monkeypatch
):
    """A fake scheduler proves empty successes cannot dispatch in a tight loop."""
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    calls = []
    scheduled = []
    schedule_changed = threading.Condition()

    class FakeTimer:
        def __init__(self, callback):
            self.callback = callback
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def fire(self):
            assert not self.cancelled
            self.callback()

    def fake_start_timer(delay, callback):
        timer = FakeTimer(callback)
        with schedule_changed:
            scheduled.append((delay, timer))
            schedule_changed.notify_all()
        return timer

    class Agent:
        session_id = session_key
        model = "test-model"
        provider = "test-provider"

        def clear_interrupt(self):
            pass

        def run_conversation(self, prompt, **kwargs):
            calls.append(prompt)
            return {"final_response": "", "messages": [], "failed": False}

    s["agent"] = Agent()
    GoalManager(session_key, default_max_turns=6).set("ship the verified slice")
    monkeypatch.setattr(server, "_start_goal_continuation_timer", fake_start_timer)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {})

    response = _call(server, "prompt.submit", session_id=sid, text="start now")

    assert response["result"]["status"] == "streaming"
    with schedule_changed:
        assert schedule_changed.wait_for(lambda: len(scheduled) == 1, timeout=5)
    assert len(calls) == 1  # not dispatched until the injected scheduler fires

    for expected_schedules in range(2, 6):
        scheduled[-1][1].fire()
        with schedule_changed:
            assert schedule_changed.wait_for(
                lambda: len(scheduled) == expected_schedules, timeout=5
            )

    scheduled[-1][1].fire()
    worker = s["_run_thread"]
    worker.join(timeout=5)
    assert not worker.is_alive()

    delays = [delay for delay, _timer in scheduled]
    assert delays[0] > 0
    assert delays == sorted(delays)
    assert delays[-1] == server._GOAL_EMPTY_BACKOFF_MAX_S
    assert all(delay <= server._GOAL_EMPTY_BACKOFF_MAX_S for delay in delays)
    state = GoalManager(session_key).state
    assert state.status == "paused"
    assert state.turns_used == state.max_turns == 6
    assert len(calls) == 6


def test_empty_success_backoff_caps_without_overflow_at_large_configured_budget(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    state = GoalManager(session_key, default_max_turns=2048).set(
        "ship the verified slice"
    )
    scheduled = []
    monkeypatch.setattr(
        server,
        "_start_goal_continuation_timer",
        lambda delay, _callback: scheduled.append(delay) or MagicMock(),
    )
    s["_goal_empty_success_streak"] = 1024

    server._schedule_empty_goal_continuation(
        1,
        sid,
        s,
        "continue the verified slice",
        goal_created_at=state.created_at,
    )

    assert scheduled == [server._GOAL_EMPTY_BACKOFF_MAX_S]
    assert s["_goal_empty_success_streak"] == 1025


@pytest.mark.parametrize("transition", ["pause", "clear", "newer-goal"])
def test_stale_empty_success_timer_cannot_resurrect_invalidated_goal_generation(
    server, session, monkeypatch, transition
):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    callbacks = []
    dispatched = []

    class FakeTimer:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    def fake_start_timer(_delay, callback):
        callbacks.append(callback)
        return FakeTimer()

    state = GoalManager(session_key).set("old goal")
    monkeypatch.setattr(server, "_start_goal_continuation_timer", fake_start_timer)
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args: dispatched.append("resurrected"),
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: True)
    server._schedule_empty_goal_continuation(
        1,
        sid,
        s,
        "continue old goal",
        goal_created_at=state.created_at,
    )
    stale_callback = callbacks[0]

    with s["history_lock"]:
        if transition == "newer-goal":
            server._new_pending_goal_activation(s, {"goal": "new goal"})
        else:
            server._goal_control_transition(s, transition, 20)

    # Simulate the hardest cancel race: callback was already dequeued when
    # cancel() ran. Generation/session/goal checks must still reject it.
    stale_callback()

    assert dispatched == []
    assert s["running"] is False


def test_nonempty_goal_progress_resets_empty_success_backoff(
    server, session, monkeypatch
):
    from hermes_cli import goals
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    responses = iter(["", "made concrete progress", ""])
    calls = []
    scheduled = []
    schedule_changed = threading.Condition()

    class FakeTimer:
        def __init__(self, callback):
            self.callback = callback

        def cancel(self):
            pass

    def fake_start_timer(delay, callback):
        with schedule_changed:
            scheduled.append((delay, FakeTimer(callback)))
            schedule_changed.notify_all()
        return scheduled[-1][1]

    class Agent:
        session_id = session_key
        model = "test-model"
        provider = "test-provider"

        def clear_interrupt(self):
            pass

        def run_conversation(self, prompt, **kwargs):
            calls.append(prompt)
            return {
                "final_response": next(responses),
                "messages": [],
                "failed": False,
            }

    s["agent"] = Agent()
    GoalManager(session_key, default_max_turns=5).set("ship the verified slice")
    monkeypatch.setattr(server, "_start_goal_continuation_timer", fake_start_timer)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_args, **_kwargs: ("continue", "progress", False, None),
    )
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {})

    _call(server, "prompt.submit", session_id=sid, text="start now")
    with schedule_changed:
        assert schedule_changed.wait_for(lambda: len(scheduled) == 1, timeout=5)
    scheduled[0][1].callback()
    with schedule_changed:
        assert schedule_changed.wait_for(lambda: len(scheduled) == 2, timeout=5)

    assert len(calls) == 3
    assert [delay for delay, _timer in scheduled] == [
        server._GOAL_EMPTY_BACKOFF_BASE_S,
        server._GOAL_EMPTY_BACKOFF_BASE_S,
    ]


def test_cleared_pending_goal_activation_cannot_be_replayed(server, session, monkeypatch):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    directive = _call(
        server, "command.dispatch", name="goal", arg="stale goal", session_id=sid
    )["result"]
    stale_activation = dict(directive["goal_activation"])

    _call(server, "command.dispatch", name="goal", arg="clear", session_id=sid)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    response = _call(
        server,
        "prompt.submit",
        session_id=sid,
        text=directive["message"],
        goal_activation=stale_activation,
    )

    assert response["error"]["code"] == 4004
    assert GoalManager(session_key).state is None


def test_two_concurrent_prompt_submits_claim_only_one_worker(server, session, monkeypatch):
    sid, _, s = session
    barrier = threading.Barrier(2)
    base_lock = threading.Lock()
    first_pass_threads = set()

    class CoordinatedLock:
        def __enter__(self):
            base_lock.acquire()
            return self

        def __exit__(self, *_exc):
            should_wait = (
                threading.get_ident() not in first_pass_threads
                and not s.get("running")
            )
            if should_wait:
                first_pass_threads.add(threading.get_ident())
            base_lock.release()
            if should_wait:
                barrier.wait(timeout=5)

    s["history_lock"] = CoordinatedLock()
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    started = []
    release_worker = threading.Event()

    def run_prompt(_rid, _sid, _session, text):
        started.append(text)
        assert release_worker.wait(timeout=5)
        with _session["history_lock"]:
            _session["running"] = False

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt)
    responses = []

    def submit(text):
        responses.append(_call(server, "prompt.submit", session_id=sid, text=text))

    threads = [threading.Thread(target=submit, args=(text,)) for text in ("one", "two")]
    for thread in threads:
        thread.start()
    deadline = time.time() + 5
    while time.time() < deadline and len(responses) < 2:
        time.sleep(0.01)
    release_worker.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(started) == 1
    assert sum("result" in response for response in responses) == 1
    assert sum(response.get("error", {}).get("code") == 4009 for response in responses) == 1


def test_interrupt_during_agent_init_rolls_back_goal_activation(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    directive = _call(
        server, "command.dispatch", name="goal", arg="init race", session_id=sid
    )["result"]
    init_started = threading.Event()
    release_init = threading.Event()

    class Agent:
        def interrupt(self):
            pass

    s["agent"] = Agent()
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)

    init_wait_calls = 0

    def wait_agent(_session, _rid):
        nonlocal init_wait_calls
        init_wait_calls += 1
        if init_wait_calls == 1:
            init_started.set()
            assert release_init.wait(timeout=5)
        return None

    monkeypatch.setattr(server, "_wait_agent", wait_agent)
    response = _call(
        server,
        "prompt.submit",
        session_id=sid,
        text=directive["message"],
        goal_activation=directive["goal_activation"],
    )
    assert init_started.wait(timeout=5)
    _call(server, "session.interrupt", session_id=sid)
    release_init.set()
    s["_run_thread"].join(timeout=5)

    assert response["result"]["status"] == "streaming"
    assert not GoalManager(session_key).has_goal()


def test_pause_wins_when_goal_judge_save_was_blocked(server, session, monkeypatch):
    from hermes_cli import goals
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    judge_started = threading.Event()
    release_judge = threading.Event()

    class Agent:
        session_id = session_key
        model = "test-model"
        provider = "test-provider"

        def clear_interrupt(self):
            pass

        def run_conversation(self, _prompt, **_kwargs):
            return {"final_response": "still working", "messages": [], "failed": False}

    def blocked_judge(*_args, **_kwargs):
        judge_started.set()
        assert release_judge.wait(timeout=5)
        return "continue", "more work remains", False, None

    s["agent"] = Agent()
    GoalManager(session_key).set("race-safe goal")
    monkeypatch.setattr(goals, "judge_goal", blocked_judge)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda _sid, _session: None)
    monkeypatch.setattr(server, "_wait_agent", lambda _session, _rid: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {})

    _call(server, "prompt.submit", session_id=sid, text="continue")
    assert judge_started.wait(timeout=5)
    pause = _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    release_judge.set()
    deadline = time.time() + 5
    while time.time() < deadline and s.get("running"):
        time.sleep(0.01)

    assert "paused" in pause["result"]["output"].lower()
    assert GoalManager(session_key).state.status == "paused"


def test_goal_state_is_isolated_to_session_profile_home(
    server, session, hermes_home, tmp_path, monkeypatch
):
    from hermes_cli.goals import GoalManager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    sid, session_key, s = session
    profile_home = tmp_path / "profile-a"
    profile_home.mkdir()
    s["profile_home"] = str(profile_home)
    directive = _call(
        server, "command.dispatch", name="goal", arg="profile-local goal", session_id=sid
    )["result"]
    _accept_directive(server, session, directive, monkeypatch)

    assert GoalManager(session_key).state is None
    token = set_hermes_home_override(profile_home)
    try:
        state = GoalManager(session_key).state
    finally:
        reset_hermes_home_override(token)
    assert state is not None
    assert state.goal == "profile-local goal"


def test_disconnect_does_not_cancel_the_backend_goal_owner(server, session, monkeypatch):
    sid, _, s = session

    class Agent:
        closed = False

        def close(self):
            self.closed = True

    transport = object()
    agent = Agent()
    s.update({"transport": transport, "running": True, "agent": agent})
    scheduled = []
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", scheduled.append)

    reaped, detached = server._close_sessions_for_transport(transport)

    assert (reaped, detached) == (0, 1)
    assert server._sessions[sid] is s
    assert s["running"] is True
    assert s["transport"] is server._detached_ws_transport
    assert agent.closed is False
    assert scheduled == [sid]
    assert not server._ws_session_is_orphaned(s)


def test_disconnect_cannot_overwrite_concurrent_reconnect_transport(
    server, session, monkeypatch
):
    sid, _, original = session
    old_transport = object()
    new_transport = object()
    disconnect_snapshot = threading.Event()
    release_disconnect = threading.Event()

    class BlockingSession(dict):
        def get(self, key, default=None):
            if key == "close_on_disconnect":
                disconnect_snapshot.set()
                assert release_disconnect.wait(5)
            return super().get(key, default)

    s = BlockingSession(original)
    s["transport"] = old_transport
    server._sessions[sid] = s
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", lambda _sid: None)
    disconnect = threading.Thread(
        target=server._close_sessions_for_transport, args=(old_transport,)
    )
    disconnect.start()
    assert disconnect_snapshot.wait(1)

    reconnect_done = threading.Event()

    def reconnect_session():
        with server._session_resume_lock:
            server._live_session_payload(sid, s, transport=new_transport)
        reconnect_done.set()

    reconnect = threading.Thread(target=reconnect_session)
    reconnect.start()
    assert not reconnect_done.wait(0.05)
    release_disconnect.set()
    disconnect.join(2)
    reconnect.join(2)

    assert reconnect_done.is_set()
    assert s["transport"] is new_transport


def test_live_session_lookup_is_profile_aware_for_same_session_key(
    server, tmp_path
):
    key = "same-id"
    home_a = (tmp_path / "profile-a").resolve()
    home_b = (tmp_path / "profile-b").resolve()
    server._sessions.clear()
    server._sessions.update(
        {
            "live-a": {"session_key": key, "profile_home": str(home_a)},
            "live-b": {"session_key": key, "profile_home": str(home_b)},
        }
    )

    assert server._find_live_session_by_key(key, profile_home=home_a)[0] == "live-a"
    assert server._find_live_session_by_key(key, profile_home=home_b)[0] == "live-b"
    assert server._resume_goal_session_for_recovery(key, home_b)[0] == "live-b"


def test_production_goal_scheduler_explicit_lifecycle_discovers_profiles(
    server, hermes_home, tmp_path, monkeypatch
):
    import hermes_constants
    from hermes_state import SessionDB

    root = tmp_path / "canonical-root"
    secondary = root / "profiles" / "work"
    secondary.mkdir(parents=True)
    db = SessionDB(secondary / "state.db")
    db.close()
    seen = set()
    recovered = threading.Event()

    def recover(coordinator):
        seen.add(Path(coordinator.home).resolve())
        if {hermes_home.resolve(), secondary.resolve()} <= seen:
            recovered.set()
        return 0

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr(server, "_GOAL_RECOVERY_SCAN_S", 0.01)
    monkeypatch.setattr(server, "_recover_active_goals_once", recover)
    monkeypatch.setattr(server, "_goal_recovery_supervisor", None)

    server._start_goal_recovery_scheduler()
    try:
        assert recovered.wait(2)
    finally:
        assert server._shutdown_goal_recovery_scheduler() is True

    assert seen == {hermes_home.resolve(), secondary.resolve()}


def test_accepted_activation_claims_durable_backend_lease(server, session, hermes_home, monkeypatch):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    directive = _call(
        server, "command.dispatch", name="goal", arg="survive backend restart", session_id=session[0]
    )["result"]
    _accept_directive(server, session, directive, monkeypatch)

    state = GoalManager(session[1]).state
    record = GoalExecutionCoordinator(hermes_home, owner_id="observer").get(session[1])
    assert record is not None
    assert record.generation == state.generation
    assert record.owner_id == server._GOAL_BACKEND_OWNER_ID
    assert record.lease_expires_at > time.time()


def test_coordinator_construction_is_single_flight_per_home_not_global(
    server, hermes_home, monkeypatch
):
    home_a = hermes_home / "coordinator-race-a"
    home_b = hermes_home / "coordinator-race-b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setattr(server, "_goal_coordinators", {})
    monkeypatch.setattr(server, "_goal_coordinator_inflight", {}, raising=False)
    monkeypatch.setattr(server, "_goal_coordinators_lock", threading.Lock())
    callers_ready = threading.Barrier(3)
    home_a_constructor_entered = threading.Event()
    release_home_a_constructor = threading.Event()
    home_b_returned = threading.Event()
    instances = {str(home_a): [], str(home_b): []}
    home_a_results = []
    home_b_results = []
    errors = []

    class Coordinator:
        def __init__(self, home):
            key = str(home)
            instances[key].append(self)
            if key == str(home_a):
                home_a_constructor_entered.set()
                if not release_home_a_constructor.wait(timeout=5):
                    raise TimeoutError("test did not release home A construction")

    monkeypatch.setattr(server, "_new_goal_execution_coordinator", Coordinator)

    def access_home_a():
        try:
            callers_ready.wait(timeout=5)
            home_a_results.append(
                server._goal_execution_coordinator({"profile_home": str(home_a)})
            )
        except BaseException as exc:
            errors.append(exc)

    def access_home_b():
        try:
            home_b_results.append(
                server._goal_execution_coordinator({"profile_home": str(home_b)})
            )
            home_b_returned.set()
        except BaseException as exc:
            errors.append(exc)

    home_a_threads = [threading.Thread(target=access_home_a) for _ in range(2)]
    for thread in home_a_threads:
        thread.start()
    callers_ready.wait(timeout=5)
    assert home_a_constructor_entered.wait(timeout=5)

    home_b_thread = threading.Thread(target=access_home_b)
    home_b_thread.start()
    try:
        assert home_b_returned.wait(timeout=0.5)
    finally:
        release_home_a_constructor.set()
        for thread in [*home_a_threads, home_b_thread]:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in [*home_a_threads, home_b_thread])
    assert errors == []
    assert len(instances[str(home_a)]) == 1
    assert len(home_a_results) == 2
    assert home_a_results[0] is home_a_results[1]
    assert len(instances[str(home_b)]) == 1
    assert home_b_results == instances[str(home_b)]


def test_failed_coordinator_construction_clears_single_flight_for_retry(
    server, hermes_home, monkeypatch
):
    coordinator_home = hermes_home / "coordinator-retry"
    coordinator_home.mkdir()
    monkeypatch.setattr(server, "_goal_coordinators", {})
    monkeypatch.setattr(server, "_goal_coordinator_inflight", {})
    monkeypatch.setattr(server, "_goal_coordinators_lock", threading.Lock())
    constructed = object()
    attempts = 0

    def construct(_home):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("initialization failed")
        return constructed

    monkeypatch.setattr(server, "_new_goal_execution_coordinator", construct)
    coordinator_session = {"profile_home": str(coordinator_home)}

    with pytest.raises(RuntimeError, match="initialization failed"):
        server._goal_execution_coordinator(coordinator_session)

    assert server._goal_coordinator_inflight == {}
    assert server._goal_execution_coordinator(coordinator_session) is constructed
    assert attempts == 2


def test_heartbeat_loss_fences_judge_persistence_and_followup(
    server, session, hermes_home, monkeypatch
):
    from hermes_cli.goals import GoalManager

    _, session_key, s = session
    state = GoalManager(session_key).set("do not persist stale judge")
    s["_goal_lease_generation"] = state.generation
    s["_goal_lease_token"] = "lost-fence"

    class LostCoordinator:
        def heartbeat(self, *_args):
            return False

    class ImmediateThread:
        def __init__(self, *, target, **_kwargs):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(server, "_goal_execution_coordinator", lambda _s=None: LostCoordinator())
    monkeypatch.setattr(server, "_goal_heartbeat_wait", lambda _stop, _seconds: False, raising=False)
    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)

    server._start_goal_lease_heartbeat(s, state.generation)

    assert s["_goal_lease_lost"] is True
    before = GoalManager(session_key).state.to_json()
    assert server._evaluate_goal_turn(s, "stale progress", []) is None
    assert GoalManager(session_key).state.to_json() == before
    assert server._goal_followup_allowed(s) is False


def test_teardown_stops_heartbeat_and_durably_releases_active_goal(
    server, session, hermes_home, monkeypatch
):
    from hermes_cli.goals import GoalManager

    _, session_key, s = session
    state = GoalManager(session_key).set("recover after teardown")
    assert server._claim_goal_execution(s, state)
    stop = s["_goal_lease_heartbeat_stop"]
    monkeypatch.setattr(server, "_finalize_session", lambda sess, **_kw: sess.__setitem__("_finalized", True))

    server._teardown_session(s, end_reason="test")

    assert stop.is_set()
    record = server._goal_execution_coordinator(s).get(session_key)
    assert record.owner_id is None
    assert record.next_run_at > 0


def test_goal_runtime_shutdown_releases_sessions_before_closing_all_coordinators(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalManager

    _, session_key, s = session
    monkeypatch.setattr(server, "_goal_coordinator_closing", threading.Event())
    monkeypatch.setattr(server, "_goal_coordinator_generation", 0)
    state = GoalManager(session_key).set("recover during shutdown")
    assert server._claim_goal_execution(s, state)
    coordinator = server._goal_execution_coordinator(s)
    events = []
    original_schedule = coordinator.schedule
    original_close = coordinator.close

    def schedule(*args, **kwargs):
        events.append("release")
        return original_schedule(*args, **kwargs)

    def close():
        events.append("close")
        return original_close()

    monkeypatch.setattr(coordinator, "schedule", schedule)
    monkeypatch.setattr(coordinator, "close", close)
    monkeypatch.setattr(server, "_finalize_session", lambda sess, **_kw: None)

    server._shutdown_goal_runtime()

    assert events == ["release", "close"]
    assert server._sessions == {}
    assert server._goal_coordinators == {}
    assert server._goal_coordinator_inflight == {}
    assert coordinator.db._conn is None


def test_shutdown_waits_for_inflight_constructor_and_cancels_its_publication(
    server, hermes_home, monkeypatch
):
    coordinator_home = hermes_home / "coordinator-shutdown-race"
    coordinator_home.mkdir()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()
    shutdown_done = threading.Event()
    closing = threading.Event()
    constructed = []
    construction_errors = []
    shutdown_results = []

    class Coordinator:
        def __init__(self, _home):
            self.closed = False
            constructed.append(self)
            constructor_entered.set()
            assert release_constructor.wait(timeout=5)

        def close(self):
            close_entered.set()
            assert release_close.wait(timeout=5)
            self.closed = True

    monkeypatch.setattr(server, "_goal_coordinators", {})
    monkeypatch.setattr(server, "_goal_coordinator_inflight", {})
    monkeypatch.setattr(server, "_goal_coordinators_lock", threading.Lock())
    monkeypatch.setattr(server, "_goal_coordinator_closing", closing, raising=False)
    monkeypatch.setattr(server, "_goal_coordinator_generation", 0, raising=False)
    monkeypatch.setattr(server, "_goal_recovery_supervisor", None)
    monkeypatch.setattr(server, "_new_goal_execution_coordinator", Coordinator)

    def construct():
        try:
            server._goal_execution_coordinator({"profile_home": str(coordinator_home)})
        except BaseException as exc:
            construction_errors.append(exc)

    def shutdown():
        shutdown_results.append(server._shutdown_goal_recovery_scheduler())
        shutdown_done.set()

    constructor_thread = threading.Thread(target=construct)
    shutdown_thread = threading.Thread(target=shutdown)
    constructor_thread.start()
    assert constructor_entered.wait(timeout=5)
    shutdown_thread.start()
    assert closing.wait(timeout=5)
    with server._goal_coordinators_lock:
        initialization_done = next(iter(server._goal_coordinator_inflight.values()))[0]
    assert not shutdown_done.is_set()

    release_constructor.set()
    assert close_entered.wait(timeout=5)
    assert not initialization_done.is_set()
    assert not shutdown_done.is_set()
    release_close.set()
    constructor_thread.join(timeout=5)
    shutdown_thread.join(timeout=5)

    assert not constructor_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert shutdown_results == [True]
    assert len(constructed) == 1
    assert constructed[0].closed is True
    assert len(construction_errors) == 1
    assert isinstance(construction_errors[0], RuntimeError)
    assert server._goal_coordinators == {}
    assert server._goal_coordinator_inflight == {}


def test_timed_out_shutdown_refuses_reopen_until_cancelled_constructor_close_finishes(
    server, hermes_home, monkeypatch
):
    from hermes_cli import goal_execution

    coordinator_home = hermes_home / "coordinator-shutdown-timeout-race"
    coordinator_home.mkdir()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    close_entered = threading.Event()
    release_close = threading.Event()
    closing = threading.Event()
    construction_errors = []
    shutdown_results = []

    class Coordinator:
        def __init__(self, _home):
            constructor_entered.set()
            assert release_constructor.wait(timeout=5)

        def close(self):
            close_entered.set()
            assert release_close.wait(timeout=5)

    class Supervisor:
        closed_coordinators = []

        def __init__(self):
            self.starts = 0

        def start(self):
            self.starts += 1

    supervisor = Supervisor()
    monkeypatch.setattr(server, "_goal_coordinators", {})
    monkeypatch.setattr(server, "_goal_coordinator_inflight", {})
    monkeypatch.setattr(server, "_goal_coordinators_lock", threading.Lock())
    monkeypatch.setattr(server, "_goal_coordinator_closing", closing, raising=False)
    monkeypatch.setattr(server, "_goal_coordinator_generation", 0, raising=False)
    monkeypatch.setattr(server, "_goal_recovery_supervisor", None)
    monkeypatch.setattr(server, "_GOAL_COORDINATOR_SHUTDOWN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(server, "_new_goal_execution_coordinator", Coordinator)
    monkeypatch.setattr(
        goal_execution,
        "get_goal_recovery_supervisor",
        lambda _key, _factory: supervisor,
    )

    def construct():
        try:
            server._goal_execution_coordinator({"profile_home": str(coordinator_home)})
        except BaseException as exc:
            construction_errors.append(exc)

    constructor_thread = threading.Thread(target=construct)
    constructor_thread.start()
    assert constructor_entered.wait(timeout=5)

    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(
            server._shutdown_goal_recovery_scheduler()
        )
    )
    shutdown_thread.start()
    assert closing.wait(timeout=5)
    release_constructor.set()
    assert close_entered.wait(timeout=5)
    shutdown_thread.join(timeout=5)

    assert not shutdown_thread.is_alive()
    assert shutdown_results == [False]
    with server._goal_coordinators_lock:
        cleanup_done = next(iter(server._goal_coordinator_inflight.values()))[0]
    assert not cleanup_done.is_set()
    with pytest.raises(
        RuntimeError, match="cannot reopen goal coordinator registry during shutdown"
    ):
        server._start_goal_recovery_scheduler()
    assert closing.is_set()
    assert supervisor.starts == 0

    release_close.set()
    constructor_thread.join(timeout=5)

    assert not constructor_thread.is_alive()
    assert cleanup_done.is_set()
    assert server._goal_coordinator_inflight == {}
    assert len(construction_errors) == 1
    assert isinstance(construction_errors[0], RuntimeError)

    server._start_goal_recovery_scheduler()

    assert not closing.is_set()
    assert server._goal_coordinator_generation == 2


def test_retry_failed_cas_keeps_exact_local_goal_ownership(
    server, session, monkeypatch
):
    from hermes_cli.goals import GoalManager

    _, session_key, s = session
    state = GoalManager(session_key).set("keep newer ownership")
    assert server._claim_goal_execution(s, state)
    coordinator = server._goal_execution_coordinator(s)
    generation = s["_goal_lease_generation"]
    claim_token = s["_goal_lease_token"]
    heartbeat_stop = s["_goal_lease_heartbeat_stop"]
    monkeypatch.setattr(coordinator, "schedule", lambda *_args, **_kwargs: False)

    server._retry_or_pause_goal_execution(s, RuntimeError("old dispatch failed"))

    assert s["_goal_lease_generation"] == generation
    assert s["_goal_lease_token"] == claim_token
    assert s["_goal_lease_heartbeat_stop"] is heartbeat_stop
    assert not heartbeat_stop.is_set()


def test_wait_schedule_stale_cas_preserves_concurrently_reclaimed_goal(
    server, session, monkeypatch
):
    _, session_key, s = session
    schedule_entered = threading.Event()
    release_schedule = threading.Event()
    old_stop = threading.Event()
    new_stop = threading.Event()
    scheduled = []

    class Coordinator:
        def schedule(self, *args, **kwargs):
            scheduled.append((args, kwargs))
            schedule_entered.set()
            assert release_schedule.wait(timeout=5)
            return False

    s.update(
        {
            "_goal_lease_generation": 1,
            "_goal_lease_token": "claim-1",
            "_goal_lease_heartbeat_stop": old_stop,
        }
    )
    monkeypatch.setattr(server, "_goal_execution_coordinator", lambda _s=None: Coordinator())
    worker = threading.Thread(
        target=server._schedule_goal_execution_from_state,
        args=(s, SimpleNamespace(generation=1, waiting_until=time.time() + 10)),
    )
    worker.start()
    assert schedule_entered.wait(timeout=5)
    with s["history_lock"]:
        s["_goal_lease_generation"] = 2
        s["_goal_lease_token"] = "claim-2"
        s["_goal_lease_heartbeat_stop"] = new_stop
    release_schedule.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert scheduled[0][0][:3] == (session_key, 1, "claim-1")
    assert s["_goal_lease_generation"] == 2
    assert s["_goal_lease_token"] == "claim-2"
    assert s["_goal_lease_heartbeat_stop"] is new_stop
    assert not new_stop.is_set()


def test_pause_failure_race_preserves_newer_goal_and_claim(
    server, session, hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    _, session_key, s = session
    manager = GoalManager(session_key)
    old_state = manager.set("old work")
    assert server._claim_goal_execution(s, old_state)
    coordinator = server._goal_execution_coordinator(s)
    old_generation = s["_goal_lease_generation"]
    old_token = s["_goal_lease_token"]
    original_pause = coordinator.pause_after_failure
    replacement = GoalExecutionCoordinator(
        hermes_home, owner_id="replacement", clock=lambda: time.time()
    )

    monkeypatch.setattr(
        coordinator,
        "get",
        lambda _key: SimpleNamespace(
            generation=old_generation,
            owner_id=server._GOAL_BACKEND_OWNER_ID,
            claim_token=old_token,
            attempt=4,
        ),
    )

    def replace_before_pause(*args, **kwargs):
        newer_state = manager.set("newer work")
        assert coordinator.invalidate(session_key, newer_state.generation)
        newer_claim = replacement.claim(session_key, newer_state.generation)
        assert newer_claim is not None
        return original_pause(*args, **kwargs)

    monkeypatch.setattr(coordinator, "pause_after_failure", replace_before_pause)
    server._retry_or_pause_goal_execution(s, RuntimeError("blocked old dispatch"))

    current = GoalManager(session_key).state
    assert current is not None
    assert current.status == "active"
    assert current.goal == "newer work"
    lease = replacement.get(session_key)
    assert lease is not None
    assert lease.owner_id == "replacement"
    assert s["_goal_lease_generation"] == old_generation
    assert s["_goal_lease_token"] == old_token


def test_full_orphan_reaper_spares_active_goal_owner_then_reaps_after_pause(
    server, session, hermes_home, monkeypatch
):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    state = GoalManager(session_key).set("finish while detached")
    assert server._claim_goal_execution(s, state)
    s.update({"transport": server._detached_ws_transport, "running": False})
    callbacks = []

    class Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
        def start(self):
            pass

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_teardown_popped_session", lambda sess, **_kw: sess is not None)

    server._schedule_ws_orphan_reap(sid)
    callbacks.pop(0)()
    assert server._sessions[sid] is s

    with s["history_lock"]:
        server._goal_control_transition(s, "pause", 20)
    assert len(callbacks) == 1  # ownership transition re-armed cleanup itself
    callbacks.pop(0)()
    assert sid not in server._sessions


def test_restart_recovery_reconstructs_normal_next_user_turn(
    server, session, hermes_home, monkeypatch
):
    from hermes_cli.goal_execution import GoalExecutionCoordinator
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    s["history"] = [{"role": "user", "content": "goal"}, {"role": "assistant", "content": "progress"}]
    original = list(s["history"])
    state = GoalManager(session_key).set("recover me")
    dead = GoalExecutionCoordinator(hermes_home, owner_id="dead", clock=lambda: 1.0)
    assert dead.claim(session_key, state.generation, lease_seconds=1)
    fresh = GoalExecutionCoordinator(hermes_home, owner_id=server._GOAL_BACKEND_OWNER_ID, clock=lambda: 10.0)
    prompts = []

    monkeypatch.setattr(server, "_resume_goal_session_for_recovery", lambda _key, _home=None: (sid, s))
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: True)
    def run(_rid, _sid, restored, prompt):
        prompts.append(prompt)
        assert restored["history"] == original
        restored["running"] = False
    monkeypatch.setattr(server, "_run_prompt_submit", run)

    assert server._recover_active_goals_once(fresh) == 1
    assert prompts == [GoalManager(session_key).next_continuation_prompt()]


def test_goal_inline_contract_preserves_exact_draft_boundaries(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    goal = (
        "Ship a verified staging slice\n"
        "verify: focused staging tests pass\n"
        "constraints: preserve production behavior\n"
        "boundaries: only touch the staging path\n"
        "stop when: deployment credentials are required"
    )
    result = _call(
        server, "command.dispatch", name="goal", arg=goal, session_id=sid
    )["result"]

    assert result["type"] == "send"
    assert result["message"] == "Ship a verified staging slice"
    assert "Completion contract:" in result["notice"]
    assert GoalManager(session_key).state is None
    activation = result["goal_activation"]
    assert activation["goal"] == "Ship a verified staging slice"
    assert activation["contract"] == {
        "outcome": "",
        "verification": "focused staging tests pass",
        "constraints": "preserve production behavior",
        "boundaries": "only touch the staging path",
        "stop_when": "deployment credentials are required",
    }


def test_goal_bare_multiline_preserves_original_text_when_contract_is_empty(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    goal = "Ship the staging slice\n  without changing production\n\tkeep this formatting"

    result = _call(
        server, "command.dispatch", name="goal", arg=goal, session_id=sid
    )["result"]

    assert result["message"] == goal
    assert result["goal_activation"]["goal"] == goal
    assert GoalManager(session_key).state is None


def test_goal_draftsmanship_is_a_bare_goal_not_a_draft_request(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    with patch("hermes_cli.goals.draft_contract") as draft:
        result = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draftsmanship matters",
            session_id=sid,
        )["result"]

    draft.assert_not_called()
    assert result["message"] == "draftsmanship matters"
    assert result["goal_activation"]["goal"] == "draftsmanship matters"
    assert GoalManager(session_key).state is None


def test_goal_draft_accepts_tab_separator(server, session):
    from hermes_cli.goals import GoalContract

    sid, _, _ = session
    contract = GoalContract(verification="focused tests pass")
    with patch("hermes_cli.goals.draft_contract", return_value=contract) as draft:
        result = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft\tShip the staging slice",
            session_id=sid,
        )["result"]

    draft.assert_called_once_with("Ship the staging slice")
    assert result["message"] == "Ship the staging slice"
    assert "Completion contract:" in result["notice"]


def test_goal_draft_failure_falls_back_honestly_to_clean_free_form_goal(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    with patch(
        "hermes_cli.goals.draft_contract", side_effect=RuntimeError("model unavailable")
    ):
        result = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft Ship a verified staging slice",
            session_id=sid,
        )["result"]

    assert result["type"] == "send"
    assert result["message"] == "Ship a verified staging slice"
    assert "Couldn't draft a contract" in result["notice"]
    assert result["goal_activation"]["goal"] == "Ship a verified staging slice"
    assert result["goal_activation"]["contract"] is None
    assert GoalManager(session_key).state is None


def test_goal_draft_none_falls_back_to_clean_free_form_goal(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, _ = session
    with patch("hermes_cli.goals.draft_contract", return_value=None):
        result = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft Ship a verified staging slice",
            session_id=sid,
        )["result"]

    assert result["message"] == "Ship a verified staging slice"
    assert "Couldn't draft a contract" in result["notice"]
    assert result["goal_activation"]["contract"] is None
    assert GoalManager(session_key).state is None


def test_goal_set_rejects_busy_session_without_drafting_or_persisting(server, session):
    from hermes_cli.goals import GoalManager

    sid, session_key, s = session
    s["running"] = True
    with patch("hermes_cli.goals.draft_contract") as draft:
        response = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="draft Ship a verified staging slice",
            session_id=sid,
        )

    draft.assert_not_called()
    assert response["error"]["code"] == 4009
    assert GoalManager(session_key).state is None


def test_goal_activation_loses_concurrent_prompt_claim_without_persisting(server, session):
    from hermes_cli.goals import GoalContract, GoalManager

    sid, session_key, s = session
    started = threading.Event()
    release = threading.Event()
    response = {}

    def slow_draft(_objective):
        started.set()
        assert release.wait(timeout=5)
        return GoalContract(verification="focused tests pass")

    def dispatch_goal():
        response.update(
            _call(
                server,
                "command.dispatch",
                name="goal",
                arg="draft Ship a verified staging slice",
                session_id=sid,
            )
        )

    worker = threading.Thread(target=dispatch_goal)
    try:
        with patch("hermes_cli.goals.draft_contract", side_effect=slow_draft):
            worker.start()
            assert started.wait(timeout=5)
            with s["history_lock"]:
                s["running"] = True
            release.set()
            worker.join(timeout=5)
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert response["error"]["code"] == 4009
    assert GoalManager(session_key).state is None


@pytest.mark.parametrize("command_name", ["goal", "/goal"])
def test_goal_draft_command_dispatch_is_offloaded_from_rpc_reader(
    server, session, command_name
):
    """A model-backed draft must not block the stdio/WebSocket reader."""
    from hermes_cli.goals import GoalContract

    class Transport:
        def __init__(self):
            self.written = []
            self.done = threading.Event()

        def write(self, obj):
            self.written.append(obj)
            self.done.set()
            return True

    sid, _, _ = session
    started = threading.Event()
    release = threading.Event()
    transport = Transport()

    def slow_draft(_objective):
        started.set()
        release.wait(timeout=1)
        return GoalContract(verification="focused tests pass")

    try:
        with patch("hermes_cli.goals.draft_contract", side_effect=slow_draft):
            response = server.dispatch(
                {
                    "id": "goal-draft",
                    "method": "command.dispatch",
                    "params": {
                        "name": command_name,
                        "arg": "draft Ship the staging slice",
                        "session_id": sid,
                    },
                },
                transport=transport,
            )
            assert response is None
            assert started.wait(timeout=1)
            assert not transport.done.is_set()
            release.set()
            assert transport.done.wait(timeout=1)
    finally:
        release.set()

    assert transport.written[0]["result"]["message"] == "Ship the staging slice"


def test_goal_pause_after_accepted_set(server, session, monkeypatch):
    sid, session_key, _ = session
    directive = _call(
        server, "command.dispatch", name="goal", arg="write a story", session_id=sid
    )["result"]
    _accept_directive(server, session, directive, monkeypatch)
    r = _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "paused" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state.status == "paused"


def test_goal_resume_claims_exactly_one_execution_lease(server, session, monkeypatch):
    sid, session_key, _ = session
    directive = _call(
        server, "command.dispatch", name="goal", arg="write a story", session_id=sid
    )["result"]
    _accept_directive(server, session, directive, monkeypatch)
    _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="resume", session_id=sid)

    from hermes_cli.goals import GoalManager

    resume = r["result"]
    assert resume["type"] == "send"
    assert "resumed" in resume["notice"].lower()
    assert GoalManager(session_key).state.status == "paused"

    calls = 0
    def count_lease():
        nonlocal calls
        calls += 1

    _accept_directive(
        server,
        session,
        resume,
        monkeypatch,
        forward_activation=False,
        on_run=count_lease,
    )

    assert calls == 1
    assert GoalManager(session_key).state.status == "active"


def test_goal_clear_removes_active_goal(server, session, monkeypatch):
    sid, session_key, _ = session
    directive = _call(
        server, "command.dispatch", name="goal", arg="write a story", session_id=sid
    )["result"]
    _accept_directive(server, session, directive, monkeypatch)
    r = _call(server, "command.dispatch", name="goal", arg="clear", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "cleared" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    # After clear the row is marked status=cleared (kept for audit);
    # ``has_goal()`` / ``is_active()`` return False so the goal loop
    # stays off and ``status`` reports "No active goal".
    mgr = GoalManager(session_key)
    assert not mgr.has_goal()
    assert not mgr.is_active()
    assert "No active goal" in mgr.status_line()


def test_goal_stop_and_done_are_clear_aliases(server, session, monkeypatch):
    sid, _, _ = session
    first = _call(
        server, "command.dispatch", name="goal", arg="first goal", session_id=sid
    )["result"]
    _accept_directive(server, session, first, monkeypatch)
    r = _call(server, "command.dispatch", name="goal", arg="stop", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()

    second = _call(
        server, "command.dispatch", name="goal", arg="second goal", session_id=sid
    )["result"]
    _accept_directive(server, session, second, monkeypatch)
    r = _call(server, "command.dispatch", name="goal", arg="done", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()


def test_goal_requires_session(server):
    r = _call(server, "command.dispatch", name="goal", arg="nope", session_id="unknown")
    assert "error" in r
    assert r["error"]["code"] == 4001


# ── slash.exec /goal routing ──────────────────────────────────────────


def test_slash_exec_routes_goal_to_command_dispatch(server, session):
    """slash.exec must route /goal directly to command.dispatch internally
    instead of returning an error.  Previously the 4018 error required the
    TUI client to retry via command.dispatch, but some clients failed the
    fallback, leaving the command empty ("empty command")."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="goal status", session_id=sid)
    # Should succeed by routing to command.dispatch internally
    assert "result" in r
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_pending_input_commands_includes_goal(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'goal' — removing it would
    silently re-break the TUI."""
    assert "goal" in server._PENDING_INPUT_COMMANDS


# ── command.dispatch /moa ────────────────────────────────────────────

def _write_moa_config(home, text):
    cfg_path = home / "config.yaml"
    cfg_path.write_text(text)


def test_moa_bare_returns_usage(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="", session_id=sid)
    # Bare /moa is usage-only now; switching to a preset is via the model picker.
    assert "error" in r
    assert "model_override" not in s


def test_moa_arg_is_always_one_shot(server, session, hermes_home):
    # Any arg (even a preset name) is a one-shot prompt through the DEFAULT
    # preset; /moa never does a sticky switch anymore.
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default: {}
    review:
      reference_models:
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, s = session
    r = _call(server, "command.dispatch", name="moa", arg="review", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "review"
    assert "one-shot" in result["notice"]
    # Lazy session (no live agent) → MoA preset pinned via model_override for
    # the build, and it is the DEFAULT preset, not the "review" arg.
    assert s["model_override"]["provider"] == "moa"
    assert s["model_override"]["model"] == "default"


def test_moa_non_preset_returns_one_shot_send(server, session, hermes_home):
    _write_moa_config(hermes_home, """
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""")
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="moa", arg="inspect this project", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "inspect this project"
    assert "one-shot" in result["notice"]


def test_pending_input_commands_includes_moa(server):
    assert "moa" in server._PENDING_INPUT_COMMANDS
