"""Factory tuple propagation through the existing Kanban worker tools."""

from __future__ import annotations

import json
from types import SimpleNamespace


class _Connection:
    def close(self) -> None:
        pass


def _factory_env(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_factory")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "41")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:worker")
    monkeypatch.setenv("HERMES_FACTORY_EXECUTION_GENERATION", "7")
    monkeypatch.setenv("HERMES_FACTORY_AUTHORIZATION_ID", "fa_exact")
    monkeypatch.setenv("HERMES_FACTORY_BINDING_DIGEST", "binding_exact")


def test_heartbeat_forwards_exact_factory_tuple(monkeypatch):
    from tools import kanban_tools as kt

    _factory_env(monkeypatch)
    calls: dict[str, dict] = {}
    fake_kb = SimpleNamespace(
        heartbeat_claim=lambda _conn, _tid, **kwargs: calls.setdefault("claim", kwargs) or True,
        heartbeat_worker=lambda _conn, _tid, **kwargs: calls.setdefault("worker", kwargs) or True,
    )
    monkeypatch.setattr(kt, "_connect", lambda **_kwargs: (fake_kb, _Connection()))

    result = json.loads(kt._handle_heartbeat({"note": "alive"}))

    assert result["ok"] is True
    assert calls["claim"] == {
        "claimer": "host:worker",
        "expected_run_id": 41,
        "expected_generation": 7,
        "authorization_id": "fa_exact",
    }
    assert calls["worker"]["expected_run_id"] == 41


def test_automatic_heartbeat_forwards_exact_factory_tuple(monkeypatch):
    from tools import kanban_tools as kt

    _factory_env(monkeypatch)
    calls: dict[str, dict] = {}
    fake_kb = SimpleNamespace(
        heartbeat_claim=lambda _conn, _tid, **kwargs: calls.setdefault("claim", kwargs) or True,
        heartbeat_worker=lambda _conn, _tid, **kwargs: calls.setdefault("worker", kwargs) or True,
    )
    monkeypatch.setattr(kt, "_connect", lambda **_kwargs: (fake_kb, _Connection()))
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)

    assert kt.heartbeat_current_worker_from_env() is True
    assert calls["claim"] == {
        "claimer": "host:worker",
        "expected_run_id": 41,
        "expected_generation": 7,
        "authorization_id": "fa_exact",
    }


def test_complete_forwards_exact_factory_closure_tuple(monkeypatch):
    from tools import kanban_tools as kt

    _factory_env(monkeypatch)
    calls: dict[str, dict] = {}
    fake_kb = SimpleNamespace(
        get_task=lambda _conn, _tid: SimpleNamespace(goal_mode=False),
        complete_task=lambda _conn, _tid, **kwargs: calls.setdefault("complete", kwargs) or True,
        latest_run=lambda _conn, _tid: SimpleNamespace(id=41),
        ArtifactPreservationError=RuntimeError,
        HallucinatedCardsError=ValueError,
    )
    monkeypatch.setattr(kt, "_connect", lambda **_kwargs: (fake_kb, _Connection()))
    monkeypatch.setattr(kt, "_stamp_worker_session_metadata", lambda _tid, metadata: metadata)

    result = json.loads(kt._handle_complete({"summary": "independently verified"}))

    assert result["ok"] is True
    assert calls["complete"]["expected_run_id"] == 41
    assert calls["complete"]["expected_generation"] == 7
    assert calls["complete"]["authorization_id"] == "fa_exact"
