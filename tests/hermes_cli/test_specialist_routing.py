from __future__ import annotations

import json
import threading
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from hermes_cli import specialist_routing as routing
from hermes_cli.specialist_routing import (
    GROK_MODEL,
    SpecialistRoutingBlocked,
    artifact_transition_allowed,
    build_brain_projection,
    build_grok_contract,
    create_probe_receipt,
    grok_role_for_scope,
    required_risk_roles,
    run_live_probe,
    select_specialist_route,
    validate_grok_contract,
)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    connection = kb.connect()
    yield connection
    connection.close()


@pytest.fixture
def project_id(conn, tmp_path):
    repo = tmp_path / "design-repo"
    repo.mkdir()
    with pdb.connect_closing() as connection:
        return pdb.create_project(
            connection, name="Design", folders=[str(repo)], primary_path=str(repo),
        )


def _successful_receipt(
    *, now: int = 1_000, provider: str = "xai-oauth",
    model: str = GROK_MODEL, profile: str = "grok-creative",
):
    def infer(**kwargs):
        assert kwargs["provider"] == provider
        assert kwargs["model"] == model
        assert kwargs["fallback_disabled"] is True
        return kwargs["expected_marker"]

    return create_probe_receipt(
        profile=profile,
        provider=provider,
        model=model,
        nonce="one-use-private-nonce",
        infer=infer,
        now=now,
        ttl_seconds=300,
    )


def test_product_matrix_allows_only_grok_design_scopes():
    positive = {
        ("webblotsen", "three_customer_concepts"): "creative_director",
        ("webblotsen", "cohort_responsive_critique"): "contrarian_reviewer",
        ("webblotsen", "private_design_room"): "creative_director",
        ("firmalotsen", "private_design_room"): "creative_director",
        ("the-foundry", "public_art_direction"): "creative_director",
        ("hbbq", "campaign"): "creative_director",
        ("hbbq", "catering"): "creative_director",
        ("hbbq", "reels"): "creative_director",
        ("hbbq", "campaign_premiumisation"): "creative_director",
        ("mission-control", "owner_view_hierarchy_critique"): "contrarian_reviewer",
        ("firmalotsen-personal", "onboarding"): "creative_director",
        ("firmalotsen-personal", "dashboard"): "creative_director",
        ("firmalotsen-personal", "empty_state"): "creative_director",
        ("firmalotsen-personal", "onboarding_positioning"): "creative_director",
        ("personalkollen", "onboarding"): "creative_director",
        ("personalkollen", "dashboard"): "creative_director",
        ("personalkollen", "empty_state"): "creative_director",
        ("lundin", "irreversible_positioning_review"): "contrarian_reviewer",
    }
    for route, role in positive.items():
        assert grok_role_for_scope(*route) == role

    for forbidden in (
        "code", "database_migration", "deploy", "security_review",
        "payroll_calculation", "sole_fact_verification", "final_e2e", "coo_orchestration",
    ):
        assert grok_role_for_scope("webblotsen", forbidden) is None


def test_visual_market_reaction_considers_grok_but_reliability_does_not():
    selector = getattr(routing, "should_consider_grok", None)
    assert callable(selector)
    for capability in ("visual", "feeling", "market_reaction"):
        assert selector(capability) is True
    for capability in ("tests", "calculation", "reliability", "security"):
        assert selector(capability) is False


def test_contract_rejects_prohibited_capability_and_scope_role_mismatch():
    prohibited = build_grok_contract(
        role="creative_director",
        capability="code",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="webblotsen",
        scope="three_customer_concepts",
        probe_receipt=_successful_receipt(),
    )
    with pytest.raises(SpecialistRoutingBlocked, match="capability"):
        validate_grok_contract(prohibited, now=1_001)

    mismatched = build_grok_contract(
        role="contrarian_reviewer",
        capability="visual_art_direction",
        profile="grok-reviewer",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=_successful_receipt(profile="grok-reviewer"),
    )
    with pytest.raises(SpecialistRoutingBlocked, match="scope"):
        validate_grok_contract(mismatched, now=1_001)


def test_probe_receipt_rejects_unknown_fields_before_persistence():
    receipt = _successful_receipt()
    receipt["raw_response"] = "must-never-persist"
    with pytest.raises(SpecialistRoutingBlocked, match="receipt"):
        build_grok_contract(
            role="creative_director",
            capability="visual_art_direction",
            profile="grok-creative",
            provider="xai-oauth",
            model=GROK_MODEL,
            project_id="p_design",
            product="the-foundry",
            scope="public_art_direction",
            probe_receipt=receipt,
        )


def test_grok_binding_requires_provider_and_model():
    receipt = _successful_receipt()
    contract = build_grok_contract(
        role="creative_director",
        capability="visual_art_direction",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=receipt,
    )
    validate_grok_contract(contract, now=1_001)
    assert contract["artifact_policy"] == {
        "proposal_first": True,
        "visible_clickable_artifact": True,
        "required_viewports": ["desktop", "mobile"],
        "exactly_one_mutable_writer": True,
        "reviewer_read_only": True,
        "review_exact_frozen_generation": True,
        "cohort_before_shared_renderer_change": False,
    }
    assert "code" in contract["prohibited_ownership"]
    assert "final_e2e" in contract["prohibited_ownership"]

    for field in ("provider", "model"):
        broken = json.loads(json.dumps(contract))
        broken["selected_route"][field] = ""
        with pytest.raises(SpecialistRoutingBlocked, match=field):
            validate_grok_contract(broken, now=1_001)


def test_auth_presence_alone_cannot_verify_model():
    contract = build_grok_contract(
        role="creative_director",
        capability="visual_art_direction",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=None,
    )
    contract["auth_present"] = True
    with pytest.raises(SpecialistRoutingBlocked, match="live probe"):
        validate_grok_contract(contract, now=1_001)


def test_failed_grok_probe_is_fail_closed_and_never_attributes_grok():
    def infer(**_kwargs):
        return "wrong model response"

    receipt = create_probe_receipt(
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        nonce="nonce",
        infer=infer,
        now=1_000,
    )
    assert receipt["model_verified"] is False
    assert receipt["attribution_allowed"] is False
    assert "wrong model response" not in json.dumps(receipt)


def test_successful_nonce_probe_records_sanitized_exact_route_receipt():
    receipt = _successful_receipt()
    encoded = json.dumps(receipt)
    assert receipt["model_verified"] is True
    assert receipt["attribution_allowed"] is True
    assert receipt["provider"] == "xai-oauth"
    assert receipt["model"] == GROK_MODEL
    assert receipt["marker_digest"]
    assert "one-use-private-nonce" not in encoded
    assert "HERMES_GROK_PROBE" not in encoded


def test_live_probe_adapter_constructs_exact_fallback_free_agent(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider
    import hermes_cli.profiles as profiles
    import run_agent

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "grok-creative")
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kwargs: {
        "provider": "xai-oauth", "api_mode": "codex_responses",
        "base_url": "https://api.x.ai/v1", "api_key": "opaque-test-token",
    })
    observed = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def chat(self, prompt):
            return prompt.rsplit(": ", 1)[-1]

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    receipt = run_live_probe(
        profile="grok-creative", provider="xai-oauth", model=GROK_MODEL,
        nonce="private-probe-nonce", now=1_000,
    )
    assert receipt["model_verified"] is True
    assert observed["provider"] == "xai-oauth"
    assert observed["model"] == GROK_MODEL
    assert observed["fallback_model"] == []
    assert observed["enabled_toolsets"] == []


def test_live_probe_rejects_profile_not_owned_by_active_home(monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "different-profile")
    with pytest.raises(SpecialistRoutingBlocked, match="active profile"):
        run_live_probe(
            profile="grok-creative", provider="xai-oauth", model=GROK_MODEL,
            nonce="private-probe-nonce", now=1_000,
        )


def test_expired_or_changed_route_probe_blocks_dispatch():
    receipt = _successful_receipt()
    contract = build_grok_contract(
        role="creative_director",
        capability="visual_art_direction",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=receipt,
    )
    with pytest.raises(SpecialistRoutingBlocked, match="expired"):
        validate_grok_contract(contract, now=1_301)

    contract["selected_route"]["provider"] = "xai"
    with pytest.raises(SpecialistRoutingBlocked, match="route"):
        validate_grok_contract(contract, now=1_001)


def test_claim_snapshots_profile_provider_model_skills_toolsets(conn, project_id):
    contract = build_grok_contract(
        role="creative_director",
        capability="visual_art_direction",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=_successful_receipt(now=1_000),
        skills=["popular-web-designs"],
        toolsets=["browser", "vision"],
    )
    contract["project_id"] = project_id
    task_id = kb.create_task(
        conn,
        title="Design",
        assignee="grok-creative",
        model_override=GROK_MODEL,
        provider_override="xai-oauth",
        project_id=project_id,
        specialist_contract=contract,
    )
    claimed = kb.claim_task(conn, task_id, claimer="worker", now=1_001)
    assert claimed is not None
    run = kb.list_runs(conn, task_id)[-1]
    provenance = run.metadata["routing_provenance"]
    assert provenance["profile"] == "grok-creative"
    assert provenance["provider"] == "xai-oauth"
    assert provenance["model"] == GROK_MODEL
    assert provenance["skills"] == ["popular-web-designs"]
    assert provenance["toolsets"] == ["browser", "vision"]
    assert provenance["probe_receipt"]["model_verified"] is True


def test_concurrent_specialist_claim_creates_one_provenance_snapshot(conn, project_id):
    contract = build_grok_contract(
        role="creative_director", capability="visual_art_direction",
        profile="grok-creative", provider="xai-oauth", model=GROK_MODEL,
        project_id=project_id, product="the-foundry", scope="public_art_direction",
        probe_receipt=_successful_receipt(now=1_000),
    )
    task_id = kb.create_task(
        conn, title="Concurrent design", assignee="grok-creative",
        model_override=GROK_MODEL, provider_override="xai-oauth",
        project_id=project_id, specialist_contract=contract,
    )
    barrier = threading.Barrier(2)
    outcomes = []

    def claim(label):
        connection = kb.connect()
        try:
            barrier.wait()
            outcomes.append(kb.claim_task(
                connection, task_id, claimer=label, now=1_001,
            ))
        finally:
            connection.close()

    workers = [threading.Thread(target=claim, args=(label,)) for label in ("a", "b")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert sum(result is not None for result in outcomes) == 1
    runs = kb.list_runs(conn, task_id)
    assert len(runs) == 1
    assert runs[0].metadata is not None
    assert runs[0].metadata["routing_provenance"]["project_id"] == project_id


def test_grok_dispatch_without_probe_blocks_before_spawn(conn, project_id):
    contract = build_grok_contract(
        role="creative_director",
        capability="visual_art_direction",
        profile="grok-creative",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="the-foundry",
        scope="public_art_direction",
        probe_receipt=None,
    )
    contract["project_id"] = project_id
    task_id = kb.create_task(
        conn,
        title="Design",
        assignee="grok-creative",
        model_override=GROK_MODEL,
        provider_override="xai-oauth",
        project_id=project_id,
        specialist_contract=contract,
    )
    with pytest.raises(SpecialistRoutingBlocked, match="probe"):
        kb.claim_task(conn, task_id, now=1_100)
    assert kb.get_task(conn, task_id).status == "ready"
    assert kb.list_runs(conn, task_id) == []

    verified_at = int(time.time())
    receipt = _successful_receipt(now=verified_at)
    assert kb.set_specialist_probe_receipt(conn, task_id, receipt) is True
    assert kb.claim_task(conn, task_id, now=verified_at) is not None
    run = kb.list_runs(conn, task_id)[0]
    assert run.metadata is not None
    provenance = run.metadata["routing_provenance"]
    assert provenance["profile"] == "grok-creative"
    assert provenance["provider"] == "xai-oauth"
    assert provenance["model"] == GROK_MODEL
    assert "one-use-private-nonce" not in json.dumps(provenance)


def test_completion_rejects_attribution_mismatch(conn, project_id):
    contract = build_grok_contract(
        role="contrarian_reviewer",
        capability="market_reaction",
        profile="grok-reviewer",
        provider="xai-oauth",
        model=GROK_MODEL,
        project_id="p_design",
        product="lundin",
        scope="irreversible_positioning_review",
        probe_receipt=_successful_receipt(now=1_000, profile="grok-reviewer"),
    )
    contract["project_id"] = project_id
    task_id = kb.create_task(
        conn,
        title="Review",
        assignee="grok-reviewer",
        model_override=GROK_MODEL,
        provider_override="xai-oauth",
        project_id=project_id,
        specialist_contract=contract,
    )
    kb.claim_task(conn, task_id, claimer="worker", now=1_001)
    with pytest.raises(SpecialistRoutingBlocked, match="attribution"):
        kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"routing_provenance": {"provider": "openai-codex", "model": "gpt-5.4"}},
        )
    assert kb.get_task(conn, task_id).status == "running"
    with pytest.raises(SpecialistRoutingBlocked, match="artifact evidence"):
        kb.complete_task(conn, task_id, summary="No artifact", metadata={})
    assert kb.get_task(conn, task_id).status == "running"
    closure = {
        "commit_sha": "f" * 40,
        "artifact": {
            "generation": "gen-1",
            "digest": "a" * 64,
            "author_profile": "grok-creative",
            "frozen": True,
            "visible_url": "http://127.0.0.1/review.html",
            "viewports": ["desktop", "mobile"],
        },
    }
    assert kb.complete_task(
        conn, task_id, summary="Verified review", metadata=closure,
    ) is True
    completed = kb.list_runs(conn, task_id)[0]
    assert completed.metadata is not None
    assert completed.metadata["routing_provenance"]["model"] == GROK_MODEL
    assert completed.metadata["brain_projection"]["commit_sha"] == "f" * 40


def test_delivery_contract_requires_roles_and_capabilities_not_only_assignee():
    receipt = _successful_receipt()
    contract = build_grok_contract(
        role="creative_director", capability="visual_feeling",
        profile="grok-creative", provider="xai-oauth", model=GROK_MODEL,
        project_id="p_design", product="webblotsen",
        scope="three_customer_concepts", probe_receipt=receipt,
    )
    for field in ("required_roles", "required_capabilities"):
        broken = json.loads(json.dumps(contract))
        broken[field] = []
        with pytest.raises(SpecialistRoutingBlocked):
            validate_grok_contract(broken, now=1_001)


def test_selector_honours_explicit_user_binding_before_preferences():
    candidates = [
        {"profile": "other", "provider": "xai-oauth", "model": GROK_MODEL,
         "roles": ["creative_director"], "capabilities": ["visual_feeling"], "priority": 100},
        {"profile": "diffen-grok", "provider": "xai", "model": GROK_MODEL,
         "roles": ["creative_director"], "capabilities": ["visual_feeling"], "priority": 1},
    ]
    selected = select_specialist_route(
        role="creative_director", capability="visual_feeling", candidates=candidates,
        explicit_binding={"profile": "diffen-grok", "provider": "xai", "model": GROK_MODEL},
    )
    assert selected["profile"] == "diffen-grok"


def test_selector_is_deterministic_and_does_not_mutate_inputs():
    candidates = [
        {"profile": "zeta", "provider": "xai-oauth", "model": GROK_MODEL,
         "roles": ["contrarian_reviewer"], "capabilities": ["market_reaction"], "priority": 5},
        {"profile": "alpha", "provider": "xai", "model": GROK_MODEL,
         "roles": ["contrarian_reviewer"], "capabilities": ["market_reaction"], "priority": 5},
    ]
    before = json.loads(json.dumps(candidates))
    first = select_specialist_route(
        role="contrarian_reviewer", capability="market_reaction", candidates=candidates,
    )
    second = select_specialist_route(
        role="contrarian_reviewer", capability="market_reaction", candidates=list(reversed(candidates)),
    )
    assert first == second
    assert candidates == before
    assert first["profile"] == "alpha"


def test_risk_flags_add_only_required_specialists():
    assert required_risk_roles({}) == []
    assert required_risk_roles({"security": True}) == ["security_reviewer"]
    assert required_risk_roles({"data_integration": True, "release": True}) == [
        "data_integration_specialist", "release_e2e_owner",
    ]


def test_review_pins_exact_non_self_authored_generation():
    artifact = {
        "generation": "gen-7", "digest": "a" * 64, "author_profile": "grok-creative",
        "frozen": True, "visible_url": "http://127.0.0.1/design.html",
        "viewports": ["desktop", "mobile"],
    }
    assert artifact_transition_allowed(
        phase="review", actor_profile="grok-reviewer", artifact=artifact,
    )["generation"] == "gen-7"
    with pytest.raises(SpecialistRoutingBlocked, match="independent"):
        artifact_transition_allowed(
            phase="review", actor_profile="grok-creative", artifact=artifact,
        )
    with pytest.raises(SpecialistRoutingBlocked, match="frozen"):
        artifact_transition_allowed(
            phase="review", actor_profile="grok-reviewer", artifact={**artifact, "frozen": False},
        )


def test_implementation_waits_for_accepted_design_generation():
    artifact = {
        "generation": "gen-8", "digest": "b" * 64, "author_profile": "grok-creative",
        "frozen": True, "accepted": False,
        "visible_url": "http://127.0.0.1/design.html",
        "viewports": ["desktop", "mobile"],
    }
    with pytest.raises(SpecialistRoutingBlocked, match="accepted"):
        artifact_transition_allowed(
            phase="implementation", actor_profile="john", artifact=artifact,
        )
    assert artifact_transition_allowed(
        phase="implementation", actor_profile="john", artifact={**artifact, "accepted": True},
    )["digest"] == "b" * 64


def test_webblotsen_renderer_repair_waits_for_three_concept_cohort():
    artifact = {
        "generation": "gen-9", "digest": "c" * 64, "author_profile": "grok-reviewer",
        "frozen": True, "accepted": True, "visible_url": "http://127.0.0.1/cohort.html",
        "viewports": ["desktop", "tablet", "mobile"], "concept_count": 3,
        "concept_ids": ["editorial", "cinematic", "utility"], "cohort_complete": True,
    }
    assert artifact_transition_allowed(
        phase="shared_renderer_change", actor_profile="john", artifact=artifact,
        product="webblotsen",
    )["cohort_complete"] is True
    for broken in (
        {**artifact, "concept_count": 2},
        {**artifact, "concept_ids": ["same", "same", "same"]},
        {**artifact, "viewports": ["desktop", "mobile"]},
        {**artifact, "cohort_complete": False},
    ):
        with pytest.raises(SpecialistRoutingBlocked):
            artifact_transition_allowed(
                phase="shared_renderer_change", actor_profile="john",
                artifact=broken, product="webblotsen",
            )


def test_brain_projection_is_redacted_idempotent_and_post_commit():
    provenance = {
        "profile": "grok-creative", "provider": "xai-oauth", "model": GROK_MODEL,
        "skills": ["popular-web-designs"], "toolsets": ["vision"],
        "project_id": "p_design", "required_roles": ["creative_director"],
        "required_capabilities": ["visual_feeling"],
        "probe_receipt": {**_successful_receipt(), "raw_prompt": "secret prompt", "token": "secret"},
    }
    first = build_brain_projection(
        task_id="t_12345678", run_id=7, provenance=provenance,
        commit_sha="d" * 40, artifact_generation="gen-9", artifact_digest="e" * 64,
    )
    second = build_brain_projection(
        task_id="t_12345678", run_id=7, provenance=provenance,
        commit_sha="d" * 40, artifact_generation="gen-9", artifact_digest="e" * 64,
    )
    assert first == second
    assert first["projection_id"]
    assert first["commit_sha"] == "d" * 40
    encoded = json.dumps(first)
    assert "secret prompt" not in encoded
    assert '"token"' not in encoded
    with pytest.raises(SpecialistRoutingBlocked, match="commit"):
        build_brain_projection(
            task_id="t_12345678", run_id=7, provenance=provenance,
            commit_sha="", artifact_generation="gen-9", artifact_digest="e" * 64,
        )
