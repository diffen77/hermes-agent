"""Task-local specialist role selection and verified inference routing.

Credential loading stays outside this module. The probe caller supplies the
exact inference adapter; this boundary emits only a credential-safe receipt,
never a prompt, nonce, raw model output, token, or exception message.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Optional
from urllib.parse import urlparse

GROK_MODEL = "grok-4.5"
GROK_PROVIDERS = frozenset({"xai", "xai-oauth"})
GROK_ROLES = frozenset({"creative_director", "contrarian_reviewer"})

_GROK_SCOPE_ROLES = {
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
    ("personalkollen", "onboarding_positioning"): "creative_director",
    ("lundin", "irreversible_positioning_review"): "contrarian_reviewer",
}

_PROHIBITED_OWNERSHIP = (
    "code", "database", "migration", "deploy", "operations", "security",
    "accounting", "payroll", "calculation", "sole_fact_verification",
    "final_acceptance", "final_e2e", "coo_orchestration",
)

_SAFE_RECEIPT_KEYS = frozenset({
    "receipt_version", "profile", "provider", "model", "route_digest",
    "marker_digest", "verified_at", "expires_at", "fallback_disabled",
    "model_verified", "attribution_allowed", "error_code",
})

_CONTRACT_KEYS = frozenset({
    "contract_version", "required_roles", "required_capabilities", "project_id",
    "product", "scope", "selected_route", "probe_receipt", "fallback_allowed",
    "artifact_policy", "artifact_target", "content_boundaries", "prohibited_ownership",
    "risk_flags", "required_risk_roles", "evidence_requirements",
})
_ROUTE_KEYS = frozenset({"profile", "provider", "model", "skills", "toolsets"})
_POLICY_KEYS = frozenset({
    "proposal_first", "visible_clickable_artifact", "required_viewports",
    "exactly_one_mutable_writer", "reviewer_read_only",
    "review_exact_frozen_generation", "cohort_before_shared_renderer_change",
})
_TARGET_KEYS = frozenset({"scope_id", "generation", "access_mode"})
_REVIEWER_TOOLSETS = frozenset({"vision"})

_GROK_FIRST_CAPABILITIES = frozenset({"visual", "feeling", "market_reaction"})

_PRODUCT_BOUNDARIES = {
    "webblotsen": ("source_grounded", "three_distinct_concepts"),
    "firmalotsen": ("source_grounded", "private_design_room"),
    "the-foundry": ("art_direction_only", "pre_implementation"),
    "hbbq": ("no_invented_menu_price_availability", "protected_copy_immutable"),
    "mission-control": ("hierarchy_cta_critique_only", "source_kanban_logic_immutable"),
    "firmalotsen-personal": ("no_payroll_law_pii_calculation",),
    "personalkollen": ("no_payroll_law_pii_calculation",),
    "lundin": ("contrarian_review_only", "coo_authority_immutable"),
}


class SpecialistRoutingBlocked(RuntimeError):
    """The requested specialist route is not safe to dispatch or attribute."""


def _normalize_product(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_scope(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _valid_visible_url(value: Any) -> bool:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and raw.lower() not in {"http://about:blank", "https://about:blank"}
    )


def should_consider_grok(capability: str) -> bool:
    """Return the stable routing preference without claiming availability."""

    return str(capability).strip().lower() in _GROK_FIRST_CAPABILITIES


def select_specialist_route(
    *, role: str, capability: str, candidates: list[dict[str, Any]],
    explicit_binding: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Select one exact eligible route deterministically without mutating input."""

    role = str(role).strip()
    capability = str(capability).strip()
    eligible: list[dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        candidate = dict(raw)
        if role not in (candidate.get("roles") or []):
            continue
        if capability not in (candidate.get("capabilities") or []):
            continue
        profile = str(candidate.get("profile") or "").strip()
        provider = str(candidate.get("provider") or "").strip().lower()
        model = str(candidate.get("model") or "").strip()
        if not profile or provider not in GROK_PROVIDERS or model != GROK_MODEL:
            continue
        candidate.update(profile=profile, provider=provider, model=model)
        eligible.append(candidate)
    if explicit_binding is not None:
        identity = (
            str(explicit_binding.get("profile") or "").strip(),
            str(explicit_binding.get("provider") or "").strip().lower(),
            str(explicit_binding.get("model") or "").strip(),
        )
        for candidate in eligible:
            if (candidate["profile"], candidate["provider"], candidate["model"]) == identity:
                return candidate
        raise SpecialistRoutingBlocked("explicit specialist binding is unavailable")
    if not eligible:
        raise SpecialistRoutingBlocked("no eligible specialist route")
    eligible.sort(key=lambda item: (
        -int(item.get("priority") or 0), item["profile"], item["provider"], item["model"],
    ))
    return eligible[0]


def required_risk_roles(risks: dict[str, Any]) -> list[str]:
    """Activate only high-risk roles requested by the delivery contract."""

    mapping = (
        ("security", "security_reviewer"),
        ("data_integration", "data_integration_specialist"),
        ("release", "release_e2e_owner"),
    )
    return [role for flag, role in mapping if risks.get(flag) is True]


def effective_specialist_spawn(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the exact sanitized task-local spawn configuration."""

    validate_grok_contract(contract, require_fresh=False, require_verified=False)
    route = contract["selected_route"]
    return {
        "profile": route["profile"],
        "provider": route["provider"],
        "model": route["model"],
        "skills": list(route["skills"]),
        "toolsets": list(route["toolsets"]),
        "access_mode": contract["artifact_target"]["access_mode"],
    }


def sanitized_specialist_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return only binding, non-secret fields safe for workers/read surfaces."""

    validate_grok_contract(contract, require_fresh=False, require_verified=False)
    route = contract["selected_route"]
    receipt = contract.get("probe_receipt")
    return {
        "contract_version": contract["contract_version"],
        "required_roles": list(contract["required_roles"]),
        "required_capabilities": list(contract["required_capabilities"]),
        "project_id": contract["project_id"],
        "product": contract["product"],
        "scope": contract["scope"],
        "selected_route": {
            key: (list(route[key]) if key in {"skills", "toolsets"} else route[key])
            for key in sorted(_ROUTE_KEYS)
        },
        "probe_status": ({
            "model_verified": receipt.get("model_verified"),
            "fallback_disabled": receipt.get("fallback_disabled"),
            "expires_at": receipt.get("expires_at"),
            "route_digest": receipt.get("route_digest"),
        } if isinstance(receipt, dict) else None),
        "fallback_allowed": False,
        "artifact_policy": dict(contract["artifact_policy"]),
        "artifact_target": dict(contract["artifact_target"]),
        "content_boundaries": list(contract["content_boundaries"]),
        "prohibited_ownership": list(contract["prohibited_ownership"]),
        "risk_flags": dict(contract["risk_flags"]),
        "required_risk_roles": list(contract["required_risk_roles"]),
        "evidence_requirements": list(contract["evidence_requirements"]),
    }


def artifact_transition_allowed(
    *, phase: str, actor_profile: str, artifact: dict[str, Any],
    product: Optional[str] = None,
) -> dict[str, Any]:
    """Validate proposal/review/implementation gates against one frozen generation."""

    if not isinstance(artifact, dict):
        raise SpecialistRoutingBlocked("closure artifact evidence envelope is invalid")
    frozen = dict(artifact)
    generation = str(frozen.get("generation") or "").strip()
    digest = str(frozen.get("digest") or "").strip().lower()
    author = str(frozen.get("author_profile") or "").strip()
    if not generation or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SpecialistRoutingBlocked("exact artifact generation and digest are required")
    if frozen.get("frozen") is not True:
        raise SpecialistRoutingBlocked("artifact generation must be frozen")
    if not _valid_visible_url(frozen.get("visible_url")):
        raise SpecialistRoutingBlocked("visible clickable artifact is required")
    if not author:
        raise SpecialistRoutingBlocked("artifact author is required")
    phase = str(phase).strip().lower()
    actor = str(actor_profile).strip()
    if phase == "review":
        if actor == author:
            raise SpecialistRoutingBlocked("review must be independent of artifact author")
    elif phase in {"implementation", "shared_renderer_change"}:
        if frozen.get("accepted") is not True:
            raise SpecialistRoutingBlocked("accepted design generation is required")
    else:
        raise SpecialistRoutingBlocked("unsupported artifact transition")
    if phase == "shared_renderer_change" and str(product or "").strip().lower() == "webblotsen":
        concept_ids = frozen.get("concept_ids")
        if (
            frozen.get("cohort_complete") is not True
            or frozen.get("concept_count") != 3
            or not isinstance(concept_ids, list)
            or len(concept_ids) != 3
            or len({str(item).strip() for item in concept_ids}) != 3
            or frozen.get("viewports") != ["desktop", "tablet", "mobile"]
        ):
            raise SpecialistRoutingBlocked(
                "Webblotsen requires three distinct whole-cohort responsive concepts"
            )
    return frozen


def specialist_evidence_requirements(product: str, scope: str) -> list[str]:
    """Return the closed lifecycle evidence field set for one contract route."""

    normalized_product = _normalize_product(product)
    normalized_scope = _normalize_scope(scope)
    route = (normalized_product, normalized_scope)
    if route in {
        ("webblotsen", "three_customer_concepts"),
        ("webblotsen", "private_design_room"),
        ("firmalotsen", "private_design_room"),
    }:
        required = {"source_refs", "concepts", "risks", "recommendation"}
    elif route == ("webblotsen", "cohort_responsive_critique"):
        required = {
            "source_refs", "concept_ids", "cohort_complete", "viewports",
            "risks", "recommendation",
        }
    elif route == ("the-foundry", "public_art_direction"):
        required = {"source_refs", "coverage", "risks", "recommendation"}
    elif normalized_product == "hbbq" and route in _GROK_SCOPE_ROLES:
        required = {
            "source_refs", "fact_boundaries_ack", "protected_copy_unchanged",
            "risks", "recommendation",
        }
    elif route == ("mission-control", "owner_view_hierarchy_critique"):
        required = {
            "source_refs", "five_second_hierarchy", "cta_scope_only",
            "risks", "recommendation",
        }
    elif normalized_product in {"firmalotsen-personal", "personalkollen"} \
            and route in _GROK_SCOPE_ROLES:
        required = {"source_refs", "exclusions", "risks", "recommendation"}
    elif route == ("lundin", "irreversible_positioning_review"):
        required = {
            "source_refs", "irreversible_positioning_review", "risks", "recommendation",
        }
    else:
        raise SpecialistRoutingBlocked("no evidence matrix exists for specialist route")
    return sorted(required)


def build_brain_projection(
    *, task_id: str, run_id: int, provenance: dict[str, Any], commit_sha: str,
    artifact_generation: str, artifact_digest: str,
) -> dict[str, Any]:
    """Build an idempotent redacted post-commit projection payload."""

    commit_sha = str(commit_sha).strip().lower()
    artifact_digest = str(artifact_digest).strip().lower()
    if len(commit_sha) != 40 or any(ch not in "0123456789abcdef" for ch in commit_sha):
        raise SpecialistRoutingBlocked("post-commit SHA is required for Brain projection")
    if len(artifact_digest) != 64 or any(ch not in "0123456789abcdef" for ch in artifact_digest):
        raise SpecialistRoutingBlocked("artifact digest is required for Brain projection")
    receipt = provenance.get("probe_receipt") if isinstance(provenance, dict) else None
    safe_receipt = (
        {key: receipt.get(key) for key in sorted(_SAFE_RECEIPT_KEYS)}
        if isinstance(receipt, dict) else None
    )
    payload = {
        "schema_version": "GROK-SR-1.0",
        "task_id": str(task_id),
        "run_id": int(run_id),
        "commit_sha": commit_sha,
        "artifact_generation": str(artifact_generation),
        "artifact_digest": artifact_digest,
        "routing_provenance": {
            key: provenance.get(key) for key in (
                "profile", "provider", "model", "skills", "toolsets",
                "project_id", "required_roles", "required_capabilities",
            )
        },
        "probe_receipt": safe_receipt,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["projection_id"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def _validate_artifact_evidence(artifact: dict[str, Any]) -> None:
    required_artifact_keys = {
        "generation", "digest", "author_profile", "frozen", "visible_url",
        "viewports", "evidence",
    }
    if set(artifact) != required_artifact_keys:
        raise SpecialistRoutingBlocked("specialist artifact evidence has unsupported fields")
    generation = str(artifact.get("generation") or "").strip()
    digest = str(artifact.get("digest") or "").strip().lower()
    author = str(artifact.get("author_profile") or "").strip()
    viewports = artifact.get("viewports")
    if (
        not generation or not author or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
        or artifact.get("frozen") is not True
        or not _valid_visible_url(artifact.get("visible_url"))
        or not isinstance(viewports, list)
        or not {"desktop", "mobile"}.issubset(set(viewports))
    ):
        raise SpecialistRoutingBlocked("specialist closure artifact is incomplete")


def validate_specialist_evidence(
    *, product: str, scope: str, artifact: dict[str, Any],
) -> None:
    """Validate the closed product/scope evidence matrix for specialist output."""

    normalized_product = _normalize_product(product)
    normalized_scope = _normalize_scope(scope)
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        raise SpecialistRoutingBlocked("artifact evidence matrix is required")
    route = (normalized_product, normalized_scope)
    required = set(specialist_evidence_requirements(normalized_product, normalized_scope))
    if set(evidence) != required:
        raise SpecialistRoutingBlocked("artifact evidence matrix contains missing or unsupported fields")

    refs = evidence.get("source_refs")
    if not isinstance(refs, list) or not refs or any(
        not _valid_visible_url(ref) or "about:blank" in ref.lower()
        for ref in refs if isinstance(ref, str)
    ) or any(not isinstance(ref, str) for ref in refs):
        raise SpecialistRoutingBlocked("artifact source_refs require real http(s) evidence")
    risks = evidence.get("risks")
    if not isinstance(risks, list) or not risks or any(
        not isinstance(item, str) or not item.strip() for item in risks
    ):
        raise SpecialistRoutingBlocked("artifact risks are required")
    if not isinstance(evidence.get("recommendation"), str) or not evidence["recommendation"].strip():
        raise SpecialistRoutingBlocked("artifact recommendation is required")

    if "concepts" in required:
        concepts = evidence.get("concepts")
        if not isinstance(concepts, list) or len(concepts) != 3:
            raise SpecialistRoutingBlocked("exactly three source-backed concepts are required")
        concept_ids: list[str] = []
        for concept in concepts:
            if not isinstance(concept, dict) or set(concept) != {"id", "title", "source_ref"}:
                raise SpecialistRoutingBlocked("concept evidence has an invalid shape")
            concept_id = str(concept.get("id") or "").strip()
            title = str(concept.get("title") or "").strip()
            source_ref = str(concept.get("source_ref") or "")
            if not concept_id or not title or source_ref not in refs:
                raise SpecialistRoutingBlocked("concept evidence is not source-backed")
            concept_ids.append(concept_id)
        if len(set(concept_ids)) != 3:
            raise SpecialistRoutingBlocked("three structurally distinct concepts are required")
    if route == ("webblotsen", "cohort_responsive_critique"):
        concept_ids = evidence.get("concept_ids")
        if (
            not isinstance(concept_ids, list) or len(concept_ids) != 3
            or len(set(concept_ids)) != 3 or evidence.get("cohort_complete") is not True
            or evidence.get("viewports") != ["desktop", "tablet", "mobile"]
        ):
            raise SpecialistRoutingBlocked("complete three-concept responsive cohort evidence is required")
    if route == ("the-foundry", "public_art_direction") and evidence.get("coverage") != [
        "brand", "photo", "type", "motion", "sv_en_tone",
    ]:
        raise SpecialistRoutingBlocked("The Foundry evidence must cover the complete art-direction system")
    if normalized_product == "hbbq" and (
        evidence.get("fact_boundaries_ack") is not True
        or evidence.get("protected_copy_unchanged") is not True
    ):
        raise SpecialistRoutingBlocked("HBBQ factual/copy boundaries must be explicitly preserved")
    if route == ("mission-control", "owner_view_hierarchy_critique") and (
        evidence.get("five_second_hierarchy") is not True
        or evidence.get("cta_scope_only") is not True
    ):
        raise SpecialistRoutingBlocked("Mission Control hierarchy and CTA boundaries are required")
    if normalized_product in {"firmalotsen-personal", "personalkollen"} and evidence.get(
        "exclusions"
    ) != ["payroll", "law", "pii", "calculation"]:
        raise SpecialistRoutingBlocked("Personal evidence must preserve the forbidden ownership boundary")
    if route == ("lundin", "irreversible_positioning_review") and evidence.get(
        "irreversible_positioning_review"
    ) is not True:
        raise SpecialistRoutingBlocked("irreversible positioning review evidence is required")


def validate_specialist_closure(
    *, contract: dict[str, Any], provenance: dict[str, Any], metadata: dict[str, Any],
    task_id: str, run_id: int,
) -> dict[str, Any]:
    """Validate frozen artifact evidence and return its safe Brain projection."""

    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise SpecialistRoutingBlocked("specialist completion requires artifact evidence")
    _validate_artifact_evidence(artifact)
    validate_specialist_evidence(
        product=str(contract.get("product") or ""),
        scope=str(contract.get("scope") or ""),
        artifact=artifact,
    )
    generation = str(artifact.get("generation") or "").strip()
    digest = str(artifact.get("digest") or "").strip().lower()
    author = str(artifact.get("author_profile") or "").strip()
    viewports = list(artifact["viewports"])
    required_viewports = contract["artifact_policy"]["required_viewports"]
    if set(viewports) != set(required_viewports):
        raise SpecialistRoutingBlocked("specialist closure viewports are incomplete")
    route_profile = contract["selected_route"]["profile"]
    role = contract["required_roles"][0]
    if role == "creative_director" and author != route_profile:
        raise SpecialistRoutingBlocked("creative artifact author does not match routed writer")
    if role == "contrarian_reviewer" and author == route_profile:
        raise SpecialistRoutingBlocked("review closure must pin a non-self-authored artifact")
    commit_sha = str(metadata.get("commit_sha") or "").strip()
    return build_brain_projection(
        task_id=task_id,
        run_id=run_id,
        provenance=provenance,
        commit_sha=commit_sha,
        artifact_generation=generation,
        artifact_digest=digest,
    )


def grok_role_for_scope(product: str, scope: str) -> Optional[str]:
    """Return the only allowed Grok role for a governed product/scope pair."""

    key = (str(product).strip().lower(), str(scope).strip().lower())
    return _GROK_SCOPE_ROLES.get(key)


def _route_digest(profile: str, provider: str, model: str) -> str:
    canonical = json.dumps(
        {"profile": profile, "provider": provider, "model": model},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_probe_receipt(
    *,
    profile: str,
    provider: str,
    model: str,
    nonce: str,
    infer: Callable[..., str],
    now: Optional[int] = None,
    ttl_seconds: int = 300,
) -> dict[str, Any]:
    """Run one fallback-disabled marker inference and return sanitized evidence."""

    profile = str(profile).strip()
    provider = str(provider).strip().lower()
    model = str(model).strip()
    if not profile or provider not in GROK_PROVIDERS or model != GROK_MODEL:
        raise SpecialistRoutingBlocked("probe requires an exact Grok provider and model route")
    if not nonce or ttl_seconds <= 0:
        raise ValueError("probe nonce and positive ttl_seconds are required")
    verified_at = int(time.time()) if now is None else int(now)
    marker = "HERMES_GROK_PROBE_" + hashlib.sha256(nonce.encode()).hexdigest()[:20]
    success = False
    error_code: Optional[str] = None
    try:
        response = infer(
            profile=profile,
            provider=provider,
            model=model,
            expected_marker=marker,
            prompt=f"Return exactly this marker and nothing else: {marker}",
            fallback_disabled=True,
        )
        success = isinstance(response, str) and response.strip() == marker
        if not success:
            error_code = "marker_mismatch"
    except Exception as exc:  # fixed classification; no exception text is persisted
        error_code = f"inference_{type(exc).__name__.lower()}"
    return {
        "receipt_version": 1,
        "profile": profile,
        "provider": provider,
        "model": model,
        "route_digest": _route_digest(profile, provider, model),
        "marker_digest": hashlib.sha256(marker.encode()).hexdigest(),
        "verified_at": verified_at,
        "expires_at": verified_at + int(ttl_seconds),
        "fallback_disabled": True,
        "model_verified": success,
        "attribution_allowed": success,
        "error_code": error_code,
    }


def run_live_probe(
    *, profile: str, provider: str, model: str, nonce: str,
    now: Optional[int] = None, ttl_seconds: int = 300,
) -> dict[str, Any]:
    """Exercise the real configured route once with tools and fallbacks off."""

    from hermes_cli.profiles import get_active_profile_name

    if get_active_profile_name() != str(profile).strip():
        raise SpecialistRoutingBlocked(
            "probe profile does not match the active profile home"
        )

    def infer(**kwargs: Any) -> str:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from run_agent import AIAgent

        runtime = resolve_runtime_provider(
            requested=kwargs["provider"], target_model=kwargs["model"]
        )
        if runtime.get("provider") != kwargs["provider"]:
            raise SpecialistRoutingBlocked("runtime resolved a different provider")
        agent = AIAgent(
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            api_mode=runtime.get("api_mode"),
            model=kwargs["model"],
            fallback_model=[],
            enabled_toolsets=[],
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        return agent.chat(kwargs["prompt"])

    return create_probe_receipt(
        profile=profile, provider=provider, model=model, nonce=nonce,
        infer=infer, now=now, ttl_seconds=ttl_seconds,
    )


def build_grok_contract(
    *,
    role: str,
    capability: str,
    profile: str,
    provider: str,
    model: str,
    project_id: str,
    product: str,
    scope: str,
    probe_receipt: Optional[dict[str, Any]],
    skills: Optional[list[str]] = None,
    toolsets: Optional[list[str]] = None,
    risks: Optional[dict[str, bool]] = None,
) -> dict[str, Any]:
    """Build the task-local contract without mutating profile defaults."""

    normalized_project = str(project_id).strip()
    normalized_product = str(product).strip().lower()
    normalized_scope = str(scope).strip().lower()
    if probe_receipt is not None:
        if not isinstance(probe_receipt, dict) or set(probe_receipt) != _SAFE_RECEIPT_KEYS:
            raise SpecialistRoutingBlocked("probe receipt contains unsupported fields")
    required_viewports = ["desktop", "mobile"]
    if normalized_product == "webblotsen":
        required_viewports.insert(1, "tablet")
    role_value = str(role).strip()
    capability_value = str(capability).strip()
    selected = select_specialist_route(
        role=role_value,
        capability=capability_value,
        candidates=[{
            "profile": str(profile).strip(),
            "provider": str(provider).strip().lower(),
            "model": str(model).strip(),
            "roles": [role_value],
            "capabilities": [capability_value],
            "skills": [str(item) for item in (skills or []) if str(item).strip()],
            "toolsets": [str(item) for item in (toolsets or []) if str(item).strip()],
        }],
    )
    scope_canonical = json.dumps(
        {"project_id": normalized_project, "product": normalized_product,
         "scope": normalized_scope},
        sort_keys=True,
        separators=(",", ":"),
    )
    normalized_risks = dict(risks or {})
    return {
        "contract_version": "GROK-SR-1.0",
        "required_roles": [role_value],
        "required_capabilities": [capability_value],
        "project_id": normalized_project,
        "product": normalized_product,
        "scope": normalized_scope,
        "selected_route": {key: selected[key] for key in _ROUTE_KEYS},
        "probe_receipt": dict(probe_receipt) if probe_receipt is not None else None,
        "fallback_allowed": False,
        "artifact_policy": {
            "proposal_first": True,
            "visible_clickable_artifact": True,
            "required_viewports": required_viewports,
            "exactly_one_mutable_writer": True,
            "reviewer_read_only": True,
            "review_exact_frozen_generation": True,
            "cohort_before_shared_renderer_change": normalized_product == "webblotsen",
        },
        "artifact_target": {
            "scope_id": hashlib.sha256(scope_canonical.encode()).hexdigest(),
            "generation": "proposal",
            "access_mode": (
                "read_only" if role_value == "contrarian_reviewer" else "mutable_writer"
            ),
        },
        "content_boundaries": list(_PRODUCT_BOUNDARIES.get(normalized_product, ())),
        "prohibited_ownership": list(_PROHIBITED_OWNERSHIP),
        "risk_flags": normalized_risks,
        "required_risk_roles": required_risk_roles(normalized_risks),
        "evidence_requirements": specialist_evidence_requirements(
            normalized_product, normalized_scope
        ),
    }


def validate_grok_contract(
    contract: dict[str, Any], *, now: Optional[int] = None,
    require_fresh: bool = True, require_verified: bool = True,
) -> None:
    """Fail closed unless role, exact route, and live receipt all agree."""

    if not isinstance(contract, dict):
        raise SpecialistRoutingBlocked("specialist contract is required")
    if set(contract) != _CONTRACT_KEYS:
        raise SpecialistRoutingBlocked("specialist contract contains unsupported fields")
    if contract.get("contract_version") != "GROK-SR-1.0":
        raise SpecialistRoutingBlocked("unsupported specialist contract version")
    roles = contract.get("required_roles")
    capabilities = contract.get("required_capabilities")
    if not isinstance(roles, list) or not roles or not all(role in GROK_ROLES for role in roles):
        raise SpecialistRoutingBlocked("Grok role is prohibited or missing")
    if not isinstance(capabilities, list) or not any(str(item).strip() for item in capabilities):
        raise SpecialistRoutingBlocked("required capability is missing")
    product = str(contract.get("product") or "").strip().lower()
    scope = str(contract.get("scope") or "").strip().lower()
    allowed_role = grok_role_for_scope(product, scope)
    if allowed_role is None or roles != [allowed_role]:
        raise SpecialistRoutingBlocked("product scope is not allowed for the Grok role")
    normalized_capabilities = {str(item).strip().lower() for item in capabilities}
    if any(
        prohibited in capability
        for capability in normalized_capabilities
        for prohibited in _PROHIBITED_OWNERSHIP
    ):
        raise SpecialistRoutingBlocked("required capability grants prohibited Grok ownership")
    route = contract.get("selected_route")
    if not isinstance(route, dict):
        raise SpecialistRoutingBlocked("selected route is missing")
    if set(route) != _ROUTE_KEYS:
        raise SpecialistRoutingBlocked("selected route contains unsupported fields")
    profile = str(route.get("profile") or "").strip()
    provider = str(route.get("provider") or "").strip().lower()
    model = str(route.get("model") or "").strip()
    if not profile:
        raise SpecialistRoutingBlocked("selected profile is missing")
    if not provider:
        raise SpecialistRoutingBlocked("selected provider is missing")
    if provider not in GROK_PROVIDERS:
        raise SpecialistRoutingBlocked("selected provider is not an exact Grok route")
    if not model:
        raise SpecialistRoutingBlocked("selected model is missing")
    if model != GROK_MODEL:
        raise SpecialistRoutingBlocked("selected model is not the exact Grok model")
    if not isinstance(route.get("skills"), list) or not isinstance(route.get("toolsets"), list):
        raise SpecialistRoutingBlocked("selected route skills/toolsets are invalid")
    if roles == ["contrarian_reviewer"] and not set(route["toolsets"]).issubset(
        _REVIEWER_TOOLSETS
    ):
        raise SpecialistRoutingBlocked("contrarian reviewer route must be genuinely read-only")
    if contract.get("fallback_allowed") is not False:
        raise SpecialistRoutingBlocked("Grok fallback must be disabled")
    risk_flags = contract.get("risk_flags")
    if (
        not isinstance(risk_flags, dict)
        or not set(risk_flags).issubset({"security", "data_integration", "release"})
        or any(value is not True for value in risk_flags.values())
        or contract.get("required_risk_roles") != required_risk_roles(risk_flags)
    ):
        raise SpecialistRoutingBlocked("risk specialist activation is invalid")
    evidence_requirements = contract.get("evidence_requirements")
    if (
        not isinstance(evidence_requirements, list)
        or evidence_requirements != specialist_evidence_requirements(product, scope)
    ):
        raise SpecialistRoutingBlocked("specialist evidence requirements are invalid")
    policy = contract.get("artifact_policy")
    if not isinstance(policy, dict) or set(policy) != _POLICY_KEYS:
        raise SpecialistRoutingBlocked("artifact policy contains unsupported fields")
    if any(
        policy.get(key) is not True for key in (
            "proposal_first", "visible_clickable_artifact",
            "exactly_one_mutable_writer", "reviewer_read_only",
            "review_exact_frozen_generation",
        )
    ):
        raise SpecialistRoutingBlocked("proposal and frozen-artifact policy is incomplete")
    target = contract.get("artifact_target")
    if not isinstance(target, dict) or set(target) != _TARGET_KEYS:
        raise SpecialistRoutingBlocked("artifact target contains unsupported fields")
    expected_mode = "read_only" if roles == ["contrarian_reviewer"] else "mutable_writer"
    scope_id = str(target.get("scope_id") or "")
    if (
        len(scope_id) != 64
        or any(ch not in "0123456789abcdef" for ch in scope_id)
        or target.get("generation") != "proposal"
        or target.get("access_mode") != expected_mode
    ):
        raise SpecialistRoutingBlocked("artifact target identity is invalid")
    receipt = contract.get("probe_receipt")
    if not require_verified and receipt is None:
        return
    if not isinstance(receipt, dict):
        raise SpecialistRoutingBlocked("successful live probe receipt is required")
    if set(receipt) != _SAFE_RECEIPT_KEYS:
        raise SpecialistRoutingBlocked("probe receipt contains unsupported fields")
    if (
        receipt.get("receipt_version") != 1
        or receipt.get("profile") != profile
        or receipt.get("provider") != provider
        or receipt.get("model") != model
    ):
        raise SpecialistRoutingBlocked("probe receipt does not match selected route")
    if not receipt.get("model_verified") or not receipt.get("attribution_allowed"):
        raise SpecialistRoutingBlocked("live probe did not verify the model")
    if receipt.get("fallback_disabled") is not True:
        raise SpecialistRoutingBlocked("probe fallback was not disabled")
    if receipt.get("route_digest") != _route_digest(profile, provider, model):
        raise SpecialistRoutingBlocked("probe route does not match selected route")
    if require_fresh:
        current = int(time.time()) if now is None else int(now)
        if current >= int(receipt.get("expires_at") or 0):
            raise SpecialistRoutingBlocked("live probe receipt expired")
