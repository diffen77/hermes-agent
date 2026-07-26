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


def artifact_transition_allowed(
    *, phase: str, actor_profile: str, artifact: dict[str, Any],
    product: Optional[str] = None,
) -> dict[str, Any]:
    """Validate proposal/review/implementation gates against one frozen generation."""

    if not isinstance(artifact, dict):
        raise SpecialistRoutingBlocked("artifact evidence is required")
    frozen = dict(artifact)
    generation = str(frozen.get("generation") or "").strip()
    digest = str(frozen.get("digest") or "").strip().lower()
    author = str(frozen.get("author_profile") or "").strip()
    if not generation or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise SpecialistRoutingBlocked("exact artifact generation and digest are required")
    if frozen.get("frozen") is not True:
        raise SpecialistRoutingBlocked("artifact generation must be frozen")
    if not str(frozen.get("visible_url") or "").strip():
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


def validate_specialist_closure(
    *, contract: dict[str, Any], provenance: dict[str, Any], metadata: dict[str, Any],
    task_id: str, run_id: int,
) -> dict[str, Any]:
    """Validate frozen artifact evidence and return its safe Brain projection."""

    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise SpecialistRoutingBlocked("specialist closure requires artifact evidence")
    required_artifact_keys = {
        "generation", "digest", "author_profile", "frozen", "visible_url", "viewports",
    }
    if set(artifact) != required_artifact_keys:
        raise SpecialistRoutingBlocked("specialist artifact evidence has unsupported fields")
    generation = str(artifact.get("generation") or "").strip()
    digest = str(artifact.get("digest") or "").strip().lower()
    author = str(artifact.get("author_profile") or "").strip()
    viewports = artifact.get("viewports")
    if (
        not generation
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
        or artifact.get("frozen") is not True
        or not str(artifact.get("visible_url") or "").strip()
        or not isinstance(viewports, list)
    ):
        raise SpecialistRoutingBlocked("specialist closure artifact is incomplete")
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
    return {
        "contract_version": "GROK-SR-1.0",
        "required_roles": [str(role).strip()],
        "required_capabilities": [str(capability).strip()],
        "project_id": normalized_project,
        "product": normalized_product,
        "scope": normalized_scope,
        "selected_route": {
            "profile": str(profile).strip(),
            "provider": str(provider).strip().lower(),
            "model": str(model).strip(),
            "skills": [str(item) for item in (skills or []) if str(item).strip()],
            "toolsets": [str(item) for item in (toolsets or []) if str(item).strip()],
        },
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
        "content_boundaries": list(_PRODUCT_BOUNDARIES.get(normalized_product, ())),
        "prohibited_ownership": list(_PROHIBITED_OWNERSHIP),
    }


def validate_grok_contract(
    contract: dict[str, Any], *, now: Optional[int] = None,
    require_fresh: bool = True, require_verified: bool = True,
) -> None:
    """Fail closed unless role, exact route, and live receipt all agree."""

    if not isinstance(contract, dict):
        raise SpecialistRoutingBlocked("specialist contract is required")
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
    if contract.get("fallback_allowed") is not False:
        raise SpecialistRoutingBlocked("Grok fallback must be disabled")
    policy = contract.get("artifact_policy")
    if not isinstance(policy, dict) or any(
        policy.get(key) is not True for key in (
            "proposal_first", "visible_clickable_artifact",
            "exactly_one_mutable_writer", "reviewer_read_only",
            "review_exact_frozen_generation",
        )
    ):
        raise SpecialistRoutingBlocked("proposal and frozen-artifact policy is incomplete")
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
