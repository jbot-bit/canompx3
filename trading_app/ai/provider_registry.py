"""Canonical AI provider/profile registry for repo-native AI surfaces.

This module centralizes provider settings, model selection, and OpenRouter
request-shaping defaults so new AI entrypoints do not scatter hardcoded model
IDs, base URLs, or routing behavior across the repo.

OpenRouter defaults are intentionally conservative:
- research/planning only
- read-only host tools only
- no hidden provider fallbacks
- require parameter support on the selected model
- deny provider-side data collection where OpenRouter supports it
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from trading_app.ai.claude_client import CLAUDE_REASONING_MODEL, CLAUDE_STRUCTURED_MODEL

ProviderName = Literal["anthropic", "openrouter"]
ResponseMode = Literal["free_text", "json_schema"]
RuntimeClass = Literal["read_only_single_turn", "read_only_tool_loop", "interactive_editor"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]


def _csv_tuple(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _profile_env_prefix(profile_id: str) -> str:
    return "CANOMPX3_AI_" + "".join(ch if ch.isalnum() else "_" for ch in profile_id.upper())


@dataclass(frozen=True)
class ProviderRouting:
    """OpenRouter provider-routing options grounded in official docs."""

    order: tuple[str, ...] = ()
    allow_fallbacks: bool = False
    require_parameters: bool = True
    data_collection: Literal["allow", "deny"] | None = "deny"
    zdr: bool | None = None

    def to_openrouter_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
        }
        if self.order:
            payload["order"] = list(self.order)
        if self.data_collection is not None:
            payload["data_collection"] = self.data_collection
        if self.zdr is not None:
            payload["zdr"] = self.zdr
        return payload


@dataclass(frozen=True)
class AIProfile:
    """Repo-level AI profile contract."""

    profile_id: str
    provider: ProviderName
    use_case: str
    model: str | None
    api_key_env: str
    base_url: str | None = None
    reasoning_enabled: bool = False
    reasoning_effort: ReasoningEffort | None = None
    response_mode: ResponseMode = "free_text"
    runtime_class: RuntimeClass = "read_only_single_turn"
    host_tools: tuple[str, ...] = ()
    context_views: tuple[str, ...] = ()
    router: ProviderRouting | None = None
    mutation_allowed: bool = False
    live_control_allowed: bool = False
    required_env: tuple[str, ...] = field(default_factory=tuple)
    required_parameters: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def env_prefix(self) -> str:
        return _profile_env_prefix(self.profile_id)

    def resolved(self) -> AIProfile:
        prefix = self.env_prefix()
        model = os.environ.get(f"{prefix}_MODEL", self.model)
        base_url = os.environ.get(f"{prefix}_BASE_URL", self.base_url)

        router = self.router
        if router is not None:
            router = ProviderRouting(
                order=_csv_tuple(os.environ.get(f"{prefix}_PROVIDER_ORDER"), router.order),
                allow_fallbacks=router.allow_fallbacks,
                require_parameters=router.require_parameters,
                data_collection=router.data_collection,
                zdr=router.zdr,
            )

        return AIProfile(
            profile_id=self.profile_id,
            provider=self.provider,
            use_case=self.use_case,
            model=model,
            api_key_env=self.api_key_env,
            base_url=base_url,
            reasoning_enabled=self.reasoning_enabled,
            reasoning_effort=self.reasoning_effort,
            response_mode=self.response_mode,
            runtime_class=self.runtime_class,
            host_tools=self.host_tools,
            context_views=self.context_views,
            router=router,
            mutation_allowed=self.mutation_allowed,
            live_control_allowed=self.live_control_allowed,
            required_env=self.required_env,
            required_parameters=self.required_parameters,
            notes=self.notes,
        )

    def missing_env(self) -> list[str]:
        required = {self.api_key_env, *self.required_env}
        return sorted(var for var in required if var and not os.environ.get(var))

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.provider == "openrouter" and not self.base_url:
            errors.append("openrouter profile missing base_url")
        if self.provider == "openrouter" and not self.model:
            errors.append(f"model not configured; set {self.env_prefix()}_MODEL for this profile")
        missing = self.missing_env()
        if missing:
            errors.append("missing required env: " + ", ".join(missing))
        # Mutation/live-control authority is rejected for read-only research profiles.
        # The interactive_editor runtime class is the dedicated escape hatch for the
        # repo-native coding-agent profile, which mutates source files by design.
        if self.runtime_class != "interactive_editor":
            if self.mutation_allowed:
                errors.append("mutation authority is not allowed for repo research profiles")
            if self.live_control_allowed:
                errors.append("live-control authority is not allowed for repo research profiles")
        if self.router is not None and self.router.allow_fallbacks:
            errors.append("openrouter research profiles must not allow provider fallbacks")
        if self.router is not None and not self.router.require_parameters:
            errors.append("openrouter research profiles must require supported parameters")
        if self.router is not None and self.router.data_collection != "deny":
            errors.append("openrouter research profiles must deny provider-side data collection")
        return errors

    def assert_ready(self) -> AIProfile:
        errors = self.validation_errors()
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_configured"] = bool(self.model)
        payload["validation_errors"] = self.validation_errors()
        if self.router is not None:
            payload["router"] = self.router.to_openrouter_dict()
        return payload


PROFILE_REGISTRY: dict[str, AIProfile] = {
    "claude_structured": AIProfile(
        profile_id="claude_structured",
        provider="anthropic",
        use_case="structured intent extraction for embedded repo queries",
        model=CLAUDE_STRUCTURED_MODEL,
        api_key_env="ANTHROPIC_API_KEY",
        required_env=("ANTHROPIC_API_KEY",),
        notes="Canonical structured-output pass for QueryAgent.",
    ),
    "claude_reasoning": AIProfile(
        profile_id="claude_reasoning",
        provider="anthropic",
        use_case="reasoning and interpretation for embedded repo queries",
        model=CLAUDE_REASONING_MODEL,
        api_key_env="ANTHROPIC_API_KEY",
        reasoning_enabled=True,
        reasoning_effort="high",
        required_env=("ANTHROPIC_API_KEY",),
        notes="Canonical reasoning pass for QueryAgent.",
    ),
    "deepseek_planning": AIProfile(
        profile_id="deepseek_planning",
        provider="openrouter",
        use_case="repo planning, architecture review, and grounded research synthesis",
        model=None,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        reasoning_enabled=True,
        reasoning_effort="high",
        runtime_class="read_only_tool_loop",
        host_tools=("get_context_view", "get_canonical_context", "query_trading_db"),
        context_views=("research", "verification"),
        router=ProviderRouting(
            order=("deepseek",), allow_fallbacks=False, require_parameters=True, data_collection="deny"
        ),
        required_env=("OPENROUTER_API_KEY",),
        required_parameters=("reasoning", "tools"),
        notes=(
            "OpenRouter-backed planning profile. Configure the exact OpenRouter "
            "model via CANOMPX3_AI_DEEPSEEK_PLANNING_MODEL."
        ),
    ),
    "deepseek_research_long_context": AIProfile(
        profile_id="deepseek_research_long_context",
        provider="openrouter",
        use_case="long-context repo research, literature-grounded synthesis, and result interpretation",
        model=None,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        reasoning_enabled=True,
        reasoning_effort="high",
        runtime_class="read_only_tool_loop",
        host_tools=("get_context_view", "get_canonical_context", "query_trading_db"),
        context_views=("research", "recent_performance"),
        router=ProviderRouting(
            order=("deepseek",), allow_fallbacks=False, require_parameters=True, data_collection="deny"
        ),
        required_env=("OPENROUTER_API_KEY",),
        required_parameters=("reasoning", "tools"),
        notes=(
            "OpenRouter-backed long-context DeepSeek profile. Configure the exact OpenRouter "
            "model via CANOMPX3_AI_DEEPSEEK_RESEARCH_LONG_CONTEXT_MODEL."
        ),
    ),
    "deepseek_structured_extraction": AIProfile(
        profile_id="deepseek_structured_extraction",
        provider="openrouter",
        use_case="schema-bound extraction and structured research outputs",
        model=None,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        response_mode="json_schema",
        runtime_class="read_only_single_turn",
        context_views=("research",),
        router=ProviderRouting(
            order=("deepseek",), allow_fallbacks=False, require_parameters=True, data_collection="deny"
        ),
        required_env=("OPENROUTER_API_KEY",),
        required_parameters=("response_format", "structured_outputs"),
        notes=(
            "OpenRouter-backed extraction profile. Configure the exact OpenRouter "
            "model via CANOMPX3_AI_DEEPSEEK_STRUCTURED_EXTRACTION_MODEL."
        ),
    ),
    "deepseek_coding": AIProfile(
        profile_id="deepseek_coding",
        provider="openrouter",
        use_case="repo-native coding-agent edits with claude-side review gating",
        model=None,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        runtime_class="interactive_editor",
        router=ProviderRouting(
            order=("deepseek",),
            allow_fallbacks=False,
            require_parameters=True,
            data_collection="deny",
            zdr=True,
        ),
        mutation_allowed=True,
        required_env=("OPENROUTER_API_KEY",),
        notes=(
            "OpenRouter-backed coding-agent profile (DeepSeek Coding Agent v4). "
            "model is None until Phase 2.5 bake-off picks the winner; assert_ready "
            "fails-closed by design until then. Edits land via aider; every commit "
            "is reviewed by claude-side gate (Phase 3) before push. Configure the "
            "selected model via CANOMPX3_AI_DEEPSEEK_CODING_MODEL."
        ),
    ),
}


def list_profiles() -> list[str]:
    return sorted(PROFILE_REGISTRY)


def list_openrouter_research_profiles() -> list[str]:
    return sorted(
        profile_id
        for profile_id, profile in PROFILE_REGISTRY.items()
        if profile.provider == "openrouter" and profile.runtime_class != "interactive_editor"
    )


def get_profile(profile_id: str) -> AIProfile:
    try:
        profile = PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown AI profile: {profile_id}") from exc
    return profile.resolved()


def get_model_name(profile_id: str) -> str | None:
    return get_profile(profile_id).model


def assert_openrouter_research_profile(profile_id: str) -> AIProfile:
    profile = get_profile(profile_id)
    if profile.provider != "openrouter":
        raise ValueError(f"{profile_id} is not an OpenRouter research profile")
    if profile.runtime_class == "interactive_editor":
        raise ValueError(
            f"{profile_id} is not an OpenRouter research profile "
            "(runtime_class=interactive_editor; use the dedicated coding-agent launcher)"
        )
    return profile.assert_ready()


def get_openrouter_request_defaults(profile_id: str) -> dict[str, Any]:
    profile = assert_openrouter_research_profile(profile_id)
    payload: dict[str, Any] = {}
    if profile.router is not None:
        payload["provider"] = profile.router.to_openrouter_dict()
    if profile.reasoning_enabled:
        reasoning: dict[str, Any] = {"enabled": True}
        if profile.reasoning_effort is not None:
            reasoning["effort"] = profile.reasoning_effort
        payload["reasoning"] = reasoning
    return payload
