"""Thin LLM client wrapper with a cost ceiling and retry-once policy.

Routes to either Anthropic or OpenRouter depending on the model name prefix.
The wrapper is intentionally tiny: it builds the request, enforces input/
output token caps, computes an estimated cost BEFORE the call, and refuses
the call if the estimate exceeds the ceiling.

Tests mock this module via ``llm_client.propose`` directly; no live network
call should occur in CI.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:  # pragma: no cover - exercised only in environments without httpx
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

# Canonical Claude model IDs (institutional-rigor §4: delegate, never re-encode).
# Imported lazily inside helpers so tests that mock at this module's boundary
# do not pull the full trading_app package into every test scope.


def _claude_models() -> tuple[str, str]:
    """Return ``(reasoning_model_id, structured_model_id)`` from the canonical
    source ``trading_app.ai.claude_client``.

    Bound at call time so unit tests can monkeypatch the import target without
    needing to reload this module.
    """
    from trading_app.ai.claude_client import (
        CLAUDE_REASONING_MODEL,
        CLAUDE_STRUCTURED_MODEL,
    )

    return CLAUDE_REASONING_MODEL, CLAUDE_STRUCTURED_MODEL


def default_reasoning_model() -> str:
    """Canonical Claude reasoning model (Opus tier). See claude_client."""
    return _claude_models()[0]


@dataclass(frozen=True)
class ProposerResult:
    yaml_text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_s: float


class CostCeilingExceeded(Exception):
    """Raised before the network call when estimated cost > ceiling."""


class LLMRefusalToGround(Exception):
    """Raised when the model returns the refusal sentinel."""


class LLMRequestError(Exception):
    """Raised on transport / API failures after the retry policy is exhausted."""


# Rough pricing tiers (USD per 1M tokens). Keyed by *tier*, not by hardcoded
# model-ID string, so the canonical Claude model IDs in
# ``trading_app.ai.claude_client`` remain the single source of truth.
# Cost ceiling is the safety; these constants are pre-flight estimates only.
_PRICING_TIER_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "opus": (15.00, 75.00),
    "sonnet": (3.00, 15.00),
    "haiku": (1.00, 5.00),
    "unknown": (20.00, 80.00),  # conservative fallback for non-Anthropic routes
}


def _tier_for_model(model: str) -> str:
    """Map a model id (or OpenRouter slug) to its pricing tier.

    Anthropic canonical ids come from ``claude_client``; OpenRouter slugs are
    detected by substring. Any unknown model falls back to ``unknown`` so cost
    estimates remain conservative.
    """
    try:
        reasoning, structured = _claude_models()
    except Exception:  # pragma: no cover - defensive; canonical import failure
        reasoning, structured = "", ""
    if model in (reasoning,) or "opus" in model.lower():
        return "opus"
    if model in (structured,) or "sonnet" in model.lower():
        return "sonnet"
    if "haiku" in model.lower():
        return "haiku"
    return "unknown"


def estimate_cost_usd(model: str, in_toks: int, out_toks: int) -> float:
    tier = _tier_for_model(model)
    in_rate, out_rate = _PRICING_TIER_USD_PER_MTOK[tier]
    return (in_toks / 1_000_000) * in_rate + (out_toks / 1_000_000) * out_rate


def _rough_token_count(text: str) -> int:
    """Token count proxy: 1 token ≈ 4 chars. Good enough for pre-flight."""
    return max(1, len(text) // 4)


def propose(
    *,
    system_prompt: str,
    fewshot: str,
    corpus_summary: str,
    adjacency_context: str,
    user_instruction: str,
    max_input_tokens: int = 30_000,
    max_output_tokens: int = 4_000,
    cost_ceiling_usd: float = 0.50,
    timeout_s: float = 120.0,
    model: str | None = None,
    api_key: str | None = None,
) -> ProposerResult:
    """Single LLM call. Returns raw response text.

    Cost is estimated BEFORE the call from rough token counts and the static
    pricing table. If the estimate exceeds ``cost_ceiling_usd`` we raise
    ``CostCeilingExceeded`` without spending anything.
    """
    if httpx is None:  # pragma: no cover
        raise LLMRequestError("httpx not installed; cannot call LLM. Install via uv sync.")

    if model is None:
        model = default_reasoning_model()

    user_msg = (
        f"FEWSHOT EXAMPLES:\n{fewshot}\n\n"
        f"LITERATURE CORPUS:\n{corpus_summary}\n\n"
        f"ADJACENCY CONTEXT:\n{adjacency_context}\n\n"
        f"USER INSTRUCTION:\n{user_instruction}\n"
    )
    in_toks = _rough_token_count(system_prompt) + _rough_token_count(user_msg)
    if in_toks > max_input_tokens:
        raise CostCeilingExceeded(f"Estimated input tokens {in_toks} > max_input_tokens {max_input_tokens}")
    estimated = estimate_cost_usd(model, in_toks, max_output_tokens)
    if estimated > cost_ceiling_usd:
        raise CostCeilingExceeded(
            f"Estimated cost ${estimated:.4f} > ceiling ${cost_ceiling_usd:.4f} "
            f"(input ~{in_toks} toks, output cap {max_output_tokens} toks, model {model})"
        )

    use_openrouter = "/" in model
    api_key = api_key or (
        os.environ.get("OPENROUTER_API_KEY") if use_openrouter else os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        raise LLMRequestError(
            f"No API key in env for model {model}. Set "
            f"{'OPENROUTER_API_KEY' if use_openrouter else 'ANTHROPIC_API_KEY'}."
        )

    if use_openrouter:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
    else:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_output_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }

    start = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 500 and attempt == 0:
                last_exc = LLMRequestError(f"Server {resp.status_code}: {resp.text[:200]}")
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.HTTPError as exc:
            last_exc = LLMRequestError(str(exc))
            if attempt == 0:
                continue
            raise last_exc from exc
    else:  # pragma: no cover
        raise last_exc if last_exc else LLMRequestError("Unknown LLM failure")

    elapsed = time.monotonic() - start

    if use_openrouter:
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_used = usage.get("prompt_tokens", in_toks)
        out_used = usage.get("completion_tokens", _rough_token_count(text))
    else:
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        in_used = usage.get("input_tokens", in_toks)
        out_used = usage.get("output_tokens", _rough_token_count(text))

    if text.strip().startswith("REFUSE: no_literature_match"):
        raise LLMRefusalToGround(text.strip())

    final_cost = estimate_cost_usd(model, in_used, out_used)
    return ProposerResult(
        yaml_text=text,
        model=model,
        input_tokens=in_used,
        output_tokens=out_used,
        cost_usd=final_cost,
        elapsed_s=elapsed,
    )


# Used by tests to bypass the live HTTP path. The CLI checks for this module-
# level attribute when --mock-llm is set.
_MOCK_RESPONSE: ProposerResult | Exception | None = None


def set_mock_response(value: ProposerResult | Exception | None) -> None:
    """Test helper: set or clear the mocked response."""
    global _MOCK_RESPONSE
    _MOCK_RESPONSE = value


def propose_with_mock_support(**kwargs) -> ProposerResult:
    """Wrapper used by the CLI; honours ``set_mock_response`` for tests."""
    if _MOCK_RESPONSE is not None:
        mock = _MOCK_RESPONSE
        if isinstance(mock, Exception):
            raise mock
        return mock
    return propose(**kwargs)


__all__ = [
    "CostCeilingExceeded",
    "LLMRefusalToGround",
    "LLMRequestError",
    "ProposerResult",
    "estimate_cost_usd",
    "propose",
    "propose_with_mock_support",
    "set_mock_response",
]
