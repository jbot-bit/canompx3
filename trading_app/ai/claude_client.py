"""Canonical Anthropic client module for embedded AI calls.

Single source of truth for model pins and client construction. Every
`trading_app/ai/` and `scripts/tools/` module that calls the Anthropic API
MUST import from here — never hardcode a `claude-*` model string, never
instantiate `anthropic.Anthropic()` directly.

Enforced by `pipeline/check_drift.py` (added in Stage 4 of
claude-api-modernization).

Model strategy (see docs/runtime/stages/claude-api-modernization.md):
  - CLAUDE_STRUCTURED_MODEL (Sonnet 4.6) — deterministic structured-output
    extraction (Pass 1 query intent, digest JSON schemas). Cheaper; keeps
    temperature semantics available via adaptive thinking.
  - CLAUDE_REASONING_MODEL (Opus 4.7) — interpretation, coaching, research
    reasoning. Adaptive thinking; 1M context at standard pricing.
"""

from __future__ import annotations

import os
from typing import Any

CLAUDE_STRUCTURED_MODEL = "claude-sonnet-4-6"
CLAUDE_REASONING_MODEL = "claude-opus-4-7"


def _load_anthropic() -> Any:
    """Import the optional Anthropic SDK only when the AI surface is used."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic SDK not installed. Install the optional 'anthropic' package "
            "to use trading_app.ai query surfaces."
        ) from exc
    return anthropic


def get_client(api_key: str | None = None) -> Any:
    """Return a configured `anthropic.Anthropic` client.

    Args:
        api_key: Explicit API key. Overrides `ANTHROPIC_API_KEY` when provided.

    Raises:
        ValueError: when no key is available from either argument or environment.
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise ValueError("ANTHROPIC_API_KEY required. Pass api_key or set env var.")
    anthropic = _load_anthropic()
    return anthropic.Anthropic(api_key=resolved_key)
