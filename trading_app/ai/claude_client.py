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

import anthropic

CLAUDE_STRUCTURED_MODEL = "claude-sonnet-4-6"
CLAUDE_REASONING_MODEL = "claude-opus-4-7"


def get_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Return a configured `anthropic.Anthropic` client.

    Args:
        api_key: Explicit API key. Overrides `ANTHROPIC_API_KEY` when provided.

    Raises:
        ValueError: when no key is available from either argument or environment.
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise ValueError(
            "ANTHROPIC_API_KEY required. Pass api_key or set env var."
        )
    return anthropic.Anthropic(api_key=resolved_key)
