"""Resolve the canonical deepseek_coding model ID for the OpenCode launcher.

Single source of truth: ``trading_app.ai.provider_registry.PROFILE_REGISTRY``.

Contract:
- Exit 0 with the model ID on stdout when the ``deepseek_coding`` profile is
  configured + valid (i.e. ``CANOMPX3_AI_DEEPSEEK_CODING_MODEL`` set, OpenRouter
  API key present, router config intact).
- Exit 1 with diagnostic messages on stderr otherwise. Stdout stays empty so
  the PowerShell caller can use raw stdout as the model ID without parsing.

Per institutional-rigor rule #4 (delegate, never re-encode), this script
delegates to ``get_profile().resolved().validation_errors()`` rather than
re-implementing model selection. If the registry contract changes, this
script does not need to.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading_app.ai.provider_registry import get_profile  # noqa: E402


def main() -> int:
    profile = get_profile("deepseek_coding")
    errors = profile.validation_errors()
    if errors:
        for err in errors:
            print(f"opencode_resolve_model: {err}", file=sys.stderr)
        return 1
    if not profile.model:
        print("opencode_resolve_model: profile resolved with empty model", file=sys.stderr)
        return 1
    print(profile.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
