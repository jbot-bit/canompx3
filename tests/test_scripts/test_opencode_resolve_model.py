"""Tests for `scripts/tools/opencode_resolve_model.py`.

The resolver is a thin shell over `trading_app.ai.provider_registry.get_profile`,
so the tests verify the contract that the OpenCode launcher relies on:

- exit 0 + model on stdout when `deepseek_coding` profile is fully configured;
- exit 1 + diagnostic on stderr otherwise (env unset, missing key, etc.);
- stdout is empty on the failure path so the PowerShell caller can use raw
  stdout as the model ID without parsing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOLVER = PROJECT_ROOT / "scripts" / "tools" / "opencode_resolve_model.py"


def _run(env_overrides: dict[str, str | None]) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    for key, val in env_overrides.items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    return subprocess.run(
        [sys.executable, str(RESOLVER)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


class TestOpencodeResolveModel:
    def test_exit_one_when_model_env_unset(self) -> None:
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": None,
                "OPENROUTER_API_KEY": "sk-or-test",
            }
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "CANOMPX3_AI_DEEPSEEK_CODING_MODEL" in result.stderr

    def test_exit_one_when_api_key_unset(self) -> None:
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "openrouter/deepseek-chat-v3.1",
                "OPENROUTER_API_KEY": None,
            }
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "OPENROUTER_API_KEY" in result.stderr

    def test_exit_zero_with_model_on_stdout_when_env_set(self) -> None:
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "openrouter/deepseek-chat-v3.1",
                "OPENROUTER_API_KEY": "sk-or-test",
            }
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert result.stdout.strip() == "openrouter/deepseek-chat-v3.1"
        assert result.stderr == ""

    def test_stdout_is_clean_for_pwsh_consumption(self) -> None:
        # The PowerShell launcher uses `& python <script>` and reads stdout
        # raw (with .Trim()). Any extra prefix/suffix would break model
        # parsing. Verify the success-path stdout is exactly the model line.
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "openrouter/some-vendor/some-model",
                "OPENROUTER_API_KEY": "sk-or-test",
            }
        )
        assert result.returncode == 0
        assert result.stdout.rstrip("\r\n") == "openrouter/some-vendor/some-model"

    def test_whitespace_only_model_rejected(self) -> None:
        # Canonical-layer fix: AIProfile.validation_errors() must reject
        # whitespace-only model strings (e.g. " ") with the same diagnostic
        # as the unset case. Without this, env=" " produces a "configured"
        # but useless model and the launcher silently picks junk.
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "   ",
                "OPENROUTER_API_KEY": "sk-or-test",
            }
        )
        assert result.returncode == 1
        assert result.stdout == ""
        assert "model not configured" in result.stderr
        assert "DEEPSEEK_CODING_MODEL" in result.stderr

    def test_resolver_does_not_validate_keys_against_openrouter_live(self) -> None:
        # Resolver is offline by contract — it only checks env presence and
        # router config. Setting a clearly-fake key still resolves OK.
        result = _run(
            {
                "CANOMPX3_AI_DEEPSEEK_CODING_MODEL": "openrouter/deepseek-chat-v3.1",
                "OPENROUTER_API_KEY": "sk-or-not-real-key-but-set",
            }
        )
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
