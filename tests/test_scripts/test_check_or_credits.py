"""Tests for `scripts/tools/check_or_credits.py`.

Live HTTP calls to OpenRouter are NOT exercised in tests. Mock fixtures
cover the WARN + non-WARN paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "tools" / "check_or_credits.py"


def _run(args: list[str], env_overrides: dict[str, str | None] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key, val in (env_overrides or {}).items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


class TestCheckOrCredits:
    def test_mock_normal_emits_balance_no_warn(self) -> None:
        result = _run(["--mock"])
        assert result.returncode == 0
        assert "remaining=" in result.stdout
        # The mock fixture has $87.66 remaining > $5 threshold → no WARN.
        assert "WARN" not in result.stderr

    def test_mock_low_emits_warn(self) -> None:
        result = _run(["--mock-low"])
        assert result.returncode == 0
        assert "WARN" in result.stderr
        assert "below threshold" in result.stderr

    def test_threshold_override_can_force_warn(self) -> None:
        # Normal mock has $87.66 remaining; threshold of $200 forces WARN.
        result = _run(["--mock", "--threshold", "200"])
        assert result.returncode == 0
        assert "WARN" in result.stderr

    def test_no_api_key_is_advisory(self) -> None:
        result = _run([], env_overrides={"OPENROUTER_API_KEY": None})
        assert result.returncode == 0
        assert "WARN" in result.stderr
        assert "OPENROUTER_API_KEY" in result.stderr

    def test_emit_balance_unit(self) -> None:
        # Direct unit test on the formatter — independent of subprocess
        # boundary, robust under coverage tooling.
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
        try:
            import check_or_credits as cor  # type: ignore
        finally:
            sys.path.pop(0)
        # Smoke: function executes without raising on valid payload.
        cor._emit_balance({"data": {"label": "k", "usage": 1.0, "limit": 100.0}}, 5.0)
        # And on missing fields → WARN path.
        cor._emit_balance({"data": {"label": "k"}}, 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
