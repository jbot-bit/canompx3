"""Tests for `scripts/tools/check_or_credits.py`.

Live HTTP calls to OpenRouter are NOT exercised in tests. Mock fixtures
cover the WARN + non-WARN paths.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "tools" / "check_or_credits.py"


def _load_module():
    """Load `check_or_credits.py` as a module without running its CLI.

    The script's `if __name__ == "__main__"` guard at line 143 means
    importing the module does not invoke `main()`. Returns a fresh
    module object each call so test isolation is preserved.
    """
    spec = importlib.util.spec_from_file_location("check_or_credits_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_fetch_live_handles_http_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        # _fetch_live's HTTPError branch (lines 58-63) — server returned
        # non-2xx. Must return None and emit a WARN naming the status.
        mod = _load_module()
        http_err = urllib.error.HTTPError(
            mod.OR_AUTH_KEY_URL,
            500,
            "Internal Server Error",
            {},
            None,  # type: ignore[arg-type]
        )
        with patch.object(mod.urllib.request, "urlopen", side_effect=http_err):
            result = mod._fetch_live("sk-or-test")
        assert result is None
        captured = capsys.readouterr()
        assert "HTTP 500" in captured.err
        assert "WARN" in captured.err

    @pytest.mark.parametrize(
        "exc",
        [
            urllib.error.URLError("dns lookup failed"),
            TimeoutError("socket timeout"),
            OSError("connection refused"),
        ],
    )
    def test_fetch_live_handles_network_error(self, capsys: pytest.CaptureFixture[str], exc: Exception) -> None:
        # _fetch_live's URLError | TimeoutError | OSError branch
        # (lines 64-69) — transport failed before any HTTP response.
        mod = _load_module()
        with patch.object(mod.urllib.request, "urlopen", side_effect=exc):
            result = mod._fetch_live("sk-or-test")
        assert result is None
        captured = capsys.readouterr()
        assert "network error" in captured.err
        assert "WARN" in captured.err

    def test_fetch_live_handles_parse_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        # _fetch_live's JSONDecodeError branch (lines 70-74) — server
        # returned 2xx but body is not valid JSON.
        mod = _load_module()

        class _FakeResp:
            def read(self) -> bytes:
                return b"not json at all"

            def __enter__(self):
                return self

            def __exit__(self, *a: object) -> None:
                return None

        with patch.object(mod.urllib.request, "urlopen", return_value=_FakeResp()):
            result = mod._fetch_live("sk-or-test")
        assert result is None
        captured = capsys.readouterr()
        assert "parse error" in captured.err
        assert "WARN" in captured.err


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
