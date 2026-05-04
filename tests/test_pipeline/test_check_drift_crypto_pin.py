"""Tests for ``check_cryptography_pin_holds`` — the cryptography<47 sidecar guard.

The check is registered with ``is_advisory=False`` so Phase 1 (regression
detection: cryptography>=47 installed alongside fastmcp) can fail-closed
and abort the commit. Phase 2 (revisit-by staleness signal) is purely
informational and MUST NOT contribute to the returned violations list,
because the executor in ``main()`` treats any non-empty list from a
non-advisory check as a blocking violation. See bug fix in this commit.

Authority:
- ``docs/runtime/stages/fix-cryptography-check-advisory-leak.md``
- ``memory/feedback_mcp_venv_drift_cryptography47.md``
- PR #175 (squash-merged ``f2fb88f1``)
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _im

import pytest

import pipeline.check_drift as cd


@pytest.fixture
def stub_versions(monkeypatch):
    """Helper: stub ``importlib.metadata.version`` to control package presence."""

    def make(installed: dict[str, str | None]):
        def fake_version(name: str) -> str:
            if name in installed and installed[name] is not None:
                return installed[name]  # type: ignore[return-value]
            raise _im.PackageNotFoundError(name)

        monkeypatch.setattr(_im, "version", fake_version)

    return make


@pytest.fixture
def stub_today(monkeypatch):
    """Helper: stub ``datetime.date.today()`` to a fixed ISO date."""

    def make(iso: str):
        target = _dt.date.fromisoformat(iso)

        class FrozenDate(_dt.date):
            @classmethod
            def today(cls):
                return target

        monkeypatch.setattr(_dt, "date", FrozenDate)

    return make


class TestCheckCryptographyPinHolds:
    def test_clean_state_returns_no_violations(self, stub_versions, stub_today, capsys):
        """cryptography<47 + fastmcp + revisit-by in future → no violations, no advisory."""
        stub_versions({"cryptography": "46.0.1", "fastmcp": "2.0.0"})
        stub_today("2026-05-01")  # well before 2026-10-29 revisit-by
        result = cd.check_cryptography_pin_holds()
        assert result == []
        out = capsys.readouterr().out
        assert "ADVISORY" not in out

    def test_cryptography_47_with_fastmcp_blocks(self, stub_versions, stub_today, capsys):
        """Phase 1 regression: cryptography>=47 + fastmcp present → 1 violation (fail-closed)."""
        stub_versions({"cryptography": "47.0.0", "fastmcp": "2.0.0"})
        stub_today("2026-05-01")
        result = cd.check_cryptography_pin_holds()
        assert len(result) == 1
        msg = result[0]
        assert "cryptography==47.0.0" in msg
        assert "FastMCP" in msg
        assert "constraints.txt" in msg

    def test_cryptography_47_without_fastmcp_passes(self, stub_versions, stub_today):
        """No fastmcp → no Authlib breakage risk, even if cryptography>=47."""
        stub_versions({"cryptography": "47.0.0", "fastmcp": None})
        stub_today("2026-05-01")
        result = cd.check_cryptography_pin_holds()
        assert result == []

    def test_revisit_by_passed_does_not_block(self, stub_versions, stub_today, capsys):
        """Regression test for the Phase 2 advisory leak.

        With clean Phase 1 state and date past revisit-by, the check must
        return [] (no blocking violation). The advisory must surface via
        stdout, not via the return value. Before the fix, this test would
        have failed: the old code appended to ``violations``, so the return
        was a 1-element list that the executor treated as a blocking
        violation.
        """
        stub_versions({"cryptography": "46.0.1", "fastmcp": "2.0.0"})
        stub_today("2027-01-01")  # 64 days past 2026-10-29 revisit-by
        result = cd.check_cryptography_pin_holds()
        assert result == [], (
            "Phase 2 staleness must not contribute to violations list — "
            "is_advisory=False would make it block, contradicting the "
            "'ADVISORY' wording. Surface via print() instead."
        )
        out = capsys.readouterr().out
        assert "ADVISORY" in out
        assert "revisit-by:2026-10-29" in out
        assert "64 day(s) overdue" in out
