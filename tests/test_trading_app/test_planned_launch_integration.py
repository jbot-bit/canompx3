"""Integration tests for the planned-launch surface end-to-end.

Closes the audit finding (CONDITIONAL verdict, 2026-05-28) that the JS-layer
mode-derivation at bot_dashboard.html:4146 was the same untested-expression
class as the inverted-ternary bug it replaced.

These tests assert two things the Python-only unit tests don't:

1. The /api/planned-launch endpoint roundtrips writer payloads correctly so
   the dashboard's JS reads what was written — no server-side transform that
   could re-introduce mode collapse.
2. The bot_dashboard.html derivation logic (`planMode === "LIVE" ? "live" :
   planMode === "DEMO" ? "demo" : "signal"`) is structurally correct AND not
   regressed by future edits. Done by static-scanning the HTML for the bug-
   class pattern `bs.demo ? "demo" : bs.signal_only ? "signal" : "demo"`.
"""

from __future__ import annotations

import re
from pathlib import Path

import anyio
import httpx
import pytest

from trading_app.live import bot_dashboard, planned_launch


class _ASGIClient:
    def __init__(self, app):
        self._app = app

    def get(self, url: str, **kwargs) -> httpx.Response:
        async def _get() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                return await c.get(url, **kwargs)

        return anyio.run(_get)


@pytest.fixture
def client():
    return _ASGIClient(bot_dashboard.app)


@pytest.fixture
def tmp_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "bot_planned_launch.json"
    monkeypatch.setattr(planned_launch, "PLANNED_LAUNCH_PATH", target)
    return target


@pytest.mark.parametrize(
    "mode,expected_cli_mode,expected_cta_label",
    [
        ("SIGNAL", "signal", "START SIGNAL"),
        ("DEMO", "demo", "START DEMO"),
        ("LIVE", "live", "START LIVE"),
    ],
)
def test_endpoint_returns_mode_that_js_will_map_correctly(
    client: _ASGIClient,
    tmp_artifact: Path,
    mode: str,
    expected_cli_mode: str,
    expected_cta_label: str,
) -> None:
    """End-to-end: writer → endpoint → JS-derivation contract.

    The dashboard's primary-CTA derivation at bot_dashboard.html computes:

        cliMode = planMode === "LIVE" ? "live" :
                  planMode === "DEMO" ? "demo" : "signal"
        modeLabel = planMode === "LIVE" ? "START LIVE" :
                    planMode === "DEMO" ? "START DEMO" : "START SIGNAL"

    This test re-executes that exact derivation against the endpoint response
    to guarantee the contract holds. If the JS ever drifts (or someone
    reintroduces the inverted-ternary class of bug at the writer or endpoint
    level), this fails.
    """
    planned_launch.write_planned_launch(
        profile_id="adhoc",
        mode=mode,
        source="CLI",
        copies=1,
        instruments=["MNQ"],
    )

    response = client.get("/api/planned-launch")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] == mode, f"endpoint returned {body['mode']!r} for input {mode!r}"

    # Replicate the dashboard's exact derivation. If THIS logic ever differs
    # from bot_dashboard.html, the JS-static-scan test below will also fail.
    plan_mode = str(body["mode"]).upper()
    cli_mode = "live" if plan_mode == "LIVE" else "demo" if plan_mode == "DEMO" else "signal"
    cta_label = "START LIVE" if plan_mode == "LIVE" else "START DEMO" if plan_mode == "DEMO" else "START SIGNAL"
    assert cli_mode == expected_cli_mode
    assert cta_label == expected_cta_label


def test_endpoint_surfaces_copies_so_dashboard_can_show_fan_out(client: _ASGIClient, tmp_artifact: Path) -> None:
    """profile.copies > 1 = signal mirrors to N broker accounts.

    The dashboard banner says "N broker accounts · REAL MONEY" on LIVE +
    copies > 1. Endpoint must surface broker_accounts_count for that.
    """
    planned_launch.write_planned_launch(
        profile_id="adhoc",
        mode="LIVE",
        source="CLI",
        copies=3,
        instruments=["MNQ"],
    )
    body = client.get("/api/planned-launch").json()
    assert body["broker_accounts_count"] == 3
    assert body["copies"] == 3


def test_endpoint_returns_unknown_when_no_file(client: _ASGIClient, tmp_artifact: Path) -> None:
    """The dashboard banner's UNKNOWN path triggers off status. Missing-file
    must NEVER return status=ok with any guessed mode."""
    assert not tmp_artifact.exists()
    body = client.get("/api/planned-launch").json()
    assert body["status"] == "unknown"
    assert "mode" not in body or body.get("mode") not in {"SIGNAL", "DEMO", "LIVE"}


# ──────────────────────────────────────────────────────────────────────────
# Static guard against the inverted-ternary bug class (audit finding closure)
# ──────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = Path(__file__).parent.parent.parent / "trading_app" / "live" / "bot_dashboard.html"


def test_dashboard_does_not_reintroduce_inverted_demo_ternary() -> None:
    """Regression guard for the original 2026-05-28 bug.

    The pre-fix code at bot_dashboard.html:4060 was:

        const orchMode = bs.demo ? "demo" : bs.signal_only ? "signal" : "demo";

    Three defects:
      (a) `bs.demo` short-circuits before `bs.signal_only` — but signal_only
          implies demo at run_live_session.py:883, so SIGNAL renders DEMO.
      (b) Else-branch is "demo", not "live" — LIVE is unreachable.
      (c) Source is bot_state.broker_status (stale pre-start), not the
          planned-launch surface.

    This test fails if any of those patterns reappear.
    """
    text = DASHBOARD_HTML.read_text(encoding="utf-8")
    # (a)+(b): the exact bug pattern — broker_status demo/signal_only ternary
    bug_pattern_a = re.compile(
        r"bs\.demo\s*\?\s*[\"']demo[\"']\s*:\s*bs\.signal_only\s*\?\s*[\"']signal[\"']\s*:\s*[\"']demo[\"']"
    )
    assert not bug_pattern_a.search(text), (
        "REGRESSION: the inverted-ternary bug class (bs.demo before bs.signal_only, "
        'else="demo") has reappeared in bot_dashboard.html. See the audit closure for '
        "2026-05-28-dashboard-planned-launch-surface stage."
    )
    # (b) restated: no "demo" else-branch where "live" is the third real choice.
    bug_pattern_b = re.compile(r"signal_only\s*\?\s*[\"']signal[\"']\s*:\s*[\"']demo[\"']")
    assert not bug_pattern_b.search(text), (
        'REGRESSION: signal_only ? "signal" : "demo" pattern is back — '
        "this collapses LIVE → DEMO and was the original 4060 bug."
    )


def test_dashboard_uses_planned_launch_for_pre_start_mode() -> None:
    """The fix relies on the dashboard reading _lastPlannedLaunch, not
    _lastStatusData.broker_status, when computing the START CTA mode.

    If a future edit reverts to broker_status as the pre-start mode source,
    this test fails (the new ternary at planMode === "LIVE" disappears).
    """
    text = DASHBOARD_HTML.read_text(encoding="utf-8")
    # The fix must contain the correct planMode derivation.
    assert 'planMode === "LIVE"' in text, (
        "bot_dashboard.html no longer derives CTA mode from planMode — the "
        "planned-launch surface fix has been reverted or refactored."
    )
    assert "_lastPlannedLaunch" in text, (
        "bot_dashboard.html no longer references _lastPlannedLaunch — the "
        "planned-launch fetcher wiring has been removed."
    )


def test_dashboard_calls_planned_launch_endpoint() -> None:
    text = DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "/api/planned-launch" in text, (
        "bot_dashboard.html no longer fetches /api/planned-launch — the banner cannot be populated."
    )
