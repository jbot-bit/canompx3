"""Test configuration for trading_app tests.

Provides shared autouse fixtures that prevent test I/O from contaminating
production runtime files.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _redirect_alerts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Redirect operator alert + bot-state writes to a per-test tmpdir.

    Prevents operator-alert, bot-state, and live-health runtime files from
    being written during the test suite (ALERT-CONTAM-N2 class; n=2 incident
    2026-05-19).

    All three module-level Path constants are read at call time, so
    monkeypatching the attribute is sufficient — no import-time capture.
    """
    import trading_app.live.alert_engine as _ae
    import trading_app.live.bot_state as _bs

    monkeypatch.setattr(_ae, "ALERTS_PATH", tmp_path / "operator_alerts.jsonl")
    monkeypatch.setattr(_bs, "STATE_FILE", tmp_path / "bot_state.json")
    monkeypatch.setattr(_bs, "LIVE_HEALTH_FILE", tmp_path / "live_health.json")
