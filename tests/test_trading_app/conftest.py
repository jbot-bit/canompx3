"""Test configuration for trading_app tests.

Provides shared autouse fixtures that prevent test I/O from contaminating
production runtime files.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _redirect_alerts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect operator alert writes to a per-test tmpdir.

    session_orchestrator._notify() calls alert_engine.record_operator_alert(),
    which appends to the module-level ALERTS_PATH constant.  Without this
    fixture every test that exercises _notify() (even indirectly) writes to
    the production data/runtime/operator_alerts.jsonl file.

    Monkeypatching the module-level attribute is sufficient because
    record_operator_alert() reads ALERTS_PATH at call time (not import time).
    """
    import trading_app.live.alert_engine as _ae

    monkeypatch.setattr(_ae, "ALERTS_PATH", tmp_path / "operator_alerts.jsonl")
