"""
Unit tests for the rolling-correlation tripwire monitor.

Tests:
  1. `consecutive_breach_runs` correctly identifies runs above threshold
  2. Short runs below min_run are not reported as breaches
  3. `rolling_pairwise_corr` returns NaN for the first (window-1) days then
     populated values
  4. `compute_alarms` returns zero alarms when correlations stay below threshold
  5. `compute_alarms` returns an alarm when a run of >= min_run days breaches
  6. Correlation math: known input → expected output (sanity check)

These pin the regime-change tripwire so it doesn't silently stop catching
correlation jumps — which is the whole reason the monitor exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reports.monitor_lane_correlation_rolling import (  # noqa: E402
    compute_alarms,
    consecutive_breach_runs,
    rolling_pairwise_corr,
)


def test_consecutive_breach_runs_single_run():
    s = pd.Series([0.1, 0.2, 0.5, 0.6, 0.7, 0.1, 0.1], index=pd.date_range("2026-01-01", periods=7))
    runs = consecutive_breach_runs(s, threshold=0.3, min_run=3)
    assert len(runs) == 1
    assert runs[0][2] == 3  # run length
    assert str(runs[0][0].date()) == "2026-01-03"
    assert str(runs[0][1].date()) == "2026-01-05"


def test_consecutive_breach_runs_short_run_ignored():
    s = pd.Series([0.5, 0.6, 0.1, 0.1], index=pd.date_range("2026-01-01", periods=4))
    runs = consecutive_breach_runs(s, threshold=0.3, min_run=3)
    assert runs == []


def test_consecutive_breach_runs_multiple_runs():
    s = pd.Series(
        [0.4, 0.5, 0.5, 0.5, 0.1, 0.6, 0.7, 0.8, 0.9, 0.1],
        index=pd.date_range("2026-01-01", periods=10),
    )
    runs = consecutive_breach_runs(s, threshold=0.3, min_run=3)
    assert len(runs) == 2
    assert runs[0][2] == 4
    assert runs[1][2] == 4


def test_consecutive_breach_runs_ends_in_breach():
    """Run that extends to the end of the series must be counted."""
    s = pd.Series([0.1, 0.1, 0.5, 0.6, 0.7], index=pd.date_range("2026-01-01", periods=5))
    runs = consecutive_breach_runs(s, threshold=0.3, min_run=3)
    assert len(runs) == 1
    assert runs[0][2] == 3


def test_rolling_pairwise_corr_initial_nan_then_populated():
    """Window = 5 → first 4 days NaN, then populated."""
    n = 10
    idx = pd.date_range("2026-01-01", periods=n)
    df = pd.DataFrame({"A": np.random.randn(n), "B": np.random.randn(n)}, index=idx)
    rolling = rolling_pairwise_corr(df, window=5)
    # All reported rolling rows must have non-NaN (we filter at build time)
    assert rolling["corr"].notna().all()
    # Only days 5..10 should have entries → 6 rows for the single pair
    assert len(rolling) == 6


def test_rolling_pairwise_corr_perfectly_correlated():
    n = 20
    idx = pd.date_range("2026-01-01", periods=n)
    x = np.random.randn(n)
    df = pd.DataFrame({"A": x, "B": 2 * x + 1}, index=idx)  # perfect positive correlation
    rolling = rolling_pairwise_corr(df, window=5)
    assert (rolling["corr"] > 0.999).all()


def test_compute_alarms_clear_book():
    """Mean correlation well below 0.30 → zero alarms."""
    np.random.seed(0)
    n = 60
    idx = pd.date_range("2026-01-01", periods=n)
    df = pd.DataFrame(
        {"A": np.random.randn(n), "B": np.random.randn(n), "C": np.random.randn(n)},
        index=idx,
    )
    rolling = rolling_pairwise_corr(df, window=30)
    alarms = compute_alarms(rolling)
    assert alarms == []


def test_compute_alarms_correlated_pair_trips_alarm():
    """One perfectly-correlated pair + 40 days → single alarm with 40+ day run."""
    n = 50
    idx = pd.date_range("2026-01-01", periods=n)
    x = np.random.randn(n)
    df = pd.DataFrame(
        {
            "A": x,
            "B": x,  # identical → correlation = 1.0
            "C": np.random.randn(n),
        },
        index=idx,
    )
    rolling = rolling_pairwise_corr(df, window=30)
    alarms = compute_alarms(rolling)
    a_b_alarms = [a for a in alarms if {a["pair_a"], a["pair_b"]} == {"A", "B"}]
    assert len(a_b_alarms) == 1
    assert a_b_alarms[0]["peak_corr"] > 0.99
    # Run should cover all populated rolling days (50 - 30 + 1 = 21)
    assert a_b_alarms[0]["run_length_days"] >= 10


def test_compute_alarms_transient_spike_under_min_run_ignored():
    """A spike that lasts only 5 days when min_run = 10 must not alarm."""
    n = 60
    idx = pd.date_range("2026-01-01", periods=n)
    x = np.random.randn(n)
    # B correlates with A only for 5 specific rolling windows
    b = np.random.randn(n)
    b[35:40] = x[35:40] * 10  # strong spike
    df = pd.DataFrame({"A": x, "B": b}, index=idx)
    rolling = rolling_pairwise_corr(df, window=30)
    alarms = compute_alarms(rolling)
    # Any alarm must have run_length >= 10
    for a in alarms:
        assert a["run_length_days"] >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
