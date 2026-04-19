from __future__ import annotations

import pandas as pd

from research.research_mnq_nyse_close_failure_mode_audit import (
    HOLDOUT_START,
    classify_rejection_reason,
    summarize_baseline,
)


def test_classify_rejection_reason_maps_phase3() -> None:
    reason = "Phase 3: 4/6 years positive (67% < 75%)"
    assert classify_rejection_reason(reason) == "year_stability"


def test_classify_rejection_reason_maps_criterion9() -> None:
    reason = "criterion_9: era 2024-2025 ExpR=-0.0512 < -0.05 (N=92)"
    assert classify_rejection_reason(reason) == "era_instability"


def test_summarize_baseline_counts_positive_years_pre_holdout_only() -> None:
    rows = pd.DataFrame(
        {
            "trading_day": pd.to_datetime(
                ["2024-01-02", "2024-02-02", "2025-01-03", "2026-01-05"]
            ),
            "orb_minutes": [5, 5, 5, 5],
            "rr_target": [1.0, 1.0, 1.0, 1.0],
            "pnl_r": [0.1, 0.2, -0.3, 0.9],
            "break_dir": ["long", "short", "long", "short"],
        }
    )
    summary = summarize_baseline(rows)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["positive_years"] == 1
    assert row["total_years"] == 2
    assert row["n_oos"] == 1
    assert row["avg_oos"] == 0.9
    assert HOLDOUT_START.year == 2026
