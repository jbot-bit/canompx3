"""Tests for research.oos_holdout.powered_oos_split.

Proves the operator-mandated invariants of the no-wait powered-OOS reform:
- temporal split (not random)
- 2026+ excluded from the selection/deploy split
- underpowered tail fails CLOSED (BLOCKED, never silent pass)
- negative OOS expectancy fails deployment
- adequately-powered positive OOS can pass
- validate_strategy-grade verdict is never NULL (always one of PASS/FAIL/BLOCKED)
- honest provenance flag for contaminated legacy candidates
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research.oos_holdout import (
    BLOCKED,
    FAIL,
    PASS,
    powered_oos_split,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM


def _series(values: list[float], start: date = date(2022, 1, 3)) -> list[tuple[date, float]]:
    """Build a (trading_day, pnl_r) list with one trade per business-ish day."""
    out: list[tuple[date, float]] = []
    d = start
    for v in values:
        out.append((d, v))
        d = d + timedelta(days=1)
    return out


def test_verdict_is_never_null() -> None:
    """Every code path returns a non-null verdict in {PASS, FAIL, BLOCKED}."""
    res = powered_oos_split(_series([0.1] * 100))
    assert res.verdict in {PASS, FAIL, BLOCKED}
    assert res.verdict is not None


def test_split_is_temporal_not_random() -> None:
    """OOS slice is the most-recent contiguous tail by trading_day."""
    # Train era strongly positive, OOS era strongly negative. A temporal split
    # MUST put the late (negative) trades in OOS; a random split would mix.
    trades = _series([0.5] * 70 + [-0.5] * 30)
    res = powered_oos_split(trades, oos_fraction=0.30)
    assert res.n_powered_oos == 30
    assert res.n_train == 70
    # OOS slice must be the late negative block.
    assert res.powered_oos_exp_r is not None and res.powered_oos_exp_r < 0
    assert res.is_exp_r is not None and res.is_exp_r > 0
    # split_start must be AFTER all the train days.
    assert res.split_start is not None
    assert res.split_start > trades[res.n_train - 1][0]


def test_2026_excluded_from_split() -> None:
    """Trades on/after HOLDOUT_SACRED_FROM are dropped before splitting."""
    pre = _series([0.2] * 60, start=date(2025, 1, 1))
    # Append 2026 trades that MUST be excluded from the powered split.
    post = [(HOLDOUT_SACRED_FROM + timedelta(days=i), 0.2) for i in range(40)]
    res = powered_oos_split(pre + post)
    # n_total counts ONLY pre-holdout eligible trades.
    assert res.n_total == 60
    assert res.split_end is not None and res.split_end < HOLDOUT_SACRED_FROM
    assert res.provenance_warning is not None
    assert "sacred calendar holdout" in res.provenance_warning


def test_underpowered_tail_fails_closed() -> None:
    """A tiny eligible sample cannot reach deploy-grade power -> BLOCKED."""
    # 30 trades total -> 9 OOS. Small effect size -> sub-0.50 power.
    trades = _series([0.02, -0.01] * 15)  # 30 trades, near-zero mean, noisy
    res = powered_oos_split(trades)
    assert res.verdict == BLOCKED
    assert res.estimated_power < 0.50
    assert "Not deployable" in res.reason or "STATISTICALLY_USELESS" in res.reason


def test_below_min_total_trades_blocked() -> None:
    """Fewer than MIN_TOTAL_TRADES eligible -> BLOCKED, fail-closed."""
    res = powered_oos_split(_series([0.3] * 20))
    assert res.verdict == BLOCKED
    assert "insufficient eligible trades" in res.reason


def test_negative_oos_expr_fails_deployment() -> None:
    """Positive IS, negative OOS, adequate power -> FAIL (not BLOCKED)."""
    # Large, consistent effect so power is high; OOS flips sign.
    trades = _series([0.6] * 200 + [-0.6] * 90)
    res = powered_oos_split(trades)
    assert res.verdict == FAIL
    assert res.powered_oos_exp_r is not None and res.powered_oos_exp_r <= 0


def test_adequately_powered_positive_oos_passes() -> None:
    """Strong, stable edge across train and OOS with power >= 0.50 -> PASS."""
    # Consistent ~0.5R per trade, low noise, large N -> high power, ratio ~1.
    trades = _series([0.5, 0.5, 0.5, 0.5, -0.5] * 80)  # 400 trades, mean 0.3
    res = powered_oos_split(trades)
    assert res.verdict == PASS
    assert res.dir_match is True
    assert res.oos_is_ratio is not None and res.oos_is_ratio >= 0.40
    assert res.estimated_power >= 0.50


def test_oos_is_ratio_floor_fails() -> None:
    """OOS positive but decayed below the 0.40*IS ratio at power -> FAIL."""
    # Strong IS edge, OOS positive but much weaker (ratio < 0.40).
    trades = _series([0.8] * 200 + [0.05] * 90)
    res = powered_oos_split(trades)
    # Either FAIL on ratio, or (if power low) BLOCKED — must not PASS.
    assert res.verdict in {FAIL, BLOCKED}
    assert res.verdict != PASS


def test_provenance_flag_marks_contaminated_split() -> None:
    """discovery_touched_recent_fraction=True -> provenance_clean False + warning."""
    trades = _series([0.5, 0.5, 0.5, 0.5, -0.5] * 80)
    res = powered_oos_split(trades, discovery_touched_recent_fraction=True)
    assert res.provenance_clean is False
    assert res.provenance_warning is not None
    assert "CONTAMINATED" in res.provenance_warning
    # Verdict is still computed (contamination is a label, not a hard block here).
    assert res.verdict in {PASS, FAIL, BLOCKED}


def test_clean_split_has_clean_provenance() -> None:
    """Default (no contamination flag) on pre-2026 data -> provenance_clean True."""
    trades = _series([0.4, 0.4, -0.2] * 60, start=date(2021, 1, 1))
    res = powered_oos_split(trades)
    assert res.provenance_clean is True


def test_to_dict_has_all_reporting_fields() -> None:
    """The reporting record exposes every operator-mandated field."""
    res = powered_oos_split(_series([0.3] * 100))
    d = res.to_dict()
    required = {
        "verdict",
        "reason",
        "n_total",
        "n_train",
        "n_powered_oos",
        "split_start",
        "split_end",
        "oos_fraction",
        "estimated_power",
        "power_tier",
        "is_exp_r",
        "powered_oos_exp_r",
        "oos_is_ratio",
        "dir_match",
        "provenance_clean",
        "provenance_warning",
    }
    assert required.issubset(d.keys())


def test_degenerate_variance_blocks() -> None:
    """Zero-variance IS (all identical) -> BLOCKED, cannot size power."""
    trades = _series([0.25] * 100)
    res = powered_oos_split(trades)
    # All-identical -> IS std 0 -> cannot size power -> BLOCKED.
    assert res.verdict == BLOCKED
    assert "degenerate IS variance" in res.reason
