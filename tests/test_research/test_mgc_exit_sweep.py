"""Tests for the one-pass multi-exit replay in
research/mgc_trend_day_exit_sweep.py::compute_all_exits.

The fixed-target, breakeven-hold, and scaleout exits are new and complex. These
pin their behavior on synthetic OHLC bars, including the sub-bar causality guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.cost_model import COST_SPECS
from research.mgc_trend_day_exit_sweep import EXIT_KEYS, compute_all_exits

SPEC = COST_SPECS["MGC"]
ENTRY, STOP = 2000.0, 1990.0  # long, hard stop 10 pts (1R = 10 pts gross)


def _bars(seq):
    return pd.DataFrame(
        [
            {"ts_utc": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i), "high": h, "low": lo, "close": c}
            for i, (h, lo, c) in enumerate(seq)
        ]
    )


def test_all_exit_keys_present():
    seq = [(2010, 2000, 2009), (2020, 2009, 2019)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert set(ex.keys()) == set(EXIT_KEYS), f"missing keys: {set(EXIT_KEYS) - set(ex.keys())}"


def test_empty_bars_all_none():
    ex = compute_all_exits(_bars([]), ENTRY, STOP, 1, SPEC)
    assert all(v is None for v in ex.values())


def test_fixed_target_hit():
    # Reaches +40 pts (4R). fixed_3R and fixed_4R should fill near their targets;
    # fixed_5R/6R never reached -> ride to close (~+4R).
    seq = [(2010, 2000, 2009), (2042, 2010, 2041), (2041, 2040, 2040)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert 2.5 < ex["fixed_3R"] < 3.1, ex["fixed_3R"]
    assert 3.5 < ex["fixed_4R"] < 4.1, ex["fixed_4R"]
    # 5R/6R never hit -> ride to close ~4R
    assert ex["fixed_5R"] > 3.0 and ex["fixed_6R"] > 3.0


def test_fixed_target_stopped_before_target():
    # Spikes +25 pts (2.5R) then hard-stops. fixed_3R never reached -> -1R.
    seq = [(2025, 2000, 2024), (2010, 1988, 1989)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert ex["fixed_3R"] <= -1.0, ex["fixed_3R"]


def test_breakeven_hold_returns_to_be():
    # Reaches +1R (arms BE), then price returns to entry -> exit ~0R (net slightly <0).
    seq = [(2012, 2000, 2011), (2013, 1999, 2000), (2001, 2000, 2000)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert -0.2 < ex["breakeven_hold"] < 0.2, ex["breakeven_hold"]


def test_breakeven_hold_rides_to_close():
    # Arms BE at +1R, never returns to entry, closes at +30 pts (3R) -> ~3R.
    seq = [(2012, 2000, 2011), (2032, 2012, 2031), (2031, 2025, 2030)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert ex["breakeven_hold"] > 2.5, ex["breakeven_hold"]


def test_breakeven_never_arms_equals_close():
    # Max +5 pts (0.5R) < 1R; never arms BE -> = hold to close.
    seq = [(2005, 2000, 2003), (2004, 2001, 2002)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert abs(ex["breakeven_hold"] - ex["hold_to_close"]) < 1e-6


def test_scaleout_banks_half_then_rides():
    # Banks 0.5 at +1R, second half rides to +30 (3R) close.
    # blended ~ 0.5*1R + 0.5*3R = ~2R (net of friction slightly less).
    seq = [(2012, 2000, 2011), (2032, 2012, 2031), (2031, 2025, 2030)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert 1.6 < ex["scaleout_1R_runner"] < 2.2, ex["scaleout_1R_runner"]


def test_scaleout_stopped_before_1R():
    seq = [(2008, 2000, 2007), (2009, 1988, 1989)]  # hard stop before +1R
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert ex["scaleout_1R_runner"] <= -1.0, ex["scaleout_1R_runner"]


def test_hold_to_close_dies_on_hard_stop():
    # Spikes +20 (2R), then hard-stops on bar 2, then "closes" higher. A hold
    # CANNOT ride through the stop -> must realize -1R, not the late close.
    seq = [(2020, 2000, 2019), (2010, 1988, 1989), (2030, 2025, 2030)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert ex["hold_to_close"] <= -1.0, ex["hold_to_close"]


def test_hold_to_close_rides_when_not_stopped():
    seq = [(2010, 2000, 2009), (2030, 2009, 2030), (2031, 2025, 2028)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert ex["hold_to_close"] > 2.0, ex["hold_to_close"]  # ~2.8R close


def test_mfe_ceiling_is_upper_bound():
    # MFE ceiling >= every realizable exit (it's the max favorable tick).
    seq = [(2010, 2000, 2009), (2050, 2010, 2030), (2030, 2020, 2025)]
    ex = compute_all_exits(_bars(seq), ENTRY, STOP, 1, SPEC)
    realizable = [v for k, v in ex.items() if k != "mfe_ceiling" and v is not None]
    assert all(ex["mfe_ceiling"] >= r - 1e-6 for r in realizable), ex


def test_short_mirror_fixed_target():
    entry_s, stop_s = 2000.0, 2010.0  # short
    seq_long = [(2010, 2000, 2009), (2042, 2010, 2041), (2041, 2040, 2040)]
    seq_short = [(2000, 1990, 1991), (1990, 1958, 1959), (1960, 1959, 1960)]
    el = compute_all_exits(_bars(seq_long), ENTRY, STOP, 1, SPEC)
    es = compute_all_exits(_bars(seq_short), entry_s, stop_s, -1, SPEC)
    assert abs(el["fixed_4R"] - es["fixed_4R"]) < 0.05, (el["fixed_4R"], es["fixed_4R"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
