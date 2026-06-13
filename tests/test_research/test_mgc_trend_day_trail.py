"""Tests for the give-back-of-peak trailing-stop sim in
research/mgc_trend_day_tail_descriptive.py::compute_trail_r.

The trail is the load-bearing NEW logic (Phase A realizable tail #3). These
tests pin its causal behavior — in particular the within-bar causality guard
that prevents a single 1m bar's low from tripping the trail its own high just
raised (a sub-bar look-ahead trap).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.cost_model import COST_SPECS
from research.mgc_trend_day_tail_descriptive import compute_trail_r

SPEC = COST_SPECS["MGC"]
ENTRY, STOP = 2000.0, 1990.0  # long, hard stop 10 pts below (1R = 10 pts gross)


def _bars(seq):
    """seq = list of (high, low, close) 1m bars."""
    return pd.DataFrame(
        [
            {"ts_utc": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i), "high": h, "low": lo, "close": c}
            for i, (h, lo, c) in enumerate(seq)
        ]
    )


def test_empty_bars_returns_none():
    assert compute_trail_r(_bars([]), ENTRY, STOP, 1, SPEC) is None


def test_zero_hard_stop_returns_none():
    assert compute_trail_r(_bars([(2005, 2000, 2004)]), ENTRY, ENTRY, 1, SPEC) is None


def test_strong_trend_holds_runs_near_close():
    # Trends to +42 pts (4.2R), holds — trail at 50% (2.1R) never hit -> ~close.
    seq = [(2005, 2000, 2004), (2015, 2004, 2014), (2030, 2014, 2029), (2042, 2030, 2041), (2041, 2040, 2040)]
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert r is not None and r > 3.0, f"strong trend should run well past 3R, got {r}"


def test_within_bar_causality_no_same_bar_trip():
    # Bar 1 makes a new high (+15pts) AND has a low (+4pts) below the trail
    # (15*0.5=7.5). A correct trail does NOT exit on this same bar — its low
    # preceded its own new high. The trade should survive to a later bar / close.
    seq = [(2005, 2000, 2004), (2015, 2004, 2014), (2016, 2015, 2016)]
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    # peak 1.6R, trail 0.8R never breached on a LATER bar -> hold to close ~1.6R.
    assert r is not None and r > 1.0, f"same-bar low must not trip a freshly-raised trail, got {r}"


def test_spike_then_collapse_trails_out():
    # Spikes to +30 pts (3R) then collapses; a LATER bar's low breaches the trail.
    seq = [(2010, 2000, 2009), (2030, 2009, 2029), (2031, 1999, 2000), (2000, 1999, 2000)]
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert r is not None and 1.0 < r < 2.0, f"giveback should trail out near 1.5R, got {r}"


def test_hard_stop_first_bar():
    seq = [(2001, 1988, 1989)]  # drops 12 pts -> hard stop
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert r is not None and r <= -1.0, f"hard stop should be <= -1R, got {r}"


def test_never_arms_holds_to_close():
    # Max +5 pts (0.5R) < arm (1R); fades to +2 pts close -> small positive net.
    seq = [(2005, 2000, 2003), (2004, 2001, 2002), (2003, 2001, 2002)]
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert r is not None and 0.0 < r < 0.5, f"unarmed hold-to-close should be small +, got {r}"


def test_short_mirror_matches_long():
    # Short version of the strong-trend case — must be symmetric.
    entry_s, stop_s = 2000.0, 2010.0  # short, stop 10 pts above
    seq_long = [(2005, 2000, 2004), (2015, 2004, 2014), (2030, 2014, 2029), (2042, 2030, 2041), (2041, 2040, 2040)]
    seq_short = [(2000, 1995, 1996), (1996, 1985, 1986), (1986, 1970, 1971), (1971, 1958, 1959), (1960, 1959, 1960)]
    r_long = compute_trail_r(_bars(seq_long), ENTRY, STOP, 1, SPEC)
    r_short = compute_trail_r(_bars(seq_short), entry_s, stop_s, -1, SPEC)
    assert r_long is not None and r_short is not None
    assert abs(r_long - r_short) < 0.01, f"long/short must be symmetric: {r_long} vs {r_short}"


def test_friction_makes_realizable_net_of_cost():
    # A +20 pt (2R gross) move held to close should be slightly < 2R net of friction.
    seq = [(2010, 2000, 2009), (2020, 2009, 2020)]
    r = compute_trail_r(_bars(seq), ENTRY, STOP, 1, SPEC)
    assert r is not None and r < 2.0, f"realizable R must be net of friction (< gross 2R), got {r}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
