from __future__ import annotations

import math

import pandas as pd

from research.mnq_comex_unfiltered_overlay_v1 import _bh_fdr, _group_stats


def test_bh_fdr_enforces_monotone_q_values() -> None:
    q_map = _bh_fdr(
        [
            ("h2", 0.04),
            ("h1", 0.01),
            ("h3", 0.03),
        ]
    )
    assert math.isclose(q_map["h1"], 0.03)
    assert math.isclose(q_map["h3"], 0.04)
    assert math.isclose(q_map["h2"], 0.04)


def test_group_stats_treats_scratch_as_zero_but_resolved_separately() -> None:
    pnl_eff = pd.Series([1.5, -1.0, 0.0])
    outcomes = pd.Series(["win", "loss", "scratch"])
    expr_scratch0, expr_resolved, wr_resolved = _group_stats(pnl_eff, outcomes)
    assert math.isclose(expr_scratch0, (1.5 - 1.0 + 0.0) / 3.0)
    assert math.isclose(expr_resolved, (1.5 - 1.0) / 2.0)
    assert math.isclose(wr_resolved, 1.0 / 2.0)
