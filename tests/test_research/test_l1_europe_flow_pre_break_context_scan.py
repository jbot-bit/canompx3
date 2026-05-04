from __future__ import annotations

import pandas as pd

from research.l1_europe_flow_pre_break_context_scan import (
    _bh_qvalues,
    _dir_match_oos,
    overall_verdict,
)


def test_bh_qvalues_monotone_and_order_preserving() -> None:
    qvals = _bh_qvalues([0.01, 0.04])
    assert len(qvals) == 2
    assert qvals[0] <= qvals[1]
    assert qvals[0] == 0.02
    assert qvals[1] == 0.04


def test_dir_match_oos_requires_minimum_oos_count() -> None:
    assert _dir_match_oos(0.10, 0.20, 4) is None
    assert _dir_match_oos(0.10, 0.20, 5) is True
    assert _dir_match_oos(0.10, -0.01, 5) is False


def test_overall_verdict_continue_beats_park_and_kill() -> None:
    rows = [
        {
            "passes_is_gates": False,
            "dir_match_oos": None,
        },
        {
            "passes_is_gates": True,
            "dir_match_oos": True,
        },
    ]
    df = pd.DataFrame(rows)
    results = [type("R", (), row)() for row in df.to_dict(orient="records")]
    assert overall_verdict(results) == "CONTINUE"
