from __future__ import annotations

import math

from research.mnq_l1_europe_flow_prebreak_context_v1 import _apply_verdicts, _bh_fdr


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


def test_family_verdict_is_park_when_only_survivor_fails_oos_dir_match() -> None:
    rows = [
        {
            "delta_ok": True,
            "expr_ok": True,
            "years_ok": True,
            "raw_p_ok": True,
            "n_ok": True,
            "operating_range_ok": True,
            "q_family": 0.01,
            "n_on_oos": 7,
            "dir_match_oos": False,
        },
        {
            "delta_ok": False,
            "expr_ok": True,
            "years_ok": True,
            "raw_p_ok": False,
            "n_ok": True,
            "operating_range_ok": True,
            "q_family": 0.50,
            "n_on_oos": 12,
            "dir_match_oos": False,
        },
    ]
    verdict = _apply_verdicts(rows)
    assert verdict == "PARK"
    assert rows[0]["verdict"] == "PARK"
    assert rows[1]["verdict"] == "KILL"
