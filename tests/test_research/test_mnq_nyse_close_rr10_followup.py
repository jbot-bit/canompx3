from __future__ import annotations

import pandas as pd

from research.mnq_nyse_close_rr10_followup import (
    build_candidate_table,
    evaluate_followup,
)


def test_build_candidate_table_marks_missing_orbg8_execution() -> None:
    experimental = pd.DataFrame(
        {
            "filter_type": ["GAP_R015", "ORB_G5_NOFRI"],
            "orb_minutes": [5, 5],
        }
    )
    table = build_candidate_table(experimental)
    orbg8 = table[table["filter_type"] == "ORB_G8"]
    assert len(orbg8) == 2
    assert not bool(orbg8["experimental_rr10_present"].any())


def test_evaluate_followup_recommends_continue_when_baseline_alive_and_orbg8_missing() -> None:
    baseline = pd.DataFrame(
        {
            "orb_minutes": [5, 15, 30],
            "avg_is": [0.08, 0.11, 0.13],
        }
    )
    experimental = pd.DataFrame(
        {
            "filter_type": ["GAP_R015", "OVNRNG_100", "ORB_G5_NOFRI"],
            "orb_minutes": [5, 5, 5],
        }
    )
    candidates = build_candidate_table(experimental)
    decision, recommendation, framing = evaluate_followup(baseline, experimental, candidates)
    assert decision == "CONTINUE with narrow prereg"
    assert "ORB_G8 RR1.0" in recommendation
    assert set(framing["verdict"]) == {"REJECT", "PARTIAL", "PRIMARY"}


def test_evaluate_followup_kills_when_broad_baseline_is_dead() -> None:
    baseline = pd.DataFrame(
        {
            "orb_minutes": [5, 15, 30],
            "avg_is": [-0.02, -0.01, -0.03],
        }
    )
    experimental = pd.DataFrame(columns=["filter_type", "orb_minutes"])
    candidates = build_candidate_table(experimental)
    decision, recommendation, _framing = evaluate_followup(baseline, experimental, candidates)
    assert decision == "KILL broad-family follow-up"
    assert "No broad positive RR1.0 baseline remains." in recommendation
