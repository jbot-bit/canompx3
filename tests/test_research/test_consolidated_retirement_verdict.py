"""Hardening tests for ``research/consolidated_retirement_verdict.py``.

Two layers of protection against future drift:

1. **Source-doc drift detection**: the script hardcodes lane-ID sets from
   four committed audit docs (retirement queue, fire-rate, SGP Jaccard,
   Mode-B grandfather). If those docs change, hardcoded sets go stale.
   These tests parse the source docs at test time and assert the
   hardcoded sets match — any divergence is a hard fail the committee
   must resolve before trusting the consolidated view.

2. **Verdict assignment unit tests**: the ``assign_verdict`` function has
   a priority ordering (RETIRE_URGENT > RETIRE_STANDARD > N_UNDERPOWERED >
   RECLASSIFY_COST > REVIEW_TIER2 > REVIEW_CAPACITY > REVIEW_C4_WT_FAIL >
   BETTER_THAN_PEERS > KEEP). Unit tests exercise each branch to guard
   against accidental reordering or gate drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research.consolidated_retirement_verdict import (
    FIRE_RATE_ARITHMETIC_ONLY,
    FIRE_RATE_OVER95,
    FIRE_RATE_ZERO_INJECTION_ARTIFACT,
    RETIREMENT_BETTER,
    RETIREMENT_HOLD,
    RETIREMENT_TIER1,
    RETIREMENT_TIER1_URGENT,
    RETIREMENT_TIER2,
    SGP_JACCARD_PAIR,
    SUBSET_T_ARITHMETIC_LIFT,
    SUBSET_T_TIER1_PASS,
    SUBSET_T_TIER1_THIN_N,
    SUBSET_T_TIER2,
    SUBSET_T_TIER4_FAIL_CONVENTIONAL,
    VERDICT_ORDER,
    assign_verdict,
    subset_t_annotation,
)
from research.mode_a_revalidation_active_setups import LaneRevalidation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = PROJECT_ROOT / "docs" / "audit" / "results"

RETIREMENT_DOC = AUDIT_DIR / "2026-04-19-mnq-retirement-queue-committee-action.md"
FIRE_RATE_DOC = AUDIT_DIR / "2026-04-19-fire-rate-audit.md"
SGP_JACCARD_DOC = AUDIT_DIR / "2026-04-19-sgp-o15-o30-jaccard.md"
PHASE_2_5_CSV = PROJECT_ROOT / "research" / "output" / "phase_2_5_portfolio_subset_t_sweep.csv"


# --- Source-doc drift detection ----------------------------------------


def _doc_text(path: Path) -> str:
    """Return full text of an audit doc; skip the test with xfail if absent."""
    if not path.exists():
        pytest.skip(f"source audit doc not found on this branch: {path.name}")
    return path.read_text(encoding="utf-8")


class TestRetirementQueueSourceSync:
    """Guard: hardcoded RETIREMENT_TIER1 / TIER2 / HOLD / BETTER must each
    appear in the retirement-queue committee-action source doc.

    Direction of assertion: every hardcoded lane MUST appear in the source
    (drift-into-script detection). Does NOT enforce the reverse direction
    because the source doc may list lanes the committee decided to carve
    out of the consolidated view. Reverse-direction enforcement would be
    too strict."""

    def test_tier1_lanes_present_in_source(self) -> None:
        text = _doc_text(RETIREMENT_DOC)
        for lane in RETIREMENT_TIER1:
            assert lane in text, (
                f"RETIREMENT_TIER1 lane `{lane}` not found in "
                f"{RETIREMENT_DOC.name} — source doc may have been revised."
            )

    def test_tier1_urgent_is_subset_of_tier1(self) -> None:
        assert RETIREMENT_TIER1_URGENT <= RETIREMENT_TIER1, (
            "RETIREMENT_TIER1_URGENT must be a subset of RETIREMENT_TIER1 — "
            "urgent implies the lane is already in Tier-1."
        )

    def test_tier2_lanes_present_in_source(self) -> None:
        text = _doc_text(RETIREMENT_DOC)
        for lane in RETIREMENT_TIER2:
            assert lane in text, (
                f"RETIREMENT_TIER2 lane `{lane}` not found in {RETIREMENT_DOC.name}"
            )

    def test_hold_lanes_present_in_source(self) -> None:
        text = _doc_text(RETIREMENT_DOC)
        for lane in RETIREMENT_HOLD:
            assert lane in text, (
                f"RETIREMENT_HOLD lane `{lane}` not found in {RETIREMENT_DOC.name}"
            )

    def test_better_lanes_present_in_source(self) -> None:
        text = _doc_text(RETIREMENT_DOC)
        for lane in RETIREMENT_BETTER:
            assert lane in text, (
                f"RETIREMENT_BETTER lane `{lane}` not found in {RETIREMENT_DOC.name}"
            )

    def test_categories_are_disjoint(self) -> None:
        """A lane can only be in one retirement category at a time —
        doctrine does not allow a lane to be simultaneously Tier-1 DECAY
        and BETTER-THAN-PEERS, etc."""
        categories = {
            "TIER1": RETIREMENT_TIER1,
            "TIER2": RETIREMENT_TIER2,
            "HOLD": RETIREMENT_HOLD,
            "BETTER": RETIREMENT_BETTER,
        }
        names = list(categories.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                overlap = categories[n1] & categories[n2]
                assert not overlap, (
                    f"Retirement categories {n1} and {n2} overlap on "
                    f"{overlap} — exactly one category per lane"
                )


class TestFireRateSourceSync:
    def test_over95_lanes_present_in_source(self) -> None:
        text = _doc_text(FIRE_RATE_DOC)
        for lane in FIRE_RATE_OVER95:
            assert lane in text, (
                f"FIRE_RATE_OVER95 lane `{lane}` not found in {FIRE_RATE_DOC.name}"
            )

    def test_arithmetic_only_lanes_present_in_source(self) -> None:
        text = _doc_text(FIRE_RATE_DOC)
        for lane in FIRE_RATE_ARITHMETIC_ONLY:
            assert lane in text, (
                f"FIRE_RATE_ARITHMETIC_ONLY lane `{lane}` not found in "
                f"{FIRE_RATE_DOC.name}"
            )

    def test_fire_rate_percentages_are_above_95(self) -> None:
        """Sanity: every FIRE_RATE_OVER95 entry must actually be > 95%."""
        for lane, pct in FIRE_RATE_OVER95.items():
            assert pct > 95.0, (
                f"FIRE_RATE_OVER95[{lane}] = {pct} is not > 95 — entry should "
                f"not be in this set unless fire-rate actually exceeds 95%"
            )

    def test_zero_injection_artifact_lanes_use_x_mes_atr60(self) -> None:
        """Sanity: the 'fires 0%' artifact class is specifically the
        cross-asset ATR filter lanes."""
        for lane in FIRE_RATE_ZERO_INJECTION_ARTIFACT:
            assert "X_MES_ATR60" in lane, (
                f"FIRE_RATE_ZERO_INJECTION_ARTIFACT entry `{lane}` is not an "
                f"X_MES_ATR60 lane — the artifact class is specifically about "
                f"cross-asset ATR lanes that need _inject_cross_asset_atrs "
                f"pre-processing."
            )


class TestSGPJaccardSourceSync:
    def test_pair_has_exactly_two_entries(self) -> None:
        """SGP Jaccard finding is a two-lane pair by construction."""
        assert len(SGP_JACCARD_PAIR) == 2

    def test_pair_is_o15_plus_o30(self) -> None:
        """Pair must be one O15 and one O30 lane."""
        suffixes = {lane[-4:] for lane in SGP_JACCARD_PAIR}
        assert suffixes == {"_O15", "_O30"}, (
            f"SGP_JACCARD_PAIR suffixes {suffixes} should be exactly "
            f"{{'_O15', '_O30'}}"
        )

    def test_pair_both_present_in_source(self) -> None:
        text = _doc_text(SGP_JACCARD_DOC)
        for lane in SGP_JACCARD_PAIR:
            assert lane in text, (
                f"SGP_JACCARD_PAIR lane `{lane}` not found in "
                f"{SGP_JACCARD_DOC.name}"
            )


# --- Verdict assignment unit tests -------------------------------------


def _rv(
    strategy_id: str,
    *,
    mode_a_n: int = 500,
    mode_a_expr: float | None = 0.10,
    mode_a_sd: float | None = 0.9,
    mode_a_sharpe: float | None = 1.0,
    c4_t_stat: float | None = 4.0,
    c4_pass_with_theory: bool | None = True,
    c4_pass_no_theory: bool | None = True,
    c7_pass: bool | None = True,
    c9_pass: bool | None = True,
    c9_violating_eras: list[str] | None = None,
    mode_b_contaminated: bool = False,
    stored_last_trade_day=None,
) -> LaneRevalidation:
    """Build a LaneRevalidation with verdict-relevant fields pre-populated.
    Defaults represent a lane that would otherwise be KEEP."""
    return LaneRevalidation(
        strategy_id=strategy_id,
        instrument=strategy_id.split("_")[0],
        orb_label="NYSE_OPEN",
        orb_minutes=5,
        rr_target=1.5,
        entry_model="E2",
        confirm_bars=1,
        filter_type="ORB_G5",
        direction="long",
        mode_a_n=mode_a_n,
        mode_a_expr=mode_a_expr,
        mode_a_sd=mode_a_sd,
        mode_a_sharpe=mode_a_sharpe,
        c4_t_stat=c4_t_stat,
        c4_pass_with_theory=c4_pass_with_theory,
        c4_pass_no_theory=c4_pass_no_theory,
        c7_pass=c7_pass,
        c9_pass=c9_pass,
        c9_violating_eras=c9_violating_eras or [],
        mode_b_contaminated=mode_b_contaminated,
        stored_last_trade_day=stored_last_trade_day,
    )


class TestAssignVerdictPriority:
    def test_retire_urgent_for_tier1_urgent_lane(self) -> None:
        lane_id = next(iter(RETIREMENT_TIER1_URGENT))
        rv = _rv(lane_id)
        verdict, reasons = assign_verdict(rv)
        assert verdict == "RETIRE_URGENT"
        assert any("NEGATIVE late Sharpe" in r for r in reasons)

    def test_retire_standard_for_tier1_non_urgent_lane(self) -> None:
        lane_id = next(iter(RETIREMENT_TIER1 - RETIREMENT_TIER1_URGENT))
        rv = _rv(lane_id)
        verdict, _ = assign_verdict(rv)
        assert verdict == "RETIRE_STANDARD"

    def test_n_underpowered_beats_reclassify_cost(self) -> None:
        """C7 fail takes precedence over fire-rate-over-95 because
        an underpowered lane can't deploy under ANY doctrine, whereas
        reclassifying to cost-screen is only valid if N is sufficient."""
        over95_lane = next(iter(FIRE_RATE_OVER95))
        rv = _rv(over95_lane, mode_a_n=88, c7_pass=False)
        verdict, _ = assign_verdict(rv)
        assert verdict == "N_UNDERPOWERED"

    def test_reclassify_cost_for_over95_fire_rate(self) -> None:
        lane_id = next(iter(FIRE_RATE_OVER95))
        rv = _rv(lane_id)
        verdict, reasons = assign_verdict(rv)
        assert verdict == "RECLASSIFY_COST"
        assert any("fire-rate" in r for r in reasons)

    def test_reclassify_cost_for_arithmetic_only(self) -> None:
        # Pick an arithmetic_only lane that is NOT also in over95 set
        candidates = FIRE_RATE_ARITHMETIC_ONLY - set(FIRE_RATE_OVER95.keys())
        assert candidates, "need an arithmetic_only lane not in over95 set"
        lane_id = next(iter(candidates))
        rv = _rv(lane_id)
        verdict, reasons = assign_verdict(rv)
        assert verdict == "RECLASSIFY_COST"
        assert any("arithmetic_only" in r for r in reasons)

    def test_review_tier2_for_tier2_lane(self) -> None:
        candidates = RETIREMENT_TIER2 - set(FIRE_RATE_OVER95.keys()) - FIRE_RATE_ARITHMETIC_ONLY
        assert candidates, (
            "test invariant: need a Tier-2 lane that doesn't also hit a "
            "higher-priority verdict"
        )
        lane_id = next(iter(candidates))
        rv = _rv(lane_id)
        verdict, _ = assign_verdict(rv)
        assert verdict == "REVIEW_TIER2"

    def test_review_capacity_for_sgp_jaccard_pair(self) -> None:
        candidates = SGP_JACCARD_PAIR - set(FIRE_RATE_OVER95.keys()) - FIRE_RATE_ARITHMETIC_ONLY - RETIREMENT_TIER1 - RETIREMENT_TIER2
        if not candidates:
            pytest.skip(
                "SGP pair fully masked by higher-priority flags — test not "
                "meaningful on current data"
            )
        lane_id = next(iter(candidates))
        rv = _rv(lane_id)
        verdict, _ = assign_verdict(rv)
        assert verdict == "REVIEW_CAPACITY"

    def test_better_than_peers_for_better_lane(self) -> None:
        candidates = (
            RETIREMENT_BETTER
            - set(FIRE_RATE_OVER95.keys())
            - FIRE_RATE_ARITHMETIC_ONLY
            - RETIREMENT_TIER1
            - RETIREMENT_TIER2
            - SGP_JACCARD_PAIR
        )
        if not candidates:
            pytest.skip(
                "BETTER lanes fully masked by higher-priority flags — test "
                "not meaningful on current data"
            )
        lane_id = next(iter(candidates))
        rv = _rv(lane_id)
        verdict, _ = assign_verdict(rv)
        assert verdict == "BETTER_THAN_PEERS"

    def test_review_c4_wt_fail_for_low_t_lane_not_in_any_set(self) -> None:
        rv = _rv(
            "MNQ_NONEXISTENT_LANE_ID",  # not in any hardcoded set
            c4_t_stat=2.5,
            c4_pass_with_theory=False,
            c4_pass_no_theory=False,
        )
        verdict, _ = assign_verdict(rv)
        assert verdict == "REVIEW_C4_WT_FAIL"

    def test_keep_for_clean_lane_not_in_any_set(self) -> None:
        rv = _rv("MNQ_NONEXISTENT_LANE_ID_CLEAN")
        verdict, reasons = assign_verdict(rv)
        assert verdict == "KEEP"

    def test_c9_fail_appears_as_secondary_reason(self) -> None:
        """A lane's primary verdict may not be C9, but c9_pass=False must
        still appear in reasons as supplementary evidence."""
        rv = _rv(
            "MNQ_CLEAN_EXCEPT_C9",
            c9_pass=False,
            c9_violating_eras=["2023"],
        )
        _, reasons = assign_verdict(rv)
        assert any("C9" in r for r in reasons), (
            f"C9 fail must be in reasons; got: {reasons}"
        )

    def test_mode_b_contamination_is_secondary_reason(self) -> None:
        from datetime import date
        rv = _rv(
            "MNQ_CLEAN_EXCEPT_MODE_B",
            mode_b_contaminated=True,
            stored_last_trade_day=date(2026, 3, 15),
        )
        _, reasons = assign_verdict(rv)
        assert any("Mode-B grandfathered" in r for r in reasons)

    def test_injection_artifact_is_secondary_reason(self) -> None:
        lane_id = next(iter(FIRE_RATE_ZERO_INJECTION_ARTIFACT))
        rv = _rv(lane_id)
        _, reasons = assign_verdict(rv)
        assert any("artifact" in r.lower() for r in reasons), (
            f"Cross-asset injection artifact must be flagged; got: {reasons}"
        )


class TestVerdictOrderCompleteness:
    def test_every_verdict_code_has_unique_position(self) -> None:
        assert len(VERDICT_ORDER) == len(set(VERDICT_ORDER))

    def test_retire_precedes_review(self) -> None:
        """Sanity: RETIRE_* codes must come before REVIEW_* codes in the
        priority order so committee sees retire candidates first."""
        retire_idx = [i for i, v in enumerate(VERDICT_ORDER) if v.startswith("RETIRE")]
        review_idx = [i for i, v in enumerate(VERDICT_ORDER) if v.startswith("REVIEW")]
        assert retire_idx and review_idx
        assert max(retire_idx) < min(review_idx), (
            f"RETIRE_* codes {retire_idx} must all come before REVIEW_* codes "
            f"{review_idx} in VERDICT_ORDER"
        )

    def test_keep_is_last(self) -> None:
        assert VERDICT_ORDER[-1] == "KEEP", (
            "KEEP is the null-action verdict; must be last so that "
            "render-by-severity lists keep-lanes at the bottom"
        )


class TestPhase25SubsetTSetsMatchCSV:
    """Drift detection: Phase 2.5 hardcoded sets must match the live CSV."""

    @staticmethod
    def _load_csv():
        if not PHASE_2_5_CSV.exists():
            pytest.skip(f"{PHASE_2_5_CSV} not present (run phase_2_5 sweep first)")
        import csv
        with PHASE_2_5_CSV.open() as f:
            return list(csv.DictReader(f))

    def test_tier1_pass_matches_csv(self):
        """Lanes with primary_flag=PASS and subset_t>=3.00 must equal SUBSET_T_TIER1_PASS."""
        rows = self._load_csv()
        csv_tier1 = {
            r["strategy_id"] for r in rows
            if r["primary_flag"] == "PASS"
            and r["subset_t"] not in ("", None)
            and abs(float(r["subset_t"])) >= 3.00
        }
        assert csv_tier1 == SUBSET_T_TIER1_PASS, (
            f"Hardcoded SUBSET_T_TIER1_PASS drifted from CSV.\n"
            f"Only in CSV: {csv_tier1 - SUBSET_T_TIER1_PASS}\n"
            f"Only in hardcoded: {SUBSET_T_TIER1_PASS - csv_tier1}"
        )

    def test_tier4_fails_conventional_matches_csv(self):
        """Lanes with |subset_t|<1.96 must equal SUBSET_T_TIER4_FAIL_CONVENTIONAL."""
        rows = self._load_csv()
        csv_tier4 = {
            r["strategy_id"] for r in rows
            if r["subset_t"] not in ("", None) and abs(float(r["subset_t"])) < 1.96
        }
        assert csv_tier4 == SUBSET_T_TIER4_FAIL_CONVENTIONAL, (
            f"Hardcoded SUBSET_T_TIER4_FAIL_CONVENTIONAL drifted from CSV.\n"
            f"Only in CSV: {csv_tier4 - SUBSET_T_TIER4_FAIL_CONVENTIONAL}\n"
            f"Only in hardcoded: {SUBSET_T_TIER4_FAIL_CONVENTIONAL - csv_tier4}"
        )

    def test_arithmetic_lift_matches_csv(self):
        """Lanes with primary_flag=ARITHMETIC_LIFT must equal SUBSET_T_ARITHMETIC_LIFT."""
        rows = self._load_csv()
        csv_lift = {
            r["strategy_id"] for r in rows
            if r["primary_flag"] == "ARITHMETIC_LIFT"
        }
        assert csv_lift == SUBSET_T_ARITHMETIC_LIFT, (
            f"Hardcoded SUBSET_T_ARITHMETIC_LIFT drifted from CSV.\n"
            f"Only in CSV: {csv_lift - SUBSET_T_ARITHMETIC_LIFT}\n"
            f"Only in hardcoded: {SUBSET_T_ARITHMETIC_LIFT - csv_lift}"
        )

    def test_tiers_are_disjoint(self):
        """No lane can appear in multiple mutually-exclusive tier sets."""
        all_sets = {
            "TIER1_PASS": SUBSET_T_TIER1_PASS,
            "TIER1_THIN_N": SUBSET_T_TIER1_THIN_N,
            "TIER2": SUBSET_T_TIER2,
            "TIER4": SUBSET_T_TIER4_FAIL_CONVENTIONAL,
        }
        for name_a, set_a in all_sets.items():
            for name_b, set_b in all_sets.items():
                if name_a >= name_b:
                    continue
                overlap = set_a & set_b
                assert not overlap, f"{name_a} and {name_b} overlap: {overlap}"


class TestPhase25Annotation:
    """Subset-t annotation returns correct string per tier."""

    def test_tier1_pass_annotation(self):
        for sid in SUBSET_T_TIER1_PASS:
            assert subset_t_annotation(sid) == "Tier 1 (t>=3.00 Chordia PASS)"

    def test_tier1_thin_n_annotation(self):
        for sid in SUBSET_T_TIER1_THIN_N:
            assert subset_t_annotation(sid) == "Tier 1 thin-N (t>=3.00, N<100)"

    def test_tier2_annotation(self):
        # Exclude ARITHMETIC_LIFT-flagged lanes (which take priority per doctrine).
        # A Tier-2 lane that also has Rule 8.3 violation should render as ARITHMETIC_LIFT.
        for sid in SUBSET_T_TIER2 - SUBSET_T_ARITHMETIC_LIFT:
            assert subset_t_annotation(sid) == "Tier 2 (t in [2.58, 3.00))"

    def test_tier4_annotation(self):
        for sid in SUBSET_T_TIER4_FAIL_CONVENTIONAL:
            assert subset_t_annotation(sid) == "Tier 4 (t<1.96 — fails p<0.05)"

    def test_arithmetic_lift_annotation_takes_priority(self):
        for sid in SUBSET_T_ARITHMETIC_LIFT:
            assert subset_t_annotation(sid) == "Rule 8.3 ARITHMETIC_LIFT (retire/reframe)"

    def test_unknown_lane_defaults_to_tier3(self):
        assert subset_t_annotation("UNKNOWN_LANE_ID_THAT_DOES_NOT_EXIST") == "Tier 3 (t in [1.96, 2.58))"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
