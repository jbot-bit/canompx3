"""Unit tests for the criterion-evaluation helpers in
``research/mode_a_revalidation_active_setups.py``.

These cover pure functions only (no DB). The integration path (load_active_setups,
compute_mode_a) is exercised by running the script end-to-end against
``gold.db``; see ``docs/audit/results/2026-04-19-mode-a-criterion-evaluation.md``
for the integration output.

Covered:
- ``compute_criterion_flags`` applies pre_registered_criteria.md § 4, 7, 9
  correctly for representative input combinations
- None inputs produce None flags (skipped, not False) so downstream rendering
  can distinguish "not evaluated" from "failed"
- Locked threshold constants match pre_registered_criteria.md verbatim — guard
  against accidental tuning per ``integrity-guardian.md`` § 8.

Not covered (require DB integration test):
- ``compute_mode_a`` full SQL pipeline
- ``load_active_setups`` query
- ``render`` full markdown pipeline (exercised by running the script)
- CrossAssetATRFilter injection branch (exercised by integration run)
"""

from __future__ import annotations

import pytest

from research.mode_a_revalidation_active_setups import (
    C4_T_NO_THEORY,
    C4_T_WITH_THEORY,
    C7_MIN_N,
    C9_ERA_THRESHOLD,
    C9_MIN_N_PER_ERA,
    LaneRevalidation,
    compute_criterion_flags,
)


def _lane(
    *,
    mode_a_n: int = 0,
    mode_a_expr: float | None = None,
    mode_a_sd: float | None = None,
    years_breakdown: dict | None = None,
) -> LaneRevalidation:
    return LaneRevalidation(
        strategy_id="TEST",
        instrument="MNQ",
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
        years_breakdown=years_breakdown or {},
    )


class TestLockedThresholds:
    """Guard against accidental threshold tuning. Constants MUST match
    pre_registered_criteria.md § Criterion 4, 7, 9 verbatim. Any change
    requires an amendment per that file's § Amendment procedure."""

    def test_c4_thresholds_match_doctrine(self) -> None:
        assert C4_T_WITH_THEORY == 3.00
        assert C4_T_NO_THEORY == 3.79

    def test_c7_threshold_matches_doctrine(self) -> None:
        assert C7_MIN_N == 100

    def test_c9_thresholds_match_doctrine(self) -> None:
        assert C9_ERA_THRESHOLD == -0.05
        assert C9_MIN_N_PER_ERA == 50


class TestCriterion4ChordiaT:
    def test_passes_with_theory_at_exactly_3(self) -> None:
        # t = ExpR / (sd/sqrt(N)) = 0.06 / (0.9 / sqrt(N)) = 3.00 → N = (3*0.9/0.06)^2 = 2025
        rv = _lane(mode_a_n=2025, mode_a_expr=0.06, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is not None
        assert abs(rv.c4_t_stat - 3.0) < 0.01
        assert rv.c4_pass_with_theory is True
        # 3.00 < 3.79 → no-theory fails
        assert rv.c4_pass_no_theory is False

    def test_fails_both_thresholds_at_low_t(self) -> None:
        # t = 0.05 * sqrt(100) / 0.9 = 0.556
        rv = _lane(mode_a_n=100, mode_a_expr=0.05, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c4_pass_with_theory is False
        assert rv.c4_pass_no_theory is False
        assert any("C4" in f for f in rv.criterion_failures)

    def test_passes_no_theory_at_3_79(self) -> None:
        # t = ExpR * sqrt(N) / sd >= 3.79 → sqrt(N) >= 3.79 * 0.9 / 0.06 = 56.85 → N >= 3233
        # Use N=3240 for safety margin against float rounding
        rv = _lane(mode_a_n=3240, mode_a_expr=0.06, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is not None
        assert rv.c4_t_stat >= C4_T_NO_THEORY
        assert rv.c4_pass_with_theory is True
        assert rv.c4_pass_no_theory is True

    def test_missing_sd_leaves_t_none(self) -> None:
        rv = _lane(mode_a_n=500, mode_a_expr=0.1, mode_a_sd=None)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is None
        assert rv.c4_pass_with_theory is None
        assert rv.c4_pass_no_theory is None

    def test_zero_n_leaves_t_none(self) -> None:
        rv = _lane(mode_a_n=0, mode_a_expr=None, mode_a_sd=None)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is None
        assert rv.c4_pass_with_theory is None
        assert rv.c4_pass_no_theory is None

    def test_zero_sd_leaves_t_none(self) -> None:
        """Degenerate: if all trades have the same pnl_r, sd=0 and t is undefined.
        Must not divide by zero."""
        rv = _lane(mode_a_n=100, mode_a_expr=0.1, mode_a_sd=0.0)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is None

    def test_negative_expr_produces_negative_t(self) -> None:
        """A lane with negative Mode A ExpR should get negative t and fail both
        thresholds (sign matters; absolute value is not what the criterion
        checks, per pre_registered_criteria.md § Criterion 4 which requires
        t ≥ +3.00 not |t|)."""
        rv = _lane(mode_a_n=2025, mode_a_expr=-0.06, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c4_t_stat is not None
        assert rv.c4_t_stat < 0
        assert rv.c4_pass_with_theory is False
        assert rv.c4_pass_no_theory is False


class TestCriterion7SampleSize:
    def test_passes_at_exactly_100(self) -> None:
        rv = _lane(mode_a_n=100, mode_a_expr=0.1, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c7_pass is True

    def test_fails_below_100(self) -> None:
        # Real-world case from 2026-04-19 revalidation: MES_CME_PRECLOSE COST_LT08 at N=88
        rv = _lane(mode_a_n=88, mode_a_expr=0.3, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c7_pass is False
        assert any("C7" in f and "88" in f for f in rv.criterion_failures)

    def test_passes_large_n(self) -> None:
        rv = _lane(mode_a_n=1000, mode_a_expr=0.1, mode_a_sd=0.9)
        compute_criterion_flags(rv)
        assert rv.c7_pass is True

    def test_zero_n_leaves_c7_none(self) -> None:
        """N=0 means no Mode A trades survived — skip, don't report FAIL,
        because the divergence-flagging layer handles 'no data' cases
        separately."""
        rv = _lane(mode_a_n=0)
        compute_criterion_flags(rv)
        assert rv.c7_pass is None


class TestCriterion9EraStability:
    def test_c9_uses_era_aggregation_not_individual_years(self) -> None:
        """REGRESSION (2026-04-19 adversarial-review catch): C9 must aggregate
        years into doctrine-specified eras per pre_registered_criteria.md §
        Criterion 9: "era-split into (2015-2019, 2020-2022, 2023, 2024-2025,
        2026)". Year-level flagging is STRICTER than doctrine — it flags a
        single bad year inside an era whose AGGREGATE ExpR is ≥ -0.05.
        """
        yrs = {
            # Era 2020-2022: one bad year, but era-aggregate passes
            2020: {"n": 100, "expr": -0.10, "positive": False},
            2021: {"n": 100, "expr": +0.15, "positive": True},
            2022: {"n": 100, "expr": +0.10, "positive": True},
            # Aggregate: (100*-0.10 + 100*+0.15 + 100*+0.10) / 300 = +0.05
            # which is NOT < -0.05, so era passes per doctrine.
        }
        rv = _lane(mode_a_n=300, mode_a_expr=0.05, mode_a_sd=0.9,
                   years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is True, (
            "C9 must aggregate years into doctrine eras; per-year flagging "
            "is stricter than doctrine specifies"
        )

    def test_c9_fails_when_era_aggregate_below_threshold(self) -> None:
        """C9 violates when an ERA's aggregate ExpR is < -0.05 AND era-N ≥ 50."""
        yrs = {
            # Era 2020-2022: all three years negative, aggregate violates
            2020: {"n": 100, "expr": -0.08, "positive": False},
            2021: {"n": 100, "expr": -0.06, "positive": False},
            2022: {"n": 100, "expr": -0.07, "positive": False},
            # Aggregate: -0.07 < -0.05 → era violates
        }
        rv = _lane(mode_a_n=300, mode_a_expr=-0.07, mode_a_sd=0.9,
                   years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is False

    def test_c9_exempts_small_era_aggregate(self) -> None:
        """An era with aggregate N < 50 is exempt (not enough data to judge)."""
        yrs = {
            2023: {"n": 40, "expr": -0.20, "positive": False},
            # Era 2023 = single year 2023, N=40 < 50 → exempt
        }
        rv = _lane(mode_a_n=40, mode_a_expr=-0.20, mode_a_sd=0.9,
                   years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is True, "Era 2023 with N<50 must be exempt"

    def test_c9_reports_era_names_not_year_ints(self) -> None:
        """c9_violating_eras must contain era NAMES (e.g. '2020-2022') not
        individual year integers — downstream rendering depends on this."""
        yrs = {2023: {"n": 100, "expr": -0.10, "positive": False}}
        rv = _lane(mode_a_n=100, mode_a_expr=-0.10, mode_a_sd=0.9,
                   years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is False
        assert rv.c9_violating_eras == ["2023"], (
            f"expected era name string, got {rv.c9_violating_eras}"
        )

    def test_passes_all_positive_years(self) -> None:
        yrs = {
            2019: {"n": 100, "expr": 0.1, "positive": True},   # era 2015-2019
            2020: {"n": 120, "expr": 0.15, "positive": True},  # era 2020-2022
            2021: {"n": 90, "expr": 0.08, "positive": True},   # era 2020-2022
        }
        rv = _lane(mode_a_n=310, mode_a_expr=0.11, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is True
        assert rv.c9_violating_eras == []

    def test_fails_on_bad_era_with_sufficient_n(self) -> None:
        """Era 2020-2022 has only 2020 (N=80, expr=-0.06). Era aggregate equals
        the single year → -0.06 < -0.05, era-N=80 ≥ 50, violates C9."""
        yrs = {
            2019: {"n": 100, "expr": 0.1, "positive": True},
            2020: {"n": 80, "expr": -0.06, "positive": False},
        }
        rv = _lane(mode_a_n=180, mode_a_expr=0.02, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is False
        assert rv.c9_violating_eras == ["2020-2022"]
        assert any("C9" in f for f in rv.criterion_failures)

    def test_exempts_small_n_era(self) -> None:
        """An era whose aggregate N < 50 is exempt — not enough data to judge."""
        yrs = {
            2019: {"n": 100, "expr": 0.1, "positive": True},
            # era 2020-2022 aggregate has only 2020 (N=30 < 50) → exempt
            2020: {"n": 30, "expr": -0.20, "positive": False},
        }
        rv = _lane(mode_a_n=130, mode_a_expr=0.05, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is True
        assert rv.c9_violating_eras == []

    def test_at_exactly_threshold_does_not_violate(self) -> None:
        """era aggregate ExpR = -0.05 exactly is NOT a violation (criterion is "< -0.05")."""
        yrs = {2020: {"n": 100, "expr": -0.05, "positive": False}}
        rv = _lane(mode_a_n=100, mode_a_expr=-0.05, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is True
        assert rv.c9_violating_eras == []

    def test_at_exactly_min_n_evaluates(self) -> None:
        """Era-aggregate N = 50 exactly IS evaluated (criterion is "N ≥ 50")."""
        yrs = {2020: {"n": 50, "expr": -0.10, "positive": False}}
        rv = _lane(mode_a_n=50, mode_a_expr=-0.10, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is False
        assert rv.c9_violating_eras == ["2020-2022"]

    def test_multiple_violating_eras_all_reported(self) -> None:
        """Multiple eras violate: era 2020-2022 aggregate AND era 2023 both bad."""
        yrs = {
            2019: {"n": 100, "expr": 0.1, "positive": True},
            # era 2020-2022 aggregate: (80*-0.07 + 90*-0.06)/170 = -0.0647 < -0.05 → violates
            2020: {"n": 80, "expr": -0.07, "positive": False},
            2021: {"n": 90, "expr": -0.06, "positive": False},
            # era 2023 single-year violates
            2023: {"n": 60, "expr": -0.08, "positive": False},
        }
        rv = _lane(mode_a_n=330, mode_a_expr=-0.04, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c9_pass is False
        # Era names appear in doctrine order, not alphabetical
        assert rv.c9_violating_eras == ["2020-2022", "2023"]

    def test_empty_year_breakdown_skips_c9(self) -> None:
        rv = _lane(mode_a_n=100, mode_a_expr=0.1, mode_a_sd=0.9, years_breakdown={})
        compute_criterion_flags(rv)
        assert rv.c9_pass is None
        assert rv.c9_violating_eras == []


class TestCombinedFailures:
    def test_multiple_criterion_failures_accumulate(self) -> None:
        """A lane can fail C4, C7, and C9 simultaneously. The failure list
        must include all three so the retirement view prioritises correctly."""
        yrs = {2020: {"n": 60, "expr": -0.10, "positive": False}}
        rv = _lane(mode_a_n=50, mode_a_expr=0.01, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        # t = 0.01 * sqrt(50) / 0.9 ≈ 0.079 — fails C4
        assert rv.c4_pass_with_theory is False
        # N=50 < 100 — fails C7
        assert rv.c7_pass is False
        # 2020 bad era — fails C9
        assert rv.c9_pass is False
        assert len(rv.criterion_failures) == 3

    def test_clean_lane_has_no_criterion_failures(self) -> None:
        yrs = {
            2019: {"n": 100, "expr": 0.1, "positive": True},
            2020: {"n": 120, "expr": 0.15, "positive": True},
        }
        rv = _lane(mode_a_n=2025, mode_a_expr=0.06, mode_a_sd=0.9, years_breakdown=yrs)
        compute_criterion_flags(rv)
        assert rv.c4_pass_with_theory is True
        assert rv.c7_pass is True
        assert rv.c9_pass is True
        assert rv.criterion_failures == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
