"""Unit tests for the cherry-pick ranker.

Covers score-formula components individually, the aggregator, edge cases
(empty queue, REVOKED entries, missing OOS), and end-to-end ranking on a
synthetic queue that mirrors the real schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.research import cherry_pick_ranker as cpr


# ----- Component-level tests -----


class TestDeflationHeadroom:
    def test_zero_when_below_threshold(self):
        assert cpr.compute_deflation_headroom(3.06) == 0.0

    def test_zero_when_equal_to_threshold(self):
        assert cpr.compute_deflation_headroom(cpr.HEAVYWEIGHT_T_THRESHOLD) == 0.0

    def test_positive_when_above_threshold(self):
        # t=4.79: headroom = 1.0/4.79 ~ 0.2087
        val = cpr.compute_deflation_headroom(4.79)
        assert 0.20 < val < 0.22

    def test_zero_on_nan(self):
        assert cpr.compute_deflation_headroom(float("nan")) == 0.0


class TestNAdequacy:
    def test_zero_when_n_zero(self):
        assert cpr.compute_n_adequacy(0) == 0.0

    def test_linear_below_target(self):
        # N=100, target=200 -> 0.5
        assert cpr.compute_n_adequacy(100) == pytest.approx(0.5, abs=1e-6)

    def test_capped_at_one(self):
        assert cpr.compute_n_adequacy(cpr.N_ADEQUACY_TARGET * 10) == 1.0

    def test_at_target(self):
        assert cpr.compute_n_adequacy(cpr.N_ADEQUACY_TARGET) == 1.0


class TestDirMatch:
    def test_same_sign_positive(self):
        oos = cpr.OOSStats(n_oos=40, expr_oos=0.05, t_oos=1.5)
        assert cpr.compute_dir_match(0.17, oos) == 1.0

    def test_same_sign_negative(self):
        oos = cpr.OOSStats(n_oos=40, expr_oos=-0.05, t_oos=-1.5)
        assert cpr.compute_dir_match(-0.17, oos) == 1.0

    def test_opposite_sign(self):
        oos = cpr.OOSStats(n_oos=40, expr_oos=-0.02, t_oos=-0.5)
        assert cpr.compute_dir_match(0.17, oos) == 0.0

    def test_zero_on_no_oos(self):
        assert cpr.compute_dir_match(0.17, None) == 0.0

    def test_zero_on_nan(self):
        oos = cpr.OOSStats(n_oos=40, expr_oos=float("nan"), t_oos=float("nan"))
        assert cpr.compute_dir_match(0.17, oos) == 0.0


class TestNonArtifact:
    def test_one_when_not_artifact(self):
        assert cpr.compute_non_artifact(False) == 1.0

    def test_zero_when_artifact(self):
        assert cpr.compute_non_artifact(True) == 0.0


class TestOOSPowerReadiness:
    def test_zero_when_no_oos(self):
        assert cpr.compute_oos_power_readiness(0.17, 3.06, 226, None) == 0.0

    def test_zero_when_n_oos_below_floor(self):
        oos = cpr.OOSStats(n_oos=cpr.OOS_N_FLOOR - 1, expr_oos=0.05, t_oos=1.5)
        assert cpr.compute_oos_power_readiness(0.17, 3.06, 226, oos) == 0.0

    def test_zero_when_pooled_t_zero(self):
        oos = cpr.OOSStats(n_oos=50, expr_oos=0.05, t_oos=1.5)
        assert cpr.compute_oos_power_readiness(0.0, 0.0, 226, oos) == 0.0

    def test_bounded_to_unit_interval(self):
        oos = cpr.OOSStats(n_oos=100, expr_oos=0.05, t_oos=1.5)
        val = cpr.compute_oos_power_readiness(0.17, 3.06, 226, oos)
        assert 0.0 <= val <= 1.0

    def test_matches_canonical_one_sample_helper(self):
        """Audit-fix regression: ranker power must equal research.oos_power.one_sample_power.

        Per evidence-auditor finding (2026-05-19), the ranker was using
        ``oos_ttest_power`` two-sample helper with IS N passed as a second
        OOS group, inflating power. Fix delegates to ``one_sample_power``;
        this test pins the delegation.
        """
        from research.oos_power import one_sample_power

        pooled_t = 3.06
        pooled_n = 226
        oos_n = 100
        cohen_d = abs(pooled_t) / (pooled_n**0.5)
        expected = float(one_sample_power(cohen_d, oos_n, alpha=0.05))
        oos = cpr.OOSStats(n_oos=oos_n, expr_oos=0.05, t_oos=1.5)
        actual = cpr.compute_oos_power_readiness(0.17, pooled_t, pooled_n, oos)
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_no_inflation_at_large_IS_N_small_OOS_N(self):
        """Audit-fix regression: large IS N must NOT inflate OOS power.

        Pre-fix: passing IS N=500 as n_oos_a gave df=500+35-2=533 and a
        non-central t with ncp scaled by sqrt((500*35)/(500+35)), yielding
        spuriously high power. Post-fix: power depends only on OOS N (35)
        and cohen_d=|t|/sqrt(IS N), which is small for any pooled_t<3.79
        with N>>100, so power must stay modest.
        """
        oos = cpr.OOSStats(n_oos=35, expr_oos=0.05, t_oos=1.5)
        # IS pooled_t=3.06, N=500 -> cohen_d=3.06/sqrt(500)=0.137 (small effect).
        # One-sample power at d=0.137, n=35 is well below 0.50.
        val = cpr.compute_oos_power_readiness(0.17, 3.06, 500, oos)
        assert val < 0.50, f"large-IS-N + small-OOS-N must not inflate power; got {val}"

    def test_pooled_expr_zero_with_nonzero_t_computes_power_from_t(self):
        """Audit second-pass open item: pooled_expr=0 AND pooled_t!=0 edge case.

        Pre-fix behavior returned 0.0 (silent fail-safe). Post-fix:
        Cohen's d derives from t and N only, so the function computes a
        non-zero power consistent with what t implies. Algebraically
        incoherent on well-formed data (t = mean*sqrt(N)/std forces t=0
        when mean=0), but the new path is the honest computation. Pinned
        here so the deliberate behavior change cannot regress unnoticed.
        """
        oos = cpr.OOSStats(n_oos=100, expr_oos=0.05, t_oos=1.5)
        # pooled_expr=0 -- algebraically incoherent with pooled_t=3.06 but the
        # function must not silently zero out; it should compute power from
        # the t-stat the runner emitted.
        val = cpr.compute_oos_power_readiness(0.0, 3.06, 226, oos)
        # Same cohen_d as test_matches_canonical_one_sample_helper.
        from research.oos_power import one_sample_power

        expected = float(one_sample_power(abs(3.06) / (226**0.5), 100, alpha=0.05))
        assert val == pytest.approx(expected, abs=1e-9)


# ----- Score breakdown total -----


class TestScoreBreakdown:
    def test_total_uses_canonical_weights(self):
        b = cpr.ScoreBreakdown(
            deflation_headroom=1.0,
            n_adequacy=1.0,
            oos_power_readiness=1.0,
            dir_match=1.0,
            non_artifact=1.0,
            era_stability_proxy=1.0,
        )
        # All components at 1.0 must total to sum(WEIGHTS.values())
        assert b.total == pytest.approx(sum(cpr.WEIGHTS.values()), abs=1e-9)

    def test_weights_sum_to_one(self):
        assert sum(cpr.WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)

    def test_zero_total_on_zero_components(self):
        b = cpr.ScoreBreakdown(
            deflation_headroom=0.0,
            n_adequacy=0.0,
            oos_power_readiness=0.0,
            dir_match=0.0,
            non_artifact=0.0,
            era_stability_proxy=0.0,
        )
        assert b.total == 0.0


# ----- OOS row parsing -----


class TestParseOOSRow:
    def test_parses_real_format(self, tmp_path):
        # Format mirrors docs/audit/results/2026-05-18-mnq-usdata1000-*.md
        md = tmp_path / "result.md"
        md.write_text(
            "## Split summary\n\n"
            "| Split | N_universe | N_fired | Fire% | Scratch | Null | "
            "ExpR | Policy EV/opp | Sharpe | t | p_two |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            "| IS | 1539 | 226 | 14.68% | 60 | 5 | 0.1708 | 0.0251 | "
            "0.2038 | 3.064 | 0.00218 |\n"
            "| OOS | 72 | 14 | 19.44% | 4 | 0 | -0.0233 | -0.0045 | "
            "-0.0263 | -0.099 | 0.92151 |\n",
            encoding="utf-8",
        )
        oos = cpr.parse_oos_row(md)
        assert oos is not None
        assert oos.n_oos == 14
        assert oos.expr_oos == pytest.approx(-0.0233, abs=1e-6)
        assert oos.t_oos == pytest.approx(-0.099, abs=1e-6)

    def test_returns_none_when_file_missing(self, tmp_path):
        assert cpr.parse_oos_row(tmp_path / "missing.md") is None

    def test_returns_none_when_no_oos_row(self, tmp_path):
        md = tmp_path / "result.md"
        md.write_text("Some markdown with no OOS row\n", encoding="utf-8")
        assert cpr.parse_oos_row(md) is None


# ----- End-to-end ranking -----


@pytest.fixture
def synthetic_queue(tmp_path):
    """Build a synthetic promote_queue.yaml + matching result MD."""
    queue = tmp_path / "promote_queue.yaml"
    results = tmp_path / "results"
    results.mkdir()
    rmd_a = results / "2026-05-18-strat-a-fast-lane-v1.md"
    rmd_b = results / "2026-05-18-strat-b-fast-lane-v1.md"
    rmd_c = results / "2026-05-18-strat-c-fast-lane-v1.md"
    # Strategy A: strong (t=4.5, big N, OOS dir-match, OOS N>=floor)
    rmd_a.write_text(
        "## Split summary\n\n"
        "| Split | N_universe | N_fired | Fire% | Scratch | Null | "
        "ExpR | Policy EV/opp | Sharpe | t | p_two |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| IS | 2000 | 300 | 15.00% | 50 | 5 | 0.2000 | 0.0300 | "
        "0.2500 | 4.500 | 0.00001 |\n"
        "| OOS | 200 | 40 | 20.00% | 5 | 0 | 0.1500 | 0.0300 | "
        "0.1800 | 2.500 | 0.01700 |\n",
        encoding="utf-8",
    )
    # Strategy B: weak (t=3.06, OOS underpowered N=14)
    rmd_b.write_text(
        "## Split summary\n\n"
        "| Split | N_universe | N_fired | Fire% | Scratch | Null | "
        "ExpR | Policy EV/opp | Sharpe | t | p_two |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| IS | 1539 | 226 | 14.68% | 60 | 5 | 0.1708 | 0.0251 | "
        "0.2038 | 3.064 | 0.00218 |\n"
        "| OOS | 72 | 14 | 19.44% | 4 | 0 | -0.0233 | -0.0045 | "
        "-0.0263 | -0.099 | 0.92151 |\n",
        encoding="utf-8",
    )
    # Strategy C: REVOKED (must NOT appear in ranking)
    rmd_c.write_text(
        "## Split summary\n\n"
        "| Split | N_universe | N_fired | Fire% | Scratch | Null | "
        "ExpR | Policy EV/opp | Sharpe | t | p_two |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| IS | 1494 | 97 | 6.49% | 10 | 1 | 0.4676 | 0.0303 | "
        "0.3300 | 3.300 | 0.00131 |\n"
        "| OOS | 60 | 5 | 8.33% | 0 | 0 | 0.0500 | 0.0042 | "
        "0.0400 | 0.500 | 0.62000 |\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "result_md": str(rmd_a.relative_to(tmp_path)).replace("\\", "/"),
                "strategy_id": "STRAT_A_E2_RR2_PROBE_LONG",
                "direction": "long",
                "pooled_t": 4.5,
                "pooled_expr": 0.2,
                "pooled_n": 300,
                "pooling_artifact": False,
                "status": "QUEUED",
            },
            {
                "result_md": str(rmd_b.relative_to(tmp_path)).replace("\\", "/"),
                "strategy_id": "STRAT_B_E1_RR1_PD_CLEAR_LONG",
                "direction": "long",
                "pooled_t": 3.064,
                "pooled_expr": 0.1708,
                "pooled_n": 226,
                "pooling_artifact": False,
                "status": "QUEUED",
            },
            {
                "result_md": str(rmd_c.relative_to(tmp_path)).replace("\\", "/"),
                "strategy_id": "STRAT_C_E2_RR2_ORB_VOL_POOLED",
                "direction": "pooled",
                "pooled_t": 3.3,
                "pooled_expr": 0.4676,
                "pooled_n": 97,
                "pooling_artifact": True,
                "status": "REVOKED",
            },
        ],
    }
    queue.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    # Tweak entries to use absolute result paths so the ranker resolves them
    # regardless of REPO_ROOT.
    payload2 = yaml.safe_load(queue.read_text(encoding="utf-8"))
    for entry in payload2["entries"]:
        # Use the basename so rank_queue_entries falls back to results_dir
        entry["result_md"] = "results/" + Path(entry["result_md"]).name
    queue.write_text(yaml.safe_dump(payload2, sort_keys=False), encoding="utf-8")
    return queue, results


def test_revoked_entries_excluded(synthetic_queue):
    queue_path, results_dir = synthetic_queue
    entries = cpr._load_queue_entries(queue_path)
    ranked = cpr.rank_queue_entries(entries, results_dir=results_dir)
    assert len(ranked) == 2, "REVOKED entry must not be scored"
    assert all(c.strategy_id != "STRAT_C_E2_RR2_ORB_VOL_POOLED" for c in ranked)


def test_strong_candidate_ranks_above_weak(synthetic_queue):
    queue_path, results_dir = synthetic_queue
    entries = cpr._load_queue_entries(queue_path)
    ranked = cpr.rank_queue_entries(entries, results_dir=results_dir)
    assert ranked[0].strategy_id == "STRAT_A_E2_RR2_PROBE_LONG"
    assert ranked[1].strategy_id == "STRAT_B_E1_RR1_PD_CLEAR_LONG"
    # Strong candidate must have positive deflation_headroom (t=4.5 > 3.79)
    assert ranked[0].score.deflation_headroom > 0
    # Weak candidate must have zero deflation_headroom (t=3.06 < 3.79)
    assert ranked[1].score.deflation_headroom == 0.0


def test_dir_match_flag_set_correctly(synthetic_queue):
    queue_path, results_dir = synthetic_queue
    entries = cpr._load_queue_entries(queue_path)
    ranked = cpr.rank_queue_entries(entries, results_dir=results_dir)
    by_id = {c.strategy_id: c for c in ranked}
    # A: IS +0.2, OOS +0.15 -> same sign
    assert by_id["STRAT_A_E2_RR2_PROBE_LONG"].dir_match is True
    # B: IS +0.17, OOS -0.02 -> opposite sign
    assert by_id["STRAT_B_E1_RR1_PD_CLEAR_LONG"].dir_match is False


def test_skip_recommended_below_floor(synthetic_queue):
    queue_path, results_dir = synthetic_queue
    entries = cpr._load_queue_entries(queue_path)
    ranked = cpr.rank_queue_entries(entries, results_dir=results_dir)
    # Strategy B should be flagged for skip (deflation=0, dir_match=0,
    # OOS power=0 due to N=14<floor) -- composite score below 0.40
    by_id = {c.strategy_id: c for c in ranked}
    weak = by_id["STRAT_B_E1_RR1_PD_CLEAR_LONG"]
    assert weak.skip_recommended is True


def test_empty_queue(tmp_path):
    queue = tmp_path / "promote_queue.yaml"
    queue.write_text(yaml.safe_dump({"schema_version": 1, "entries": []}), encoding="utf-8")
    entries = cpr._load_queue_entries(queue)
    ranked = cpr.rank_queue_entries(entries, results_dir=tmp_path)
    assert ranked == []


def test_missing_queue_file(tmp_path):
    entries = cpr._load_queue_entries(tmp_path / "does-not-exist.yaml")
    assert entries == []


def test_pooled_t_zero_is_preserved_not_nan_coerced(tmp_path):
    """Code-review A- residual: pooled_t=0 must not falsy-coerce to NaN.

    Previously `float(entry.get("pooled_t") or float("nan"))` collapsed a
    legitimate 0.0 into NaN. A degenerate t=0 fast-lane result is incoherent
    but should not be silently coerced -- the early-return guard at
    abs(pooled_t)<1e-9 in compute_oos_power_readiness already handles it
    semantically. Preserve the input.
    """
    queue = tmp_path / "promote_queue.yaml"
    results = tmp_path / "results"
    results.mkdir()
    rmd = results / "rstub.md"
    rmd.write_text(
        "## Split summary\n\n"
        "| Split | N_universe | N_fired | Fire% | Scratch | Null | "
        "ExpR | Policy EV/opp | Sharpe | t | p_two |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| IS | 500 | 100 | 20.00% | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |\n"
        "| OOS | 100 | 50 | 50.00% | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |\n",
        encoding="utf-8",
    )
    queue.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "result_md": "results/rstub.md",
                        "strategy_id": "STRAT_ZERO_T_E2_RR1",
                        "direction": "long",
                        "pooled_t": 0.0,
                        "pooled_expr": 0.0,
                        "pooled_n": 100,
                        "pooling_artifact": False,
                        "status": "QUEUED",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    entries = cpr._load_queue_entries(queue)
    ranked = cpr.rank_queue_entries(entries, results_dir=results)
    assert len(ranked) == 1
    # pooled_t MUST be the literal 0.0, not NaN
    assert ranked[0].pooled_t == 0.0, f"pooled_t=0.0 must be preserved, got {ranked[0].pooled_t!r}"
    # And the score must still safely compute (early-return at t<1e-9 fires
    # in oos_power; deflation_headroom is 0 below threshold; dir_match is 0)
    assert ranked[0].total_score >= 0.0


def test_csv_columns_match_contract(synthetic_queue, tmp_path):
    queue_path, results_dir = synthetic_queue
    entries = cpr._load_queue_entries(queue_path)
    ranked = cpr.rank_queue_entries(entries, results_dir=results_dir)
    out = tmp_path / "ranking.csv"
    cpr.write_csv(ranked, out)
    text = out.read_text(encoding="utf-8")
    # Header line must contain every documented column
    header = text.splitlines()[0]
    for col in cpr.CSV_COLUMNS:
        assert col in header, f"missing column {col!r} in CSV header"


def test_canonical_threshold_constant():
    """Anchor: HEAVYWEIGHT_T_THRESHOLD is the Chordia 2018 no-theory value."""
    # NB: Check #160 in pipeline/check_drift.py enforces parity at commit time.
    # This is a fast-feedback test for the same anchor.
    assert cpr.HEAVYWEIGHT_T_THRESHOLD == 3.79
