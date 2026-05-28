"""Unit tests for the Stage 4a NYSE_PREOPEN bounded runner.

Stage 4a contract: the runner ships with these tests, but Stage 4a does NOT
run the runner against canonical gold.db data. These tests exercise the
runner's logic against synthetic in-memory DuckDB seeds and against the
promoted prereg's SHA-locked state.

Test classes
------------

- ``TestPromotedPrereg`` — prereg promoted out of drafts/; SHA stable; loader
  accepts; metadata matches the runner's locked constants.
- ``TestCellEnumeration`` — K=27 enumeration is in lock order and exhaustive.
- ``TestNFPSplit`` — NFP-split partitioning matches
  ``pipeline.calendar_filters.is_nfp_day`` exactly; sum of partitions
  reconstitutes the input.
- ``TestHoldoutSplit`` — strict Mode A IS/OOS partition uses
  ``HOLDOUT_SACRED_FROM`` boundary.
- ``TestDSTCount`` — DST-regime count exercises ``pipeline.dst.is_us_dst``.
- ``TestCellStats`` — per-cell stats math (Chordia t, sharpe, OOS power)
  against a small handcrafted seed with known answers.
- ``TestBHFDR`` — BH-FDR at K=27 composes correctly; NaN p-values fail the
  gate cleanly.
- ``TestGradeCell`` — promotion gate fires the right verdict label for the
  six prereg-defined paths (PASS / FAIL_CHORDIA / DST_IMBALANCE / DEAD_OOS
  / CONDITIONAL_OOS_UNDERPOWERED).
- ``TestSyntheticEndToEnd`` — full ``compute_full_verdict`` against an
  in-memory DuckDB seed with 27 cells; verdict labels resolve.
- ``TestPrereqCheck`` — ``check_prereq`` emits the expected PASS lines for
  prereg / NFP / NYSE-holiday / OOS-power; SKIP for gold.db when no con.
- ``TestRunnerCLI`` — the CLI refuses verdict-emission in 4a; ``--dry-run-cells``
  enumerates exactly 27; ``--check-prereq`` exits 0.

The Stage 4b run-against-canonical path is NOT exercised here. A separate
integration smoke test (env-var gated) is intentionally omitted from Stage
4a — exercising the canonical layer is Stage 4b's contract.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import duckdb
import pytest

from pipeline.calendar_filters import is_nfp_day
from pipeline.dst import is_us_dst
from research.mnq_nyse_preopen_e2_nfp_spillover_v1 import (
    K_FAMILY,
    ORB_MINUTES,
    PROMOTED_PREREG_SHA,
    RR_TARGETS,
    SPLITS,
    CellSpec,
    OutcomeRow,
    apply_split,
    build_parser,
    check_prereq,
    compose_bh_fdr,
    compute_cell_stats,
    compute_full_verdict,
    count_dst_regimes,
    dry_run_cells,
    enumerate_cells,
    fetch_canonical_cell_rows,
    grade_cell,
    grade_family,
    load_promoted_prereg,
    main,
    prereg_path,
    split_by_holdout,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM
from trading_app.hypothesis_loader import compute_file_sha


# ---------------------------------------------------------------------------
# Promoted prereg
# ---------------------------------------------------------------------------


class TestPromotedPrereg:
    def test_prereg_file_exists_outside_drafts(self) -> None:
        path = prereg_path()
        assert path.exists(), f"Promoted prereg missing at {path}"
        assert path.parent.name == "hypotheses", (
            "Prereg must be promoted out of drafts/ (parent dir must be 'hypotheses')"
        )

    def test_drafts_does_not_still_contain_promoted_file(self) -> None:
        drafts = (
            Path(__file__).resolve().parents[2]
            / "docs"
            / "audit"
            / "hypotheses"
            / "drafts"
            / "2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.draft.yaml"
        )
        assert not drafts.exists(), f"Old drafts/ copy still present at {drafts}; git mv should have removed it."

    def test_prereg_sha_matches_locked_constant(self) -> None:
        sha = compute_file_sha(prereg_path())
        assert sha == PROMOTED_PREREG_SHA, f"SHA drift on promoted prereg: file={sha}, locked={PROMOTED_PREREG_SHA}"

    def test_load_promoted_prereg_returns_expected_metadata(self) -> None:
        meta = load_promoted_prereg()
        assert meta["name"] == "mnq_nyse_preopen_e2_nfp_spillover_v1"
        assert meta["total_expected_trials"] == K_FAMILY
        assert meta["holdout_date"] == HOLDOUT_SACRED_FROM
        assert meta.get("metadata", {}).get("theory_grant") is False


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------


class TestCellEnumeration:
    def test_k_family_is_27(self) -> None:
        assert K_FAMILY == 27
        assert len(ORB_MINUTES) * len(RR_TARGETS) * len(SPLITS) == 27

    def test_enumerate_cells_produces_exactly_27(self) -> None:
        cells = enumerate_cells()
        assert len(cells) == 27

    def test_enumerate_cells_in_lock_order(self) -> None:
        cells = enumerate_cells()
        # First cell: O5 RR1.0 all_days, last: O30 RR2.0 non_nfp_days
        assert cells[0] == CellSpec(orb_minutes=5, rr_target=1.0, split="all_days")
        assert cells[-1] == CellSpec(orb_minutes=30, rr_target=2.0, split="non_nfp_days")

    def test_enumerate_cells_unique(self) -> None:
        cells = enumerate_cells()
        assert len(set(cells)) == 27, "Cells must be unique"

    def test_enumerate_cells_covers_full_grid(self) -> None:
        cells = enumerate_cells()
        for o in ORB_MINUTES:
            for rr in RR_TARGETS:
                for split in SPLITS:
                    assert CellSpec(orb_minutes=o, rr_target=rr, split=split) in cells


# ---------------------------------------------------------------------------
# NFP split
# ---------------------------------------------------------------------------


def _row(d: date, pnl: float = 0.1) -> OutcomeRow:
    return OutcomeRow(trading_day=d, pnl_r=pnl, entry_ts=None)


class TestNFPSplit:
    def test_nfp_days_only_matches_canonical(self) -> None:
        # June 7 2024 = first Friday of June = NFP. June 14 = third Friday = not NFP.
        nfp = date(2024, 6, 7)
        non_nfp = date(2024, 6, 14)
        assert is_nfp_day(nfp) is True
        assert is_nfp_day(non_nfp) is False

        rows = [_row(nfp), _row(non_nfp), _row(date(2024, 7, 5))]  # 7/5 first Friday July = NFP
        nfp_only = apply_split(rows, "nfp_days_only")
        non_nfp_only = apply_split(rows, "non_nfp_days")

        assert {r.trading_day for r in nfp_only} == {nfp, date(2024, 7, 5)}
        assert {r.trading_day for r in non_nfp_only} == {non_nfp}

    def test_partition_is_exhaustive(self) -> None:
        rows = [_row(date(2024, 6, d)) for d in range(3, 29)]
        assert len(rows) == 26
        all_days = apply_split(rows, "all_days")
        nfp = apply_split(rows, "nfp_days_only")
        non_nfp = apply_split(rows, "non_nfp_days")
        assert len(all_days) == len(rows)
        assert len(nfp) + len(non_nfp) == len(all_days)

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            apply_split([_row(date(2024, 1, 5))], "weekly_pattern")


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------


class TestHoldoutSplit:
    def test_holdout_boundary_is_inclusive_for_oos(self) -> None:
        before = _row(date(2025, 12, 31))  # IS
        boundary = _row(HOLDOUT_SACRED_FROM)  # OOS (>=)
        after = _row(date(2026, 3, 1))  # OOS
        is_rows, oos_rows = split_by_holdout([before, boundary, after])
        assert [r.trading_day for r in is_rows] == [date(2025, 12, 31)]
        assert [r.trading_day for r in oos_rows] == [HOLDOUT_SACRED_FROM, date(2026, 3, 1)]


# ---------------------------------------------------------------------------
# DST count
# ---------------------------------------------------------------------------


class TestDSTCount:
    def test_count_dst_regimes_matches_pipeline(self) -> None:
        # Jan = EST (winter), Jul = EDT (summer), per pipeline.dst.is_us_dst
        winter = date(2024, 1, 10)
        summer = date(2024, 7, 10)
        assert is_us_dst(winter) is False
        assert is_us_dst(summer) is True
        n_est, n_edt = count_dst_regimes([_row(winter), _row(summer), _row(summer)])
        assert n_est == 1
        assert n_edt == 2


# ---------------------------------------------------------------------------
# Per-cell stats
# ---------------------------------------------------------------------------


class TestCellStats:
    def test_zero_rows_produces_nan_stats(self) -> None:
        spec = CellSpec(orb_minutes=5, rr_target=1.0, split="all_days")
        stats = compute_cell_stats(spec, [])
        assert stats.n_is_on == 0
        assert stats.n_oos_on == 0
        # NaN equality requires explicit check
        import math

        assert math.isnan(stats.expr_is)
        assert math.isnan(stats.t_is)

    def test_known_t_stat_recovered(self) -> None:
        # 200 rows in IS, mean=0.20, std=1.0 -> t = 0.20*sqrt(200)/1.0 ~= 2.828
        spec = CellSpec(orb_minutes=5, rr_target=1.0, split="all_days")
        rows: list[OutcomeRow] = []
        for i in range(200):
            d = date(2022, 1, 1)
            d = date(d.year, d.month, 1 + (i % 27))  # arbitrary IS dates < HOLDOUT
            # Deterministic alternating pattern with mean ~0.20
            pnl = 0.20 + (1.0 if i % 2 == 0 else -1.0)
            rows.append(OutcomeRow(trading_day=d, pnl_r=pnl, entry_ts=None))
        stats = compute_cell_stats(spec, rows)
        assert stats.n_is_on == 200
        # t = mean*sqrt(N)/std; with the +/-1 alternation around 0.20:
        #   mean = 0.20, std ≈ 1.0005, t ≈ 0.20 * sqrt(200) / 1.0005 ≈ 2.827
        assert 2.7 < stats.t_is < 2.95
        assert stats.expr_is == pytest.approx(0.20, rel=0.001)

    def test_oos_power_gated_by_n(self) -> None:
        # IS with strong signal, OOS empty -> power == 0
        spec = CellSpec(orb_minutes=5, rr_target=1.0, split="all_days")
        rows = [OutcomeRow(trading_day=date(2022, 6, 1 + i % 27), pnl_r=0.5, entry_ts=None) for i in range(150)]
        stats = compute_cell_stats(spec, rows)
        assert stats.n_oos_on == 0
        assert stats.oos_power == 0.0
        assert stats.oos_power_tier == "STATISTICALLY_USELESS"


# ---------------------------------------------------------------------------
# BH-FDR composition
# ---------------------------------------------------------------------------


class TestBHFDR:
    def test_compose_bh_fdr_simple(self) -> None:
        # All small p-values -> all reject
        rejects, qs = compose_bh_fdr([0.001, 0.002, 0.003], alpha=0.05)
        assert all(rejects)
        assert all(q < 0.05 for q in qs)

    def test_compose_bh_fdr_nan_treated_as_one(self) -> None:
        # NaN should fail the gate (treated as p=1.0), not crash
        import math

        rejects, _ = compose_bh_fdr([0.001, math.nan], alpha=0.05)
        assert bool(rejects[0]) is True
        assert bool(rejects[1]) is False

    def test_compose_bh_fdr_empty(self) -> None:
        rejects, qs = compose_bh_fdr([], alpha=0.05)
        assert rejects == []
        assert qs == []


# ---------------------------------------------------------------------------
# Per-cell grading
# ---------------------------------------------------------------------------


def _stats_factory(
    *,
    t_is: float = 4.0,
    n_is_on: int = 200,
    expr_is: float = 0.15,
    expr_oos: float = 0.10,
    n_est_is: int = 100,
    n_edt_is: int = 100,
    n_oos_on: int = 50,
    oos_power: float = 0.30,
) -> "object":
    from research.mnq_nyse_preopen_e2_nfp_spillover_v1 import CellStats

    spec = CellSpec(orb_minutes=5, rr_target=1.0, split="all_days")
    dir_match = (expr_is > 0 and expr_oos > 0) or (expr_is < 0 and expr_oos < 0)
    return CellStats(
        spec=spec,
        n_is_on=n_is_on,
        expr_is=expr_is,
        sharpe_is=expr_is,
        t_is=t_is,
        p_one_sided=0.0001,
        n_oos_on=n_oos_on,
        expr_oos=expr_oos,
        n_est_is=n_est_is,
        n_edt_is=n_edt_is,
        dir_match_oos=dir_match,
        oos_power=oos_power,
        oos_power_tier=(
            "CAN_REFUTE"
            if oos_power >= 0.80
            else ("DIRECTIONAL_ONLY" if oos_power >= 0.50 else "STATISTICALLY_USELESS")
        ),
    )


class TestGradeCell:
    def test_pass_chordia_strict(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=4.0), bh_q=0.001, bh_reject=True)
        assert verdict.verdict_label == "PASS_CHORDIA_STRICT"
        assert verdict.pass_chordia_strict is True
        assert verdict.dst_balance_verdict == "BALANCED"

    def test_fail_chordia_low_t(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=2.5), bh_q=0.001, bh_reject=True)
        assert verdict.verdict_label == "FAIL_CHORDIA_STRICT"
        assert "t_IS=2.500" in verdict.verdict_reason

    def test_fail_chordia_low_n(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=4.0, n_is_on=50), bh_q=0.001, bh_reject=True)
        assert verdict.verdict_label == "FAIL_CHORDIA_STRICT"
        assert "N_IS_on=50" in verdict.verdict_reason

    def test_fail_chordia_negative_expr(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=4.0, expr_is=-0.15, expr_oos=-0.10), bh_q=0.001, bh_reject=True)
        assert verdict.verdict_label == "FAIL_CHORDIA_STRICT"

    def test_fail_chordia_bh_reject_false(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=4.0), bh_q=0.20, bh_reject=False)
        assert verdict.verdict_label == "FAIL_CHORDIA_STRICT"
        assert "BH q=0.2000" in verdict.verdict_reason

    def test_unverified_dst_imbalance_takes_precedence(self) -> None:
        verdict = grade_cell(_stats_factory(t_is=4.0, n_est_is=10, n_edt_is=190), bh_q=0.001, bh_reject=True)
        assert verdict.verdict_label == "UNVERIFIED_DST_IMBALANCE"
        assert verdict.dst_balance_verdict == "EST_THIN"
        assert verdict.pass_chordia_strict is False

    def test_conditional_when_oos_underpowered_dir_mismatch(self) -> None:
        # Strong IS, OOS sign opposes, but power < CAN_REFUTE -> CONDITIONAL not DEAD
        verdict = grade_cell(
            _stats_factory(t_is=4.0, expr_is=0.15, expr_oos=-0.05, oos_power=0.30),
            bh_q=0.001,
            bh_reject=True,
        )
        assert verdict.verdict_label == "CONDITIONAL_OOS_UNDERPOWERED"

    def test_dead_when_oos_refutes_at_can_refute_power(self) -> None:
        # OOS sign opposes IS AND power >= CAN_REFUTE -> DEAD per RULE 3.3
        verdict = grade_cell(
            _stats_factory(t_is=4.0, expr_is=0.15, expr_oos=-0.05, oos_power=0.85),
            bh_q=0.001,
            bh_reject=True,
        )
        assert verdict.verdict_label == "DEAD_OOS_REFUTES"


class TestGradeFamily:
    def test_partial_family_refuses(self) -> None:
        # Only 5 stats provided; family demands 27
        partial = [_stats_factory(t_is=4.0) for _ in range(5)]
        with pytest.raises(RuntimeError, match="K_FAMILY"):
            grade_family(partial)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Synthetic end-to-end against in-memory DuckDB
# ---------------------------------------------------------------------------


def _seed_synthetic_db(con: duckdb.DuckDBPyConnection) -> None:
    """Create an orb_outcomes-shaped table and seed 27-cell uniform data."""
    con.execute(
        """
        CREATE TABLE orb_outcomes (
            trading_day DATE,
            symbol VARCHAR,
            orb_label VARCHAR,
            orb_minutes INTEGER,
            rr_target DOUBLE,
            confirm_bars INTEGER,
            entry_model VARCHAR,
            entry_ts TIMESTAMPTZ,
            pnl_r DOUBLE
        )
        """
    )
    # Seed: 200 IS days (mix of EST/EDT/NFP) + 30 OOS days per (orb, rr)
    is_dates: list[date] = []
    for i in range(200):
        # Spread across 2022 Jan (EST) and 2022 Jul (EDT) to hit DST balance
        if i < 100:
            d = date(2022, 1, 3 + (i % 28))
        else:
            d = date(2022, 7, 1 + (i % 30))
        is_dates.append(d)
    oos_dates = [date(2026, 1, 5) + __import__("datetime").timedelta(days=i) for i in range(30)]
    all_dates = is_dates + oos_dates

    for o in ORB_MINUTES:
        for rr in RR_TARGETS:
            for i, d in enumerate(all_dates):
                # Deterministic small positive mean with noise
                pnl = 0.10 + (1.0 if i % 2 == 0 else -1.0)
                con.execute(
                    "INSERT INTO orb_outcomes VALUES (?, 'MNQ', 'NYSE_PREOPEN', ?, ?, 1, 'E2', NULL, ?)",
                    [d, o, rr, pnl],
                )


class TestSyntheticEndToEnd:
    def test_compute_full_verdict_resolves_all_27(self) -> None:
        con = duckdb.connect(":memory:")
        try:
            _seed_synthetic_db(con)
            verdicts = compute_full_verdict(con)
            assert len(verdicts) == K_FAMILY
            allowed_labels = {
                "PASS_CHORDIA_STRICT",
                "FAIL_CHORDIA_STRICT",
                "UNVERIFIED_DST_IMBALANCE",
                "DEAD_OOS_REFUTES",
                "CONDITIONAL_OOS_UNDERPOWERED",
            }
            for v in verdicts:
                assert v.verdict_label in allowed_labels, f"Unexpected label: {v.verdict_label}"
        finally:
            con.close()

    def test_fetch_canonical_cell_rows_filters_to_session(self) -> None:
        con = duckdb.connect(":memory:")
        try:
            _seed_synthetic_db(con)
            # Insert a contaminating row on a different session — must NOT be fetched
            con.execute(
                "INSERT INTO orb_outcomes VALUES (?, 'MNQ', 'NYSE_OPEN', 5, 1.0, 1, 'E2', NULL, 99.0)",
                [date(2022, 1, 5)],
            )
            rows = fetch_canonical_cell_rows(con, 5, 1.0)
            assert all(r.pnl_r != 99.0 for r in rows), "Contaminating NYSE_OPEN row leaked into NYSE_PREOPEN fetch"
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Prereq check
# ---------------------------------------------------------------------------


class TestPrereqCheck:
    def test_check_prereq_no_db_emits_skip(self) -> None:
        lines = check_prereq(None)
        joined = "\n".join(lines)
        assert "PASS prereg loaded" in joined
        assert "PASS pipeline.calendar_filters.is_nfp_day" in joined
        assert "PASS pipeline.market_calendar.is_nyse_holiday wired" in joined
        assert "PASS research.oos_power" in joined
        assert "SKIP canonical-coverage" in joined

    def test_check_prereq_with_db_reads_coverage(self) -> None:
        con = duckdb.connect(":memory:")
        try:
            _seed_synthetic_db(con)
            lines = check_prereq(con)
            joined = "\n".join(lines)
            assert "PASS orb_outcomes coverage" in joined
        finally:
            con.close()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestRunnerCLI:
    def test_no_flag_refuses(self) -> None:
        # main returns 2 and prints to stderr when no mode flag passed
        rc = main([])
        assert rc == 2

    def test_dry_run_cells_returns_27(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--dry-run-cells"])
        assert rc == 0
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        assert len(lines) == K_FAMILY

    def test_dry_run_cells_function(self) -> None:
        # Internal helper matches the CLI output line count
        assert len(dry_run_cells()) == K_FAMILY

    def test_parser_has_no_emit_verdict_flag(self) -> None:
        # Stage 4a contract: verdict emission is Stage 4b. Confirm no --emit-verdict.
        parser = build_parser()
        actions = {a.dest for a in parser._actions}
        assert "emit_verdict" not in actions
        assert "check_prereq" in actions
        assert "dry_run_cells" in actions
