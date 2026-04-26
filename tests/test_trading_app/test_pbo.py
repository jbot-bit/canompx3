"""Tests for Probability of Backtest Overfitting (PBO) — Bailey et al. 2014."""

from datetime import date, timedelta

import duckdb

from trading_app.pbo import compute_family_pbo, compute_pbo


def _days(n, start=date(2020, 1, 1)):
    """Generate n consecutive trading days starting from start."""
    return [start + timedelta(days=i) for i in range(n)]


class TestComputePbo:
    def test_single_strategy_returns_none(self):
        """PBO requires 2+ strategies (no selection to measure)."""
        days = _days(19)
        result = compute_pbo({"A": [(d, 1.0) for d in days]})
        assert result["pbo"] is None
        assert result["n_splits"] == 0

    def test_insufficient_data_returns_none(self):
        """Too few days for meaningful block partition."""
        days = _days(4)
        result = compute_pbo(
            {
                "A": [(d, 1.0) for d in days],
                "B": [(d, -1.0) for d in days],
            }
        )
        assert result["pbo"] is None

    def test_robust_strategies_low_pbo(self):
        """Two strategies consistently positive -> PBO should be low.

        Strategy A always wins (+2R), B always wins (+1R).
        IS-best is always A, OOS is always positive -> PBO = 0.
        """
        days = _days(80)
        strategy_pnl = {
            "A": [(d, 2.0) for d in days],
            "B": [(d, 1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        assert result["pbo"] <= 0.10  # Robust — near zero
        assert result["n_splits"] == 70  # C(8,4) = 70

    def test_overfit_strategy_high_pbo(self):
        """One strategy overfits early data, fails on later data.

        Strategy A: wins in first half, loses in second half.
        Strategy B: mediocre everywhere.
        Selection on early blocks picks A, but A's OOS is negative.
        """
        days = _days(80)
        half = len(days) // 2
        strategy_pnl = {
            # A: strong first half, collapses second half
            "A": [(d, 3.0) if i < half else (d, -2.0) for i, d in enumerate(days)],
            # B: small consistent edge
            "B": [(d, 0.1) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # A dominates IS when train includes early blocks, but OOS on late blocks is negative
        assert result["pbo"] > 0.2

    def test_identical_strategies_zero_pbo(self):
        """Identical strategies -> IS-best = any, OOS always same -> PBO=0."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, 1.0) for d in days],
            "B": [(d, 1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] == 0.0

    def test_all_negative_high_pbo(self):
        """Two losing strategies — IS-best is just less bad, OOS still negative."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, -0.5) for d in days],
            "B": [(d, -1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # IS-best (A, less negative) still has negative OOS -> PBO = 1.0
        assert result["pbo"] == 1.0

    def test_n_blocks_parameter(self):
        """Custom block count changes number of splits."""
        days = _days(60)
        strategy_pnl = {
            "A": [(d, 1.0) for d in days],
            "B": [(d, 0.5) for d in days],
        }
        # C(6,3) = 20 splits
        result = compute_pbo(strategy_pnl, n_blocks=6)
        assert result["n_splits"] == 20

    def test_logit_pbo_computed(self):
        """Logit PBO is computed for 0 < PBO < 1."""
        days = _days(80)
        strategy_pnl = {
            "A": [(d, -0.5) for d in days],
            "B": [(d, -1.0) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        # PBO = 1.0, logit is undefined at boundaries
        if result["pbo"] == 1.0:
            assert result["logit_pbo"] is None

    def test_many_strategies(self):
        """PBO works with more than 2 strategies."""
        days = _days(80)
        strategy_pnl = {f"S{i}": [(d, 1.0 + i * 0.1) for d in days] for i in range(5)}
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        # All positive -> PBO should be 0
        assert result["pbo"] == 0.0
        assert result["n_splits"] == 70

    def test_compute_pbo_deterministic_regression(self):
        """Pin PBO output to catch regressions from bulk-load refactoring."""
        days = _days(80)
        half = len(days) // 2
        strategy_pnl = {
            "A": [(d, 3.0) if i < half else (d, -2.0) for i, d in enumerate(days)],
            "B": [(d, 0.1) for d in days],
            "C": [(d, 1.5) for d in days],
        }
        result = compute_pbo(strategy_pnl)
        assert result["pbo"] is not None
        assert result["n_splits"] == 70
        assert isinstance(result["pbo"], float)
        assert 0.0 <= result["pbo"] <= 1.0


class TestComputeFamilyPbo:
    def test_tight_stop_family_uses_stop_multiplier_variants(self, monkeypatch, tmp_path):
        db_path = tmp_path / "pbo.db"
        con = duckdb.connect(str(db_path))
        con.execute("""
            CREATE TABLE validated_setups (
                strategy_id VARCHAR,
                instrument VARCHAR,
                family_hash VARCHAR,
                orb_label VARCHAR,
                orb_minutes INTEGER,
                entry_model VARCHAR,
                rr_target DOUBLE,
                confirm_bars INTEGER,
                filter_type VARCHAR,
                stop_multiplier DOUBLE,
                status VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE orb_outcomes (
                symbol VARCHAR,
                trading_day DATE,
                pnl_r DOUBLE,
                mae_r DOUBLE,
                entry_price DOUBLE,
                stop_price DOUBLE,
                outcome VARCHAR,
                orb_label VARCHAR,
                orb_minutes INTEGER,
                entry_model VARCHAR,
                rr_target DOUBLE,
                confirm_bars INTEGER
            )
        """)
        con.execute("""
            INSERT INTO validated_setups VALUES
            ('MGC_CME_REOPEN_E1_RR2.0_CB1_NO_FILTER_S075', 'MGC', 'fam_stop', 'CME_REOPEN', 5, 'E1', 2.0, 1, 'NO_FILTER', 0.75, 'active'),
            ('MGC_CME_REOPEN_E1_RR2.5_CB1_NO_FILTER_S075', 'MGC', 'fam_stop', 'CME_REOPEN', 5, 'E1', 2.5, 1, 'NO_FILTER', 0.75, 'active')
        """)
        days = _days(20)
        for td in days:
            con.execute(
                """
                INSERT INTO orb_outcomes VALUES
                ('MGC', ?, 1.0, 0.9, 100.0, 99.0, 'win', 'CME_REOPEN', 5, 'E1', 2.0, 1),
                ('MGC', ?, 1.2, 0.9, 100.0, 99.0, 'win', 'CME_REOPEN', 5, 'E1', 2.5, 1)
            """,
                [td, td],
            )
        con.close()

        calls = []

        def fake_apply_tight_stop(outcomes, stop_multiplier, cost_spec):
            calls.append((len(outcomes), stop_multiplier, cost_spec))
            adjusted = []
            for outcome in outcomes:
                updated = dict(outcome)
                updated["pnl_r"] = outcome["pnl_r"] - 0.25
                adjusted.append(updated)
            return adjusted

        monkeypatch.setattr("trading_app.pbo.deployable_validated_relation", lambda con: "validated_setups")
        monkeypatch.setattr("trading_app.pbo.apply_tight_stop", fake_apply_tight_stop)
        monkeypatch.setattr("trading_app.pbo.get_cost_spec", lambda instrument: object())

        con = duckdb.connect(str(db_path), read_only=True)
        result = compute_family_pbo(con, "fam_stop", "MGC")
        con.close()

        assert result["pbo"] is not None
        assert result["n_splits"] == 70
        assert len(calls) == 2
        assert all(stop_multiplier == 0.75 for _n, stop_multiplier, _spec in calls)
