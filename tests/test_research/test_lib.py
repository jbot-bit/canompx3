"""Tests for research.lib -- shared research utilities."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from research.lib.audit import assert_no_inflation
from research.lib.db import connect_db, query_df
from research.lib.io import format_stats_table, output_dir, write_csv, write_markdown
from research.lib.stats import (
    bh_fdr,
    compute_metrics,
    expanding_stat,
    mannwhitney_2s,
    ttest_1s,
    year_by_year,
)

_HAS_SCIPY = True
try:
    import scipy  # noqa: F401
except ImportError:
    _HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


# ── audit.py ─────────────────────────────────────────────────────────────


class TestAssertNoInflation:
    """Join audit: n_after must not exceed n_before."""

    def test_equal_counts_passes(self):
        assert_no_inflation(100, 100, context="test")

    def test_fewer_rows_passes(self):
        assert_no_inflation(100, 95, context="test")

    def test_inflated_raises(self):
        with pytest.raises(ValueError, match="inflated"):
            assert_no_inflation(100, 300, context="my_join")

    def test_inflated_includes_context(self):
        with pytest.raises(ValueError, match="my_join"):
            assert_no_inflation(50, 51, context="my_join")

    def test_zero_counts_passes(self):
        assert_no_inflation(0, 0, context="empty")


# ── stats.py: ttest_1s ──────────────────────────────────────────────────


@needs_scipy
class TestTtest1s:
    """One-sample t-test returning (n, mean, win_rate, t_stat, p_value)."""

    def test_known_positive(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        n, mean, wr, t, p = ttest_1s(arr)
        assert n == 10
        assert mean == pytest.approx(5.5)
        assert wr == pytest.approx(1.0)  # all > 0
        assert t > 0
        assert p < 0.001

    def test_mixed_pnl(self):
        arr = np.array([1.5, -1.0, 1.5, -1.0, 1.5, -1.0, 1.5, -1.0, 1.5, -1.0])
        n, mean, wr, t, p = ttest_1s(arr)
        assert n == 10
        assert wr == pytest.approx(0.5)

    def test_too_few_returns_nan(self):
        arr = np.array([1.0, 2.0])
        n, mean, wr, t, p = ttest_1s(arr)
        assert n == 2
        assert np.isnan(t)
        assert np.isnan(p)

    def test_nan_stripped(self):
        arr = np.array([1.0, float("nan"), 2.0, float("nan"), 3.0,
                        4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        n, mean, wr, t, p = ttest_1s(arr)
        assert n == 10  # 2 NaN stripped

    def test_empty_returns_nan(self):
        n, mean, wr, t, p = ttest_1s(np.array([]))
        assert n == 0
        assert np.isnan(mean)


# ── stats.py: bh_fdr ────────────────────────────────────────────────────


class TestBhFdr:
    """Benjamini-Hochberg FDR correction."""

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        rejected = bh_fdr(p_values, q=0.10)
        assert rejected == {0, 1, 2}

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        rejected = bh_fdr(p_values, q=0.10)
        assert rejected == set()

    def test_partial_rejection(self):
        # 5 tests, first two have very low p
        p_values = [0.001, 0.003, 0.20, 0.50, 0.90]
        rejected = bh_fdr(p_values, q=0.10)
        assert 0 in rejected
        assert 1 in rejected
        assert 4 not in rejected

    def test_empty_input(self):
        assert bh_fdr([], q=0.10) == set()

    def test_single_significant(self):
        assert bh_fdr([0.01], q=0.10) == {0}

    def test_single_not_significant(self):
        assert bh_fdr([0.50], q=0.10) == set()


# ── stats.py: compute_metrics ───────────────────────────────────────────


class TestComputeMetrics:
    """Standard strategy metrics bundle."""

    def test_positive_strategy(self):
        pnls = [1.5, -1.0, 1.5, -1.0, 2.0, -1.0, 1.5, -1.0, 1.5, -1.0]
        m = compute_metrics(pnls)
        assert m is not None
        assert m["n"] == 10
        assert m["win_rate"] == pytest.approx(0.5)
        assert m["avg_r"] > 0
        assert m["total_r"] == pytest.approx(3.0)
        assert m["sharpe"] > 0
        assert m["max_dd"] >= 0

    def test_all_nan_returns_none(self):
        assert compute_metrics([float("nan"), float("nan")]) is None

    def test_empty_returns_none(self):
        assert compute_metrics([]) is None

    def test_single_value(self):
        m = compute_metrics([1.5])
        assert m["n"] == 1
        assert m["avg_r"] == pytest.approx(1.5)


# ── stats.py: mannwhitney_2s ────────────────────────────────────────────


@needs_scipy
class TestMannWhitney:
    """Two-sample Mann-Whitney U test."""

    def test_different_groups(self):
        a = [10, 11, 12, 13, 14]
        b = [1, 2, 3, 4, 5]
        u, p = mannwhitney_2s(a, b)
        assert p < 0.05

    def test_too_few_returns_nan(self):
        u, p = mannwhitney_2s([1, 2], [3, 4])
        assert np.isnan(p)


# ── stats.py: year_by_year ──────────────────────────────────────────────


@needs_scipy
class TestYearByYear:
    """Per-year breakdown of PnL stats."""

    def test_two_years(self):
        df = pd.DataFrame({
            "trading_day": pd.to_datetime(
                ["2024-03-01"] * 15 + ["2025-03-01"] * 15
            ),
            "pnl_r": [1.0] * 10 + [-0.5] * 5 + [0.5] * 10 + [-1.0] * 5,
        })
        result = year_by_year(df)
        assert len(result) == 2
        assert list(result["year"]) == [2024, 2025]
        assert all(result["n"] == 15)


# ── stats.py: expanding_stat ────────────────────────────────────────────


class TestExpandingStat:
    """Expanding-window mean (no lookahead)."""

    def test_expanding_mean(self):
        df = pd.DataFrame({"atr": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = expanding_stat(df, "atr", min_periods=2)
        assert np.isnan(result.iloc[0])  # min_periods=2
        assert result.iloc[1] == pytest.approx(15.0)
        assert result.iloc[4] == pytest.approx(30.0)


# ── db.py ────────────────────────────────────────────────────────────────


class TestConnectDb:
    """Context manager for DuckDB connections."""

    def test_connect_returns_connection(self, tmp_path, monkeypatch):
        db = tmp_path / "test.db"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE t (x INT)")
        con.execute("INSERT INTO t VALUES (42)")
        con.close()
        monkeypatch.setattr("research.lib.db.GOLD_DB_PATH", db)
        with connect_db() as c:
            result = c.execute("SELECT x FROM t").fetchone()
        assert result[0] == 42

    def test_connect_closes_on_exit(self, tmp_path, monkeypatch):
        db = tmp_path / "test.db"
        duckdb.connect(str(db)).close()
        monkeypatch.setattr("research.lib.db.GOLD_DB_PATH", db)
        with connect_db() as c:
            pass
        # Connection should be closed -- no way to query
        # (DuckDB doesn't expose .closed, so just verify no exception in block)


class TestQueryDf:
    """One-shot query convenience function."""

    def test_returns_dataframe(self, tmp_path, monkeypatch):
        db = tmp_path / "test.db"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE t (x INT, y TEXT)")
        con.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        con.close()
        monkeypatch.setattr("research.lib.db.GOLD_DB_PATH", db)
        df = query_df("SELECT * FROM t ORDER BY x")
        assert len(df) == 2
        assert list(df.columns) == ["x", "y"]
        assert df["x"].tolist() == [1, 2]


# ── io.py ────────────────────────────────────────────────────────────────


class TestOutputDir:
    """Output directory resolution."""

    def test_returns_research_output(self):
        d = output_dir()
        assert d.name == "output"
        assert d.parent.name == "research"

    def test_creates_if_missing(self, tmp_path, monkeypatch):
        fake_output = tmp_path / "research" / "output"
        monkeypatch.setattr("research.lib.io._OUTPUT_DIR", fake_output)
        result = output_dir()
        assert result.exists()


class TestWriteCsv:
    """CSV output helper."""

    def test_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("research.lib.io._OUTPUT_DIR", tmp_path)
        df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        path = write_csv(df, "test.csv")
        assert path.exists()
        assert path.name == "test.csv"
        content = path.read_text()
        assert "x,y" in content


class TestWriteMarkdown:
    """Markdown output helper."""

    def test_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("research.lib.io._OUTPUT_DIR", tmp_path)
        path = write_markdown("# Hello", "test.md")
        assert path.exists()
        assert path.read_text() == "# Hello"


class TestFormatStatsTable:
    """Markdown stats table formatting."""

    def test_formats_table(self):
        results = {
            "ALL": {"n": 500, "mean_r": 0.12, "win_rate": 0.54, "p_value": 0.003},
            "G4": {"n": 320, "mean_r": 0.18, "win_rate": 0.57, "p_value": 0.001},
        }
        table = format_stats_table(results)
        assert "ALL" in table
        assert "G4" in table
        assert "500" in table
        assert "|" in table  # markdown table

    def test_empty_dict(self):
        table = format_stats_table({})
        assert "|" in table  # still has header
