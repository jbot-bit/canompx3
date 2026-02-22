# Research Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `research/lib/` — a thin utility package that eliminates ~25 lines of boilerplate per research script by consolidating DB connections, statistical tests, query building, join auditing, and output formatting.

**Architecture:** 5 small modules (`db.py`, `stats.py`, `query.py`, `audit.py`, `io.py`) with a top-level `__init__.py` re-exporting the most-used functions. Each module is independently testable with zero DB dependency (except `db.py` which wraps DuckDB). Forward-only adoption — existing scripts migrate opportunistically.

**Tech Stack:** Python, DuckDB, scipy.stats, numpy, pandas

**Design Doc:** `docs/plans/2026-02-22-research-lib-design.md`

---

### Task 1: Package Skeleton + audit.py (simplest module first)

**Files:**
- Create: `research/lib/__init__.py`
- Create: `research/lib/audit.py`
- Create: `tests/test_research/test_lib.py`

**Step 1: Create the package directory and empty `__init__.py`**

```bash
mkdir -p research/lib
touch research/lib/__init__.py
```

**Step 2: Write the failing test for `assert_no_inflation`**

In `tests/test_research/test_lib.py`:

```python
"""Tests for research.lib — shared research utilities."""

import pytest

from research.lib.audit import assert_no_inflation


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
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_research/test_lib.py::TestAssertNoInflation -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'research.lib.audit'`

**Step 4: Write `audit.py`**

In `research/lib/audit.py`:

```python
"""Join verification — catches silent row inflation from bad JOINs."""


def assert_no_inflation(n_before: int, n_after: int, context: str = "") -> None:
    """Raise ValueError if a JOIN inflated row count.

    Usage:
        n_raw = len(outcomes_df)
        merged = outcomes_df.merge(features_df, on=[...])
        assert_no_inflation(n_raw, len(merged), context="my_analysis")
    """
    if n_after > n_before:
        tag = f" [{context}]" if context else ""
        raise ValueError(
            f"Row count inflated{tag}: {n_before} → {n_after}. "
            f"Check JOIN columns (missing orb_minutes?)."
        )
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_research/test_lib.py::TestAssertNoInflation -x -q
```
Expected: 5 passed

**Step 6: Commit**

```bash
git add research/lib/__init__.py research/lib/audit.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): add audit.py with assert_no_inflation"
```

---

### Task 2: stats.py — ttest_1s and bh_fdr

**Files:**
- Create: `research/lib/stats.py`
- Modify: `tests/test_research/test_lib.py` (append)

**Step 1: Write failing tests for `ttest_1s`**

Append to `tests/test_research/test_lib.py`:

```python
import numpy as np

from research.lib.stats import ttest_1s, bh_fdr


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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_research/test_lib.py::TestTtest1s -x -q
python -m pytest tests/test_research/test_lib.py::TestBhFdr -x -q
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write `stats.py`**

In `research/lib/stats.py`:

```python
"""Statistical tests and metrics for research scripts.

Canonical implementations — use these instead of inline copies.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def ttest_1s(arr, mu: float = 0.0) -> tuple[int, float, float, float, float]:
    """One-sample t-test. Returns (n, mean, win_rate, t_stat, p_value).

    NaN values are stripped. Returns NaN for t/p if n < 10.

    Extracted from research_compressed_spring.py:71 and research_avoid_crosscheck.py:68
    (identical implementations).
    """
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    if n < 10:
        return n, float(a.mean()), float((a > 0).mean()), float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(a, mu)
    return n, float(a.mean()), float((a > 0).mean()), float(t), float(p)


def bh_fdr(p_values: list | np.ndarray, q: float = 0.10) -> set[int]:
    """Benjamini-Hochberg FDR correction. Returns set of rejected hypothesis indices.

    Extracted from research_compressed_spring.py:90 and research_avoid_crosscheck.py:77
    (identical implementations).
    """
    n = len(p_values)
    if n == 0:
        return set()
    ranked = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [q * (k + 1) / n for k in range(n)]
    max_k = -1
    for k, (_, p) in enumerate(ranked):
        if p <= thresholds[k]:
            max_k = k
    if max_k < 0:
        return set()
    return {idx for idx, _ in ranked[: max_k + 1]}


def mannwhitney_2s(a, b) -> tuple[float, float]:
    """Two-sample Mann-Whitney U test. Returns (u_stat, p_value)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def compute_metrics(pnls) -> dict | None:
    """Compute standard strategy metrics. Returns None if all NaN.

    Returns dict with keys: n, win_rate, avg_r, sharpe, max_dd, total_r.
    """
    a = np.array(pnls, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return None
    n = len(a)
    total_r = float(a.sum())
    avg_r = float(a.mean())
    win_rate = float((a > 0).mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sharpe = avg_r / std if std > 0 else 0.0
    # Max drawdown in R-units
    cumulative = np.cumsum(a)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0
    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 4),
        "total_r": round(total_r, 4),
    }


def year_by_year(
    df: pd.DataFrame, date_col: str = "trading_day", value_col: str = "pnl_r"
) -> pd.DataFrame:
    """Per-year breakdown: n, mean, win_rate, p_value.

    Returns DataFrame with columns: year, n, mean, win_rate, p_value.
    """
    df = df.copy()
    df["_year"] = pd.to_datetime(df[date_col]).dt.year
    rows = []
    for year, group in df.groupby("_year"):
        vals = group[value_col].dropna().values
        n, mean, wr, _, p = ttest_1s(vals)
        rows.append({"year": int(year), "n": n, "mean": round(mean, 4),
                      "win_rate": round(wr, 4), "p_value": round(p, 6) if not np.isnan(p) else None})
    return pd.DataFrame(rows)


def expanding_stat(
    df: pd.DataFrame, col: str, min_periods: int = 20
) -> pd.Series:
    """Expanding-window mean (no lookahead). Returns Series aligned to df index."""
    return df[col].expanding(min_periods=min_periods).mean()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_research/test_lib.py -x -q
```
Expected: all passed

**Step 5: Commit**

```bash
git add research/lib/stats.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): add stats.py with ttest_1s, bh_fdr, compute_metrics"
```

---

### Task 3: stats.py — compute_metrics, mannwhitney, year_by_year, expanding_stat

**Files:**
- Modify: `tests/test_research/test_lib.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_research/test_lib.py`:

```python
from research.lib.stats import compute_metrics, mannwhitney_2s, year_by_year, expanding_stat


class TestComputeMetrics:
    """Standard strategy metrics bundle."""

    def test_positive_strategy(self):
        pnls = [1.5, -1.0, 1.5, -1.0, 2.0, -1.0, 1.5, -1.0, 1.5, -1.0]
        m = compute_metrics(pnls)
        assert m is not None
        assert m["n"] == 10
        assert m["win_rate"] == pytest.approx(0.5)
        assert m["avg_r"] > 0
        assert m["total_r"] == pytest.approx(2.0)
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


class TestExpandingStat:
    """Expanding-window mean (no lookahead)."""

    def test_expanding_mean(self):
        df = pd.DataFrame({"atr": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result = expanding_stat(df, "atr", min_periods=2)
        assert np.isnan(result.iloc[0])  # min_periods=2
        assert result.iloc[1] == pytest.approx(15.0)
        assert result.iloc[4] == pytest.approx(30.0)
```

**Step 2: Run tests**

```bash
python -m pytest tests/test_research/test_lib.py -x -q
```
Expected: all passed (implementation already in Task 2)

**Step 3: Commit**

```bash
git add tests/test_research/test_lib.py
git commit -m "test(research-lib): add tests for compute_metrics, mannwhitney, year_by_year, expanding_stat"
```

---

### Task 4: db.py — Database Connection

**Files:**
- Create: `research/lib/db.py`
- Modify: `tests/test_research/test_lib.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_research/test_lib.py`:

```python
import duckdb

from research.lib.db import connect_db, query_df


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
        # Connection should be closed — no way to query
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
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_research/test_lib.py::TestConnectDb -x -q
```
Expected: FAIL

**Step 3: Write `db.py`**

In `research/lib/db.py`:

```python
"""Database connection lifecycle for research scripts.

All research scripts should use connect_db() or query_df() instead of
inline duckdb.connect() + os.environ boilerplate.
"""

from contextlib import contextmanager

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH


@contextmanager
def connect_db(read_only: bool = True):
    """Open a DuckDB connection to gold.db. Closes on exit.

    Usage:
        with connect_db() as con:
            df = con.execute(sql).fetchdf()
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Opens and closes connection automatically.

    Usage:
        df = query_df("SELECT * FROM orb_outcomes WHERE symbol = ?", ["MGC"])
    """
    with connect_db() as con:
        if params:
            return con.execute(sql, params).fetchdf()
        return con.execute(sql).fetchdf()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_research/test_lib.py -x -q
```
Expected: all passed

**Step 5: Commit**

```bash
git add research/lib/db.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): add db.py with connect_db, query_df"
```

---

### Task 5: io.py — Output Formatting

**Files:**
- Create: `research/lib/io.py`
- Modify: `tests/test_research/test_lib.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_research/test_lib.py`:

```python
from research.lib.io import output_dir, write_csv, write_markdown, format_stats_table


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
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_research/test_lib.py::TestOutputDir -x -q
```
Expected: FAIL

**Step 3: Write `io.py`**

In `research/lib/io.py`:

```python
"""Output directory and formatting helpers for research scripts.

Replaces OUTPUT_DIR.mkdir(parents=True, exist_ok=True) boilerplate
found in 70+ scripts.
"""

from pathlib import Path

import pandas as pd

# Canonical output dir — can be monkeypatched in tests
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def output_dir() -> Path:
    """Return research/output/ directory, creating if needed."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write DataFrame to research/output/<filename>."""
    path = output_dir() / filename
    df.to_csv(path, index=False)
    return path


def write_markdown(text: str, filename: str) -> Path:
    """Write text to research/output/<filename>."""
    path = output_dir() / filename
    path.write_text(text)
    return path


def format_stats_table(results: dict) -> str:
    """Format stats dict as markdown table.

    Input: {"label": {"n": int, "mean_r": float, "win_rate": float, "p_value": float}}
    Output: markdown table string.
    """
    lines = [
        "| Label | N | Mean R | WR | p-value |",
        "|-------|---:|-------:|----:|--------:|",
    ]
    for label, m in results.items():
        n = m.get("n", 0)
        mean_r = m.get("mean_r", 0)
        wr = m.get("win_rate", 0)
        p = m.get("p_value", None)
        p_str = f"{p:.4f}" if p is not None else "—"
        lines.append(f"| {label} | {n} | {mean_r:+.4f} | {wr:.1%} | {p_str} |")
    return "\n".join(lines)
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_research/test_lib.py -x -q
```
Expected: all passed

**Step 5: Commit**

```bash
git add research/lib/io.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): add io.py with output_dir, write_csv, write_markdown, format_stats_table"
```

---

### Task 6: query.py — SQL Builder + DST Split

**Files:**
- Create: `research/lib/query.py`
- Modify: `tests/test_research/test_lib.py` (append)

**Step 1: Write failing tests**

Append to `tests/test_research/test_lib.py`:

```python
from research.lib.query import SAFE_JOIN, outcomes_query, session_col, with_dst_split


class TestSafeJoin:
    """SAFE_JOIN constant contains canonical triple-join."""

    def test_contains_three_join_columns(self):
        assert "o.trading_day = d.trading_day" in SAFE_JOIN
        assert "o.symbol = d.symbol" in SAFE_JOIN
        assert "o.orb_minutes = d.orb_minutes" in SAFE_JOIN


class TestSessionCol:
    """session_col() builds column name from orb_label + stem."""

    def test_basic(self):
        assert session_col("1000", "size") == "orb_1000_size"

    def test_break_dir(self):
        assert session_col("0900", "break_dir") == "orb_0900_break_dir"


class TestOutcomesQuery:
    """outcomes_query() builds safe SQL with triple-join."""

    def test_basic_query(self):
        sql = outcomes_query("MGC", "1000", "E0")
        assert "o.trading_day = d.trading_day" in sql
        assert "o.symbol = d.symbol" in sql
        assert "o.orb_minutes = d.orb_minutes" in sql
        assert "'MGC'" in sql
        assert "'1000'" in sql
        assert "'E0'" in sql

    def test_extra_cols(self):
        sql = outcomes_query("MGC", "1000", "E0", extra_cols=["d.atr_5d"])
        assert "d.atr_5d" in sql

    def test_filters(self):
        sql = outcomes_query("MGC", "1000", "E0",
                             filters=["d.orb_1000_size >= 4"])
        assert "d.orb_1000_size >= 4" in sql

    def test_date_range(self):
        sql = outcomes_query("MGC", "1000", "E0",
                             date_range=("2021-01-01", "2025-12-31"))
        assert "2021-01-01" in sql
        assert "2025-12-31" in sql

    def test_selects_pnl_r(self):
        sql = outcomes_query("MGC", "1000", "E0")
        assert "o.pnl_r" in sql

    def test_filters_null_pnl(self):
        sql = outcomes_query("MGC", "1000", "E0")
        assert "o.pnl_r IS NOT NULL" in sql


class TestWithDstSplit:
    """with_dst_split() wraps base SQL with DST ON/OFF filters."""

    def test_us_regime(self):
        base = "SELECT * FROM t"
        on_sql, off_sql = with_dst_split(base, session="0900", regime_source="US")
        assert "us_dst" in on_sql
        assert "us_dst" in off_sql
        # ON = summer (DST active), OFF = winter
        assert "= TRUE" in on_sql or "= true" in on_sql.lower()
        assert "= FALSE" in off_sql or "= false" in off_sql.lower()

    def test_uk_regime(self):
        base = "SELECT * FROM t"
        on_sql, off_sql = with_dst_split(base, session="1800", regime_source="UK")
        assert "uk_dst" in on_sql
        assert "uk_dst" in off_sql
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_research/test_lib.py::TestOutcomesQuery -x -q
```
Expected: FAIL

**Step 3: Write `query.py`**

In `research/lib/query.py`:

```python
"""Composable SQL query builder for research scripts.

Bakes in the canonical triple-join (SAFE_JOIN) to prevent the most common
research bug: missing orb_minutes in the JOIN producing 3x row inflation.
"""

# Canonical triple-join — ALWAYS use this when joining orb_outcomes to daily_features.
# See .claude/rules/daily-features-joins.md for rationale.
SAFE_JOIN = """\
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes"""


def session_col(orb_label: str, stem: str) -> str:
    """Build daily_features column name: orb_{label}_{stem}.

    Example: session_col("1000", "size") -> "orb_1000_size"
    """
    return f"orb_{orb_label}_{stem}"


def outcomes_query(
    instrument: str,
    session: str,
    entry_model: str,
    extra_cols: list[str] | None = None,
    filters: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> str:
    """Build standard outcomes+features query with safe triple-join.

    Returns SQL string that SELECTs trading_day, pnl_r, outcome, and any
    extra_cols from the joined outcomes/features tables.

    Usage:
        sql = outcomes_query("MGC", "1000", "E0",
            extra_cols=["d.atr_5d"],
            filters=["d.orb_1000_size >= 4"],
            date_range=("2021-01-01", "2025-12-31"),
        )
        df = query_df(sql)
    """
    cols = ["o.trading_day", "o.pnl_r", "o.outcome", "o.mae_r", "o.mfe_r"]
    if extra_cols:
        cols.extend(extra_cols)
    select = ", ".join(cols)

    wheres = [
        f"o.symbol = '{instrument}'",
        f"o.orb_label = '{session}'",
        f"o.entry_model = '{entry_model}'",
        "o.outcome IN ('win', 'loss', 'early_exit')",
        "o.pnl_r IS NOT NULL",
    ]
    if filters:
        wheres.extend(filters)
    if date_range:
        wheres.append(f"o.trading_day >= '{date_range[0]}'")
        wheres.append(f"o.trading_day <= '{date_range[1]}'")
    where = "\n    AND ".join(wheres)

    return f"SELECT {select}\n{SAFE_JOIN}\n    WHERE {where}"


def with_dst_split(
    base_sql: str,
    session: str,
    regime_source: str,
) -> tuple[str, str]:
    """Wrap a base SQL query with DST regime filtering.

    Returns (dst_on_sql, dst_off_sql) — summer and winter variants.
    regime_source: "US" for 0900/0030/2300 sessions, "UK" for 1800.

    DST columns (us_dst, uk_dst) live in daily_features.
    MANDATORY per CLAUDE.md: any analysis touching DST-sensitive sessions
    MUST split by regime and report both halves.
    """
    col = "d.us_dst" if regime_source.upper() == "US" else "d.uk_dst"
    # Wrap base_sql as CTE to add DST filter
    on_sql = f"WITH base AS ({base_sql})\nSELECT * FROM base WHERE {col} = TRUE"
    off_sql = f"WITH base AS ({base_sql})\nSELECT * FROM base WHERE {col} = FALSE"
    return on_sql, off_sql
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_research/test_lib.py -x -q
```
Expected: all passed

**Step 5: Commit**

```bash
git add research/lib/query.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): add query.py with SAFE_JOIN, outcomes_query, with_dst_split"
```

---

### Task 7: __init__.py — Re-exports + Final Test Run

**Files:**
- Modify: `research/lib/__init__.py`

**Step 1: Write failing test for top-level imports**

Append to `tests/test_research/test_lib.py`:

```python
class TestTopLevelImports:
    """research.lib re-exports commonly used functions."""

    def test_stats_available(self):
        from research.lib import ttest_1s, bh_fdr, compute_metrics
        assert callable(ttest_1s)
        assert callable(bh_fdr)
        assert callable(compute_metrics)

    def test_db_available(self):
        from research.lib import connect_db, query_df
        assert callable(connect_db)
        assert callable(query_df)

    def test_query_available(self):
        from research.lib import outcomes_query, session_col, SAFE_JOIN
        assert callable(outcomes_query)
        assert callable(session_col)
        assert isinstance(SAFE_JOIN, str)

    def test_audit_available(self):
        from research.lib import assert_no_inflation
        assert callable(assert_no_inflation)

    def test_io_available(self):
        from research.lib import output_dir, write_csv, write_markdown, format_stats_table
        assert callable(output_dir)
        assert callable(write_csv)
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_research/test_lib.py::TestTopLevelImports -x -q
```
Expected: FAIL — `ImportError: cannot import name 'ttest_1s' from 'research.lib'`

**Step 3: Write `__init__.py`**

In `research/lib/__init__.py`:

```python
"""research.lib — shared utilities for research scripts.

Usage:
    from research.lib import query_df, outcomes_query, ttest_1s, bh_fdr, write_markdown
"""

from research.lib.audit import assert_no_inflation
from research.lib.db import connect_db, query_df
from research.lib.io import format_stats_table, output_dir, write_csv, write_markdown
from research.lib.query import SAFE_JOIN, outcomes_query, session_col, with_dst_split
from research.lib.stats import (
    bh_fdr,
    compute_metrics,
    expanding_stat,
    mannwhitney_2s,
    ttest_1s,
    year_by_year,
)

__all__ = [
    "assert_no_inflation",
    "connect_db",
    "query_df",
    "format_stats_table",
    "output_dir",
    "write_csv",
    "write_markdown",
    "SAFE_JOIN",
    "outcomes_query",
    "session_col",
    "with_dst_split",
    "bh_fdr",
    "compute_metrics",
    "expanding_stat",
    "mannwhitney_2s",
    "ttest_1s",
    "year_by_year",
]
```

**Step 4: Run full test suite**

```bash
python -m pytest tests/test_research/test_lib.py -v
```
Expected: all passed

**Step 5: Run project-wide tests to verify no regressions**

```bash
python -m pytest tests/ --ignore=tests/test_trader_logic.py --ignore=tests/test_trading_app/test_integration.py --ignore=tests/test_trading_app/test_paper_trader.py -x -q -n 4 --dist loadscope
```
Expected: all passed

**Step 6: Commit**

```bash
git add research/lib/__init__.py tests/test_research/test_lib.py
git commit -m "feat(research-lib): wire up __init__.py re-exports, complete package"
```

---

### Task 8: Final Verification + Ruff

**Step 1: Run ruff on new code**

```bash
ruff check research/lib/ tests/test_research/test_lib.py
```
Expected: no errors

**Step 2: Run drift check**

```bash
python pipeline/check_drift.py
```
Expected: all passed

**Step 3: Run full pre-commit suite**

```bash
python -m pytest tests/ --ignore=tests/test_trader_logic.py --ignore=tests/test_trading_app/test_integration.py --ignore=tests/test_trading_app/test_paper_trader.py -x -q -n 4 --dist loadscope
```
Expected: all passed

---

## Implementation Notes

**Reference files for extracting implementations:**
- `ttest_1s`: `research/research_compressed_spring.py:71-77` (canonical copy)
- `bh()`: `research/research_compressed_spring.py:90-104` (canonical copy)
- `SAFE_JOIN`: `research/multi_instrument_scan.py:21-27` (only named constant)
- `OUTPUT_DIR pattern`: `research/research_avoid_crosscheck.py:40-46` (typical boilerplate)
- `DB connection pattern`: `research/research_dst_edge_audit.py:39` (uses `pipeline.paths`)

**DST column names in daily_features:** `us_dst` (boolean), `uk_dst` (boolean) — built by `pipeline/build_daily_features.py:832-833`.

**DST regime mapping:** US for sessions 0900/0030/2300, UK for 1800 — per `CLAUDE.md` and `pipeline-patterns.md`.

**Do NOT:**
- Retrofit existing scripts (forward-only adoption)
- Create a base class framework
- Add calendar event helpers (only 2 scripts use these)
- Touch `research/_alt_strategy_utils.py`
