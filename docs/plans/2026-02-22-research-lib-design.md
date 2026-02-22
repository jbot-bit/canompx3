# Research Library Design — `research/lib/`

**Date:** 2026-02-22
**Status:** Approved
**Scope:** Forward-only — new scripts use the library; existing scripts migrate opportunistically
**Approach:** Thin wrappers (no base class framework)

## Problem

~100 research scripts duplicate the same patterns: DB connections, statistical tests, query templates,
output boilerplate. Key risks from copy-paste:

- **Bug propagation:** 3 different BH FDR implementations exist (manual, numpy, scipy API)
- **Join inflation:** Missing `orb_minutes` in triple-join triples row count silently
- **Lookahead leaks:** No shared expanding-window helpers
- **DST violations:** Easy to forget mandatory DST regime splitting

## Package Structure

```
research/lib/
  __init__.py          # Re-exports common functions
  db.py                # Connection lifecycle (~40 lines)
  stats.py             # Statistical tests + metrics (~120 lines)
  query.py             # Composable query builder (~90 lines)
  audit.py             # Join verification (~30 lines)
  io.py                # Output dir + formatters (~60 lines)
```

Total: ~340 lines across 5 modules + `__init__.py`.

## Module Specifications

### `db.py` — Database Connection

```python
from research.lib.db import connect_db, query_df

# One-shot convenience (opens, executes, closes)
df = query_df("SELECT * FROM orb_outcomes WHERE symbol = 'MGC'")

# Multi-query context manager
with connect_db() as con:
    df1 = con.execute(sql1).fetchdf()
    df2 = con.execute(sql2).fetchdf()
```

**Functions:**
- `connect_db(read_only=True) -> contextmanager[DuckDBPyConnection]` — uses `pipeline.paths.GOLD_DB_PATH`
- `query_df(sql, params=None) -> pd.DataFrame` — one-shot open/execute/close

**Eliminates:** `os.environ.get("DUCKDB_PATH", ...)` scattered in 50+ scripts, leaked connections.

### `stats.py` — Statistical Tests & Metrics

```python
from research.lib.stats import ttest_1s, bh_fdr, compute_metrics, year_by_year, expanding_stat

n, mean, wr, t, p = ttest_1s(pnl_array)
rejected_indices = bh_fdr(p_values, q=0.10)
metrics = compute_metrics(pnl_array)  # {n, wr, avg_r, sharpe, max_dd, total_r}
yearly_df = year_by_year(df, date_col="trading_day", value_col="pnl_r")
df["atr_exp"] = expanding_stat(df, col="atr", min_periods=20)
```

**Functions:**
- `ttest_1s(arr, mu=0.0) -> (n, mean, win_rate, t_stat, p_value)` — canonical one-sample t-test
- `bh_fdr(p_values, q=0.10) -> set[int]` — Benjamini-Hochberg, returns rejected hypothesis indices
- `compute_metrics(pnls) -> dict` — win_rate, avg_r, sharpe, max_dd, total_r, n
- `mannwhitney_2s(a, b) -> (u_stat, p_value)` — two-sample comparison
- `year_by_year(df, date_col, value_col) -> DataFrame` — per-year n, mean, wr, p-value
- `expanding_stat(df, col, min_periods) -> Series` — expanding-window statistic (no lookahead)

**Eliminates:** 2 identical `ttest_1s` copies, 3 divergent BH implementations, inline metrics in 35+ scripts.

### `query.py` — Composable Query Builder

```python
from research.lib.query import outcomes_query, session_col, with_dst_split, SAFE_JOIN

sql = outcomes_query(
    instrument="MGC", session="1000", entry_model="E0",
    extra_cols=["d.atr_5d", "d.rsi_5d"],
    filters=["d.orb_1000_size >= 4", "d.dow_label IN ('Mon','Tue')"],
    date_range=("2021-01-01", "2025-12-31"),
)

col = session_col("1000", "size")  # -> "orb_1000_size"

dst_on_sql, dst_off_sql = with_dst_split(base_sql, session="0900", regime_source="US")
```

**Constants:**
- `SAFE_JOIN` — the canonical triple-join fragment:
  ```sql
  ON o.trading_day = d.trading_day AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
  ```

**Functions:**
- `outcomes_query(instrument, session, entry_model, extra_cols=None, filters=None, date_range=None) -> str` — builds the standard outcomes+features query with safe JOIN baked in
- `session_col(orb_label, stem) -> str` — returns `f"orb_{orb_label}_{stem}"`
- `with_dst_split(base_sql, session, regime_source) -> (str, str)` — wraps query with DST ON/OFF filtering

**Eliminates:** 17+ hand-written load functions, missing `orb_minutes` join bugs, DST split omissions.

### `audit.py` — Join Verification

```python
from research.lib.audit import assert_no_inflation

n_raw = len(outcomes_df)
merged = outcomes_df.merge(features_df, on=["trading_day", "symbol", "orb_minutes"])
assert_no_inflation(n_raw, len(merged), context="my_analysis")
```

**Functions:**
- `assert_no_inflation(n_before, n_after, context="") -> None` — raises `ValueError` if `n_after > n_before`

**Eliminates:** Silent row-count inflation from missing join columns.

### `io.py` — Output & Formatting

```python
from research.lib.io import output_dir, write_csv, write_markdown, format_stats_table

write_csv(df, "compressed_spring_results.csv")
write_markdown(report_text, "compressed_spring_findings.md")

table_md = format_stats_table({
    "ALL": {"n": 500, "mean_r": 0.12, "win_rate": 0.54, "p_value": 0.003},
    "G4":  {"n": 320, "mean_r": 0.18, "win_rate": 0.57, "p_value": 0.001},
})
```

**Functions:**
- `output_dir() -> Path` — returns `research/output/`, creates if needed
- `write_csv(df, filename) -> Path` — writes DataFrame to output dir
- `write_markdown(text, filename) -> Path` — writes text to output dir
- `format_stats_table(results_dict) -> str` — formats stats as markdown table (N, mean_R, WR, p-value)

**Eliminates:** `OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` in 70+ scripts, inconsistent table formatting.

## Usage Example

Before (current ~25 lines of boilerplate):
```python
import os, sys, duckdb
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "gold.db"))
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ... 15 more lines of connection, query, stats boilerplate
```

After (~5 lines):
```python
from research.lib import query_df, outcomes_query, ttest_1s, bh_fdr, write_markdown

def main():
    df = query_df(outcomes_query("MGC", "1000", "E0", extra_cols=["d.atr_5d"]))
    n, mean, wr, t, p = ttest_1s(df["pnl_r"])
    write_markdown(report, "my_findings.md")
```

## What's NOT in Scope

- **Retrofitting existing scripts** — migrate opportunistically when touching them
- **Base class / framework** — no `ResearchScript` inheritance
- **Calendar events** (FOMC/NFP) — only 2 scripts use these, not worth centralizing yet
- **`_alt_strategy_utils.py`** — stays as-is, used by 2 archive scripts only

## Testing

- `tests/test_research/test_lib.py` — unit tests for stats functions (known inputs/outputs), query builder (SQL generation), audit assertions
- No integration tests against live DB — research lib is purely functional
