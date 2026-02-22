# Prospective Signal Tracker — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone script + DB table to prospectively track the MGC 0900 prev=LOSS signal toward N=100 confirmation.

**Architecture:** New `prospective_signals` table created via migration in `init_db.py`. Standalone script `scripts/tools/prospective_tracker.py` queries `daily_features` + `orb_outcomes` with LAG(), populates the table, and prints a stats report. Idempotent DELETE+INSERT per run.

**Tech Stack:** Python, DuckDB, scipy.stats (t-test)

---

### Task 1: Schema Migration

**Files:**
- Modify: `pipeline/init_db.py` (add migration block at ~line 372, before `con.commit()`)
- Test: `tests/test_pipeline/test_schema.py`

**Step 1: Write the failing test**

Add to `tests/test_pipeline/test_schema.py`:

```python
def test_creates_prospective_signals_table(self, tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path, force=False)

    con = duckdb.connect(str(db_path))
    tables = [t[0] for t in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()]
    con.close()

    assert "prospective_signals" in tables

def test_prospective_signals_columns(self, tmp_path):
    db_path = tmp_path / "test.db"
    init_db(db_path, force=False)

    con = duckdb.connect(str(db_path))
    cols = [c[0] for c in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'prospective_signals'"
    ).fetchall()]
    con.close()

    expected = ["signal_id", "trading_day", "symbol", "session",
                "prev_day_outcome", "orb_size", "entry_model",
                "confirm_bars", "rr_target", "outcome", "pnl_r",
                "is_prospective", "freeze_date", "created_at"]
    for col in expected:
        assert col in cols, f"Missing column: {col}"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_schema.py::TestInitDb::test_creates_prospective_signals_table -v`
Expected: FAIL — `prospective_signals` not in tables

**Step 3: Add CREATE TABLE to init_db.py**

Add this constant near the top of `pipeline/init_db.py` (after existing schema constants):

```python
PROSPECTIVE_SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prospective_signals (
    signal_id        VARCHAR NOT NULL,
    trading_day      DATE NOT NULL,
    symbol           VARCHAR NOT NULL,
    session          INTEGER NOT NULL,
    prev_day_outcome VARCHAR NOT NULL,
    orb_size         DOUBLE,
    entry_model      VARCHAR NOT NULL,
    confirm_bars     INTEGER NOT NULL,
    rr_target        DOUBLE NOT NULL,
    outcome          VARCHAR,
    pnl_r            DOUBLE,
    is_prospective   BOOLEAN NOT NULL,
    freeze_date      DATE NOT NULL,
    created_at       TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (signal_id, trading_day)
);
"""
```

Add the table creation in `init_db()`, after the GARCH migration block (~line 372), before `con.commit()`:

```python
con.execute(PROSPECTIVE_SIGNALS_SCHEMA)
logger.info("  prospective_signals: created (or already exists)")
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline/test_schema.py -v`
Expected: All PASS including the two new tests

**Step 5: Run init_db against gold.db to create the table**

Run: `python pipeline/init_db.py`
Expected: Log line `prospective_signals: created (or already exists)`

**Step 6: Commit**

```bash
git add pipeline/init_db.py tests/test_pipeline/test_schema.py
git commit -m "feat: add prospective_signals table schema"
```

---

### Task 2: Core Tracker Script — Query + Populate

**Files:**
- Create: `scripts/tools/prospective_tracker.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""
Prospective tracker for prior-day outcome signals.

Tracks qualifying days for frozen hypotheses, accumulating prospective
evidence toward confirmation thresholds (N=100 re-evaluation, N=150 deployment).

Usage:
    python scripts/tools/prospective_tracker.py
    python scripts/tools/prospective_tracker.py --freeze-date 2026-02-22
"""
import argparse
import datetime
import logging

import duckdb
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Signal definitions
# ---------------------------------------------------------------------------
DEFAULT_FREEZE_DATE = datetime.date(2026, 2, 22)

SIGNALS = {
    "MGC_0900_PREV_LOSS": {
        "symbol": "MGC",
        "session": 900,
        "orb_label": "0900",
        "prev_outcome_filter": "LOSS",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 2.0,
        "min_orb_pts": 4.0,
    },
}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def fetch_qualifying_days(con, sig: dict) -> list[tuple]:
    """
    Return qualifying days for a prior-day outcome signal.

    Uses LAG() on daily_features (where orb_{session}_outcome lives),
    then joins orb_outcomes for the realized pnl_r.
    Mirrors research/research_prev_day_signal.py query pattern.
    """
    label = sig["orb_label"]
    outcome_col = f"orb_{label}_outcome"
    size_col = f"orb_{label}_size"

    sql = f"""
        WITH lag_feats AS (
            SELECT
                trading_day,
                symbol,
                {outcome_col}  AS curr_outcome,
                LAG({outcome_col}) OVER (
                    PARTITION BY symbol ORDER BY trading_day
                )              AS prev_outcome,
                {size_col}     AS orb_size_pts
            FROM daily_features
            WHERE orb_minutes = 5
              AND symbol = ?
        )
        SELECT
            l.trading_day,
            l.prev_outcome,
            l.orb_size_pts,
            o.outcome,
            o.pnl_r
        FROM lag_feats l
        JOIN orb_outcomes o
          ON  l.trading_day  = o.trading_day
          AND l.symbol       = o.symbol
          AND o.orb_label    = ?
          AND o.orb_minutes  = 5
          AND o.entry_model  = ?
          AND o.rr_target    = ?
          AND o.confirm_bars = ?
          AND o.outcome IS NOT NULL
          AND o.pnl_r   IS NOT NULL
        WHERE l.prev_outcome = ?
          AND l.orb_size_pts IS NOT NULL
          AND l.orb_size_pts >= ?
        ORDER BY l.trading_day
    """
    return con.execute(sql, [
        sig["symbol"],
        sig["orb_label"],
        sig["entry_model"],
        sig["rr_target"],
        sig["confirm_bars"],
        sig["prev_outcome_filter"],
        sig["min_orb_pts"],
    ]).fetchall()


# ---------------------------------------------------------------------------
# Populate
# ---------------------------------------------------------------------------
def populate_signal(con, signal_id: str, sig: dict, freeze_date: datetime.date):
    """DELETE+INSERT all qualifying days for this signal."""
    rows = fetch_qualifying_days(con, sig)
    logger.info(f"{signal_id}: {len(rows)} qualifying days found")

    con.execute(
        "DELETE FROM prospective_signals WHERE signal_id = ?",
        [signal_id],
    )

    for trading_day, prev_outcome, orb_size, outcome, pnl_r in rows:
        is_prospective = trading_day >= freeze_date
        con.execute("""
            INSERT INTO prospective_signals
                (signal_id, trading_day, symbol, session,
                 prev_day_outcome, orb_size, entry_model,
                 confirm_bars, rr_target, outcome, pnl_r,
                 is_prospective, freeze_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            signal_id,
            trading_day,
            sig["symbol"],
            sig["session"],
            prev_outcome,
            orb_size,
            sig["entry_model"],
            sig["confirm_bars"],
            sig["rr_target"],
            outcome,
            pnl_r,
            is_prospective,
            freeze_date,
        ])

    con.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats(pnl_values: list[float]) -> dict:
    """Compute N, avgR, WR, t-stat, p-value from a list of pnl_r values."""
    n = len(pnl_values)
    if n == 0:
        return {"N": 0, "avgR": 0.0, "WR": 0.0, "t": 0.0, "p": 1.0}

    avg_r = sum(pnl_values) / n
    wr = sum(1 for x in pnl_values if x > 0) / n * 100

    if n >= 2:
        t_stat, p_val = stats.ttest_1samp(pnl_values, 0.0)
    else:
        t_stat, p_val = 0.0, 1.0

    return {"N": n, "avgR": avg_r, "WR": wr, "t": t_stat, "p": p_val}


def compute_yearly_stats(rows: list[tuple]) -> dict:
    """Group by year and compute stats per year."""
    yearly = {}
    for trading_day, pnl_r in rows:
        yr = trading_day.year if hasattr(trading_day, 'year') else int(str(trading_day)[:4])
        yearly.setdefault(yr, []).append(pnl_r)
    return {yr: compute_stats(vals) for yr, vals in sorted(yearly.items())}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(con, signal_id: str, sig: dict, freeze_date: datetime.date):
    """Print console report for a signal."""
    retro_rows = con.execute("""
        SELECT trading_day, pnl_r FROM prospective_signals
        WHERE signal_id = ? AND is_prospective = FALSE
        ORDER BY trading_day
    """, [signal_id]).fetchall()

    prosp_rows = con.execute("""
        SELECT trading_day, pnl_r FROM prospective_signals
        WHERE signal_id = ? AND is_prospective = TRUE
        ORDER BY trading_day
    """, [signal_id]).fetchall()

    retro_pnl = [r[1] for r in retro_rows]
    prosp_pnl = [r[1] for r in prosp_rows]
    combined_pnl = retro_pnl + prosp_pnl

    retro_s = compute_stats(retro_pnl)
    prosp_s = compute_stats(prosp_pnl)
    combined_s = compute_stats(combined_pnl)

    label = sig["orb_label"]
    em = sig["entry_model"]
    cb = sig["confirm_bars"]
    rr = sig["rr_target"]
    prev = sig["prev_outcome_filter"]
    g = sig["min_orb_pts"]

    print()
    print(f"=== Prospective Signal Tracker: {signal_id} ===")
    print(f"Signal: MGC {label} {em} CB{cb} RR{rr} G{int(g)}+ | prev_day = {prev}")
    print(f"Freeze date: {freeze_date}")
    print()

    def fmt_stats(s):
        return f"  N={s['N']}  avgR={s['avgR']:+.3f}  WR={s['WR']:.1f}%  t={s['t']:+.2f}  p={s['p']:.4f}"

    print("--- RETROSPECTIVE (before freeze) ---")
    print(fmt_stats(retro_s))
    print()
    print("--- PROSPECTIVE (after freeze) ---")
    if prosp_s["N"] == 0:
        print("  No prospective data yet (freeze date is today or in the future)")
    else:
        print(fmt_stats(prosp_s))
    print()
    print("--- COMBINED ---")
    print(fmt_stats(combined_s))
    print()

    # Progress bar
    prosp_n = prosp_s["N"]
    target = 100
    pct = min(prosp_n / target * 100, 100)
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "=" * filled + "." * (bar_len - filled)
    print("--- PROGRESS ---")
    print(f"  Prospective N: {prosp_n:3d} / {target}  [{bar}] {pct:.0f}%")
    print(f"  Next milestone:  N=100 -> formal re-evaluation")
    print(f"  Final milestone: N=150 -> full validation pipeline")

    if prosp_n >= 150:
        print()
        print("  *** THRESHOLD REACHED: N=150 prospective ***")
        print("  *** ACTION: Run full validation pipeline  ***")
        print("  *** Consider 1.5x position-size overlay   ***")
    elif prosp_n >= 100:
        print()
        print("  *** THRESHOLD REACHED: N=100 prospective ***")
        print("  *** ACTION: Formal re-evaluation required ***")

    # Year-by-year (prospective only)
    if prosp_rows:
        print()
        print("--- YEAR-BY-YEAR (prospective only) ---")
        yearly = compute_yearly_stats(prosp_rows)
        for yr, s in yearly.items():
            print(f"  {yr}:  N={s['N']}  avgR={s['avgR']:+.3f}  WR={s['WR']:.1f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prospective tracker for prior-day outcome signals"
    )
    parser.add_argument(
        "--freeze-date",
        type=lambda s: datetime.date.fromisoformat(s),
        default=DEFAULT_FREEZE_DATE,
        help="Date from which tracking is prospective (default: 2026-02-22)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(GOLD_DB_PATH),
        help="Path to DuckDB database",
    )
    args = parser.parse_args()

    con = duckdb.connect(args.db_path)

    for signal_id, sig in SIGNALS.items():
        populate_signal(con, signal_id, sig, args.freeze_date)
        print_report(con, signal_id, sig, args.freeze_date)

    con.close()


if __name__ == "__main__":
    main()
```

**Step 2: Run the script against gold.db**

Run: `python scripts/tools/prospective_tracker.py`
Expected: Report showing ~49 retrospective days, 0 prospective days (freeze date is today)

**Step 3: Verify N matches research findings**

Check that retrospective N=49 and avgR is close to +0.585. If not, investigate query differences vs `research/research_prev_day_signal.py`.

**Step 4: Commit**

```bash
git add scripts/tools/prospective_tracker.py
git commit -m "feat: prospective tracker for prior-day outcome signal"
```

---

### Task 3: Integration Test

**Files:**
- Create: `tests/test_tools/test_prospective_tracker.py`

**Step 1: Write integration test**

```python
"""
Tests for scripts/tools/prospective_tracker.py

Uses an in-memory DuckDB with synthetic data to verify:
- Qualifying day detection via LAG()
- Retrospective vs prospective tagging
- Stats computation
"""
import datetime
import duckdb
import pytest

from scripts.tools.prospective_tracker import (
    fetch_qualifying_days,
    populate_signal,
    compute_stats,
    SIGNALS,
)


@pytest.fixture
def tracker_db(tmp_path):
    """Create a minimal DB with daily_features + orb_outcomes for testing."""
    db_path = tmp_path / "test.db"
    con = duckdb.connect(str(db_path))

    # Minimal daily_features
    con.execute("""
        CREATE TABLE daily_features (
            trading_day DATE,
            symbol VARCHAR,
            orb_minutes INT,
            orb_0900_outcome VARCHAR,
            orb_0900_size DOUBLE
        )
    """)

    # Minimal orb_outcomes
    con.execute("""
        CREATE TABLE orb_outcomes (
            trading_day DATE,
            symbol VARCHAR,
            orb_label VARCHAR,
            orb_minutes INT,
            entry_model VARCHAR,
            confirm_bars INT,
            rr_target DOUBLE,
            outcome VARCHAR,
            pnl_r DOUBLE
        )
    """)

    # Prospective signals table
    con.execute("""
        CREATE TABLE prospective_signals (
            signal_id        VARCHAR NOT NULL,
            trading_day      DATE NOT NULL,
            symbol           VARCHAR NOT NULL,
            session          INTEGER NOT NULL,
            prev_day_outcome VARCHAR NOT NULL,
            orb_size         DOUBLE,
            entry_model      VARCHAR NOT NULL,
            confirm_bars     INTEGER NOT NULL,
            rr_target        DOUBLE NOT NULL,
            outcome          VARCHAR,
            pnl_r            DOUBLE,
            is_prospective   BOOLEAN NOT NULL,
            freeze_date      DATE NOT NULL,
            created_at       TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (signal_id, trading_day)
        )
    """)

    # Insert 5 days of synthetic data
    # Day 1: WIN (sets up LAG for Day 2)
    # Day 2: prev=WIN, LOSS outcome -> not qualifying (prev != LOSS)
    # Day 3: prev=LOSS, WIN outcome, orb_size=5.0 -> QUALIFYING
    # Day 4: prev=WIN, WIN outcome -> not qualifying
    # Day 5: prev=WIN, LOSS outcome -> not qualifying
    days = [
        (datetime.date(2025, 1, 1), "WIN",  6.0, "WIN",   2.0),
        (datetime.date(2025, 1, 2), "LOSS", 5.5, "LOSS", -1.0),
        (datetime.date(2025, 1, 3), "WIN",  5.0, "WIN",   2.0),
        (datetime.date(2025, 1, 4), "WIN",  4.5, "WIN",   1.5),
        (datetime.date(2025, 1, 5), "LOSS", 3.5, "LOSS", -1.0),  # orb < 4.0, not qualifying
    ]

    for day, df_outcome, orb_size, oo_outcome, pnl in days:
        con.execute("""
            INSERT INTO daily_features (trading_day, symbol, orb_minutes, orb_0900_outcome, orb_0900_size)
            VALUES (?, 'MGC', 5, ?, ?)
        """, [day, df_outcome, orb_size])

        con.execute("""
            INSERT INTO orb_outcomes (trading_day, symbol, orb_label, orb_minutes, entry_model, confirm_bars, rr_target, outcome, pnl_r)
            VALUES (?, 'MGC', '0900', 5, 'E0', 1, 2.0, ?, ?)
        """, [day, oo_outcome, pnl])

    con.commit()
    return con


class TestFetchQualifyingDays:
    def test_finds_prev_loss_days(self, tracker_db):
        sig = SIGNALS["MGC_0900_PREV_LOSS"]
        rows = fetch_qualifying_days(tracker_db, sig)
        # Only Day 3 qualifies: prev=LOSS (Day 2 was LOSS), orb_size=5.0 >= 4.0
        assert len(rows) == 1
        assert rows[0][0] == datetime.date(2025, 1, 3)

    def test_excludes_small_orb(self, tracker_db):
        """Day 5 has prev=WIN anyway, but also orb < 4.0 — double exclusion."""
        sig = SIGNALS["MGC_0900_PREV_LOSS"]
        rows = fetch_qualifying_days(tracker_db, sig)
        dates = [r[0] for r in rows]
        assert datetime.date(2025, 1, 5) not in dates


class TestPopulateSignal:
    def test_populates_table(self, tracker_db):
        sig = SIGNALS["MGC_0900_PREV_LOSS"]
        freeze = datetime.date(2025, 1, 3)
        n = populate_signal(tracker_db, "MGC_0900_PREV_LOSS", sig, freeze)
        assert n == 1

        rows = tracker_db.execute(
            "SELECT * FROM prospective_signals WHERE signal_id = 'MGC_0900_PREV_LOSS'"
        ).fetchall()
        assert len(rows) == 1

    def test_retrospective_tag(self, tracker_db):
        sig = SIGNALS["MGC_0900_PREV_LOSS"]
        freeze = datetime.date(2025, 6, 1)  # after all data
        populate_signal(tracker_db, "MGC_0900_PREV_LOSS", sig, freeze)

        rows = tracker_db.execute(
            "SELECT is_prospective FROM prospective_signals WHERE signal_id = 'MGC_0900_PREV_LOSS'"
        ).fetchall()
        assert all(r[0] == False for r in rows)

    def test_idempotent(self, tracker_db):
        sig = SIGNALS["MGC_0900_PREV_LOSS"]
        freeze = datetime.date(2025, 6, 1)
        populate_signal(tracker_db, "MGC_0900_PREV_LOSS", sig, freeze)
        populate_signal(tracker_db, "MGC_0900_PREV_LOSS", sig, freeze)

        rows = tracker_db.execute(
            "SELECT COUNT(*) FROM prospective_signals WHERE signal_id = 'MGC_0900_PREV_LOSS'"
        ).fetchone()
        assert rows[0] == 1  # no duplicates


class TestComputeStats:
    def test_empty(self):
        s = compute_stats([])
        assert s["N"] == 0
        assert s["p"] == 1.0

    def test_positive_signal(self):
        s = compute_stats([1.0, 2.0, 0.5, 1.5])
        assert s["N"] == 4
        assert s["avgR"] == pytest.approx(1.25)
        assert s["WR"] == 100.0
        assert s["p"] < 0.05

    def test_single_value(self):
        s = compute_stats([1.0])
        assert s["N"] == 1
        assert s["p"] == 1.0  # can't t-test with N=1
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools/test_prospective_tracker.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_tools/test_prospective_tracker.py
git commit -m "test: integration tests for prospective tracker"
```

---

### Task 4: Smoke Test Against Production DB

**Step 1: Run the tracker**

Run: `python scripts/tools/prospective_tracker.py`

**Step 2: Verify output**

Check:
- Retrospective N should be ~49 (matching research findings)
- avgR should be close to +0.585
- Prospective N should be 0 (freeze date is today)
- No errors or warnings

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass (no regressions)

**Step 4: Final commit (all files together)**

```bash
git add pipeline/init_db.py scripts/tools/prospective_tracker.py tests/
git commit -m "feat: prior-day outcome signal prospective tracker

Adds prospective_signals table and standalone tracker script for
MGC 0900 prev=LOSS signal (BH FDR survivor, N=49, avgR=+0.585R).
Tracks toward N=100 confirmation threshold."
```
