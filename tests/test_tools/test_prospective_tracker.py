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
            orb_CME_REOPEN_outcome VARCHAR,
            orb_CME_REOPEN_size DOUBLE
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
            pnl_r DOUBLE,
            ambiguous_bar BOOLEAN DEFAULT FALSE,
            ts_outcome VARCHAR,
            ts_pnl_r DOUBLE,
            ts_exit_ts TIMESTAMPTZ
        )
    """)

    # Prospective signals table
    con.execute("""
        CREATE TABLE prospective_signals (
            signal_id        VARCHAR NOT NULL,
            trading_day      DATE NOT NULL,
            symbol           VARCHAR NOT NULL,
            session          VARCHAR NOT NULL,
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

    # Insert 5 days of synthetic data (LOWERCASE outcomes)
    # Day 1: win (sets up LAG for Day 2)
    # Day 2: prev=win, loss outcome -> not qualifying (prev != loss)
    # Day 3: prev=loss, win outcome, orb_size=5.0 -> QUALIFYING
    # Day 4: prev=win, win outcome -> not qualifying
    # Day 5: prev=win, loss outcome, orb_size=3.5 -> not qualifying (orb < 4.0)
    days = [
        (datetime.date(2025, 1, 1), "win",  6.0, "win",   2.0),
        (datetime.date(2025, 1, 2), "loss", 5.5, "loss", -1.0),
        (datetime.date(2025, 1, 3), "win",  5.0, "win",   2.0),
        (datetime.date(2025, 1, 4), "win",  4.5, "win",   1.5),
        (datetime.date(2025, 1, 5), "loss", 3.5, "loss", -1.0),
    ]

    for day, df_outcome, orb_size, oo_outcome, pnl in days:
        con.execute("""
            INSERT INTO daily_features (trading_day, symbol, orb_minutes, orb_CME_REOPEN_outcome, orb_CME_REOPEN_size)
            VALUES (?, 'MGC', 5, ?, ?)
        """, [day, df_outcome, orb_size])

        con.execute("""
            INSERT INTO orb_outcomes (trading_day, symbol, orb_label, orb_minutes, entry_model, confirm_bars, rr_target, outcome, pnl_r)
            VALUES (?, 'MGC', 'CME_REOPEN', 5, 'E2', 1, 2.0, ?, ?)
        """, [day, oo_outcome, pnl])

    con.commit()
    return con


class TestFetchQualifyingDays:
    def test_finds_prev_loss_days(self, tracker_db):
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        rows = fetch_qualifying_days(tracker_db, sig)
        # Only Day 3 qualifies: prev=loss (Day 2 was loss), orb_size=5.0 >= 4.0
        assert len(rows) == 1
        assert rows[0][0] == datetime.date(2025, 1, 3)

    def test_excludes_small_orb(self, tracker_db):
        """Day 5 has prev=win anyway, but also orb < 4.0."""
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        rows = fetch_qualifying_days(tracker_db, sig)
        dates = [r[0] for r in rows]
        assert datetime.date(2025, 1, 5) not in dates


class TestPopulateSignal:
    def test_populates_table(self, tracker_db):
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        freeze = datetime.date(2025, 1, 3)
        n = populate_signal(tracker_db, "MGC_CME_REOPEN_PREV_LOSS", sig, freeze)
        assert n == 1

        rows = tracker_db.execute(
            "SELECT * FROM prospective_signals WHERE signal_id = 'MGC_CME_REOPEN_PREV_LOSS'"
        ).fetchall()
        assert len(rows) == 1

    def test_retrospective_tag(self, tracker_db):
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        freeze = datetime.date(2025, 6, 1)  # after all data
        populate_signal(tracker_db, "MGC_CME_REOPEN_PREV_LOSS", sig, freeze)

        rows = tracker_db.execute(
            "SELECT is_prospective FROM prospective_signals WHERE signal_id = 'MGC_CME_REOPEN_PREV_LOSS'"
        ).fetchall()
        assert all(r[0] == False for r in rows)

    def test_prospective_tag(self, tracker_db):
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        freeze = datetime.date(2025, 1, 1)  # before all data
        populate_signal(tracker_db, "MGC_CME_REOPEN_PREV_LOSS", sig, freeze)

        rows = tracker_db.execute(
            "SELECT is_prospective FROM prospective_signals WHERE signal_id = 'MGC_CME_REOPEN_PREV_LOSS'"
        ).fetchall()
        assert all(r[0] == True for r in rows)

    def test_idempotent(self, tracker_db):
        sig = SIGNALS["MGC_CME_REOPEN_PREV_LOSS"]
        freeze = datetime.date(2025, 6, 1)
        populate_signal(tracker_db, "MGC_CME_REOPEN_PREV_LOSS", sig, freeze)
        populate_signal(tracker_db, "MGC_CME_REOPEN_PREV_LOSS", sig, freeze)

        rows = tracker_db.execute(
            "SELECT COUNT(*) FROM prospective_signals WHERE signal_id = 'MGC_CME_REOPEN_PREV_LOSS'"
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
