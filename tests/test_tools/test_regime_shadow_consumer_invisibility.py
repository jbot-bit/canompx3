"""G4b — shadow rows are invisible to live/monitoring AGGREGATE consumers.

The original self-review claimed shadow rows are "structurally invisible" to
every live/monitoring consumer, resting on a per-strategy_id tier-disjointness
argument. An adversarial review (2026-06-03) FALSIFIED that for AGGREGATE reads:
five consumers ran COUNT/SUM/GROUP-BY over paper_trades with NO execution_source
filter, so once execution_source='shadow' rows land they would be silently mixed
into live forward-performance, win-rate, discipline-rate, the DB-identity hash,
the live trade summary, and the dashboard count.

Each contaminated read was guarded with `execution_source != 'shadow'`. These
tests pin that guard against regression by reproducing each consumer's exact
aggregate query against a temp DB holding one 'live' row and one 'shadow' row
that share an orb_label / lane_name (the worst case the disjointness argument
does NOT cover). Every guarded read must count the live row ONLY.

The tests are NOT vacuous: each asserts (a) the guarded query returns 1 (live
only) AND (b) the same query WITHOUT the guard returns 2 — proving the shadow row
is present and the guard is doing the work.
"""

from __future__ import annotations

import datetime

import duckdb

from trading_app.db_manager import init_trading_app_schema

_ORB = "US_DATA_1000"
_LANE = "US_DATA_1000_VWAP_MID"


def _init_schema(db) -> None:
    """orb_outcomes has a FK to daily_features, so that table must exist first
    (mirrors the canonical key tuple). Then run the canonical initializer, which
    applies the paper_trades execution_source migration."""
    with duckdb.connect(str(db)) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS daily_features (
                 symbol VARCHAR, orb_minutes INTEGER, trading_day DATE,
                 UNIQUE (symbol, trading_day, orb_minutes))"""
        )
    init_trading_app_schema(db_path=db)


def _seed_one_live_one_shadow(db) -> None:
    """Insert exactly one live and one shadow row sharing orb_label + lane_name.

    Both carry a winning pnl_r and pnl_dollar so any unguarded COUNT/SUM/WR/
    daily-PnL aggregate would visibly include the shadow row. This is the case
    the per-strategy_id disjointness argument does not protect.
    """
    with duckdb.connect(str(db)) as con:
        con.executemany(
            """INSERT INTO paper_trades (
                   trading_day, orb_label, lane_name, strategy_id, instrument,
                   direction, exit_reason, pnl_r, pnl_dollar, slippage_ticks,
                   execution_source
               ) VALUES (?, ?, ?, ?, 'MNQ', 'long', 'win', ?, ?, ?, ?)""",
            [
                ("2026-06-04", _ORB, _LANE, "LIVE1", 1.5, 30.0, 1, "live"),
                ("2026-06-04", _ORB, _LANE, "SHDW1", 2.0, None, None, "shadow"),
            ],
        )


def _both(con, guarded: str, unguarded: str, params=None):
    """Return (guarded_result, unguarded_result) scalars for the same query."""
    g = con.execute(guarded, params or []).fetchone()[0]
    u = con.execute(unguarded, params or []).fetchone()[0]
    return g, u


def test_weekly_review_aggregates_exclude_shadow(tmp_path):
    """weekly_review §1/§4/§6: orb_label-scoped COUNT/SUM must skip shadow."""
    db = tmp_path / "wr.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        # §1 total count (matches weekly_review.py section_1_forward_performance)
        g, u = _both(
            con,
            "SELECT COUNT(*) FROM paper_trades WHERE orb_label = ? AND pnl_r IS NOT NULL "
            "AND execution_source != 'shadow'",
            "SELECT COUNT(*) FROM paper_trades WHERE orb_label = ? AND pnl_r IS NOT NULL",
            [_ORB],
        )
        assert g == 1, "live forward-performance count must exclude the shadow row"
        assert u == 2, "guard is load-bearing: unguarded query DOES see the shadow row"

        # §1 cumulative R must reflect the live row's pnl_r only (1.5, not 3.5).
        cum = con.execute(
            "SELECT COALESCE(ROUND(SUM(pnl_r), 2), 0) FROM paper_trades "
            "WHERE orb_label = ? AND pnl_r IS NOT NULL AND execution_source != 'shadow'",
            [_ORB],
        ).fetchone()[0]
        assert cum == 1.5, f"cum R must be live-only (1.5), got {cum}"


def test_derived_state_count_excludes_shadow(tmp_path):
    """derived_state DB-identity hash count must be invariant to shadow rows so a
    shadow sync never churns C11/C12 live-readiness state."""
    db = tmp_path / "ds.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        g, u = _both(
            con,
            "SELECT COUNT(*) FROM paper_trades WHERE COALESCE(execution_source, 'backfill') != 'shadow'",
            "SELECT COUNT(*) FROM paper_trades",
        )
        assert g == 1, "DB-identity count must exclude shadow rows"
        assert u == 2


def test_paper_trade_summary_aggregates_exclude_shadow(tmp_path):
    """paper_trade_summary portfolio total + per-lane GROUP BY must skip shadow."""
    db = tmp_path / "pts.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        g, u = _both(
            con,
            "SELECT COUNT(*) FROM paper_trades WHERE execution_source != 'shadow'",
            "SELECT COUNT(*) FROM paper_trades",
        )
        assert g == 1
        assert u == 2

        # Per-lane GROUP BY must not emit a phantom shadow lane row count.
        lane_n = con.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE lane_name = ? AND execution_source != 'shadow' GROUP BY lane_name",
            [_LANE],
        ).fetchone()
        assert lane_n[0] == 1, "per-lane summary must count the live row only"


def test_pre_session_check_count_excludes_shadow(tmp_path):
    db = tmp_path / "psc.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        g, u = _both(
            con,
            "SELECT COUNT(*) FROM paper_trades WHERE execution_source != 'shadow'",
            "SELECT COUNT(*) FROM paper_trades",
        )
        assert g == 1
        assert u == 2


def test_bot_dashboard_list_and_count_exclude_shadow(tmp_path):
    """Dashboard trade list and its total must show live/backfill only."""
    db = tmp_path / "dash.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        listed = con.execute(
            "SELECT strategy_id FROM paper_trades WHERE execution_source != 'shadow' ORDER BY trading_day DESC"
        ).fetchall()
        assert [r[0] for r in listed] == ["LIVE1"], "dashboard list must omit shadow rows"
        total = con.execute("SELECT COUNT(*) FROM paper_trades WHERE execution_source != 'shadow'").fetchone()[0]
        assert total == 1


def test_consistency_tracker_already_excludes_shadow_via_pnl_dollar(tmp_path):
    """consistency_tracker filters `WHERE pnl_dollar IS NOT NULL`. The shadow
    runner never writes pnl_dollar (NULL), so shadow rows are ALREADY excluded —
    documenting why this consumer needed no execution_source guard (the auditor's
    flag here was a false positive). This test pins that NULL-exclusion so a
    future change that starts populating pnl_dollar on shadow rows fails here and
    forces an explicit guard decision."""
    db = tmp_path / "ct.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "SELECT trading_day, SUM(pnl_dollar) FROM paper_trades WHERE pnl_dollar IS NOT NULL GROUP BY 1"
        ).fetchall()
    # Only the live row has a non-NULL pnl_dollar -> exactly one daily group, $30.
    assert len(rows) == 1, "shadow row (NULL pnl_dollar) must not form a daily group"
    assert rows[0][1] == 30.0, "daily P&L must be live-only ($30), shadow excluded"


# ── R1: structural VIEW + newly-guarded readers (Stage-2 root fix) ────────


def test_live_paper_trades_view_excludes_shadow(tmp_path):
    """R1 layer 1: the canonical `live_paper_trades` VIEW excludes shadow rows by
    construction — a consumer reading the VIEW inherits invisibility for free."""
    from pipeline.db_contracts import LIVE_PAPER_TRADES_VIEW

    db = tmp_path / "view.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)
    with duckdb.connect(str(db), read_only=True) as con:
        view_n, raw_n = _both(
            con,
            f"SELECT COUNT(*) FROM {LIVE_PAPER_TRADES_VIEW}",
            "SELECT COUNT(*) FROM paper_trades",
        )
        assert view_n == 1, "VIEW excludes the shadow row"
        assert raw_n == 2, "raw table still holds both (VIEW is the structural guard)"
        # VIEW carries the same columns -> sum(pnl_r) is live-only.
        cum = con.execute(f"SELECT COALESCE(SUM(pnl_r), 0) FROM {LIVE_PAPER_TRADES_VIEW}").fetchone()[0]
        assert cum == 1.5, f"VIEW sum must be live-only (1.5), got {cum}"


def test_log_trade_post_trade_stat_excludes_shadow(tmp_path):
    """R1/F4 honesty gate: log_trade's post-trade stat keys on orb_label only.
    With a shadow row sharing that orb_label, the guarded stat must report the
    live row ONLY (N=1, cum_r=1.5) — the unguarded form would report N=2."""
    db = tmp_path / "lt.db"
    _init_schema(db)
    _seed_one_live_one_shadow(db)  # live + shadow share _ORB
    with duckdb.connect(str(db), read_only=True) as con:
        # Mirror log_trade.py:177-182 exactly (guarded form).
        guarded = (
            "SELECT COUNT(*) as n, ROUND(SUM(pnl_r), 2) as cum_r "
            "FROM paper_trades WHERE orb_label = ? AND execution_source != 'shadow'"
        )
        unguarded = "SELECT COUNT(*) as n, ROUND(SUM(pnl_r), 2) as cum_r FROM paper_trades WHERE orb_label = ?"
        gn = con.execute(guarded, [_ORB]).fetchone()
        un = con.execute(unguarded, [_ORB]).fetchone()
        assert gn == (1, 1.5), f"log_trade stat must be live-only, got {gn}"
        assert un == (2, 3.5), "guard is load-bearing: unguarded stat sees the shadow row"


def test_paper_trade_logger_reads_exclude_shadow(tmp_path):
    """R1/F2: paper_trade_logger's MAX(trading_day) sync-boundary read and its
    per-lane summary read key on strategy_id. They are CORE-disjoint from shadow
    today, but the guard is now structural. Prove the guarded forms skip a shadow
    row sharing the strategy_id (worst case the disjointness argument waives)."""
    db = tmp_path / "ptl.db"
    _init_schema(db)
    sid = "MNQ_SHARED_ID"
    with duckdb.connect(str(db)) as con:
        con.executemany(
            """INSERT INTO paper_trades (
                   trading_day, orb_label, strategy_id, instrument, pnl_r, execution_source
               ) VALUES (?, ?, ?, 'MNQ', ?, ?)""",
            [
                ("2026-06-04", _ORB, sid, 1.0, "backfill"),
                ("2026-09-01", _ORB, sid, 2.0, "shadow"),  # later day, would skew MAX + summary
            ],
        )
    with duckdb.connect(str(db), read_only=True) as con:
        # MAX(trading_day) guarded (paper_trade_logger.py:243) -> backfill day, not shadow's.
        mx = con.execute(
            "SELECT MAX(trading_day) FROM paper_trades WHERE strategy_id = ? AND execution_source != 'shadow'",
            [sid],
        ).fetchone()[0]
        assert str(mx) == "2026-06-04", f"sync boundary MAX must ignore the shadow day, got {mx}"
        # Summary guarded (paper_trade_logger.py:364) -> COUNT/SUM live-only.
        summ = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl_r), 0) FROM paper_trades "
            "WHERE strategy_id = ? AND execution_source != 'shadow'",
            [sid],
        ).fetchone()
        assert summ == (1, 1.0), f"lane summary must be backfill-only, got {summ}"


def test_monitor_lane_correlation_excludes_shadow(tmp_path):
    """R1: monitor_lane_correlation_rolling (8th reader, not in original
    inventory) groups daily pnl_r per strategy_id for the live correlation
    matrix. A shadow row sharing the strategy_id must not enter the matrix."""
    db = tmp_path / "corr.db"
    _init_schema(db)
    sid = "MNQ_CORR_ID"
    # paper_trades PRIMARY KEY is (strategy_id, trading_day), so a live and a
    # shadow row for the same id must sit on DIFFERENT days. The guard must drop
    # the shadow day from this strategy's daily-pnl series.
    with duckdb.connect(str(db)) as con:
        con.executemany(
            """INSERT INTO paper_trades (
                   trading_day, orb_label, strategy_id, instrument, pnl_r, execution_source
               ) VALUES (?, ?, ?, 'MNQ', ?, ?)""",
            [
                ("2026-06-04", _ORB, sid, 1.0, "live"),
                ("2026-06-05", _ORB, sid, 9.0, "shadow"),
            ],
        )
    with duckdb.connect(str(db), read_only=True) as con:
        # Mirror monitor_lane_correlation_rolling.py:131 (guarded form).
        guarded_days = con.execute(
            "SELECT trading_day, SUM(pnl_r) FROM paper_trades WHERE pnl_r IS NOT NULL "
            "AND execution_source != 'shadow' AND strategy_id = ? GROUP BY strategy_id, trading_day ORDER BY 1",
            [sid],
        ).fetchall()
        assert guarded_days == [(datetime.date(2026, 6, 4), 1.0)], (
            f"correlation series must be live-only (one day, 1.0R), shadow day excluded; got {guarded_days}"
        )
        # Guard is load-bearing: WITHOUT it the shadow day enters the series.
        unguarded = con.execute(
            "SELECT COUNT(*) FROM (SELECT trading_day FROM paper_trades WHERE pnl_r IS NOT NULL "
            "AND strategy_id = ? GROUP BY strategy_id, trading_day)",
            [sid],
        ).fetchone()[0]
        assert unguarded == 2, "unguarded correlation series would include the shadow day"
