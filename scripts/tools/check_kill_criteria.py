"""
Kill criteria monitor for MNQ RR1.0 raw baseline paper trading.

Two modes:
  1. Live mode (default): reads live_journal.db for broker-execution results
  2. Batch mode (--from-outcomes): reads orb_outcomes from gold.db for
     pre-computed trade results. Use for daily batch forward testing.

Checks against pre-registered thresholds from:
    docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md

Kill criteria (frozen):
    1. After 100 trades/session: actual ExpR < +0.03R -> STOP that session
    2. After 100 trades: actual slippage > 3 ticks average -> STOP (cost model wrong)
       (Live mode only -- slippage is baked into outcomes in batch mode)
    3. After 200 trades total: combined portfolio ExpR < +0.05R -> STOP everything
    4. O'Brien-Fleming sequential monitoring -- checked at interim fractions

Usage:
    python scripts/tools/check_kill_criteria.py                          # live mode
    python scripts/tools/check_kill_criteria.py --from-outcomes          # batch mode (2026+)
    python scripts/tools/check_kill_criteria.py --from-outcomes --since 2026-01-01
    python scripts/tools/check_kill_criteria.py --journal-path /path/to/live_journal.db
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import duckdb

# =========================================================================
# Pre-registered kill thresholds (FROZEN -- from pre-registration doc)
# =========================================================================

SESSION_EXPR_THRESHOLD = 0.03      # ExpR per session after N trades
SESSION_TRADE_THRESHOLD = 100      # Min trades before session kill fires
SLIPPAGE_TICK_THRESHOLD = 3.0      # Max avg slippage (ticks) before STOP
SLIPPAGE_TRADE_THRESHOLD = 100     # Min trades before slippage kill fires
PORTFOLIO_EXPR_THRESHOLD = 0.05    # Portfolio-wide ExpR after N trades
PORTFOLIO_TRADE_THRESHOLD = 200    # Min trades before portfolio kill fires

# Pre-registered sessions (from docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md)
PRE_REGISTERED_SESSIONS = frozenset({
    "NYSE_OPEN",       # BH FDR PASS
    "COMEX_SETTLE",    # BH FDR PASS
    "CME_PRECLOSE",    # Pre-registered for single-test 2026
})

# MNQ tick size for slippage conversion
MNQ_TICK_SIZE = 0.25

# =========================================================================
# O'Brien-Fleming sequential monitoring boundaries
# =========================================================================

# Lan-DeMets O'Brien-Fleming spending function approximation.
# z(t) = z_alpha / sqrt(t), one-sided for kill on underperformance.
OBF_FINAL_Z = 1.645  # one-sided alpha=0.05


def obf_boundary(fraction: float) -> float:
    """O'Brien-Fleming z-boundary at information fraction `fraction` (0,1]."""
    if fraction <= 0:
        return float("inf")
    return OBF_FINAL_Z / math.sqrt(fraction)


def compute_obf_p_equivalent(z_boundary: float) -> float:
    """Convert z-boundary to one-sided p-value equivalent (for reporting)."""
    from statistics import NormalDist

    nd = NormalDist()
    return 1.0 - nd.cdf(z_boundary)


# =========================================================================
# Data loaders
# =========================================================================


def _default_journal_path() -> Path:
    return Path(__file__).parent.parent.parent / "live_journal.db"


def load_journal(
    journal_path: Path, instrument: str = "MNQ"
) -> list[dict]:
    """Load completed trades from live_journal.db for the given instrument."""
    if not journal_path.exists():
        return []

    with duckdb.connect(str(journal_path), read_only=True) as con:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        if "live_trades" not in tables:
            return []

        result = con.execute(
            """
            SELECT
                trade_id, trading_day, instrument, strategy_id,
                direction, entry_model, engine_entry, engine_exit,
                fill_entry, fill_exit, actual_r, expected_r,
                slippage_pts, pnl_dollars, exit_reason, session_mode,
                created_at, exited_at
            FROM live_trades
            WHERE instrument = ?
              AND exited_at IS NOT NULL
              AND actual_r IS NOT NULL
            ORDER BY trading_day, created_at
            """,
            [instrument],
        )
        cols = [d[0] for d in result.description]
        rows = result.fetchall()

    return [dict(zip(cols, row, strict=False)) for row in rows]


def load_from_outcomes(
    instrument: str = "MNQ",
    entry_model: str = "E2",
    rr_target: float = 1.0,
    confirm_bars: int = 1,
    orb_minutes: int = 5,
    since: date | None = None,
    exclude_sessions: frozenset[str] = frozenset({"NYSE_CLOSE"}),
) -> list[dict]:
    """Load completed trades from orb_outcomes (gold.db) for batch mode.

    Returns the same list[dict] shape as load_journal() so all downstream
    check functions work identically.
    """
    from pipeline.paths import GOLD_DB_PATH

    if since is None:
        since = date(2026, 1, 1)

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        result = con.execute(
            """
            SELECT
                trading_day, orb_label, pnl_r, entry_price, exit_price,
                risk_dollars, outcome
            FROM orb_outcomes
            WHERE symbol = ?
              AND entry_model = ?
              AND rr_target = ?
              AND confirm_bars = ?
              AND orb_minutes = ?
              AND trading_day >= ?
              AND pnl_r IS NOT NULL
            ORDER BY trading_day, orb_label
            """,
            [instrument, entry_model, rr_target, confirm_bars, orb_minutes, since],
        )
        cols = [d[0] for d in result.description]
        rows = result.fetchall()

    trades = []
    for row in rows:
        r = dict(zip(cols, row, strict=False))
        if r["orb_label"] in exclude_sessions:
            continue
        sid = f"{instrument}_{r['orb_label']}_{entry_model}_RR{rr_target}_CB{confirm_bars}_NO_FILTER"
        trades.append({
            "trade_id": f"batch_{r['trading_day']}_{r['orb_label']}",
            "trading_day": r["trading_day"],
            "instrument": instrument,
            "strategy_id": sid,
            "actual_r": float(r["pnl_r"]),
            "slippage_pts": None,  # E2 slippage baked into entry_price
            "pnl_dollars": None,
            "session_mode": "batch",
        })

    return trades


# =========================================================================
# Session extraction
# =========================================================================


def extract_session(strategy_id: str) -> str:
    """Extract orb_label from strategy_id like 'MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER'."""
    parts = strategy_id.split("_")
    for i, part in enumerate(parts):
        if part in ("E1", "E2", "E3"):
            return "_".join(parts[1:i])
    return "_".join(parts[1:-3])  # fallback


# =========================================================================
# Kill criteria checks
# =========================================================================


def check_session_expr(trades: list[dict]) -> list[dict]:
    """Check per-session ExpR against kill threshold."""
    sessions: dict[str, list[float]] = {}
    for t in trades:
        sess = extract_session(t["strategy_id"])
        sessions.setdefault(sess, []).append(t["actual_r"])

    results = []
    for sess, rs in sorted(sessions.items()):
        n = len(rs)
        expr = sum(rs) / n if n > 0 else 0.0
        active = n >= SESSION_TRADE_THRESHOLD
        killed = active and expr < SESSION_EXPR_THRESHOLD

        results.append({
            "session": sess,
            "trades": n,
            "expr": expr,
            "threshold": SESSION_EXPR_THRESHOLD,
            "min_trades": SESSION_TRADE_THRESHOLD,
            "active": active,
            "verdict": "KILL" if killed else ("PASS" if active else "WAITING"),
            "pre_registered": sess in PRE_REGISTERED_SESSIONS,
        })
    return results


def check_slippage(trades: list[dict], tick_size: float = MNQ_TICK_SIZE) -> dict:
    """Check average slippage in ticks against kill threshold."""
    slippage_pts = [t["slippage_pts"] for t in trades if t["slippage_pts"] is not None]
    n = len(slippage_pts)
    if n == 0:
        return {
            "trades_with_slippage": 0,
            "avg_slippage_pts": 0.0,
            "avg_slippage_ticks": 0.0,
            "threshold_ticks": SLIPPAGE_TICK_THRESHOLD,
            "active": False,
            "verdict": "WAITING",
        }

    avg_pts = sum(abs(s) for s in slippage_pts) / n
    avg_ticks = avg_pts / tick_size if tick_size > 0 else 0.0
    active = n >= SLIPPAGE_TRADE_THRESHOLD
    killed = active and avg_ticks > SLIPPAGE_TICK_THRESHOLD

    return {
        "trades_with_slippage": n,
        "avg_slippage_pts": avg_pts,
        "avg_slippage_ticks": avg_ticks,
        "threshold_ticks": SLIPPAGE_TICK_THRESHOLD,
        "active": active,
        "verdict": "KILL" if killed else ("PASS" if active else "WATCHING"),
    }


def check_portfolio_expr(trades: list[dict]) -> dict:
    """Check portfolio-wide ExpR against kill threshold."""
    n = len(trades)
    total_r = sum(t["actual_r"] for t in trades)
    expr = total_r / n if n > 0 else 0.0
    active = n >= PORTFOLIO_TRADE_THRESHOLD
    killed = active and expr < PORTFOLIO_EXPR_THRESHOLD

    return {
        "total_trades": n,
        "total_r": total_r,
        "expr": expr,
        "threshold": PORTFOLIO_EXPR_THRESHOLD,
        "min_trades": PORTFOLIO_TRADE_THRESHOLD,
        "active": active,
        "verdict": "KILL" if killed else ("PASS" if active else "WAITING"),
    }


def check_obf_sequential(
    trades: list[dict], target_trades_per_session: int = 100
) -> list[dict]:
    """O'Brien-Fleming sequential monitoring per session."""
    sessions: dict[str, list[float]] = {}
    for t in trades:
        sess = extract_session(t["strategy_id"])
        sessions.setdefault(sess, []).append(t["actual_r"])

    results = []
    for sess, rs in sorted(sessions.items()):
        n = len(rs)
        if n < 10:
            results.append({
                "session": sess,
                "trades": n,
                "fraction": n / target_trades_per_session,
                "verdict": "TOO_EARLY",
            })
            continue

        mean_r = sum(rs) / n
        sd_r = math.sqrt(sum((r - mean_r) ** 2 for r in rs) / n) if n > 1 else 1.0
        se = sd_r / math.sqrt(n) if sd_r > 0 else 1.0
        z_stat = mean_r / se

        fraction = min(n / target_trades_per_session, 1.0)
        boundary = obf_boundary(fraction)
        early_kill = z_stat < -boundary

        results.append({
            "session": sess,
            "trades": n,
            "fraction": round(fraction, 3),
            "expr": round(mean_r, 4),
            "z_stat": round(z_stat, 3),
            "boundary": round(boundary, 3),
            "kill_boundary": round(-boundary, 3),
            "p_equiv": round(compute_obf_p_equivalent(boundary), 4),
            "verdict": "EARLY_KILL" if early_kill else "CONTINUE",
            "pre_registered": sess in PRE_REGISTERED_SESSIONS,
        })
    return results


# =========================================================================
# Reporting
# =========================================================================


def print_report(
    trades: list[dict],
    instrument: str,
    target_trades: int,
    batch_mode: bool = False,
) -> bool:
    """Print full kill criteria report. Returns True if all clear, False if any KILL."""
    today = date.today()
    mode_label = "BATCH (orb_outcomes)" if batch_mode else "LIVE (live_journal.db)"
    print(f"\n{'=' * 72}")
    print(f"  KILL CRITERIA MONITOR -- {instrument} Raw Baseline")
    print(f"  Date: {today}   Trades: {len(trades)}   Mode: {mode_label}")
    print(f"{'=' * 72}")

    if not trades:
        print("\n  No completed trades found. Nothing to check.")
        if batch_mode:
            print(f"  Refresh data: python scripts/tools/forward_test.py --instrument {instrument}")
        else:
            print(f"  Start paper trading: python scripts/run_live_session.py "
                  f"--instrument {instrument} --signal-only --raw-baseline")
        print(f"{'=' * 72}\n")
        return True

    any_kill = False

    # --- 1. Per-session ExpR ---
    print(f"\n{'-' * 72}")
    print(f"  1. SESSION EXPECTANCY (kill < {SESSION_EXPR_THRESHOLD}R after {SESSION_TRADE_THRESHOLD} trades)")
    print(f"{'-' * 72}")
    session_results = check_session_expr(trades)
    print(f"  {'Session':<20} {'Trades':>6} {'ExpR':>8} {'Status':>10} {'Pre-Reg':>8}")
    print(f"  {'-' * 56}")
    for r in session_results:
        pre = "YES" if r["pre_registered"] else ""
        status = r["verdict"]
        marker = " *** KILL ***" if status == "KILL" else ""
        print(f"  {r['session']:<20} {r['trades']:>6} {r['expr']:>+8.4f} {status:>10} {pre:>8}{marker}")
        if status == "KILL":
            any_kill = True

    # --- 2. Slippage ---
    print(f"\n{'-' * 72}")
    print(f"  2. SLIPPAGE (kill > {SLIPPAGE_TICK_THRESHOLD} ticks avg after {SLIPPAGE_TRADE_THRESHOLD} trades)")
    print(f"{'-' * 72}")
    if batch_mode:
        print("  SKIPPED (batch mode -- E2 slippage baked into orb_outcomes entry_price)")
        print("  Slippage kill criterion only applies to live broker execution.")
    else:
        slip = check_slippage(trades)
        print(f"  Trades with slippage data: {slip['trades_with_slippage']}")
        print(f"  Avg slippage: {slip['avg_slippage_pts']:.3f} pts ({slip['avg_slippage_ticks']:.2f} ticks)")
        print(f"  Threshold: {slip['threshold_ticks']:.1f} ticks")
        print(f"  Verdict: {slip['verdict']}")
        if slip["verdict"] == "KILL":
            print("  *** KILL -- COST MODEL WRONG ***")
            any_kill = True

    # --- 3. Portfolio ExpR ---
    print(f"\n{'-' * 72}")
    print(f"  3. PORTFOLIO EXPECTANCY (kill < {PORTFOLIO_EXPR_THRESHOLD}R after {PORTFOLIO_TRADE_THRESHOLD} trades)")
    print(f"{'-' * 72}")
    pf = check_portfolio_expr(trades)
    print(f"  Total trades: {pf['total_trades']}")
    print(f"  Total R: {pf['total_r']:+.2f}")
    print(f"  Portfolio ExpR: {pf['expr']:+.4f}")
    print(f"  Verdict: {pf['verdict']}")
    if pf["verdict"] == "KILL":
        print("  *** KILL -- STOP EVERYTHING ***")
        any_kill = True

    # --- 4. O'Brien-Fleming sequential monitoring ---
    print(f"\n{'-' * 72}")
    print(f"  4. O'BRIEN-FLEMING SEQUENTIAL MONITORING (target {target_trades} trades/session)")
    print(f"{'-' * 72}")
    obf = check_obf_sequential(trades, target_trades)
    print(f"  {'Session':<20} {'N':>5} {'Frac':>6} {'ExpR':>8} {'z':>7} {'Bound':>7} {'Status':>12}")
    print(f"  {'-' * 68}")
    for r in obf:
        if r["verdict"] == "TOO_EARLY":
            print(f"  {r['session']:<20} {r['trades']:>5} {r['fraction']:>6.3f} {'--':>8} {'--':>7} {'--':>7} {'TOO_EARLY':>12}")
        else:
            marker = " ***" if r["verdict"] == "EARLY_KILL" else ""
            print(
                f"  {r['session']:<20} {r['trades']:>5} {r['fraction']:>6.3f} "
                f"{r['expr']:>+8.4f} {r['z_stat']:>+7.3f} {r['kill_boundary']:>+7.3f} "
                f"{r['verdict']:>12}{marker}"
            )
        if r["verdict"] == "EARLY_KILL":
            any_kill = True

    # --- Summary ---
    print(f"\n{'=' * 72}")
    if any_kill:
        print("  *** KILL CRITERIA TRIGGERED -- ACTION REQUIRED ***")
        print("  Review the flagged items above. Pre-registration binds: if kill fires, STOP.")
    else:
        print("  ALL CLEAR -- no kill criteria triggered")
        waiting = sum(1 for r in session_results if r["verdict"] == "WAITING")
        if waiting:
            print(f"  ({waiting} sessions still accumulating trades)")
    print(f"{'=' * 72}\n")

    return not any_kill


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Kill criteria monitor for paper trading forward test",
    )
    parser.add_argument(
        "--journal-path",
        type=Path,
        default=None,
        help=f"Path to live_journal.db (default: {_default_journal_path()})",
    )
    parser.add_argument(
        "--instrument",
        default="MNQ",
        help="Instrument to monitor (default: MNQ)",
    )
    parser.add_argument(
        "--target-trades",
        type=int,
        default=100,
        help="Target trades per session for OBF monitoring (default: 100)",
    )
    parser.add_argument(
        "--from-outcomes",
        action="store_true",
        default=False,
        help="Batch mode: read from orb_outcomes (gold.db) instead of live_journal.db",
    )
    parser.add_argument(
        "--since",
        type=date.fromisoformat,
        default=None,
        help="Start date for batch mode (default: 2026-01-01)",
    )
    args = parser.parse_args()

    if args.from_outcomes:
        trades = load_from_outcomes(
            instrument=args.instrument,
            since=args.since,
        )
        ok = print_report(trades, args.instrument, args.target_trades, batch_mode=True)
    else:
        journal_path = args.journal_path or _default_journal_path()
        trades = load_journal(journal_path, args.instrument)
        ok = print_report(trades, args.instrument, args.target_trades, batch_mode=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
