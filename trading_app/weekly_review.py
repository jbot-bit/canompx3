"""Phase 1 weekly operations review.

Run every Monday 08:00 AEST. All data from DB — no manual input.

Usage:
    python -m trading_app.weekly_review
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
MONITOR_DIR = Path(__file__).resolve().parents[1] / "data" / "forward_monitoring"

SESSIONS = ["NYSE_CLOSE", "SINGAPORE_OPEN", "COMEX_SETTLE", "NYSE_OPEN", "TOKYO_OPEN"]


def section_0_account_health():
    """SECTION 0 — Account HWM DD status (before all other sections)."""
    print("\n  SECTION 0 - ACCOUNT HEALTH (Dollar DD Tracker)")
    print(
        f"  {'Account':<15} {'Firm':<12} {'Equity':>10} {'HWM':>10} {'DD Used':>10} {'DD Limit':>10} {'%Used':>8} {'Status':>10}"
    )
    print("  " + "-" * 90)

    hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
    if not hwm_files:
        print("  No HWM tracker files found. Will init on first live session.")
        return

    for f in hwm_files:
        if "CORRUPT" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            acct = data.get("account_id", "?")
            firm = data.get("firm", "?")
            equity = data.get("last_equity", 0)
            hwm = data.get("hwm_dollars", 0)
            used = data.get("dd_used_dollars", 0)
            limit_d = data.get("dd_limit_dollars", 0)
            pct = data.get("dd_pct_used", 0)
            halted = data.get("halt_triggered", False)

            status = "HALTED" if halted else ("WARNING" if pct >= 0.75 else "OK")
            flag = " ***" if pct >= 0.50 else ""
            pct_display = pct * 100 if pct <= 1.0 else pct  # stored as decimal (0.75) or pct (75.0)
            print(
                f"  {acct:<15} {firm:<12} ${equity:>9,.0f} ${hwm:>9,.0f} "
                f"${used:>9,.0f} ${limit_d:>9,.0f} {pct_display:>7.1f}% {status:>10}{flag}"
            )

            # Session history
            sessions = data.get("session_log", [])
            if sessions:
                recent = sessions[-3:]  # last 3 sessions
                for s in recent:
                    s_dd = s.get("session_dd", 0)
                    print(
                        f"    {s.get('date', '?')[:10]}: "
                        f"start=${s.get('start_equity', 0):,.0f} "
                        f"end=${s.get('end_equity', 0):,.0f} "
                        f"dd=${s_dd:+,.0f}"
                    )
        except Exception as e:
            print(f"  ERROR: {f.name}: {e}")


def section_0b_consistency():
    """SECTION 0b — Consistency rule and payout eligibility."""
    print("\n  SECTION 0b - COMPLIANCE MONITORS")

    try:
        from trading_app.consistency_tracker import (
            check_consistency,
            check_payout_eligibility,
            check_account_idle,
            check_microscalp_compliance,
        )

        # Consistency
        for firm in ["apex", "topstep"]:
            result = check_consistency(firm=firm, instrument="MNQ")
            if result is not None:
                print(
                    f"  Consistency ({firm} {result.limit_pct:.0%}): "
                    f"best day ${result.best_day_pnl:.0f} ({result.best_day_date}) "
                    f"= {result.windfall_pct:.1f}% of ${result.total_profit:.0f} — {result.status}"
                )

        # Payout eligibility
        pe = check_payout_eligibility(firm="apex", instrument="MNQ")
        print(
            f"  Payout eligibility: {pe.trading_days}/{pe.min_trading_days} trading days, "
            f"{pe.profitable_days_50}/{pe.min_profitable_days} profitable days ($50+) — "
            f"{'ELIGIBLE' if pe.eligible else 'NOT YET'}"
        )
        for note in pe.notes:
            print(f"    -> {note}")

        # Idle check (Tradeify)
        idle_status, idle_days = check_account_idle(instrument="MNQ")
        if idle_status != "OK":
            print(f"  Tradeify idle: {idle_status} — {idle_days} days since last trade")
        else:
            print(f"  Tradeify idle: OK ({idle_days} days since last trade)")

        # Microscalp
        ms = check_microscalp_compliance(instrument="MNQ")
        if ms is not None:
            print(
                f"  Microscalp (Tradeify): {ms.pct_trades_over_10s:.0f}% trades >10s, "
                f"{ms.pct_profit_from_over_10s:.0f}% profit >10s — "
                f"{'COMPLIANT' if ms.compliant else 'BREACH'}"
            )
    except Exception as e:
        print(f"  Compliance monitors unavailable: {e}")


def section_1_forward_performance(con, week_start: date):
    """Per-lane forward performance this week and running totals."""
    print("\n  SECTION 1 - FORWARD PERFORMANCE")
    print(f"  Week starting: {week_start}")
    print(f"  {'Lane':<25} {'This Wk N':>10} {'Wk CumR':>10} {'Total N':>10} {'Total CumR':>10} {'WR%':>8}")
    print("  " + "-" * 80)

    for session in SESSIONS:
        # This week
        wk = con.execute(
            """SELECT COUNT(*), COALESCE(ROUND(SUM(pnl_r), 2), 0)
               FROM paper_trades WHERE orb_label = ? AND trading_day >= ? AND pnl_r IS NOT NULL""",
            [session, week_start],
        ).fetchone()
        # Total
        tot = con.execute(
            """SELECT COUNT(*), COALESCE(ROUND(SUM(pnl_r), 2), 0),
                      ROUND(AVG(CASE WHEN pnl_r > 0 THEN 1.0 ELSE 0.0 END)*100, 1)
               FROM paper_trades WHERE orb_label = ? AND pnl_r IS NOT NULL""",
            [session],
        ).fetchone()
        print(f"  {session:<25} {wk[0]:>10} {wk[1]:>+10.2f} {tot[0]:>10} {tot[1]:>+10.2f} {tot[2] or 0:>7.1f}%")


def section_2_slippage(con):
    """Slippage actuals vs model."""
    print("\n  SECTION 2 - SLIPPAGE ACTUALS VS MODEL")
    row = con.execute(
        """SELECT COUNT(*), ROUND(AVG(slippage_ticks), 2), ROUND(MAX(slippage_ticks), 0)
           FROM paper_trades WHERE execution_source = 'live' AND slippage_ticks IS NOT NULL"""
    ).fetchone()
    live_n = row[0] if row else 0
    if live_n == 0:
        print("  No live trades with slippage data yet. Model assumes 1 tick.")
        print(f"  MNQ slippage pilot: 0/30 live trades recorded.")
        return
    print(f"  Live trades with slippage: {live_n}")
    print(f"  Mean actual slippage: {row[1]} ticks (model: 1 tick)")
    print(f"  Max actual slippage: {row[2]} ticks")
    if row[1] and row[1] > 2.0:
        print("  ** FLAG: Mean actual slippage > 2 ticks — edge at risk **")
    print(f"  MNQ slippage pilot: {live_n}/30 live trades recorded.")


def section_3_dd_events():
    """DD circuit breaker events."""
    print("\n  SECTION 3 - DD CIRCUIT BREAKER EVENTS")
    cb_file = STATE_DIR / "dd_circuit_breaker.json"
    if not cb_file.exists():
        print("  No circuit breaker events recorded.")
        return
    try:
        data = json.loads(cb_file.read_text())
        if data.get("session_halt"):
            print(f"  ACTIVE HALT: {data.get('date')} — {data.get('reason', 'unknown')}")
        else:
            print("  No active halt.")
    except Exception:
        print("  Circuit breaker state file unreadable.")


def section_4_signal_quality(con, week_start: date):
    """Signal quality — signals generated vs taken."""
    print("\n  SECTION 4 - SIGNAL QUALITY (EXECUTION DISCIPLINE)")
    # Count signals from orb_outcomes vs trades in paper_trades
    for session in SESSIONS[:4]:  # MNQ only
        signals = con.execute(
            """SELECT COUNT(DISTINCT trading_day) FROM orb_outcomes
               WHERE symbol = 'MNQ' AND orb_label = ? AND entry_model = 'E2'
                 AND outcome IS NOT NULL AND outcome != 'scratch'
                 AND trading_day >= ?""",
            [session, week_start],
        ).fetchone()[0]
        taken = con.execute(
            """SELECT COUNT(*) FROM paper_trades
               WHERE orb_label = ? AND trading_day >= ? AND pnl_r IS NOT NULL""",
            [session, week_start],
        ).fetchone()[0]
        rate = (taken / signals * 100) if signals > 0 else 0
        flag = " ** DISCIPLINE_ISSUE" if 0 < rate < 80 else ""
        print(f"  {session:<25} signals={signals:>3}, taken={taken:>3}, rate={rate:.0f}%{flag}")


def section_5_forward_monitor():
    """Forward monitor status."""
    print("\n  SECTION 5 - FORWARD MONITOR STATUS")
    print("  Running forward_monitor.py...")
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "scripts.tools.forward_monitor"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        # Print just the table lines
        for line in result.stdout.split("\n"):
            if line.strip() and not line.startswith("="):
                print(f"  {line}")
    else:
        print(f"  ERROR: {result.stderr[:200]}")


def section_6_mgc_shadow(con):
    """MGC shadow trading progress."""
    print("\n  SECTION 6 - MGC SHADOW PROGRESS")
    row = con.execute(
        """SELECT COUNT(*) FROM paper_trades WHERE orb_label = 'TOKYO_OPEN' AND pnl_r IS NOT NULL"""
    ).fetchone()
    shadow_n = row[0] if row else 0
    remaining = max(0, 250 - shadow_n)

    # Trade frequency from 2025+
    freq = (
        con.execute(
            """SELECT COUNT(*)*1.0 / GREATEST(COUNT(DISTINCT DATE_TRUNC('week', trading_day)), 1)
           FROM orb_outcomes WHERE symbol='MGC' AND orb_label='TOKYO_OPEN'
             AND entry_model='E2' AND rr_target=2.0 AND confirm_bars=1 AND orb_minutes=5
             AND outcome IS NOT NULL AND outcome != 'scratch'
             AND trading_day >= '2025-01-01'"""
        ).fetchone()[0]
        or 0
    )

    weeks_to_250 = remaining / freq if freq > 0 else 9999
    est_date = date.today() + timedelta(weeks=weeks_to_250)

    print(f"  Current shadow N: {shadow_n} / 250")
    print(f"  Remaining: {remaining}")
    print(f"  Historical rate: {freq:.1f} trades/week")
    print(f"  Estimated N=250 date: {est_date}")

    # Regime check
    atr_row = con.execute(
        """SELECT ROUND(AVG(atr_20), 1) FROM daily_features
           WHERE symbol='MGC' AND orb_minutes=5 AND trading_day >= CURRENT_DATE - 20"""
    ).fetchone()
    if atr_row and atr_row[0]:
        regime = "FAVORABLE" if atr_row[0] > 30 else "UNFAVORABLE"
        print(f"  MGC ATR-20 (recent): {atr_row[0]} — regime: {regime}")


def section_7_phase2_blockers():
    """Phase 2 blocker status."""
    print("\n  SECTION 7 - PHASE 2 BLOCKERS")
    blockers = [
        ("Tradovate rate limit", "pending"),
        ("e2e sim retest (429 handling)", "not started"),
        ("DD circuit breaker code integration", "not started"),
        ("MNQ tbbo pilot", "not started"),
    ]
    for name, status in blockers:
        check = "[ ]" if status != "done" else "[x]"
        print(f"  {check} {name}: {status}")


def section_8_orb_monitor(con):
    """SECTION 8 — ORB size monitor per lane (risk management)."""
    print("\n  SECTION 8 - ORB SIZE MONITOR")

    try:
        from trading_app.prop_profiles import get_lane_registry
    except ImportError:
        print("  Cannot load lane registry — skipping ORB monitor.")
        return

    registry = get_lane_registry()
    print(
        f"  {'Lane':<25} {'Cap':>8} {'20d Med':>8} {'5d Med':>8} {'Trend':>10} {'Alert':>20}"
    )
    print("  " + "-" * 90)

    for label, info in sorted(registry.items()):
        sess = info["orb_label"]
        om = info["orb_minutes"]
        cap_risk = info.get("max_orb_size_pts")  # cap on risk_points (stop distance)
        stop_mult = info.get("stop_multiplier", 1.0) or 1.0

        # Convert risk_points cap to raw ORB equivalent for monitoring.
        # risk_points ~ orb_size * stop_mult, so orb_cap ~ risk_cap / stop_mult.
        cap_orb = cap_risk / stop_mult if cap_risk is not None and stop_mult > 0 else None
        cap_str = f"~{cap_orb:.0f} pts" if cap_orb is not None else "NO CAP"

        # Defensive: session label must be a valid identifier (prevents SQL injection
        # if registry is ever corrupted — source is trusted but defense-in-depth).
        if not sess.replace("_", "").isalnum():
            print(f"  {label:<25} INVALID SESSION LABEL: {sess!r}")
            continue

        try:
            # Rolling 20-day and 5-day medians from orb_outcomes (break-days only)
            row_20 = con.execute(
                f"""
                SELECT MEDIAN(d."orb_{sess}_size")
                FROM orb_outcomes o
                JOIN daily_features d ON o.trading_day = d.trading_day
                    AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
                  AND o.entry_model = 'E2' AND o.confirm_bars = 1
                  AND d."orb_{sess}_size" IS NOT NULL AND d."orb_{sess}_size" > 0
                  AND o.trading_day >= CURRENT_DATE - 30
                """,
                [info["instrument"], sess, om],
            ).fetchone()
            med_20 = row_20[0] if row_20 and row_20[0] else 0

            row_5 = con.execute(
                f"""
                SELECT MEDIAN(d."orb_{sess}_size")
                FROM orb_outcomes o
                JOIN daily_features d ON o.trading_day = d.trading_day
                    AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
                WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
                  AND o.entry_model = 'E2' AND o.confirm_bars = 1
                  AND d."orb_{sess}_size" IS NOT NULL AND d."orb_{sess}_size" > 0
                  AND o.trading_day >= CURRENT_DATE - 7
                """,
                [info["instrument"], sess, om],
            ).fetchone()
            med_5 = row_5[0] if row_5 and row_5[0] else 0

            trend = "EXPANDING" if med_5 > med_20 > 0 else ("CONTRACTING" if 0 < med_5 < med_20 else "STABLE")

            alert = ""
            if cap_orb is not None and med_20 > 0:
                if med_20 >= cap_orb:
                    # Count recent skip rate (raw ORB >= equivalent cap)
                    total = con.execute(
                        f"""
                        SELECT COUNT(*) FROM orb_outcomes o
                        JOIN daily_features d ON o.trading_day = d.trading_day
                            AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
                        WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
                          AND o.entry_model = 'E2' AND o.confirm_bars = 1
                          AND d."orb_{sess}_size" IS NOT NULL
                          AND o.trading_day >= CURRENT_DATE - 30
                        """,
                        [info["instrument"], sess, om],
                    ).fetchone()[0]
                    over_cap = con.execute(
                        f"""
                        SELECT COUNT(*) FROM orb_outcomes o
                        JOIN daily_features d ON o.trading_day = d.trading_day
                            AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
                        WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
                          AND o.entry_model = 'E2' AND o.confirm_bars = 1
                          AND d."orb_{sess}_size" >= ?
                          AND o.trading_day >= CURRENT_DATE - 30
                        """,
                        [info["instrument"], sess, om, cap_orb],
                    ).fetchone()[0]
                    pct = over_cap / total * 100 if total > 0 else 0
                    alert = f"AT CAP {pct:.0f}% skip"
                elif med_20 >= cap_orb * 0.80:
                    alert = "APPROACHING CAP"

            print(
                f"  {label:<25} {cap_str:>8} {med_20:>7.1f}p {med_5:>7.1f}p {trend:>10} {alert:>20}"
            )
        except Exception as e:
            print(f"  {label:<25} {cap_str:>8} {'ERROR':>8}  {str(e)[:40]}")


def run_review():
    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)

        today = date.today()
        week_start = today - timedelta(days=today.weekday())

        print(f"{'=' * 80}")
        print(f"PHASE 1 WEEKLY OPERATIONS REVIEW | {today}")
        print(f"{'=' * 80}")

        section_0_account_health()
        section_0b_consistency()
        section_1_forward_performance(con, week_start)
        section_2_slippage(con)
        section_3_dd_events()
        section_4_signal_quality(con, week_start)
        section_5_forward_monitor()
        section_6_mgc_shadow(con)
        section_7_phase2_blockers()
        section_8_orb_monitor(con)

        print(f"\n{'=' * 80}")
        print("END OF WEEKLY REVIEW")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    run_review()
