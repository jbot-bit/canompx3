"""Pre-session system health check — HARD GATE for manual Phase 1 trading.

No check passing = no trade that session. This is not advisory.

Usage:
    python -m trading_app.pre_session_check --session NYSE_CLOSE
    python -m trading_app.pre_session_check --session SINGAPORE_OPEN
    python -m trading_app.pre_session_check --session COMEX_SETTLE
    python -m trading_app.pre_session_check --session NYSE_OPEN
    python -m trading_app.pre_session_check --session TOKYO_OPEN  (MGC shadow)
"""

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Lane definitions — imported from canonical source (prop_profiles.py)
from trading_app.prop_profiles import get_lane_registry

LANE_DEFS = get_lane_registry()


def check_data_freshness(con, instrument: str) -> tuple[bool, str]:
    """Check gold.db has recent data for this instrument."""
    row = con.execute("SELECT MAX(ts_utc) FROM bars_1m WHERE symbol = ?", [instrument]).fetchone()
    if not row or not row[0]:
        return False, f"No bars_1m data for {instrument}"
    latest = row[0]
    now_utc = datetime.now(timezone.utc)
    if hasattr(latest, "timestamp"):
        gap_hours = (now_utc.timestamp() - latest.timestamp()) / 3600
    else:
        gap_hours = 999
    if gap_hours > 48:
        return False, f"Data stale: latest bar {latest} ({gap_hours:.0f}h ago)"
    return True, f"Latest bar: {latest} ({gap_hours:.1f}h ago)"


def check_paper_trades_accessible(con) -> tuple[bool, str]:
    """Check paper_trades table is accessible."""
    try:
        n = con.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        return True, f"paper_trades: {n} rows"
    except Exception as e:
        return False, f"paper_trades inaccessible: {e}"


def check_dd_circuit_breaker() -> tuple[bool, str]:
    """Check if DD circuit breaker has halted trading today."""
    cb_file = STATE_DIR / "dd_circuit_breaker.json"
    if not cb_file.exists():
        return True, "No DD halt file (first session)"
    try:
        data = json.loads(cb_file.read_text())
        if data.get("session_halt") and data.get("date") == str(date.today()):
            return False, f"DD CIRCUIT BREAKER ACTIVE: halted at {data.get('reason', 'unknown')}"
        return True, "DD circuit breaker: clear"
    except Exception:
        return True, "DD halt file unreadable (proceeding)"


def check_daily_equity() -> tuple[bool, str]:
    """Check today's equity tracking state."""
    eq_file = STATE_DIR / f"equity_{date.today()}.json"
    if not eq_file.exists():
        # Create it
        data = {"date": str(date.today()), "starting_equity": None, "current_dd": 0.0}
        eq_file.write_text(json.dumps(data, indent=2))
        return True, "Equity file created for today. Record starting equity before first trade."
    data = json.loads(eq_file.read_text())
    dd = data.get("current_dd", 0.0)
    if dd <= -1000:
        return False, f"DAILY DD LIMIT BREACHED: ${dd:.0f} (limit -$1,000)"
    if dd <= -800:
        return True, f"WARNING: daily DD at ${dd:.0f} (halt at -$1,000)"
    return True, f"Daily DD: ${dd:.0f}"


def check_slippage_pilot_progress(con) -> str:
    """Check progress toward MNQ slippage pilot (30 live trades needed)."""
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE execution_source = 'live' AND slippage_ticks IS NOT NULL"
        ).fetchone()
        n = row[0] if row else 0
        return f"MNQ slippage pilot: {n}/30 live trades with slippage recorded"
    except Exception:
        return "MNQ slippage pilot: cannot query"


def check_hwm_tracker() -> tuple[bool, str]:
    """Check account HWM DD tracker status."""
    from pathlib import Path as _P

    hwm_dir = _P(__file__).resolve().parents[1] / "data" / "state"
    hwm_files = list(hwm_dir.glob("account_hwm_*.json"))
    if not hwm_files:
        return True, "No HWM tracker active (first session — will init from broker)"

    results = []
    any_halt = False
    for f in hwm_files:
        if "CORRUPT" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            acct = data.get("account_id", "?")
            hwm = data.get("hwm_dollars", 0)
            used = data.get("dd_used_dollars", 0)
            limit = data.get("dd_limit_dollars", 0)
            pct = data.get("dd_pct_used", 0)
            halted = data.get("halt_triggered", False)
            hwm_date = (data.get("hwm_timestamp") or "")[:10]
            remaining = limit - used

            if halted:
                any_halt = True
                results.append(f"HALT {acct}: DD ${used:.0f} >= limit ${limit:.0f}")
            elif pct >= 0.75:
                results.append(
                    f"WARN {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — "
                    f"${remaining:.0f} remaining. HWM ${hwm:.0f} on {hwm_date}"
                )
            else:
                results.append(f"OK {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — ${remaining:.0f} remaining")
        except Exception as e:
            results.append(f"ERROR reading {f.name}: {e}")

    msg = " | ".join(results)
    return (not any_halt), f"DD TRACKER: {msg}"


def check_consistency_rule() -> tuple[bool, str]:
    """Check prop firm consistency rule status."""
    try:
        from trading_app.consistency_tracker import check_consistency

        result = check_consistency(firm="apex", instrument="MNQ")
        if result is None:
            return True, "Consistency: no trades yet"
        status_ok = result.status != "BREACH"
        return status_ok, (
            f"Consistency ({result.limit_pct:.0%} rule): best day ${result.best_day_pnl:.0f} on {result.best_day_date} "
            f"= {result.windfall_pct:.1f}% of ${result.total_profit:.0f} total — {result.status}"
        )
    except Exception as e:
        return True, f"Consistency check unavailable: {e}"


def check_forward_monitor_freshness() -> tuple[bool, str]:
    """Check if forward_monitor has been run recently."""
    fm_file = Path(__file__).resolve().parents[1] / "data" / "forward_monitoring" / "latest.json"
    if not fm_file.exists():
        return True, "WARN: forward_monitor never run. Run: python -m scripts.tools.forward_monitor"
    try:
        data = json.loads(fm_file.read_text())
        ts = data.get("timestamp", "")
        return True, f"Forward monitor last run: {ts}"
    except Exception:
        return True, "WARN: forward_monitor output unreadable"


def check_signal_exists(con, session: str, lane: dict, today: date) -> tuple[bool, str]:
    """Check if today has outcome data for this session (signal fired)."""
    instrument = lane["instrument"]
    orb_min = lane["orb_minutes"]
    row = con.execute(
        """SELECT COUNT(*) FROM orb_outcomes
           WHERE symbol = ? AND orb_label = ? AND orb_minutes = ?
             AND entry_model = 'E2' AND trading_day = ?""",
        [instrument, session, orb_min, today],
    ).fetchone()
    n = row[0] if row else 0
    if n > 0:
        return True, f"Signal exists for {session} today ({n} outcomes)"
    return True, f"No signal yet for {session} today (may not have fired yet)"


def run_checks(session: str) -> bool:
    """Run all checks for a session. Returns True if GO."""
    if session not in LANE_DEFS:
        print(f"ERROR: Unknown session '{session}'. Valid: {', '.join(LANE_DEFS.keys())}")
        return False

    lane = LANE_DEFS[session]
    today = date.today()
    results = []

    with duckdb.connect(str(GOLD_DB_PATH), read_only=True) as con:
        configure_connection(con)

        # Data freshness
        ok, msg = check_data_freshness(con, lane["instrument"])
        results.append(("Data freshness", ok, msg))

        ok, msg = check_paper_trades_accessible(con)
        results.append(("Paper trades table", ok, msg))

        ok, msg = check_forward_monitor_freshness()
        results.append(("Forward monitor", ok, msg))

        # Account state
        ok, msg = check_hwm_tracker()
        results.append(("HWM DD tracker", ok, msg))

        ok, msg = check_dd_circuit_breaker()
        results.append(("DD circuit breaker (intraday)", ok, msg))

        ok, msg = check_daily_equity()
        results.append(("Daily equity", ok, msg))

        ok, msg = check_consistency_rule()
        results.append(("Consistency rule", ok, msg))

        # Signal
        ok, msg = check_signal_exists(con, session, lane, today)
        results.append(("Signal check", ok, msg))

        # Slippage pilot progress
        slip_msg = check_slippage_pilot_progress(con)
        results.append(("Slippage pilot", True, slip_msg))

    # Lane 2 enforcement
    if lane.get("is_half_size"):
        results.append(("Lane 2 sizing", True, "0.5x MANDATORY — 1 micro lot max"))

    # Shadow-only check
    if lane.get("shadow_only"):
        results.append(("Shadow mode", True, "MGC TOKYO_OPEN: shadow-trade ONLY, no real capital"))

    # Print results
    all_pass = True
    print(f"\n{'=' * 70}")
    print(f"PRE-SESSION CHECK: {session} | {today} | {lane['instrument']}")
    print(f"Strategy: {lane['strategy_id']}")
    orb_cap = lane.get("max_orb_size_pts")
    cap_str = f"{orb_cap:.0f} pts" if orb_cap else "NONE"
    print(f"Filter: {lane['filter_type']} | RR: {lane['rr_target']} | ORB: O{lane['orb_minutes']} | ORB Cap: {cap_str}")
    print(f"{'=' * 70}")

    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        indicator = "[+]" if ok else "[X]"
        print(f"  {indicator} {name}: {msg}")

    print(f"{'=' * 70}")
    gate = "GO" if all_pass else "NO-GO"
    print(f"  GATE STATUS: {gate}")
    if not all_pass:
        print("  ACTION: Fix FAIL items above before trading this session.")
    print(f"{'=' * 70}")

    # Lane 2 confirmation
    if lane.get("is_half_size") and all_pass:
        print("\n  ** Lane 2 (SINGAPORE_OPEN RR4.0) is 0.5x sizing. **")
        print("  ** You MUST enter only 1 micro lot. **")

    # Log to file
    log_data = {
        "date": str(today),
        "session": session,
        "gate": gate,
        "checks": [{"name": n, "pass": o, "detail": m} for n, o, m in results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file = STATE_DIR / f"session_checklist_{today}_{session}.json"
    log_file.write_text(json.dumps(log_data, indent=2))

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Pre-session health check (hard gate)")
    parser.add_argument("--session", required=True, choices=list(LANE_DEFS.keys()), help="Session to check")
    args = parser.parse_args()
    ok = run_checks(args.session)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
