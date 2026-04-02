#!/usr/bin/env python3
"""Trade signal alerter — Telegram push notifications for manual execution.

Two-phase alert design:
  Phase 1 (PRE-SESSION, -10min): Reminder + pre-session filter check (OVNRNG).
    For ORB-dependent filters (COST, ORB_VOL), reminds trader to check after ORB forms.
  Phase 2 (POST-ORB, +orb_minutes+1): ORB formed, evaluates ALL filters, sends levels.

daily_features is batch-built (not real-time), so Phase 2 computes ORB from bars_1m.

Usage:
    python scripts/infra/signal_alert.py              # Daemon mode
    python scripts/infra/signal_alert.py --once        # Check now
    python scripts/infra/signal_alert.py --test        # Send test alert
    pythonw scripts/infra/signal_alert.py              # Background (no console)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import duckdb

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.paper_trade_logger import _build_lanes
from trading_app.prop_profiles import ACCOUNT_PROFILES

# Build lanes from first active profile
_active_profile = next((pid for pid, p in ACCOUNT_PROFILES.items() if p.active), None)
LANES = _build_lanes(_active_profile) if _active_profile else ()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "signal_alert.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("signal_alert")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

ALERT_LEAD_MINUTES = 10  # Phase 1: minutes before session
POST_ORB_BUFFER_MINUTES = 1  # Phase 2: minutes after ORB window closes
ALERT_EXPIRE_MINUTES = 15  # Stop alerting after this
CHECK_INTERVAL_SECONDS = 60
STOP_MULTIPLIER = 0.75  # All deployed lanes use S0.75

# Pre-session filters (can evaluate before ORB forms)
PRE_SESSION_FILTERS = {"OVNRNG_100", "NO_FILTER"}

# ORB-dependent filters (need ORB data, evaluate post-ORB)
ORB_DEPENDENT_FILTERS = {"COST_LT08", "COST_LT10", "COST_LT12", "COST_LT15",
                         "ORB_VOL_8K", "ORB_G4", "ORB_G5", "ORB_G6", "ORB_G8"}

# Max ORB size per lane (from prop_profiles — execution safety)
MAX_ORB_SIZE: dict[str, float] = {
    "CME_PRE_COST": 120.0,
    "COMEX_ORBVOL": 80.0,
    "EUROPE_OVNRNG": 80.0,
    "NYSE_CLOSE_OVNRNG": 100.0,
    "TOKYO_COST": 80.0,
}


def send_telegram(text: str) -> bool:
    """Send a Telegram message."""
    import json
    import urllib.parse
    import urllib.request

    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Telegram not configured")
        return False

    data = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data)
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("ok", False)
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

BRISBANE_TZ = timezone(timedelta(hours=10))


def brisbane_now() -> datetime:
    return datetime.now(UTC).astimezone(BRISBANE_TZ)


def compute_trading_day(brisbane_dt: datetime) -> date:
    """Trading day: 09:00 Brisbane → next 09:00. Before 09:00 = previous day."""
    if brisbane_dt.hour < 9:
        return (brisbane_dt - timedelta(days=1)).date()
    return brisbane_dt.date()


def get_session_time(orb_label: str, trading_day: date) -> tuple[int, int]:
    """Session start in Brisbane (hour, minute)."""
    return SESSION_CATALOG[orb_label]["resolver"](trading_day)


# ---------------------------------------------------------------------------
# Real-time ORB computation from bars_1m
# ---------------------------------------------------------------------------


def compute_live_orb(instrument: str, orb_label: str, orb_minutes: int,
                     trading_day: date) -> dict | None:
    """Compute ORB high/low/size from bars_1m for the current session.

    Returns None if insufficient data (ORB window not complete yet).
    """
    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            # Get session start time for this trading day
            sess_h, sess_m = get_session_time(orb_label, trading_day)

            # Convert Brisbane session time to UTC
            utc_h = (sess_h - 10) % 24
            # If crossing midnight (Brisbane time > 14:00 maps to same UTC day,
            # but Brisbane < 10:00 maps to previous UTC day)
            if sess_h < 10:
                # Session is after midnight Brisbane = same UTC day
                utc_date = trading_day + timedelta(days=1)
            else:
                utc_date = trading_day

            orb_start_utc = f"{utc_date} {utc_h:02d}:{sess_m:02d}:00"
            orb_end_utc = f"TIMESTAMP '{orb_start_utc}' + INTERVAL '{orb_minutes} minutes'"

            row = con.execute(
                f"""
                SELECT MAX(high) as orb_high, MIN(low) as orb_low,
                       MAX(high) - MIN(low) as orb_size,
                       COUNT(*) as bar_count
                FROM bars_1m
                WHERE symbol = ?
                  AND ts_utc >= TIMESTAMP '{orb_start_utc}'
                  AND ts_utc < {orb_end_utc}
                """,
                [instrument],
            ).fetchone()

            if row is None or row[3] < orb_minutes:
                return None  # ORB window not complete

            return {
                "orb_high": row[0],
                "orb_low": row[1],
                "orb_size": row[2],
            }
        finally:
            con.close()
    except duckdb.IOException:
        return None  # DB locked
    except duckdb.BinderException as e:
        log.error(f"SQL error in compute_live_orb: {e}")
        return None


# ---------------------------------------------------------------------------
# Pre-session filter check (overnight range from bars_1m)
# ---------------------------------------------------------------------------


def check_overnight_range(instrument: str, trading_day: date) -> tuple[float, bool] | None:
    """Query overnight range from daily_features (batch-built, may be stale).

    Returns (value, is_current) or None. is_current=False means using prev day's data.
    """
    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            # Use daily_features from the most recent completed trading day
            td = trading_day
            row = con.execute(
                f"""
                SELECT overnight_range
                FROM daily_features
                WHERE symbol = ? AND trading_day = ? AND orb_minutes = 5
                LIMIT 1
                """,
                [instrument, str(td)],
            ).fetchone()

            if row and row[0] is not None:
                return float(row[0]), True

            # Try previous trading day if today's not built yet
            td_prev = td - timedelta(days=1)
            row = con.execute(
                """
                SELECT overnight_range
                FROM daily_features
                WHERE symbol = ? AND trading_day = ? AND orb_minutes = 5
                LIMIT 1
                """,
                [instrument, str(td_prev)],
            ).fetchone()

            if row and row[0] is not None:
                return float(row[0]), False  # prev day — stale

            return None
        finally:
            con.close()
    except (duckdb.IOException, duckdb.BinderException):
        return None


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


def format_pre_session_alert(lane, session_time_str: str,
                             filter_result: str) -> str:
    """Phase 1: Pre-session reminder."""
    return (
        f"<b>SESSION APPROACHING</b>\n"
        f"\n"
        f"<b>{lane.instrument} {lane.orb_label}</b>\n"
        f"Time: {session_time_str} Brisbane\n"
        f"Lane: {lane.lane_name}\n"
        f"\n"
        f"Entry: E2 stop-market (both directions)\n"
        f"RR: {lane.rr_target} | Stop: S{STOP_MULTIPLIER}\n"
        f"Filter: {lane.filter_type}\n"
        f"{filter_result}\n"
        f"\n"
        f"<i>Watch for ORB formation on TopStepX</i>"
    )


def format_post_orb_alert(lane, orb: dict, session_time_str: str,
                          filter_pass: bool) -> str:
    """Phase 2: ORB formed, trade levels ready."""
    orb_high = orb["orb_high"]
    orb_low = orb["orb_low"]
    orb_size = orb["orb_size"]

    max_size = MAX_ORB_SIZE.get(lane.lane_name)
    size_warn = ""
    if max_size and orb_size > max_size:
        size_warn = f"\n<b>WARNING: ORB {orb_size:.1f} > max {max_size:.0f} — SKIP</b>"

    if not filter_pass:
        return (
            f"<b>ORB FORMED — FILTER FAIL</b>\n"
            f"\n"
            f"<b>{lane.instrument} {lane.orb_label}</b>\n"
            f"ORB: {orb_low:.2f} — {orb_high:.2f} ({orb_size:.1f} pts)\n"
            f"Filter: {lane.filter_type} FAIL — NO TRADE"
        )

    return (
        f"<b>TRADE SIGNAL</b>\n"
        f"\n"
        f"<b>{lane.instrument} {lane.orb_label}</b>\n"
        f"Lane: {lane.lane_name}\n"
        f"\n"
        f"ORB: {orb_low:.2f} — {orb_high:.2f} ({orb_size:.1f} pts)\n"
        f"  BUY STOP: {orb_high:.2f}\n"
        f"  SELL STOP: {orb_low:.2f}\n"
        f"\n"
        f"RR: {lane.rr_target} | Stop: S{STOP_MULTIPLIER}\n"
        f"Filter: {lane.filter_type} PASS"
        f"{size_warn}\n"
        f"\n"
        f"<i>Place stop-market orders on TopStepX now</i>"
    )


# ---------------------------------------------------------------------------
# Filter evaluation (post-ORB)
# ---------------------------------------------------------------------------


def evaluate_filter_post_orb(lane, orb: dict) -> bool:
    """Evaluate a lane's filter using live ORB data. Returns True if passes."""
    orb_size = orb.get("orb_size", 0)
    if orb_size <= 0:
        return False

    ft = lane.filter_type

    if ft == "NO_FILTER":
        return True

    if ft in ("COST_LT08", "COST_LT10"):
        from pipeline.cost_model import COST_SPECS

        friction = COST_SPECS[lane.instrument].total_friction
        threshold = 0.08 if ft == "COST_LT08" else 0.10
        return (friction / (orb_size * 2.0)) < threshold

    if ft.startswith("ORB_G"):
        # ORB size gate: G4=4pts, G5=5pts, etc.
        min_size = int(ft.split("G")[1])
        return orb_size >= min_size

    if ft == "ORB_VOL_8K":
        # Can't evaluate without volume data from bars_1m
        # Conservative: pass (the trader can see volume on chart)
        return True

    if ft.startswith("OVNRNG"):
        # Already checked in pre-session, always pass here
        return True

    # Unknown filter — fail-closed
    log.warning(f"Unknown filter type: {ft}")
    return False


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def scan_and_alert(alerted: dict[str, set[str]]) -> dict[str, set[str]]:
    """Scan lanes, send phase 1 (pre-session) and phase 2 (post-ORB) alerts.

    alerted keys: "pre" and "post", each a set of lane_date keys.
    """
    now = brisbane_now()
    today_cal = now.date()
    now_min = now.hour * 60 + now.minute

    for lane in LANES:
        # Determine trading day for this session
        try:
            sess_h, sess_m = get_session_time(lane.orb_label, today_cal)
        except Exception:
            continue

        # Compute trading day (sessions before 09:00 Brisbane = previous day)
        if sess_h < 9:
            trading_day = today_cal - timedelta(days=1)
        else:
            trading_day = today_cal

        sess_min = sess_h * 60 + sess_m
        time_str = f"{sess_h:02d}:{sess_m:02d}"

        # --- Phase 1: Pre-session reminder ---
        pre_key = f"{lane.lane_name}_{today_cal}_pre"
        pre_start = sess_min - ALERT_LEAD_MINUTES
        pre_end = sess_min

        if pre_key not in alerted.get("pre", set()) and pre_start <= now_min <= pre_end:
            filter_result = ""
            if lane.filter_type in PRE_SESSION_FILTERS:
                if lane.filter_type.startswith("OVNRNG"):
                    threshold = 100  # OVNRNG_100
                    result = check_overnight_range(lane.instrument, trading_day)
                    if result is not None:
                        ovn, is_current = result
                        stale = "" if is_current else " (prev day — verify)"
                        if ovn >= threshold:
                            filter_result = f"Overnight range: {ovn:.0f} pts >= {threshold} PASS{stale}"
                        else:
                            filter_result = f"Overnight range: {ovn:.0f} pts < {threshold} FAIL — SKIP{stale}"
                            alerted.setdefault("post", set()).add(
                                f"{lane.lane_name}_{today_cal}_post"
                            )  # No need for phase 2
                    else:
                        filter_result = "Overnight range: data unavailable — check manually"
                else:
                    filter_result = "No filter — always eligible"
            else:
                filter_result = f"Check {lane.filter_type} after ORB forms ({lane.orb_minutes}min)"

            msg = format_pre_session_alert(lane, time_str, filter_result)
            ok = send_telegram(msg)
            if ok:
                log.info(f"  PRE-SESSION: {lane.lane_name} ({time_str})")
            alerted.setdefault("pre", set()).add(pre_key)

        # --- Phase 2: Post-ORB trade signal ---
        post_key = f"{lane.lane_name}_{today_cal}_post"
        post_start = sess_min + lane.orb_minutes + POST_ORB_BUFFER_MINUTES
        post_end = post_start + ALERT_EXPIRE_MINUTES

        if post_key not in alerted.get("post", set()) and post_start <= now_min <= post_end:
            orb = compute_live_orb(lane.instrument, lane.orb_label,
                                   lane.orb_minutes, trading_day)
            if orb is None:
                log.info(f"  POST-ORB: {lane.lane_name} — ORB data unavailable, will retry")
                continue  # Don't mark as alerted — retry next cycle

            filter_pass = evaluate_filter_post_orb(lane, orb)
            msg = format_post_orb_alert(lane, orb, time_str, filter_pass)
            ok = send_telegram(msg)
            if ok:
                status = "SIGNAL" if filter_pass else "FILTER FAIL"
                log.info(f"  POST-ORB: {lane.lane_name} — {status}")
            alerted.setdefault("post", set()).add(post_key)

    return alerted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if "--test" in sys.argv:
        ok = send_telegram(
            "<b>SIGNAL TEST</b>\n\ncanompx3 trade alerter is working.\n"
            "You will receive alerts before each session."
        )
        print(f"Test alert: {'OK' if ok else 'FAIL'}")
        return

    if "--once" in sys.argv:
        log.info("One-shot scan...")
        scan_and_alert({"pre": set(), "post": set()})
        return

    # Daemon mode
    log.info("=" * 50)
    log.info("  SIGNAL ALERTER — daemon mode")
    log.info(f"  Lanes: {len(LANES)}")
    log.info(f"  Pre-session: {ALERT_LEAD_MINUTES}min before")
    log.info(f"  Post-ORB: +orb_minutes + {POST_ORB_BUFFER_MINUTES}min")
    log.info("=" * 50)

    for lane in LANES:
        try:
            h, m = get_session_time(lane.orb_label, date.today())
            log.info(f"  {lane.lane_name:<18} {lane.orb_label:<18} {h:02d}:{m:02d} Bris")
        except Exception:
            log.info(f"  {lane.lane_name:<18} {lane.orb_label:<18} (time error)")

    alerted: dict[str, set[str]] = {"pre": set(), "post": set()}

    while True:
        try:
            alerted = scan_and_alert(alerted)

            # Reset at 09:00 Brisbane (new trading day)
            now = brisbane_now()
            if now.hour == 9 and now.minute < 2:
                alerted = {"pre": set(), "post": set()}

        except Exception as e:
            log.error(f"Scan error: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
