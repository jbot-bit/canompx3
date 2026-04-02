#!/usr/bin/env python3
"""Trade signal alerter — Telegram push notifications for manual execution.

Scans deployed lanes, checks if session is approaching, evaluates filters
against today's daily_features, and sends Telegram alerts with trade details.

Runs as a daemon (checks every 60s) or one-shot mode.

Usage:
    python scripts/infra/signal_alert.py              # Daemon mode
    python scripts/infra/signal_alert.py --once        # Check now, alert if applicable
    python scripts/infra/signal_alert.py --test        # Send test alert
    pythonw scripts/infra/signal_alert.py              # Background (no console)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import duckdb

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH
from trading_app.paper_trade_logger import LANES

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
# Telegram
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Minutes before session to send alert
ALERT_LEAD_MINUTES = 10

# Minutes after session start to stop alerting (avoid stale alerts)
ALERT_EXPIRE_MINUTES = 5

# Check interval in daemon mode
CHECK_INTERVAL_SECONDS = 60


def send_telegram(text: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    import json
    import urllib.parse
    import urllib.request

    if not BOT_TOKEN or not CHAT_ID:
        log.warning("Telegram not configured (missing BOT_TOKEN or CHAT_ID)")
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
# Session time helpers
# ---------------------------------------------------------------------------


def get_brisbane_now() -> datetime:
    """Current time in Brisbane (UTC+10, no DST)."""
    from datetime import UTC, timezone

    utc_now = datetime.now(UTC)
    brisbane_offset = timezone(timedelta(hours=10))
    return utc_now.astimezone(brisbane_offset)


def get_session_brisbane_time(orb_label: str, trading_day: date) -> tuple[int, int]:
    """Get session start time in Brisbane hours/minutes."""
    info = SESSION_CATALOG[orb_label]
    return info["resolver"](trading_day)


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------


def evaluate_filter(lane, trading_day: date) -> dict | None:
    """Check if today's data passes the lane's filter.

    Returns dict with ORB details if filter passes, None if it fails or data missing.
    """
    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        try:
            # Get today's daily_features for this session
            row = con.execute(
                """
                SELECT orb_high, orb_low, orb_size, atr_20,
                       overnight_range_pts, orb_dollar_volume
                FROM daily_features
                WHERE trading_day = ? AND symbol = ? AND orb_minutes = ?
                ORDER BY trading_day DESC
                LIMIT 1
                """,
                [str(trading_day), lane.instrument, lane.orb_minutes],
            ).fetchone()

            if row is None:
                return None

            orb_high, orb_low, orb_size, atr_20, overnight_range, orb_dollar_volume = row

            if orb_size is None or orb_size <= 0:
                return None

            # Evaluate filter
            filter_pass = True
            if lane.filter_sql:
                # Build the evaluation context
                check = con.execute(
                    f"""
                    SELECT CASE WHEN {lane.filter_sql} THEN 1 ELSE 0 END
                    FROM daily_features
                    WHERE trading_day = ? AND symbol = ? AND orb_minutes = ?
                    LIMIT 1
                    """,
                    [str(trading_day), lane.instrument, lane.orb_minutes],
                ).fetchone()
                filter_pass = check is not None and check[0] == 1

            if not filter_pass:
                return None

            return {
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_size": orb_size,
                "atr_20": atr_20,
            }

        finally:
            con.close()

    except duckdb.IOException:
        # DB locked — skip this check, will retry next cycle
        return None
    except Exception as e:
        log.error(f"Filter eval error for {lane.lane_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Alert formatting
# ---------------------------------------------------------------------------


def format_alert(lane, orb_data: dict, session_time_str: str) -> str:
    """Format a trade signal alert message."""
    orb_high = orb_data["orb_high"]
    orb_low = orb_data["orb_low"]
    orb_size = orb_data["orb_size"]

    # E2 = stop-market at ORB boundary (both directions)
    stop_mult = "S0.75" if "_S075" in (lane.strategy_id or "") else "S1.0"

    return (
        f"<b>TRADE SIGNAL</b>\n"
        f"\n"
        f"<b>{lane.instrument} {lane.orb_label}</b>\n"
        f"Lane: {lane.lane_name}\n"
        f"Session: {session_time_str} Brisbane\n"
        f"\n"
        f"Entry: E2 (stop-market at ORB break)\n"
        f"  LONG above {orb_high:.2f}\n"
        f"  SHORT below {orb_low:.2f}\n"
        f"ORB size: {orb_size:.2f} pts\n"
        f"RR target: {lane.rr_target}\n"
        f"Stop: {stop_mult}\n"
        f"Filter: {lane.filter_type} PASS\n"
        f"\n"
        f"<i>Place stop-market order on TopStepX</i>"
    )


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------


def scan_and_alert(already_alerted: set[str]) -> set[str]:
    """Scan all lanes, alert if session approaching and filter passes.

    Returns updated set of alerted lane+date keys (to avoid duplicates).
    """
    now = get_brisbane_now()
    today = now.date()
    now_minutes = now.hour * 60 + now.minute

    for lane in LANES:
        alert_key = f"{lane.lane_name}_{today}"

        # Skip if already alerted today
        if alert_key in already_alerted:
            continue

        # Get session time
        try:
            sess_h, sess_m = get_session_brisbane_time(lane.orb_label, today)
        except Exception:
            continue

        sess_minutes = sess_h * 60 + sess_m
        alert_start = sess_minutes - ALERT_LEAD_MINUTES
        alert_end = sess_minutes + ALERT_EXPIRE_MINUTES

        # Handle midnight crossing (e.g., sessions at 00:00-06:00 Brisbane)
        if alert_start < 0:
            alert_start += 1440
        if now_minutes < alert_start and alert_start > 1200 and now_minutes < 300:
            # After midnight, session was before midnight
            continue

        # Check if we're in the alert window
        if alert_start <= now_minutes <= alert_end:
            # Evaluate filter
            orb_data = evaluate_filter(lane, today)

            if orb_data is None:
                log.info(f"  {lane.lane_name}: filter FAIL or no data — skip")
                already_alerted.add(alert_key)  # Don't retry today
                continue

            # Format and send
            time_str = f"{sess_h:02d}:{sess_m:02d}"
            msg = format_alert(lane, orb_data, time_str)
            ok = send_telegram(msg)

            if ok:
                log.info(f"  ALERT SENT: {lane.lane_name} ({lane.orb_label} {time_str})")
            else:
                log.error(f"  ALERT FAILED: {lane.lane_name}")

            already_alerted.add(alert_key)

    return already_alerted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    if "--test" in sys.argv:
        ok = send_telegram(
            "<b>SIGNAL TEST</b>\n\ncanompx3 trade alerter is working.\nYou will receive alerts before each session."
        )
        print(f"Test alert: {'OK' if ok else 'FAIL'}")
        return

    if "--once" in sys.argv:
        log.info("One-shot scan...")
        scan_and_alert(set())
        return

    # Daemon mode
    log.info("=" * 50)
    log.info("  SIGNAL ALERTER — daemon mode")
    log.info(f"  Lanes: {len(LANES)}")
    log.info(f"  Alert lead: {ALERT_LEAD_MINUTES} min before session")
    log.info(f"  Check interval: {CHECK_INTERVAL_SECONDS}s")
    log.info("=" * 50)

    for lane in LANES:
        try:
            h, m = get_session_brisbane_time(lane.orb_label, date.today())
            log.info(f"  {lane.lane_name:<18} {lane.orb_label:<18} {h:02d}:{m:02d} Brisbane")
        except Exception:
            log.info(f"  {lane.lane_name:<18} {lane.orb_label:<18} (time error)")

    already_alerted: set[str] = set()

    while True:
        try:
            already_alerted = scan_and_alert(already_alerted)

            # Reset alerts at midnight Brisbane
            now = get_brisbane_now()
            if now.hour == 0 and now.minute < 2:
                already_alerted.clear()

        except Exception as e:
            log.error(f"Scan error: {e}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
