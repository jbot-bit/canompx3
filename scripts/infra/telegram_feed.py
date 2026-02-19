"""
Process monitor -> Telegram alerts.

Checks outcome_builder heartbeat files and alerts if stale.
Runs silently on login. Zero token cost — pure Python, no LLM calls.

Usage:
    pythonw scripts/infra/telegram_feed.py          # daemon (no console)
    python scripts/infra/telegram_feed.py --once "msg"  # one-off message
"""

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

BOT_TOKEN = "8572496011:AAFFDahKzbGbROndyPSFbH52VjoyCcmPWT0"
CHAT_ID = "6812728770"

HEARTBEAT_PATHS = [
    Path(r"C:\Users\joshd\canompx3\outcome_builder.heartbeat"),
    Path(r"C:\db\outcome_builder.heartbeat"),
]
STALE_MINUTES = 20
CHECK_INTERVAL = 300  # 5 minutes


def send_telegram(text: str) -> bool:
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("ok", False)
    except Exception:
        return False


def monitor():
    alerted = set()

    while True:
        for hb_path in HEARTBEAT_PATHS:
            if not hb_path.exists():
                continue

            age_min = (time.time() - hb_path.stat().st_mtime) / 60

            if age_min > STALE_MINUTES and str(hb_path) not in alerted:
                content = hb_path.read_text().strip()
                send_telegram(
                    f"<b>STALE HEARTBEAT</b>\n"
                    f"outcome_builder hasn't updated in {age_min:.0f}min\n"
                    f"Last: {content}\n"
                    f"Process may be dead"
                )
                alerted.add(str(hb_path))

            elif age_min <= STALE_MINUTES:
                alerted.discard(str(hb_path))

        time.sleep(CHECK_INTERVAL)


def main():
    if "--once" in sys.argv:
        idx = sys.argv.index("--once")
        text = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "ping"
        send_telegram(text)
        return

    monitor()


if __name__ == "__main__":
    main()
