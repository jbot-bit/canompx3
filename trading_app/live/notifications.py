"""Best-effort Telegram notifications for live trading events.

Wraps scripts/infra/telegram_feed.send_telegram() with fail-safe behavior.
Notification failure must NEVER affect the trading loop.
"""

import logging

log = logging.getLogger(__name__)


def notify(instrument: str, message: str) -> bool:
    """Send Telegram notification. Never raises. True on success, False on failure.

    The bool return lets startup self-tests distinguish a working Telegram pipe
    from a silently broken one. Existing call sites that ignore the return
    value keep their historical contract (never raises).
    """
    try:
        from scripts.infra.telegram_feed import send_telegram

        send_telegram(f"[{instrument}] {message}")
        return True
    except Exception as exc:
        log.warning("Notification failed for %s: %s", instrument, exc)
        return False
