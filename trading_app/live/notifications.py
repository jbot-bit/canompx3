"""Best-effort Telegram notifications for live trading events.

Wraps scripts/infra/telegram_feed.send_telegram() with fail-safe behavior.
Notification failure must NEVER affect the trading loop.
"""


def notify(instrument: str, message: str) -> None:
    """Send Telegram notification. Never raises."""
    try:
        from scripts.infra.telegram_feed import send_telegram

        send_telegram(f"[{instrument}] {message}")
    except Exception:
        pass  # notification failure must never affect trading
