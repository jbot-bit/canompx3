"""Tests for trading_app.live.notifications.notify() bool contract.

notify() must:
- Return True when send_telegram() succeeds.
- Return False when send_telegram() raises (broken Telegram config).
- Never raise itself — trading loop must not be affected.
"""

from unittest.mock import patch

from trading_app.live import notifications


def test_notify_returns_true_on_success():
    with patch("scripts.infra.telegram_feed.send_telegram") as send:
        send.return_value = None
        assert notifications.notify("MNQ", "hello") is True
    send.assert_called_once_with("[MNQ] hello")


def test_notify_returns_false_when_send_telegram_raises():
    with patch("scripts.infra.telegram_feed.send_telegram", side_effect=RuntimeError("bad token")):
        assert notifications.notify("MGC", "drift") is False


def test_notify_never_raises_on_arbitrary_exception():
    with patch("scripts.infra.telegram_feed.send_telegram", side_effect=Exception("net dead")):
        try:
            result = notifications.notify("MES", "test")
        except Exception:
            raise AssertionError("notify() must never raise on Exception subclasses")
    assert result is False
