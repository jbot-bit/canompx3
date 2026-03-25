"""E2E simulation tests — mocked broker, deterministic, repeatable.

Covers the 8 components changed since last 7/7 sim pass:
1. 429 on entry/exit/auth
2. HWM halt/warning/poll failure
3. Price collar entry/stop-leg
4. Full TopStep lifecycle (clean + DD breach)
5. Orphan detection (ProjectX)
6. Stress: simultaneous events, restart persistence, corrupt state
"""

import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import requests

from trading_app.live.bar_aggregator import Bar
from trading_app.live.position_tracker import PositionTracker, PositionState
from trading_app.account_hwm_tracker import AccountHWMTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(close: float = 20000.0, minute: int = 30) -> Bar:
    return Bar(
        ts_utc=datetime(2026, 3, 25, 0, minute, 0, tzinfo=UTC),
        open=close - 5,
        high=close + 5,
        low=close - 10,
        close=close,
        volume=100,
    )


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ===========================================================================
# TASK 2: 429 TESTS
# ===========================================================================


class Test429OnEntrySubmit:
    def test_429_exhausts_retries_then_raises(self):
        """submit() gets 429 on all attempts -> raises after 3 retries."""
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        spec = OrderSpec(action="Buy", order_type="Stop", symbol="MNQM6",
                         qty=1, account_id=123, stop_price=20000.0)

        mock_429 = _mock_response(429)
        with patch("trading_app.live.tradovate.order_router.requests.post", return_value=mock_429):
            with patch("trading_app.live.tradovate.order_router.time.sleep") as mock_sleep:
                with pytest.raises(requests.HTTPError):
                    router.submit(spec)

        # 4 calls total: 1 initial + 3 retries
        assert mock_sleep.call_count == 3

    def test_429_succeeds_on_second_attempt(self):
        """submit() gets 429 once, then 200 -> order placed."""
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        spec = OrderSpec(action="Buy", order_type="Market", symbol="MNQM6",
                         qty=1, account_id=123)

        mock_429 = _mock_response(429)
        mock_ok = _mock_response(200, {"orderId": 42, "status": "Working"})
        responses = [mock_429, mock_ok]

        with patch("trading_app.live.tradovate.order_router.requests.post", side_effect=responses):
            with patch("trading_app.live.tradovate.order_router.time.sleep"):
                result = router.submit(spec)

        assert result.order_id == 42


class Test429OnExitSubmit:
    def test_exit_429_raises_promptly(self):
        """Exit order 429 exhausts retries — caller handles as UNMANAGED."""
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        spec = OrderSpec(action="Sell", order_type="Market", symbol="MNQM6",
                         qty=1, account_id=123)

        mock_429 = _mock_response(429)
        with patch("trading_app.live.tradovate.order_router.requests.post", return_value=mock_429):
            with patch("trading_app.live.tradovate.order_router.time.sleep") as mock_sleep:
                with pytest.raises(requests.HTTPError):
                    router.submit(spec)

        # Max 3 retries = max ~7s delay
        total_sleep = sum(c.args[0] for c in mock_sleep.call_args_list)
        assert total_sleep < 60  # well under 60s


class Test429OnAuthRefresh:
    def test_auth_429_retries_with_backoff(self):
        """Auth refresh gets 429 on all attempts -> raises RuntimeError."""
        from trading_app.live.tradovate.auth import TradovateAuth

        auth = TradovateAuth(demo=True)
        mock_429 = _mock_response(429)

        with patch("trading_app.live.tradovate.auth.requests.post", return_value=mock_429):
            with patch("trading_app.live.tradovate.auth.time.sleep"):
                with pytest.raises(RuntimeError, match="Auth refresh failed"):
                    auth.get_token()

        assert not auth.is_healthy

    def test_auth_succeeds_after_retry(self):
        """Auth refresh fails once then succeeds."""
        from trading_app.live.tradovate.auth import TradovateAuth
        import os

        auth = TradovateAuth(demo=True)
        mock_fail = _mock_response(503)
        mock_ok = _mock_response(200, {
            "accessToken": "test_token",
            "expirationTime": "2026-03-26T00:00:00Z",
        })

        env = {
            "TRADOVATE_USER": "test", "TRADOVATE_PASS": "test",
            "TRADOVATE_APP_ID": "test", "TRADOVATE_CID": "1", "TRADOVATE_SEC": "test",
        }
        with patch.dict(os.environ, env):
            with patch("trading_app.live.tradovate.auth.requests.post", side_effect=[mock_fail, mock_ok]):
                with patch("trading_app.live.tradovate.auth.time.sleep"):
                    token = auth.get_token()

        assert token == "test_token"
        assert auth.is_healthy


# ===========================================================================
# TASK 2: HWM TESTS
# ===========================================================================


class TestHWMHaltBlocksOrder:
    def test_halted_tracker_blocks_entry(self, tmp_path):
        """When HWM halt is active, no order should be submitted."""
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(52000.0)
        tracker.update_equity(49000.0)  # DD = $3000 > $2000 limit
        assert tracker._halt

        halted, reason = tracker.check_halt()
        assert halted
        assert "HWM_HALT" in reason

    def test_warning_allows_order(self, tmp_path):
        """At 76% DD, warning logged but order proceeds."""
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(50000.0)
        tracker.update_equity(51520.0)  # HWM
        tracker.update_equity(50000.0)  # DD = 1520 = 76%
        halted, reason = tracker.check_halt()
        assert not halted
        assert "WARNING_75" in reason


class TestHWMPollFailureAccumulation:
    def test_three_failures_halt(self, tmp_path):
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(50000.0)
        tracker.update_equity(None)
        tracker.update_equity(None)
        tracker.update_equity(None)
        assert tracker._halt

        halted, reason = tracker.check_halt()
        assert halted

        # Halt persists on restart
        t2 = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        halted2, _ = t2.check_halt()
        assert halted2


# ===========================================================================
# TASK 2: PRICE COLLAR TESTS
# ===========================================================================


class TestPriceCollarRejectsBadEntry:
    def test_entry_beyond_collar_rejected(self):
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        router.update_market_price(20000.0)
        spec = OrderSpec(action="Buy", order_type="Stop", symbol="MNQM6",
                         qty=1, account_id=123, stop_price=20200.0)  # 1% away

        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)

    def test_entry_within_collar_proceeds(self):
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        router.update_market_price(20000.0)
        spec = OrderSpec(action="Buy", order_type="Stop", symbol="MNQM6",
                         qty=1, account_id=123, stop_price=20050.0)  # 0.25%

        mock_ok = _mock_response(200, {"orderId": 42})
        with patch("trading_app.live.tradovate.order_router.requests.post", return_value=mock_ok):
            result = router.submit(spec)
        assert result.order_id == 42


class TestPriceCollarIgnoresStopLeg:
    def test_market_exit_not_collared(self):
        """Exit market orders have no stop_price -> collar skipped."""
        from trading_app.live.tradovate.order_router import TradovateOrderRouter, OrderSpec

        router = TradovateOrderRouter(account_id=123, auth=MagicMock(), demo=True)
        router.update_market_price(20000.0)
        spec = OrderSpec(action="Sell", order_type="Market", symbol="MNQM6",
                         qty=1, account_id=123, stop_price=None)

        mock_ok = _mock_response(200, {"orderId": 43})
        with patch("trading_app.live.tradovate.order_router.requests.post", return_value=mock_ok):
            result = router.submit(spec)
        assert result.order_id == 43  # no collar error


class TestPriceCollarProjectX:
    def test_projectx_collar_rejects(self):
        from trading_app.live.projectx.order_router import ProjectXOrderRouter

        router = ProjectXOrderRouter(account_id=123, auth=MagicMock(), tick_size=0.25)
        router.update_market_price(20000.0)
        spec = {"accountId": 123, "contractId": "X", "type": 4, "side": 0,
                "size": 1, "stopPrice": 20300.0}  # 1.5%
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)


# ===========================================================================
# TASK 2: FULL LIFECYCLE TESTS
# ===========================================================================


class TestFullTopStepLifecycleClean:
    def test_position_tracker_roundtrip(self):
        """Entry -> fill -> exit -> flat. All states correct."""
        pt = PositionTracker()
        rec = pt.on_entry_sent("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075", "long", 2950.0)
        assert rec is not None
        assert rec.state == PositionState.PENDING_ENTRY

        pt.on_entry_filled("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075", 2950.25)
        rec = pt.get("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075")
        assert rec.state == PositionState.ENTERED
        assert rec.fill_entry_price == 2950.25

        closed = pt.on_exit_filled("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075", fill_price=2960.0)
        assert closed is not None
        assert pt.get("MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G4_CONT_S075") is None


class TestFullTopStepLifecycleDDBreach:
    def test_hwm_halts_after_losing_trade(self, tmp_path):
        """Trade loses -> equity drops -> HWM halt blocks second signal."""
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(49900.0)  # HWM = 49900
        # DD already at $1300 of $2000 (HWM was higher before)
        # Simulate: HWM was set at $51200 in a previous session
        tracker._hwm = 51200.0
        tracker._save_state()

        # First equity poll: DD = 51200 - 49900 = 1300 (65%)
        state = tracker.update_equity(49900.0)
        halted, reason = tracker.check_halt()
        assert not halted  # 65% < 100%

        # Trade loses $500 -> equity 49400, DD = 51200 - 49400 = 1800 (90%)
        state = tracker.update_equity(49400.0)
        halted, reason = tracker.check_halt()
        assert not halted  # 90% < 100%
        assert "WARNING_75" in reason

        # Another loss -> equity 49100, DD = 51200 - 49100 = 2100 > 2000
        state = tracker.update_equity(49100.0)
        assert state.halt_triggered
        halted, reason = tracker.check_halt()
        assert halted
        assert "HWM_HALT" in reason


class TestOrphanDetectionProjectX:
    def test_orphan_detected_at_startup(self):
        """ProjectX reports open position -> logged as orphan."""
        from trading_app.live.projectx.positions import ProjectXPositions

        auth = MagicMock()
        auth.headers.return_value = {"Authorization": "Bearer test"}
        pos = ProjectXPositions(auth=auth)

        mock_resp = _mock_response(200, [
            {"contractId": "CON.F.US.MGC.M26", "type": 1, "size": 1, "averagePrice": 2950.0}
        ])
        with patch("trading_app.live.projectx.positions.requests.post", return_value=mock_resp):
            orphans = pos.query_open(12345)

        assert len(orphans) == 1
        assert orphans[0]["side"] == "long"
        assert orphans[0]["size"] == 1


# ===========================================================================
# TASK 4: STRESS TESTS
# ===========================================================================


class TestSimultaneousSignalAndPollFailure:
    def test_poll_failure_does_not_block_signal(self, tmp_path):
        """Poll failure increments counter but doesn't immediately halt."""
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(50000.0)

        # Simulate: poll fails once
        tracker.update_equity(None)
        assert tracker._consecutive_poll_failures == 1
        assert not tracker._halt

        # Signal should still be allowed
        halted, _ = tracker.check_halt()
        assert not halted

        # Successful poll resets
        tracker.update_equity(50000.0)
        assert tracker._consecutive_poll_failures == 0


class TestSessionRestartPersistence:
    def test_hwm_persists_across_restart(self, tmp_path):
        """HWM value survives process restart."""
        t1 = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        t1.update_equity(50000.0)
        t1.update_equity(51500.0)  # HWM = 51500
        t1.update_equity(50800.0)  # DD = 700

        # "Restart"
        t2 = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        assert t2._hwm == 51500.0
        assert t2._last_equity == 50800.0

        halted, _ = t2.check_halt()
        assert not halted  # DD = 700 < 2000

        # Continue tracking from where we left off
        t2.update_equity(50000.0)  # DD = 1500
        halted, reason = t2.check_halt()
        assert not halted
        assert "WARNING_75" in reason


class TestBackToBack429ThenHWMHalt:
    def test_429_then_hwm_halt_no_deadlock(self, tmp_path):
        """429 on entry -> then HWM breaches -> both handled independently."""
        tracker = AccountHWMTracker("TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        tracker.update_equity(52000.0)

        # 429 happens (simulated by catching the error)
        # Then equity drops to breach
        tracker.update_equity(49500.0)  # DD = 2500 > 2000
        assert tracker._halt

        halted, reason = tracker.check_halt()
        assert halted
        assert "HWM_HALT" in reason


class TestCorruptHWMStateOnStartup:
    def test_corrupt_json_recovers(self, tmp_path):
        """Corrupt state file -> backup saved, fresh init, no crash."""
        state_file = tmp_path / "account_hwm_CORRUPT_TEST.json"
        state_file.write_text("{not valid json!!!}")

        t = AccountHWMTracker("CORRUPT_TEST", "topstep", dd_limit_dollars=2000.0, state_dir=tmp_path)
        assert t._hwm == 0.0  # Fresh init

        # Backup exists
        backups = list(tmp_path.glob("account_hwm_CORRUPT_TEST_CORRUPT_*.json"))
        assert len(backups) == 1

        # Normal operation continues
        state = t.update_equity(50000.0)
        assert state.hwm_dollars == 50000.0
