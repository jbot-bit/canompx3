"""Tests for Tradovate broker integration + BrokerDispatcher."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from trading_app.live.broker_base import BrokerRouter
from trading_app.live.broker_dispatcher import BrokerDispatcher
from trading_app.live.tradovate.auth import TradovateAuth
from trading_app.live.tradovate.contracts import TradovateContracts
from trading_app.live.tradovate.http import RateLimitExhausted, request_with_retry
from trading_app.live.tradovate.order_router import TradovateOrderRouter
from trading_app.live.tradovate.positions import TradovatePositions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_auth():
    """TradovateAuth-like mock with base_url and headers."""
    auth = MagicMock(spec=TradovateAuth)
    auth.base_url = "https://demo.tradovateapi.com/v1"
    auth.headers.return_value = {"Authorization": "Bearer test-token"}
    auth.get_token.return_value = "test-token"
    return auth


@pytest.fixture()
def router(mock_auth):
    """TradovateOrderRouter with mock auth."""
    return TradovateOrderRouter(account_id=12345, auth=mock_auth, tick_size=0.25)


# ---------------------------------------------------------------------------
# TradovateAuth
# ---------------------------------------------------------------------------


class TestTradovateAuth:
    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEVICE_ID": "dev1",
            "TRADOVATE_DEMO": "1",
        },
    )
    def test_demo_base_url(self):
        auth = TradovateAuth()
        assert auth.base_url == "https://demo.tradovateapi.com/v1"

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
        },
    )
    def test_live_base_url_default(self):
        auth = TradovateAuth()
        assert auth.base_url == "https://live.tradovateapi.com/v1"

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    def test_login_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "accessToken": "tok123",
            "mdAccessToken": "md456",
            "userId": 42,
        }
        mock_post.return_value = mock_resp

        auth = TradovateAuth()
        token = auth.get_token()

        assert token == "tok123"
        assert auth.md_token == "md456"
        assert auth.user_id == 42
        assert auth.is_healthy is True

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    def test_login_pticket_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"p-ticket": "abc123"}
        mock_post.return_value = mock_resp

        auth = TradovateAuth()
        with pytest.raises(RuntimeError, match="2FA p-ticket"):
            auth.get_token()

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    @patch("trading_app.live.tradovate.auth.time.sleep")
    def test_login_retry_then_success(self, mock_sleep, mock_post):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.HTTPError("503")
        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {"accessToken": "tok", "userId": 1}
        mock_post.side_effect = [fail_resp, ok_resp]

        auth = TradovateAuth()
        token = auth.get_token()
        assert token == "tok"
        assert auth.is_healthy is True
        mock_sleep.assert_called_once()

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    def test_login_all_retries_fail(self, mock_post):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.HTTPError("503")
        mock_post.return_value = fail_resp

        auth = TradovateAuth()
        with patch("trading_app.live.tradovate.auth.time.sleep"), pytest.raises(
            RuntimeError, match="auth failed after"
        ):
            auth.get_token()
        assert auth.is_healthy is False

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    def test_headers_trigger_login(self):
        auth = TradovateAuth()
        with patch.object(auth, "get_token", return_value="tok"):
            h = auth.headers()
        assert h["Authorization"] == "Bearer tok"
        assert h["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# TradovateOrderRouter
# ---------------------------------------------------------------------------


class TestTradovateOrderRouter:
    def test_build_e2_stop_order(self, router):
        spec = router.build_order_spec("long", "E2", 100.50, "MNQM6", qty=2)
        assert spec["orderType"] == "Stop"
        assert spec["stopPrice"] == 100.50
        assert spec["action"] == "Buy"
        assert spec["orderQty"] == 2
        assert spec["isAutomated"] is True
        assert "_intent" in spec

    def test_build_e1_market_order(self, router):
        spec = router.build_order_spec("short", "E1", 0, "MGCM6")
        assert spec["orderType"] == "Market"
        assert spec["action"] == "Sell"
        assert "stopPrice" not in spec
        assert spec["isAutomated"] is True

    def test_unsupported_entry_model_raises(self, router):
        with pytest.raises(ValueError, match="E3"):
            router.build_order_spec("long", "E3", 100, "MNQM6")

    def test_build_exit_spec(self, router):
        spec = router.build_exit_spec("long", "MNQM6", qty=1)
        assert spec["action"] == "Sell"
        assert spec["orderType"] == "Market"
        assert spec["isAutomated"] is True

    def test_build_exit_spec_short(self, router):
        spec = router.build_exit_spec("short", "MNQM6")
        assert spec["action"] == "Buy"

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_submit_strips_intent(self, mock_req, router):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"orderId": 999, "fillPrice": 100.5}
        mock_req.return_value = mock_resp

        spec = router.build_order_spec("long", "E2", 100.5, "MNQM6")
        assert "_intent" in spec

        router.submit(spec)
        # Verify _intent was stripped from the wire payload
        call_kwargs = mock_req.call_args
        wire_body = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert "_intent" not in wire_body

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_submit_success(self, mock_req, router):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"orderId": 42, "fillPrice": 100.0}
        mock_req.return_value = mock_resp

        spec = router.build_order_spec("long", "E1", 0, "MNQM6")
        result = router.submit(spec)
        assert result["order_id"] == 42
        assert result["status"] == "submitted"
        assert result["fill_price"] == 100.0

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_submit_no_orderid_raises(self, mock_req, router):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"errorText": "bad order"}
        mock_req.return_value = mock_resp

        spec = router.build_order_spec("long", "E1", 0, "MNQM6")
        with pytest.raises(RuntimeError, match="no orderId"):
            router.submit(spec)

    def test_price_collar_rejects_deviation(self, router):
        router.update_market_price(100.0)
        spec = router.build_order_spec("long", "E2", 110.0, "MNQM6")  # 10% deviation
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)

    def test_price_collar_accepts_close_price(self, router):
        router.update_market_price(100.0)
        spec = router.build_order_spec("long", "E2", 100.3, "MNQM6")  # 0.3% — within 0.5%
        # Would need mock for the HTTP call, but collar check is BEFORE submit
        # Verify collar doesn't reject
        with patch("trading_app.live.tradovate.order_router.request_with_retry") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"orderId": 1}
            mock_req.return_value = mock_resp
            result = router.submit(spec)
            assert result["order_id"] == 1

    def test_bracket_spec_long(self, router):
        bracket = router.build_bracket_spec("long", "MNQM6", 100.0, 99.0, 102.0)
        assert bracket is not None
        assert bracket["bracket1"]["action"] == "Sell"  # target = sell
        assert bracket["bracket1"]["orderType"] == "Limit"
        assert bracket["bracket2"]["action"] == "Sell"  # stop = sell
        assert bracket["bracket2"]["orderType"] == "Stop"
        assert bracket["bracket2"]["stopPrice"] == 99.0

    def test_bracket_spec_short(self, router):
        bracket = router.build_bracket_spec("short", "MNQM6", 100.0, 101.0, 98.0)
        assert bracket is not None
        assert bracket["bracket1"]["action"] == "Buy"  # target = buy to close
        assert bracket["bracket2"]["action"] == "Buy"  # stop = buy to close

    def test_tick_size_positive_required(self, mock_auth):
        with pytest.raises(ValueError, match="tick_size must be positive"):
            TradovateOrderRouter(account_id=1, auth=mock_auth, tick_size=0)

    def test_no_auth_submit_raises(self):
        router = TradovateOrderRouter(account_id=1, auth=None, tick_size=0.25)
        with pytest.raises(RuntimeError, match="No auth"):
            router.submit({"orderType": "Market"})

    def test_no_auth_url_raises(self):
        router = TradovateOrderRouter(account_id=1, auth=None, tick_size=0.25)
        with pytest.raises(RuntimeError, match="No auth"):
            router._url("/test")

    def test_auth_missing_base_url_raises(self):
        auth = MagicMock(spec=["headers", "get_token", "refresh_if_needed"])
        router = TradovateOrderRouter(account_id=1, auth=auth, tick_size=0.25)
        with pytest.raises(RuntimeError, match="missing base_url"):
            router._url("/test")


# ---------------------------------------------------------------------------
# TradovateContracts
# ---------------------------------------------------------------------------


class TestTradovateContracts:
    def test_auth_missing_base_url_raises(self):
        auth = MagicMock(spec=["headers", "get_token", "refresh_if_needed"])
        with pytest.raises(RuntimeError, match="missing base_url"):
            TradovateContracts(auth=auth)

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_resolve_account_id(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {"id": 100, "name": "ACCT1", "active": True},
            {"id": 200, "name": "ACCT2", "active": False},
            {"id": 300, "name": "ACCT3", "active": True},
        ]
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        account_id = contracts.resolve_account_id()
        assert account_id == 100

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_resolve_all_accounts(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {"id": 100, "name": "A1", "active": True},
            {"id": 200, "name": "A2", "active": True},
        ]
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        accounts = contracts.resolve_all_account_ids()
        assert len(accounts) == 2
        assert accounts[0] == (100, "A1")

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_no_active_accounts_raises(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        with pytest.raises(RuntimeError, match="No active"):
            contracts.resolve_account_id()

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_resolve_front_month_success(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"name": "MNQM6", "contractSymbol": "MNQM6"}]
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        symbol = contracts.resolve_front_month("MNQ")
        assert symbol == "MNQM6"

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_resolve_front_month_no_contracts_raises(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        with pytest.raises(RuntimeError, match="No contracts found"):
            contracts.resolve_front_month("MNQ")

    @patch("trading_app.live.tradovate.contracts.request_with_retry")
    def test_resolve_front_month_empty_symbol_raises(self, mock_req, mock_auth):
        """API returns a contract object but both name fields are absent/empty."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"contractMaturityId": 999}]  # no name/contractSymbol
        mock_req.return_value = mock_resp

        contracts = TradovateContracts(auth=mock_auth)
        with pytest.raises(RuntimeError, match="no usable symbol"):
            contracts.resolve_front_month("MNQ")


# ---------------------------------------------------------------------------
# TradovatePositions
# ---------------------------------------------------------------------------


class TestTradovatePositions:
    def test_auth_missing_base_url_raises(self):
        auth = MagicMock(spec=["headers", "get_token", "refresh_if_needed"])
        with pytest.raises(RuntimeError, match="missing base_url"):
            TradovatePositions(auth=auth)

    @patch("trading_app.live.tradovate.positions.request_with_retry")
    def test_query_open_filters_by_account(self, mock_req, mock_auth):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {"accountId": 100, "contractId": "C1", "netPos": 2, "netPrice": 50.0},
            {"accountId": 999, "contractId": "C2", "netPos": 1, "netPrice": 60.0},
            {"accountId": 100, "contractId": "C3", "netPos": 0, "netPrice": 0},  # flat
            {"accountId": 100, "contractId": "C4", "netPos": -1, "netPrice": 70.0},
        ]
        mock_req.return_value = mock_resp

        positions = TradovatePositions(auth=mock_auth)
        result = positions.query_open(100)
        assert len(result) == 2
        assert result[0]["side"] == "long"
        assert result[0]["size"] == 2
        assert result[1]["side"] == "short"
        assert result[1]["size"] == 1


# ---------------------------------------------------------------------------
# HTTP retry
# ---------------------------------------------------------------------------


class TestRequestWithRetry:
    @patch("trading_app.live.tradovate.http.time.sleep")
    @patch("trading_app.live.tradovate.http.requests.post")
    def test_429_retry_then_success(self, mock_post, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        mock_post.side_effect = [resp_429, resp_ok]

        result = request_with_retry("POST", "http://test/api", {})
        assert result.status_code == 200
        mock_sleep.assert_called_once()

    @patch("trading_app.live.tradovate.http.requests.post")
    def test_429_exhausted_raises(self, mock_post):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_post.return_value = resp_429

        with patch("trading_app.live.tradovate.http.time.sleep"), pytest.raises(
            RateLimitExhausted
        ):
            request_with_retry("POST", "http://test/api", {})

    @patch("trading_app.live.tradovate.http.requests.get")
    def test_non_429_returns_immediately(self, mock_get):
        resp = MagicMock()
        resp.status_code = 500
        mock_get.return_value = resp

        result = request_with_retry("GET", "http://test/api", {})
        assert result.status_code == 500
        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# BrokerDispatcher
# ---------------------------------------------------------------------------


class TestBrokerDispatcher:
    def _make_router(self, account_id: int = 1) -> MagicMock:
        router = MagicMock(spec=BrokerRouter)
        router.account_id = account_id
        router.auth = MagicMock()
        return router

    def test_init_calls_super(self):
        primary = self._make_router(account_id=10)
        dispatcher = BrokerDispatcher(primary)
        assert dispatcher.account_id == 10
        assert dispatcher.auth is primary.auth

    def test_submit_fans_out_to_secondaries(self):
        primary = self._make_router()
        secondary = self._make_router(account_id=2)
        primary.submit.return_value = {"order_id": 100, "status": "submitted"}
        secondary.build_order_spec.return_value = {"adapted": True}
        secondary.submit.return_value = {"order_id": 200, "status": "submitted"}

        dispatcher = BrokerDispatcher(primary, [secondary])
        spec = {
            "action": "Buy",
            "_intent": {"direction": "long", "entry_model": "E2", "entry_price": 100, "symbol": "X", "qty": 1},
        }

        result = dispatcher.submit(spec)
        assert result["order_id"] == 100
        secondary.build_order_spec.assert_called_once_with(
            direction="long", entry_model="E2", entry_price=100, symbol="X", qty=1
        )
        secondary.submit.assert_called_once()

    def test_secondary_failure_does_not_block_primary(self):
        primary = self._make_router()
        secondary = self._make_router(account_id=2)
        primary.submit.return_value = {"order_id": 100, "status": "submitted"}
        secondary.build_order_spec.side_effect = RuntimeError("secondary down")

        dispatcher = BrokerDispatcher(primary, [secondary])
        spec = {
            "action": "Buy",
            "_intent": {"direction": "long", "entry_model": "E1", "entry_price": 0, "symbol": "X", "qty": 1},
        }
        result = dispatcher.submit(spec)
        assert result["order_id"] == 100  # primary still succeeded

    def test_exit_spec_attaches_exit_intent(self):
        primary = self._make_router()
        primary.build_exit_spec.return_value = {"action": "Sell", "orderType": "Market"}

        dispatcher = BrokerDispatcher(primary)
        spec = dispatcher.build_exit_spec("long", "MNQM6", qty=1)
        assert "_exit_intent" in spec
        assert spec["_exit_intent"]["direction"] == "long"

    def test_exit_routing_to_secondaries(self):
        primary = self._make_router()
        secondary = self._make_router(account_id=2)
        primary.submit.return_value = {"order_id": 100, "status": "submitted"}
        secondary.build_exit_spec.return_value = {"action": "Sell", "orderType": "Market"}
        secondary.submit.return_value = {"order_id": 200, "status": "submitted"}

        dispatcher = BrokerDispatcher(primary, [secondary])
        spec = {"action": "Sell", "orderType": "Market", "_exit_intent": {"direction": "long", "symbol": "X", "qty": 1}}
        result = dispatcher.submit(spec)

        assert result["order_id"] == 100
        secondary.build_exit_spec.assert_called_once_with(direction="long", symbol="X", qty=1)

    def test_update_market_price_forwards_to_all(self):
        primary = self._make_router()
        sec1 = self._make_router(account_id=2)
        sec2 = self._make_router(account_id=3)

        dispatcher = BrokerDispatcher(primary, [sec1, sec2])
        dispatcher.update_market_price(100.5)

        primary.update_market_price.assert_called_once_with(100.5)
        sec1.update_market_price.assert_called_once_with(100.5)
        sec2.update_market_price.assert_called_once_with(100.5)

    def test_query_order_status_delegates_to_primary(self):
        primary = self._make_router()
        primary.query_order_status.return_value = {"order_id": 42, "status": "Filled", "fill_price": 100.0}

        dispatcher = BrokerDispatcher(primary)
        result = dispatcher.query_order_status(42)
        assert result["status"] == "Filled"
        primary.query_order_status.assert_called_once_with(42)

    def test_all_routers_property(self):
        primary = self._make_router()
        sec = self._make_router(account_id=2)
        dispatcher = BrokerDispatcher(primary, [sec])
        assert len(dispatcher.all_routers) == 2

    def test_supports_native_brackets_delegates(self):
        primary = self._make_router()
        primary.supports_native_brackets.return_value = True
        dispatcher = BrokerDispatcher(primary)
        assert dispatcher.supports_native_brackets() is True

    def test_same_broker_fanout_swaps_account_id(self):
        primary = self._make_router(account_id=1)
        secondary = self._make_router(account_id=2)
        primary.submit.return_value = {"order_id": 100, "status": "submitted"}
        secondary.submit.return_value = {"order_id": 200, "status": "submitted"}

        dispatcher = BrokerDispatcher(primary, [secondary])
        # Spec WITHOUT _intent — same-broker fan-out path
        spec = {"accountId": 1, "action": "Buy", "orderType": "Market"}
        dispatcher.submit(spec)
        # Secondary should get the spec with swapped accountId
        submitted = secondary.submit.call_args[0][0]
        assert submitted["accountId"] == 2


# ---------------------------------------------------------------------------
# BrokerFactory
# ---------------------------------------------------------------------------


class TestBrokerFactory:
    def test_valid_brokers(self):
        from trading_app.live.broker_factory import VALID_BROKERS

        assert "projectx" in VALID_BROKERS
        assert "tradovate" in VALID_BROKERS

    def test_invalid_broker_raises(self):
        from trading_app.live.broker_factory import create_broker_components

        with pytest.raises(ValueError, match="Unknown broker"):
            create_broker_components(broker="fake_broker")

    @patch.dict("os.environ", {"BROKER": "tradovate", "TRADOVATE_DEMO": "1"})
    def test_create_tradovate_components(self):
        # Auth will try to load env vars but won't login until get_token()
        from trading_app.live.broker_factory import create_broker_components

        components = create_broker_components(broker="tradovate")
        assert components["feed_class"] is None  # Tradovate has no feed
        assert components["router_class"].__name__ == "TradovateOrderRouter"
        assert components["contracts_class"].__name__ == "TradovateContracts"
        assert components["positions_class"].__name__ == "TradovatePositions"


# ---------------------------------------------------------------------------
# Bloomey review gap fixes
# ---------------------------------------------------------------------------


class TestTokenRenewal:
    """GAP-004: Token renewal path was untested."""

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    def test_renew_success(self, mock_post):
        """Renewal returns new token without full re-login."""
        from trading_app.live.tradovate.auth import TradovateAuth

        # Initial login
        login_resp = MagicMock()
        login_resp.raise_for_status = MagicMock()
        login_resp.json.return_value = {"accessToken": "old_tok", "userId": 1}

        # Renewal response
        renew_resp = MagicMock()
        renew_resp.raise_for_status = MagicMock()
        renew_resp.json.return_value = {"accessToken": "new_tok"}

        mock_post.side_effect = [login_resp, renew_resp]

        auth = TradovateAuth()
        auth.get_token()  # Initial login
        assert auth._access_token == "old_tok"

        # Force renewal
        auth._acquired_at = 0  # Expire the token
        auth._renew_or_login()
        assert auth._access_token == "new_tok"

    @patch.dict(
        "os.environ",
        {
            "TRADOVATE_USERNAME": "test",
            "TRADOVATE_PASSWORD": "pass",
            "TRADOVATE_CID": "123",
            "TRADOVATE_SEC": "secret",
            "TRADOVATE_DEMO": "1",
        },
    )
    @patch("trading_app.live.tradovate.auth.requests.post")
    def test_renew_fails_falls_back_to_login(self, mock_post):
        """If renewal fails, falls back to full login."""
        import requests as req
        from trading_app.live.tradovate.auth import TradovateAuth

        login_resp = MagicMock()
        login_resp.raise_for_status = MagicMock()
        login_resp.json.return_value = {"accessToken": "tok1", "userId": 1}

        renew_fail = MagicMock()
        renew_fail.raise_for_status.side_effect = req.RequestException("renewal failed")

        relogin_resp = MagicMock()
        relogin_resp.raise_for_status = MagicMock()
        relogin_resp.json.return_value = {"accessToken": "tok2", "userId": 1}

        mock_post.side_effect = [login_resp, renew_fail, relogin_resp]

        auth = TradovateAuth()
        auth.get_token()
        auth._acquired_at = 0
        auth._renew_or_login()
        assert auth._access_token == "tok2"  # Fell back to full login


class TestQueryOpenOrders:
    """GAP-005: query_open_orders was untested."""

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_returns_list(self, mock_req, router):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {"orderId": 1, "symbol": "MNQM6", "orderType": "Stop"},
            {"orderId": 2, "symbol": "MNQM6", "orderType": "Limit"},
        ]
        mock_req.return_value = mock_resp

        orders = router.query_open_orders()
        assert len(orders) == 2
        assert orders[0]["orderId"] == 1

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_returns_empty_on_non_list(self, mock_req, router):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "something"}
        mock_req.return_value = mock_resp

        orders = router.query_open_orders()
        assert orders == []

    def test_returns_empty_no_auth(self):
        from trading_app.live.tradovate.order_router import TradovateOrderRouter

        router = TradovateOrderRouter(account_id=1, auth=None, tick_size=0.25)
        assert router.query_open_orders() == []


class TestBracketPriceCollar:
    """Price collar validates ENTRY stop only. Bracket targets are NOT collared.
    This matches ProjectX behavior and prevents rejecting RR2.5+ bracket targets."""

    def test_bracket_target_far_from_market_passes(self, router):
        """RR2.5 bracket target at ~1% from market should NOT be rejected."""
        router.update_market_price(100.0)
        spec = router.build_order_spec("long", "E2", 100.1, "MNQM6")
        # Target at 102.5 = 2.5% deviation — would fail old bracket collar but should pass now
        bracket = router.build_bracket_spec("long", "MNQM6", 100.1, 99.0, 102.5)
        merged = router.merge_bracket_into_entry(spec, bracket)

        with patch("trading_app.live.tradovate.order_router.request_with_retry") as mock_req:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"orderId": 1}
            mock_req.return_value = mock_resp
            result = router.submit(merged)
            assert result["order_id"] == 1

    def test_entry_stop_still_collared(self, router):
        """Entry stop price far from market should still be rejected."""
        router.update_market_price(100.0)
        spec = router.build_order_spec("long", "E2", 110.0, "MNQM6")  # 10% deviation on entry stop
        with pytest.raises(ValueError, match="PRICE_COLLAR_REJECTED"):
            router.submit(spec)


class TestCancelBracketOrders:
    """CRITICAL-003 follow-up: cancel_bracket_orders works on Tradovate."""

    @patch("trading_app.live.tradovate.order_router.request_with_retry")
    def test_cancels_matching_brackets(self, mock_req, router):
        # query_open_orders returns brackets
        query_resp = MagicMock()
        query_resp.raise_for_status = MagicMock()
        query_resp.json.return_value = [
            {"orderId": 10, "symbol": "MNQM6", "orderType": "Limit"},
            {"orderId": 11, "symbol": "MNQM6", "orderType": "Stop"},
            {"orderId": 12, "symbol": "MGCM6", "orderType": "Stop"},  # Different symbol
        ]

        # cancel responses
        cancel_resp = MagicMock()
        cancel_resp.raise_for_status = MagicMock()

        mock_req.side_effect = [query_resp, cancel_resp, cancel_resp]

        cancelled = router.cancel_bracket_orders("MNQM6")
        assert cancelled == 2  # Only MNQM6 brackets, not MGCM6
