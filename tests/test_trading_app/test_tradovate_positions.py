"""Tests for TradovatePositions.query_open()."""

from unittest.mock import MagicMock, patch

import pytest


class TestTradovatePositions:
    def _make_positions(self, demo=True):
        from trading_app.live.tradovate.positions import TradovatePositions

        auth = MagicMock()
        auth.headers.return_value = {"Authorization": "Bearer fake"}
        return TradovatePositions(auth=auth, demo=demo)

    def test_query_open_returns_standard_format(self):
        """Response normalized to [{contract_id, side, size, avg_price}]."""
        pos = self._make_positions()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"contractId": 123, "netPos": 2, "netPrice": 2350.0, "accountId": 100},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("trading_app.live.tradovate.positions.requests.get", return_value=mock_resp):
            result = pos.query_open(100)

        assert len(result) == 1
        assert result[0]["contract_id"] == 123
        assert result[0]["side"] == "long"
        assert result[0]["size"] == 2
        assert result[0]["avg_price"] == 2350.0

    def test_query_open_filters_flat_positions(self):
        """netPos=0 positions excluded."""
        pos = self._make_positions()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"contractId": 123, "netPos": 0, "netPrice": 0.0, "accountId": 100},
            {"contractId": 456, "netPos": -1, "netPrice": 2360.0, "accountId": 100},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("trading_app.live.tradovate.positions.requests.get", return_value=mock_resp):
            result = pos.query_open(100)

        assert len(result) == 1
        assert result[0]["contract_id"] == 456
        assert result[0]["side"] == "short"

    def test_query_open_filters_by_account(self):
        """Only positions for the given account_id returned."""
        pos = self._make_positions()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"contractId": 123, "netPos": 1, "netPrice": 2350.0, "accountId": 100},
            {"contractId": 456, "netPos": 1, "netPrice": 2360.0, "accountId": 999},
        ]
        mock_resp.raise_for_status = MagicMock()
        with patch("trading_app.live.tradovate.positions.requests.get", return_value=mock_resp):
            result = pos.query_open(100)

        assert len(result) == 1
        assert result[0]["contract_id"] == 123

    def test_query_open_handles_empty_response(self):
        """Empty list when no positions."""
        pos = self._make_positions()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        with patch("trading_app.live.tradovate.positions.requests.get", return_value=mock_resp):
            result = pos.query_open(100)

        assert result == []

    def test_query_open_uses_correct_url(self):
        """Demo mode uses demo URL, live uses live URL."""
        for demo, expected_base in [(True, "demo"), (False, "live")]:
            pos = self._make_positions(demo=demo)
            mock_resp = MagicMock()
            mock_resp.json.return_value = []
            mock_resp.raise_for_status = MagicMock()
            with patch("trading_app.live.tradovate.positions.requests.get", return_value=mock_resp) as mock_get:
                pos.query_open(100)
                url = mock_get.call_args[0][0]
                assert expected_base in url
