"""Tests for ProjectX account-id resolution — singular fail-closed guard.

Capital review A (2026-06-06) defense-in-depth: the SINGULAR resolve_account_id()
must NOT silently take accounts[0] when the broker exposes >1 active account, or
LIVE orders could route to a different account than the C11 survival proof used.
"""

from unittest.mock import MagicMock, patch

import pytest

from trading_app.live.projectx.contract_resolver import ProjectXContracts


def _make_contracts() -> ProjectXContracts:
    """Build a ProjectXContracts with a stubbed HTTP client (no network)."""
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    auth.failure_hook = None
    with patch("trading_app.live.projectx.contract_resolver.BrokerHTTPClient"):
        contracts = ProjectXContracts(auth=auth)
    contracts._http = MagicMock()
    return contracts


class TestProjectXResolveAccountId:
    def test_single_account_returns_it(self):
        contracts = _make_contracts()
        contracts._http.post_json.return_value = [{"id": 555, "name": "XFA-50K"}]
        assert contracts.resolve_account_id() == 555

    def test_single_account_via_accountId_field(self):
        contracts = _make_contracts()
        contracts._http.post_json.return_value = [{"accountId": 777, "name": "XFA"}]
        assert contracts.resolve_account_id() == 777

    def test_multiple_active_accounts_raises_ambiguous(self):
        contracts = _make_contracts()
        contracts._http.post_json.return_value = [
            {"id": 100, "name": "XFA-50K"},
            {"id": 200, "name": "EVAL-COMBINE"},
        ]
        with pytest.raises(RuntimeError, match="ambiguous"):
            contracts.resolve_account_id()

    def test_multiple_accounts_error_names_them(self):
        contracts = _make_contracts()
        contracts._http.post_json.return_value = [
            {"id": 100, "name": "XFA-50K"},
            {"id": 200, "name": "EVAL-COMBINE"},
        ]
        with pytest.raises(RuntimeError) as exc:
            contracts.resolve_account_id()
        msg = str(exc.value)
        assert "XFA-50K" in msg and "EVAL-COMBINE" in msg
        assert "--account-id" in msg

    def test_no_accounts_raises(self):
        contracts = _make_contracts()
        contracts._http.post_json.return_value = []
        with pytest.raises(RuntimeError, match="No active ProjectX accounts"):
            contracts.resolve_account_id()
