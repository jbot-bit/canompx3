"""Test ProjectXPositions -- mocked HTTP."""

from unittest.mock import MagicMock, patch


def _make_auth():
    """Return a minimal auth stub with headers()."""
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    return auth


def test_query_account_metadata_returns_dict_when_found():
    """query_account_metadata returns the full account dict for the matching id."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"id": 11111111, "name": "50KXFA-other", "simulated": False},
        {"id": 20372221, "name": "50KTC-V2-451890-20372221", "balance": 47_000.0, "simulated": True},
    ]
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        meta = pos.query_account_metadata(20372221)

        assert meta is not None
        assert meta["id"] == 20372221
        assert meta["name"] == "50KTC-V2-451890-20372221"
        assert meta["simulated"] is True
        mock_post.assert_called_once()
        assert "/api/Account/search" in mock_post.call_args[0][0]


def test_query_account_metadata_returns_none_when_not_found():
    """Missing account id returns None, not a stale fallback."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"id": 11111111, "name": "50KXFA-other"},
    ]
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        meta = pos.query_account_metadata(99999999)
        assert meta is None


def test_query_account_metadata_returns_none_on_http_error():
    """HTTP failure must not raise — return None so the caller trusts profile config."""
    from trading_app.live.projectx.positions import ProjectXPositions

    with patch("requests.post", side_effect=ConnectionError("broker down")):
        pos = ProjectXPositions(_make_auth())
        meta = pos.query_account_metadata(20372221)
        assert meta is None


# ── query_equity falsy-or fix (2026-04-25) ────────────────────────────────────
# Pre-fix: `acct.get("balance") or acct.get("cashBalance")` collapsed 0.0 → None
# so day-1 XFA accounts (which legitimately start at $0 per topstep_mll_article.md:64)
# silently failed and broke F-1 EOD seeding via _apply_broker_reality_check.
# These tests pin the explicit `is None` behavior — mutation: revert to `or` and
# test_query_equity_returns_zero_for_zero_balance fails.


def test_query_equity_returns_zero_for_zero_balance():
    """$0.00 balance must return 0.0, NOT None. Day-1 XFA accounts start at
    $0 per topstep_mll_article.md:64 — failing this masks F-1 EOD seeding.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "accounts": [
            {"id": 21944866, "name": "EXPRESS-V2-451890-53179846", "balance": 0.0, "simulated": True},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        equity = pos.query_equity(21944866)
        assert equity == 0.0, "Day-1 XFA $0 balance must return 0.0, not None"
        assert equity is not None


def test_query_equity_returns_balance_for_funded_account():
    """Funded account returns the float balance — sanity check non-zero path."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "accounts": [
            {"id": 20859313, "name": "50KTC-V2-451890-20372221", "balance": 44_587.30, "simulated": True},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(20859313) == 44_587.30


def test_query_equity_falls_back_to_cashBalance_when_balance_missing():
    """If `balance` field missing, fall through to `cashBalance`. The explicit
    None-check (not falsy `or`) preserves this fallback for accounts that
    only expose `cashBalance`."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "accounts": [
            {"id": 99999999, "name": "alt-shape", "cashBalance": 12_345.67},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(99999999) == 12_345.67


def test_query_equity_returns_none_when_account_id_not_in_response():
    """Genuine 'account not found' (id missing from broker response) → None.
    The misleading 'not found' log path is preserved for actual misses."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"accounts": [{"id": 11111111, "balance": 100.0}]}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(99999999) is None
