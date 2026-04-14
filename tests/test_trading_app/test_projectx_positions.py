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
