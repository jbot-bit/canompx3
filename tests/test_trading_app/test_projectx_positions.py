"""Test ProjectXPositions -- mocked HTTP.

As of 2026-05-18 the positions module funnels through BrokerHTTPClient. Tests
patch requests.Session.request so the classifier still runs but the network
is mocked.
"""

from unittest.mock import MagicMock, patch


def _make_auth():
    """Return a minimal auth stub with headers() + refresh_if_needed()."""
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    auth.refresh_if_needed = MagicMock()
    return auth


def _ok_resp(body):
    r = MagicMock()
    r.status_code = 200
    r.headers = {}
    r.json.return_value = body
    r.text = str(body)
    r.raise_for_status = MagicMock()
    return r


def test_query_account_metadata_returns_dict_when_found():
    """query_account_metadata returns the full account dict for the matching id."""
    body = [
        {"id": 11111111, "name": "50KXFA-other", "simulated": False},
        {"id": 20372221, "name": "50KTC-V2-451890-20372221", "balance": 47_000.0, "simulated": True},
    ]

    with patch("requests.Session.request", return_value=_ok_resp(body)) as mock_req:
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        meta = pos.query_account_metadata(20372221)

        assert meta is not None
        assert meta["id"] == 20372221
        assert meta["name"] == "50KTC-V2-451890-20372221"
        assert meta["simulated"] is True
        mock_req.assert_called_once()
        called_url = mock_req.call_args.args[1] if len(mock_req.call_args.args) > 1 else mock_req.call_args.kwargs.get("url", "")
        assert "/api/Account/search" in called_url


def test_query_account_metadata_returns_none_when_not_found():
    """Missing account id returns None, not a stale fallback."""
    body = [{"id": 11111111, "name": "50KXFA-other"}]

    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        meta = pos.query_account_metadata(99999999)
        assert meta is None


def test_query_account_metadata_returns_none_on_http_error():
    """HTTP failure must not raise — return None so the caller trusts profile config.

    The new BrokerHTTPClient retries class-A errors; we want the FINAL failure
    to be swallowed by query_account_metadata (existing contract). Patch the
    session to keep raising ConnectionError on every attempt.
    """
    import requests as _requests
    from trading_app.live.projectx.positions import ProjectXPositions

    with patch("requests.Session.request", side_effect=_requests.ConnectionError("broker down")):
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
    body = {
        "accounts": [
            {"id": 21944866, "name": "EXPRESS-V2-451890-53179846", "balance": 0.0, "simulated": True},
        ]
    }

    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        equity = pos.query_equity(21944866)
        assert equity == 0.0, "Day-1 XFA $0 balance must return 0.0, not None"
        assert equity is not None


def test_query_equity_returns_balance_for_funded_account():
    """Funded account returns the float balance — sanity check non-zero path."""
    body = {
        "accounts": [
            {"id": 20859313, "name": "50KTC-V2-451890-20372221", "balance": 44_587.30, "simulated": True},
        ]
    }

    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(20859313) == 44_587.30


def test_query_equity_falls_back_to_cashBalance_when_balance_missing():
    """If `balance` field missing, fall through to `cashBalance`. The explicit
    None-check (not falsy `or`) preserves this fallback for accounts that
    only expose `cashBalance`."""
    body = {
        "accounts": [
            {"id": 99999999, "name": "alt-shape", "cashBalance": 12_345.67},
        ]
    }

    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(99999999) == 12_345.67


def test_query_equity_returns_none_when_account_id_not_in_response():
    """Genuine 'account not found' (id missing from broker response) → None.
    The misleading 'not found' log path is preserved for actual misses."""
    body = {"accounts": [{"id": 11111111, "balance": 100.0}]}

    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        assert pos.query_equity(99999999) is None


# ── Stage 3 readiness — query_equity_with_age returns EquityReading ──────────


def test_query_equity_with_age_returns_live_reading():
    body = {"accounts": [{"id": 20859313, "balance": 44_587.30}]}
    with patch("requests.Session.request", return_value=_ok_resp(body)):
        from trading_app.live.http_client import EquityReading
        from trading_app.live.projectx.positions import ProjectXPositions

        pos = ProjectXPositions(_make_auth())
        reading = pos.query_equity_with_age(20859313)
        assert isinstance(reading, EquityReading)
        assert reading.source == "live"
        assert reading.value == 44_587.30
        assert reading.age_s == 0.0


def test_query_equity_with_age_serves_cache_on_transient_failure():
    """First call succeeds → cached. Second call (transient failure) returns cache + age."""
    import time as _time
    import requests as _requests
    from trading_app.live.projectx.positions import ProjectXPositions

    ok_body = {"accounts": [{"id": 20859313, "balance": 44_587.30}]}
    ok = _ok_resp(ok_body)

    pos = ProjectXPositions(_make_auth())

    # First call — live read populates cache.
    with patch("requests.Session.request", return_value=ok):
        first = pos.query_equity_with_age(20859313)
        assert first.source == "live"

    _time.sleep(0.01)

    # Second call — transient connection errors exhaust retries; cache served.
    err = _requests.ConnectionError("rst")
    err.__cause__ = ConnectionResetError(10054, "rst")
    with patch("requests.Session.request", side_effect=err):
        cached = pos.query_equity_with_age(20859313)
    assert cached.source == "cache"
    assert cached.value == 44_587.30
    assert cached.age_s > 0
