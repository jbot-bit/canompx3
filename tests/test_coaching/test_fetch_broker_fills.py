"""Tests for broker fill fetcher."""

import json
from unittest.mock import MagicMock, patch

from scripts.tools.fetch_broker_fills import (
    fetch_topstepx_accounts,
    load_coach_state,
    normalize_topstepx_fill,
    save_coach_state,
    save_fills,
)

# -- TopstepX account fetch --------------------------------------------------

MOCK_ACCOUNTS_RESPONSE = {
    "accounts": [
        {"id": 19858923, "name": "50KTC-V2-451890-20967121", "status": "Active"},
        {"id": 17189309, "name": "100KTC-V1-382610-18623456", "status": "Active"},
    ],
    "success": True,
}


def test_fetch_topstepx_accounts():
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_ACCOUNTS_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    with patch("scripts.tools.fetch_broker_fills.requests.post", return_value=mock_resp):
        accounts = fetch_topstepx_accounts(headers={"Authorization": "Bearer fake"})

    assert len(accounts) == 2
    assert accounts[0]["id"] == 19858923


# -- TopstepX fill normalization ----------------------------------------------

# Real API format: side=int (1=BUY, 2=SELL), creationTimestamp, fees (not commission)
MOCK_FILL_REAL = {
    "id": 2241049355,
    "accountId": 19858923,
    "contractId": "CON.F.US.MNQ.H26",
    "creationTimestamp": "2026-03-06T13:43:48.140526+00:00",
    "side": 1,
    "size": 4,
    "price": 24740.75,
    "profitAndLoss": 556.0,
    "fees": 1.48,
    "orderId": 2587762443,
}

# Legacy format (action string) for backwards compat
MOCK_FILL_LEGACY = {
    "id": 2241049355,
    "accountId": 19858923,
    "contractId": "CON.F.US.MNQ.H26",
    "timestamp": "2026-03-06T13:43:48.140526+00:00",
    "action": "Buy",
    "size": 4,
    "price": 24740.75,
    "profitAndLoss": 556.0,
    "commission": 1.48,
    "orderId": 2587762443,
}


def test_normalize_topstepx_fill_real_api():
    """Real API: side=1 (BUY), creationTimestamp, fees."""
    fill = normalize_topstepx_fill(MOCK_FILL_REAL, account_name="50KTC-V2-451890-20967121")
    assert fill["fill_id"] == "topstepx-2241049355"
    assert fill["broker"] == "topstepx"
    assert fill["instrument"] == "MNQ"
    assert fill["side"] == "BUY"
    assert fill["size"] == 4
    assert fill["price"] == 24740.75
    assert fill["pnl"] == 556.0
    assert fill["fees"] == 1.48
    assert "2026-03-06" in fill["timestamp"]


def test_normalize_topstepx_fill_sell_side():
    fill = normalize_topstepx_fill({**MOCK_FILL_REAL, "side": 2}, account_name="test")
    assert fill["side"] == "SELL"


def test_normalize_topstepx_fill_legacy_format():
    """Backwards compat: action string, timestamp, commission."""
    fill = normalize_topstepx_fill(MOCK_FILL_LEGACY, account_name="test")
    assert fill["side"] == "BUY"
    assert fill["fees"] == 1.48
    assert "2026-03-06" in fill["timestamp"]


def test_normalize_topstepx_fill_null_pnl():
    fill_data = {**MOCK_FILL_REAL, "profitAndLoss": None}
    fill = normalize_topstepx_fill(fill_data, account_name="test")
    assert fill["pnl"] == 0.0


def test_normalize_topstepx_fill_extracts_instrument_from_contract_id():
    """CON.F.US.MGC.J26 -> MGC"""
    fill_data = {**MOCK_FILL_REAL, "contractId": "CON.F.US.MGC.J26"}
    fill = normalize_topstepx_fill(fill_data, account_name="test")
    assert fill["instrument"] == "MGC"


# -- JSONL save/load ---------------------------------------------------------


def test_save_fills_appends_jsonl(tmp_path):
    out = tmp_path / "fills.jsonl"
    fills = [
        {"fill_id": "topstepx-1", "broker": "topstepx", "price": 100.0},
        {"fill_id": "topstepx-2", "broker": "topstepx", "price": 200.0},
    ]
    save_fills(fills, path=out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["fill_id"] == "topstepx-1"


def test_save_fills_appends_not_overwrites(tmp_path):
    out = tmp_path / "fills.jsonl"
    save_fills([{"fill_id": "a"}], path=out)
    save_fills([{"fill_id": "b"}], path=out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2


def test_save_fills_dedup_prevents_duplicates(tmp_path):
    out = tmp_path / "fills.jsonl"
    fills = [{"fill_id": "f1", "price": 100}]
    assert save_fills(fills, path=out) == 1
    # Re-save same fill_id — should skip
    assert save_fills(fills, path=out) == 0
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 1


def test_save_fills_dedup_allows_new(tmp_path):
    out = tmp_path / "fills.jsonl"
    save_fills([{"fill_id": "f1"}], path=out)
    save_fills([{"fill_id": "f1"}, {"fill_id": "f2"}], path=out)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2  # f1 deduped, f2 appended


# -- Coach state persistence --------------------------------------------------


def test_coach_state_roundtrip(tmp_path):
    state_path = tmp_path / "coach_state.json"
    state = {"last_fetch": "2026-03-06T14:00:00Z", "accounts": {"topstepx-123": {"last_fill_id": 999}}}
    save_coach_state(state, path=state_path)
    loaded = load_coach_state(path=state_path)
    assert loaded["accounts"]["topstepx-123"]["last_fill_id"] == 999


def test_coach_state_returns_empty_if_missing(tmp_path):
    state = load_coach_state(path=tmp_path / "nonexistent.json")
    assert state["accounts"] == {}
