"""Tests for broker fill fetcher."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools.fetch_broker_fills import (
    fetch_topstepx_accounts,
    fetch_topstepx_fills,
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

MOCK_FILL = {
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


def test_normalize_topstepx_fill():
    fill = normalize_topstepx_fill(MOCK_FILL, account_name="50KTC-V2-451890-20967121")
    assert fill["fill_id"] == "topstepx-2241049355"
    assert fill["broker"] == "topstepx"
    assert fill["instrument"] == "MNQ"
    assert fill["side"] == "BUY"
    assert fill["size"] == 4
    assert fill["price"] == 24740.75
    assert fill["pnl"] == 556.0
    assert fill["fees"] == 1.48


def test_normalize_topstepx_fill_null_pnl():
    fill_data = {**MOCK_FILL, "profitAndLoss": None}
    fill = normalize_topstepx_fill(fill_data, account_name="test")
    assert fill["pnl"] == 0.0


def test_normalize_topstepx_fill_extracts_instrument_from_contract_id():
    """CON.F.US.MGC.J26 -> MGC"""
    fill_data = {**MOCK_FILL, "contractId": "CON.F.US.MGC.J26"}
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
