---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# AI Trading Coach — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI trading coach that fetches broker fills, reconstructs round-trip trades, generates session digests via Claude API, maintains an evolving trader profile, and integrates with existing discipline coach and Pinecone knowledge base.

**Architecture:** Broker fill fetcher → trade matcher → session digest engine (Claude API) → interactive chat. Data flows through JSONL files (`data/broker_fills.jsonl`, `data/broker_trades.jsonl`, `data/coaching_digests.jsonl`) with a persistent trader profile (`data/trader_profile.json`). Reuses existing `ProjectXAuth` for TopstepX, `TradovateAuth` for Tradovate.

**Tech Stack:** Python 3.12, requests, anthropic SDK, existing ProjectXAuth/TradovateAuth, Pinecone Assistant API, JSONL storage, pytest

**Design Doc:** `docs/plans/2026-03-06-ai-coach-design.md`

---

### Task 0: Data directory + trader profile bootstrap

**Files:**
- Create: `data/trader_profile.json`
- Create: `data/coach_state.json`

**Step 1: Create initial trader profile**

Create `data/trader_profile.json`:
```json
{
  "version": 1,
  "last_updated": "2026-03-06",
  "strengths": [],
  "growth_edges": [],
  "behavioral_patterns": [],
  "goals": [],
  "session_tendencies": {},
  "emotional_profile": {
    "tilt_indicators": [],
    "calm_indicators": []
  },
  "account_summary": {}
}
```

**Step 2: Create initial coach state**

Create `data/coach_state.json`:
```json
{
  "last_fetch": null,
  "accounts": {}
}
```

**Step 3: Commit**

```bash
git add data/trader_profile.json data/coach_state.json
git commit -m "feat(coach): bootstrap trader profile and coach state"
```

---

### Task 1: Broker fill fetcher — TopstepX

**Files:**
- Create: `scripts/tools/fetch_broker_fills.py`
- Create: `tests/test_coaching/test_fetch_broker_fills.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/__init__.py` (empty) and `tests/test_coaching/test_fetch_broker_fills.py`:

```python
"""Tests for broker fill fetcher."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools.fetch_broker_fills import (
    fetch_topstepx_accounts,
    fetch_topstepx_fills,
    normalize_topstepx_fill,
    save_fills,
    load_coach_state,
    save_coach_state,
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_fetch_broker_fills.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `scripts/tools/fetch_broker_fills.py`:

```python
#!/usr/bin/env python3
"""Fetch broker fills from TopstepX and Tradovate APIs.

Normalizes to unified fill schema, appends to data/broker_fills.jsonl.
Tracks incremental fetch state via data/coach_state.json.

Usage:
    python scripts/tools/fetch_broker_fills.py                # fetch all brokers
    python scripts/tools/fetch_broker_fills.py --broker topstepx  # TopstepX only
    python scripts/tools/fetch_broker_fills.py --broker tradovate  # Tradovate only
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
load_dotenv()

from trading_app.live.projectx.auth import ProjectXAuth, BASE_URL

log = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
FILLS_PATH = DATA_DIR / "broker_fills.jsonl"
COACH_STATE_PATH = DATA_DIR / "coach_state.json"


# ---------------------------------------------------------------------------
# Coach state persistence
# ---------------------------------------------------------------------------

def load_coach_state(*, path: Path = COACH_STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"last_fetch": None, "accounts": {}}


def save_coach_state(state: dict, *, path: Path = COACH_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def save_fills(fills: list[dict], *, path: Path = FILLS_PATH) -> int:
    """Append fills to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for fill in fills:
            fh.write(json.dumps(fill) + "\n")
    return len(fills)


# ---------------------------------------------------------------------------
# TopstepX
# ---------------------------------------------------------------------------

def fetch_topstepx_accounts(*, headers: dict) -> list[dict]:
    """Fetch active trading accounts from TopstepX."""
    resp = requests.post(
        f"{BASE_URL}/api/Account/search",
        json={"onlyActiveAccounts": True},
        headers={**headers, "Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("accounts", data) if isinstance(data, dict) else data


def fetch_topstepx_fills(account_id: int, *, headers: dict, since_id: int | None = None) -> list[dict]:
    """Fetch trade fills for a single TopstepX account."""
    resp = requests.post(
        f"{BASE_URL}/api/Trade/search",
        json={"accountId": account_id},
        headers={**headers, "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    trades = data.get("trades", data) if isinstance(data, dict) else data
    if since_id is not None:
        trades = [t for t in trades if t.get("id", 0) > since_id]
    return trades


def _extract_instrument(contract_id: str) -> str:
    """Extract instrument symbol from ProjectX contractId. CON.F.US.MNQ.H26 -> MNQ"""
    match = re.search(r"\.([A-Z0-9]+)\.[A-Z]\d{2}$", contract_id)
    return match.group(1) if match else contract_id


def normalize_topstepx_fill(raw: dict, *, account_name: str) -> dict:
    """Normalize a TopstepX fill to the unified schema."""
    return {
        "fill_id": f"topstepx-{raw['id']}",
        "broker": "topstepx",
        "account_id": raw["accountId"],
        "account_name": account_name,
        "instrument": _extract_instrument(raw.get("contractId", "")),
        "contract_id": raw.get("contractId", ""),
        "timestamp": raw.get("timestamp", ""),
        "side": raw.get("action", "").upper(),
        "size": raw.get("size", 0),
        "price": raw.get("price", 0.0),
        "pnl": raw.get("profitAndLoss") or 0.0,
        "fees": raw.get("commission") or 0.0,
        "order_id": raw.get("orderId"),
    }


# ---------------------------------------------------------------------------
# Fetch orchestration
# ---------------------------------------------------------------------------

def fetch_all_topstepx(state: dict) -> list[dict]:
    """Fetch fills from all TopstepX accounts. Updates state in-place."""
    auth = ProjectXAuth()
    hdrs = auth.headers()
    accounts = fetch_topstepx_accounts(headers=hdrs)
    all_fills = []

    for acct in accounts:
        acct_id = acct["id"]
        acct_name = acct.get("name", str(acct_id))
        state_key = f"topstepx-{acct_id}"
        acct_state = state["accounts"].get(state_key, {})
        since_id = acct_state.get("last_fill_id")

        raw_fills = fetch_topstepx_fills(acct_id, headers=hdrs, since_id=since_id)
        if not raw_fills:
            print(f"  {acct_name}: no new fills")
            continue

        normalized = [normalize_topstepx_fill(f, account_name=acct_name) for f in raw_fills]
        all_fills.extend(normalized)

        max_fill = max(raw_fills, key=lambda t: t.get("id", 0))
        state["accounts"][state_key] = {
            "last_fill_id": max_fill["id"],
            "last_fill_ts": max_fill.get("timestamp", ""),
        }
        print(f"  {acct_name}: {len(normalized)} new fills")

    return all_fills


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch broker fills")
    parser.add_argument("--broker", choices=["topstepx", "tradovate", "all"], default="all")
    args = parser.parse_args()

    state = load_coach_state()
    all_fills = []

    if args.broker in ("topstepx", "all"):
        print("Fetching TopstepX fills...")
        try:
            fills = fetch_all_topstepx(state)
            all_fills.extend(fills)
        except Exception as exc:
            print(f"  TopstepX fetch failed: {exc}")

    if args.broker in ("tradovate", "all"):
        print("Tradovate: skipping (auth not configured)")

    if all_fills:
        n = save_fills(all_fills)
        print(f"\nSaved {n} fills to {FILLS_PATH}")
    else:
        print("\nNo new fills to save.")

    state["last_fetch"] = datetime.now(UTC).isoformat()
    save_coach_state(state)
    print(f"Coach state updated: {COACH_STATE_PATH}")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_fetch_broker_fills.py -v`
Expected: 7 tests PASS

**Step 5: Lint and commit**

```bash
ruff format scripts/tools/fetch_broker_fills.py tests/test_coaching/
ruff check scripts/tools/fetch_broker_fills.py tests/test_coaching/
git add scripts/tools/fetch_broker_fills.py tests/test_coaching/ data/trader_profile.json data/coach_state.json
git commit -m "feat(coach): P1 broker fill fetcher — TopstepX + JSONL storage"
```

---

### Task 2: Trade matcher — fills to round-trip trades

**Files:**
- Create: `scripts/tools/trade_matcher.py`
- Create: `tests/test_coaching/test_trade_matcher.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_trade_matcher.py`:

```python
"""Tests for trade matcher — fills -> round-trip trades."""

import json
from pathlib import Path

import pytest

from scripts.tools.trade_matcher import (
    match_fills_to_trades,
    classify_trade_type,
    load_fills,
)


def _make_fill(fill_id, account_id, instrument, timestamp, side, size, price, pnl=0, fees=0):
    return {
        "fill_id": fill_id,
        "broker": "topstepx",
        "account_id": account_id,
        "account_name": "test-account",
        "instrument": instrument,
        "timestamp": timestamp,
        "side": side,
        "size": size,
        "price": price,
        "pnl": pnl,
        "fees": fees,
    }


class TestClassifyTradeType:
    def test_scalp_under_5min(self):
        assert classify_trade_type(120) == "scalp"

    def test_swing_5_to_60min(self):
        assert classify_trade_type(600) == "swing"

    def test_position_over_60min(self):
        assert classify_trade_type(7200) == "position"


class TestSimpleRoundTrip:
    """BUY 4, then SELL 4 -> one LONG round-trip trade."""

    def test_simple_long(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 4, 24800.0, 0, 1.48),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:02:00Z", "SELL", 4, 24810.0, 40.0, 1.48),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t["direction"] == "LONG"
        assert t["instrument"] == "MNQ"
        assert t["entry_price_avg"] == 24800.0
        assert t["exit_price_avg"] == 24810.0
        assert t["size"] == 4
        assert t["hold_seconds"] == 120
        assert t["trade_type"] == "scalp"
        assert t["num_fills"] == 2

    def test_simple_short(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "SELL", 2, 24800.0, 0, 1.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:01:00Z", "BUY", 2, 24790.0, 20.0, 1.0),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        assert trades[0]["direction"] == "SHORT"


class TestMultiFillRoundTrip:
    """Scale-in: BUY 2 + BUY 2, then SELL 4 -> one trade with VWAP entry."""

    def test_scale_in_vwap(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:00:30Z", "BUY", 2, 24805.0),
            _make_fill("f3", 123, "MNQ", "2026-03-06T13:02:00Z", "SELL", 4, 24810.0, 30.0, 2.96),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 1
        t = trades[0]
        assert t["entry_price_avg"] == pytest.approx(24802.5)  # VWAP: (2*24800 + 2*24805) / 4
        assert t["size"] == 4
        assert t["num_fills"] == 3


class TestPositionFlip:
    """BUY 2, SELL 4 -> close LONG (2), open SHORT (2). Two trades."""

    def test_flip_generates_two_trades(self):
        fills = [
            _make_fill("f1", 123, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 123, "MNQ", "2026-03-06T13:01:00Z", "SELL", 4, 24810.0),
            _make_fill("f3", 123, "MNQ", "2026-03-06T13:03:00Z", "BUY", 2, 24805.0),
        ]
        trades = match_fills_to_trades(fills)
        assert len(trades) == 2
        assert trades[0]["direction"] == "LONG"
        assert trades[1]["direction"] == "SHORT"


class TestMultiAccountIsolation:
    """Fills from different accounts don't cross-match."""

    def test_accounts_isolated(self):
        fills = [
            _make_fill("f1", 111, "MNQ", "2026-03-06T13:00:00Z", "BUY", 2, 24800.0),
            _make_fill("f2", 222, "MNQ", "2026-03-06T13:00:00Z", "SELL", 2, 24810.0),
        ]
        trades = match_fills_to_trades(fills)
        # No round-trips: each account has an open position, neither is closed
        assert len(trades) == 0


class TestLoadFills:
    def test_load_fills_from_jsonl(self, tmp_path):
        path = tmp_path / "fills.jsonl"
        path.write_text(
            json.dumps({"fill_id": "a", "timestamp": "2026-03-06T13:00:00Z"}) + "\n"
            + json.dumps({"fill_id": "b", "timestamp": "2026-03-06T13:01:00Z"}) + "\n"
        )
        fills = load_fills(path=path)
        assert len(fills) == 2
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_trade_matcher.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `scripts/tools/trade_matcher.py`:

```python
#!/usr/bin/env python3
"""Match raw broker fills into round-trip trades.

Reads data/broker_fills.jsonl, outputs data/broker_trades.jsonl.
Position tracking per (account_id, instrument). VWAP entry/exit pricing.

Usage:
    python scripts/tools/trade_matcher.py                    # match all fills
    python scripts/tools/trade_matcher.py --date 2026-03-06  # specific date
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
FILLS_PATH = DATA_DIR / "broker_fills.jsonl"
TRADES_PATH = DATA_DIR / "broker_trades.jsonl"


def load_fills(*, path: Path = FILLS_PATH) -> list[dict]:
    """Load fills from JSONL."""
    if not path.exists():
        return []
    fills = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            fills.append(json.loads(line))
    return fills


def classify_trade_type(hold_seconds: float) -> str:
    if hold_seconds < 300:
        return "scalp"
    elif hold_seconds < 3600:
        return "swing"
    return "position"


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def match_fills_to_trades(fills: list[dict]) -> list[dict]:
    """Convert a list of fills into round-trip trade records.

    Algorithm:
    1. Group fills by (account_id, instrument)
    2. Sort by timestamp within each group
    3. Track running position; emit trade when position returns to zero or flips
    """
    # Group by (account_id, instrument)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for fill in fills:
        key = (fill["account_id"], fill["instrument"])
        groups[key].append(fill)

    all_trades = []

    for (account_id, instrument), group_fills in groups.items():
        group_fills.sort(key=lambda f: f["timestamp"])

        position = 0  # positive = long, negative = short
        entry_fills: list[dict] = []
        trade_counter = 0

        for fill in group_fills:
            side = fill["side"].upper()
            size = fill["size"]
            signed = size if side == "BUY" else -size

            new_position = position + signed

            # Same direction as existing position — accumulate
            if position == 0 or (position > 0 and signed > 0) or (position < 0 and signed < 0):
                entry_fills.append(fill)
                position = new_position
                continue

            # Opposite direction — partial or full close, or flip
            if abs(signed) <= abs(position):
                # Partial or full close
                position = new_position
                if position == 0:
                    # Full close — emit trade
                    trade_counter += 1
                    trade = _build_trade(
                        entry_fills, fill, account_id, instrument, trade_counter
                    )
                    all_trades.append(trade)
                    entry_fills = []
                # If partial close, don't emit yet
            else:
                # Position flip — close current, open new
                trade_counter += 1
                trade = _build_trade(
                    entry_fills, fill, account_id, instrument, trade_counter
                )
                all_trades.append(trade)

                # Remaining size opens new position
                entry_fills = [fill]  # The flip fill starts new position
                position = new_position

    return all_trades


def _build_trade(
    entry_fills: list[dict],
    exit_fill: dict,
    account_id: int,
    instrument: str,
    counter: int,
) -> dict:
    """Build a round-trip trade record from entry fills + exit fill."""
    # VWAP entry price
    total_entry_value = sum(f["price"] * f["size"] for f in entry_fills)
    total_entry_size = sum(f["size"] for f in entry_fills)
    entry_price_avg = total_entry_value / total_entry_size if total_entry_size else 0

    entry_time = entry_fills[0]["timestamp"]
    exit_time = exit_fill["timestamp"]

    entry_dt = _parse_ts(entry_time)
    exit_dt = _parse_ts(exit_time)
    hold_seconds = (exit_dt - entry_dt).total_seconds()

    direction = "LONG" if entry_fills[0]["side"].upper() == "BUY" else "SHORT"

    total_fees = sum(f.get("fees", 0) for f in entry_fills) + exit_fill.get("fees", 0)
    total_pnl = sum(f.get("pnl", 0) for f in entry_fills) + exit_fill.get("pnl", 0)

    date_str = entry_dt.strftime("%Y-%m-%d")
    trade_id = f"{entry_fills[0]['broker']}-{account_id}-{date_str}-{counter:03d}"

    return {
        "trade_id": trade_id,
        "broker": entry_fills[0]["broker"],
        "account_name": entry_fills[0].get("account_name", str(account_id)),
        "instrument": instrument,
        "direction": direction,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price_avg": round(entry_price_avg, 6),
        "exit_price_avg": round(exit_fill["price"], 6),
        "size": total_entry_size,
        "pnl_dollar": total_pnl,
        "fees": total_fees,
        "hold_seconds": hold_seconds,
        "num_fills": len(entry_fills) + 1,
        "trade_type": classify_trade_type(hold_seconds),
        "source": "manual",
        "strategy_id": None,
    }


def save_trades(trades: list[dict], *, path: Path = TRADES_PATH) -> int:
    """Append trades to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for trade in trades:
            fh.write(json.dumps(trade) + "\n")
    return len(trades)


def main():
    parser = argparse.ArgumentParser(description="Match fills to round-trip trades")
    parser.add_argument("--date", help="Filter fills by date (YYYY-MM-DD)")
    args = parser.parse_args()

    fills = load_fills()
    if args.date:
        fills = [f for f in fills if f.get("timestamp", "").startswith(args.date)]

    print(f"Loaded {len(fills)} fills")
    trades = match_fills_to_trades(fills)
    print(f"Matched {len(trades)} round-trip trades")

    if trades:
        n = save_trades(trades)
        print(f"Saved {n} trades to {TRADES_PATH}")

        for t in trades:
            emoji = "+" if t["pnl_dollar"] >= 0 else ""
            print(f"  {t['trade_id']}: {t['direction']} {t['instrument']} "
                  f"x{t['size']} {emoji}${t['pnl_dollar']:.0f} ({t['trade_type']}, "
                  f"{t['hold_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_trade_matcher.py -v`
Expected: 8 tests PASS

**Step 5: Lint and commit**

```bash
ruff format scripts/tools/trade_matcher.py tests/test_coaching/test_trade_matcher.py
ruff check scripts/tools/trade_matcher.py tests/test_coaching/test_trade_matcher.py
git add scripts/tools/trade_matcher.py tests/test_coaching/test_trade_matcher.py
git commit -m "feat(coach): P2 trade matcher — fills to round-trip trades"
```

---

### Task 3: Source detection — system vs manual trades

**Files:**
- Modify: `scripts/tools/trade_matcher.py`
- Create: `tests/test_coaching/test_source_detection.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_source_detection.py`:

```python
"""Tests for source detection — matching broker trades to system signals."""

import json
from pathlib import Path

import pytest

from scripts.tools.trade_matcher import detect_source


def _make_trade(entry_time, instrument):
    return {"entry_time": entry_time, "instrument": instrument, "source": "manual", "strategy_id": None}


def _make_signal(ts, instrument, strategy_id, signal_type="SIGNAL_ENTRY"):
    return {"ts": ts, "instrument": instrument, "strategy_id": strategy_id, "type": signal_type}


class TestSourceDetection:
    def test_matches_within_60s(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:00:30+00:00", "MNQ", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "system"
        assert trade["strategy_id"] == "strat-001"

    def test_no_match_beyond_60s(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:05:00+00:00", "MNQ", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "manual"
        assert trade["strategy_id"] is None

    def test_no_match_wrong_instrument(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        signals = [_make_signal("2026-03-06T13:00:10+00:00", "MGC", "strat-001")]
        detect_source(trade, signals)
        assert trade["source"] == "manual"

    def test_empty_signals(self):
        trade = _make_trade("2026-03-06T13:00:00+00:00", "MNQ")
        detect_source(trade, [])
        assert trade["source"] == "manual"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_source_detection.py -v`
Expected: FAIL with `ImportError`

**Step 3: Add detect_source to trade_matcher.py**

Add to `scripts/tools/trade_matcher.py`:

```python
def detect_source(trade: dict, signals: list[dict], *, tolerance_s: float = 60.0) -> None:
    """Match a trade to a system signal. Mutates trade in-place."""
    if not signals:
        return
    trade_ts = _parse_ts(trade["entry_time"])
    for sig in signals:
        if sig.get("type") not in ("SIGNAL_ENTRY", "ORDER_ENTRY"):
            continue
        if sig.get("instrument") != trade["instrument"]:
            continue
        sig_ts = _parse_ts(sig["ts"])
        if abs((trade_ts - sig_ts).total_seconds()) <= tolerance_s:
            trade["source"] = "system"
            trade["strategy_id"] = sig.get("strategy_id")
            return
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_source_detection.py -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
ruff format scripts/tools/trade_matcher.py tests/test_coaching/test_source_detection.py
git add scripts/tools/trade_matcher.py tests/test_coaching/test_source_detection.py
git commit -m "feat(coach): P2 source detection — system vs manual trade classification"
```

---

### Task 4: Session digest engine — Claude API integration

**Files:**
- Create: `scripts/tools/coaching_digest.py`
- Create: `tests/test_coaching/test_coaching_digest.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_coaching_digest.py`:

```python
"""Tests for coaching digest engine."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools.coaching_digest import (
    build_digest_prompt,
    parse_digest_response,
    apply_profile_patch,
    load_trader_profile,
    save_trader_profile,
)


class TestBuildDigestPrompt:
    def test_includes_profile_and_trades(self):
        profile = {"version": 1, "strengths": [], "growth_edges": []}
        trades = [{"trade_id": "t1", "pnl_dollar": 100}]
        prompt = build_digest_prompt(profile, trades)
        assert "version" in prompt
        assert "t1" in prompt
        assert "pnl_dollar" in prompt


class TestParseDigestResponse:
    def test_parses_valid_json(self):
        raw = json.dumps({
            "digest": {
                "summary": "Good session",
                "trade_grades": [{"trade_id": "t1", "grade": "A", "reason": "Clean entry"}],
                "patterns_observed": ["patience"],
                "coaching_note": "Keep it up.",
                "metrics": {"trades": 1, "win_rate": 1.0, "gross_pnl": 100, "fees": 2, "net_pnl": 98},
            },
            "profile_patch": {
                "strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}],
            },
        })
        digest, patch = parse_digest_response(raw)
        assert digest["summary"] == "Good session"
        assert len(patch["strengths"]) == 1

    def test_handles_markdown_fenced_json(self):
        raw = "```json\n" + json.dumps({
            "digest": {"summary": "x", "trade_grades": [], "patterns_observed": [],
                       "coaching_note": "y", "metrics": {}},
            "profile_patch": {},
        }) + "\n```"
        digest, patch = parse_digest_response(raw)
        assert digest["summary"] == "x"


class TestApplyProfilePatch:
    def test_adds_new_strength(self):
        profile = {"version": 1, "strengths": [], "growth_edges": []}
        patch = {"strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}]}
        apply_profile_patch(profile, patch)
        assert len(profile["strengths"]) == 1
        assert profile["version"] == 2

    def test_updates_existing_strength(self):
        profile = {
            "version": 3,
            "strengths": [{"trait": "patience", "confidence": 0.5, "evidence_count": 2}],
            "growth_edges": [],
        }
        patch = {"strengths": [{"trait": "patience", "confidence": 0.7, "evidence_count": 3}]}
        apply_profile_patch(profile, patch)
        assert profile["strengths"][0]["confidence"] == 0.7
        assert profile["version"] == 4

    def test_does_not_increment_version_on_empty_patch(self):
        profile = {"version": 5, "strengths": [], "growth_edges": []}
        apply_profile_patch(profile, {})
        assert profile["version"] == 5


class TestProfilePersistence:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = {"version": 1, "strengths": [{"trait": "x"}], "growth_edges": []}
        save_trader_profile(profile, path=path)
        loaded = load_trader_profile(path=path)
        assert loaded["strengths"][0]["trait"] == "x"
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_coaching_digest.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `scripts/tools/coaching_digest.py`:

```python
#!/usr/bin/env python3
"""AI coaching digest engine — generates session analysis via Claude API.

Reads broker_trades.jsonl, trader_profile.json, and daily_features.
Generates coaching digest + profile update patch.

Usage:
    python scripts/tools/coaching_digest.py                    # today's session
    python scripts/tools/coaching_digest.py --date 2026-03-06  # specific date
"""

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime, date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "trader_profile.json"
TRADES_PATH = DATA_DIR / "broker_trades.jsonl"
DIGESTS_PATH = DATA_DIR / "coaching_digests.jsonl"
TRADING_RULES_PATH = PROJECT_ROOT / "TRADING_RULES.md"

SYSTEM_PROMPT = """You are a professional trading coach at a proprietary trading firm. You analyze trade data with the precision of a quant and the empathy of a mentor.

Your job is to:
1. Grade each trade (A/B/C/D/F) based on execution quality, timing, and discipline
2. Identify behavioral patterns (both positive and negative)
3. Write a coaching note that is honest, specific, and actionable
4. Generate a profile patch that updates the trader's evolving model

Be direct. Use evidence from the trades. Never praise without substance. Never criticize without a path forward.

RESPOND WITH VALID JSON ONLY — no markdown fencing, no commentary outside the JSON."""

DIGEST_SCHEMA = """{
  "digest": {
    "summary": "1-2 sentence session summary",
    "trade_grades": [{"trade_id": "...", "grade": "A|B|C|D|F", "reason": "..."}],
    "patterns_observed": ["pattern_name_1", "pattern_name_2"],
    "coaching_note": "2-3 paragraphs of coaching feedback",
    "metrics": {"trades": N, "win_rate": 0.XX, "gross_pnl": X, "fees": X, "net_pnl": X}
  },
  "profile_patch": {
    "strengths": [{"trait": "...", "confidence": 0.0-1.0, "evidence_count": N}],
    "growth_edges": [{"trait": "...", "confidence": 0.0-1.0, "evidence_count": N}],
    "behavioral_patterns": [{"pattern": "...", "trigger": "...", "frequency": "...", "avg_cost": X}]
  }
}"""


# ---------------------------------------------------------------------------
# Profile I/O
# ---------------------------------------------------------------------------

def load_trader_profile(*, path: Path = PROFILE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": 1, "last_updated": "", "strengths": [],
        "growth_edges": [], "behavioral_patterns": [], "goals": [],
        "session_tendencies": {}, "emotional_profile": {"tilt_indicators": [], "calm_indicators": []},
        "account_summary": {},
    }


def save_trader_profile(profile: dict, *, path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile["last_updated"] = date.today().isoformat()
    path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_digest_prompt(profile: dict, trades: list[dict], trading_rules_excerpt: str = "") -> str:
    """Build the user prompt for the Claude API call."""
    parts = []
    if trading_rules_excerpt:
        parts.append(f"## Trading Rules\n{trading_rules_excerpt[:2000]}")
    parts.append(f"## Current Trader Profile\n```json\n{json.dumps(profile, indent=2)}\n```")
    parts.append(f"## Today's Trades\n```json\n{json.dumps(trades, indent=2)}\n```")
    parts.append(f"## Required Output Schema\n```json\n{DIGEST_SCHEMA}\n```")
    parts.append("Generate the digest and profile patch now. JSON only.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_digest_response(raw: str) -> tuple[dict, dict]:
    """Parse Claude's response into (digest, profile_patch)."""
    # Strip markdown fencing if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    data = json.loads(cleaned)
    return data["digest"], data.get("profile_patch", {})


# ---------------------------------------------------------------------------
# Profile patching
# ---------------------------------------------------------------------------

def apply_profile_patch(profile: dict, patch: dict) -> None:
    """Merge a profile patch into the existing profile. Mutates in-place."""
    if not patch:
        return

    changed = False
    for list_field in ("strengths", "growth_edges", "behavioral_patterns"):
        if list_field not in patch:
            continue
        existing = profile.setdefault(list_field, [])
        for new_item in patch[list_field]:
            # Find existing by trait/pattern name
            key_field = "trait" if list_field != "behavioral_patterns" else "pattern"
            match = next((e for e in existing if e.get(key_field) == new_item.get(key_field)), None)
            if match:
                match.update(new_item)
            else:
                existing.append(new_item)
            changed = True

    if changed:
        profile["version"] = profile.get("version", 0) + 1


# ---------------------------------------------------------------------------
# Digest generation
# ---------------------------------------------------------------------------

def generate_digest(trades: list[dict], *, profile_path: Path = PROFILE_PATH) -> dict | None:
    """Generate a coaching digest via Claude API. Returns the digest dict or None on failure."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return None

    profile = load_trader_profile(path=profile_path)
    rules_excerpt = ""
    if TRADING_RULES_PATH.exists():
        rules_excerpt = TRADING_RULES_PATH.read_text(encoding="utf-8")[:2000]

    user_prompt = build_digest_prompt(profile, trades, rules_excerpt)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw_text = response.content[0].text
    digest, patch = parse_digest_response(raw_text)

    # Apply profile patch
    version_before = profile.get("version", 1)
    apply_profile_patch(profile, patch)
    save_trader_profile(profile, path=profile_path)

    # Enrich digest with metadata
    digest["date"] = trades[0]["entry_time"][:10] if trades else date.today().isoformat()
    digest["accounts"] = list({t.get("account_name", "") for t in trades})
    digest["profile_version_before"] = version_before
    digest["profile_version_after"] = profile.get("version", version_before)

    return digest


def save_digest(digest: dict, *, path: Path = DIGESTS_PATH) -> None:
    """Append digest to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(digest) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate coaching digest")
    parser.add_argument("--date", help="Date to analyze (YYYY-MM-DD), default today")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    # Load trades for the target date
    if not TRADES_PATH.exists():
        print(f"No trades file at {TRADES_PATH}. Run trade_matcher.py first.")
        return

    all_trades = []
    for line in TRADES_PATH.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            all_trades.append(json.loads(line))

    day_trades = [t for t in all_trades if t.get("entry_time", "").startswith(target_date)]
    if not day_trades:
        print(f"No trades found for {target_date}")
        return

    print(f"Generating digest for {target_date} ({len(day_trades)} trades)...")
    digest = generate_digest(day_trades)
    if digest:
        save_digest(digest)
        print(f"\nDigest saved to {DIGESTS_PATH}")
        print(f"Coaching note:\n{digest.get('coaching_note', 'N/A')}")
    else:
        print("Digest generation failed.")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_coaching_digest.py -v`
Expected: 7 tests PASS

**Step 5: Commit**

```bash
ruff format scripts/tools/coaching_digest.py tests/test_coaching/test_coaching_digest.py
ruff check scripts/tools/coaching_digest.py tests/test_coaching/test_coaching_digest.py
git add scripts/tools/coaching_digest.py tests/test_coaching/test_coaching_digest.py
git commit -m "feat(coach): P3 session digest engine — Claude API + profile patching"
```

---

### Task 5: Interactive chat — trading coach CLI

**Files:**
- Create: `scripts/tools/trading_coach.py`
- Create: `tests/test_coaching/test_trading_coach.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_trading_coach.py`:

```python
"""Tests for interactive trading coach."""

import json
from pathlib import Path

import pytest

from scripts.tools.trading_coach import build_chat_system_prompt, load_recent_digests


class TestBuildChatSystemPrompt:
    def test_includes_profile(self):
        profile = {"version": 1, "strengths": [{"trait": "patience"}]}
        prompt = build_chat_system_prompt(profile, [])
        assert "patience" in prompt
        assert "personal coach" in prompt.lower() or "trading coach" in prompt.lower()

    def test_includes_recent_digests(self):
        profile = {"version": 1}
        digests = [{"date": "2026-03-06", "coaching_note": "Great discipline today"}]
        prompt = build_chat_system_prompt(profile, digests)
        assert "Great discipline today" in prompt


class TestLoadRecentDigests:
    def test_loads_last_n(self, tmp_path):
        path = tmp_path / "digests.jsonl"
        lines = [json.dumps({"date": f"2026-03-0{i}", "summary": f"day {i}"}) for i in range(1, 8)]
        path.write_text("\n".join(lines) + "\n")
        recent = load_recent_digests(n=3, path=path)
        assert len(recent) == 3
        assert recent[-1]["date"] == "2026-03-07"

    def test_returns_empty_if_missing(self, tmp_path):
        recent = load_recent_digests(n=5, path=tmp_path / "nonexistent.jsonl")
        assert recent == []
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_trading_coach.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `scripts/tools/trading_coach.py`:

```python
#!/usr/bin/env python3
"""Interactive AI trading coach — conversational CLI.

Loads trader profile + recent coaching digests into system prompt.
Uses Claude API for conversation. Supports Pinecone RAG for historical queries.

Usage:
    python scripts/tools/trading_coach.py          # start chat
    python scripts/tools/trading_coach.py --query "Why do I keep losing on Fridays?"
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "trader_profile.json"
DIGESTS_PATH = DATA_DIR / "coaching_digests.jsonl"
TRADING_RULES_PATH = PROJECT_ROOT / "TRADING_RULES.md"


def load_recent_digests(n: int = 5, *, path: Path = DIGESTS_PATH) -> list[dict]:
    """Load the last N coaching digests."""
    if not path.exists():
        return []
    digests = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            digests.append(json.loads(line))
    return digests[-n:]


def build_chat_system_prompt(profile: dict, recent_digests: list[dict]) -> str:
    """Build the system prompt for interactive chat."""
    parts = [
        "You are this trader's personal trading coach. You know their patterns, "
        "strengths, and growth edges from your ongoing analysis of their trades.",
        "",
        "## Trader Profile",
        f"```json\n{json.dumps(profile, indent=2)}\n```",
    ]

    if recent_digests:
        parts.append("\n## Recent Coaching Digests")
        for d in recent_digests:
            parts.append(f"\n### {d.get('date', 'Unknown date')}")
            if d.get("coaching_note"):
                parts.append(d["coaching_note"])
            if d.get("metrics"):
                parts.append(f"Metrics: {json.dumps(d['metrics'])}")

    parts.extend([
        "",
        "## Your Role",
        "- Answer questions using evidence from the profile and trading history",
        "- Be honest and direct — the trader needs truth, not comfort",
        "- Reference specific trades, patterns, and metrics when relevant",
        "- If asked about something not in the data, say so clearly",
    ])

    return "\n".join(parts)


def chat_loop():
    """Run interactive chat with the trading coach."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return

    from scripts.tools.coaching_digest import load_trader_profile

    profile = load_trader_profile()
    digests = load_recent_digests()
    system_prompt = build_chat_system_prompt(profile, digests)

    client = anthropic.Anthropic()
    messages = []

    print("Trading Coach (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        print(f"\nCoach: {assistant_text}")


def single_query(query: str):
    """Ask a single question and print the response."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return

    from scripts.tools.coaching_digest import load_trader_profile

    profile = load_trader_profile()
    digests = load_recent_digests()
    system_prompt = build_chat_system_prompt(profile, digests)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    print(response.content[0].text)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Coach")
    parser.add_argument("--query", "-q", help="Single question (no interactive mode)")
    args = parser.parse_args()

    if args.query:
        single_query(args.query)
    else:
        chat_loop()


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_trading_coach.py -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
ruff format scripts/tools/trading_coach.py tests/test_coaching/test_trading_coach.py
ruff check scripts/tools/trading_coach.py tests/test_coaching/test_trading_coach.py
git add scripts/tools/trading_coach.py tests/test_coaching/test_trading_coach.py
git commit -m "feat(coach): P4 interactive chat — CLI trading coach"
```

---

### Task 6: Discipline coach integration

**Files:**
- Modify: `ui/discipline_data.py` (add profile loading for priming)
- Modify: `ui/discipline.py` (show AI coaching note in pre-session priming)
- Create: `tests/test_coaching/test_discipline_integration.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_discipline_integration.py`:

```python
"""Tests for discipline coach integration with AI coach."""

import json
from pathlib import Path

import pytest

from ui.discipline_data import load_coaching_note


class TestLoadCoachingNote:
    def test_returns_latest_note(self, tmp_path):
        digests_path = tmp_path / "digests.jsonl"
        digests_path.write_text(
            json.dumps({"date": "2026-03-05", "coaching_note": "Old note"}) + "\n"
            + json.dumps({"date": "2026-03-06", "coaching_note": "Latest note"}) + "\n"
        )
        note = load_coaching_note(digests_path=digests_path)
        assert note == "Latest note"

    def test_returns_none_if_no_digests(self, tmp_path):
        note = load_coaching_note(digests_path=tmp_path / "nonexistent.jsonl")
        assert note is None

    def test_returns_none_if_no_coaching_note_field(self, tmp_path):
        digests_path = tmp_path / "digests.jsonl"
        digests_path.write_text(json.dumps({"date": "2026-03-06", "summary": "no note"}) + "\n")
        note = load_coaching_note(digests_path=digests_path)
        assert note is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_discipline_integration.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_coaching_note'`

**Step 3: Add load_coaching_note to discipline_data.py**

Add at the bottom of `ui/discipline_data.py`:

```python
# -- AI Coach integration ---------------------------------------------------

COACHING_DIGESTS_PATH = _DATA_DIR / "coaching_digests.jsonl"


def load_coaching_note(*, digests_path: Path = COACHING_DIGESTS_PATH) -> str | None:
    """Load the latest coaching note for pre-session priming. Returns None if unavailable."""
    records = _load_jsonl(digests_path, label="coaching_digest")
    if not records:
        return None
    # Get last record with a coaching_note
    for record in reversed(records):
        note = record.get("coaching_note")
        if note:
            return note
    return None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_discipline_integration.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
ruff format ui/discipline_data.py tests/test_coaching/test_discipline_integration.py
git add ui/discipline_data.py tests/test_coaching/test_discipline_integration.py
git commit -m "feat(coach): P5 discipline coach integration — coaching note for priming"
```

---

### Task 7: Pinecone sync — coaching tier

**Files:**
- Modify: `scripts/tools/pinecone_manifest.json` (add coaching tier)
- Modify: `scripts/tools/sync_pinecone.py` (collect coaching files)
- Create: `tests/test_coaching/test_pinecone_coaching.py`

**Step 1: Write the failing tests**

Create `tests/test_coaching/test_pinecone_coaching.py`:

```python
"""Tests for Pinecone coaching tier sync."""

import json
from pathlib import Path

import pytest


class TestManifestHasCoachingTier:
    def test_coaching_tier_exists_in_manifest(self):
        manifest_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "tools" / "pinecone_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tiers = manifest["content_tiers"]
        assert "coaching" in tiers, "coaching tier missing from pinecone_manifest.json"
        assert "files" in tiers["coaching"]


class TestCoachingTierCollection:
    def test_collects_profile_and_digests(self, tmp_path):
        # Create mock coaching files
        profile = tmp_path / "trader_profile.json"
        profile.write_text(json.dumps({"version": 1}))
        digests = tmp_path / "coaching_digests.jsonl"
        digests.write_text(json.dumps({"date": "2026-03-06"}) + "\n")

        from scripts.tools.sync_pinecone import collect_coaching_files

        files = collect_coaching_files(data_dir=tmp_path)
        paths = [str(p) for p, _ in files]
        assert any("trader_profile" in p for p in paths)
        assert any("coaching_digests" in p for p in paths)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_coaching/test_pinecone_coaching.py -v`
Expected: FAIL (manifest assertion fails, ImportError for collect_coaching_files)

**Step 3: Add coaching tier to manifest**

Add to `scripts/tools/pinecone_manifest.json` in `content_tiers`:

```json
"coaching": {
  "description": "AI coaching digests and trader profile — synced for chat RAG",
  "files": ["data/trader_profile.json", "data/coaching_digests.jsonl"]
}
```

**Step 4: Add collect_coaching_files to sync_pinecone.py**

Add a new function and integrate it into `collect_all_files()`:

```python
def collect_coaching_files(*, data_dir: Path = PROJECT_ROOT / "data") -> list[tuple[Path, str]]:
    """Collect coaching files for Pinecone sync."""
    files = []
    for filename in ("trader_profile.json", "coaching_digests.jsonl"):
        path = data_dir / filename
        if path.exists():
            files.append((path, f"coaching/{filename}"))
    return files
```

In `collect_all_files()`, after the generated tier collection, add:

```python
    # --- Coaching files ---
    if "coaching" in tiers:
        collected["coaching"] = collect_coaching_files()
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_coaching/test_pinecone_coaching.py -v`
Expected: 2 tests PASS

**Step 6: Commit**

```bash
ruff format scripts/tools/sync_pinecone.py tests/test_coaching/test_pinecone_coaching.py
git add scripts/tools/pinecone_manifest.json scripts/tools/sync_pinecone.py tests/test_coaching/test_pinecone_coaching.py
git commit -m "feat(coach): P6 Pinecone sync — coaching tier for digests + profile"
```

---

### Task 8: End-to-end smoke test

**Files:**
- Create: `tests/test_coaching/test_e2e_coaching.py`

**Step 1: Write end-to-end test (mocked Claude API)**

Create `tests/test_coaching/test_e2e_coaching.py`:

```python
"""End-to-end smoke test: fills -> trades -> digest (mocked Claude)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.tools.fetch_broker_fills import save_fills
from scripts.tools.trade_matcher import match_fills_to_trades, save_trades
from scripts.tools.coaching_digest import (
    generate_digest,
    load_trader_profile,
    save_trader_profile,
    DIGESTS_PATH,
)


@pytest.fixture
def coaching_data_dir(tmp_path):
    """Set up a temporary coaching data directory."""
    profile = {
        "version": 1, "last_updated": "", "strengths": [],
        "growth_edges": [], "behavioral_patterns": [], "goals": [],
        "session_tendencies": {},
        "emotional_profile": {"tilt_indicators": [], "calm_indicators": []},
        "account_summary": {},
    }
    profile_path = tmp_path / "trader_profile.json"
    save_trader_profile(profile, path=profile_path)
    return tmp_path, profile_path


def test_full_pipeline_mocked(coaching_data_dir):
    """Fills -> trades -> digest with mocked Claude API."""
    tmp_path, profile_path = coaching_data_dir

    # 1. Create fills
    fills = [
        {
            "fill_id": "topstepx-1", "broker": "topstepx", "account_id": 123,
            "account_name": "test-acct", "instrument": "MNQ",
            "timestamp": "2026-03-06T13:00:00+00:00", "side": "BUY",
            "size": 2, "price": 24800.0, "pnl": 0, "fees": 1.0,
        },
        {
            "fill_id": "topstepx-2", "broker": "topstepx", "account_id": 123,
            "account_name": "test-acct", "instrument": "MNQ",
            "timestamp": "2026-03-06T13:02:00+00:00", "side": "SELL",
            "size": 2, "price": 24810.0, "pnl": 20.0, "fees": 1.0,
        },
    ]
    fills_path = tmp_path / "fills.jsonl"
    save_fills(fills, path=fills_path)

    # 2. Match trades
    trades = match_fills_to_trades(fills)
    assert len(trades) == 1
    assert trades[0]["direction"] == "LONG"
    assert trades[0]["pnl_dollar"] == 20.0

    # 3. Generate digest (mock Claude API)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps({
        "digest": {
            "summary": "1 MNQ trade, net +$20",
            "trade_grades": [{"trade_id": trades[0]["trade_id"], "grade": "B", "reason": "Clean entry"}],
            "patterns_observed": ["patience"],
            "coaching_note": "Good discipline on this trade.",
            "metrics": {"trades": 1, "win_rate": 1.0, "gross_pnl": 20, "fees": 2, "net_pnl": 18},
        },
        "profile_patch": {
            "strengths": [{"trait": "patience", "confidence": 0.6, "evidence_count": 1}],
        },
    }))]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("scripts.tools.coaching_digest.anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = mock_client
        digest = generate_digest(trades, profile_path=profile_path)

    assert digest is not None
    assert digest["summary"] == "1 MNQ trade, net +$20"

    # 4. Verify profile was updated
    updated_profile = load_trader_profile(path=profile_path)
    assert updated_profile["version"] == 2
    assert len(updated_profile["strengths"]) == 1
    assert updated_profile["strengths"][0]["trait"] == "patience"
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_coaching/test_e2e_coaching.py -v`
Expected: PASS

**Step 3: Run full coaching test suite**

Run: `python -m pytest tests/test_coaching/ -v`
Expected: All tests PASS (~23 tests)

**Step 4: Commit**

```bash
ruff format tests/test_coaching/test_e2e_coaching.py
git add tests/test_coaching/test_e2e_coaching.py
git commit -m "test(coach): end-to-end smoke test — fills to digest with mocked Claude"
```

---

### Task 9: Final integration — ruff, full test suite, REPO_MAP

**Files:**
- Modify: `REPO_MAP.md` (auto-regen)

**Step 1: Lint all coaching code**

```bash
ruff format scripts/tools/fetch_broker_fills.py scripts/tools/trade_matcher.py scripts/tools/coaching_digest.py scripts/tools/trading_coach.py tests/test_coaching/
ruff check scripts/tools/fetch_broker_fills.py scripts/tools/trade_matcher.py scripts/tools/coaching_digest.py scripts/tools/trading_coach.py tests/test_coaching/
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All tests PASS (including existing pipeline tests + new coaching tests)

**Step 3: Run drift checks**

```bash
python pipeline/check_drift.py
```

Expected: All checks PASS

**Step 4: Regenerate REPO_MAP**

```bash
python scripts/tools/gen_repo_map.py
```

**Step 5: Commit**

```bash
git add REPO_MAP.md
git commit -m "docs: update REPO_MAP after AI coaching feature"
```

---

## Dependency Chain

```
Task 0: Data bootstrap (profile + state)
  └── Task 1: Broker fill fetcher (TopstepX)
      └── Task 2: Trade matcher (fills -> round-trips)
          ├── Task 3: Source detection (system vs manual)
          └── Task 4: Session digest engine (Claude API)
              ├── Task 5: Interactive chat
              ├── Task 6: Discipline coach integration
              └── Task 7: Pinecone sync (coaching tier)
                  └── Task 8: E2E smoke test
                      └── Task 9: Final integration
```

Tasks 5, 6, 7 can be parallelized after Task 4.
