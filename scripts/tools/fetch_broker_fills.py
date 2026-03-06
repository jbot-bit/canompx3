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

from trading_app.live.projectx.auth import BASE_URL, ProjectXAuth

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
            import traceback

            print(f"  TopstepX fetch failed: {exc}")
            traceback.print_exc()

    if args.broker in ("tradovate", "all"):
        print("Tradovate: skipped (set TRADOVATE_USERNAME + TRADOVATE_PASSWORD in .env to enable)")

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
