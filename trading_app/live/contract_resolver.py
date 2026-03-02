"""
Resolves instrument shortname to current Tradovate front-month contract symbol.

Calls Tradovate /contract/find, returns the nearest-expiry contract name
(e.g. 'MGCM6', 'MNQM6'). Contracts roll quarterly — call this once per session
start rather than caching across sessions.
"""
import requests

from .tradovate_auth import TradovateAuth

DEMO_BASE = "https://demo.tradovate.com/v1"
LIVE_BASE = "https://live.tradovate.com/v1"

# Tradovate product base names (same as our instrument codes for all active instruments)
PRODUCT_MAP = {
    "MGC": "MGC",
    "MNQ": "MNQ",
    "MES": "MES",
    "M2K": "M2K",
}


def resolve_front_month(instrument: str, auth: TradovateAuth, demo: bool = True) -> str:
    """
    Return the current front-month contract symbol for an instrument.

    Example: 'MGC' → 'MGCM6' (June 2026)
    """
    if instrument not in PRODUCT_MAP:
        raise ValueError(f"Unknown instrument '{instrument}'. Valid: {list(PRODUCT_MAP)}")

    base = DEMO_BASE if demo else LIVE_BASE
    resp = requests.get(
        f"{base}/contract/find",
        params={"name": PRODUCT_MAP[instrument]},
        headers=auth.headers(),
        timeout=5,
    )
    resp.raise_for_status()
    contracts = resp.json()

    if not contracts:
        raise RuntimeError(f"No contracts found for {instrument} ({PRODUCT_MAP[instrument]})")

    # Take the nearest expiry
    front = sorted(contracts, key=lambda c: c.get("expirationDate", ""))[0]
    return front["name"]
