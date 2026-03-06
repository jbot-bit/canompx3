"""
Resolves instrument shortname to current Tradovate front-month contract symbol.

Calls Tradovate /contract/find, returns the nearest-expiry contract name
(e.g. 'MGCM6', 'MNQM6'). Contracts roll quarterly -- call this once per session
start rather than caching across sessions.
"""

import logging
from datetime import date

import requests

from ..broker_base import BrokerAuth, BrokerContracts

DEMO_BASE = "https://demo.tradovateapi.com/v1"
LIVE_BASE = "https://live.tradovateapi.com/v1"

log = logging.getLogger(__name__)

# Tradovate product base names (same as our instrument codes for all active instruments)
PRODUCT_MAP = {
    "MGC": "MGC",
    "MNQ": "MNQ",
    "MES": "MES",
    "M2K": "M2K",
}


def resolve_account_id(auth: BrokerAuth, demo: bool = True) -> int:
    """
    Return the numeric account ID for the authenticated user.

    For prop firm accounts (Apex, TopstepX) there is typically one account.
    If multiple accounts exist, returns the first active one.
    """
    base = DEMO_BASE if demo else LIVE_BASE
    resp = requests.get(
        f"{base}/account/list",
        headers=auth.headers(),
        timeout=5,
    )
    resp.raise_for_status()
    accounts = resp.json()
    if not accounts:
        raise RuntimeError("No Tradovate accounts found for authenticated user")
    # Prefer the first account -- prop firm setups typically have exactly one
    acct = accounts[0]
    acct_id = acct["id"]
    acct_name = acct.get("name", "unknown")
    log.info(
        "Auto-resolved account: %s (id=%d) -- %s mode",
        acct_name,
        acct_id,
        "DEMO" if demo else "LIVE",
    )
    return acct_id


def resolve_front_month(instrument: str, auth: BrokerAuth, demo: bool = True) -> str:
    """
    Return the current front-month contract symbol for an instrument.

    Example: 'MGC' -> 'MGCM6' (June 2026)
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

    # Filter out expired contracts (expirationDate < today)
    today = date.today().isoformat()
    active = [c for c in contracts if c.get("expirationDate", "") >= today]
    if not active:
        raise RuntimeError(f"All contracts for {instrument} are expired")

    # Take the nearest future expiry (front month)
    front = sorted(active, key=lambda c: c.get("expirationDate", ""))[0]
    return front["name"]


class TradovateContracts(BrokerContracts):
    """Tradovate contract resolution via REST API."""

    def __init__(self, auth: BrokerAuth, demo: bool = True):
        super().__init__(auth)
        self.demo = demo

    def resolve_account_id(self) -> int:
        """Return the numeric account ID for the authenticated user."""
        return resolve_account_id(self.auth, self.demo)

    def resolve_front_month(self, instrument: str) -> str:
        """Return current front-month contract symbol for an instrument."""
        return resolve_front_month(instrument, self.auth, self.demo)
