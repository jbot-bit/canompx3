# Multi-Broker Live Trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the live trading stack from Tradovate-only to multi-broker (ProjectX + Tradovate), with ProjectX as the primary broker for TopstepX prop firm trading.

**Architecture:** 5 ABCs (BrokerAuth, BrokerFeed, BrokerRouter, BrokerContracts, BrokerPositions) + factory pattern. Existing Tradovate code moves into `tradovate/` submodule. New `projectx/` submodule implements ProjectX Gateway API. SessionOrchestrator becomes broker-agnostic.

**Tech Stack:** Python 3.12, `requests`, `websockets` (Tradovate), `pysignalr` (ProjectX SignalR), DuckDB, existing ExecutionEngine/BarAggregator.

---

## Background

### Design Doc
Full design at `docs/plans/2026-03-06-multi-broker-design.md`. M2.5 reviewed and approved with amendments.

### Why ProjectX First
Tradovate API requires a personal live funded account ($1,000+) + $25/mo subscription. Apex prop firm accounts CANNOT get API access. TopstepX (ProjectX) explicitly supports prop firm API access at $14.50/mo.

### User's ProjectX Account
Account code: `50KTC-V2-451890-20967121`. API key to be generated from ProjectX dashboard (Settings > API tab > ProjectX Linking).

### Key API Endpoints (ProjectX Gateway)
- **Base:** `https://api.thefuturesdesk.projectx.com`
- **Auth:** POST `/api/Auth/loginKey` `{userName, apiKey}` -> JWT (24h)
- **Validate:** POST `/api/Auth/validate` -> new JWT
- **Accounts:** POST `/api/Account/search` `{onlyActiveAccounts: true}`
- **Contracts:** POST `/api/Contract/available` `{live: false}`
- **Order:** POST `/api/Order/place` `{accountId, contractId, type, side, size, stopPrice?, brackets?}`
- **Positions:** POST `/api/Position/searchOpen` `{accountId}`
- **Market Hub:** `wss://rtc.thefuturesdesk.projectx.com/hubs/market` (SignalR)

---

## Task 0: Broker Abstraction Layer (ABCs + Factory)

**Files:**
- Create: `trading_app/live/broker_base.py`
- Create: `trading_app/live/broker_factory.py`
- Test: `tests/test_trading_app/test_broker_base.py`

**Step 1: Write the failing test**

```python
# tests/test_trading_app/test_broker_base.py
"""Test broker abstraction layer."""
import pytest
from trading_app.live.broker_base import (
    BrokerAuth,
    BrokerContracts,
    BrokerFeed,
    BrokerPositions,
    BrokerRouter,
)
from trading_app.live.broker_factory import create_broker_components


def test_abc_cannot_instantiate():
    """ABCs should not be directly instantiable."""
    with pytest.raises(TypeError):
        BrokerAuth()
    with pytest.raises(TypeError):
        BrokerFeed(auth=None, on_bar=None)
    with pytest.raises(TypeError):
        BrokerRouter(account_id=0, auth=None)
    with pytest.raises(TypeError):
        BrokerContracts(auth=None)
    with pytest.raises(TypeError):
        BrokerPositions(auth=None)


def test_factory_unknown_broker():
    with pytest.raises(ValueError, match="Unknown broker"):
        create_broker_components("nonexistent")


def test_factory_returns_correct_types():
    """Factory should return objects implementing all 5 ABCs."""
    # We test with tradovate since it's the existing implementation
    # ProjectX will be added in Task 2
    components = create_broker_components("tradovate", demo=True)
    assert "auth" in components
    assert "contracts" in components
    assert isinstance(components["auth"], BrokerAuth)
    assert isinstance(components["contracts"], BrokerContracts)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_broker_base.py -v`
Expected: FAIL — `broker_base` module does not exist yet.

**Step 3: Create broker_base.py**

```python
# trading_app/live/broker_base.py
"""Abstract base classes for multi-broker support.

All broker implementations (Tradovate, ProjectX) must implement these ABCs.
SessionOrchestrator depends ONLY on these interfaces, never on concrete classes.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from .bar_aggregator import Bar


class BrokerAuth(ABC):
    """Authenticate with a broker and manage session tokens."""

    @abstractmethod
    def get_token(self) -> str:
        """Return a valid session/access token, refreshing if needed."""
        ...

    @abstractmethod
    def headers(self) -> dict:
        """Return Authorization header dict for REST calls."""
        ...

    @abstractmethod
    def refresh_if_needed(self) -> None:
        """Proactively refresh token if near expiry."""
        ...


class BrokerFeed(ABC):
    """Stream real-time market data and produce 1-minute bars."""

    def __init__(
        self,
        auth: BrokerAuth,
        on_bar: Callable[[Bar], Coroutine[Any, Any, None]],
        **kwargs,
    ):
        self.auth = auth
        self.on_bar = on_bar

    @abstractmethod
    async def run(self, symbol: str) -> None:
        """Connect and stream bars. Reconnects on disconnect."""
        ...

    @abstractmethod
    def flush(self, symbol: str = "") -> Bar | None:
        """Force-close current bar at session end."""
        ...


class BrokerRouter(ABC):
    """Route orders to the broker."""

    def __init__(self, account_id: int, auth: BrokerAuth | None, **kwargs):
        self.account_id = account_id
        self.auth = auth

    @abstractmethod
    def build_order_spec(
        self,
        direction: str,
        entry_model: str,
        entry_price: float,
        symbol: str,
        qty: int = 1,
    ) -> dict:
        """Build a broker-specific order payload."""
        ...

    @abstractmethod
    def submit(self, spec: dict) -> dict:
        """Submit order, return {order_id, status}."""
        ...

    @abstractmethod
    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        """Build a market close order."""
        ...

    @abstractmethod
    def cancel(self, order_id: int) -> None:
        """Cancel an open order."""
        ...

    @abstractmethod
    def supports_native_brackets(self) -> bool:
        """Whether broker supports stop/target brackets on entry order."""
        ...


class BrokerContracts(ABC):
    """Resolve accounts and contract symbols."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        self.auth = auth

    @abstractmethod
    def resolve_account_id(self) -> int:
        """Return the numeric account ID for trading."""
        ...

    @abstractmethod
    def resolve_front_month(self, instrument: str) -> str:
        """Return current front-month contract symbol for an instrument."""
        ...


class BrokerPositions(ABC):
    """Query broker for open positions (crash recovery + EOD reconciliation)."""

    def __init__(self, auth: BrokerAuth, **kwargs):
        self.auth = auth

    @abstractmethod
    def query_open(self, account_id: int) -> list[dict]:
        """Return open positions: [{contract_id, side, size, avg_price}]."""
        ...
```

**Step 4: Create broker_factory.py**

```python
# trading_app/live/broker_factory.py
"""Factory for creating broker components from a broker name."""

import os

from .broker_base import BrokerAuth, BrokerContracts, BrokerFeed, BrokerPositions, BrokerRouter


def get_broker_name() -> str:
    """Read broker name from env, default to 'projectx'."""
    return os.environ.get("BROKER", "projectx").lower()


def create_broker_components(
    broker: str | None = None,
    demo: bool = True,
) -> dict:
    """Create all broker components for the given broker.

    Returns dict with keys: auth, feed_class, router_class, contracts_class, positions_class.
    Feed/router/contracts/positions are CLASSES (not instances) — caller instantiates with args.
    Auth is an INSTANCE (needs to be shared across components).
    """
    if broker is None:
        broker = get_broker_name()

    if broker == "projectx":
        from .projectx.auth import ProjectXAuth
        from .projectx.contract_resolver import ProjectXContracts
        from .projectx.data_feed import ProjectXDataFeed
        from .projectx.order_router import ProjectXOrderRouter
        from .projectx.positions import ProjectXPositions

        auth = ProjectXAuth()
        return {
            "auth": auth,
            "feed_class": ProjectXDataFeed,
            "router_class": ProjectXOrderRouter,
            "contracts_class": ProjectXContracts,
            "positions_class": ProjectXPositions,
        }

    elif broker == "tradovate":
        from .tradovate.auth import TradovateAuth
        from .tradovate.contract_resolver import TradovateContracts
        from .tradovate.data_feed import TradovateDataFeed
        from .tradovate.order_router import TradovateOrderRouter
        from .tradovate.positions import TradovatePositions

        auth = TradovateAuth(demo=demo)
        return {
            "auth": auth,
            "feed_class": TradovateDataFeed,
            "router_class": TradovateOrderRouter,
            "contracts_class": TradovateContracts,
            "positions_class": TradovatePositions,
        }

    else:
        raise ValueError(f"Unknown broker: '{broker}'. Valid: 'projectx', 'tradovate'")
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_trading_app/test_broker_base.py -v`
Expected: First two tests PASS. Third test (`test_factory_returns_correct_types`) will FAIL because Tradovate submodule doesn't exist yet — that's OK, we'll fix it in Task 1.

**Step 6: Commit**

```bash
git add trading_app/live/broker_base.py trading_app/live/broker_factory.py tests/test_trading_app/test_broker_base.py
git commit -m "feat: broker abstraction layer — 5 ABCs + factory"
```

---

## Task 1: Refactor Tradovate into Submodule

**Files:**
- Create: `trading_app/live/tradovate/__init__.py`
- Create: `trading_app/live/tradovate/auth.py` (move from `tradovate_auth.py`)
- Create: `trading_app/live/tradovate/data_feed.py` (move from `data_feed.py`)
- Create: `trading_app/live/tradovate/order_router.py` (move from `order_router.py`)
- Create: `trading_app/live/tradovate/contract_resolver.py` (move from `contract_resolver.py`)
- Create: `trading_app/live/tradovate/positions.py` (new — stub)
- Modify: `trading_app/live/tradovate_auth.py` (thin re-export)
- Modify: `trading_app/live/data_feed.py` (thin re-export)
- Modify: `trading_app/live/order_router.py` (thin re-export)
- Modify: `trading_app/live/contract_resolver.py` (thin re-export)
- Test: `tests/test_trading_app/test_order_router.py` (must still pass)

**Step 1: Create tradovate submodule directory**

```bash
mkdir -p trading_app/live/tradovate
```

**Step 2: Move tradovate_auth.py -> tradovate/auth.py**

Copy existing `tradovate_auth.py` to `tradovate/auth.py`. Make `TradovateAuth` implement `BrokerAuth` ABC. Key changes:
- Import `BrokerAuth` from `broker_base`
- `class TradovateAuth(BrokerAuth):`
- Add `refresh_if_needed()` method (calls `_refresh()` if within 120s of expiry)
- Keep all existing logic intact

**Step 3: Move data_feed.py -> tradovate/data_feed.py**

Copy existing `data_feed.py` to `tradovate/data_feed.py`. Make `DataFeed` implement `BrokerFeed` ABC:
- Rename class to `TradovateDataFeed`
- `class TradovateDataFeed(BrokerFeed):`
- Constructor calls `super().__init__(auth, on_bar)`
- Import auth from `.auth` instead of `..tradovate_auth`

**Step 4: Move order_router.py -> tradovate/order_router.py**

Copy existing `order_router.py` to `tradovate/order_router.py`. Make `OrderRouter` implement `BrokerRouter`:
- Rename to `TradovateOrderRouter`
- `class TradovateOrderRouter(BrokerRouter):`
- `supports_native_brackets()` returns `False`
- `build_order_spec` and `submit` return dicts instead of dataclasses (match ABC)
- Keep `OrderSpec` and `OrderResult` as internal helpers

**Step 5: Move contract_resolver.py -> tradovate/contract_resolver.py**

Copy existing `contract_resolver.py` to `tradovate/contract_resolver.py`:
- Create `TradovateContracts(BrokerContracts)` wrapping existing functions
- `resolve_account_id()` and `resolve_front_month(instrument)` call existing logic

**Step 6: Create tradovate/positions.py (stub)**

```python
# trading_app/live/tradovate/positions.py
"""Tradovate position queries — stub for future API access."""

import logging
from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)


class TradovatePositions(BrokerPositions):
    def query_open(self, account_id: int) -> list[dict]:
        log.warning("Tradovate position query not implemented (no API access). Returning empty.")
        return []
```

**Step 7: Create tradovate/__init__.py**

```python
# trading_app/live/tradovate/__init__.py
"""Tradovate broker implementation."""
from .auth import TradovateAuth
from .contract_resolver import TradovateContracts
from .data_feed import TradovateDataFeed
from .order_router import TradovateOrderRouter
from .positions import TradovatePositions

__all__ = [
    "TradovateAuth",
    "TradovateContracts",
    "TradovateDataFeed",
    "TradovateOrderRouter",
    "TradovatePositions",
]
```

**Step 8: Create thin re-export shims for backwards compatibility**

Replace old files with re-exports:

```python
# trading_app/live/tradovate_auth.py
"""Backwards-compat re-export. Real implementation in tradovate/auth.py."""
from .tradovate.auth import TradovateAuth  # noqa: F401
```

```python
# trading_app/live/data_feed.py
"""Backwards-compat re-export. Real implementation in tradovate/data_feed.py."""
from .tradovate.data_feed import TradovateDataFeed as DataFeed  # noqa: F401
```

```python
# trading_app/live/order_router.py
"""Backwards-compat re-export. Real implementation in tradovate/order_router.py."""
from .tradovate.order_router import TradovateOrderRouter as OrderRouter  # noqa: F401
from .tradovate.order_router import OrderResult, OrderSpec  # noqa: F401
```

```python
# trading_app/live/contract_resolver.py
"""Backwards-compat re-export. Real implementation in tradovate/contract_resolver.py."""
from .tradovate.contract_resolver import (  # noqa: F401
    TradovateContracts,
    resolve_account_id,
    resolve_front_month,
)
```

**Step 9: Run all tests**

Run: `python -m pytest tests/test_trading_app/test_order_router.py tests/test_trading_app/test_bar_aggregator.py tests/test_trading_app/test_broker_base.py -v`
Expected: ALL PASS — old imports still work via re-exports, new factory test passes.

**Step 10: Commit**

```bash
git add trading_app/live/tradovate/ trading_app/live/tradovate_auth.py trading_app/live/data_feed.py trading_app/live/order_router.py trading_app/live/contract_resolver.py
git commit -m "refactor: move Tradovate into submodule with ABC implementations"
```

---

## Task 2: ProjectX Auth + Contract Discovery

**Files:**
- Create: `trading_app/live/projectx/__init__.py`
- Create: `trading_app/live/projectx/auth.py`
- Create: `trading_app/live/projectx/contract_resolver.py`
- Create: `trading_app/live/projectx/positions.py`
- Modify: `.env` (add BROKER, PROJECTX_USER, PROJECTX_API_KEY)
- Test: `tests/test_trading_app/test_projectx_auth.py`

**Step 1: Write the failing test**

```python
# tests/test_trading_app/test_projectx_auth.py
"""Test ProjectX auth — mocked HTTP."""
from unittest.mock import patch, MagicMock
import pytest


def test_projectx_auth_login():
    """Auth should POST to /api/Auth/loginKey and return JWT."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "token": "test_jwt_token_123",
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            from trading_app.live.projectx.auth import ProjectXAuth
            auth = ProjectXAuth()
            token = auth.get_token()
            assert token == "test_jwt_token_123"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/api/Auth/loginKey" in call_args[0][0]
            assert call_args[1]["json"]["userName"] == "testuser"
            assert call_args[1]["json"]["apiKey"] == "testkey"


def test_projectx_auth_headers():
    """Headers should use Bearer scheme."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"token": "jwt123", "success": True, "errorCode": 0, "errorMessage": None}
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}):
        with patch("requests.post", return_value=mock_resp):
            from trading_app.live.projectx.auth import ProjectXAuth
            auth = ProjectXAuth()
            headers = auth.headers()
            assert headers["Authorization"] == "Bearer jwt123"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_projectx_auth.py -v`
Expected: FAIL — `projectx` module doesn't exist.

**Step 3: Implement ProjectXAuth**

```python
# trading_app/live/projectx/auth.py
"""ProjectX Gateway API authentication.

POST /api/Auth/loginKey with {userName, apiKey} -> JWT token (24h).
Refresh via POST /api/Auth/validate before expiry.
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv

from ..broker_base import BrokerAuth

load_dotenv()
log = logging.getLogger(__name__)

BASE_URL = "https://api.thefuturesdesk.projectx.com"


class ProjectXAuth(BrokerAuth):
    def __init__(self):
        self._token: str | None = None
        self._acquired_at: float = 0
        self._token_lifetime: float = 23 * 3600  # refresh after 23h (token lasts 24h)

    def get_token(self) -> str:
        if self._token and time.time() < self._acquired_at + self._token_lifetime:
            return self._token
        return self._login()

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def refresh_if_needed(self) -> None:
        if self._token is None or time.time() >= self._acquired_at + self._token_lifetime:
            self._validate_or_login()

    def _login(self) -> str:
        user = os.environ["PROJECTX_USER"]
        api_key = os.environ["PROJECTX_API_KEY"]
        resp = requests.post(
            f"{BASE_URL}/api/Auth/loginKey",
            json={"userName": user, "apiKey": api_key},
            headers={"Content-Type": "application/json", "Accept": "text/plain"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"ProjectX auth failed: {data.get('errorMessage', data)}")
        self._token = data["token"]
        self._acquired_at = time.time()
        log.info("ProjectX auth: token acquired")
        return self._token

    def _validate_or_login(self) -> None:
        """Try to validate existing token; fall back to full login."""
        if self._token is None:
            self._login()
            return
        try:
            resp = requests.post(
                f"{BASE_URL}/api/Auth/validate",
                headers=self.headers(),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and data.get("newToken"):
                self._token = data["newToken"]
                self._acquired_at = time.time()
                log.info("ProjectX auth: token refreshed via validate")
            else:
                self._login()
        except Exception:
            log.warning("ProjectX token validate failed, falling back to full login")
            self._login()
```

**Step 4: Implement ProjectXContracts**

```python
# trading_app/live/projectx/contract_resolver.py
"""ProjectX contract resolution and account discovery."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerContracts

log = logging.getLogger(__name__)

BASE_URL = "https://api.thefuturesdesk.projectx.com"

# Instrument -> ProjectX symbolId prefix. VERIFIED against /api/Contract/available at runtime.
# These are starting guesses — resolve_front_month() searches dynamically.
INSTRUMENT_SEARCH_TERMS = {
    "MGC": ["MGC", "Micro Gold"],
    "MNQ": ["MNQ", "Micro Nasdaq", "Micro E-mini Nasdaq"],
    "MES": ["MES", "Micro E-mini S&P", "Micro S&P"],
    "M2K": ["M2K", "Micro E-mini Russell", "Micro Russell"],
}


class ProjectXContracts(BrokerContracts):
    def __init__(self, auth: BrokerAuth, **kwargs):
        super().__init__(auth, **kwargs)
        self._contract_cache: dict[str, str] = {}  # instrument -> contractId

    def resolve_account_id(self) -> int:
        resp = requests.post(
            f"{BASE_URL}/api/Account/search",
            json={"onlyActiveAccounts": True},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        accounts = data.get("accounts", [])
        if not accounts:
            raise RuntimeError("No active ProjectX accounts found")
        acct = accounts[0]
        log.info("ProjectX account: %s (id=%d, canTrade=%s)", acct["name"], acct["id"], acct.get("canTrade"))
        return acct["id"]

    def resolve_front_month(self, instrument: str) -> str:
        if instrument in self._contract_cache:
            return self._contract_cache[instrument]

        resp = requests.post(
            f"{BASE_URL}/api/Contract/available",
            json={"live": False},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        contracts = data.get("contracts", [])

        search_terms = INSTRUMENT_SEARCH_TERMS.get(instrument, [instrument])
        for contract in contracts:
            name = contract.get("name", "")
            desc = contract.get("description", "")
            cid = contract.get("id", "")
            if contract.get("activeContract") and any(
                term.upper() in name.upper() or term.upper() in desc.upper() or term.upper() in cid.upper()
                for term in search_terms
            ):
                self._contract_cache[instrument] = cid
                log.info("ProjectX contract: %s -> %s (%s)", instrument, cid, desc)
                return cid

        # Log all available for debugging
        log.error("Could not find contract for %s. Available contracts:", instrument)
        for c in contracts[:30]:
            log.error("  %s: %s (%s)", c.get("id"), c.get("name"), c.get("description"))
        raise RuntimeError(f"No active ProjectX contract found for {instrument}")
```

**Step 5: Implement ProjectXPositions**

```python
# trading_app/live/projectx/positions.py
"""ProjectX position queries for crash recovery."""

import logging

import requests

from ..broker_base import BrokerAuth, BrokerPositions

log = logging.getLogger(__name__)

BASE_URL = "https://api.thefuturesdesk.projectx.com"


class ProjectXPositions(BrokerPositions):
    def query_open(self, account_id: int) -> list[dict]:
        resp = requests.post(
            f"{BASE_URL}/api/Position/searchOpen",
            json={"accountId": account_id},
            headers=self.auth.headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        positions = data.get("positions", [])
        result = []
        for p in positions:
            result.append({
                "contract_id": p.get("contractId"),
                "side": "long" if p.get("type") == 1 else "short",
                "size": p.get("size", 0),
                "avg_price": p.get("averagePrice", 0),
            })
        if result:
            log.warning("Found %d orphaned positions on session start", len(result))
        return result
```

**Step 6: Create projectx/__init__.py**

```python
# trading_app/live/projectx/__init__.py
"""ProjectX (TopstepX) broker implementation."""
```

**Step 7: Update .env with ProjectX credentials**

Add to `.env` (do NOT commit):
```env
BROKER=projectx
PROJECTX_USER=<your_projectx_username>
PROJECTX_API_KEY=<your_api_key>
```

**Step 8: Run tests**

Run: `python -m pytest tests/test_trading_app/test_projectx_auth.py tests/test_trading_app/test_broker_base.py -v`
Expected: ALL PASS.

**Step 9: Commit**

```bash
git add trading_app/live/projectx/
git commit -m "feat: ProjectX auth + contract resolver + positions"
```

---

## Task 3: ProjectX Order Router

**Files:**
- Create: `trading_app/live/projectx/order_router.py`
- Test: `tests/test_trading_app/test_projectx_router.py`

**Step 1: Write the failing test**

```python
# tests/test_trading_app/test_projectx_router.py
"""Test ProjectX order routing — mocked HTTP."""
from unittest.mock import patch, MagicMock


def test_projectx_market_buy():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"orderId": 9056, "success": True, "errorCode": 0, "errorMessage": None}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        from trading_app.live.projectx.order_router import ProjectXOrderRouter
        router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
        spec = router.build_order_spec(
            direction="long",
            entry_model="E1",
            entry_price=2950.0,
            symbol="CON.F.US.MGC.M26",
            qty=1,
        )
        result = router.submit(spec)
        assert result["order_id"] == 9056
        call_body = mock_post.call_args[1]["json"]
        assert call_body["accountId"] == 123
        assert call_body["type"] == 2  # Market
        assert call_body["side"] == 0  # Bid (buy)
        assert call_body["size"] == 1


def test_projectx_stop_sell():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"orderId": 9057, "success": True, "errorCode": 0, "errorMessage": None}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp):
        from trading_app.live.projectx.order_router import ProjectXOrderRouter
        router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
        spec = router.build_order_spec(
            direction="short",
            entry_model="E2",
            entry_price=2950.0,
            symbol="CON.F.US.MGC.M26",
            qty=1,
        )
        assert spec["type"] == 4  # Stop
        assert spec["side"] == 1  # Ask (sell)
        assert spec["stopPrice"] == 2950.0


def test_projectx_supports_brackets():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}
    from trading_app.live.projectx.order_router import ProjectXOrderRouter
    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert router.supports_native_brackets() is True
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_projectx_router.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement ProjectXOrderRouter**

```python
# trading_app/live/projectx/order_router.py
"""ProjectX order routing via REST API.

Order types: 1=Limit, 2=Market, 4=Stop, 5=TrailingStop
Sides: 0=Bid (buy), 1=Ask (sell)
"""

import logging
import time

import requests

from ..broker_base import BrokerAuth, BrokerRouter

log = logging.getLogger(__name__)

BASE_URL = "https://api.thefuturesdesk.projectx.com"


class ProjectXOrderRouter(BrokerRouter):
    def __init__(self, account_id: int, auth: BrokerAuth | None, **kwargs):
        super().__init__(account_id, auth, **kwargs)

    def build_order_spec(
        self,
        direction: str,
        entry_model: str,
        entry_price: float,
        symbol: str,
        qty: int = 1,
    ) -> dict:
        side = 0 if direction == "long" else 1  # 0=Bid(buy), 1=Ask(sell)

        if entry_model == "E1":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 2,  # Market
                "side": side,
                "size": qty,
            }
        elif entry_model == "E2":
            return {
                "accountId": self.account_id,
                "contractId": symbol,
                "type": 4,  # Stop
                "side": side,
                "size": qty,
                "stopPrice": entry_price,
            }
        else:
            raise ValueError(f"Entry model '{entry_model}' not supported live. Use E1 or E2.")

    def submit(self, spec: dict) -> dict:
        if self.auth is None:
            raise RuntimeError("No auth — cannot submit orders without ProjectXAuth")

        t0 = time.monotonic()
        resp = requests.post(
            f"{BASE_URL}/api/Order/place",
            json=spec,
            headers={**self.auth.headers(), "Content-Type": "application/json"},
            timeout=5,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            raise RuntimeError(f"ProjectX order failed: {data.get('errorMessage', data)}")

        order_id = data.get("orderId", 0)
        log.info(
            "ProjectX order placed: side=%d type=%d qty=%d -> orderId=%d (%.0fms)",
            spec.get("side", -1),
            spec.get("type", -1),
            spec.get("size", 0),
            order_id,
            elapsed_ms,
        )
        if elapsed_ms > 1000:
            log.warning("Order submission took %.0fms", elapsed_ms)
        return {"order_id": order_id, "status": "submitted"}

    def build_exit_spec(self, direction: str, symbol: str, qty: int = 1) -> dict:
        side = 1 if direction == "long" else 0  # Reverse: close long=sell, close short=buy
        return {
            "accountId": self.account_id,
            "contractId": symbol,
            "type": 2,  # Market
            "side": side,
            "size": qty,
        }

    def cancel(self, order_id: int) -> None:
        if self.auth is None:
            log.error("Cannot cancel order %d — no auth", order_id)
            return
        resp = requests.post(
            f"{BASE_URL}/api/Order/cancel",
            json={"orderId": order_id},
            headers=self.auth.headers(),
            timeout=5,
        )
        resp.raise_for_status()
        log.info("ProjectX order cancelled: orderId=%d", order_id)

    def supports_native_brackets(self) -> bool:
        return True
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_trading_app/test_projectx_router.py -v`
Expected: ALL PASS.

**Step 5: Commit**

```bash
git add trading_app/live/projectx/order_router.py tests/test_trading_app/test_projectx_router.py
git commit -m "feat: ProjectX order router with native bracket support"
```

---

## Task 4: ProjectX SignalR Data Feed

**Files:**
- Create: `trading_app/live/projectx/data_feed.py`
- Modify: `pyproject.toml` (add `pysignalr` dependency)
- Test: `tests/test_trading_app/test_projectx_feed.py`

**Step 1: Add pysignalr dependency**

```bash
uv add pysignalr
```

If `pysignalr` has Windows issues, fall back to:
```bash
uv add signalrcore
```

**Step 2: Write the failing test**

```python
# tests/test_trading_app/test_projectx_feed.py
"""Test ProjectX data feed — verify it creates bars from quotes."""
from datetime import UTC, datetime
from unittest.mock import MagicMock

from trading_app.live.bar_aggregator import Bar


def test_projectx_feed_creates_bar_from_quote():
    """Verify that a GatewayQuote message produces ticks for BarAggregator."""
    from trading_app.live.projectx.data_feed import ProjectXDataFeed

    bars_received = []

    async def on_bar(bar: Bar):
        bars_received.append(bar)

    mock_auth = MagicMock()
    mock_auth.get_token.return_value = "test_token"

    feed = ProjectXDataFeed(auth=mock_auth, on_bar=on_bar)

    # Simulate a GatewayQuote message
    quote = {
        "lastPrice": 2950.25,
        "volume": 100,
        "timestamp": "2026-03-06T08:00:00Z",
    }
    # The feed should be able to parse this into a tick
    price, vol = feed.parse_quote(quote)
    assert price == 2950.25
    assert vol == 100
```

**Step 3: Implement ProjectXDataFeed**

This is the most complex piece — SignalR WebSocket connection to the ProjectX market hub. The implementation must:
1. Connect to `wss://rtc.thefuturesdesk.projectx.com/hubs/market`
2. Subscribe to `SubscribeContractQuotes` and `SubscribeContractTrades`
3. Handle `GatewayQuote` and `GatewayTrade` events
4. Feed prices into BarAggregator
5. Call `on_bar()` for each completed 1-minute bar

The exact implementation depends on which SignalR library works on Windows. Write the core structure, test with signal-only mode.

**Step 4: Run tests**

Run: `python -m pytest tests/test_trading_app/test_projectx_feed.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add trading_app/live/projectx/data_feed.py tests/test_trading_app/test_projectx_feed.py pyproject.toml uv.lock
git commit -m "feat: ProjectX SignalR data feed"
```

---

## Task 5: Wire SessionOrchestrator to Broker Factory

**Files:**
- Modify: `trading_app/live/session_orchestrator.py`
- Modify: `scripts/run_live_session.py` (add `--broker` flag)
- Test: existing tests must still pass

**Step 1: Modify SessionOrchestrator.__init__**

Replace hardcoded Tradovate imports with broker factory:

```python
# Key changes to __init__:
def __init__(self, instrument: str, broker: str = "projectx", demo: bool = True,
             account_id: int = 0, signal_only: bool = False):
    # ... existing trading_day logic ...

    # Create broker components via factory
    from .broker_factory import create_broker_components
    components = create_broker_components(broker, demo=demo)
    self.auth = components["auth"]
    contracts_cls = components["contracts_class"]
    positions_cls = components["positions_class"]

    contracts = contracts_cls(auth=self.auth, demo=demo)
    self.positions = positions_cls(auth=self.auth)

    # ... existing portfolio build logic (unchanged) ...

    if signal_only:
        self.order_router = None
    else:
        if account_id == 0:
            account_id = contracts.resolve_account_id()
        router_cls = components["router_class"]
        self.order_router = router_cls(account_id=account_id, auth=self.auth)

    # Contract resolution
    self.contract_symbol = contracts.resolve_front_month(instrument)

    # Position reconciliation on startup (M2.5 P0)
    if not signal_only and account_id:
        orphans = self.positions.query_open(account_id)
        if orphans:
            log.critical("ORPHANED POSITIONS DETECTED: %s", orphans)

    # ... rest of existing init ...
```

**Step 2: Modify run() to use broker feed**

```python
async def run(self) -> None:
    from .broker_factory import create_broker_components
    components = create_broker_components(self._broker_name, demo=self.demo)
    feed_cls = components["feed_class"]
    feed = feed_cls(self.auth, on_bar=self._on_bar, demo=self.demo)
    log.info("Starting live feed: %s (broker: %s)", self.contract_symbol, self._broker_name)
    await feed.run(self.contract_symbol)
```

**Step 3: Add --broker flag to run_live_session.py**

```python
parser.add_argument(
    "--broker",
    default=None,
    help="Broker: 'projectx' or 'tradovate' (default: from BROKER env var)",
)
```

Pass `broker=args.broker` to `SessionOrchestrator()`.

**Step 4: Run tests**

Run: `python -m pytest tests/ -x -q`
Expected: ALL PASS.

**Step 5: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: PASS.

**Step 6: Commit**

```bash
git add trading_app/live/session_orchestrator.py scripts/run_live_session.py
git commit -m "feat: wire SessionOrchestrator to broker factory"
```

---

## Task 6: Auth Test + First Signal-Only Session

**Files:** None to create — all code exists.

**Step 1: Get ProjectX API credentials**

1. Open TopstepX platform
2. Click Settings (gear icon) > API tab > ProjectX Linking
3. Go to `dashboard.projectx.com`
4. Register / log in
5. Sidebar > Subscriptions > ProjectX API Access
6. Use promo code `topstep` for 50% off ($14.50/mo)
7. Generate API key from settings

**Step 2: Fill .env**

```env
BROKER=projectx
PROJECTX_USER=your_username
PROJECTX_API_KEY=your_api_key
```

**Step 3: Test auth**

```bash
python -c "
from trading_app.live.projectx.auth import ProjectXAuth
auth = ProjectXAuth()
token = auth.get_token()
print(f'Token: {token[:30]}...')
print(f'Headers: {auth.headers()}')
"
```
Expected: Token string printed, no errors.

**Step 4: Test contract discovery**

```bash
python -c "
from trading_app.live.projectx.auth import ProjectXAuth
from trading_app.live.projectx.contract_resolver import ProjectXContracts
auth = ProjectXAuth()
contracts = ProjectXContracts(auth=auth)
acct_id = contracts.resolve_account_id()
print(f'Account ID: {acct_id}')
for inst in ['MGC', 'MNQ', 'MES', 'M2K']:
    try:
        cid = contracts.resolve_front_month(inst)
        print(f'{inst} -> {cid}')
    except Exception as e:
        print(f'{inst} -> ERROR: {e}')
"
```
Expected: Account ID and contract IDs for all 4 instruments.

**Step 5: Run signal-only session**

```bash
python scripts/run_live_session.py --instrument MGC --broker projectx --signal-only
```

Expected: Connects to ProjectX market hub, receives quotes, builds bars, logs signals.

**Step 6: Verify clean shutdown**

Press Ctrl+C. Verify `post_session()` runs.

---

## Task 7: Drift Checks + Cleanup

**Files:**
- Modify: `pipeline/check_drift.py` (add broker-related checks)
- Modify: `trading_app/live/__init__.py` (update comment)

**Step 1: Add drift check for BROKER env var**

```python
def check_broker_env_var() -> list[str]:
    """Ensure BROKER env var is set to a valid value when live modules are used."""
    valid = {"projectx", "tradovate"}
    broker = os.environ.get("BROKER", "")
    if broker and broker.lower() not in valid:
        return [f"  BROKER={broker!r} is not valid. Must be one of: {valid}"]
    return []
```

**Step 2: Update existing Tradovate URL drift check**

Extend check #71 to also scan `projectx/` files for correct base URLs.

**Step 3: Run full test suite + drift**

```bash
python -m pytest tests/ -x -q
python pipeline/check_drift.py
```

**Step 4: Commit**

```bash
git add pipeline/check_drift.py trading_app/live/__init__.py
git commit -m "chore: broker drift checks + cleanup"
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| SignalR library doesn't work on Windows | Test `pysignalr` first; fallback to `signalrcore`; worst case use REST polling |
| ProjectX contract IDs don't match guesses | Dynamic discovery via `/api/Contract/available` — logs all available contracts |
| Token expires mid-session | `refresh_if_needed()` called proactively; 401 triggers re-auth |
| Position orphaning on crash | `query_open()` on session start detects orphans |
| Old imports break | Thin re-export shims maintain all existing import paths |

## Validation Checklist

- [ ] All 5 ABCs defined and tested (broker_base.py)
- [ ] Tradovate code moved to submodule, old imports work via re-exports
- [ ] ProjectX auth works with real credentials
- [ ] Contract discovery finds MGC/MNQ/MES/M2K on ProjectX
- [ ] Signal-only session connects and receives bars
- [ ] Order router builds correct ProjectX order payloads
- [ ] Position query works (for crash recovery)
- [ ] `--broker` flag works on CLI
- [ ] Drift checks pass
- [ ] All existing tests pass
