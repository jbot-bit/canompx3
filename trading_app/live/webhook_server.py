"""
TradingView webhook server — receives alerts and places orders via Tradovate.

Architecture:
    TradingView alert (Pine Script) → POST /trade JSON → this server → OrderRouter → Tradovate

Setup:
    1. Add WEBHOOK_SECRET to .env
    2. python scripts/run_webhook_server.py          # starts on localhost:8765
    3. cloudflared tunnel run orb-webhook            # exposes via persistent HTTPS URL
    4. Paste the public URL into TradingView Alert → Webhook URL

TradingView Pine Script alert message (JSON string in the alert dialog):
    {"instrument":"MGC","direction":"long","action":"entry","qty":1,"entry_model":"E1","secret":"YOUR_SECRET"}

Endpoints:
    GET  /health  — liveness check
    POST /trade   — place a trade (entry or exit)
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

load_dotenv()

log = logging.getLogger(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
DEMO = os.environ.get("WEBHOOK_DEMO", "true").lower() != "false"
PORT = int(os.environ.get("WEBHOOK_PORT", "8765"))

# Rate limiting: max 3 orders per 60 seconds (guards against runaway alerts)
_ORDER_TIMESTAMPS: deque[float] = deque(maxlen=10)
RATE_LIMIT_ORDERS = 3
RATE_LIMIT_WINDOW = 60.0  # seconds


# ── Lazy-initialized singletons (created on first request) ──────────────────
_auth = None
_account_id: int | None = None
_contract_cache: dict[str, tuple[str, float]] = {}  # instrument → (symbol, resolved_at)
CONTRACT_TTL = 3600.0  # re-resolve front month after 1 hour


def _get_auth():
    global _auth
    if _auth is None:
        from trading_app.live.tradovate.auth import TradovateAuth

        _auth = TradovateAuth(demo=DEMO)
        log.info("TradovateAuth initialized (demo=%s)", DEMO)
    return _auth


def _get_account_id() -> int:
    global _account_id
    if _account_id is None:
        from trading_app.live.tradovate.contract_resolver import resolve_account_id

        _account_id = resolve_account_id(_get_auth(), demo=DEMO)
    return _account_id


def _get_contract(instrument: str) -> str:
    """Resolve front-month contract symbol, cached for 1 hour."""
    from trading_app.live.tradovate.contract_resolver import resolve_front_month

    now = time.monotonic()
    if instrument in _contract_cache:
        symbol, resolved_at = _contract_cache[instrument]
        if now - resolved_at < CONTRACT_TTL:
            return symbol
    symbol = resolve_front_month(instrument, _get_auth(), demo=DEMO)
    _contract_cache[instrument] = (symbol, now)
    log.info("Resolved contract: %s → %s", instrument, symbol)
    return symbol


# ── Lifespan: warm up auth on startup ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Webhook server starting — demo=%s port=%d", DEMO, PORT)
    if not WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET env var is required — refusing to start without authentication")
    try:
        # Warm up auth + account ID in a thread (synchronous HTTP call)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _get_account_id)
        log.info("Auth ready — account_id=%d", _account_id)
    except Exception as e:
        log.error("Auth warm-up failed: %s — orders will be rejected until auth resolves", e)
    yield
    log.info("Webhook server shutting down")


app = FastAPI(title="ORB Webhook Server", lifespan=lifespan)


# ── Request / Response models ────────────────────────────────────────────────
class TradeRequest(BaseModel):
    instrument: str
    direction: str  # "long" | "short"
    action: str = "entry"  # "entry" | "exit"
    qty: int = 1
    entry_model: str = "E1"  # "E1" (market) | "E2" (stop-market)
    entry_price: float | None = None  # required only for E2 stop orders
    secret: str = ""

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.lower()
        if v not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got '{v}'")
        return v

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        v = v.lower()
        if v not in ("entry", "exit"):
            raise ValueError(f"action must be 'entry' or 'exit', got '{v}'")
        return v

    @field_validator("entry_model")
    @classmethod
    def validate_entry_model(cls, v: str) -> str:
        v = v.upper()
        if v not in ("E1", "E2"):
            raise ValueError(f"entry_model must be 'E1' or 'E2', got '{v}'")
        return v

    @field_validator("instrument")
    @classmethod
    def validate_instrument(cls, v: str) -> str:
        return v.upper()


class TradeResponse(BaseModel):
    order_id: int
    status: str
    contract: str
    action: str
    direction: str
    qty: int
    demo: bool
    timestamp: str


# ── Helpers ──────────────────────────────────────────────────────────────────
def _check_rate_limit() -> None:
    """Raise HTTPException if rate limit exceeded."""
    now = time.monotonic()
    # Drop timestamps outside the window
    while _ORDER_TIMESTAMPS and now - _ORDER_TIMESTAMPS[0] > RATE_LIMIT_WINDOW:
        _ORDER_TIMESTAMPS.popleft()
    if len(_ORDER_TIMESTAMPS) >= RATE_LIMIT_ORDERS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {RATE_LIMIT_ORDERS} orders per {RATE_LIMIT_WINDOW:.0f}s",
        )
    _ORDER_TIMESTAMPS.append(now)


def _place_order(req: TradeRequest, contract: str) -> int:
    """Build and submit order. Returns order_id. Runs in a thread executor."""
    from trading_app.live.tradovate.order_router import TradovateOrderRouter as OrderRouter

    router = OrderRouter(account_id=_get_account_id(), auth=_get_auth(), demo=DEMO)

    if req.action == "entry":
        if req.entry_model == "E2" and req.entry_price is None:
            raise ValueError("entry_price is required for E2 stop-market orders")
        spec = router.build_order_spec(
            direction=req.direction,
            entry_model=req.entry_model,
            entry_price=req.entry_price or 0.0,
            symbol=contract,
            qty=req.qty,
        )
    else:  # exit
        spec = router.build_exit_spec(
            direction=req.direction,
            symbol=contract,
            qty=req.qty,
        )

    result = router.submit(spec)
    return result.get("order_id") if isinstance(result, dict) else getattr(result, "order_id", None)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "demo": DEMO,
        "account_id": _account_id,
        "rate_limit": f"{RATE_LIMIT_ORDERS} orders/{RATE_LIMIT_WINDOW:.0f}s",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/trade", response_model=TradeResponse)
async def trade(req: TradeRequest, request: Request):
    # 1. Auth check
    if not hmac.compare_digest(req.secret, WEBHOOK_SECRET):
        log.warning("Rejected webhook from %s — invalid secret", request.client)
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    # 2. Rate limit
    _check_rate_limit()

    # 3. Resolve contract (cached, async-safe)
    loop = asyncio.get_running_loop()
    try:
        contract = await loop.run_in_executor(None, _get_contract, req.instrument)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Contract resolution failed: {e}") from e

    # 4. Place order (synchronous HTTP in thread pool to avoid blocking event loop)
    try:
        order_id = await loop.run_in_executor(None, _place_order, req, contract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.error("Order failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order failed: {e}") from e

    log.info(
        "WEBHOOK ORDER: %s %s %s qty=%d → orderId=%d (%s)",
        req.action,
        req.instrument,
        req.direction,
        req.qty,
        order_id,
        "DEMO" if DEMO else "LIVE",
    )

    return TradeResponse(
        order_id=order_id,
        status="submitted",
        contract=contract,
        action=req.action,
        direction=req.direction,
        qty=req.qty,
        demo=DEMO,
        timestamp=datetime.now(UTC).isoformat(),
    )
