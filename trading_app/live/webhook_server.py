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

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

load_dotenv()

log = logging.getLogger(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
DEMO = os.environ.get("WEBHOOK_DEMO", "true").lower() != "false"
PORT = int(os.environ.get("WEBHOOK_PORT", "8765"))
BROKER = os.environ.get("WEBHOOK_BROKER", "tradovate")  # "tradovate" | "projectx"

# Dedup: reject identical (instrument, direction, action) within window (guards against TV double-fire)
DEDUP_WINDOW = float(os.environ.get("WEBHOOK_DEDUP_SECONDS", "10"))
_DEDUP_CACHE: dict[str, tuple[float, TradeResponse]] = {}  # key → (monotonic_ts, cached_response)

# Rate limiting: max 3 orders per 60 seconds (guards against runaway alerts)
_ORDER_TIMESTAMPS: deque[float] = deque(maxlen=10)
RATE_LIMIT_ORDERS = 3
RATE_LIMIT_WINDOW = 60.0  # seconds

# Position limit: max 1 open position per instrument (guards against double entries)
MAX_OPEN_POSITIONS = int(os.environ.get("WEBHOOK_MAX_POSITIONS", "1"))
_OPEN_POSITIONS: dict[str, int] = {}

# Max qty per order (guards against fat-finger or bad alert data)
MAX_ORDER_QTY = int(os.environ.get("WEBHOOK_MAX_QTY", "5"))

# Instrument allowlist — only instruments in ACTIVE_ORB_INSTRUMENTS can be traded
_ALLOWED_INSTRUMENTS = frozenset(ACTIVE_ORB_INSTRUMENTS)


# ── Lazy-initialized singletons (created on first request) ──────────────────
_auth = None
_account_id: int | None = None
_contract_cache: dict[str, tuple[str, float]] = {}  # instrument → (symbol, resolved_at)
CONTRACT_TTL = 3600.0  # re-resolve front month after 1 hour


_broker_components = None


def _get_broker():
    """Lazy-initialize broker components via factory. Supports tradovate and projectx."""
    global _broker_components
    if _broker_components is None:
        from trading_app.live.broker_factory import create_broker_components

        _broker_components = create_broker_components(BROKER, demo=DEMO)
        log.info("Broker initialized: %s (demo=%s)", BROKER, DEMO)
    return _broker_components


def _get_auth():
    global _auth
    if _auth is None:
        _auth = _get_broker()["auth"]
    return _auth


def _get_account_id() -> int:
    global _account_id
    if _account_id is None:
        contracts_cls = _get_broker()["contracts_class"]
        contracts = contracts_cls(auth=_get_auth())
        _account_id = contracts.resolve_account_id()
    return _account_id


def _get_contract(instrument: str) -> str:
    """Resolve front-month contract symbol, cached for 1 hour."""
    now = time.monotonic()
    if instrument in _contract_cache:
        symbol, resolved_at = _contract_cache[instrument]
        if now - resolved_at < CONTRACT_TTL:
            return symbol
    contracts_cls = _get_broker()["contracts_class"]
    contracts = contracts_cls(auth=_get_auth())
    symbol = contracts.resolve_front_month(instrument)
    _contract_cache[instrument] = (symbol, now)
    log.info("Resolved contract: %s → %s", instrument, symbol)
    return symbol


# ── Lifespan: validate secret, keep broker auth lazy ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Webhook server starting — demo=%s port=%d", DEMO, PORT)
    if not WEBHOOK_SECRET:
        raise RuntimeError("WEBHOOK_SECRET env var is required — refusing to start without authentication")
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
    order_id: int | None
    status: str
    contract: str
    action: str
    direction: str
    qty: int
    demo: bool
    timestamp: str


# ── Helpers ──────────────────────────────────────────────────────────────────
def _dedup_key(req: TradeRequest) -> str:
    return f"{req.instrument}:{req.direction}:{req.action}"


def _check_dedup(req: TradeRequest) -> TradeResponse | None:
    """Return cached response if identical request arrived within DEDUP_WINDOW, else None."""
    if DEDUP_WINDOW <= 0:
        return None
    key = _dedup_key(req)
    now = time.monotonic()
    # Prune expired entries (cheap — cache is tiny, at most one per key)
    expired = [k for k, (ts, _) in _DEDUP_CACHE.items() if now - ts > DEDUP_WINDOW]
    for k in expired:
        del _DEDUP_CACHE[k]
    if key in _DEDUP_CACHE:
        ts, cached = _DEDUP_CACHE[key]
        if now - ts <= DEDUP_WINDOW:
            log.warning("DEDUP: blocked duplicate %s within %.1fs", key, now - ts)
            return TradeResponse(
                order_id=cached.order_id,
                status="deduplicated",
                contract=cached.contract,
                action=cached.action,
                direction=cached.direction,
                qty=cached.qty,
                demo=cached.demo,
                timestamp=cached.timestamp,
            )
    return None


def _cache_response(req: TradeRequest, resp: TradeResponse) -> None:
    """Cache a successful response for dedup."""
    if DEDUP_WINDOW > 0:
        _DEDUP_CACHE[_dedup_key(req)] = (time.monotonic(), resp)


def _check_position_limit(req: TradeRequest) -> None:
    """Block entry if instrument already at max open positions. Exits always allowed."""
    if req.action != "entry":
        return
    current = _OPEN_POSITIONS.get(req.instrument, 0)
    if current >= MAX_OPEN_POSITIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Position limit: {req.instrument} already has {current} open position(s)",
        )


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


def _place_order(req: TradeRequest, contract: str) -> int | None:
    """Build and submit order. Returns order_id (or None if unavailable). Runs in a thread executor."""
    router_cls = _get_broker()["router_class"]
    router = router_cls(account_id=_get_account_id(), auth=_get_auth(), demo=DEMO)

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
    # OrderResult may be a dict (legacy) or dataclass (current) — handle both
    if isinstance(result, dict):
        return result.get("order_id")
    return getattr(result, "order_id", None)


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

    # 2. Instrument allowlist — reject unknown instruments
    if req.instrument not in _ALLOWED_INSTRUMENTS:
        log.warning("Rejected webhook: unknown instrument %s", req.instrument)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown instrument '{req.instrument}'. Allowed: {sorted(_ALLOWED_INSTRUMENTS)}",
        )

    # 3. Qty cap — reject oversized orders
    if req.qty > MAX_ORDER_QTY:
        log.warning("Rejected webhook: qty %d exceeds max %d", req.qty, MAX_ORDER_QTY)
        raise HTTPException(
            status_code=400,
            detail=f"Qty {req.qty} exceeds max {MAX_ORDER_QTY}",
        )

    # 4. Dedup check — block identical (instrument, direction, action) within window
    cached = _check_dedup(req)
    if cached is not None:
        return cached

    # 5. Rate limit
    _check_rate_limit()

    # 6. Position limit (entry only)
    _check_position_limit(req)

    # 7. Resolve contract (cached, async-safe)
    loop = asyncio.get_running_loop()
    try:
        contract = await loop.run_in_executor(None, _get_contract, req.instrument)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Contract resolution failed: {e}") from e

    # 8. Place order (synchronous HTTP in thread pool to avoid blocking event loop)
    try:
        order_id = await loop.run_in_executor(None, _place_order, req, contract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.error("Order failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order failed: {e}") from e

    log.info(
        "WEBHOOK ORDER: %s %s %s qty=%d → orderId=%s (%s)",
        req.action,
        req.instrument,
        req.direction,
        req.qty,
        order_id,
        "DEMO" if DEMO else "LIVE",
    )

    resp = TradeResponse(
        order_id=order_id,
        status="submitted",
        contract=contract,
        action=req.action,
        direction=req.direction,
        qty=req.qty,
        demo=DEMO,
        timestamp=datetime.now(UTC).isoformat(),
    )

    # 9. Update position tracking
    if req.action == "entry":
        _OPEN_POSITIONS[req.instrument] = _OPEN_POSITIONS.get(req.instrument, 0) + 1
    elif req.action == "exit":
        _OPEN_POSITIONS[req.instrument] = max(0, _OPEN_POSITIONS.get(req.instrument, 0) - 1)

    # 10. Cache for dedup
    _cache_response(req, resp)

    return resp
