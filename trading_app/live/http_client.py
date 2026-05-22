"""Canonical HTTP client for broker REST APIs.

Replaces five separate retry implementations across the live-trading codebase
with a single classified-retry client that every broker module funnels through.

Failure-mode taxonomy (each class has distinct retry semantics):
    A. DNS / connect failure        → retry with backoff
    B. Read timeout (slow body)     → retry with backoff (bounded by deadline)
    C. TCP RST mid-request          → retry with backoff
    D. 5xx server error             → retry with backoff
    E. 429 rate limit               → respect Retry-After, then backoff
    F. 401 auth expiry              → refresh hook + single retry
    G. Malformed JSON / success:false → never retry (raise BrokerProtocolError)

Every retry respects a per-call deadline_ms so retries cannot blow caller's SLA.

Per-call CircuitBreaker hook (Stage 4 wires this): record_failure(class) +
record_success() are invoked on every attempt, letting orchestration block
calls when an endpoint has been failing systemically.

Grounding:
    - Plan reference: docs/specs/live_execution_resilience.md (Stage 5)
    - Existing pattern: projectx.order_router._backoff_wait (jitter math)
    - Existing pattern: circuit_breaker.CircuitBreaker (state transitions)

NOTE: This client is the ONLY sanctioned way for trading_app/live/projectx/**
and trading_app/live/tradovate/** to perform HTTP. A drift check (Stage 5)
forbids direct requests.{get,post}() in those subpackages.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)


# ─── Typed exception hierarchy ──────────────────────────────────────────────


class BrokerHTTPError(Exception):
    """Root broker HTTP failure. Always carries the failure class letter."""

    def __init__(self, message: str, error_class: str, *, last_response: requests.Response | None = None):
        super().__init__(message)
        self.error_class = error_class
        self.last_response = last_response


class BrokerTransientError(BrokerHTTPError):
    """Class A/B/C/D — retries exhausted. Caller may abort or escalate."""


class BrokerAuthError(BrokerHTTPError):
    """Class F — auth refresh attempted and still failing. Token state is bad."""


class BrokerRateLimitExhausted(BrokerHTTPError):
    """Class E — 429 retries exhausted. Caller MUST surface; never swallow."""


class BrokerProtocolError(BrokerHTTPError):
    """Class G — malformed payload / application-level success=false. No retry."""


class BrokerPermanentError(BrokerHTTPError):
    """4xx other than 401/429 — permanent. No retry."""


# ─── Retry policy ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryPolicy:
    """Per-call retry budget. Keep deadline shorter than caller's SLA."""

    max_attempts: int = 5  # includes first attempt
    base_backoff_s: float = 0.5
    backoff_factor: float = 2.0
    backoff_max_s: float = 8.0
    jitter_factor: float = 0.2  # ±20%
    deadline_s: float = 12.0  # total wall-clock budget across attempts
    max_429_attempts: int = 3  # tighter than general retries
    retry_on_status: tuple[int, ...] = (500, 502, 503, 504)


# Conservative reads policy — idempotent GETs and lookups.
READ_POLICY = RetryPolicy()

# Order placement / cancellation policy — same retry math but caller MUST
# present an idempotency key. This is enforced at the call site (order_router),
# not by the client itself.
ORDER_POLICY = RetryPolicy(max_attempts=4, deadline_s=10.0)

# Auth policy — login is rare; short deadline keeps startup fast on success.
AUTH_POLICY = RetryPolicy(max_attempts=3, deadline_s=6.0, base_backoff_s=1.0, backoff_max_s=4.0)


# ─── Failure classification ────────────────────────────────────────────────


def _classify_exception(exc: BaseException) -> str:
    """Map a requests exception to the failure-class letter."""
    import socket

    if isinstance(exc, requests.ConnectionError):
        cause = exc.__cause__ or exc.__context__
        if isinstance(cause, ConnectionResetError):
            return "C"
        if isinstance(cause, socket.gaierror):
            return "A"
        return "A"
    if isinstance(exc, requests.Timeout):
        return "B"
    return "G"


def _classify_response(resp: requests.Response) -> str | None:
    """Map an HTTP response to a class letter, or None if it should not retry."""
    if resp.status_code == 429:
        return "E"
    if resp.status_code == 401:
        return "F"
    if 500 <= resp.status_code < 600:
        return "D"
    return None


# ─── Backoff math (lifted from order_router._backoff_wait + capped) ────────


def _backoff_wait(attempt: int, policy: RetryPolicy) -> float:
    """Exponential backoff with ±jitter, clamped to policy.backoff_max_s."""
    base = min(policy.base_backoff_s * (policy.backoff_factor**attempt), policy.backoff_max_s)
    jitter = base * policy.jitter_factor * (2 * random.random() - 1)
    return max(0.05, base + jitter)


def _parse_retry_after(resp: requests.Response, fallback_s: float) -> float:
    """Honor Retry-After header (seconds or HTTP-date). Fall back to backoff."""
    val = resp.headers.get("Retry-After")
    if not val:
        return fallback_s
    try:
        return max(0.05, float(val))
    except ValueError:
        return fallback_s


# ─── Failure hook (Stage 4 wires CircuitBreaker.record_failure here) ───────


@dataclass
class _NoopFailureHook:
    """Default failure hook — does nothing. Stage 4 replaces this with circuit
    breakers per (broker, endpoint_class)."""

    def record_failure(self, error_class: str) -> None:  # noqa: B027
        return

    def record_success(self) -> None:  # noqa: B027
        return

    def should_allow_request(self) -> bool:
        return True


# ─── The client ────────────────────────────────────────────────────────────


@dataclass
class BrokerHTTPClient:
    """Session-backed HTTP client with classified retry + deadline propagation.

    Construction:
        client = BrokerHTTPClient(
            base_url="https://api.example.com",
            refresh_token=auth_obj.refresh_if_needed,  # optional, for 401
            failure_hook=circuit_breaker,              # optional, Stage 4
        )

    Use:
        data = client.post_json("/api/Order/place", auth_headers, body, ORDER_POLICY)
        # or
        resp = client.request("POST", "/api/...", headers=h, json=body, policy=READ_POLICY)
    """

    base_url: str
    refresh_token: Callable[[], None] | None = None
    failure_hook: Any = field(default_factory=_NoopFailureHook)
    # `session` is typed as Any to permit duck-typed stubs in tests (anything
    # exposing `.request(method, url, **kwargs)`).
    session: Any = None
    name: str = "broker"

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()
        self.base_url = self.base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
        timeout: float = 10.0,
        policy: RetryPolicy = READ_POLICY,
    ) -> requests.Response:
        """Perform an HTTP request with classified retry + deadline budgeting.

        `path` may be absolute (http://...) or relative to `self.base_url`.
        `timeout` is the PER-ATTEMPT socket timeout (read + connect). The
        overall budget is policy.deadline_s.
        """
        if not self.failure_hook.should_allow_request():
            raise BrokerTransientError(
                f"{self.name}: circuit breaker OPEN — request refused",
                error_class="X",
            )

        url = path if path.startswith("http") else f"{self.base_url}{path if path.startswith('/') else '/' + path}"

        deadline = time.monotonic() + policy.deadline_s
        attempts_remaining = policy.max_attempts
        attempts_429 = policy.max_429_attempts
        attempt = 0
        refresh_used = False

        while True:
            attempt += 1
            attempts_remaining -= 1
            now = time.monotonic()
            if now >= deadline:
                msg = f"{self.name}: deadline {policy.deadline_s:.1f}s exceeded on {path}"
                log.error(msg)
                self.failure_hook.record_failure("B")
                raise BrokerTransientError(msg, error_class="B")

            try:
                resp = self.session.request(  # type: ignore[union-attr]
                    method,
                    url,
                    headers=headers,
                    json=json,
                    params=params,
                    timeout=min(timeout, max(0.5, deadline - now)),
                )
            except requests.RequestException as exc:
                error_class = _classify_exception(exc)
                self.failure_hook.record_failure(error_class)
                if attempts_remaining <= 0 or time.monotonic() >= deadline:
                    msg = f"{self.name}: {error_class} exhausted on {path} after {attempt} attempts ({exc})"
                    log.error(msg)
                    raise BrokerTransientError(msg, error_class=error_class) from exc
                wait = _backoff_wait(attempt - 1, policy)
                log.warning(
                    "%s: class=%s %s on %s (attempt %d/%d) — retrying in %.1fs",
                    self.name,
                    error_class,
                    type(exc).__name__,
                    path,
                    attempt,
                    policy.max_attempts,
                    wait,
                )
                _sleep_bounded(wait, deadline)
                continue

            # Got a response — classify status.
            cls = _classify_response(resp)
            if cls is None:
                # 2xx or 3xx or non-retryable 4xx. raise_for_status handled by caller.
                if resp.status_code >= 400:
                    self.failure_hook.record_failure("perm")
                    raise BrokerPermanentError(
                        f"{self.name}: HTTP {resp.status_code} on {path}: {resp.text[:200]}",
                        error_class="perm",
                        last_response=resp,
                    )
                # NOTE: success recording happens at the parse layer (post_json / get_json)
                # so a 200 + {"success": false} body counts as failure not success. Raw
                # callers (Tradovate request_with_retry, projectx order_router.cancel)
                # record success themselves via _record_response_success() helper.
                return resp

            if cls == "F":
                # 401 — try refresh once, then retry once.
                if refresh_used or self.refresh_token is None or attempts_remaining <= 0:
                    self.failure_hook.record_failure("F")
                    raise BrokerAuthError(
                        f"{self.name}: 401 on {path} after refresh attempt",
                        error_class="F",
                        last_response=resp,
                    )
                refresh_used = True
                log.warning("%s: 401 on %s — invoking token refresh", self.name, path)
                try:
                    self.refresh_token()
                except Exception as exc:
                    self.failure_hook.record_failure("F")
                    raise BrokerAuthError(
                        f"{self.name}: refresh_token raised on 401 retry: {exc}",
                        error_class="F",
                        last_response=resp,
                    ) from exc
                continue

            if cls == "E":
                attempts_429 -= 1
                if attempts_429 < 0 or attempts_remaining <= 0:
                    self.failure_hook.record_failure("E")
                    raise BrokerRateLimitExhausted(
                        f"{self.name}: 429 exhausted on {path}",
                        error_class="E",
                        last_response=resp,
                    )
                wait = _parse_retry_after(resp, _backoff_wait(attempt - 1, policy))
                self.failure_hook.record_failure("E")
                log.warning("%s: 429 on %s — retry-after %.1fs", self.name, path, wait)
                _sleep_bounded(wait, deadline)
                continue

            # cls == "D" — 5xx
            self.failure_hook.record_failure("D")
            if attempts_remaining <= 0 or time.monotonic() >= deadline:
                raise BrokerTransientError(
                    f"{self.name}: 5xx {resp.status_code} on {path} after {attempt} attempts",
                    error_class="D",
                    last_response=resp,
                )
            wait = _backoff_wait(attempt - 1, policy)
            log.warning(
                "%s: HTTP %d on %s (attempt %d/%d) — retrying in %.1fs",
                self.name,
                resp.status_code,
                path,
                attempt,
                policy.max_attempts,
                wait,
            )
            _sleep_bounded(wait, deadline)
            continue

    def post_json(
        self,
        path: str,
        headers: dict,
        body: dict,
        policy: RetryPolicy = READ_POLICY,
        *,
        timeout: float = 10.0,
    ) -> dict:
        """POST JSON body, return parsed JSON. Raises BrokerProtocolError on
        non-JSON response or {"success": false} payload (success-field is
        ProjectX/Tradovate convention)."""
        resp = self.request("POST", path, headers=headers, json=body, timeout=timeout, policy=policy)
        try:
            data = _parse_json_or_protocol(self.name, path, resp)
        except BrokerProtocolError:
            self.failure_hook.record_failure("G")
            raise
        self.failure_hook.record_success()
        return data

    def get_json(
        self,
        path: str,
        headers: dict,
        params: dict | None = None,
        policy: RetryPolicy = READ_POLICY,
        *,
        timeout: float = 10.0,
    ) -> dict:
        resp = self.request("GET", path, headers=headers, params=params, timeout=timeout, policy=policy)
        try:
            data = _parse_json_or_protocol(self.name, path, resp)
        except BrokerProtocolError:
            self.failure_hook.record_failure("G")
            raise
        self.failure_hook.record_success()
        return data

    def record_response_success(self) -> None:
        """Raw callers (Tradovate request_with_retry, projectx fast-path) call this
        after a returned response has been validated at the caller's layer.
        Mirrors what post_json/get_json do internally."""
        self.failure_hook.record_success()


def _parse_json_or_protocol(name: str, path: str, resp: requests.Response) -> dict:
    try:
        data = resp.json()
    except ValueError as exc:
        raise BrokerProtocolError(
            f"{name}: non-JSON response on {path}: {resp.text[:200]}",
            error_class="G",
            last_response=resp,
        ) from exc
    if isinstance(data, dict) and data.get("success") is False:
        raise BrokerProtocolError(
            f"{name}: success=false on {path}: {data.get('errorMessage', data)}",
            error_class="G",
            last_response=resp,
        )
    return data


def _sleep_bounded(seconds: float, deadline: float) -> None:
    """Sleep at most until `deadline`, never longer than requested."""
    now = time.monotonic()
    remaining = max(0.0, deadline - now)
    time.sleep(min(seconds, remaining))


# ─── Equity reading with age (for Stage 3 broker-health-tick) ──────────────


@dataclass(frozen=True)
class EquityReading:
    """Equity snapshot with age-since-broker-acknowledged.

    Stage 3 orchestrator consumes this to gate the kill-switch SLA.
    """

    value: float | None
    age_s: float
    source: str  # "live" | "cache" | "missing"
