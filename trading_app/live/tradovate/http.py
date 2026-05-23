"""Shared HTTP utilities for Tradovate API calls.

Thin compatibility shim over trading_app.live.http_client.BrokerHTTPClient.
All retry/classification logic now lives in the canonical client. The two
exported names (request_with_retry, RateLimitExhausted) are preserved for
the existing Tradovate call sites (order_router, positions, contracts).
"""

import logging

import requests

from ..http_client import (
    ORDER_POLICY,
    READ_POLICY,
    RetryPolicy,
    BrokerHTTPClient,
    BrokerRateLimitExhausted,
)

log = logging.getLogger(__name__)


# Re-export under the legacy name so existing imports continue to work.
# `isinstance(exc, RateLimitExhausted)` still resolves correctly for callers.
RateLimitExhausted = BrokerRateLimitExhausted


def request_with_retry(
    method: str,
    url: str,
    headers: dict,
    json_body: dict | None = None,
    timeout: float = 10,
    failure_hook: object | None = None,
    policy: RetryPolicy = READ_POLICY,
) -> requests.Response:
    """HTTP request with classified retry. Compatibility wrapper.

    Internally delegates to a per-call BrokerHTTPClient configured with the
    given ``policy`` (default READ_POLICY for backwards-compat).

    Order-mutating callers (submit, cancel) must pass ORDER_POLICY explicitly
    to cap retries at 4 attempts / 10 s and avoid duplicate-order risk on
    transient failures. See PR301-TRADO-IDEMPOTENCY (ralph iter 205).

    ``failure_hook`` (Stage 4): orchestrator-wired CircuitBreaker. When the
    caller has access to ``auth.failure_hook`` (set by SessionOrchestrator),
    pass it through; otherwise the client uses _NoopFailureHook.
    """
    # Tradovate URLs are always absolute; base_url is unused at this seam.
    kwargs_for_client: dict = {"base_url": "", "name": "tradovate"}
    if failure_hook is not None:
        kwargs_for_client["failure_hook"] = failure_hook
    client = BrokerHTTPClient(**kwargs_for_client)
    resp = client.request(
        method,
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
        policy=policy,
    )
    # request() no longer records success at the HTTP layer (deferred to parse).
    # Raw callers like this shim must record success themselves; the caller's
    # resp.raise_for_status() + resp.json() happens after this returns and any
    # protocol error there raises, never reaching the CB success path.
    client.record_response_success()
    return resp
