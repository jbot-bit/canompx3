"""Shared HTTP utilities for Tradovate API calls.

Thin compatibility shim over trading_app.live.http_client.BrokerHTTPClient.
All retry/classification logic now lives in the canonical client. The two
exported names (request_with_retry, RateLimitExhausted) are preserved for
the existing Tradovate call sites (order_router, positions, contracts).
"""

import logging

import requests

from ..http_client import (
    READ_POLICY,
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
) -> requests.Response:
    """HTTP request with classified retry. Compatibility wrapper.

    Internally delegates to a per-call BrokerHTTPClient configured with
    READ_POLICY. Callers should migrate to BrokerHTTPClient directly for
    new code paths.
    """
    # Tradovate URLs are always absolute; base_url is unused at this seam.
    client = BrokerHTTPClient(base_url="", name="tradovate")
    return client.request(
        method,
        url,
        headers=headers,
        json=json_body,
        timeout=timeout,
        policy=READ_POLICY,
    )
