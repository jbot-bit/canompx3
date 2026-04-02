"""Shared HTTP utilities for Tradovate API calls.

Rate-limit retry with exponential backoff, used by all Tradovate modules.
"""

import logging
import random
import time

import requests

log = logging.getLogger(__name__)

_429_MAX_RETRIES = 3
_429_BACKOFF_BASE = 1.0
_429_BACKOFF_MAX = 30.0
_429_JITTER_FACTOR = 0.2


class RateLimitExhausted(Exception):
    """Raised when 429 retries are exhausted."""

    pass


def _backoff_wait(attempt: int) -> float:
    base_wait = min(_429_BACKOFF_BASE * (2**attempt), _429_BACKOFF_MAX)
    jitter = base_wait * _429_JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.1, base_wait + jitter)


def request_with_retry(
    method: str,
    url: str,
    headers: dict,
    json_body: dict | None = None,
    timeout: float = 10,
) -> requests.Response:
    """HTTP request with 429 rate-limit retry and exponential backoff."""
    func = requests.post if method == "POST" else requests.get
    kwargs: dict = {"headers": headers, "timeout": timeout}
    if json_body is not None:
        kwargs["json"] = json_body
    for attempt in range(_429_MAX_RETRIES + 1):
        resp = func(url, **kwargs)
        if resp.status_code != 429:
            return resp
        if attempt < _429_MAX_RETRIES:
            wait = _backoff_wait(attempt)
            log.warning("HTTP 429 on %s (attempt %d) — retrying in %.1fs", url.split("/")[-1], attempt + 1, wait)
            time.sleep(wait)
    raise RateLimitExhausted(f"429 exhausted after {_429_MAX_RETRIES + 1} attempts on {url.split('/')[-1]}")
