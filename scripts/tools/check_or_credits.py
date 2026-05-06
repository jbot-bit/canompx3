"""Advisory OpenRouter credit-balance check for the OpenCode launcher.

Calls ``GET https://openrouter.ai/api/v1/auth/key`` (the conventional
endpoint that returns ``{ data: { usage, limit, label, is_free_tier,
rate_limit } }``). Prints the remaining credit balance and emits a WARN
to stderr when below the threshold (default $5). **Always exits 0 —
advisory only**; the launcher must keep working when credits are tight
or the endpoint is down.

Per the spec doc § Maintenance, the endpoint URL must be falsified via
``curl -H "Authorization: Bearer $OPENROUTER_API_KEY"
https://openrouter.ai/api/v1/auth/key`` before this script is promoted
from opt-in (``OPENCODE_AGENT_CHECK_CREDITS=1``) to default-on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

OR_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"
DEFAULT_THRESHOLD_USD = 5.0
HTTP_TIMEOUT_SECONDS = 5.0

_MOCK_NORMAL_PAYLOAD = {
    "data": {
        "label": "mock-key",
        "usage": 12.34,
        "limit": 100.00,
        "is_free_tier": False,
        "rate_limit": {"requests": 200, "interval": "10s"},
    }
}

_MOCK_LOW_PAYLOAD = {
    "data": {
        "label": "mock-key",
        "usage": 96.50,
        "limit": 100.00,
        "is_free_tier": False,
        "rate_limit": {"requests": 200, "interval": "10s"},
    }
}


def _fetch_live(api_key: str) -> dict | None:
    req = urllib.request.Request(
        OR_AUTH_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print(
            f"[check_or_credits] WARN: HTTP {exc.code} from {OR_AUTH_KEY_URL}: {exc.reason}",
            file=sys.stderr,
        )
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"[check_or_credits] WARN: network error contacting {OR_AUTH_KEY_URL}: {exc}",
            file=sys.stderr,
        )
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        print(f"[check_or_credits] WARN: parse error: {exc}", file=sys.stderr)
        return None


def _emit_balance(payload: dict, threshold: float) -> None:
    data = payload.get("data") or {}
    usage = data.get("usage")
    limit = data.get("limit")
    label = data.get("label", "?")
    if usage is None or limit is None:
        print(
            f"[check_or_credits] WARN: response missing usage/limit fields: {payload!r}",
            file=sys.stderr,
        )
        return
    remaining = float(limit) - float(usage)
    free_tier = data.get("is_free_tier", False)
    suffix = " (free tier)" if free_tier else ""
    print(
        f"[check_or_credits] OpenRouter key={label} usage=${usage:.2f} "
        f"limit=${limit:.2f} remaining=${remaining:.2f}{suffix}"
    )
    if remaining < threshold:
        print(
            f"[check_or_credits] WARN: remaining credit ${remaining:.2f} "
            f"is below threshold ${threshold:.2f} — refill at https://openrouter.ai/credits",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a mocked OpenRouter response (for tests).",
    )
    parser.add_argument(
        "--mock-low",
        action="store_true",
        help="Mock variant: usage near limit (forces WARN).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_USD,
        help=f"USD remaining-balance WARN threshold (default {DEFAULT_THRESHOLD_USD}).",
    )
    args = parser.parse_args()

    if args.mock or args.mock_low:
        payload = _MOCK_LOW_PAYLOAD if args.mock_low else _MOCK_NORMAL_PAYLOAD
        _emit_balance(payload, args.threshold)
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print(
            "[check_or_credits] WARN: OPENROUTER_API_KEY not set; cannot check credits.",
            file=sys.stderr,
        )
        return 0

    payload = _fetch_live(api_key)
    if payload is None:
        return 0  # Advisory: endpoint failures are non-fatal.
    _emit_balance(payload, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
