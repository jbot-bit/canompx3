"""Stage 2 tests — order idempotency + post-failure reconcile.

Scenarios:
  1. Place succeeds first try — one journal entry, one broker call.
  2. Place 429 then 200 — same customTag on both attempts (idempotency held).
  3. Place transient failure after broker accept — reconcile fingerprint
     locates the open order and adopts its id (no duplicate placement).
  4. Place transient failure before broker accept — reconcile finds nothing,
     original transient error surfaces, caller can retry safely.

Reference: ProjectX API spec (resources/projectx_api_spec_2026_05_16.md) —
customTag unique-per-account-forever; searchOpen does NOT return customTag,
so reconcile is fingerprint-based.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from trading_app.live.http_client import BrokerHTTPClient, BrokerTransientError
from trading_app.live.projectx.order_router import ProjectXOrderRouter


def _ok(json_body):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.headers = {}
    r.json.return_value = json_body
    r.text = str(json_body)
    return r


def _status(code, headers=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = code
    r.headers = headers or {}
    r.json.side_effect = ValueError("no body")
    r.text = ""
    return r


class _Scripted:
    """Returns / raises items in order; records calls."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._script:
            raise AssertionError(f"unexpected extra call: {method} {url}")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _build_router(script, *, base_url="https://api.test"):
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer t"}
    auth.refresh_if_needed = MagicMock()
    session = _Scripted(script)
    router = ProjectXOrderRouter(account_id=12345, auth=auth, tick_size=0.10)
    router._http = BrokerHTTPClient(
        base_url=base_url,
        refresh_token=auth.refresh_if_needed,
        session=session,
        name="projectx-orders-test",
    )
    return router, session


def _spec_long_market(symbol="MGCZ5", qty=1):
    return {
        "accountId": 12345,
        "contractId": symbol,
        "type": 2,  # Market
        "side": 0,  # Bid
        "size": qty,
        "_intent": {"direction": "long"},
    }


# ─── Scenario 1 — happy path ───────────────────────────────────────────────


def test_place_succeeds_first_try():
    router, session = _build_router([
        _ok({"success": True, "orderId": 4242}),
    ])
    out = router.submit(_spec_long_market())
    assert out["order_id"] == 4242
    assert out["status"] == "submitted"
    assert out["client_order_id"]  # non-empty
    assert len(session.calls) == 1
    sent_body = session.calls[0]["json"]
    assert sent_body["customTag"] == out["client_order_id"]


# ─── Scenario 2 — 429 then 200, same customTag preserved ───────────────────


def test_place_429_then_success_uses_same_customtag():
    router, session = _build_router([
        _status(429, headers={"Retry-After": "0.01"}),
        _ok({"success": True, "orderId": 4243}),
    ])
    out = router.submit(_spec_long_market())
    assert out["order_id"] == 4243
    assert len(session.calls) == 2
    # Idempotency held — both POSTs carry the same customTag.
    tags = [c["json"]["customTag"] for c in session.calls]
    assert tags[0] == tags[1]
    assert tags[0] == out["client_order_id"]


# ─── Scenario 3 — RST after server accept, reconcile finds the open order ──


def test_place_rst_after_accept_reconciles_via_fingerprint():
    """After place attempts exhaust, reconcile queries searchOpen and finds
    an order matching the (contract, side, size, type) fingerprint that was
    created within the reconcile window. Adopt its id; no duplicate place."""
    rst = requests.ConnectionError("rst")
    rst.__cause__ = ConnectionResetError(10054, "forcibly closed")

    # All 4 ORDER_POLICY attempts fail with RST, then the reconcile path runs
    # which calls searchOpen and finds the pre-existing order.
    searchopen_resp = _ok({
        "orders": [
            {
                "id": 9999,
                "contractId": "MGCZ5",
                "side": 0,
                "size": 1,
                "type": 2,
            }
        ]
    })
    router, session = _build_router([rst, rst, rst, rst, searchopen_resp])

    out = router.submit(_spec_long_market())
    assert out["order_id"] == 9999
    assert out["status"] == "submitted_reconciled"
    # 4 place attempts + 1 reconcile (searchOpen)
    assert len(session.calls) == 5


# ─── Scenario 4 — RST before server accept, reconcile finds nothing ────────


def test_place_rst_before_accept_no_reconcile_raises_transient():
    """Same RST pattern, but searchOpen returns empty — no fingerprint match.
    Original BrokerTransientError must bubble up so the caller can retry
    SAFELY (the broker definitively never received the place)."""
    rst = requests.ConnectionError("rst")
    rst.__cause__ = ConnectionResetError(10054, "forcibly closed")

    router, session = _build_router([rst, rst, rst, rst, _ok({"orders": []})])

    with pytest.raises(BrokerTransientError) as exc:
        router.submit(_spec_long_market())
    assert exc.value.error_class == "C"
    assert len(session.calls) == 5


# ─── Scenario 5 — Reconcile refuses ambiguous fingerprint ──────────────────


def test_place_rst_ambiguous_fingerprint_refuses_to_adopt():
    """If two open orders match the fingerprint, we cannot tell which is ours;
    refuse to adopt and surface the original transient error."""
    rst = requests.ConnectionError("rst")
    rst.__cause__ = ConnectionResetError(10054, "forcibly closed")
    searchopen_resp = _ok({
        "orders": [
            {"id": 9999, "contractId": "MGCZ5", "side": 0, "size": 1, "type": 2},
            {"id": 10000, "contractId": "MGCZ5", "side": 0, "size": 1, "type": 2},
        ]
    })
    router, session = _build_router([rst, rst, rst, rst, searchopen_resp])

    with pytest.raises(BrokerTransientError):
        router.submit(_spec_long_market())


# ─── customTag generator stays unique ──────────────────────────────────────


def test_client_order_ids_are_unique():
    from trading_app.live.projectx.order_router import generate_client_order_id
    ids = {generate_client_order_id() for _ in range(1000)}
    assert len(ids) == 1000
