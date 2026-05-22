"""Tests for trading_app.live.http_client.BrokerHTTPClient.

Covers the failure-mode taxonomy A-G plus deadline propagation and 401-refresh
path. Uses a stub requests.Session injected at construction — no `responses`
library required.

Reference: docs/specs/live_execution_resilience.md § Failure-Mode Taxonomy
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

import pytest
import requests

from trading_app.live.http_client import (
    AUTH_POLICY,
    ORDER_POLICY,
    READ_POLICY,
    BrokerAuthError,
    BrokerHTTPClient,
    BrokerPermanentError,
    BrokerProtocolError,
    BrokerRateLimitExhausted,
    BrokerTransientError,
    RetryPolicy,
)


def _make_resp(status: int, json_body: dict | list | None = None, *, headers: dict | None = None) -> MagicMock:
    """Build a MagicMock that quacks like requests.Response."""
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    if json_body is None:
        r.json.side_effect = ValueError("no body")
        r.text = ""
    else:
        r.json.return_value = json_body
        r.text = str(json_body)
    return r


class _ScriptedSession:
    """Stub session that returns / raises per a scripted sequence."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls: list[dict] = []

    def request(self, method: str, url: str, **kwargs):  # noqa: D401
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._script:
            raise AssertionError(f"Unexpected extra call: {method} {url}")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def fast_policy():
    """Tight policy so tests don't burn real wall-clock on backoffs."""
    return RetryPolicy(
        max_attempts=4,
        base_backoff_s=0.001,
        backoff_factor=1.5,
        backoff_max_s=0.01,
        jitter_factor=0.0,
        deadline_s=2.0,
        max_429_attempts=2,
    )


# ─── Class A — DNS / connect ───────────────────────────────────────────────


def test_class_a_dns_retries_then_success(fast_policy):
    session = _ScriptedSession(
        [
            requests.ConnectionError("name resolution failed"),
            _make_resp(200, {"ok": True}),
        ]
    )
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}
    assert len(session.calls) == 2


def test_class_a_dns_exhaustion_raises_transient(fast_policy):
    err = requests.ConnectionError("nope")
    err.__cause__ = socket.gaierror("dns")
    session = _ScriptedSession([err, err, err, err])
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerTransientError) as exc:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert exc.value.error_class == "A"


# ─── Class B — read timeout ────────────────────────────────────────────────


def test_class_b_timeout_retries(fast_policy):
    session = _ScriptedSession(
        [
            requests.Timeout("read timeout"),
            _make_resp(200, {"ok": True}),
        ]
    )
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}


def test_class_b_deadline_budget_refuses_after_budget():
    """3 timeouts × ~1s budget → eventually deadline exceeded."""
    policy = RetryPolicy(
        max_attempts=10,
        base_backoff_s=0.5,
        backoff_factor=1.0,
        backoff_max_s=0.5,
        jitter_factor=0.0,
        deadline_s=0.1,  # very tight
        max_429_attempts=10,
    )
    session = _ScriptedSession([requests.Timeout("t")] * 10)
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerTransientError) as exc:
        client.post_json("/p", {}, {}, policy=policy)
    assert exc.value.error_class in ("B", "B")  # exhaustion or deadline


# ─── Class C — TCP RST ─────────────────────────────────────────────────────


def test_class_c_tcp_rst_retries_then_success(fast_policy):
    err = requests.ConnectionError("rst")
    err.__cause__ = ConnectionResetError(10054, "forcibly closed by remote host")
    session = _ScriptedSession([err, _make_resp(200, {"ok": True})])
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}


def test_class_c_classification_is_c_on_exhaustion(fast_policy):
    err = requests.ConnectionError("rst")
    err.__cause__ = ConnectionResetError(10054, "rst")
    session = _ScriptedSession([err] * 4)
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerTransientError) as exc:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert exc.value.error_class == "C"


# ─── Class D — 5xx ─────────────────────────────────────────────────────────


def test_class_d_5xx_retries_then_success(fast_policy):
    session = _ScriptedSession(
        [
            _make_resp(503),
            _make_resp(200, {"ok": True}),
        ]
    )
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}


def test_class_d_5xx_exhaustion(fast_policy):
    session = _ScriptedSession([_make_resp(500)] * 4)
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerTransientError) as exc:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert exc.value.error_class == "D"


# ─── Class E — 429 ─────────────────────────────────────────────────────────


def test_class_e_429_respects_retry_after_and_succeeds(fast_policy):
    session = _ScriptedSession(
        [
            _make_resp(429, headers={"Retry-After": "0.01"}),
            _make_resp(200, {"ok": True}),
        ]
    )
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}


def test_class_e_429_exhaustion(fast_policy):
    session = _ScriptedSession([_make_resp(429)] * 5)
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerRateLimitExhausted) as exc:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert exc.value.error_class == "E"


# ─── Class F — 401 refresh-and-retry ───────────────────────────────────────


def test_class_f_401_invokes_refresh_then_retries(fast_policy):
    refresh = MagicMock()
    session = _ScriptedSession(
        [
            _make_resp(401),
            _make_resp(200, {"ok": True}),
        ]
    )
    client = BrokerHTTPClient(
        base_url="https://x.test",
        session=session,
        refresh_token=refresh,
        name="t",
    )
    data = client.post_json("/p", {}, {}, policy=fast_policy)
    assert data == {"ok": True}
    refresh.assert_called_once()


def test_class_f_401_without_refresh_raises_auth_error(fast_policy):
    session = _ScriptedSession([_make_resp(401)])
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerAuthError) as exc:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert exc.value.error_class == "F"


def test_class_f_401_double_failure_raises_auth_error(fast_policy):
    """Refresh callback succeeds but 401 again — still an auth error."""
    refresh = MagicMock()
    session = _ScriptedSession([_make_resp(401), _make_resp(401)])
    client = BrokerHTTPClient(
        base_url="https://x.test",
        session=session,
        refresh_token=refresh,
        name="t",
    )
    with pytest.raises(BrokerAuthError):
        client.post_json("/p", {}, {}, policy=fast_policy)


# ─── Class G — malformed JSON / success=false ──────────────────────────────


def test_class_g_non_json_raises_protocol(fast_policy):
    session = _ScriptedSession([_make_resp(200)])  # json() raises ValueError
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerProtocolError):
        client.post_json("/p", {}, {}, policy=fast_policy)


def test_class_g_success_false_raises_protocol(fast_policy):
    session = _ScriptedSession(
        [
            _make_resp(200, {"success": False, "errorMessage": "bad payload"}),
        ]
    )
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerProtocolError):
        client.post_json("/p", {}, {}, policy=fast_policy)


# ─── 4xx permanent ─────────────────────────────────────────────────────────


def test_4xx_permanent_no_retry(fast_policy):
    session = _ScriptedSession([_make_resp(404)])
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t")
    with pytest.raises(BrokerPermanentError):
        client.post_json("/p", {}, {}, policy=fast_policy)
    # Only one call — no retry on permanent.
    assert len(session.calls) == 1


# ─── Policy defaults stay constructible ────────────────────────────────────


def test_policy_defaults_construct():
    assert READ_POLICY.max_attempts == 5
    assert ORDER_POLICY.max_attempts == 4
    assert AUTH_POLICY.max_attempts == 3


# ─── Stage 4 — failure_hook receives the classified error letter ───────────


class _RecordingHook:
    """Hook stub that captures every call so we can assert on (class, count)."""

    def __init__(self):
        self.failures: list[str] = []
        self.successes: int = 0
        self.allow: bool = True

    def record_failure(self, error_class: str) -> None:
        self.failures.append(error_class)

    def record_success(self) -> None:
        self.successes += 1

    def should_allow_request(self) -> bool:
        return self.allow


def test_failure_hook_receives_class_d_on_5xx():
    """5xx storm — hook records 'D' for each attempt before exhaustion."""
    session = _ScriptedSession([_make_resp(503), _make_resp(503)])
    hook = _RecordingHook()
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t", failure_hook=hook)
    with pytest.raises(BrokerTransientError):
        client.request("POST", "/p", policy=RetryPolicy(max_attempts=2, deadline_s=2.0, base_backoff_s=0.01))
    # Each 5xx response invokes record_failure("D"); two attempts → two D's.
    assert hook.failures == ["D", "D"]
    assert hook.successes == 0


def test_failure_hook_receives_class_b_on_timeout():
    """Read-timeout (class B) propagates to the hook."""
    session = _ScriptedSession([requests.Timeout("slow"), _make_resp(200, {"ok": True})])
    hook = _RecordingHook()
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t", failure_hook=hook)
    client.post_json("/p", {}, {}, policy=RetryPolicy(max_attempts=2, deadline_s=2.0, base_backoff_s=0.01))
    # First attempt is class-B timeout, then success.
    assert "B" in hook.failures
    assert hook.successes == 1


def test_failure_hook_should_allow_request_blocks_call(fast_policy):
    """When hook.should_allow_request() is False, client raises before any HTTP."""
    session = _ScriptedSession([])  # no responses scripted — should never be called
    hook = _RecordingHook()
    hook.allow = False
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t", failure_hook=hook)
    with pytest.raises(BrokerTransientError) as excinfo:
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert "circuit breaker open" in str(excinfo.value).lower()
    # No HTTP attempted.
    assert session.calls == []


def test_failure_hook_records_class_g_on_success_false(fast_policy):
    """200 + success=false body → hook records class-G failure, not success."""
    session = _ScriptedSession(
        [_make_resp(200, {"success": False, "errorMessage": "bad payload"})]
    )
    hook = _RecordingHook()
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t", failure_hook=hook)
    with pytest.raises(BrokerProtocolError):
        client.post_json("/p", {}, {}, policy=fast_policy)
    assert hook.failures == ["G"]
    assert hook.successes == 0


def test_failure_hook_records_class_g_on_non_json(fast_policy):
    """200 with non-JSON body → hook records class-G failure, not success."""
    session = _ScriptedSession([_make_resp(200)])  # json() raises ValueError
    hook = _RecordingHook()
    client = BrokerHTTPClient(base_url="https://x.test", session=session, name="t", failure_hook=hook)
    with pytest.raises(BrokerProtocolError):
        client.get_json("/p", {}, policy=fast_policy)
    assert hook.failures == ["G"]
    assert hook.successes == 0
