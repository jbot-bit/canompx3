"""CSRF middleware tests — OriginAllowlistMiddleware on /api/action/kill.

Nine cases covering the allow/block decision tree:
  1. GET with no Origin passes (safe method)
  2. POST same-origin (http://localhost:8080) passes
  3. POST cross-origin (http://evil.example) blocked 403
  4. POST no-Origin no-Referer blocked 403 when PYTEST_CURRENT_TEST is unset
  5. POST no-Origin allowed when PYTEST_CURRENT_TEST is set (normal pytest env)
  6. POST Referer-only same-origin passes (Origin absent, Referer present + matching)
  7. POST cross-origin Referer blocked 403 (Origin absent, Referer present but wrong)
  8. PUT cross-origin blocked 403 (non-POST mutating method also gated)
  9. POST 127.0.0.1 origin passes (loopback variant in allowed set)
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import trading_app.live.bot_dashboard as dashboard


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(dashboard, "STOP_FILE", tmp_path / "live_session.stop")
    return TestClient(dashboard.app, raise_server_exceptions=False)


# ── Case 1: safe method (GET) passes without any Origin ──────────────────────

def test_get_no_origin_passes(client: TestClient) -> None:
    resp = client.get("/api/action/kill")
    assert resp.status_code != 403


# ── Case 2: POST same-origin passes ──────────────────────────────────────────

def test_post_same_origin_passes(client: TestClient) -> None:
    resp = client.post(
        "/api/action/kill",
        headers={"Origin": f"http://localhost:{dashboard.PORT}"},
    )
    assert resp.status_code != 403


# ── Case 3: POST cross-origin blocked ────────────────────────────────────────

def test_post_cross_origin_blocked(client: TestClient) -> None:
    resp = client.post(
        "/api/action/kill",
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403


# ── Case 4: POST no-Origin no-Referer blocked in prod-like env ───────────────

def test_post_no_origin_no_referer_blocked_in_prod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dashboard, "STOP_FILE", tmp_path / "live_session.stop")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    # Build a fresh client after env mutation so middleware sees the change
    prod_client = TestClient(dashboard.app, raise_server_exceptions=False)
    resp = prod_client.post("/api/action/kill")
    assert resp.status_code == 403


# ── Case 5: POST no-Origin allowed when PYTEST_CURRENT_TEST is set ───────────

def test_post_no_origin_allowed_under_pytest(client: TestClient) -> None:
    # PYTEST_CURRENT_TEST is set automatically by pytest — standard TestClient
    # calls without an Origin header must still reach the route.
    assert os.environ.get("PYTEST_CURRENT_TEST"), "must run under pytest for this case"
    resp = client.post("/api/action/kill")
    assert resp.status_code != 403


# ── Case 6: Referer-only same-origin passes ──────────────────────────────────

def test_post_referer_only_same_origin_passes(client: TestClient) -> None:
    resp = client.post(
        "/api/action/kill",
        headers={"Referer": f"http://localhost:{dashboard.PORT}/"},
    )
    assert resp.status_code != 403


# ── Case 7: Cross-origin Referer blocked ─────────────────────────────────────

def test_post_cross_origin_referer_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dashboard, "STOP_FILE", tmp_path / "live_session.stop")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    prod_client = TestClient(dashboard.app, raise_server_exceptions=False)
    resp = prod_client.post(
        "/api/action/kill",
        headers={"Referer": "http://evil.example/page"},
    )
    assert resp.status_code == 403


# ── Case 8: PUT cross-origin blocked (non-POST mutating method) ───────────────

def test_put_cross_origin_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dashboard, "STOP_FILE", tmp_path / "live_session.stop")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    prod_client = TestClient(dashboard.app, raise_server_exceptions=False)
    resp = prod_client.put(
        "/api/action/kill",
        headers={"Origin": "http://evil.example"},
    )
    assert resp.status_code == 403


# ── Case 9: POST 127.0.0.1 origin passes (loopback variant) ──────────────────

def test_post_127_origin_passes(client: TestClient) -> None:
    resp = client.post(
        "/api/action/kill",
        headers={"Origin": f"http://127.0.0.1:{dashboard.PORT}"},
    )
    assert resp.status_code != 403
