"""Tests for webhook server fail-closed secret validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_webhook_rejects_wrong_secret():
    """Requests with wrong secret get 403."""
    # Patch the module-level secret so lifespan won't raise
    with patch.dict("os.environ", {"WEBHOOK_SECRET": "correct-secret"}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = "correct-secret"

        from fastapi.testclient import TestClient

        client = TestClient(ws.app, raise_server_exceptions=False)

        resp = client.post(
            "/trade",
            json={
                "instrument": "MGC",
                "direction": "long",
                "action": "entry",
                "qty": 1,
                "secret": "wrong-secret",
            },
        )
        assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_webhook_health_endpoint():
    """Health endpoint returns 200."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": "test-secret"}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = "test-secret"

        from fastapi.testclient import TestClient

        client = TestClient(ws.app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


def test_webhook_lifespan_blocks_empty_secret():
    """Server refuses to start when WEBHOOK_SECRET is empty."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": ""}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = ""

        from fastapi.testclient import TestClient

        with pytest.raises(RuntimeError, match="WEBHOOK_SECRET env var is required"):
            with TestClient(ws.app):
                pass  # should never reach here
