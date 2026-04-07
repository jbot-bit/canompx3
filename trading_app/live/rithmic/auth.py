"""Rithmic authentication and connection manager.

Manages a persistent async_rithmic RithmicClient in a background thread.
Bridges async library to sync BrokerAuth interface via run_coroutine_threadsafe.

Env vars (all required for live trading):
    RITHMIC_USER         — Rithmic login username
    RITHMIC_PASSWORD     — Rithmic login password
    RITHMIC_SYSTEM_NAME  — Environment name (e.g. "Rithmic Paper Trading", "Rithmic 01")
    RITHMIC_APP_NAME     — 4-char app prefix (assigned after conformance test)
    RITHMIC_APP_VERSION  — App version string (e.g. "1.0.0")
    RITHMIC_GATEWAY      — Gateway URI (e.g. "wss://rituz00100.rithmic.com:443")

Verified against: async_rithmic 1.5.9 source (plants/order.py, client.py, enums.py)
"""

import asyncio
import logging
import os
import threading
from typing import Any

from ..broker_base import BrokerAuth

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 30.0  # seconds to wait for initial connection
_BRIDGE_TIMEOUT = 10.0  # seconds to wait for async→sync bridge calls


class RithmicAuth(BrokerAuth):
    """Rithmic connection-based auth. Manages async_rithmic client in background thread.

    Unlike ProjectX/Tradovate (REST + token), Rithmic uses persistent WebSocket
    connections. Auth is handled at connection time, not per-request.
    """

    def __init__(self):
        # async_rithmic.RithmicClient at runtime. Typed as Any because
        # async_rithmic may not be installed in dev environments (pyright
        # can't resolve the import) and the library has no type stubs.
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._auth_healthy = False

        # Read credentials from env (fail-fast if missing)
        self._user = os.environ.get("RITHMIC_USER", "")
        self._password = os.environ.get("RITHMIC_PASSWORD", "")
        self._system_name = os.environ.get("RITHMIC_SYSTEM_NAME", "Rithmic Paper Trading")
        self._app_name = os.environ.get("RITHMIC_APP_NAME", "CANO")
        self._app_version = os.environ.get("RITHMIC_APP_VERSION", "1.0.0")
        self._gateway = os.environ.get("RITHMIC_GATEWAY", "")

    def _ensure_connected(self) -> None:
        """Lazy-connect on first use. Creates background thread with asyncio event loop."""
        if self._connected and self._client is not None and self._auth_healthy:
            return

        if not self._user or not self._password or not self._gateway:
            raise RuntimeError(
                "Rithmic credentials not configured. Set RITHMIC_USER, RITHMIC_PASSWORD, "
                "RITHMIC_GATEWAY environment variables."
            )

        # Import async_rithmic lazily to avoid import errors when not using Rithmic
        from async_rithmic import OrderPlacement, RithmicClient, SysInfraType

        client = RithmicClient(
            user=self._user,
            password=self._password,
            system_name=self._system_name,
            app_name=self._app_name,
            app_version=self._app_version,
            url=self._gateway,
            manual_or_auto=OrderPlacement.AUTO,
        )

        # Create background event loop in daemon thread
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="rithmic-event-loop",
        )

        # Assign to instance BEFORE starting, so disconnect() can clean up on failure
        self._client = client
        self._loop = loop
        self._thread = thread
        thread.start()

        # Connect ORDER_PLANT + PNL_PLANT (skip TICKER — ProjectX handles data)
        future = asyncio.run_coroutine_threadsafe(
            client.connect(plants=[SysInfraType.ORDER_PLANT, SysInfraType.PNL_PLANT]),
            loop,
        )
        try:
            future.result(timeout=_CONNECT_TIMEOUT)
            self._connected = True
            self._auth_healthy = True
            log.info(
                "Rithmic connected: system=%s gateway=%s accounts=%d",
                self._system_name,
                self._gateway,
                len(client.accounts) if client.accounts else 0,
            )
        except Exception as e:
            # Clean up partially-created resources to prevent leak on retry
            self._auth_healthy = False
            self._connected = False
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
            self._client = None
            self._loop = None
            self._thread = None
            log.critical("Rithmic connection FAILED: %s", e)
            raise RuntimeError(f"Rithmic connection failed: {e}") from e

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread.

        _loop is set in _ensure_connected before this thread is started —
        guaranteed non-None here.
        """
        assert self._loop is not None, "RithmicAuth._run_loop called before _loop assigned"
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def client(self) -> Any:
        """Access the underlying RithmicClient. Connects on first use.

        Returns async_rithmic.RithmicClient (guaranteed non-None after
        _ensure_connected returns successfully — raises RuntimeError
        otherwise). Typed Any because async_rithmic has no stubs.
        """
        self._ensure_connected()
        return self._client

    @property
    def is_healthy(self) -> bool:
        return self._auth_healthy

    def run_async(self, coro, timeout: float = _BRIDGE_TIMEOUT):
        """Bridge an async coroutine to sync. Runs in the background event loop.

        This is the primary mechanism for all Rithmic operations:
        the BrokerRouter methods are sync, but async_rithmic is async-only.
        """
        self._ensure_connected()
        if self._loop is None:
            raise RuntimeError("Rithmic event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            self._auth_healthy = False
            raise RuntimeError(f"Rithmic async bridge timed out after {timeout}s") from None
        except Exception as e:
            self._auth_healthy = False
            log.error("Rithmic async bridge error: %s", e)
            raise

    def get_token(self) -> str:
        """BrokerAuth interface compliance. Rithmic doesn't use bearer tokens.

        Returns a placeholder — auth is handled at WebSocket connection level.
        Ensures connection is established on first call.
        """
        self._ensure_connected()
        return "rithmic-connection-based"

    def headers(self) -> dict:
        """BrokerAuth interface compliance. Rithmic doesn't use HTTP headers.

        Returns empty dict — all communication is via Protocol Buffer WebSocket.
        """
        return {}

    def disconnect(self) -> None:
        """Gracefully disconnect from Rithmic and stop the background event loop.

        Important for prop firms where connection limits apply (Bulenox: 3 simultaneous).
        Ghost sessions from ungraceful disconnects could consume account slots.
        """
        if self._client is not None and self._connected and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.disconnect(), self._loop
                )
                future.result(timeout=5.0)
            except Exception as e:
                log.warning("Rithmic disconnect error (non-fatal): %s", e)
            finally:
                self._connected = False
                self._auth_healthy = False

        if self._loop is not None:
            try:
                if self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass  # Loop already closed/stopped — safe to ignore

    def refresh_if_needed(self) -> None:
        """Check connection health. Reconnect is handled by async_rithmic auto-reconnect.

        The library has built-in reconnection with linear backoff (10s, 20s, ...120s).
        This method just ensures we're connected.
        """
        if not self._connected or not self._auth_healthy:
            self._ensure_connected()
