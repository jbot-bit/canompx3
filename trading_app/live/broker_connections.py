"""Broker connection manager — persistent credential store with runtime auth lifecycle.

Stores connections in data/broker_connections.json (gitignored, localhost-only).
Falls back to .env vars if no connections file exists (backward compatible).
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_CONNECTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "broker_connections.json"

BROKER_TYPES = {
    "projectx": {
        "display": "TopStepX (ProjectX)",
        "fields": [
            {"key": "username", "label": "Username / Email", "type": "text", "required": True},
            {"key": "api_key", "label": "API Key", "type": "password", "required": True},
            {
                "key": "base_url",
                "label": "Base URL",
                "type": "text",
                "required": False,
                "default": "https://api.thefuturesdesk.projectx.com",
            },
        ],
    },
    "tradovate": {
        "display": "Tradovate",
        "fields": [
            {"key": "user", "label": "Username", "type": "text", "required": True},
            {"key": "password", "label": "Password", "type": "password", "required": True},
            {"key": "cid", "label": "Client ID (CID)", "type": "text", "required": True},
            {"key": "sec", "label": "API Secret", "type": "password", "required": True},
            {"key": "demo", "label": "Demo Mode", "type": "checkbox", "required": False, "default": "0"},
        ],
    },
    "rithmic": {
        "display": "Rithmic (Bulenox / Elite)",
        "fields": [
            {"key": "user", "label": "Username", "type": "text", "required": True},
            {"key": "password", "label": "Password", "type": "password", "required": True},
            {"key": "gateway", "label": "Gateway URL", "type": "text", "required": True},
            {
                "key": "system_name",
                "label": "System Name",
                "type": "text",
                "required": False,
                "default": "Rithmic Paper Trading",
            },
        ],
    },
}


class ConnectionState:
    __slots__ = ("auth", "status", "last_error", "connected_at", "account_count")

    def __init__(self):
        self.auth: object | None = None
        self.status: str = "disconnected"
        self.last_error: str | None = None
        self.connected_at: str | None = None
        self.account_count: int = 0


class BrokerConnectionManager:
    def __init__(self):
        self._connections: list[dict] = []
        self._states: dict[str, ConnectionState] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if _CONNECTIONS_PATH.exists():
                try:
                    self._connections = json.loads(_CONNECTIONS_PATH.read_text(encoding="utf-8"))
                    log.info("Loaded %d broker connections from file", len(self._connections))
                except Exception as e:
                    log.warning("Failed to load connections: %s — .env fallback", e)
                    self._connections = []
                    self._migrate_from_env()
            else:
                self._migrate_from_env()
            for conn in self._connections:
                if conn["id"] not in self._states:
                    self._states[conn["id"]] = ConnectionState()

    def _migrate_from_env(self) -> None:
        if os.environ.get("PROJECTX_API_KEY"):
            self._connections.append(
                {
                    "id": f"env-projectx-{uuid.uuid4().hex[:6]}",
                    "broker_type": "projectx",
                    "display_name": "TopStepX (from .env)",
                    "enabled": True,
                    "source": "env",
                    "credentials": {
                        "username": os.environ.get("PROJECTX_USERNAME", ""),
                        "api_key": os.environ.get("PROJECTX_API_KEY", ""),
                        "base_url": os.environ.get("PROJECTX_BASE_URL", "https://api.thefuturesdesk.projectx.com"),
                    },
                }
            )
            log.info("Migrated ProjectX connection from .env")
        if os.environ.get("TRADOVATE_CID"):
            self._connections.append(
                {
                    "id": f"env-tradovate-{uuid.uuid4().hex[:6]}",
                    "broker_type": "tradovate",
                    "display_name": "Tradovate (from .env)",
                    "enabled": False,
                    "source": "env",
                    "credentials": {
                        "user": os.environ.get("TRADOVATE_USER", ""),
                        "password": os.environ.get("TRADOVATE_PASS", ""),
                        "cid": os.environ.get("TRADOVATE_CID", ""),
                        "sec": os.environ.get("TRADOVATE_SEC", ""),
                        "demo": os.environ.get("TRADOVATE_DEMO", "0"),
                    },
                }
            )

    def save(self) -> None:
        with self._lock:
            try:
                _CONNECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
                _CONNECTIONS_PATH.write_text(json.dumps(self._connections, indent=2), encoding="utf-8")
            except Exception as e:
                log.warning("Failed to save connections: %s", e)

    def list_connections(self) -> list[dict]:
        result = []
        for conn in self._connections:
            state = self._states.get(conn["id"], ConnectionState())
            result.append(
                {
                    "id": conn["id"],
                    "broker_type": conn["broker_type"],
                    "display_name": conn["display_name"],
                    "enabled": conn.get("enabled", True),
                    "source": conn.get("source", "ui"),
                    "status": state.status,
                    "last_error": state.last_error,
                    "connected_at": state.connected_at,
                    "account_count": state.account_count,
                    "credentials": {k: "***" if v else "" for k, v in conn.get("credentials", {}).items()},
                }
            )
        return result

    def get_enabled_connections(self) -> list[dict]:
        return [c for c in self._connections if c.get("enabled", True)]

    def add_connection(self, broker_type: str, display_name: str, credentials: dict) -> dict:
        if broker_type not in BROKER_TYPES:
            raise ValueError(f"Unknown broker type: {broker_type}")
        for field in BROKER_TYPES[broker_type]["fields"]:
            if field["required"] and not credentials.get(field["key"]):
                raise ValueError(f"Missing required field: {field['label']}")
            if not field["required"] and field["key"] not in credentials:
                credentials[field["key"]] = field.get("default", "")
        conn_id = f"{broker_type}-{uuid.uuid4().hex[:8]}"
        conn = {
            "id": conn_id,
            "broker_type": broker_type,
            "display_name": display_name,
            "enabled": True,
            "credentials": credentials,
            "source": "ui",
        }
        with self._lock:
            self._connections.append(conn)
            self._states[conn_id] = ConnectionState()
        self.save()
        return {"id": conn_id, "broker_type": broker_type, "display_name": display_name}

    def remove_connection(self, conn_id: str) -> bool:
        with self._lock:
            if not any(c["id"] == conn_id for c in self._connections):
                return False
            self._disconnect_auth(conn_id)
            self._connections = [c for c in self._connections if c["id"] != conn_id]
            self._states.pop(conn_id, None)
        self.save()
        return True

    def toggle_connection(self, conn_id: str) -> dict | None:
        with self._lock:
            conn = next((c for c in self._connections if c["id"] == conn_id), None)
            if not conn:
                return None
            conn["enabled"] = not conn.get("enabled", True)
            if not conn["enabled"]:
                self._disconnect_auth(conn_id)
        self.save()
        return {"id": conn_id, "enabled": conn["enabled"]}

    def test_connection(self, broker_type: str, credentials: dict) -> dict:
        try:
            auth = self._create_auth(broker_type, credentials)
            auth.get_token()
            return {"success": True, "message": "Connected successfully"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def connect(self, conn_id: str) -> None:
        conn = next((c for c in self._connections if c["id"] == conn_id), None)
        if not conn:
            raise ValueError(f"Unknown connection: {conn_id}")
        state = self._states.setdefault(conn_id, ConnectionState())
        try:
            auth = self._create_auth(conn["broker_type"], conn["credentials"])
            auth.get_token()
            state.auth = auth
            state.status = "connected"
            state.last_error = None
            state.connected_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            log.info("Connected: %s (%s)", conn["display_name"], conn["broker_type"])
        except Exception as e:
            state.auth = None
            state.status = "error"
            state.last_error = str(e)
            raise

    def connect_all_enabled(self) -> None:
        for conn in self._connections:
            if conn.get("enabled", True):
                try:
                    self.connect(conn["id"])
                except Exception as e:
                    log.warning(
                        "connect_all_enabled: failed to connect '%s' (%s): %s",
                        conn.get("display_name", conn["id"]),
                        conn.get("broker_type", "?"),
                        e,
                    )

    def get_auth(self, conn_id: str):
        state = self._states.get(conn_id)
        return state.auth if state else None

    def update_account_count(self, conn_id: str, count: int) -> None:
        state = self._states.get(conn_id)
        if state:
            state.account_count = count

    def _create_auth(self, broker_type: str, credentials: dict):
        if broker_type == "projectx":
            if credentials.get("username"):
                os.environ["PROJECTX_USERNAME"] = credentials["username"]
            if credentials.get("api_key"):
                os.environ["PROJECTX_API_KEY"] = credentials["api_key"]
            if credentials.get("base_url"):
                os.environ["PROJECTX_BASE_URL"] = credentials["base_url"]
            from trading_app.live.projectx.auth import ProjectXAuth

            return ProjectXAuth()
        elif broker_type == "tradovate":
            if credentials.get("user"):
                os.environ["TRADOVATE_USER"] = credentials["user"]
            if credentials.get("password"):
                os.environ["TRADOVATE_PASS"] = credentials["password"]
            if credentials.get("cid"):
                os.environ["TRADOVATE_CID"] = credentials["cid"]
            if credentials.get("sec"):
                os.environ["TRADOVATE_SEC"] = credentials["sec"]
            from trading_app.live.tradovate.auth import TradovateAuth

            return TradovateAuth()
        elif broker_type == "rithmic":
            raise NotImplementedError("Rithmic requires running session")
        raise ValueError(f"Unknown broker type: {broker_type}")

    def _disconnect_auth(self, conn_id: str) -> None:
        state = self._states.get(conn_id)
        if state and state.auth:
            state.auth = None
            state.status = "disconnected"
            state.connected_at = None
            state.account_count = 0


connection_manager = BrokerConnectionManager()
