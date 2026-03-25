"""
Tradovate OAuth token management. Auto-renews before expiry.

Reads credentials from .env (loaded via python-dotenv):
    TRADOVATE_USER=your_email
    TRADOVATE_PASS=your_password
    TRADOVATE_APP_ID=Sample App
    TRADOVATE_APP_VERSION=1.0
    TRADOVATE_CID=your_cid
    TRADOVATE_SEC=your_secret

TopstepX: use your TopstepX Tradovate account credentials here.
"""

import logging
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

from ..broker_base import BrokerAuth

load_dotenv()
log = logging.getLogger(__name__)

LIVE_BASE = "https://live.tradovateapi.com/v1"
DEMO_BASE = "https://demo.tradovateapi.com/v1"

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4


class TradovateAuth(BrokerAuth):
    def __init__(self, demo: bool = True):
        self.base = DEMO_BASE if demo else LIVE_BASE
        self._token: str | None = None
        self._expires_at: float = 0
        self._auth_healthy = True

    def get_token(self) -> str:
        """Return a valid access token, refreshing if within 60s of expiry."""
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        return self._refresh_with_retry()

    @property
    def is_healthy(self) -> bool:
        """True if last auth attempt succeeded."""
        return self._auth_healthy

    def _refresh_with_retry(self) -> str:
        """Refresh token with retry and exponential backoff."""
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                token = self._refresh()
                self._auth_healthy = True
                return token
            except Exception as e:
                last_error = e
                wait = _BACKOFF_BASE * (2**attempt)
                log.warning(
                    "Auth refresh attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        self._auth_healthy = False
        log.critical(
            "Auth refresh FAILED after %d attempts: %s — orders will fail until resolved",
            _MAX_RETRIES, last_error,
        )
        raise RuntimeError(
            f"Auth refresh failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def _refresh(self) -> str:
        resp = requests.post(
            f"{self.base}/auth/accesstokenrequest",
            json={
                "name": os.environ["TRADOVATE_USER"],
                "password": os.environ["TRADOVATE_PASS"],
                "appId": os.environ["TRADOVATE_APP_ID"],
                "appVersion": os.environ.get("TRADOVATE_APP_VERSION", "1.0"),
                "cid": int(os.environ["TRADOVATE_CID"]),
                "sec": os.environ["TRADOVATE_SEC"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["accessToken"]
        exp = datetime.fromisoformat(data["expirationTime"].replace("Z", "+00:00"))
        self._expires_at = exp.timestamp()
        return self._token

    def headers(self) -> dict:
        """Return Authorization header dict for REST calls."""
        return {"Authorization": f"Bearer {self.get_token()}"}

    def refresh_if_needed(self) -> None:
        """Proactively refresh token if within 120s of expiry."""
        if self._token is None or time.time() >= self._expires_at - 120:
            self._refresh_with_retry()
