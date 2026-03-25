"""ProjectX Gateway API authentication.

POST /api/Auth/loginKey with {userName, apiKey} -> JWT token (24h).
Refresh via POST /api/Auth/validate before expiry.
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv

from ..broker_base import BrokerAuth

load_dotenv()
log = logging.getLogger(__name__)

# Base URLs — configurable via PROJECTX_BASE_URL env var.
# Canonical: https://api.thefuturesdesk.projectx.com (per official API spec).
# See docs/reference/PROJECTX_API_REFERENCE.md for ground truth.
_DEFAULT_BASE = "https://api.thefuturesdesk.projectx.com"
BASE_URL = os.environ.get("PROJECTX_BASE_URL", _DEFAULT_BASE).rstrip("/")
MARKET_HUB_URL = BASE_URL.replace("://api.", "://rtc.") + "/hubs/market"
USER_HUB_URL = BASE_URL.replace("://api.", "://rtc.") + "/hubs/user"


_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds: 1, 2, 4


class ProjectXAuth(BrokerAuth):
    def __init__(self):
        self._token: str | None = None
        self._acquired_at: float = 0
        self._token_lifetime: float = 23 * 3600  # refresh after 23h (token lasts 24h)
        self._auth_healthy = True

    def get_token(self) -> str:
        if self._token and time.time() < self._acquired_at + self._token_lifetime:
            return self._token
        return self._login_with_retry()

    @property
    def is_healthy(self) -> bool:
        return self._auth_healthy

    def _login_with_retry(self) -> str:
        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                token = self._login()
                self._auth_healthy = True
                return token
            except Exception as e:
                last_error = e
                wait = _BACKOFF_BASE * (2**attempt)
                log.warning(
                    "Auth login attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    import time as _t

                    _t.sleep(wait)

        self._auth_healthy = False
        log.critical(
            "Auth login FAILED after %d attempts: %s — orders will fail until resolved",
            _MAX_RETRIES, last_error,
        )
        raise RuntimeError(
            f"Auth login failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def refresh_if_needed(self) -> None:
        if self._token is None or time.time() >= self._acquired_at + self._token_lifetime:
            log.debug("ProjectX auth: proactive token refresh triggered")
            self._validate_or_login()

    def _login(self) -> str:
        user = os.environ.get("PROJECTX_USERNAME") or os.environ["PROJECTX_USER"]
        api_key = os.environ["PROJECTX_API_KEY"]
        resp = requests.post(
            f"{BASE_URL}/api/Auth/loginKey",
            json={"userName": user, "apiKey": api_key},
            headers={"Content-Type": "application/json", "Accept": "text/plain"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"ProjectX auth failed: {data.get('errorMessage', data)}")
        self._token = data["token"]
        self._acquired_at = time.time()
        log.info("ProjectX auth: token acquired")
        return self._token

    def _validate_or_login(self) -> None:
        """Try to validate existing token; fall back to full login."""
        if self._token is None:
            self._login()
            return
        try:
            resp = requests.post(
                f"{BASE_URL}/api/Auth/validate",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            new_token = data.get("token") or data.get("newToken")
            if data.get("success") and new_token:
                self._token = new_token
                self._acquired_at = time.time()
                log.info("ProjectX auth: token refreshed via validate")
            else:
                self._login()
        except requests.RequestException:
            log.warning("ProjectX token validate failed, falling back to full login")
            self._login()
