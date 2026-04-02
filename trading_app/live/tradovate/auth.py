"""Tradovate API authentication.

POST /v1/auth/accesstokenrequest with {name, password, cid, sec, deviceId}
→ {accessToken, mdAccessToken, expirationTime, userId}.

Token lifetime: ~24h (expirationTime in response). Refresh via /auth/renewaccesstoken.

Env vars:
    TRADOVATE_USER      — account username (also accepts TRADOVATE_USERNAME)
    TRADOVATE_PASS      — account password (also accepts TRADOVATE_PASSWORD)
    TRADOVATE_CID       — client app ID (from Tradovate API key page)
    TRADOVATE_SEC       — API secret (from Tradovate API key page)
    TRADOVATE_DEVICE_ID — unique device identifier (generate once, reuse)
    TRADOVATE_DEMO      — set to "1" for demo environment (default: live)
"""

import logging
import os
import time
import uuid

import requests
from dotenv import load_dotenv

from ..broker_base import BrokerAuth

load_dotenv()
log = logging.getLogger(__name__)

LIVE_URL = "https://live.tradovateapi.com/v1"
DEMO_URL = "https://demo.tradovateapi.com/v1"

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0


def _base_url() -> str:
    return DEMO_URL if os.environ.get("TRADOVATE_DEMO", "0") == "1" else LIVE_URL


class TradovateAuth(BrokerAuth):
    """Authenticate with Tradovate REST API."""

    def __init__(self):
        self._access_token: str | None = None
        self._md_token: str | None = None
        self._user_id: int | None = None
        self._acquired_at: float = 0
        self._token_lifetime: float = 75 * 60  # refresh after 75min (tokens expire at 90min)
        self._auth_healthy = True
        self._base = _base_url()

    def get_token(self) -> str:
        if self._access_token and time.time() < self._acquired_at + self._token_lifetime:
            return self._access_token
        return self._login_with_retry()

    @property
    def md_token(self) -> str | None:
        """Market data access token (separate from trading token)."""
        return self._md_token

    @property
    def user_id(self) -> int | None:
        return self._user_id

    @property
    def is_healthy(self) -> bool:
        return self._auth_healthy

    @property
    def base_url(self) -> str:
        return self._base

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def refresh_if_needed(self) -> None:
        if self._access_token is None or time.time() >= self._acquired_at + self._token_lifetime:
            log.debug("Tradovate auth: proactive token refresh")
            self._renew_or_login()

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
                    "Tradovate auth attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                    wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)

        self._auth_healthy = False
        log.critical("Tradovate auth FAILED after %d attempts: %s", _MAX_RETRIES, last_error)
        raise RuntimeError(f"Tradovate auth failed after {_MAX_RETRIES} attempts: {last_error}") from last_error

    def _login(self) -> str:
        name = os.environ.get("TRADOVATE_USER") or os.environ["TRADOVATE_USERNAME"]
        password = os.environ.get("TRADOVATE_PASS") or os.environ["TRADOVATE_PASSWORD"]
        cid = int(os.environ["TRADOVATE_CID"])
        sec = os.environ["TRADOVATE_SEC"]
        device_id = os.environ.get("TRADOVATE_DEVICE_ID", str(uuid.uuid4()))

        resp = requests.post(
            f"{self._base}/auth/accesstokenrequest",
            json={
                "name": name,
                "password": password,
                "appId": "canompx3-bot",
                "appVersion": "1.0",
                "cid": cid,
                "sec": sec,
                "deviceId": device_id,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Tradovate returns error as p-ticket challenge or errorText
        if "p-ticket" in data:
            raise RuntimeError("Tradovate requires 2FA p-ticket — complete manually first")
        if "errorText" in data:
            raise RuntimeError(f"Tradovate auth failed: {data['errorText']}")

        token: str = data["accessToken"]
        self._access_token = token
        self._md_token = data.get("mdAccessToken")
        self._user_id = data.get("userId")
        self._acquired_at = time.time()
        log.info("Tradovate auth: token acquired (userId=%s, demo=%s)", self._user_id, self._base == DEMO_URL)
        return token

    def _renew_or_login(self) -> None:
        """Try token renewal; fall back to full login with retry."""
        if self._access_token is None:
            self._login_with_retry()
            return
        try:
            resp = requests.post(
                f"{self._base}/auth/renewaccesstoken",
                headers={"Authorization": f"Bearer {self._access_token}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("accessToken"):
                self._access_token = data["accessToken"]
                self._md_token = data.get("mdAccessToken", self._md_token)
                self._acquired_at = time.time()
                log.info("Tradovate auth: token renewed")
            else:
                self._login_with_retry()
        except requests.RequestException:
            log.warning("Tradovate token renewal failed, falling back to full login with retry")
            self._login_with_retry()
