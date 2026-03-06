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

import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

LIVE_BASE = "https://live.tradovateapi.com/v1"
DEMO_BASE = "https://demo.tradovateapi.com/v1"


class TradovateAuth:
    def __init__(self, demo: bool = True):
        self.base = DEMO_BASE if demo else LIVE_BASE
        self._token: str | None = None
        self._expires_at: float = 0

    def get_token(self) -> str:
        """Return a valid access token, refreshing if within 60s of expiry."""
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        return self._refresh()

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
