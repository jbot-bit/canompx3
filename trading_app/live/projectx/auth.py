"""ProjectX Gateway API authentication.

POST /api/Auth/loginKey with {userName, apiKey} -> JWT token (24h).
Refresh via POST /api/Auth/validate before expiry.

All HTTP traffic flows through trading_app.live.http_client.BrokerHTTPClient
for classified retry + deadline propagation. Direct requests.* calls are
forbidden in this subpackage (Stage 5 drift check).
"""

import logging
import os
import time

from trading_app.live.env_bootstrap import load_runtime_env

from ..broker_base import BrokerAuth
from ..http_client import (
    AUTH_POLICY,
    BrokerHTTPClient,
    BrokerProtocolError,
)

load_runtime_env()
log = logging.getLogger(__name__)

# Base URLs — configurable via PROJECTX_BASE_URL env var.
# Canonical: https://api.thefuturesdesk.projectx.com (per official API spec).
# See docs/reference/PROJECTX_API_REFERENCE.md for ground truth.
_DEFAULT_BASE = "https://api.thefuturesdesk.projectx.com"


def projectx_base_url() -> str:
    load_runtime_env()
    return os.environ.get("PROJECTX_BASE_URL", _DEFAULT_BASE).rstrip("/")


def projectx_market_hub_url(base_url: str | None = None) -> str:
    base = (base_url or projectx_base_url()).rstrip("/")
    return base.replace("://api.", "://rtc.") + "/hubs/market"


def projectx_user_hub_url(base_url: str | None = None) -> str:
    base = (base_url or projectx_base_url()).rstrip("/")
    return base.replace("://api.", "://rtc.") + "/hubs/user"


BASE_URL = projectx_base_url()
MARKET_HUB_URL = projectx_market_hub_url(BASE_URL)
USER_HUB_URL = projectx_user_hub_url(BASE_URL)


def _projectx_env_error(missing_keys: tuple[str, ...]) -> RuntimeError:
    load_result = load_runtime_env()
    attempted = ", ".join(str(path) for path in load_result.attempted_paths) or "none"
    loaded = str(load_result.env_path) if load_result.env_path else "none"
    return RuntimeError(
        "Missing ProjectX env var(s): "
        f"{', '.join(missing_keys)}; checked shell environment and runtime .env paths: "
        f"{attempted}; loaded_env={loaded}"
    )


def _projectx_login_credentials() -> tuple[str, str]:
    user = os.environ.get("PROJECTX_USERNAME") or os.environ.get("PROJECTX_USER")
    missing: list[str] = []
    if not user:
        missing.append("PROJECTX_USERNAME or PROJECTX_USER")
    api_key = os.environ.get("PROJECTX_API_KEY")
    if not api_key:
        missing.append("PROJECTX_API_KEY")
    if missing:
        raise _projectx_env_error(tuple(missing))
    return user, api_key


class ProjectXAuth(BrokerAuth):
    def __init__(self):
        super().__init__()
        self._token: str | None = None
        self._acquired_at: float = 0
        self._token_lifetime: float = 23 * 3600  # refresh after 23h (token lasts 24h)
        self._auth_healthy = True
        # Auth bootstrap has no refresh hook — login IS the refresh.
        # Auth's own HTTP client is intentionally NOT wired to the circuit
        # breaker (Stage 4 wiring): one-time login at startup happens before
        # the orchestrator's breaker is constructed, and a login failure is
        # already a fail-loud RuntimeError. Continuous broker reads
        # (positions/contracts/orders) DO get the breaker — they read
        # ``self.failure_hook`` (set by orchestrator) at their own __init__.
        self._http = BrokerHTTPClient(base_url=projectx_base_url(), name="projectx-auth")

    def get_token(self) -> str:
        if self._token and time.time() < self._acquired_at + self._token_lifetime:
            return self._token
        return self._login_with_retry()

    @property
    def is_healthy(self) -> bool:
        return self._auth_healthy

    def _login_with_retry(self) -> str:
        try:
            token = self._login()
            self._auth_healthy = True
            return token
        except Exception as exc:
            self._auth_healthy = False
            log.critical(
                "Auth login FAILED after retries (%s) — orders will fail until resolved",
                exc,
            )
            raise

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def refresh_if_needed(self) -> None:
        if self._token is None or time.time() >= self._acquired_at + self._token_lifetime:
            log.debug("ProjectX auth: proactive token refresh triggered")
            self._validate_or_login()

    def _login(self) -> str:
        user, api_key = _projectx_login_credentials()
        data = self._http.post_json(
            "/api/Auth/loginKey",
            headers={"Content-Type": "application/json", "Accept": "text/plain"},
            body={"userName": user, "apiKey": api_key},
            policy=AUTH_POLICY,
            timeout=10,
        )
        token = data.get("token")
        if not token:
            raise BrokerProtocolError(
                f"projectx-auth: login response missing token: {data}",
                error_class="G",
            )
        self._token = str(token)
        self._acquired_at = time.time()
        log.info("ProjectX auth: token acquired")
        return self._token

    def _validate_or_login(self) -> None:
        """Try to validate existing token; fall back to full login."""
        if self._token is None:
            self._login()
            return
        try:
            data = self._http.post_json(
                "/api/Auth/validate",
                headers={"Authorization": f"Bearer {self._token}"},
                body={},
                policy=AUTH_POLICY,
                timeout=10,
            )
            new_token = data.get("token") or data.get("newToken")
            if new_token:
                self._token = new_token
                self._acquired_at = time.time()
                log.info("ProjectX auth: token refreshed via validate")
            else:
                self._login()
        except Exception as exc:
            log.warning("ProjectX token validate failed (%s), falling back to full login", exc)
            self._login()
