"""Test ProjectX auth -- mocked HTTP.

Note: as of 2026-05-18 the auth path runs through BrokerHTTPClient which uses
requests.Session().request(...). We patch Session.request to intercept the
call without bypassing the classifier in http_client.
"""

from unittest.mock import MagicMock, patch

import pytest


def _ok_resp(body):
    r = MagicMock()
    r.status_code = 200
    r.headers = {}
    r.json.return_value = body
    r.text = str(body)
    return r


def test_projectx_auth_login():
    """Auth should POST to /api/Auth/loginKey and return JWT."""
    mock_resp = _ok_resp(
        {
            "token": "test_jwt_token_123",
            "success": True,
            "errorCode": 0,
            "errorMessage": None,
        }
    )

    with patch.dict(
        "os.environ", {"PROJECTX_USERNAME": "testuser", "PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}
    ):
        with patch("requests.Session.request", return_value=mock_resp) as mock_req:
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            token = auth.get_token()
            assert token == "test_jwt_token_123"
            mock_req.assert_called_once()
            call_kwargs = mock_req.call_args.kwargs
            call_url = mock_req.call_args.args[1] if len(mock_req.call_args.args) > 1 else call_kwargs.get("url", "")
            assert "/api/Auth/loginKey" in call_url
            assert call_kwargs["json"]["userName"] == "testuser"
            assert call_kwargs["json"]["apiKey"] == "testkey"


def test_projectx_auth_headers():
    """Headers should use Bearer scheme."""
    mock_resp = _ok_resp(
        {
            "token": "jwt123",
            "success": True,
            "errorCode": 0,
            "errorMessage": None,
        }
    )

    with patch.dict(
        "os.environ", {"PROJECTX_USERNAME": "testuser", "PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}
    ):
        with patch("requests.Session.request", return_value=mock_resp):
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            headers = auth.headers()
            assert headers["Authorization"] == "Bearer jwt123"


def test_projectx_auth_is_broker_auth():
    """ProjectXAuth must be a BrokerAuth."""
    mock_resp = _ok_resp({"token": "t", "success": True, "errorCode": 0, "errorMessage": None})

    with patch.dict("os.environ", {"PROJECTX_USER": "u", "PROJECTX_API_KEY": "k"}):
        with patch("requests.Session.request", return_value=mock_resp):
            from trading_app.live.broker_base import BrokerAuth
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            assert isinstance(auth, BrokerAuth)


def test_projectx_credentials_accept_projectx_username_without_legacy_user(monkeypatch):
    monkeypatch.setenv("PROJECTX_USERNAME", "canonical_user")
    monkeypatch.delenv("PROJECTX_USER", raising=False)
    monkeypatch.setenv("PROJECTX_API_KEY", "canonical_key")

    from trading_app.live.projectx import auth as auth_module

    assert auth_module._projectx_login_credentials() == ("canonical_user", "canonical_key")


def test_projectx_credentials_report_missing_username_without_keyerror(monkeypatch):
    monkeypatch.delenv("PROJECTX_USERNAME", raising=False)
    monkeypatch.delenv("PROJECTX_USER", raising=False)
    monkeypatch.setenv("PROJECTX_API_KEY", "canonical_secret_key_value")

    from trading_app.live.projectx import auth as auth_module

    with pytest.raises(RuntimeError, match="PROJECTX_USERNAME or PROJECTX_USER") as excinfo:
        auth_module._projectx_login_credentials()

    message = str(excinfo.value)
    assert "checked shell environment" in message
    assert ".env" in message
    assert "canonical_secret_key_value" not in message


def test_projectx_base_url_resolves_after_module_import(monkeypatch):
    from trading_app.live.projectx import auth as auth_module

    monkeypatch.setenv("PROJECTX_BASE_URL", "https://api.dynamic-projectx.test/")

    assert auth_module.projectx_base_url() == "https://api.dynamic-projectx.test"
    assert auth_module.projectx_market_hub_url() == "https://rtc.dynamic-projectx.test/hubs/market"
    assert auth_module.projectx_user_hub_url() == "https://rtc.dynamic-projectx.test/hubs/user"

    auth = auth_module.ProjectXAuth()
    assert auth._http.base_url == "https://api.dynamic-projectx.test"
