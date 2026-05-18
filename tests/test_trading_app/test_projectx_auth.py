"""Test ProjectX auth -- mocked HTTP.

Note: as of 2026-05-18 the auth path runs through BrokerHTTPClient which uses
requests.Session().request(...). We patch Session.request to intercept the
call without bypassing the classifier in http_client.
"""

from unittest.mock import MagicMock, patch


def _ok_resp(body):
    r = MagicMock()
    r.status_code = 200
    r.headers = {}
    r.json.return_value = body
    r.text = str(body)
    return r


def test_projectx_auth_login():
    """Auth should POST to /api/Auth/loginKey and return JWT."""
    mock_resp = _ok_resp({
        "token": "test_jwt_token_123",
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    })

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
    mock_resp = _ok_resp({
        "token": "jwt123",
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    })

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
