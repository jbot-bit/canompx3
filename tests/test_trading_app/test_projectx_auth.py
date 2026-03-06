"""Test ProjectX auth -- mocked HTTP."""

from unittest.mock import MagicMock, patch


def test_projectx_auth_login():
    """Auth should POST to /api/Auth/loginKey and return JWT."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "token": "test_jwt_token_123",
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            token = auth.get_token()
            assert token == "test_jwt_token_123"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/api/Auth/loginKey" in call_args[0][0]
            assert call_args[1]["json"]["userName"] == "testuser"
            assert call_args[1]["json"]["apiKey"] == "testkey"


def test_projectx_auth_headers():
    """Headers should use Bearer scheme."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "token": "jwt123",
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"PROJECTX_USER": "testuser", "PROJECTX_API_KEY": "testkey"}):
        with patch("requests.post", return_value=mock_resp):
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            headers = auth.headers()
            assert headers["Authorization"] == "Bearer jwt123"


def test_projectx_auth_is_broker_auth():
    """ProjectXAuth must be a BrokerAuth."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"token": "t", "success": True, "errorCode": 0, "errorMessage": None}
    mock_resp.raise_for_status = MagicMock()

    with patch.dict("os.environ", {"PROJECTX_USER": "u", "PROJECTX_API_KEY": "k"}):
        with patch("requests.post", return_value=mock_resp):
            from trading_app.live.broker_base import BrokerAuth
            from trading_app.live.projectx.auth import ProjectXAuth

            auth = ProjectXAuth()
            assert isinstance(auth, BrokerAuth)
