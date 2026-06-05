from __future__ import annotations

import importlib
import os


def test_tradovate_auth_loads_canonical_runtime_env(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    worktree_root = tmp_path / "linked-worktree"
    canonical_root = tmp_path / "canonical-root"
    worktree_root.mkdir()
    canonical_root.mkdir()
    (canonical_root / ".env").write_text(
        "\n".join(
            [
                "TRADOVATE_USERNAME=canonical_trader",
                "TRADOVATE_PASSWORD=canonical_password",
                "TRADOVATE_CID=123",
                "TRADOVATE_SEC=canonical_secret",
                "TRADOVATE_DEMO=1",
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "TRADOVATE_USERNAME",
        "TRADOVATE_PASSWORD",
        "TRADOVATE_CID",
        "TRADOVATE_SEC",
        "TRADOVATE_DEMO",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_bootstrap, "PROJECT_ROOT", worktree_root)
    monkeypatch.setattr(env_bootstrap, "CANONICAL_RUNTIME_ROOT", canonical_root)

    import trading_app.live.tradovate.auth as tradovate_auth

    tradovate_auth = importlib.reload(tradovate_auth)
    auth = tradovate_auth.TradovateAuth()

    assert os.environ["TRADOVATE_USERNAME"] == "canonical_trader"
    assert os.environ["TRADOVATE_CID"] == "123"
    assert auth.base_url == "https://demo.tradovateapi.com/v1"
