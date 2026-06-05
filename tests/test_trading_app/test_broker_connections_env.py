from __future__ import annotations

import importlib


def test_broker_connections_migrates_projectx_from_canonical_env(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    worktree_root = tmp_path / "linked-worktree"
    canonical_root = tmp_path / "canonical-root"
    worktree_root.mkdir()
    canonical_root.mkdir()
    (canonical_root / ".env").write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=canonical_user",
                "PROJECTX_API_KEY=canonical_key_12345678901234567890",
                "PROJECTX_BASE_URL=https://api.canonical-broker.test",
            ]
        ),
        encoding="utf-8",
    )
    (worktree_root / ".env.example").write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=example_user",
                "PROJECTX_API_KEY=example_key_12345678901234567890",
            ]
        ),
        encoding="utf-8",
    )

    for key in ("PROJECTX_USERNAME", "PROJECTX_USER", "PROJECTX_API_KEY", "PROJECTX_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env_bootstrap, "PROJECT_ROOT", worktree_root)
    monkeypatch.setattr(env_bootstrap, "CANONICAL_RUNTIME_ROOT", canonical_root)

    import trading_app.live.broker_connections as broker_connections

    broker_connections = importlib.reload(broker_connections)
    monkeypatch.setattr(broker_connections, "_CONNECTIONS_PATH", tmp_path / "data" / "broker_connections.json")

    manager = broker_connections.BrokerConnectionManager()
    manager.load()

    connections = manager.get_enabled_connections()
    assert len(connections) == 1
    credentials = connections[0]["credentials"]
    assert credentials["username"] == "canonical_user"
    assert credentials["api_key"] == "canonical_key_12345678901234567890"
    assert credentials["base_url"] == "https://api.canonical-broker.test"
