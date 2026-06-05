from __future__ import annotations

import os


def test_load_runtime_env_prefers_canonical_runtime_root(monkeypatch, tmp_path):
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
                "PROJECTX_BASE_URL=https://api.canonical.test",
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

    monkeypatch.delenv("PROJECTX_USERNAME", raising=False)
    monkeypatch.delenv("PROJECTX_USER", raising=False)
    monkeypatch.delenv("PROJECTX_API_KEY", raising=False)
    monkeypatch.delenv("PROJECTX_BASE_URL", raising=False)
    monkeypatch.setattr(env_bootstrap, "PROJECT_ROOT", worktree_root)
    monkeypatch.setattr(env_bootstrap, "CANONICAL_RUNTIME_ROOT", canonical_root)

    result = env_bootstrap.load_runtime_env()

    assert result.loaded is True
    assert result.env_path == canonical_root / ".env"
    assert os.environ["PROJECTX_USERNAME"] == "canonical_user"
    assert os.environ["PROJECTX_API_KEY"] == "canonical_key_12345678901234567890"
    assert os.environ["PROJECTX_BASE_URL"] == "https://api.canonical.test"


def test_load_runtime_env_never_loads_env_example(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    worktree_root = tmp_path / "linked-worktree"
    canonical_root = tmp_path / "canonical-root"
    worktree_root.mkdir()
    canonical_root.mkdir()
    (canonical_root / ".env.example").write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=example_user",
                "PROJECTX_API_KEY=example_key_12345678901234567890",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("PROJECTX_USERNAME", raising=False)
    monkeypatch.delenv("PROJECTX_USER", raising=False)
    monkeypatch.delenv("PROJECTX_API_KEY", raising=False)
    monkeypatch.setattr(env_bootstrap, "PROJECT_ROOT", worktree_root)
    monkeypatch.setattr(env_bootstrap, "CANONICAL_RUNTIME_ROOT", canonical_root)

    result = env_bootstrap.load_runtime_env()

    assert result.loaded is False
    assert result.env_path is None
    assert "PROJECTX_USERNAME" not in os.environ
    assert "PROJECTX_API_KEY" not in os.environ


def test_load_runtime_env_does_not_override_shell_env(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    canonical_root = tmp_path / "canonical-root"
    canonical_root.mkdir()
    (canonical_root / ".env").write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=canonical_user",
                "PROJECTX_API_KEY=canonical_key_12345678901234567890",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PROJECTX_USERNAME", "shell_user")
    monkeypatch.delenv("PROJECTX_API_KEY", raising=False)
    monkeypatch.setattr(env_bootstrap, "PROJECT_ROOT", tmp_path / "worktree")
    monkeypatch.setattr(env_bootstrap, "CANONICAL_RUNTIME_ROOT", canonical_root)

    result = env_bootstrap.load_runtime_env()

    assert result.loaded is True
    assert os.environ["PROJECTX_USERNAME"] == "shell_user"
    assert os.environ["PROJECTX_API_KEY"] == "canonical_key_12345678901234567890"


def test_detect_tracked_env_example_secret_shapes_redacts_values(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    env_example = tmp_path / ".env.example"
    env_example.write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=real.user@broker.invalid",
                "PROJECTX_API_KEY=px_real_key_123456789012345678901234567890",
                "PROJECTX_BASE_URL=https://api.thefuturesdesk.projectx.com",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(env_bootstrap, "_is_git_tracked", lambda _root, _rel: True)

    warning = env_bootstrap.detect_tracked_env_example_secret_shapes(tmp_path)

    assert warning is not None
    assert warning.keys == ("PROJECTX_USERNAME", "PROJECTX_API_KEY")
    message = warning.format_message()
    assert ".env.example" in message
    assert "PROJECTX_USERNAME" in message
    assert "PROJECTX_API_KEY" in message
    assert "real.user@broker.invalid" not in message
    assert "px_real_key_123456789012345678901234567890" not in message


def test_detect_tracked_env_example_ignores_placeholders(monkeypatch, tmp_path):
    from trading_app.live import env_bootstrap

    (tmp_path / ".env.example").write_text(
        "\n".join(
            [
                "PROJECTX_USERNAME=your_username",
                "PROJECTX_API_KEY=your_projectx_api_key",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(env_bootstrap, "_is_git_tracked", lambda _root, _rel: True)

    assert env_bootstrap.detect_tracked_env_example_secret_shapes(tmp_path) is None
