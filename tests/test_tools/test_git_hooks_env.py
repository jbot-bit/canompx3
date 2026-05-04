from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pre_commit_prefers_wsl_venv_before_windows_venv_on_posix_shells():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert "IS_WINDOWS_SHELL=0" in text
    assert "SCRIPT_DIR/.venv-wsl/bin/python" in text
    assert "SCRIPT_DIR/.venv/Scripts/python.exe" in text
    assert text.index("SCRIPT_DIR/.venv-wsl/bin/python") < text.index("SCRIPT_DIR/.venv/Scripts/python.exe")
    assert '[[ "$candidate" == *"/Scripts/"* ]]' in text
    assert "POSIX/WSL pre-commit selected Windows Python" in text
    assert "Windows pre-commit selected WSL Python" in text
    assert "continue" in text


def test_pre_commit_ruff_prefers_wsl_venv_before_windows_ruff_on_posix_shells():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert "SCRIPT_DIR/.venv-wsl/bin/ruff" in text
    assert "SCRIPT_DIR/.venv/Scripts/ruff.exe" in text
    assert text.index("SCRIPT_DIR/.venv-wsl/bin/ruff") < text.index("SCRIPT_DIR/.venv/Scripts/ruff.exe")
    assert "POSIX/WSL pre-commit selected Windows ruff" in text


def test_post_commit_prefers_wsl_venv_before_windows_venv_on_posix_shells():
    text = (ROOT / ".githooks" / "post-commit").read_text(encoding="utf-8")

    assert "IS_WINDOWS_SHELL=0" in text
    assert "SCRIPT_DIR/.venv-wsl/bin/python" in text
    assert "SCRIPT_DIR/.venv/Scripts/python.exe" in text
    assert text.index("SCRIPT_DIR/.venv-wsl/bin/python") < text.index("SCRIPT_DIR/.venv/Scripts/python.exe")
    assert '[[ "$PYTHON" == *"/Scripts/"* || "$PYTHON" == *.exe ]]' in text
    assert '[[ "$PYTHON" == *"/.venv-wsl/"* ]]' in text
