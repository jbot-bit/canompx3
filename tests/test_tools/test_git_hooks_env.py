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


def test_pre_commit_drift_skips_advisory_but_keeps_crg_marker():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    drift_cmd = "pipeline/check_drift.py --skip-crg-advisory --skip-advisory"
    assert drift_cmd in text
    assert "--skip-crg-advisory" in text
    assert "--skip-advisory" in text


def test_pre_commit_emits_stage_timing_and_releases_commit_lock():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert "_finish_stage_timing" in text
    assert "PRE-COMMIT TIMING total" in text
    assert "CANOMPX3_PRECOMMIT_LOCK_OWNER" in text
    assert 'commit_serialize.py" release' in text
    assert text.index("_finish_stage_timing") < text.index('commit_serialize.py" release')


def test_pre_commit_pytest_fast_path_excludes_slow_tests_and_uses_repo_basetemp():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert '-m "not slow"' in text
    assert '--basetemp "$PYTEST_BASETEMP"' in text
    assert "git rev-parse --git-common-dir" in text
    assert "pytest-tmp/precommit-$$" in text
    assert 'mkdir -p "$PYTEST_COMMON_DIR/pytest-tmp"' in text


def test_post_commit_prefers_wsl_venv_before_windows_venv_on_posix_shells():
    text = (ROOT / ".githooks" / "post-commit").read_text(encoding="utf-8")

    assert "IS_WINDOWS_SHELL=0" in text
    assert "SCRIPT_DIR/.venv-wsl/bin/python" in text
    assert "SCRIPT_DIR/.venv/Scripts/python.exe" in text
    assert text.index("SCRIPT_DIR/.venv-wsl/bin/python") < text.index("SCRIPT_DIR/.venv/Scripts/python.exe")
    assert '[[ "$PYTHON" == *"/Scripts/"* || "$PYTHON" == *.exe ]]' in text
    assert '[[ "$PYTHON" == *"/.venv-wsl/"* ]]' in text
