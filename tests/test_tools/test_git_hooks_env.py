import re
import shutil
import subprocess
from pathlib import Path

import pytest

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


# --- Path-scoped drift gate (branch drift-supersonic, 2026-06-05) ---------------
#
# The pre-commit hook skips the heavy ~172s drift step for commits that stage ONLY
# docs-safe markdown/notes (fail-closed: any code/config/unknown path → full drift).
# Two layers of tests:
#   * marker assertions — house style, cheap, catch removal of the gate.
#   * behavioral assertions — source the bash classifier and run real paths through
#     it. These catch a `case`-ordering bug (e.g. *.md matching before docs/audit/*)
#     that a text assertion never would. This is the load-bearing safety test.

_BASH = shutil.which("bash")


def test_pre_commit_drift_gate_markers_present():
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    # The classifier function and the two operator-specified log lines.
    assert "_drift_path_is_docs_safe()" in text
    assert "DRIFT: skipped path-safe docs-only commit" in text
    assert "DRIFT: running full check due to" in text
    # Deletions must be included (delete-only .py must not look like an empty set).
    assert "--diff-filter=ACMD" in text
    # Empty staged set is fail-closed (full drift), not docs-safe.
    assert "empty staged set (fail-closed)" in text
    # The full drift command is still the path-scoped FALSE branch — unchanged flags.
    assert "pipeline/check_drift.py --skip-crg-advisory --skip-advisory" in text


def test_pre_commit_drift_gate_denylist_checked_before_md_allowlist():
    """case-order safety: denylist patterns must appear before the *.md branch."""
    text = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    body = text[text.index("_drift_path_is_docs_safe()") :]
    body = body[: body.index("\n}\n")]
    # docs/audit/* and the code-surface deny patterns must be matched before *.md.
    assert body.index("docs/audit/*) return 1") < body.index("*.md) return 0")
    assert body.index("pipeline/*") < body.index("*.md) return 0")
    assert body.index("*.py|") < body.index("*.md) return 0")


def _classify(path: str) -> str:
    """Source the classifier out of the live hook and ask it about one path.

    Returns 'SAFE' (docs-only, drift skippable) or 'FULL' (run full drift).
    Extracting from the real hook (not a copy) means the test tracks the hook;
    a future edit that breaks ordering fails here.
    """
    hook = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    m = re.search(r"^_drift_path_is_docs_safe\(\) \{\n.*?^\}\n", hook, re.DOTALL | re.MULTILINE)
    assert m, "could not extract _drift_path_is_docs_safe from pre-commit"
    func = m.group(0)
    script = func + '\nif _drift_path_is_docs_safe "$1"; then echo SAFE; else echo FULL; fi\n'
    proc = subprocess.run(
        [_BASH, "-c", script, "_", path],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


@pytest.mark.skipif(_BASH is None, reason="bash not available")
@pytest.mark.parametrize(
    "path",
    [
        "HANDOFF.md",
        "REPO_MAP.md",
        "README.md",
        "docs/plans/active/2026-06/x.md",
        "docs/runtime/stages/drift-path-scope-gate.md",
        "docs/handoffs/archived/2026-06-03.md",
    ],
)
def test_classifier_docs_safe_paths_skip(path):
    assert _classify(path) == "SAFE", f"{path} should be docs-safe (skip drift)"


@pytest.mark.skipif(_BASH is None, reason="bash not available")
@pytest.mark.parametrize(
    "path",
    [
        # Code / config surfaces — must run full drift.
        "pipeline/check_drift.py",
        "trading_app/prop_profiles.py",
        "scripts/tools/foo.py",
        "tests/test_x.py",
        ".githooks/pre-commit",
        "pyproject.toml",
        "uv.lock",
        ".mcp.json",
        ".python-version",
        "lane_allocation.json",
        "config.yaml",
        # .md files that live under denylisted surfaces — fail-closed despite .md.
        ".claude/rules/foo.md",
        "docs/audit/results/x.md",
        "docs/audit/2026.md",
        "pipeline/notes.md",
        # Unknown shapes — fail-closed.
        "Makefile",
        "assets/logo.png",
        "",
    ],
)
def test_classifier_code_and_unknown_paths_run_full(path):
    assert _classify(path) == "FULL", f"{path} must run full drift (fail-closed)"


def test_pre_push_full_drift_gate_present_and_unflagged():
    """pre-push runs the FULL drift set (no --skip flags) and fails closed."""
    pre_push = ROOT / ".githooks" / "pre-push"
    assert pre_push.exists(), "pre-push safety gate must exist"
    text = pre_push.read_text(encoding="utf-8")
    # Full set: the check_drift.py INVOCATION carries NO --skip flags. (Comments
    # may mention the flags by name when explaining why pre-push omits them, so
    # assert against the actual command line, not the whole file text.)
    invocation = [ln for ln in text.splitlines() if "check_drift.py" in ln and not ln.lstrip().startswith("#")]
    assert invocation, "pre-push must invoke check_drift.py"
    for ln in invocation:
        assert "--skip-crg-advisory" not in ln, f"pre-push must run FULL drift: {ln}"
        assert "--skip-advisory" not in ln, f"pre-push must run FULL drift: {ln}"
    # Fail-closed: a drift failure blocks the push.
    assert "BLOCKED (pre-push)" in text
    assert "exit 1" in text
