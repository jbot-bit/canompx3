"""Tests for the autopilot deterministic tier classifier and PreToolUse guard.

The classifier is the single source of truth for "safe to do unattended (A)"
vs "block + flag (B)". These tests lock in:
  - every Tier-B path from autonomy-contract.md classifies B,
  - docs/config/tests classify A,
  - an unknown pipeline/trading_app path classifies B (fail-CLOSED),
  - the adversarial edge cases found during the build (case-insensitivity,
    rm -rf on a capital path, git-push word boundary, read vs write of a
    canon path),
  - a Tier-B edit attempt is BLOCKED by the hook and writes a journal line.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))

import tier_guard  # noqa: E402

# ── Path classification ────────────────────────────────────────
TIER_B_PATHS = [
    "trading_app/live/engine.py",
    "trading_app/broker_client.py",
    "trading_app/execution_engine.py",
    "trading_app/session_orchestrator.py",
    "trading_app/prop_profiles.py",
    "trading_app/prop_portfolio.py",
    "trading_app/account_survival.py",
    "trading_app/risk_manager.py",
    "pipeline/dst.py",
    "pipeline/cost_model.py",
    "pipeline/asset_configs.py",
    "pipeline/paths.py",
    "pipeline/holdout_policy.py",
    "trading_app/holdout_policy.py",
    "gold.db",
    "data/foo.db",
    "pipeline/schema_v2.sql",
    "trading_app/live_config.py",
    "trading_app/lane_allocation.py",
]

TIER_A_PATHS = [
    "docs/foo.md",
    "docs/prompts/autopilot-task-template.md",
    "scripts/autopilot/run_autopilot.sh",
    "tests/test_autopilot/test_tier_guard.py",
    ".claude/rules/some-rule.md",
    "README.md",
    "scripts/tools/gen_report.py",
]

FAIL_CLOSED_PATHS = [
    "pipeline/unknown_new.py",
    "trading_app/config.py",
    "trading_app/strategy_research.py",
    "pipeline/",
    "",  # empty -> fail closed
]


@pytest.mark.parametrize("path", TIER_B_PATHS)
def test_tier_b_paths_classify_b(path):
    tier, reason = tier_guard.classify_path(path)
    assert tier == "B", f"{path} should be Tier B, got {tier} ({reason})"


@pytest.mark.parametrize("path", TIER_A_PATHS)
def test_tier_a_paths_classify_a(path):
    tier, reason = tier_guard.classify_path(path)
    assert tier == "A", f"{path} should be Tier A, got {tier} ({reason})"


@pytest.mark.parametrize("path", FAIL_CLOSED_PATHS)
def test_unknown_prod_paths_fail_closed_to_b(path):
    tier, _ = tier_guard.classify_path(path)
    assert tier == "B", f"unknown prod path {path!r} must fail closed to B"


# ── Adversarial edge cases found during the build ──────────────
def test_case_insensitive_path_match():
    # Uppercase must NOT bypass the Tier-B table.
    assert tier_guard.classify_path("TRADING_APP/LIVE/engine.py")[0] == "B"
    assert tier_guard.classify_path("Pipeline/DST.py")[0] == "B"


def test_windows_absolute_path_still_classified():
    p = "C:\\Users\\joshd\\canompx3\\trading_app\\prop_profiles.py"
    assert tier_guard.classify_path(p)[0] == "B"


def test_dotdot_path_does_not_escape():
    assert tier_guard.classify_path("docs/../trading_app/prop_profiles.py")[0] == "B"


# ── Command classification ─────────────────────────────────────
TIER_B_COMMANDS = [
    "git push origin main",
    "git push",
    "python scripts/run_live_session.py --live",
    "python scripts/run_live_session.py --demo",
    "git reset --hard HEAD~1",
    "git clean -fd",
    "rm -rf trading_app/live",
    "rm -fr pipeline/",
    "duckdb gold.db 'DROP TABLE x'",
    "echo x > trading_app/prop_profiles.py",
    "sed -i s/a/b/ pipeline/dst.py",
    "python -m refresh_control_state",
]

TIER_A_COMMANDS = [
    "git diff HEAD",
    "git status",
    "ls docs/",
    "cat trading_app/prop_profiles.py",  # read of a canon file is OK
    "head pipeline/dst.py",
    "git pushx",  # word-boundary: not a push
    "pytest tests/test_autopilot/",
]


@pytest.mark.parametrize("cmd", TIER_B_COMMANDS)
def test_tier_b_commands(cmd):
    tier, reason = tier_guard.classify_action("Bash", {"command": cmd})
    assert tier == "B", f"{cmd!r} should be Tier B, got {tier} ({reason})"


@pytest.mark.parametrize("cmd", TIER_A_COMMANDS)
def test_tier_a_commands(cmd):
    tier, reason = tier_guard.classify_action("Bash", {"command": cmd})
    assert tier == "A", f"{cmd!r} should be Tier A, got {tier} ({reason})"


def test_read_only_tools_are_tier_a():
    for tool in ("Read", "Grep", "Glob"):
        assert tier_guard.classify_action(tool, {})[0] == "A"


# ── The PreToolUse hook: blocks Tier-B edit + writes journal ───
def _run_hook(event: dict, run_id: str, tmp_path: Path):
    """Invoke the hook as a subprocess (its real execution path)."""
    hook = PROJECT_ROOT / ".claude" / "hooks" / "autopilot-tier-guard.py"
    env = dict(os.environ)
    env["AUTOPILOT_RUN"] = "1"
    env["AUTOPILOT_RUN_ID"] = run_id
    return subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=15,
    )


def test_hook_blocks_tier_b_edit_and_journals(tmp_path):
    run_id = "test-block-run"
    journal = PROJECT_ROOT / "docs" / "runtime" / "autopilot" / f"{run_id}.jsonl"
    if journal.exists():
        journal.unlink()
    event = {
        "tool_name": "Edit",
        "tool_input": {"file_path": "trading_app/prop_profiles.py"},
    }
    try:
        r = _run_hook(event, run_id, tmp_path)
        assert r.returncode == 2, f"expected block (exit 2), got {r.returncode}: {r.stderr}"
        assert "BLOCKED_TIER_B" in (r.stdout + r.stderr)
        assert journal.exists(), "journal line not written"
        lines = [json.loads(ln) for ln in journal.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert any(e.get("event") == "BLOCKED_TIER_B" for e in lines)
    finally:
        if journal.exists():
            journal.unlink()


def test_hook_allows_tier_a_edit(tmp_path):
    r = _run_hook(
        {"tool_name": "Edit", "tool_input": {"file_path": "docs/foo.md"}},
        "test-allow-run",
        tmp_path,
    )
    assert r.returncode == 0


def test_hook_inert_without_autopilot_env(tmp_path):
    # Without AUTOPILOT_RUN=1 the hook must be a no-op even on a Tier-B edit.
    hook = PROJECT_ROOT / ".claude" / "hooks" / "autopilot-tier-guard.py"
    env = {k: v for k, v in os.environ.items() if k != "AUTOPILOT_RUN"}
    r = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "trading_app/prop_profiles.py"}}),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=15,
    )
    assert r.returncode == 0
    assert r.stdout.strip() == ""


def test_hook_fails_open_on_bad_input(tmp_path):
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / ".claude" / "hooks" / "autopilot-tier-guard.py")],
        input="not json",
        capture_output=True,
        text=True,
        env={**os.environ, "AUTOPILOT_RUN": "1"},
        cwd=str(PROJECT_ROOT),
        timeout=15,
    )
    assert r.returncode == 0  # fail-open
