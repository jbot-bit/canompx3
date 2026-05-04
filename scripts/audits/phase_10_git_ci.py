#!/usr/bin/env python3
"""Phase 10 — Git & CI Hygiene.

Source: SYSTEM_AUDIT.md Phase 10 (lines 410-427)

Checks git hooks, CI pipeline, and repo cleanliness.
"""

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audits import PROJECT_ROOT, AuditPhase, Severity


def _git_config(key: str) -> str:
    """Get a git config value."""
    r = subprocess.run(
        ["git", "config", "--get", key],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def _git_status() -> str:
    """Get git status output."""
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    return r.stdout.strip() if r.returncode == 0 else ""


def main():
    audit = AuditPhase(phase_num=10, name="Git & CI Hygiene")
    audit.print_header()

    # ── 10A: Git Hooks ──
    print("\n--- 10A. Git Hooks ---")
    pre_commit = PROJECT_ROOT / ".githooks" / "pre-commit"
    if pre_commit.exists():
        audit.check_passed(".githooks/pre-commit exists")
        # Check executable
        if os.access(pre_commit, os.X_OK):
            audit.check_passed(".githooks/pre-commit is executable")
        else:
            audit.check_failed(".githooks/pre-commit is NOT executable")
            audit.add_finding(
                Severity.MEDIUM,
                "CONFIG_DRIFT",
                claimed="pre-commit hook is executable",
                actual="Missing execute permission",
                evidence=f"stat {pre_commit}",
                fix_type="CODE_FIX",
            )
    else:
        audit.check_failed(".githooks/pre-commit does NOT exist")
        audit.add_finding(
            Severity.HIGH,
            "GATE_MISSING",
            claimed="Pre-commit hook exists",
            actual="File not found",
            evidence=str(pre_commit),
            fix_type="CODE_FIX",
        )

    hooks_path = _git_config("core.hooksPath")
    if hooks_path == ".githooks":
        audit.check_passed("git config core.hooksPath = .githooks")
    else:
        audit.check_failed(f"git config core.hooksPath = '{hooks_path}' (expected '.githooks')")
        audit.add_finding(
            Severity.HIGH,
            "CONFIG_DRIFT",
            claimed="core.hooksPath = .githooks",
            actual=f"core.hooksPath = '{hooks_path}'",
            evidence="git config --get core.hooksPath",
            fix_type="CONFIG_FIX",
        )

    # ── 10B: CI Pipeline ──
    print("\n--- 10B. CI Pipeline ---")
    workflows_dir = PROJECT_ROOT / ".github" / "workflows"
    if workflows_dir.exists():
        ci_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        if ci_files:
            audit.check_passed(f"CI config: {len(ci_files)} workflow file(s) in .github/workflows/")
            for f in ci_files:
                print(f"         {f.name}")
                # Check for push/PR triggers
                content = f.read_text(encoding="utf-8")
                has_push = "push:" in content or "push" in content
                has_pr = "pull_request" in content
                triggers = []
                if has_push:
                    triggers.append("push")
                if has_pr:
                    triggers.append("PR")
                if triggers:
                    print(f"           triggers: {', '.join(triggers)}")
                # Check for drift/test commands
                has_drift = "check_drift" in content
                has_tests = "pytest" in content
                checks = []
                if has_drift:
                    checks.append("drift")
                if has_tests:
                    checks.append("tests")
                if checks:
                    print(f"           runs: {', '.join(checks)}")
                if not has_drift:
                    audit.add_finding(
                        Severity.MEDIUM,
                        "GATE_MISSING",
                        claimed="CI runs drift checks",
                        actual=f"No check_drift reference in {f.name}",
                        evidence=str(f.relative_to(PROJECT_ROOT)),
                        fix_type="CONFIG_FIX",
                    )
        else:
            audit.check_failed("No workflow files in .github/workflows/")
            audit.add_finding(
                Severity.MEDIUM,
                "GATE_MISSING",
                claimed="CI pipeline exists",
                actual="No .yml/.yaml files in .github/workflows/",
                evidence=str(workflows_dir),
                fix_type="CONFIG_FIX",
            )
    else:
        audit.check_info("No .github/workflows/ directory found")

    # ── 10C: Repo Cleanliness ──
    print("\n--- 10C. Repo Cleanliness ---")

    # Uncommitted changes
    status = _git_status()
    if status:
        modified = [line for line in status.splitlines() if line.startswith(" M") or line.startswith("M ")]
        untracked = [line for line in status.splitlines() if line.startswith("??")]
        if modified:
            audit.check_info(f"{len(modified)} modified file(s)")
            for m in modified[:5]:
                print(f"         {m}")
        if untracked:
            audit.check_info(f"{len(untracked)} untracked file(s)")
    else:
        audit.check_passed("Working tree clean")

    # .py files in project root that should be in subdirectory
    root_py = [f for f in PROJECT_ROOT.glob("*.py") if f.name not in {"conftest.py", "setup.py", "pyproject.toml"}]
    if root_py:
        audit.check_info(f"{len(root_py)} .py file(s) in project root")
        for f in root_py:
            print(f"         {f.name}")
    else:
        audit.check_passed("No stray .py files in project root")

    # Scratch/temp files
    scratch_patterns = ["*.tmp", "*.bak", "*.swp", "*.orig", "scratch_*"]
    scratch_files = []
    for pattern in scratch_patterns:
        scratch_files.extend(PROJECT_ROOT.glob(pattern))
    if scratch_files:
        audit.check_info(f"{len(scratch_files)} scratch/temp file(s) in project root")
        for f in scratch_files:
            print(f"         {f.name}")
    else:
        audit.check_passed("No scratch/temp files in project root")

    audit.run_and_exit()


if __name__ == "__main__":
    main()
