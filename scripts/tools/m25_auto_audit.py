#!/usr/bin/env python
"""
M2.5 auto-audit: automatically audits changed pipeline/trading_app files.

IMPORTANT: M2.5 is a SECOND-OPINION scanner only. Its findings are
UNVERIFIED SUGGESTIONS — they have a high false-positive rate (~70% in
stress testing). NEVER act on M2.5 findings without Claude Code verification.

Workflow:
    1. M2.5 flags potential issues (this script)
    2. Claude Code cross-references each finding against actual code
    3. Only VERIFIED findings get implemented
    4. CLAUDE.md is the authority — M2.5 suggestions that contradict it are wrong

Usage:
    # Audit uncommitted changes vs HEAD
    python scripts/tools/m25_auto_audit.py

    # Audit staged files only (used by pre-commit hook)
    python scripts/tools/m25_auto_audit.py --staged

    # Audit changes since a specific ref
    python scripts/tools/m25_auto_audit.py --since HEAD~3

    # Audit specific files
    python scripts/tools/m25_auto_audit.py --files pipeline/dst.py trading_app/config.py

    # Quick mode: only first critical finding per file, no saved output
    python scripts/tools/m25_auto_audit.py --quick

    # Non-blocking: print summary only, don't fail on findings
    python scripts/tools/m25_auto_audit.py --advisory

Setup:
    Set MINIMAX_API_KEY in your .env or environment.
    Without it, the script silently exits 0 (no-op).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.m25_audit import load_api_key, read_files, audit, AUDIT_MODES  # noqa: E402

# ── File → audit mode mapping ────────────────────────────────────────
# Files that touch SQL joins or daily_features get "joins" mode.
# Files that touch strategy logic get "bias" mode.
# Everything else gets "bugs" mode.
MODE_MAP = {
    # Pipeline — joins/data integrity focus
    "pipeline/build_daily_features.py": "joins",
    "pipeline/build_bars_5m.py": "joins",
    "pipeline/init_db.py": "joins",
    # Pipeline — bug focus
    "pipeline/ingest_dbn.py": "bugs",
    "pipeline/dst.py": "bugs",
    "pipeline/check_drift.py": "bugs",
    "pipeline/cost_model.py": "bugs",
    "pipeline/asset_configs.py": "bugs",
    "pipeline/health_check.py": "bugs",
    # Trading app — bias/statistical focus
    "trading_app/outcome_builder.py": "bias",
    "trading_app/strategy_discovery.py": "bias",
    "trading_app/strategy_validator.py": "bias",
    # Trading app — bug focus
    "trading_app/config.py": "bugs",
    "trading_app/entry_rules.py": "bugs",
    "trading_app/paper_trader.py": "bugs",
    "trading_app/live_config.py": "bugs",
    "trading_app/mcp_server.py": "bugs",
    # ML module — bias focus (ML-specific patterns)
    "trading_app/ml/config.py": "bias",
    "trading_app/ml/features.py": "bias",
    "trading_app/ml/meta_label.py": "bias",
    "trading_app/ml/cpcv.py": "bias",
    "trading_app/ml/predict_live.py": "bugs",
    "trading_app/ml/evaluate.py": "bias",
    "trading_app/ml/evaluate_validated.py": "bias",
    "trading_app/ml/importance.py": "bugs",
}

DEFAULT_MODE = "bugs"

# Only audit files in these directories (trading_app/ includes trading_app/ml/)
AUDIT_DIRS = ("pipeline/", "trading_app/")


def get_changed_files(
    staged: bool = False,
    since: str | None = None,
) -> list[str]:
    """Get changed .py files in pipeline/ or trading_app/ via git."""
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    elif since:
        cmd = ["git", "diff", "--name-only", since]
    else:
        # Uncommitted changes (both staged and unstaged) vs HEAD
        cmd = ["git", "diff", "--name-only", "HEAD"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        if result.stderr.strip():
            print(f"  WARNING: git failed — {result.stderr.strip()}")
        return []

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    # Filter to .py files in audit directories
    return [f for f in files if f.endswith(".py") and any(f.startswith(d) for d in AUDIT_DIRS)]


def pick_mode(filepath: str) -> str:
    """Select the best audit mode for a file."""
    return MODE_MAP.get(filepath, DEFAULT_MODE)


def run_audit(
    files: list[str],
    quick: bool = False,
    advisory: bool = False,
    output_dir: Path | None = None,
) -> int:
    """Run M2.5 audit on files. Returns 0 if clean, 1 if findings."""
    try:
        api_key = load_api_key()
    except SystemExit:
        # No API key — silent skip
        return 0

    findings_count = 0
    total_files = len(files)

    for i, filepath in enumerate(files, 1):
        mode = pick_mode(filepath)
        print(f"  [{i}/{total_files}] {filepath} ({mode})...", end="", flush=True)

        try:
            file_content = read_files([filepath])
        except SystemExit:
            print(" SKIP (file not found)")
            continue

        system_prompt = AUDIT_MODES[mode]
        if quick:
            system_prompt += (
                "\n\nBe extremely concise. Report ONLY the single most critical "
                "finding (if any). If no real issues, say 'CLEAN'."
            )

        try:
            # Always use Lightning for automated scans — faster, cheaper, same accuracy
            result = audit(file_content, system_prompt, api_key, fast=True)
        except Exception as e:
            print(f" ERROR ({type(e).__name__})")
            continue

        # Check if findings exist (heuristic: look for severity markers)
        has_findings = any(
            marker in result.upper() for marker in ["CRITICAL", "WARNING", "BUG", "ERROR", "VULNERABILITY", "ISSUE"]
        )
        is_clean = "CLEAN" in result.upper() and not has_findings

        if is_clean:
            print(" CLEAN")
        else:
            findings_count += 1
            print(f" FINDINGS")

        # Save detailed output
        if output_dir and not quick:
            safe_name = filepath.replace("/", "_").replace("\\", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            out_path = output_dir / f"m25_auto_{safe_name}_{ts}.md"
            out_path.write_text(
                f"# M2.5 Auto-Audit: {filepath}\n"
                f"**Mode:** {mode} | **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"{result}\n",
                encoding="utf-8",
            )

    # Summary
    print()
    if findings_count == 0:
        print(f"  M2.5 audit: {total_files} file(s) CLEAN")
    else:
        print(f"  M2.5 audit: {findings_count}/{total_files} file(s) have findings")
        if not advisory:
            print("  (Review findings above. Use --advisory to suppress exit code.)")

    if advisory:
        return 0
    return 1 if findings_count > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="M2.5 auto-audit for changed pipeline/trading_app files",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Audit staged files only (for pre-commit hook)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Audit changes since this git ref (e.g., HEAD~3, main)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Audit specific files (overrides git detection)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: concise output, no saved files",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Non-blocking: always exit 0 regardless of findings",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save detailed audit output (default: research/output/)",
    )
    args = parser.parse_args()

    # Check for API key early — silent exit if not set
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("MINIMAX_API_KEY"):
        # No API key = no-op (don't break workflows)
        sys.exit(0)

    # Determine files to audit
    if args.files:
        files = [f for f in args.files if f.endswith(".py") and any(f.startswith(d) for d in AUDIT_DIRS)]
    else:
        files = get_changed_files(staged=args.staged, since=args.since)

    if not files:
        print("  M2.5 auto-audit: no pipeline/trading_app .py files changed")
        sys.exit(0)

    # Output directory
    output_dir = None
    if not args.quick:
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "research" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  M2.5 auto-audit: {len(files)} file(s) to review")
    exit_code = run_audit(files, quick=args.quick, advisory=args.advisory, output_dir=output_dir)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
