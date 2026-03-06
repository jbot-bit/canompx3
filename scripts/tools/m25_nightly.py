#!/usr/bin/env python
"""
M2.5 nightly batch audit — runs all core pipeline files, saves results to
research/output/m25/, and writes a one-page summary you can glance at.

Designed to run at 2 AM via Windows Task Scheduler so it's waiting for you
in the morning. Uses Lightning model throughout to stay within budget.

Usage:
    python scripts/tools/m25_nightly.py          # Run full batch
    python scripts/tools/m25_nightly.py --setup  # Print Task Scheduler setup command
    python scripts/tools/m25_nightly.py --summary # Print last summary only (no API calls)
"""

from __future__ import annotations

import sys
from datetime import datetime, date
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.tools.m25_audit import (  # noqa: E402
    AUDIT_MODES,
    load_api_key,
    read_files,
    audit,
    show_budget,
)

OUTPUT_DIR = PROJECT / "research" / "output" / "m25"

# All core files to audit nightly, with the most appropriate mode.
# Order matters: cheap/quick files first so partial runs are still useful.
NIGHTLY_TARGETS: list[tuple[str, str]] = [
    # Pipeline — data integrity
    ("pipeline/dst.py", "bugs"),
    ("pipeline/cost_model.py", "bugs"),
    ("pipeline/asset_configs.py", "bugs"),
    ("pipeline/build_daily_features.py", "joins"),
    ("pipeline/build_bars_5m.py", "joins"),
    ("pipeline/ingest_dbn.py", "bugs"),
    # Trading app — statistical correctness
    ("trading_app/outcome_builder.py", "bias"),
    ("trading_app/strategy_discovery.py", "bias"),
    ("trading_app/strategy_validator.py", "bias"),
    ("trading_app/config.py", "bugs"),
    # ML — bias focus
    ("trading_app/ml/meta_label.py", "bias"),
    ("trading_app/ml/cpcv.py", "bias"),
]

# Sentinel file read by session-start hook to remind you of pending findings
SENTINEL = PROJECT / ".m25_pending_review"


def run_nightly(quick: bool = False) -> None:
    try:
        api_key = load_api_key()
    except SystemExit:
        print("ERROR: MINIMAX_API_KEY not set — skipping nightly audit.")
        return

    today = date.today().isoformat()
    out_dir = OUTPUT_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = [
        f"# M2.5 Nightly Audit — {today}",
        f"*Generated {datetime.now().strftime('%H:%M')} | Lightning model | {len(NIGHTLY_TARGETS)} files*",
        "",
        "| File | Mode | Result |",
        "|---|---|---|",
    ]

    findings_files: list[str] = []
    clean_count = 0

    for i, (filepath, mode) in enumerate(NIGHTLY_TARGETS, 1):
        label = f"[{i}/{len(NIGHTLY_TARGETS)}] {filepath}"
        print(f"  {label} ({mode})...", end="", flush=True)

        try:
            file_content = read_files([filepath])
        except SystemExit:
            print(" SKIP")
            summary_lines.append(f"| `{filepath}` | {mode} | ⚪ SKIP |")
            continue

        system_prompt = AUDIT_MODES[mode]
        if quick:
            system_prompt += (
                "\n\nBe extremely concise. Report ONLY the single most critical "
                "finding (if any). If no real issues, say 'CLEAN'."
            )

        try:
            result = audit(file_content, system_prompt, api_key, fast=True)
        except Exception as e:
            print(f" ERROR ({e})")
            summary_lines.append(f"| `{filepath}` | {mode} | 🔴 ERROR |")
            continue

        # Classify result
        has_findings = any(m in result.upper() for m in ["CRITICAL", "HIGH", "BUG", "VULNERABILITY", "WRONG"])
        is_clean = "CLEAN" in result.upper() and not has_findings

        if is_clean:
            status = "✅ CLEAN"
            clean_count += 1
            print(" CLEAN")
        else:
            status = "⚠️ FINDINGS"
            findings_files.append(filepath)
            print(" FINDINGS")

        summary_lines.append(f"| `{filepath}` | {mode} | {status} |")

        # Save detailed result
        safe = filepath.replace("/", "_").replace("\\", "_")
        detail_path = out_dir / f"{safe}.md"
        detail_path.write_text(
            f"# M2.5 Audit: {filepath}\n**Mode:** {mode} | **Date:** {today} | **Model:** Lightning\n\n{result}\n",
            encoding="utf-8",
        )

    # Summary footer
    total = len(NIGHTLY_TARGETS)
    summary_lines += [
        "",
        f"**{clean_count}/{total} clean** | **{len(findings_files)} with findings**",
        "",
    ]

    if findings_files:
        summary_lines += [
            "## Files needing review",
            "",
            *[f"- `{f}`" for f in findings_files],
            "",
            f"Details in: `research/output/m25/{today}/`",
        ]
        # Write sentinel so reminder hook fires next session
        SENTINEL.write_text(
            f"date={today}\nfiles={','.join(findings_files)}\n",
            encoding="utf-8",
        )
    else:
        # Clear sentinel if all clean
        if SENTINEL.exists():
            SENTINEL.unlink()

    summary_path = OUTPUT_DIR / f"{today}_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print(f"  Summary: {clean_count}/{total} clean → {summary_path}")
    if findings_files:
        print(f"  ⚠ Findings in {len(findings_files)} file(s) — review before next trade")
    show_budget()


def print_summary() -> None:
    """Print the most recent nightly summary without calling the API."""
    if not OUTPUT_DIR.exists():
        print("No nightly summaries found yet. Run without --summary first.")
        return
    summaries = sorted(OUTPUT_DIR.glob("*_summary.md"))
    if not summaries:
        print("No nightly summaries found yet. Run without --summary first.")
        return
    latest = summaries[-1]
    print(latest.read_text(encoding="utf-8"))


def print_setup() -> None:
    """Print the Windows Task Scheduler command to register the nightly job."""
    python = sys.executable.replace("\\", "\\\\")
    script = str(Path(__file__).resolve()).replace("\\", "\\\\")
    project = str(PROJECT.resolve()).replace("\\", "\\\\")

    print("# Run this ONCE to register the nightly M2.5 audit at 02:00 AM:")
    print()
    print(f'schtasks /Create /TN "M2.5 Nightly Audit" /TR "{python} {script}" /SC DAILY /ST 02:00 /F')
    print()
    print("# To verify it's registered:")
    print('schtasks /Query /TN "M2.5 Nightly Audit"')
    print()
    print("# To remove it:")
    print('schtasks /Delete /TN "M2.5 Nightly Audit" /F')
    print()
    print(f"# Note: runs in cwd={project}")
    print("# Make sure MINIMAX_API_KEY is in your user environment variables,")
    print("# not just in .env (Task Scheduler doesn't load .env files).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="M2.5 nightly batch audit")
    parser.add_argument("--setup", action="store_true", help="Print Task Scheduler setup command")
    parser.add_argument("--summary", action="store_true", help="Print last summary, no API calls")
    parser.add_argument("--quick", action="store_true", help="One finding per file, faster")
    args = parser.parse_args()

    if args.setup:
        print_setup()
    elif args.summary:
        print_summary()
    else:
        run_nightly(quick=args.quick)
