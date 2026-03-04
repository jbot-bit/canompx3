#!/usr/bin/env python
"""
M2.5 ML Integration Audit — self-identifying, one-shot.

Auto-discovers all ML module files + context docs, sends to MiniMax M2.5
with a domain-expert prompt for pre-production ML review.

Usage:
    python scripts/tools/m25_ml_audit.py              # full audit, saves to research/output/
    python scripts/tools/m25_ml_audit.py --dry-run     # show what files would be sent
    python scripts/tools/m25_ml_audit.py --output X.md # custom output path

Setup:
    MINIMAX_API_KEY must be set in .env or environment.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.m25_audit import load_api_key, audit, ARCHITECTURE_CONTEXT  # noqa: E402

# ── Self-identification: what to send ────────────────────────────────

# ML module — the audit target
ML_DIR = PROJECT_ROOT / "trading_app" / "ml"

# Context docs — grounding for the reviewer
CONTEXT_DOCS = [
    "CLAUDE.md",
    "REPO_MAP.md",
    "MARKET_PLAYBOOK.md",
    "RESEARCH_RULES.md",
    "TRADING_RULES.md",
]

# Related files outside ml/ that the prompt references
RELATED_FILES = [
    "scripts/migrations/backfill_wf_columns.py",
]

MAX_LINES_PER_FILE = 2000  # truncate very large files


# ── File discovery ───────────────────────────────────────────────────

def discover_files() -> list[tuple[str, Path]]:
    """Auto-discover all files to include. Returns (label, path) pairs."""
    files: list[tuple[str, Path]] = []

    # 1. All .py files in trading_app/ml/ (recursive)
    if ML_DIR.exists():
        for p in sorted(ML_DIR.rglob("*.py")):
            rel = p.relative_to(PROJECT_ROOT)
            files.append((f"[ML] {rel}", p))

    # 2. Context docs from project root
    for name in CONTEXT_DOCS:
        p = PROJECT_ROOT / name
        if p.exists():
            files.append((f"[CONTEXT] {name}", p))

    # 3. Related files
    for name in RELATED_FILES:
        p = PROJECT_ROOT / name
        if p.exists():
            files.append((f"[RELATED] {name}", p))

    return files


def read_all(files: list[tuple[str, Path]]) -> str:
    """Read and concatenate all files with headers."""
    parts = []
    for label, path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            parts.append(f"### {label}\n*Could not read: {e}*")
            continue

        lines = content.splitlines()
        suffix = path.suffix.lower()
        fence = "python" if suffix == ".py" else "markdown"

        if len(lines) > MAX_LINES_PER_FILE:
            content = "\n".join(lines[:MAX_LINES_PER_FILE])
            content += f"\n\n... [TRUNCATED — {len(lines)} total lines, showing first {MAX_LINES_PER_FILE}]"

        parts.append(f"### {label} ({len(lines)} lines)\n```{fence}\n{content}\n```")

    return "\n\n---\n\n".join(parts)


# ── The audit prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are the Head of Quantitative Research at a systematic macro fund managing \
$2B AUM across futures markets. You hold a PhD in Financial Engineering and \
have 18 years building production ML systems for systematic trading — including \
experience at top-tier quantitative firms.

You are conducting a pre-production review of a meta-labeling ML system for \
ORB (Opening Range Breakout) futures trading. Real capital is at risk — be \
direct, avoid hedging, and name serious risks plainly.

{ARCHITECTURE_CONTEXT}

---

**Additional ML-Specific Context (READ BEFORE AUDITING)**

- The core edge is **ORB size (G5+)** — a structural arithmetic fact. Small \
ORBs are unprofitable regardless of ML overlay.
- Entry model bias is documented: E0 purged (3 compounding biases), E1/E2 active.
- **3-way time-ordered split**: 60% train / 20% val (threshold opt) / 20% test (frozen OOS).
- **CPCV** (de Prado) runs within training data as a quality gate — this IS \
walk-forward validation for ML (de Prado's recommended alternative to rolling WF).
- The strategy pipeline (separate from ML) has full parallel walk-forward \
testing in strategy_validator.py. Do NOT claim "no walk-forward" exists.
- Threshold optimization runs ONLY on val set. Test set is frozen and never \
touched by optimization. The test set IS the multiple-testing correction.
- `skip_filter=False` (default) means ML trains only on filter-eligible days — \
it CANNOT learn the filter boundary because filtered-out days aren't in training.
- RESEARCH_RULES.md is binding: mechanism required for every finding.

---

**Audit Scope — Structure your review as graded sections:**

For each section, provide:
- GRADE (A/B/C/D/F) with 1-sentence justification
- SPECIFIC FINDINGS (cite file:line where applicable)
- RECOMMENDED IMPROVEMENTS ranked by impact/effort ratio
- RISK RATING (CRITICAL / HIGH / MEDIUM / LOW)

**Sections:**

1. **FEATURE ENGINEERING** — economic motivation, stationarity, regime robustness, \
missing values, redundancy, what's missing
2. **MODEL ARCHITECTURE** — RF vs alternatives, hyperparameters, per-session design
3. **DATA SPLIT & LEAKAGE** — 3-way split adequacy, threshold optimization leakage, \
CPCV purge/embargo, per-session data volumes
4. **QUALITY GATES & DEPLOYMENT** — gate thresholds, fail-open design, monitoring gaps
5. **STATISTICAL METHODOLOGY** — binary target limitations, threshold search, \
ablation of ML vs filter-only
6. **INSTITUTIONAL GAPS** — regime detection, calibration, position sizing, ensemble

End with: OVERALL ASSESSMENT and top 3 highest-ROI improvements.

**CRITICAL**: Do NOT recommend things already implemented. Check the code and \
architecture context BEFORE flagging. If you're unsure whether a guard exists \
in a file you weren't given, say so explicitly rather than assuming it's missing.
"""


def main():
    parser = argparse.ArgumentParser(
        description="M2.5 ML Integration Audit (self-identifying)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show discovered files and token estimate, don't call API",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Custom output path (default: research/output/m25_ml_audit_YYYYMMDD_HHMM.md)",
    )
    args = parser.parse_args()

    # Discover files
    files = discover_files()
    if not files:
        print("ERROR: No files discovered. Is trading_app/ml/ present?", file=sys.stderr)
        sys.exit(1)

    print(f"M2.5 ML Audit — discovered {len(files)} files:")
    for label, path in files:
        lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        print(f"  {label} ({lines} lines)")

    if args.dry_run:
        content = read_all(files)
        chars = len(content) + len(SYSTEM_PROMPT)
        print(f"\nTotal payload: ~{chars:,} chars (~{chars // 4:,} tokens estimate)")
        print("Dry run — no API call made.")
        sys.exit(0)

    # Load API key
    api_key = load_api_key()

    # Build payload
    print("\nReading files...")
    file_content = read_all(files)
    chars = len(file_content) + len(SYSTEM_PROMPT)
    print(f"Payload: ~{chars:,} chars (~{chars // 4:,} tokens)")
    print("Sending to MiniMax M2.5 (this may take 2-5 minutes)...")

    result = audit(file_content, SYSTEM_PROMPT, api_key, include_context=False)

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_dir = PROJECT_ROOT / "research" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"m25_ml_audit_{ts}.md"

    header = (
        f"# M2.5 ML Integration Audit\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"**Files audited:** {len(files)}\n"
        f"**Payload:** ~{chars:,} chars\n\n"
        f"---\n\n"
    )
    out_path.write_text(header + result, encoding="utf-8")
    print(f"\nSaved to: {out_path}")

    # Also print to stdout (handle Windows cp1252 encoding)
    print("\n" + "=" * 72)
    try:
        print(result)
    except UnicodeEncodeError:
        print(result.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
