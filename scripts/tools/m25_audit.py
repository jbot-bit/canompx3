#!/usr/bin/env python
"""
MiniMax M2.5 independent audit tool.

Second-opinion code review: sends files or queries to MiniMax M2.5
for independent analysis, separate from Claude.

IMPORTANT: M2.5 findings are UNVERIFIED SUGGESTIONS with a ~70% false
positive rate. Always cross-reference with Claude Code before acting.
CLAUDE.md is the authority — M2.5 cannot override it.

Usage:
    # Review a single file
    python scripts/tools/m25_audit.py pipeline/build_daily_features.py

    # Review multiple files
    python scripts/tools/m25_audit.py trading_app/outcome_builder.py trading_app/strategy_validator.py

    # Custom audit prompt
    python scripts/tools/m25_audit.py pipeline/dst.py --prompt "Check for DST bugs and timezone issues"

    # Preset audit modes
    python scripts/tools/m25_audit.py trading_app/outcome_builder.py --mode bias
    python scripts/tools/m25_audit.py pipeline/build_bars_5m.py --mode joins
    python scripts/tools/m25_audit.py trading_app/strategy_validator.py --mode bugs

    # Save output to file
    python scripts/tools/m25_audit.py pipeline/ingest_dbn.py --output audit_result.md

Setup:
    Set MINIMAX_API_KEY in your .env or environment:
        export MINIMAX_API_KEY=your-key-here
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────
API_URL = "https://api.minimax.io/v1/chat/completions"
MODEL = "MiniMax-M2.5"
MAX_CONTEXT = 200000  # M2.5 total context window (input + output) in tokens
MAX_TOKENS = 131072   # 128K default — auto-reduced if input is large
API_TIMEOUT = 600.0   # seconds — large files need more time

# ── Architecture context ────────────────────────────────────────────
# Prepended to every audit to prevent M2.5's known false positive patterns.
# These are FACTS about the codebase that M2.5 consistently gets wrong
# because it can't trace cross-file architecture.
ARCHITECTURE_CONTEXT = """\
IMPORTANT — Known architecture patterns (DO NOT flag these as bugs):

1. DuckDB replacement scans: This project uses DuckDB, which natively references
   in-scope pandas DataFrames in SQL queries (e.g., `SELECT * FROM chunk_df`).
   No `con.register()` is needed. This is documented DuckDB behavior, not a bug.

2. Multi-stage validation pipeline: outcome_builder.py pre-computes ALL parameter
   combinations (grid search). Statistical correction (Benjamini-Hochberg FDR)
   is applied DOWNSTREAM in strategy_validator.py. Do NOT flag the grid search
   as "data snooping" — the correction exists in a different file.

3. ML quality gates: The ML system has a 4-gate quality system (delta_r >= 0,
   CPCV AUC >= 0.50, test AUC > 0.52, skip rate <= 85%). Do NOT evaluate any
   single gate in isolation — they work as a combined system.

4. Fail-open ML design: When ML models are missing or fail, the system takes
   all trades (fail-open). This is intentional for a trading system where
   missing a trade costs more than a false positive.

5. atexit exception handling: `except Exception: pass` in atexit handlers is
   correct — raising during interpreter shutdown produces noise tracebacks.

6. daily_features has 3 rows per (trading_day, symbol) — one per orb_minutes
   (5, 15, 30). Any JOIN MUST include orb_minutes or rows triple.

7. -999.0 as NaN sentinel: sklearn RF cannot handle NaN. The -999.0 fill is
   intentional. For level-proximity features, -999.0 means "no prior level
   exists" — a meaningful domain value. For other features, real values are
   in the 0-5 range, so -999.0 is safely distinct.

8. Cost model: All P&L calculations deduct round-trip friction (commission +
   spread + slippage) via to_r_multiple() in cost_model.py. Costs are handled.

9. Dead instruments: MCL, SIL, M6E, MBT were tested and found to have zero
   ORB edge — this is a validated research finding, not missing data.

Only flag issues that are REAL given these architectural facts.
"""

AUDIT_MODES = {
    "general": (
        "You are a senior quant developer reviewing code for a futures trading pipeline.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — specific praise for sound engineering decisions\n"
        "2. **FINDINGS** — real issues with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — concrete fixes ranked by impact\n\n"
        "Check for:\n"
        "- Bugs, logic errors, off-by-one errors\n"
        "- Error handling gaps (silent failures, swallowed exceptions)\n"
        "- Code quality issues (dead code, unreachable branches)\n"
        "- Resource management (unclosed connections, file handles)\n\n"
        "Be specific. Cite line numbers. Only flag real issues, not style preferences.\n"
        "Do NOT flag patterns explained in the architecture context above."
    ),
    "bias": (
        "You are the Head of Quantitative Research at a systematic futures fund.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — specific praise for bias prevention measures already in place\n"
        "2. **FINDINGS** — real bias risks with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — concrete improvements ranked by impact\n\n"
        "Audit for:\n"
        "- LOOK-AHEAD BIAS: Features derived from post-entry information\n"
        "- SURVIVORSHIP BIAS: Ignoring dead instruments when drawing conclusions\n"
        "- DATA SNOOPING: Multiple hypothesis testing without BH FDR correction "
        "(NOTE: BH FDR IS applied in strategy_validator.py — check if the file "
        "you're reviewing is upstream of that correction before flagging)\n"
        "- OVERFITTING: Too few samples (N<30) or too many free parameters\n"
        "- TRANSACTION COST ILLUSION: Ignoring spread, slippage, or commission "
        "(NOTE: cost_model.py handles this — check before flagging)\n\n"
        "This is a futures ORB breakout system. daily_features has 3 rows per "
        "(trading_day, symbol) — one per orb_minutes (5, 15, 30).\n\n"
        "Be specific. Cite line numbers. Distinguish between issues in THIS file "
        "vs issues handled in OTHER files you cannot see."
    ),
    "joins": (
        "You are a database expert auditing SQL queries in a trading pipeline.\n\n"
        "Structure your review as:\n"
        "1. **SAFE PATTERNS** — correctly implemented SQL patterns\n"
        "2. **FINDINGS** — real SQL issues with severity\n"
        "3. **RECOMMENDATIONS** — fixes with corrected SQL\n\n"
        "Check for:\n"
        "- JOIN correctness: daily_features has 3 rows per (trading_day, symbol) — "
        "one per orb_minutes. Any JOIN MUST include orb_minutes or rows triple.\n"
        "- Missing WHERE clauses that could cause row explosion\n"
        "- LAG()/window functions without proper PARTITION BY or WHERE orb_minutes = 5\n"
        "- Aggregations that could double-count due to JOIN fan-out\n"
        "- DELETE+INSERT idempotency — verify the DELETE range matches the INSERT range\n\n"
        "NOTE: This project uses DuckDB, which can reference in-scope pandas "
        "DataFrames directly in SQL (replacement scans). Do NOT flag this as a bug.\n\n"
        "Cite line numbers. Show the problematic SQL and the fix."
    ),
    "bugs": (
        "You are a Python expert doing a thorough bug hunt on a trading system.\n\n"
        "Structure your review as:\n"
        "1. **WELL-DONE** — good defensive coding patterns you observe\n"
        "2. **FINDINGS** — real bugs with severity (CRITICAL / HIGH / MEDIUM / LOW)\n"
        "3. **RECOMMENDATIONS** — fixes ranked by impact\n\n"
        "Check for:\n"
        "- Type errors, None handling, missing edge cases\n"
        "- Off-by-one errors in date ranges, slicing, indexing\n"
        "- Timezone bugs (UTC vs local, naive vs aware datetimes)\n"
        "- Resource leaks (unclosed files, connections)\n"
        "- Subprocess calls without return code checks\n"
        "- Silent failures (except: pass, catch-all exception handlers)\n"
        "- Variable shadowing, mutation of shared state\n\n"
        "IMPORTANT: `except Exception: pass` in atexit handlers is correct Python "
        "for shutdown cleanup. Do NOT flag this pattern.\n\n"
        "Be specific. Cite line numbers. Only flag actual bugs, not style."
    ),
    "improvements": (
        "You are the Head of Quantitative Research at a systematic macro fund "
        "managing $2B AUM. You hold a PhD in Financial Engineering and have 18 years "
        "building production ML/quant systems for systematic trading.\n\n"
        "Structure your review as graded sections:\n\n"
        "For each section, provide:\n"
        "- GRADE (A/B/C/D/F) with 1-sentence justification\n"
        "- SPECIFIC FINDINGS (cite file:line where applicable)\n"
        "- RECOMMENDED IMPROVEMENTS ranked by impact/effort ratio\n"
        "- RISK RATING for each finding (CRITICAL / HIGH / MEDIUM / LOW)\n\n"
        "Evaluate:\n"
        "1. CODE QUALITY — architecture, error handling, testability\n"
        "2. STATISTICAL RIGOR — bias prevention, validation methodology\n"
        "3. PRODUCTION READINESS — monitoring, alerting, failure modes\n"
        "4. INSTITUTIONAL GAPS — what would a Bloomberg/Two Sigma system have?\n\n"
        "End with: OVERALL ASSESSMENT and top 3 highest-ROI improvements.\n\n"
        "Be specific. Distinguish between what's already well-done, genuine risks, "
        "and concrete improvements. Do NOT recommend things already implemented "
        "(check the architecture context above first)."
    ),
}


def load_api_key() -> str:
    """Load MiniMax API key from env or .env file."""
    load_dotenv()
    key = os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        print("ERROR: MINIMAX_API_KEY not set.", file=sys.stderr)
        print("  Set it in your .env file or run:", file=sys.stderr)
        print("    export MINIMAX_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)
    return key


def read_files(paths: list[str]) -> str:
    """Read and concatenate files with headers."""
    parts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            # Try relative to project root
            path = Path(__file__).parent.parent.parent / p
        if not path.exists():
            print(f"WARNING: File not found: {p}", file=sys.stderr)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        # Truncate very large files
        lines = content.splitlines()
        if len(lines) > 2000:
            content = "\n".join(lines[:2000])
            content += f"\n\n... [TRUNCATED — {len(lines)} total lines, showing first 2000]"
        parts.append(f"### File: {p}\n```python\n{content}\n```")
    if not parts:
        print("ERROR: No readable files provided.", file=sys.stderr)
        sys.exit(1)
    return "\n\n".join(parts)


def audit(
    file_content: str,
    system_prompt: str,
    api_key: str,
    user_prompt: str | None = None,
    *,
    include_context: bool = True,
    timeout: float | None = None,
) -> str:
    """Send code to MiniMax M2.5 for review.

    Args:
        file_content: Code to audit (concatenated file contents).
        system_prompt: The system prompt / audit mode instruction.
        api_key: MiniMax API key.
        user_prompt: Optional additional user instruction prepended to file content.
        include_context: Prepend ARCHITECTURE_CONTEXT to system prompt (default True).
            Set False for custom prompts that provide their own context.
        timeout: API timeout in seconds (default: API_TIMEOUT = 600s).
    """
    if user_prompt:
        full_prompt = f"{user_prompt}\n\n{file_content}"
    else:
        full_prompt = file_content

    # Prepend architecture context to reduce false positives
    full_system = system_prompt
    if include_context:
        full_system = f"{ARCHITECTURE_CONTEXT}\n---\n\n{system_prompt}"

    # Auto-size output tokens: estimate input tokens, leave rest for output
    input_chars = len(full_system) + len(full_prompt)
    input_tokens_est = input_chars // 4  # ~4 chars/token heuristic
    available_output = MAX_CONTEXT - input_tokens_est - 1000  # 1K safety margin
    effective_max_tokens = max(4096, min(MAX_TOKENS, available_output))

    if available_output < MAX_TOKENS:
        print(
            f"  Auto-sized output: {effective_max_tokens:,} tokens "
            f"(input ~{input_tokens_est:,}, context limit {MAX_CONTEXT:,})",
            file=sys.stderr,
        )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": full_prompt},
        ],
        "max_tokens": effective_max_tokens,
        "temperature": 0.1,  # Low temp for precise analysis
    }

    effective_timeout = timeout or API_TIMEOUT

    for attempt in range(2):
        try:
            with httpx.Client(timeout=effective_timeout) as client:
                resp = client.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
            break
        except httpx.ReadTimeout:
            if attempt == 0:
                print("  TIMEOUT — retrying with extended timeout...", file=sys.stderr)
                effective_timeout = min(effective_timeout * 2, 900.0)
            else:
                raise

    if resp.status_code != 200:
        print(f"ERROR: API returned {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="MiniMax M2.5 independent code audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Modes: general, bias, joins, bugs, improvements",
    )
    parser.add_argument("files", nargs="+", help="Files to audit")
    parser.add_argument(
        "--mode",
        choices=AUDIT_MODES.keys(),
        default="general",
        help="Preset audit mode (default: general)",
    )
    parser.add_argument("--prompt", help="Custom audit prompt (overrides --mode)")
    parser.add_argument("--output", "-o", help="Save output to file")
    args = parser.parse_args()

    api_key = load_api_key()
    file_content = read_files(args.files)

    system_prompt = args.prompt if args.prompt else AUDIT_MODES[args.mode]

    print(f"Auditing {len(args.files)} file(s) with M2.5 [{args.mode}]...", file=sys.stderr)

    result = audit(file_content, system_prompt, api_key)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
