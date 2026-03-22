#!/usr/bin/env python3
"""Stage-gate guard v2.1: hard-blocks production edits outside approved scope.

Enforcement layers:
1. Explicitly safe files → pass
2. Non-production files → pass
3. Production code → requires STAGE_STATE.md with correct mode + scope
4. NEVER_TRIVIAL files → cannot use TRIVIAL mode
"""

import json
import sys
from pathlib import Path

STAGE_STATE = Path("docs/runtime/STAGE_STATE.md")

# ── Explicitly safe: never gated ──────────────────────────────────────
ALWAYS_ALLOWED = (
    # Tests
    "tests/",
    "test_",
    # Claude config / meta
    ".claude/",
    # Stage state + design output (skills must write these)
    "docs/runtime/",
    "docs/plans/",
    # Session handoff + auto-generated
    "HANDOFF.md",
    "REPO_MAP.md",
    ".gitignore",
    ".gitkeep",
    # Safe script directories (read-only reports, infra tooling)
    "scripts/reports/",
    "scripts/infra/",
)

# Prefix-matched safe scripts
SAFE_SCRIPT_PREFIXES = (
    "scripts/tools/gen_",
    "scripts/tools/project_pulse",
    "scripts/tools/sync_pinecone",
    "scripts/tools/generate_trade_sheet",
)

# ── Production code: requires stage gate ──────────────────────────────
PROD_PATHS = (
    "pipeline/",
    "trading_app/",
    "scripts/",
)

# ── Files that CANNOT use TRIVIAL mode (defense-in-depth) ────────────
NEVER_TRIVIAL = (
    # Pipeline core
    "pipeline/build_daily_features",
    "pipeline/build_bars_5m",
    "pipeline/ingest_dbn",
    "pipeline/init_db",
    "pipeline/run_pipeline",
    "pipeline/dst.py",
    "pipeline/check_drift",
    "pipeline/asset_configs",
    "pipeline/cost_model",
    "pipeline/paths.py",
    "pipeline/health_check",
    # Trading app core
    "trading_app/config.py",
    "trading_app/outcome_builder",
    "trading_app/strategy_discovery",
    "trading_app/strategy_validator",
    "trading_app/entry_rules",
    "trading_app/live_config",
    "trading_app/live/",
    # Protected scripts
    "scripts/tools/build_edge_families",
    "scripts/tools/audit_behavioral",
)


def normalize(p):
    p = p.replace("\\", "/")
    if "canompx3/" in p:
        p = p.split("canompx3/", 1)[-1]
    return p


def parse_field(content, field):
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped.split(":", 1)[1].strip()
    return None


def parse_scope_lock(content):
    if "## Scope Lock" not in content:
        return []
    scope_text = content.split("## Scope Lock")[1].split("##")[0]
    paths = []
    for line in scope_text.strip().splitlines():
        cleaned = line.strip().lstrip("- ").strip("`").strip()
        if cleaned:
            paths.append(cleaned.replace("\\", "/"))
    return paths


def main():
    event = json.load(sys.stdin)
    file_path = normalize(event.get("tool_input", {}).get("file_path", ""))

    # ── Layer 1: Explicitly safe files ────────────────────────────────
    if any(marker in file_path for marker in ALWAYS_ALLOWED):
        sys.exit(0)

    # Safe script prefixes
    if any(file_path.startswith(p) or p in file_path for p in SAFE_SCRIPT_PREFIXES):
        sys.exit(0)

    # ── Layer 2: Not production code → pass ───────────────────────────
    if not any(marker in file_path for marker in PROD_PATHS):
        sys.exit(0)

    # ── Layer 3: Production code — enforce stage gate ─────────────────

    if not STAGE_STATE.exists():
        print(
            "STAGE-GATE BLOCK: No active stage.\n"
            "  Run /stage-gate to classify before editing production code.\n"
            "  For quick mechanical fixes: /stage-gate trivial fix in [filename]",
            file=sys.stderr,
        )
        sys.exit(2)

    content = STAGE_STATE.read_text(encoding="utf-8")
    mode = parse_field(content, "mode")

    if not mode:
        print(
            "STAGE-GATE BLOCK: STAGE_STATE.md has no mode field.\n"
            "  Run /stage-gate to reclassify.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── TRIVIAL mode — hardened ───────────────────────────────────────
    if mode == "TRIVIAL":
        if any(marker in file_path for marker in NEVER_TRIVIAL):
            print(
                f"STAGE-GATE BLOCK: {file_path} cannot use TRIVIAL mode.\n"
                f"  This file touches pipeline logic, config, schema, or validation.\n"
                f"  Reclassify via /stage-gate with full staging.",
                file=sys.stderr,
            )
            sys.exit(2)
        sys.exit(0)

    # ── Non-IMPLEMENTATION mode → block ───────────────────────────────
    if mode != "IMPLEMENTATION":
        print(
            f"STAGE-GATE BLOCK: Mode is {mode}, not IMPLEMENTATION.\n"
            f"  Cannot edit production code ({file_path}) during {mode}.\n"
            f"  → Finish {mode} first, then /stage-gate to IMPLEMENTATION\n"
            f"  → /stage-gate reclassify (abandons current stage)\n"
            f"  → /stage-gate trivial (only if non-core mechanical fix)",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── IMPLEMENTATION mode — check scope lock ────────────────────────
    scope_lock = parse_scope_lock(content)
    if scope_lock:
        if not any(
            file_path.endswith(sp) or sp.endswith(file_path)
            or Path(file_path).name == Path(sp).name
            for sp in scope_lock
        ):
            print(
                f"STAGE-GATE BLOCK: {file_path} not in scope_lock.\n"
                f"  Allowed: {', '.join(scope_lock)}\n"
                f"  → Add to scope_lock in docs/runtime/STAGE_STATE.md if needed\n"
                f"  → Otherwise this is scope creep — defer to later stage",
                file=sys.stderr,
            )
            sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
