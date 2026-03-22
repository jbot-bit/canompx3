#!/usr/bin/env python3
"""Stage-gate guard v2.2: hard-blocks production edits outside approved scope.

Enforcement layers:
1. Explicitly safe files → pass
2. Non-production files → pass
3. Production code → requires STAGE_STATE.md with correct mode + scope
4. NEVER_TRIVIAL files → cannot use TRIVIAL mode

v2.2 fixes (from code review):
- Scope lock parser handles both markdown (## Scope Lock) and YAML (scope_lock:) formats
- Removed Path.name fallback in scope matching (was a bypass vector)
- Directory entries in scope_lock now match files inside them via startswith
- ALWAYS_ALLOWED test_ check moved to filename-only (was matching backtest_*.py)
- SAFE_SCRIPT_PREFIXES uses startswith only (removed substring fallback)
- NEVER_TRIVIAL expanded: execution_engine, risk_manager, paper_trader, walkforward, etc.
"""

import json
import sys
from pathlib import Path

STAGE_STATE = Path("docs/runtime/STAGE_STATE.md")

# ── Explicitly safe DIRECTORIES (path substring match) ────────────────
SAFE_DIRS = (
    "tests/",
    ".claude/",
    "docs/runtime/",
    "docs/plans/",
    "scripts/reports/",
    "scripts/infra/",
)

# ── Explicitly safe FILENAMES (matched against filename only) ─────────
SAFE_FILENAMES = (
    "HANDOFF.md",
    "REPO_MAP.md",
    ".gitignore",
    ".gitkeep",
)

# ── Safe script prefixes (startswith only, no substring) ──────────────
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
    "pipeline/db_config",
    "pipeline/session_guard",
    # Trading app core
    "trading_app/config.py",
    "trading_app/outcome_builder",
    "trading_app/strategy_discovery",
    "trading_app/strategy_validator",
    "trading_app/entry_rules",
    "trading_app/live_config",
    "trading_app/live/",
    "trading_app/execution_engine",
    "trading_app/execution_spec",
    "trading_app/risk_manager",
    "trading_app/paper_trader",
    "trading_app/walkforward",
    "trading_app/strategy_fitness",
    "trading_app/mcp_server",
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
    """Parse scope_lock from either markdown or YAML format.

    Markdown: ## Scope Lock section with - path items
    YAML: scope_lock: key with - path items or inline [list]
    """
    paths = []

    # Format 1: Markdown section (## Scope Lock)
    if "## Scope Lock" in content:
        scope_text = content.split("## Scope Lock")[1].split("##")[0]
        for line in scope_text.strip().splitlines():
            cleaned = line.strip().lstrip("- ").strip("`").strip()
            if cleaned:
                paths.append(cleaned.replace("\\", "/"))
        return paths

    # Format 2: YAML key (scope_lock:)
    in_scope = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("scope_lock:"):
            rest = stripped.split(":", 1)[1].strip()
            # Inline list: scope_lock: [file1.py, file2.py]
            if rest.startswith("["):
                items = rest.strip("[]").split(",")
                for item in items:
                    cleaned = item.strip().strip("'\"").replace("\\", "/")
                    if cleaned:
                        paths.append(cleaned)
                return paths
            in_scope = True
            continue
        if in_scope:
            if stripped.startswith("- "):
                val = stripped[2:].strip().strip("'\"").replace("\\", "/")
                if val:
                    paths.append(val)
            elif stripped and not stripped.startswith("#"):
                break  # next YAML key

    return paths


def is_always_allowed(file_path):
    """Check if file is explicitly safe (never needs stage gate)."""
    fname = Path(file_path).name

    # Safe directories (substring in path)
    if any(d in file_path for d in SAFE_DIRS):
        return True

    # Safe filenames (exact filename match, not substring)
    if fname in SAFE_FILENAMES:
        return True

    # Test files by filename prefix (not path substring — avoids backtest_*.py)
    if fname.startswith("test_"):
        return True

    # Safe script prefixes (startswith only)
    if any(file_path.startswith(p) for p in SAFE_SCRIPT_PREFIXES):
        return True

    return False


def matches_scope(file_path, scope_paths):
    """Check if file_path matches any entry in scope_lock.

    Handles:
    - Exact path match: pipeline/dst.py
    - Relative suffix match: file_path ends with /scope_entry
    - Directory patterns: scope entry ends with / → file must be inside
    """
    for sp in scope_paths:
        if sp.endswith("/"):
            # Directory pattern: trading_app/live/ matches trading_app/live/broker.py
            if file_path.startswith(sp):
                return True
        else:
            # Exact or suffix match (no filename-only fallback)
            if file_path == sp or file_path.endswith("/" + sp):
                return True
    return False


def main():
    event = json.load(sys.stdin)
    file_path = normalize(event.get("tool_input", {}).get("file_path", ""))

    # ── Layer 1: Explicitly safe files ────────────────────────────────
    if is_always_allowed(file_path):
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
        if not matches_scope(file_path, scope_lock):
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
