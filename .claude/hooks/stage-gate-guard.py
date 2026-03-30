#!/usr/bin/env python3
"""Stage-gate guard v3.0: multi-agent safe, hard-blocks production edits outside approved scope.

Enforcement layers:
1. Explicitly safe files → pass
2. Non-production files → pass
3. Production code → requires active stage with correct mode + scope
4. NEVER_TRIVIAL files → cannot use TRIVIAL mode

v3.0 changes (multi-agent):
- Reads ALL stage files: docs/runtime/STAGE_STATE.md + docs/runtime/stages/*.md
- Edit allowed if ANY active stage permits it (union of scope_locks)
- Auto-trivial writes to stages/auto_trivial.md (not the shared file)
- Codex writes to stages/codex.md, Claude uses STAGE_STATE.md — no conflicts
- Blast-radius required per-stage (unchanged)

v2.3 fixes (from simulation):
- F1: Auto-creates TRIVIAL STAGE_STATE for non-core files when no state exists
- F2: IMPLEMENTATION without scope_lock → blocks (was wide-open)
- F3: Better error messages with exact remediation steps
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Primary stage file (Claude Code convention)
STAGE_STATE = Path("docs/runtime/STAGE_STATE.md")
# Directory for additional agent stage files (Codex, worktrees, etc.)
STAGES_DIR = Path("docs/runtime/stages")

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


def parse_blast_radius(content):
    """Parse blast_radius from single-line YAML, multi-line YAML list, or markdown section.

    Accepted formats:
      blast_radius: entry_rules.py, config.py; tests: test_entry_rules.py
      blast_radius:\n  - entry_rules.py\n  - config.py
      ## Blast Radius\n- entry_rules.py\n- config.py
    """
    # Format 1: Markdown section (## Blast Radius)
    if "## Blast Radius" in content:
        section = content.split("## Blast Radius")[1].split("##")[0].split("---")[0]
        text = section.strip()
        if text:
            return text

    # Format 2: YAML (single-line or multi-line list)
    found_key = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("blast_radius:"):
            value = stripped.split(":", 1)[1].strip()
            if value:
                return value  # Single-line: blast_radius: some text here
            found_key = True
            continue
        if found_key:
            if stripped.startswith("- "):
                # Multi-line YAML list — collect all items
                items = [stripped[2:].strip()]
                remaining = content.split(line, 1)[1] if line in content else ""
                for next_line in remaining.splitlines():
                    ns = next_line.strip()
                    if ns.startswith("- "):
                        items.append(ns[2:].strip())
                    elif ns and not ns.startswith("#"):
                        break
                return "; ".join(items)
            elif stripped and not stripped.startswith("#"):
                break  # Next YAML key, no list found

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


def load_all_stages():
    """Load all stage files: primary STAGE_STATE.md + stages/*.md.

    Returns list of (path, content, mode, scope_lock, blast_radius) tuples.
    """
    stages = []

    # Primary file (Claude Code convention)
    if STAGE_STATE.exists():
        try:
            content = STAGE_STATE.read_text(encoding="utf-8")
            mode = parse_field(content, "mode") or parse_field(content, "stage")
            if mode:
                stages.append((
                    str(STAGE_STATE),
                    content,
                    mode,
                    parse_scope_lock(content),
                    parse_blast_radius(content),
                ))
        except (OSError, UnicodeDecodeError):
            pass

    # Additional stage files (Codex, worktrees, auto-trivial, etc.)
    if STAGES_DIR.is_dir():
        for f in sorted(STAGES_DIR.glob("*.md")):
            try:
                content = f.read_text(encoding="utf-8")
                mode = parse_field(content, "mode")
                if mode:
                    stages.append((
                        str(f),
                        content,
                        mode,
                        parse_scope_lock(content),
                        parse_blast_radius(content),
                    ))
            except (OSError, UnicodeDecodeError):
                pass

    return stages


def main():
    event = json.load(sys.stdin)
    file_path = normalize(event.get("tool_input", {}).get("file_path", ""))

    # ── Layer 1: Explicitly safe files ────────────────────────────────
    if is_always_allowed(file_path):
        sys.exit(0)

    # ── Layer 2: Not production code → pass ───────────────────────────
    if not any(marker in file_path for marker in PROD_PATHS):
        sys.exit(0)

    # ── Layer 3: Production code — enforce stage gate (multi-agent) ───

    stages = load_all_stages()

    if not stages:
        # No stage files at all — auto-create TRIVIAL for non-core
        is_core = any(marker in file_path for marker in NEVER_TRIVIAL)
        if not is_core:
            STAGES_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            auto_file = STAGES_DIR / "auto_trivial.md"
            auto_file.write_text(
                f"---\ntask: auto-trivial edit of {file_path}\n"
                f"mode: TRIVIAL\nscope: [{file_path}]\n"
                f"updated: {now}\nagent: auto\n---\n",
                encoding="utf-8",
            )
            print(
                f"STAGE-GATE: Auto-created TRIVIAL state for {file_path}.\n"
                f"  Non-core file — proceeding. Delete docs/runtime/stages/auto_trivial.md when done.",
                file=sys.stderr,
            )
            sys.exit(0)
        # Core file with no state — hard block
        print(
            f"STAGE-GATE BLOCK: No active stage. {file_path} is a core file.\n"
            f"  Core files cannot use TRIVIAL mode — full staging required.\n"
            f"  → Run /stage-gate to classify and create an approved stage.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Check if ANY active stage permits this edit ───────────────────

    # Pass 1: TRIVIAL stages allow any non-core file
    for path, content, mode, scope, blast in stages:
        if mode == "TRIVIAL":
            if any(marker in file_path for marker in NEVER_TRIVIAL):
                continue  # This TRIVIAL stage can't help with core files
            sys.exit(0)  # Non-core + any TRIVIAL stage → allowed

    # Pass 2: IMPLEMENTATION stages — check scope_lock
    impl_stages = [(p, c, m, s, b) for p, c, m, s, b in stages if m == "IMPLEMENTATION"]

    for path, content, mode, scope, blast in impl_stages:
        if not scope:
            continue  # No scope_lock = this stage can't help
        if not blast or len(blast.strip()) < 30:
            continue  # No blast_radius = this stage is incomplete
        if matches_scope(file_path, scope):
            sys.exit(0)  # File is in this stage's scope — allowed

    # Pass 3: No stage permits this edit — explain why

    # Check if it's a core file blocked by TRIVIAL
    is_core = any(marker in file_path for marker in NEVER_TRIVIAL)
    has_trivial = any(m == "TRIVIAL" for _, _, m, _, _ in stages)
    if is_core and has_trivial:
        print(
            f"STAGE-GATE BLOCK: {file_path} is a core file and cannot use TRIVIAL mode.\n"
            f"  → Run /stage-gate with full staging for core files.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Check for non-IMPLEMENTATION modes blocking
    non_impl = [m for _, _, m, _, _ in stages if m not in ("TRIVIAL", "IMPLEMENTATION")]
    if non_impl:
        print(
            f"STAGE-GATE BLOCK: Active stage(s) in {', '.join(non_impl)} mode, not IMPLEMENTATION.\n"
            f"  Cannot edit production code ({file_path}) during {non_impl[0]}.\n"
            f"  → Finish the current stage, then /stage-gate to IMPLEMENTATION",
            file=sys.stderr,
        )
        sys.exit(2)

    # IMPLEMENTATION stages exist but file not in any scope_lock
    if impl_stages:
        all_scope = []
        for _, _, _, scope, _ in impl_stages:
            if scope:
                all_scope.extend(scope)
        # Check for missing blast_radius specifically
        stages_no_blast = [p for p, _, m, s, b in impl_stages if s and (not b or len(b.strip()) < 30)]
        if stages_no_blast:
            print(
                f"STAGE-GATE BLOCK: blast_radius missing or too brief (<30 chars) in: {', '.join(stages_no_blast)}\n"
                f"  Every IMPLEMENTATION stage needs blast_radius listing affected files, tests, and downstream consumers.\n"
                f"  → Add blast_radius to the stage file before editing production code.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            f"STAGE-GATE BLOCK: {file_path} not in scope_lock.\n"
            f"  Allowed: {', '.join(all_scope) if all_scope else '(no scope defined)'}\n"
            f"  → Edit your stage file to add this file if genuinely needed\n"
            f"  → Otherwise defer to a later stage (scope creep)",
            file=sys.stderr,
        )
        sys.exit(2)

    # Shouldn't reach here, but safety net
    print(
        f"STAGE-GATE BLOCK: No stage permits editing {file_path}.\n"
        f"  → Run /stage-gate to create an approved stage.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
