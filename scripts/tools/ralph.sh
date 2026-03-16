#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════
# Ralph — Canonical Entrypoint
# ════════════════════════════════════════════════════════════════
# Single command surface for all Ralph operations.
#
# Usage:
#   bash scripts/tools/ralph.sh once                     # Single audit iteration
#   bash scripts/tools/ralph.sh batch [--iterations N]    # Headless batch (default 5)
#   bash scripts/tools/ralph.sh loop                      # Continuous loop (stop: touch ralph_loop.stop)
#   bash scripts/tools/ralph.sh review [--last N]         # Post-batch Opus review
#   bash scripts/tools/ralph.sh audit                     # Run behavioral audit only (no fixes)
#   bash scripts/tools/ralph.sh doctor                    # Preflight health check
#   bash scripts/tools/ralph.sh help                      # This help text
#
# Shared flags:
#   --iterations N    Number of iterations (batch mode, default 5)
#   --last N          Commits to review (review mode, default 5)
#   --scope FILE      Scope to a specific file
#   --output-dir DIR  Override artifact output directory
#   --json            Also write JSON artifacts
#   --review          Run post-batch review automatically after batch
#   --fail-on-findings  Exit 1 if any findings produced
#
# Exit codes:
#   0  Clean success
#   1  Findings produced / review failed / --fail-on-findings triggered
#   2  Runtime/config error
#   3  Repo health failure (doctor mode)
#   4  Verification failure
#
# Artifact output:
#   docs/ralph-loop/logs/              Master logs (existing)
#   artifacts/ralph/<timestamp>/       Per-run artifacts (when --json or --output-dir)
#
# ════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Resolve project root ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Resolve claude CLI ────────────────────────────────────────
CLAUDE=""
for p in "claude" \
         "/c/Users/joshd/.local/bin/claude.exe" \
         "/mnt/c/Users/joshd/.local/bin/claude.exe" \
         "$HOME/.local/bin/claude"; do
    if command -v "$p" &>/dev/null || [[ -x "$p" ]]; then
        CLAUDE="$p"
        break
    fi
done

# ── Delegate scripts ──────────────────────────────────────────
HEADLESS="$SCRIPT_DIR/ralph_headless.sh"
LOOP_RUNNER="$PROJECT_ROOT/scripts/ralph_loop_runner.sh"
REVIEW="$SCRIPT_DIR/ralph_review.sh"

# ── Defaults ──────────────────────────────────────────────────
ITERATIONS=5
LAST=5
SCOPE=""
OUTPUT_DIR=""
JSON=false
AUTO_REVIEW=false
FAIL_ON_FINDINGS=false
SUBCOMMAND="${1:-help}"
shift 2>/dev/null || true

# ── Parse flags ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations)   ITERATIONS="$2"; shift 2 ;;
        --last)         LAST="$2"; shift 2 ;;
        --scope)        SCOPE="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --json)         JSON=true; shift ;;
        --review)       AUTO_REVIEW=true; shift ;;
        --fail-on-findings) FAIL_ON_FINDINGS=true; shift ;;
        -h|--help)      SUBCOMMAND="help"; shift ;;
        *)              echo "Unknown flag: $1"; exit 2 ;;
    esac
done

# ── Artifact directory ────────────────────────────────────────
setup_artifacts() {
    if [[ -n "$OUTPUT_DIR" ]]; then
        ARTIFACT_DIR="$OUTPUT_DIR"
    else
        ARTIFACT_DIR="artifacts/ralph/$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$ARTIFACT_DIR"
    echo "$ARTIFACT_DIR"
}

# ── Doctor: preflight checks ─────────────────────────────────
do_doctor() {
    echo "════════════════════════════════════════════"
    echo "  RALPH DOCTOR — Preflight Health Check"
    echo "════════════════════════════════════════════"
    local failures=0

    # Git
    if command -v git &>/dev/null && git rev-parse --is-inside-work-tree &>/dev/null; then
        echo "  [OK] Git repository"
    else
        echo "  [FAIL] Not a git repository"
        ((failures++))
    fi

    # Claude CLI
    if [[ -n "$CLAUDE" ]]; then
        echo "  [OK] Claude CLI: $CLAUDE"
    else
        echo "  [FAIL] Claude CLI not found"
        ((failures++))
    fi

    # Python
    if command -v python &>/dev/null; then
        echo "  [OK] Python: $(python --version 2>&1)"
    else
        echo "  [FAIL] Python not found"
        ((failures++))
    fi

    # Agent file
    if [[ -f ".claude/agents/ralph-loop.md" ]]; then
        echo "  [OK] Agent: .claude/agents/ralph-loop.md"
    else
        echo "  [FAIL] Missing .claude/agents/ralph-loop.md"
        ((failures++))
    fi

    # Skill file
    if [[ -f ".claude/skills/ralph/SKILL.md" ]]; then
        echo "  [OK] Skill: .claude/skills/ralph/SKILL.md"
    else
        echo "  [FAIL] Missing .claude/skills/ralph/SKILL.md"
        ((failures++))
    fi

    # Audit state files
    for f in "docs/ralph-loop/ralph-loop-audit.md" \
             "docs/ralph-loop/ralph-loop-history.md" \
             "docs/ralph-loop/deferred-findings.md"; do
        if [[ -f "$f" ]]; then
            echo "  [OK] State: $f"
        else
            echo "  [WARN] Missing: $f (will be created on first run)"
        fi
    done

    # Ledger and centrality (V2 features)
    for f in "docs/ralph-loop/ralph-ledger.json" \
             "docs/ralph-loop/import_centrality.json"; do
        if [[ -f "$f" ]]; then
            echo "  [OK] V2: $f"
        else
            echo "  [WARN] Missing V2 file: $f (run ralph_build_ledger.py / import_graph.py)"
        fi
    done

    # Delegate scripts
    for s in "$HEADLESS" "$LOOP_RUNNER" "$REVIEW"; do
        if [[ -f "$s" ]]; then
            echo "  [OK] Script: $s"
        else
            echo "  [FAIL] Missing: $s"
            ((failures++))
        fi
    done

    # Behavioral audit
    if [[ -f "scripts/tools/audit_behavioral.py" ]]; then
        echo "  [OK] Behavioral: scripts/tools/audit_behavioral.py"
    else
        echo "  [FAIL] Missing: scripts/tools/audit_behavioral.py"
        ((failures++))
    fi

    # Output dir writable
    local test_dir="artifacts/ralph/.doctor_test"
    if mkdir -p "$test_dir" 2>/dev/null && rmdir "$test_dir" 2>/dev/null; then
        echo "  [OK] Artifact dir writable"
    else
        echo "  [FAIL] Cannot write to artifacts/ralph/"
        ((failures++))
    fi

    echo "════════════════════════════════════════════"
    if [[ $failures -eq 0 ]]; then
        echo "  All checks passed."
        return 0
    else
        echo "  $failures check(s) FAILED."
        return 3
    fi
}

# ── Subcommand dispatch ───────────────────────────────────────
case "$SUBCOMMAND" in
    once)
        echo "Ralph: single iteration"
        if [[ -n "$CLAUDE" ]]; then
            SCOPE_ARG=""
            [[ -n "$SCOPE" ]] && SCOPE_ARG="Scope: $SCOPE"
            $CLAUDE -p "Run /ralph $SCOPE_ARG" --allowedTools "Read,Edit,Write,Bash,Grep,Glob" --max-turns 25
            exit_code=$?
        else
            echo "ERROR: Claude CLI required for 'once' mode" >&2
            exit 2
        fi

        if $JSON; then
            ARTIFACT_DIR=$(setup_artifacts)
            cp docs/ralph-loop/ralph-loop-audit.md "$ARTIFACT_DIR/audit.md" 2>/dev/null || true
            python scripts/tools/ralph_build_ledger.py 2>/dev/null || true
            cp docs/ralph-loop/ralph-ledger.json "$ARTIFACT_DIR/ledger.json" 2>/dev/null || true
            echo "Artifacts: $ARTIFACT_DIR"
        fi

        exit ${exit_code:-0}
        ;;

    batch)
        echo "Ralph: batch mode ($ITERATIONS iterations)"
        SCOPE_ARG=""
        [[ -n "$SCOPE" ]] && SCOPE_ARG="$SCOPE"

        bash "$HEADLESS" "$ITERATIONS" "$SCOPE_ARG"
        exit_code=$?

        if $AUTO_REVIEW; then
            echo ""
            echo "Ralph: post-batch review"
            bash "$REVIEW" "$ITERATIONS"
        fi

        if $JSON; then
            ARTIFACT_DIR=$(setup_artifacts)
            cp docs/ralph-loop/ralph-loop-audit.md "$ARTIFACT_DIR/audit.md" 2>/dev/null || true
            python scripts/tools/ralph_build_ledger.py 2>/dev/null || true
            cp docs/ralph-loop/ralph-ledger.json "$ARTIFACT_DIR/ledger.json" 2>/dev/null || true
            echo "Artifacts: $ARTIFACT_DIR"
        fi

        exit ${exit_code:-0}
        ;;

    loop)
        echo "Ralph: continuous loop (stop: touch ralph_loop.stop)"
        bash "$LOOP_RUNNER"
        ;;

    review)
        echo "Ralph: reviewing last $LAST commits"
        bash "$REVIEW" "$LAST"
        ;;

    audit)
        echo "Ralph: behavioral audit only (no fixes)"
        echo ""
        python pipeline/check_drift.py
        drift_exit=$?
        python scripts/tools/audit_behavioral.py
        audit_exit=$?
        ruff check pipeline/ trading_app/ scripts/ --quiet
        ruff_exit=$?

        echo ""
        echo "════════════════════════════════════════════"
        echo "  Drift:      $([ $drift_exit -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "  Behavioral: $([ $audit_exit -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "  Ruff:       $([ $ruff_exit -eq 0 ] && echo 'PASS' || echo 'FAIL')"
        echo "════════════════════════════════════════════"

        if $FAIL_ON_FINDINGS && [[ $drift_exit -ne 0 || $audit_exit -ne 0 || $ruff_exit -ne 0 ]]; then
            exit 1
        fi
        exit 0
        ;;

    doctor)
        do_doctor
        exit $?
        ;;

    help|-h|--help|"")
        head -32 "${BASH_SOURCE[0]}" | tail -30
        exit 0
        ;;

    *)
        echo "Unknown subcommand: $SUBCOMMAND"
        echo "Run: bash scripts/tools/ralph.sh help"
        exit 2
        ;;
esac
