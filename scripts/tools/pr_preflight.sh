#!/usr/bin/env bash
# pr_preflight.sh — verify a branch is ready for PR. No push, no PR creation.
#
# Usage:
#   scripts/tools/pr_preflight.sh                   # current branch vs origin/main
#   scripts/tools/pr_preflight.sh --base <ref>      # current branch vs <ref>
#   scripts/tools/pr_preflight.sh --quiet           # exit-code only, no body
#
# Exit codes:
#   0 — clean and ready
#   1 — uncommitted changes / dirty tree
#   2 — base branch not on origin (stacked-base abort)
#   3 — no commits ahead of base
#   4 — protected-path touched (trading_app/holdout/lane_allocation/live_config)
#   5 — git fetch failed (offline?) — soft warning only, does not block
#  64 — bad CLI args
#
# Project rules enforced:
#   - .claude/rules/branch-discipline.md — show log/diff scope before any action
#   - memory/feedback_no_push_to_other_terminal_branch.md — never auto-push
#   - memory/feedback_codex_stack_collapse.md — stacked base detection
#   - memory/feedback_gha_merge_ref_staleness.md — refresh ref via head-branch push only

set -euo pipefail

BASE="origin/main"
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            BASE="$2"
            shift 2
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        --help|-h)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 64
            ;;
    esac
done

say() {
    [[ "$QUIET" -eq 1 ]] || printf '%s\n' "$*"
}

die() {
    local code="$1"
    shift
    printf 'BLOCKED [%d]: %s\n' "$code" "$*" >&2
    exit "$code"
}

# Fetch quietly; tolerate offline mode.
if ! git fetch origin --quiet 2>/dev/null; then
    say "WARNING: git fetch origin failed — base ref may be stale"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" == "HEAD" ]]; then
    die 5 "detached HEAD; checkout a branch first"
fi

# 1. Clean tree (ignoring untracked-only)
DIRTY="$(git status --porcelain | grep -vE '^\?\?' || true)"
if [[ -n "$DIRTY" ]]; then
    say "===== DIRTY-TREE CHECK ====="
    say "Working tree has uncommitted changes:"
    git status --short | grep -vE '^\?\?' || true
    say ""
    die 1 "commit or stash before opening PR"
fi

# 2. Resolve base ref
if ! git rev-parse --verify --quiet "$BASE" >/dev/null; then
    die 2 "base ref '$BASE' does not resolve locally"
fi

# 3. Stacked-base detection — if base isn't on origin (and isn't an origin/* ref),
#    publishing the PR would target a branch reviewers cannot see.
if [[ "$BASE" != origin/* ]]; then
    REMOTE_HEAD="$(git ls-remote --heads origin "$BASE" 2>/dev/null || true)"
    if [[ -z "$REMOTE_HEAD" ]]; then
        say "===== STACKED-BASE ABORT ====="
        say "Base branch '$BASE' is not on origin."
        say ""
        say "Two options:"
        say "  STACK    — push '$BASE' first, then re-run preflight:"
        say "             git push -u origin $BASE"
        say "             scripts/tools/pr_preflight.sh --base $BASE"
        say ""
        say "  RETARGET — open PR against origin/main; PR diff will include"
        say "             the un-merged base commits. Note this in the PR body."
        say "             scripts/tools/pr_preflight.sh --base origin/main"
        die 2 "stacked base not on origin; choose STACK or RETARGET"
    fi
fi

# 4. Diff scope (branch-discipline.md HARD RULE)
AHEAD_COMMITS="$(git rev-list --count "$BASE..HEAD")"
BEHIND_COMMITS="$(git rev-list --count "HEAD..$BASE")"

if [[ "$AHEAD_COMMITS" -eq 0 ]]; then
    die 3 "no commits ahead of $BASE; nothing to PR"
fi

say "===== DIFF SCOPE (branch-discipline.md) ====="
say "Current branch : $CURRENT_BRANCH"
say "Base ref       : $BASE"
say "Commits ahead  : $AHEAD_COMMITS"
say "Commits behind : $BEHIND_COMMITS"
say ""
say "----- git log --oneline $BASE..HEAD -----"
git log --oneline "$BASE..HEAD"
say ""
say "----- git diff --stat $BASE..HEAD -----"
git diff --stat "$BASE..HEAD"
say ""

# 5. Protected-path scan
PROTECTED_REGEX='^trading_app/(holdout_policy|prop_profiles|live/|risk_manager|outcome_builder|asset_configs|cost_model|config\.py)|^pipeline/(check_drift|paths|dst|cost_model|asset_configs)\.py$|holdout_policy|live_config|lane_allocation|validated_setups\.py'
PROTECTED_HITS="$(git diff --name-only "$BASE..HEAD" | grep -E "$PROTECTED_REGEX" || true)"

if [[ -n "$PROTECTED_HITS" ]]; then
    say "===== PROTECTED-PATH NOTICE ====="
    say "PR touches protected files (review extra carefully):"
    printf '  %s\n' $PROTECTED_HITS
    say ""
    HARD_BLOCK_REGEX='trading_app/(holdout_policy|live/(execution_engine|risk_manager|order_router))\.py$|live_config\.py$|lane_allocation\.json$'
    HARD_HITS="$(printf '%s\n' $PROTECTED_HITS | grep -E "$HARD_BLOCK_REGEX" || true)"
    if [[ -n "$HARD_HITS" ]]; then
        say "  HARD-BLOCK violations:"
        printf '    %s\n' $HARD_HITS
        die 4 "hard-block protected path touched; abort"
    fi
fi

say "===== READY ====="
say "Branch '$CURRENT_BRANCH' is preflight-clean for PR vs $BASE."
say "  Next: scripts/tools/pr_open.sh --base $BASE          (dry-run)"
say "        scripts/tools/pr_open.sh --base $BASE --push   (open PR)"
exit 0
