#!/usr/bin/env bash
# pr_open.sh — open a PR with full preflight gating.
#
# Usage:
#   scripts/tools/pr_open.sh                              # dry-run vs origin/main
#   scripts/tools/pr_open.sh --push                       # actually push + open PR
#   scripts/tools/pr_open.sh --base <ref>                 # custom base
#   scripts/tools/pr_open.sh --body-file <path>           # explicit body file
#   scripts/tools/pr_open.sh --title "<title>"            # explicit title (else inferred from HEAD subject)
#   scripts/tools/pr_open.sh --web                        # open the PR in a browser instead of inline
#   scripts/tools/pr_open.sh --draft                      # mark PR as draft
#
# Body resolution order:
#   1. --body-file <path>  (explicit)
#   2. docs/pr_bodies/<sanitized-branch-slug>.md  (auto-discovered)
#   3. gh pr create --fill  (commit messages)
#
# Constraints:
#   - DRY-RUN by default — pass --push to actually push.
#   - NEVER uses gh pr merge --auto (per memory/feedback_gh_pr_merge_auto_silent_register.md).
#   - Aborts on stacked-base (where base not on origin)
#     per memory/feedback_codex_stack_collapse.md.
#   - Aborts on dirty working tree.
#   - Aborts on zero commits ahead.
#   - Delegates preflight to scripts/tools/pr_preflight.sh; if preflight aborts, this aborts.
#
# Exit codes:
#   0 — success (dry-run shown OR PR opened)
#   1 — preflight aborted (see preflight exit code)
#   2 — gh CLI not installed
#   3 — body file not found
#   4 — push or gh pr create failed
#  64 — bad CLI args

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PREFLIGHT="$REPO_ROOT/scripts/tools/pr_preflight.sh"

BASE="origin/main"
PUSH=0
BODY_FILE=""
TITLE=""
WEB=0
DRAFT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)        BASE="$2"; shift 2 ;;
        --push)        PUSH=1; shift ;;
        --body-file)   BODY_FILE="$2"; shift 2 ;;
        --title)       TITLE="$2"; shift 2 ;;
        --web)         WEB=1; shift ;;
        --draft)       DRAFT=1; shift ;;
        --help|-h)
            sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown arg: $1" >&2
            exit 64
            ;;
    esac
done

if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh CLI not installed; see https://cli.github.com/" >&2
    exit 2
fi

# 1. Run preflight (its exit code propagates).
"$PREFLIGHT" --base "$BASE"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BRANCH_SLUG="${CURRENT_BRANCH//\//-}"

# 2. Resolve body file
if [[ -n "$BODY_FILE" ]]; then
    if [[ ! -f "$BODY_FILE" ]]; then
        echo "ERROR: --body-file '$BODY_FILE' not found" >&2
        exit 3
    fi
    BODY_SOURCE="$BODY_FILE (explicit)"
elif [[ -f "$REPO_ROOT/docs/pr_bodies/${BRANCH_SLUG}.md" ]]; then
    BODY_FILE="$REPO_ROOT/docs/pr_bodies/${BRANCH_SLUG}.md"
    BODY_SOURCE="$BODY_FILE (auto-discovered from branch slug)"
else
    BODY_FILE=""
    BODY_SOURCE="gh pr create --fill (commit messages)"
fi

# 3. Resolve title
if [[ -z "$TITLE" ]]; then
    TITLE="$(git log -1 --pretty=%s HEAD)"
fi

# 4. Show plan
echo ""
echo "===== PR-OPEN PLAN ====="
echo "Branch       : $CURRENT_BRANCH"
echo "Base         : $BASE"
echo "Title        : $TITLE"
echo "Body source  : $BODY_SOURCE"
echo "Push         : $([[ "$PUSH" -eq 1 ]] && echo 'YES' || echo 'NO (dry-run)')"
echo "Web mode     : $([[ "$WEB" -eq 1 ]] && echo 'YES' || echo 'NO')"
echo "Draft        : $([[ "$DRAFT" -eq 1 ]] && echo 'YES' || echo 'NO')"
echo ""

if [[ "$PUSH" -eq 0 ]]; then
    echo "DRY-RUN — re-run with --push to push and open the PR."
    exit 0
fi

# 5. Push (per memory/feedback_no_push_to_other_terminal_branch.md, --push is the explicit OK)
echo "===== PUSH ====="
if ! git push -u origin "$CURRENT_BRANCH"; then
    echo "ERROR: git push failed" >&2
    exit 4
fi
echo ""

# 6. Strip 'origin/' prefix for gh's --base (gh expects branch name, not remote ref)
GH_BASE="${BASE#origin/}"

# 7. gh pr create — never --auto
GH_ARGS=( --base "$GH_BASE" --head "$CURRENT_BRANCH" --title "$TITLE" )

if [[ -n "$BODY_FILE" ]]; then
    GH_ARGS+=( --body-file "$BODY_FILE" )
else
    GH_ARGS+=( --fill )
fi

if [[ "$DRAFT" -eq 1 ]]; then
    GH_ARGS+=( --draft )
fi

if [[ "$WEB" -eq 1 ]]; then
    GH_ARGS+=( --web )
fi

echo "===== OPEN PR ====="
echo "gh pr create ${GH_ARGS[*]}"
echo ""
if ! PR_URL="$(gh pr create "${GH_ARGS[@]}")"; then
    echo "ERROR: gh pr create failed" >&2
    exit 4
fi

echo ""
echo "===== OPENED ====="
echo "$PR_URL"
echo ""
echo "Note: auto-merge is NOT enabled. Run \`gh pr merge --merge <number>\` manually after CI passes."
exit 0
