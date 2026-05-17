# Silent-Rot T1 — Pre-commit hook fail-stop hygiene

task: Promote the trade-window backfill region of `.githooks/pre-commit` from fail-open ADVISORY to FAIL-STOP, and harden the CRG update region with a loud-WARN to stderr + 3-strike persistent counter that flips to FAIL-STOP after 3 consecutive update failures. Eliminates the two genuine fail-open regions identified by inline verification (2026-05-17 self-audit of original Phase-1 recon).
mode: IMPLEMENTATION

## Scope Lock
- .githooks/pre-commit

## Blast Radius
- `.githooks/pre-commit` line 344-352 (backfill region): switches `set +e`-captured `BACKFILL_RC` from advisory print to fail-stop `exit 1` on non-zero. Justification: today's "drift check may catch residual mismatches" is partial — drift check #45 catches the *symptom* (validated_setups vs trade_windows mismatch) but does not guarantee detection of every backfill *failure mode* (partial writes, transient DuckDB locks). Validator/builder code already trusts `validated_setups`; silent backfill failure is a data-integrity risk.
- `.githooks/pre-commit` line 386-411 (CRG region): redirects ADVISORY message to stderr (`>&2`) so it surfaces distinctly in CI logs; reads/writes a counter file at `$GIT_DIR_PATH/.crg-update-failures`. Counter increments on rc!=0, resets to 0 on success, triggers `exit 1` at count>=3. Asymmetry rationale: CRG is a *navigation* surface (per `feedback_crg_no_graph_storm.md`), not a *truth* layer. Transient flake (per `feedback_mcp_partial_install_state_2026_05_01.md`) does not justify hard-stop; sustained drift does.
- Reads (truth): no canonical DB or config files. Writes: `$GIT_DIR_PATH/.crg-update-failures` (per-worktree counter file; not version-controlled).
- Downstream effects: every `git commit` against this repo. Tighter fail-closure on backfill prevents stale `validated_setups`; CRG counter prevents accumulating-staleness on the graph DB used by `/crg-*` skills.
- No production Python touched. No companion test file exists for the hook itself — verification is by manual end-to-end hook invocation.

## Pre-decisions
- Backfill region (line 344-352): **FAIL-STOP**.
- CRG region (line 386-411): **loud-WARN to stderr + 3-strike persistent counter** (plan's recommended option; lower false-positive rate suits a non-truth navigation surface).
- Counter file: `$GIT_DIR_PATH/.crg-update-failures` — same dir as `.claude.pid`. Simple plaintext integer. Missing file = 0. Worktree-local (each worktree has its own `.git/worktrees/<name>/`).
- `.gitignore` update: not needed — `$GIT_DIR_PATH` is already `.git/` (or `.git/worktrees/<name>/`), which is never tracked by git.

## Why this is not "doc-only"
The pre-commit hook is the canonical gate between local edits and the repo. Demoting its advisory-only regions to fail-stop / counted-fail-stop changes the trust boundary of every commit. Treated as production infra change.

## Acceptance criteria
1. Backfill region (line 344-352) is FAIL-STOP. Manual test: rename `scripts/migrations/backfill_validated_trade_windows.py` to a non-existent path, attempt commit on a clean repo, observe `exit 1` and `BLOCKED` message instead of `ADVISORY`. Restore filename, re-attempt commit, observe `PASSED`.
2. CRG region (line 386-411): on rc==0, counter file resets to 0 and prints `PASSED`. On rc!=0, prints `ADVISORY: CRG update failed (rc=$CRG_RC) — graph may be stale (consecutive failures: <N>/3)` to **stderr** and increments counter. On count>=3, exits 1 with `BLOCKED: 3 consecutive CRG update failures`.
3. `git commit` on a clean working tree of `main` still succeeds (minus the 2 advisory lines on success paths — counter print is 0/3 only on fresh state).
4. `python pipeline/check_drift.py` still passes with no regression (135 checks; advisory unchanged at 20 since T1 doesn't touch the drift check list).
5. No new `set +e` regions added outside the existing 2 locations.
6. Adversarial audit: **EXEMPT per `.claude/rules/adversarial-audit-gate.md` § Exemption path** — severity LOW, infra change (hook script), not a truth-layer fix. Commit body MUST contain `AUDIT-SKIPPED: T1 LOW-severity infra, not a CRIT/HIGH truth-layer fix per adversarial-audit-gate.md § Gate trigger`.
7. Push to `main` direct (per `feedback_direct_push_vs_pr_flow_token_cost.md` — no capital-class path touched). CI must be green for non-Windows-runner-hang jobs; `Tests with coverage` hang allowed per plan exception (subject of T5).
