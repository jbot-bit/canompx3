---
task: IMPLEMENTATION — promote `judgment-review-nudge.py` (PostToolUse, NEVER blocks) to a soft-block guard at PreToolUse on `git commit`. Today: `[judgment]`-tagged commits touching capital-class paths (`trading_app/live/`, `risk_manager.py`, `execution_engine.py`, `session_orchestrator.py`, `pipeline/`) emit a one-line nudge AFTER the commit lands — a polite reminder with zero teeth. After this stage: PreToolUse fires BEFORE `git commit`, parses the staged-message + staged diff (via `git diff --cached --name-only` and `git -c sequence.editor=true commit --dry-run --no-status` or staged message file), and BLOCKS if (a) commit subject/body contains `[judgment]` AND (b) staged paths include capital-class prefixes AND (c) no review-mention regex match in the body AND (d) no fresh `.judgment-review-ts` marker (existing 60-minute suppress window honored). Override via trailing `# --audit-acknowledged` flag-marker on the bash command (mirrors `shared-state-commit-guard.py` semantics). Fail-open contract preserved: every read error / missing git / malformed event / subprocess failure exits 0. Adds drift-check parity ensuring the hook matches the existing capital-class prefix list from `judgment-review-nudge.py` (single source of truth, avoid the canonical-inline-copy class bug). Targeted tests: prompt block cases, override flag, body-mention suppression, marker-file suppression, non-capital-class allow, no-[judgment] allow, fail-open paths.
mode: IMPLEMENTATION
scope_lock:
  - .claude/hooks/judgment-review-soft-block.py
  - .claude/settings.json
  - pipeline/check_drift.py
  - .claude/hooks/tests/test_judgment_review_soft_block.py

## Blast Radius

- **`.claude/hooks/judgment-review-soft-block.py`** — NEW (~140 lines). PreToolUse(Bash) hook. Reads stdin JSON event, extracts the bash command string, matches against `\bgit\s+commit\b` (skip otherwise — exit 0). Parses commit message from the command (handles `-m "msg"`, `-m @'...'@`, `-m @file`, `--message=...`, `-F file`). Reads staged paths via `git diff --cached --name-only`. Applies the four gate predicates (see task). Imports `_CAPITAL_PATH_PREFIXES` + `_REVIEW_MENTION_PATTERNS` + `_MARKER_PATH` + `_SUPPRESS_SECONDS` directly from sibling `judgment_review_nudge.py` via a module-level `importlib.util.spec_from_file_location` shim (NOT a from-import, since hook scripts run as top-level processes not as a package). This is the canonical-source delegation per `institutional-rigor.md` § 4 — avoid the inline-copy parity class bug. Override syntax: trailing `# --audit-acknowledged` in the bash command string is stripped before parsing and acts as bypass (mirrors `shared-state-commit-guard.py` "# --shared-state-ack").
- **`.claude/settings.json`** — register the new hook under `hooks.PreToolUse[*]` with matcher `Bash`. Existing `judgment-review-nudge.py` PostToolUse registration UNCHANGED (the nudge still fires on the next bash call after the commit for the false-negative case where the commit doesn't match our trigger but the user still wants a reminder).
- **`pipeline/check_drift.py`** — add ONE new check: `check_judgment_review_capital_paths_parity`. Re-loads `judgment_review_nudge._CAPITAL_PATH_PREFIXES` (canonical) and `judgment_review_soft_block._CAPITAL_PATH_PREFIXES` (proxy), asserts equality. Mutation-probe per `feedback_injection_test_catches_float_repr_class_bug.md` — write the proxy as a function attribute fetched at check time, not literal-inlined. (If the soft-block imports via the importlib shim then the proxy IS the canonical and check is trivially true. The check exists to make the soft-block REFACTORING-SAFE: if a future edit accidentally inlines the prefixes, drift catches it.)
- **`tests/test_hooks/test_judgment_review_soft_block.py`** — NEW. 9 tests:
  1. `[judgment]` + capital-class staged path + clean body → BLOCK (exit 2, stderr names review skill).
  2. `[judgment]` + capital-class staged path + body mentions "code-review" → ALLOW (exit 0).
  3. `[judgment]` + capital-class staged path + override flag `# --audit-acknowledged` → ALLOW.
  4. `[judgment]` + capital-class staged path + fresh marker file (within 60min) → ALLOW.
  5. `[judgment]` + ONLY non-capital staged path (e.g., `docs/`) → ALLOW.
  6. No `[judgment]` tag → ALLOW even on capital-class paths (nudge layer covers this).
  7. Malformed bash command (no `git commit`) → ALLOW.
  8. Subprocess failure (mock `git diff` to fail) → fail-open ALLOW.
  9. Override flag is stripped from the bash command before any other parsing.
- **Reads:** `git diff --cached --name-only`, optional commit-message file from `-F`, sibling hook module file. **Writes:** none (stderr only). **No DB touch. No production code path change. No git-state mutation.**
- **Concurrent safety:** PreToolUse hooks run synchronously before the tool executes; no race with the existing PostToolUse nudge. The nudge still runs after the commit for false-negative coverage; we are adding a forcing function, not replacing the nudge.
- **Test blast radius:** new test file only; existing test suite untouched.

## Non-goals (deferred)

- Auto-fire `/code-review` subagent on stage closeout — that is stage (B), separate file.
- Pyright diff-ratchet — stage (D), parked.
- Windows-runner mutex hang fix — stage (C), tracked in `feedback_ci_windows_runner_hang_test_work_capsule.md`.
- Replacing the PostToolUse nudge with the soft-block — keep both layers; the nudge catches cases the soft-block parser misses (e.g., commit via IDE).

## Done criteria

1. All 9 tests in `test_judgment_review_soft_block.py` pass (show output).
2. `python pipeline/check_drift.py` passes with the new parity check counted (159 PASSED, 0 violations expected).
3. Manual smoke: a) `git commit -m "[judgment] HIGH: touch trading_app/live/foo.py"` with `trading_app/live/foo.py` staged → BLOCKED with stderr naming `/code-review`. b) Same with body `… includes code-review pass` → ALLOWED. c) Same with `# --audit-acknowledged` trailing → ALLOWED.
4. `grep -r "judgment-review-soft-block" .claude/settings.json` confirms registration.
5. Self-review against `institutional-rigor.md` §§ 4 (no inline copy of capital-class prefixes — delegate to canonical), 5 (no dead code), 6 (no silent failures — every except logs to stderr), 8 (verify before claim).
6. **No adversarial-audit gate dispatch required** — this stage touches `.claude/hooks/` + `.claude/settings.json` + adds a parity drift check + adds a test file. Zero `trading_app/live/`, `risk_manager.py`, `execution_engine.py`, `session_orchestrator.py`, or `pipeline/` truth-layer modifications. The one `pipeline/check_drift.py` edit is additive (new check function), not a behavioral change to existing logic. Per gate trigger criteria, severity is MEDIUM at most and capital-class blast is zero — gate is not required. (If this judgement is wrong, the BLOCKING soft-block itself would fire on commit — a desirable property.)
7. Companion entry in `memory/feedback_*.md` if any new failure-class surfaces during implementation.

## Execution ordering

1. Write `.claude/hooks/judgment-review-soft-block.py` with the importlib shim sourcing canonical constants from `judgment-review-nudge.py`.
2. Write `tests/test_hooks/test_judgment_review_soft_block.py`; run isolated; iterate until all 9 pass.
3. Register hook in `.claude/settings.json` under PreToolUse → Bash.
4. Add `check_judgment_review_capital_paths_parity` to `pipeline/check_drift.py`; run drift; expect 159 PASSED.
5. Manual smoke (done-criterion 3).
6. Commit. The first commit AFTER hook registration will test the hook in production — if I forgot to mention review in the body, the soft-block fires and I'll add `# --audit-acknowledged` after dispatching `evidence-auditor` … but this stage is `.claude/hooks/` not capital-class, so the hook won't fire on this commit. Self-test deferred to the next capital-class change.

## Risk register

- **HIGHEST:** false-positive block on a legitimate commit (e.g., commit message contains literal `[judgment]` substring inside a quoted shell argument that isn't the commit message itself). Mitigation: parser limits matching to the `-m`/`--message`/`-F` payload region; override flag is the operator escape hatch.
- **MEDIUM:** parsing `-m @'...'@` here-strings on PowerShell. Mitigation: detect here-string opener and read until matching closer; fall through to fail-open on unmatched here-string.
- **LOW:** Windows path separator drift between `git diff --cached` output (`/` on Windows) and `_CAPITAL_PATH_PREFIXES` (already uses `/`). Already aligned in `judgment-review-nudge.py`.

## Cross-references

- Predecessor: `.claude/hooks/judgment-review-nudge.py` (nudge layer, retained).
- Override pattern: `.claude/hooks/shared-state-commit-guard.py` (`# --shared-state-ack`).
- Fail-open contract: `.claude/rules/branch-flip-protection.md` § "Fail-safe guarantee".
- Doctrine being mechanised: `.claude/rules/adversarial-audit-gate.md`.
- Memory: `project_review_enforcement_gaps_and_plan_2026_05_23.md` (the plan), `feedback_drift_check_systemexit_escape_n1_2026_05_23.md` (sibling lesson on exception nets).
- n=3 threshold: `feedback_n3_same_class_doctrine_threshold.md` — every `[judgment]` capital-class commit is one instance of the "doctrine-OK / mechanism-MISSING" class; threshold met for forcing-function (this stage).
---
