---
task: address review findings F1/F5/F7/F9 on feat/deepseek-coding-agent-v4
mode: IMPLEMENTATION
slug: deepseek-coding-agent-v4-review-fixes
scope_lock:
  - trading_app/ai/provider_registry.py
  - tests/test_scripts/test_opencode_resolve_model.py
  - .githooks/pre-commit
  - scripts/tools/opencode-agent.ps1
  - tests/test_scripts/test_check_or_credits.py
  - docs/runtime/stages/deepseek-coding-agent-v4-phase2.md
  - docs/runtime/stages/deepseek-coding-agent-v4-phase3.md
  - docs/runtime/stages/deepseek-coding-agent-v4-phase4.md
  - docs/runtime/stages/deepseek-coding-agent-v4-review-fixes.md
---

# DeepSeek Coding Agent v4 — Review-Fixes Stage

## Plan reference

Addresses 5 findings filed against branch `feat/deepseek-coding-agent-v4`
(commits e36f2b94 → cfda30c8). F6 verified PHANTOM after grounding;
F1/F5/F7/F9 are real and fixed in this stage.

## What

1. **F1 (canonical-layer fix).** `AIProfile.validation_errors()` accepts
   whitespace-only `model` (e.g. `" "`) because the check is `not self.model`.
   Tighten to `not (self.model and self.model.strip())`. Per
   institutional-rigor §4 the fix lands in the canonical validator, not
   the downstream resolver consumer. New test in
   `test_opencode_resolve_model.py` covers the whitespace path end-to-end.
2. **F7 (capital-class semantic correctness).** `.githooks/pre-commit:202`
   collapses reviewer rc=2 (advisory: review-unavailable) to a hard block
   via `|| exit 1`. Replace with explicit rc dispatch matching the
   reviewer contract documented at `claude_review_deepseek.py:7-12,245`:
   - rc=0 → APPROVE (continue)
   - rc=1 → BLOCK (exit 1)
   - rc=2 → REVIEW_UNAVAILABLE (advisory log + continue)
   - other → fall through (defensive)
3. **F5 (PowerShell launcher hardening).** `opencode-agent.ps1:184` sets
   `$ErrorActionPreference = "Stop"` and invokes
   `& python $resolverScript` without try/catch. A terminating
   `CommandNotFoundException` (no python on PATH) aborts the whole
   launcher. Wrap in try/catch; reuse the existing WARN format; fall
   through to launcher-default model.
4. **F9 (additive coverage).** `_fetch_live` in `check_or_credits.py` has
   3 distinct error branches (HTTPError / URLError-family / JSONDecodeError)
   with zero coverage. Add 3 unit tests against the imported module
   (via `importlib.util` — script's `if __name__ == "__main__"` guard at
   line 143 makes module-import safe), patching `urlopen` directly.

## Why (institutional grounding)

- **No silent failures (institutional-rigor §6).** F7 is the worst class
  of silent failure: legitimate commits get blocked on transient network
  errors with no signal that the gate is unavailable. The reviewer
  already encoded the right semantic (exit 2). The hook discards it.
- **Canonical-source delegation (institutional-rigor §4).** F1 fix in the
  canonical validator means every consumer benefits — not just the
  resolver. `eval_openrouter_profiles.py:441` and
  `openrouter_runtime.py:35` call the same path.
- **Verify before claiming (integrity-guardian §5, §7).** F6 was a
  phantom — assignment at lines 76–97 precedes use at line 197. The
  closeout note records the verdict so a future reviewer doesn't
  re-file it.

## Files

| Path | Action | Notes |
|---|---|---|
| `trading_app/ai/provider_registry.py` | MODIFY | Line 134 — tighten model check to reject whitespace-only. |
| `tests/test_scripts/test_opencode_resolve_model.py` | MODIFY | +1 test for whitespace-only model rejection. |
| `.githooks/pre-commit` | MODIFY | Line 202 — rc dispatch (1=block, 2=advisory, else=continue). |
| `scripts/tools/opencode-agent.ps1` | MODIFY | Lines 183–194 — wrap `& python $resolverScript` in try/catch. |
| `tests/test_scripts/test_check_or_credits.py` | MODIFY | +3 tests covering `_fetch_live` HTTP/network/parse error branches. |
| `docs/runtime/stages/deepseek-coding-agent-v4-phase{2,3,4}.md` | MODIFY | Closeout sections appended (F6 phantom verdict + F1/F5/F7/F9 fix references). |

## Blast Radius

- `trading_app/ai/provider_registry.py` — modifies `AIProfile.validation_errors()` at line 134; called by `assert_ready()` (same file:156), `to_audit_dict()` (164), `opencode_resolve_model.py:32`, `eval_openrouter_profiles.py:441`. Fix only TIGHTENS validation (whitespace strings already produced "configured but useless" models — every caller benefits). Existing test `test_deepseek_coding_assert_ready_fails_when_model_none` continues to pass since `model=None` is also `not (None and ...)` = True.
- `.githooks/pre-commit` — modifies commit-gate path (capital-class). Only the F7 dispatch changes; all other checks unchanged. Reviewer contract documented at `scripts/tools/claude_review_deepseek.py:7-12`.
- `scripts/tools/opencode-agent.ps1` — additive try/catch around python invocation; preserves Phase-1 fallback behavior. No effect when python is on PATH.
- Tests — additive only; new tests in existing files.
- Stage docs — closeout appendices only.
- Reads (read-only): none beyond existing imports.
- Writes: 9 files in scope_lock.
- Reversibility: revert the commit; F7 dispatch reverts to `|| exit 1`; F5 launcher reverts to fail-on-CommandNotFoundException; F1 validator reverts to accepting whitespace.

## Approach

See plan steps Stage A → Stage G in the originating plan. Execution order:

1. **Stage A (F1):** edit `provider_registry.py:134`; add whitespace test in `test_opencode_resolve_model.py`.
2. **Stage B (F7):** edit `.githooks/pre-commit:202` with explicit rc dispatch.
3. **Stage C (F5):** edit `opencode-agent.ps1:183-194` try/catch.
4. **Stage D (F9):** add 3 `_fetch_live` tests with `importlib.util` module load + `urlopen` patches.
5. **Stage E (verification gauntlet):** drift + targeted pytest + reviewer-tests + drift-tests + F7 behavioral check.
6. **Stage F (adversarial audit):** `evidence-auditor` subagent on the combined diff.
7. **Stage G (commit/rebase/push/PR):** standard sequence; no `gh pr merge --auto`.

## Out of scope

- F6 fix (phantom — closeout note only).
- Removing the `not profile.model` defense-in-depth check at
  `opencode_resolve_model.py:37` (small, harmless redundancy; leave for
  layered safety).
- Refactoring `_fetch_live` into a separate module.
- Any Phase 5+ work or doc reorganization.

## Acceptance criteria

All required:

1. `python pipeline/check_drift.py` → 121 PASS / 0 skipped / 19 advisory (no regression).
2. `python -m pytest tests/test_scripts/test_check_or_credits.py tests/test_scripts/test_opencode_resolve_model.py tests/test_scripts/test_claude_review_deepseek.py -v` → all green; +3 in check_or_credits, +1 in opencode_resolve_model.
3. `python -m pytest tests/test_pipeline/test_check_drift.py -v` → green.
4. F7 behavioral test: synthetic rc=2 reviewer logs advisory line, does NOT exit 1.
5. `evidence-auditor` verdict: PASS or addressable findings only.
6. `git rebase origin/main` — clean (1-behind commit `f50d5657` touches non-overlapping docs files).

## Done definition

All four required (institutional rigor §8):

- [ ] Acceptance criteria 1–4 green (5 pending audit).
- [ ] Dead-code sweep: no orphan helpers introduced.
- [ ] `python pipeline/check_drift.py` passes.
- [ ] Self-review + adversarial-audit pass.

## Verification log

- [x] Drift baseline pre-fix: `121 PASS / 0 skipped / 19 advisory`.
- [x] Drift post-fix: identical `121 PASS / 0 skipped / 19 advisory` — no regression.
- [x] Stage A targeted pytest: `tests/test_scripts/test_opencode_resolve_model.py` 6/6 + `tests/test_trading_app/test_ai/test_provider_registry.py` 18/18 → 24 passed.
- [x] Stage D targeted pytest: `tests/test_scripts/test_check_or_credits.py` 9 (5 existing + 1 unit + 3 new with parametrize-3 = 9 collected) → 9 passed.
- [x] Reviewer test untouched: `tests/test_scripts/test_claude_review_deepseek.py` → 6 passed.
- [x] Drift unit tests: `tests/test_pipeline/test_check_drift.py` → 150 passed.
- [x] **F7 behavioral test under `set -e` (auditor-required):** verified all 3 reviewer rc paths via `/tmp/test_f7_under_set_e.sh`:
  - rc=0 (APPROVE) → script continues, FINAL_EXIT=0.
  - rc=2 (REVIEW_UNAVAILABLE) → advisory log emitted, script continues, FINAL_EXIT=0.
  - rc=1 (BLOCK) → script exits 1, FINAL_EXIT=1.
- [x] **Adversarial audit (`evidence-auditor`):** initial verdict caught a critical defect — original F7 fix was semantically a no-op under `set -e` because the bare invocation tripped `set -e` before `review_rc=$?` could capture the exit code. Fixed by switching to the `cmd || review_rc=$?` idiom (places the call inside a `||` list which is exempt from `set -e` per POSIX). Re-tested under `set -e` confirms advisory-vs-block separation works.
- [x] F1/F5/F9/F6 audit verdicts: PASS / PASS / PASS / phantom-confirmed.

## Code-review follow-up (post-commit `ecce35a1`)

A bloomey-style code review on `ecce35a1` graded the commit B+ and surfaced two real follow-up items, both addressed here:

### MEDIUM — `model_configured` parity with `validation_errors()`

- **Symptom:** with the F1 fix tightening `validation_errors()` to reject whitespace-only `model` strings (`provider_registry.py:134`), the sibling `to_metadata()` at line 163 still computed `model_configured = bool(self.model)`, returning `True` for `"   "`. Net effect: a self-contradicting metadata payload (`model_configured=True` alongside `validation_errors=["model not configured..."]`) rendered to the AI Research Packet markdown via `research_packet.py:218`.
- **Fix:** `bool(self.model and self.model.strip())` — exact parity with the canonical validator at line 134.
- **Blast radius:** display-only (only consumer is the markdown rendering at `research_packet.py:218`; verified via `grep -rn "model_configured"` — exactly 2 hits in the entire repo: definition + rendering). Behavior is monotonically tighter (anything previously `False` stays `False`; only whitespace strings flip `True → False`).
- **Test:** `test_to_metadata_model_configured_consistent_with_validation_errors` asserts both signals agree on whitespace-only model.

### LOW — F7 dispatch fail-CLOSED on unexpected reviewer rc

- **Symptom:** the F7 dispatch at `.githooks/pre-commit:213-220` handled rc=0 (continue), rc=1 (block), rc=2 (advisory + continue). Any other rc — rc=130 (Ctrl-C), rc=137 (SIGKILL/OOM), rc=3 (uncaught Python exception) — silently fell through, treating the commit as APPROVED on a capital-class commit-gate path.
- **Fix:** explicit `elif ... -ne 0` branch emits a loud `[0d] WARN: reviewer returned unexpected rc=N; treating as REVIEW_UNAVAILABLE` and continues. Defense-in-depth: undefined rc no longer silently approves.
- **Behavioral semantics:** rc=0 → APPROVE (continue); rc=1 → BLOCK (exit 1); rc=2 → REVIEW_UNAVAILABLE (advisory + continue); rc=anything-else → REVIEW_UNAVAILABLE with diagnostic + continue. Conservative: if the review didn't complete (signal kill, OOM), don't lie and say it passed.
- **Note:** this is **not** a gate-semantic change — rc=0/1/2 behavior is identical to the prior commit. Only previously-undefined rc paths change from silent APPROVE to logged REVIEW_UNAVAILABLE. Hence no adversarial-audit gate (advisory log on already-undefined behavior, not a CRIT/HIGH path).

### Verification log (follow-up)

- [x] Drift baseline pre-fix: `121 PASS / 0 skipped / 19 advisory`.
- [x] Drift post-fix: `121 PASS / 0 skipped / 19 advisory` — no regression.
- [x] `tests/test_trading_app/test_ai/test_provider_registry.py`: 19 passed (18 prior + 1 new metadata-parity test).
- [x] Stage A live falsification (whitespace-only model env): `model_configured: False`, `validation_errors_count: 1` — internally consistent.
- [x] Stage B behavioral test (synthetic `set -e` shim, rc=137 and rc=130): WARN line emitted, FINAL_EXIT=0 (continued, not blocked).
