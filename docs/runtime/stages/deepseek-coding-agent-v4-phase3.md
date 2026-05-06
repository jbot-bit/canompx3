---
task: DeepSeek Coding Agent v4 — Phase 3 (Claude-side review gate)
mode: IMPLEMENTATION
slug: deepseek-coding-agent-v4-phase3
scope_lock:
  - scripts/tools/claude_review_deepseek.py
  - scripts/tools/opencode-agent.ps1
  - .githooks/pre-commit
  - tests/test_scripts/test_claude_review_deepseek.py
  - tests/test_pipeline/test_check_drift.py
  - docs/runtime/stages/deepseek-coding-agent-v4-phase3.md
  - docs/specs/opencode_agent.md
---

# DeepSeek Coding Agent v4 — Phase 3 (Claude-side review gate)

## Plan reference

Stage B of the OpenCode integration plan. Phase 1 registered the
`check_deepseek_review_gate_intact` drift check as a no-op; Phase 2 left
it as a no-op. Phase 3 lands the actual `# 0d.` step in
`.githooks/pre-commit` and the canonical reviewer script
`scripts/tools/claude_review_deepseek.py`. Phase 1's regex matches
`claude_review_deepseek` (verbatim), so the script must keep that name.

## What

1. **Reviewer script.** `scripts/tools/claude_review_deepseek.py` — reads
   `git diff --cached`, calls Claude (seven-sins rubric inlined as a
   hermetic constant), parses JSON verdict, exits 0/1/2 per
   APPROVE/BLOCK/REVIEW_UNAVAILABLE.
2. **Pre-commit step 0d.** `.githooks/pre-commit` gains a step between
   the existing 0c and 1 that runs the reviewer when
   `OPENCODE_AGENT_ACTIVE=1` is exported.
3. **Launcher activation.** `scripts/tools/opencode-agent.ps1` exports
   `$env:OPENCODE_AGENT_ACTIVE = "1"` before spawning OpenCode.
4. **Tests.** Reviewer tests cover skip-when-inactive, mock APPROVE,
   mock BLOCK, doc-only-skip, and tiny-diff-skip. Drift-check test
   confirms Phase 1's `check_deepseek_review_gate_intact` now passes
   with the marker present + invocation present (positive case).

## Why (institutional grounding)

- **Adversarial-audit gate** (`.claude/rules/adversarial-audit-gate.md`):
  the reviewer is the formalized after-fix review for an LLM coding
  agent. Single-agent self-review is insufficient; an independent
  Claude pass is the gate.
- **Delegate, never re-encode** (institutional-rigor §4): reviewer
  imports `trading_app.ai.claude_client.get_client` + `CLAUDE_REASONING_MODEL`,
  never instantiates `anthropic.Anthropic` directly.
- **No silent failures** (institutional-rigor §6): network/parse errors
  return exit 2 (REVIEW_UNAVAILABLE) with explicit stderr; user must
  consciously `--no-verify` to bypass.
- **Phase 1 contract preserved**: drift check
  `check_deepseek_review_gate_intact` flips automatically when the
  marker lands; no code change in `pipeline/check_drift.py` is needed
  this phase (the regex already exists and finds `claude_review_deepseek`
  in the new pre-commit step).

## Files

| Path | Action | Notes |
|---|---|---|
| `scripts/tools/claude_review_deepseek.py` | CREATE | ~150 lines. Hermetic seven-sins rubric, JSON-out, mock flag for tests. |
| `scripts/tools/opencode-agent.ps1` | MODIFY | Export `$env:OPENCODE_AGENT_ACTIVE = "1"` before spawning OpenCode. |
| `.githooks/pre-commit` | MODIFY | Add step 0d after 0c; runs reviewer only when env var set. |
| `tests/test_scripts/test_claude_review_deepseek.py` | CREATE | Mock-mode coverage; no live Claude calls. |
| `tests/test_pipeline/test_check_drift.py` | MODIFY | Add positive test (marker present + invocation present → no violations). |
| `docs/specs/opencode_agent.md` | MODIFY | Update Review Gate section with concrete script path + exit-code contract. |

## Blast Radius

- `scripts/tools/claude_review_deepseek.py` — NEW; called only by pre-commit step 0d when launcher activates the gate. Imports canonical `trading_app.ai.claude_client` (read-only). No DB writes. No live-trading effects. Risk: low.
- `scripts/tools/opencode-agent.ps1` — single-line export added before launch. Affects only sessions launched through the launcher. Risk: low.
- `.githooks/pre-commit` — new step is gated on `OPENCODE_AGENT_ACTIVE=1`; Claude/manual commits are byte-identical to before. Risk: low (gated).
- `tests/test_scripts/test_claude_review_deepseek.py` — additive only.
- `tests/test_pipeline/test_check_drift.py` — additive positive test for the existing Phase 1 check; no behavior change.
- `docs/specs/opencode_agent.md` — documentation update.
- Reads (read-only): canonical Anthropic client API, staged git diff.
- Writes: 7 files in scope_lock.
- Reversibility: revert the commit; pre-commit step 0d falls back to no-op (the env var is the gate).

## Approach

1. **Reviewer is hermetic.** Inline the seven-sins rubric verbatim from `.claude/rules/quant-agent-identity.md`. This avoids a runtime dependency on the rule file. Drift check #137 catches rule-file moves; the constant is reviewed at PR time.
2. **Mock mode is the test surface.** Live Claude calls are NOT exercised in tests (cost + flakiness). Mock mode produces deterministic APPROVE/BLOCK responses; integration is verified manually via Stage B criterion 6.
3. **Diff filters before API call.** Skip on empty diff, < 5 line diff, doc-only diff (no code review surface). The user pays for nothing.
4. **Network-fail = exit 2.** REVIEW_UNAVAILABLE is advisory; `--no-verify` bypass is explicit, never silent. The drift check still fires if the marker is present but the invocation is missing — that's the structural guarantee.

## Out of scope (Phase 3)

- `check_or_credits.py` — Phase 4 (Stage C).
- Adding the rubric drift check (extra-scope; Phase 1's existing reference-paths check #137 covers rule-file moves).
- Live integration test against the Anthropic API — not run in CI; manual gate per criterion 6.

## Acceptance criteria

All required:

1. With `OPENCODE_AGENT_ACTIVE=1` and a staged diff, `python scripts/tools/claude_review_deepseek.py --mock --rubric-pass` → exit 0.
2. With `OPENCODE_AGENT_ACTIVE=1` and a staged diff, `python scripts/tools/claude_review_deepseek.py --mock --rubric-fail` → exit 1, stderr has FINDINGS.
3. Without `OPENCODE_AGENT_ACTIVE`, `python scripts/tools/claude_review_deepseek.py` exits 0 immediately (no Claude call attempted; no diff read).
4. `python pipeline/check_drift.py` passes — `check_deepseek_review_gate_intact` is now a real blocking check; the marker is present in pre-commit; the canonical invocation is present.
5. `python -m pytest tests/test_scripts/test_claude_review_deepseek.py tests/test_pipeline/test_check_drift.py` passes.
6. **Live smoke test (manual; user-supervised):**
   - In an OpenCode session, make a trivial code change. Commit. Reviewer runs, prints APPROVE, commit lands.
   - Make a deliberately-bad change (hardcode a second `openrouter/anything-else` literal). Commit. Reviewer flags it (HIGH/CRITICAL), exit 1, commit blocked.

## Done definition (Phase 3 only)

All four required (institutional rigor §8):

- [ ] Acceptance criteria 1–6 green with execution evidence.
- [ ] Dead-code sweep: `grep -r "claude_review_deepseek" --include="*.py" --include="*.sh" --include="*.ps1"` shows only the new code paths and references.
- [ ] `python pipeline/check_drift.py` passes.
- [ ] Self-review pass.

## Verification log

- [x] Acceptance criterion 1: with `OPENCODE_AGENT_ACTIVE=1` and a staged code diff, `python scripts/tools/claude_review_deepseek.py --mock --rubric-pass` → exit 0 (verified via `test_mock_rubric_pass_returns_zero`).
- [x] Acceptance criterion 2: with `OPENCODE_AGENT_ACTIVE=1` and a staged code diff, `python scripts/tools/claude_review_deepseek.py --mock --rubric-fail` → exit 1, stderr has FINDINGS table (`test_mock_rubric_fail_returns_one_with_findings`).
- [x] Acceptance criterion 3: without `OPENCODE_AGENT_ACTIVE`, the script returns 0 immediately. Verified directly: `python scripts/tools/claude_review_deepseek.py; echo $?` → 0 (no Claude call attempted).
- [x] Acceptance criterion 4: `python pipeline/check_drift.py` → `NO DRIFT DETECTED: 121 checks passed [OK]`. `check_deepseek_review_gate_intact` is now blocking; the marker is present in `.githooks/pre-commit` step 0d AND the canonical invocation `claude_review_deepseek` is wired.
- [x] Acceptance criterion 5: `python -m pytest tests/test_scripts/test_claude_review_deepseek.py tests/test_pipeline/test_check_drift.py` → **156 passed** (149 prior + 6 new reviewer + 1 positive review-gate).

## Live smoke test (Stage B criterion 6)

User-supervised; pending. Per spec doc § Review Gate: `OPENCODE_AGENT_ACTIVE=1`-set commit with a clean code change should APPROVE; same env with a hardcoded `openrouter/<vendor>/<model>` literal lacking the canonical-default-fallback annotation should BLOCK.
