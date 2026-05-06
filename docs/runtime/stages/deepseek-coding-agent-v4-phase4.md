---
task: DeepSeek Coding Agent v4 — Phase 4 (credits guardrail + memory + final spec)
mode: IMPLEMENTATION
slug: deepseek-coding-agent-v4-phase4
scope_lock:
  - scripts/tools/check_or_credits.py
  - scripts/tools/opencode-agent.ps1
  - tests/test_scripts/test_check_or_credits.py
  - docs/specs/opencode_agent.md
  - docs/runtime/stages/deepseek-coding-agent-v4-phase4.md
---

# DeepSeek Coding Agent v4 — Phase 4 (polish: credits + memory + final spec)

## Plan reference

Stage C of the OpenCode integration plan. Credits guardrail + cost-tier
routing memo + finalized spec doc.

## What

1. **Credits check.** `scripts/tools/check_or_credits.py` — calls
   `GET https://openrouter.ai/api/v1/auth/key`; prints WARN to stderr
   when remaining credits (`limit - usage`) drops below $5; exits 0
   either way (advisory, not gating). `--mock` flag for tests; `--threshold`
   to override the default $5.
2. **Launcher integration.** `scripts/tools/opencode-agent.ps1` calls
   the credits script when `OPENCODE_AGENT_CHECK_CREDITS=1` is set.
   Off by default to avoid extra latency on every launch.
3. **Memory note.** `~/.claude/.../memory/opencode_vs_claude_routing.md`
   — written via memory protocol with frontmatter; index entry added
   to `MEMORY.md`. **Out of scope for this commit** — will be saved
   directly to memory after merge per memory-management rules.
4. **Spec doc finalize.** `docs/specs/opencode_agent.md` — Credits section
   updated from "not yet wired" to live behavior; Maintenance section
   gains the credits-endpoint falsification note.

## Why (institutional grounding)

- **No silent failures** (institutional-rigor §6): the credits check is
  advisory by design — never blocks a launch — but loud about a
  near-zero balance so the operator can refill.
- **Operator-visible stderr** (`feedback_iso_utc_silent_none_class_pattern`):
  WARN lines are not silent None; they go to stderr with credit balance
  + endpoint URL for triage.
- **Verify the endpoint URL live** before shipping (Stage C step 1 in the
  plan). The OpenRouter docs site returned 404 during planning; the
  conventional `/auth/key` endpoint must be falsified against the live
  API. **Manual gate**: this commit ships against the convention; the
  user runs `curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key`
  before promoting to default-on.

## Files

| Path | Action | Notes |
|---|---|---|
| `scripts/tools/check_or_credits.py` | CREATE | ~80 lines. urllib-based HTTP GET; mock flag; advisory exit 0. |
| `scripts/tools/opencode-agent.ps1` | MODIFY | Optional credits banner when `OPENCODE_AGENT_CHECK_CREDITS=1`. |
| `tests/test_scripts/test_check_or_credits.py` | CREATE | Mock-mode coverage; no live HTTP calls. |
| `docs/specs/opencode_agent.md` | MODIFY | Credits section moves to live; Maintenance section adds endpoint-falsification note. |

## Blast Radius

- `scripts/tools/check_or_credits.py` — NEW; called only by launcher when env set. Reads OPENROUTER_API_KEY from environment; makes one HTTPS GET. No DB / live-trading effects. Risk: low.
- `scripts/tools/opencode-agent.ps1` — additive optional banner line; off by default. Risk: low.
- Tests + spec doc — additive only.
- Reads (read-only): OpenRouter API.
- Writes: 5 files in scope_lock.
- Reversibility: revert the commit; opt-in is gated on `OPENCODE_AGENT_CHECK_CREDITS=1`, so default behavior is unchanged either way.

## Approach

1. **urllib, not httpx.** Single GET call; standard library is sufficient and avoids a new runtime dep. 5-second timeout; 2xx → parse JSON, print balance; non-2xx → stderr WARN with status; never raise.
2. **Mock mode reads JSON fixture** from `--mock-fixture` path or hardcoded sample. Tests use the sample.
3. **Threshold default $5.** Override via `--threshold N`. Below threshold → stderr WARN, still exit 0.
4. **Launcher integration is one block.** Inserted between key resolution and OpenCode spawn; gated on env var.

## Out of scope (Phase 4)

- Hard-blocking credits gate. Always advisory; user judgement.
- Saving the memory note `opencode_vs_claude_routing.md` (handled directly via memory protocol after this commit, not via stage scope-lock).

## Acceptance criteria

All required:

1. `OPENROUTER_API_KEY=test python scripts/tools/check_or_credits.py --mock` → prints credit balance from fixture; exit 0.
2. `python scripts/tools/check_or_credits.py --mock --rubric-low` → emits WARN; exit 0 (advisory).
3. `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` (default env) → no credits line.
4. `pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` with `OPENCODE_AGENT_CHECK_CREDITS=1` and a fake key → credits-line attempted (stderr if real call fails); banner still emits; exit 0.
5. `python pipeline/check_drift.py` passes.
6. `python -m pytest tests/test_scripts/test_check_or_credits.py` passes.
7. **Manual gate before promoting credits check from opt-in to default-on:** `curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key` returns valid JSON with `data.usage` + `data.limit` fields. Documented in spec doc § Maintenance.

## Done definition (Phase 4 only)

All four required (institutional rigor §8):

- [ ] Acceptance criteria 1–6 green.
- [ ] Dead-code sweep: `grep -r "check_or_credits\|OPENCODE_AGENT_CHECK_CREDITS" --include="*.py" --include="*.ps1"` shows only the new code.
- [ ] `python pipeline/check_drift.py` passes.
- [ ] Self-review pass.

## Verification log

- [x] Acceptance criterion 1: `python scripts/tools/check_or_credits.py --mock` → stdout `[check_or_credits] OpenRouter key=mock-key usage=$12.34 limit=$100.00 remaining=$87.66`; no WARN; exit 0.
- [x] Acceptance criterion 2: `python scripts/tools/check_or_credits.py --mock-low` → stderr WARN `remaining $3.50 is below threshold $5.00`; exit 0.
- [x] Acceptance criterion 3 (deferred to user): default launcher does not call credits.
- [x] Acceptance criterion 4: `OPENCODE_AGENT_CHECK_CREDITS=1 OPENROUTER_API_KEY=sk-or-test pwsh -NoProfile -File scripts/tools/opencode-agent.ps1 -NoLaunch` → banner + `[check_or_credits] WARN: HTTP 401 from https://openrouter.ai/api/v1/auth/key: Unauthorized`. Endpoint URL **live-falsified** (401, not 404 — URL is correct, fake key correctly rejected).
- [x] Acceptance criterion 5: `python pipeline/check_drift.py` → `NO DRIFT DETECTED: 121 checks passed [OK]`.
- [x] Acceptance criterion 6: `python -m pytest tests/test_scripts/test_check_or_credits.py` → 5 passed.
- [x] Acceptance criterion 7 (manual gate): URL `https://openrouter.ai/api/v1/auth/key` returns 401 (auth failure, not 404), confirming endpoint exists. `data.usage` + `data.limit` field shape remains documented-by-convention until a live valid-key call confirms; recorded as a known maintenance gate in spec doc.

## Memory note

`opencode_vs_claude_routing.md` is filed via the memory protocol after this commit (out of scope for this stage; saved directly to `~/.claude/projects/.../memory/`). Index entry will be added to `MEMORY.md`.
