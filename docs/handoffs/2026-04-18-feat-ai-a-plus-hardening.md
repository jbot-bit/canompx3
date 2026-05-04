---
date: 2026-04-18
type: session-handover
prior_handover: docs/handoffs/2026-04-18-session-handover.md
slug: feat-ai-a-plus-hardening
---

# 2026-04-18 Session Handover — Ralph Prompt Refactor + feat(ai) A+ Hardening

## TL;DR

Two workstreams shipped to `origin/main`:

1. **Ralph Loop prompt refactor** — thin core + doctrine-load preamble replaces stale inlined Seven Sins + canonical-sources tables. File-gated doctrine loading for target-area-specific rules (backtesting-methodology / RESEARCH_RULES / mechanism_priors / etc.). Iter 168 validated: findings now cite canonical rule numbers; token spend **dropped** 100K → 76K vs iter 167 (file-gated loads cut exploration waste).
2. **feat(ai) Stages 1-4 code review → A+** — Bloomey-grade institutional review of the 4 claude-api-modernization commits found one **CRITICAL production bug** hidden by `MagicMock` tests, plus one MEDIUM silent-failure and one LOW. All fixed. New test class `TestSDKSurfaceGuards` closes the MagicMock-hides-TypeError class of failure.

No research or trading-data work this session.

---

## Commits landed (in order)

| Hash | Tag | Purpose |
|------|-----|---------|
| `6b49db72` | `[mechanical]` | Ralph prompt refactor — thin core + doctrine preamble |
| `c0217727` | `[mechanical]` | Ralph iter 167 state-file orphan cleanup |
| `9784fa03` | `[mechanical]` | Ralph iter 168 audit-only (topstep_scaling_plan, 2 LOW ACCEPTABLE) |
| `66bd82ef` | `[judgment]` | corpus.py fail-closed on CRITICAL + query_agent logger on empty TextBlock |
| `a92f9c35` | `[judgment]` | cache_control crash fix on messages.parse() + 3 SDK-surface guards |
| `77a45464` | `[mechanical]` | Symmetric Pass-2 SDK guard for messages.create() |

All pushed. Working tree clean except unrelated Codex-session modifications (`.codex/*`, `HANDOFF.md`) — not mine.

---

## The CRITICAL bug — for posterity

`query_agent.py:163` passed `cache_control={"type":"ephemeral"}` as a top-level kwarg to `client.messages.parse()`. SDK has no such param on `parse()` — strict signature, no `**kwargs` — so every real API call would have raised:

```
TypeError: Messages.parse() got an unexpected keyword argument 'cache_control'
```

The existing `test_extract_intent_applies_cache_control` test asserted this exact broken shape (`call_kwargs['cache_control']`) and **passed** because `agent.client` is `MagicMock` — mocks accept any kwargs silently.

**Fix shape:** `cache_control` moved off `parse()` onto a `TextBlockParam` inside the `system` list (SDK-documented caching pattern for `parse()`).

**Lesson codified:** new `TestSDKSurfaceGuards` class in `tests/test_trading_app/test_ai/test_query_agent.py` uses `inspect.signature.bind()` to validate the exact kwargs we pass against the real `anthropic.resources.messages.Messages` signature — zero network, zero mock, catches what MagicMock hides. 4 guard tests, symmetric coverage of Pass-1 (`parse()`) and Pass-2 (`create()`).

---

## Ralph prompt — what changed structurally

`.claude/agents/ralph-loop.md` now has:

- **Step 0a (new, always):** `cat integrity-guardian.md + institutional-rigor.md` per iteration — replaces inlined Seven Sins + canonical-sources tables (which were stale vs canonical docs)
- **Step 1a (new, conditional):** target-area → doctrine doc table. e.g., research/ code loads `backtesting-methodology.md`; entry/sizing logic loads `mechanism_priors.md`; promotion/validation loads `pre_registered_criteria.md`. ≤1 doc per iteration beyond Step 0a pair.
- **Step 1c (pattern scan):** pointers to canonical rules with rule-number citations (e.g., "integrity-guardian.md § 6"). Three Ralph-specific sins kept inline (async safety, state persistence gap, contract drift) — not yet codified canonically.
- **Plan / history / final-report templates:** require `Doctrine cited` field. No doctrine citation = finding is not grounded.

Net +31 lines in prompt. Token budget per iter **improved** — iter 168 spent 76K vs iter 167's 100K.

---

## Ralph Loop state

- **Last iter:** 168 (`topstep_scaling_plan.py`, Priority 3 medium, audit-only, 2 LOW ACCEPTABLE)
- **`consecutive_low_only`:** 5 → next iter 169 MUST find Priority 1/2 candidate or triggers DIMINISHING_RETURNS exit
- **Priority 2 candidate queued:** `trading_app/db_manager.py` (critical tier, 13 importers, stale since iter 120 / 2026-03-16)
- **Open deferred debt:** 2 items (SR-L6, PP-167). See `docs/ralph-loop/deferred-findings.md`.

---

## Bloomey grade progression

| After commit | Grade | Why |
|--------------|-------|-----|
| Pre-review baseline | B+ | MEDIUM soft-fail on corpus + UNSUPPORTED SDK-surface items |
| `66bd82ef` | A- | Soft-fail + silent-empty-text-block fixed |
| `a92f9c35` | A | CRITICAL cache_control bug fixed + Pass-1 SDK guards added |
| `77a45464` | A+ | Symmetric Pass-2 SDK guard |

**Final: A+.** Zero silent failures, canonical compliance, SDK shape production-verified via `inspect.signature.bind()`, regression guards that fail fast on MagicMock-hidden bugs.

---

## Local-only change (not committed)

`.claude/hooks/post-edit-pipeline.py` is gitignored (per-user config). I edited it to prefer `.venv/Scripts/python.exe` over system `python` for subprocess calls — fixes the `Check 16: All imports resolve` failure where the hook's Python lacked `anthropic`. Hook now exits 0 cleanly. Fix persists across sessions because the working-tree file is not reset.

If the pattern matters for cross-machine setup: consider adding a tracked installer at `scripts/infra/install_hooks.py` in a future session. Not urgent — drift checks run clean via `.venv` for tests and pre-commit.

---

## Open threads for next session

1. **No research in flight.** Per the prior 04-18 handover: H2 book closed (Path C NULL), A4c garch allocator parked. Next research priorities unchanged: Phase D volume pilot (D-0 lane MNQ COMEX_SETTLE), Tier 1/2 non-garch hypothesis scan, or shadow H2/H1 signal-only tracking.
2. **Ralph iter 169:** If run, will either audit `db_manager.py` (Priority 2 stale re-audit) or trigger DIMINISHING_RETURNS.
3. **Pre-existing pyright diagnostics** (not this session's debt): `get_db_stats` fetchone None-guard in `corpus.py:138`; minor unused imports in test files. Low priority.

---

## Files touched this session

Production:
- `trading_app/ai/corpus.py` — fail-closed on CRITICAL missing files
- `trading_app/ai/query_agent.py` — logger on empty TextBlock + cache_control on TextBlockParam (not top-level)

Tests:
- `tests/test_trading_app/test_ai/test_corpus.py` — new `test_critical_missing_raises`
- `tests/test_trading_app/test_ai/test_query_agent.py` — rewritten `test_extract_intent_applies_cache_control` + new `TestSDKSurfaceGuards` class (4 tests)

Agent prompt:
- `.claude/agents/ralph-loop.md` — thin core + doctrine-load preamble refactor

Local-only (gitignored):
- `.claude/hooks/post-edit-pipeline.py` — `.venv` Python preference

Ralph state (committed by the agent itself):
- `docs/ralph-loop/{ralph-loop-audit.md, ralph-loop-history.md, ralph-ledger.json, deferred-findings.md}`

---

## Verification status at session end

- **59 → 113** ai/ tests pass via `.venv` (Python 3.13.9 with anthropic 0.96.0)
- Drift check via `.venv` Python: **103/0/6 advisory** (clean)
- Hook exit 0 on edit of `trading_app/ai/corpus.py` (post hook-env fix)
- 4/4 `TestSDKSurfaceGuards` green — covers both `messages.parse` and `messages.create` shapes
