---
task: code-review-catchup-batch3
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
blast_radius: |
  Batch 3 reviews 9 dashboard commits (b978302d..b669de56 + 3 pre-cockpit-v3
  commits 11eb9f70, f48f7930, 157010e4). Touches dashboard SSE, action
  endpoints, poller retirement, kill-modal wiring. Read-only review with
  same-session implementation per CLAUDE.md mandate. Capital impact is
  MEDIUM (kill-modal is the proof case for the original code-review gap).
---

# Code Review Catch-Up — Batch 3 Resume Point

**Stage:** Batch 2 DONE, Batch 3 PENDING
**Model required:** Opus 4.7 (plan mandates Opus through Batch 6)
**Last commit:** `17d8b5cd` — `[code-review batch 2] session_orchestrator: 2 operator-clarity findings`
**Date:** 2026-05-15

## Batch 2 Outcome (DONE)

- 9 commits in scope reviewed independently
- Zero CRITICAL or HIGH findings
- 2 operator-clarity findings implemented (signal_record type rename, bracket-cancel warning split)
- 131/131 drift checks pass; 3118/3128 tests pass; 10 pre-existing skips
- Committed `17d8b5cd` on `main` (not yet pushed)

## Deferred Findings From Batch 2 (do NOT lose)

To be addressed in a separate cleanup batch after Batch 6, or sooner if Josh prioritizes:

1. **BrokerDispatcher dead-class** — 41 lines of API-parity infrastructure with zero production callsites confirmed on current main. Either wire it or delete it. Per `memory/feedback_code_review_dead_class_detection.md`.
2. **Pre-existing pyright errors cluster** — Josh's explicit ask: "we need to fix them after". Pattern:
   - `trading_app/live/session_orchestrator.py:2942/2959` — `Optional[None]` member access (cancel, query_order_status without None guard)
   - `tests/test_trading_app/test_session_orchestrator.py` — test fixture types bypass BrokerAuth / BrokerRouter / BrokerPositions / Bar protocols (FakeAuth, FakeRouter, FakePositions, FakeBar). `assert_not_called` / `return_value` on MethodType also flagged.
   - This is a class-bug requiring refactor (proper test fixture base classes), not patching. Per `institutional-rigor.md` §3.
3. **Tradovate verify_bracket_legs ID-heuristic mis-attribution risk** — if two entries land on the same contract back-to-back, the higher-ID-than-entry heuristic could mis-attribute legs. Mitigated by short verification window but worth a follow-up audit before any back-to-back paper-trade activation.

## Batch 3 Scope (RESUME HERE)

**Scope command (run first to refresh diff):**
```
git diff b978302d^..b669de56 -- trading_app/live/bot_dashboard.py trading_app/live/bot_dashboard.html
```

**Commits in range (cockpit-v3 stages):**
- `b978302d` Stage 1: signals-recent endpoint + legacy snapshot
- `ea5d5129` Stage 2: SSE event stream + /api/bars-recent
- `74ab60ec` Stage 2.1: adversarial-audit response
- `6f9a4eba` Stage 3: chart + SSE wiring + hold-to-kill
- `b80d6046` Stage 4: drawer + retire/gate redundant pollers
- `b669de56` harden(session-isolation): auto-recover stale Claude session locks

**Pre-cockpit-v3 dashboard commits (separate, also in scope):**
- `11eb9f70` Trade Book panel + paused-lane badge (PR #248)
- `f48f7930` cap paper_trades query at 5000 + truncation flag
- `157010e4` Automate dashboard readiness flow

## Batch 3 Focus Areas (from plan)

1. **SSE subscriber leak** — `/api/stream` must have a client disconnect handler removing the subscriber, or memory grows in long sessions
2. **Kill-path wiring** — HoldToKill modal (proof-case bug `f75157fe`) must stay intact through all 4 cockpit-v3 stages; verify `/api/action/kill` survives the drawer/poller retirement
3. **CSRF middleware preservation** — `/api/action/kill`, `/api/action/start`, etc. must still route through the origin-allowlist middleware added in Batch 1 (`db8df761`)
4. **Poller retirement completeness** — every `_retire_poller` call site must have a cleanup guard; no zombie pollers feeding stale data
5. **Localhost-only assertion** — server must still bind localhost-only (capital-class boundary)
6. **Cockpit-v4 Stage 1/2 revert risk** — Stages `82510553` Stage 1/2 landed today; verify no merge-conflict-style re-introduction of pre-fix HoldToKill logic

## Disconfirming Checks (per plan)

Run BEFORE accepting any cockpit-v3 commit as clean:

```bash
# SSE subscriber lifecycle
grep -n "subscribers" trading_app/live/bot_dashboard.py
grep -n "disconnect\|on_close\|remove.*subscriber" trading_app/live/bot_dashboard.py

# Kill-path through CSRF
grep -n "/api/action/kill\|csrf\|origin" trading_app/live/bot_dashboard.py

# Poller cleanup
grep -n "_retire_poller\|poller" trading_app/live/bot_dashboard.py

# Localhost binding
grep -n "127.0.0.1\|localhost\|host=" trading_app/live/bot_dashboard.py
```

## Per-Session Protocol (from plan)

1. Stay on Opus 4.7 — Sonnet missed HoldToKill modal bypass; this batch is where it shipped
2. Run scope command, get actual diff (not cached summary)
3. For each commit: PREMISE → TRACE → EVIDENCE → VERDICT
4. Implement findings same-session per CLAUDE.md mandate
5. Run `python pipeline/check_drift.py` (must hit 131/131)
6. Run `python -m pytest tests/test_trading_app/ -x -q` (3118 baseline)
7. Commit with `[code-review batch 3]` prefix (USE -F FILE, not -m heredoc — PowerShell leaked `@` on Batch 2 first attempt)
8. Update plan, mark Batch 3 DONE with date

## Bias Guards (do NOT skip on resume)

1. Adversarial-audit responses (Stage 2.1 `74ab60ec`) are CLAIMS, not proof — read the actual fix
2. Stage 4 poller-retirement is highest-risk for poller-leak bugs (silent failure class)
3. HoldToKill is the proof case for the original gap — re-verify it's still wired correctly
4. `b669de56` session-isolation auto-recover sounds defensive but auto-recover paths often hide silent failures — verify it fails LOUD on truly-corrupt locks, not just stale ones

## Commit Message File Pattern (REQUIRED)

PowerShell here-strings leak `@` into commit subjects. Use file-based:

```
Write commit message to tmp/batch3-commit-msg.txt then:
git commit -F tmp/batch3-commit-msg.txt
```

## Resume Instructions

After `/clear`:
1. Re-read this file (`docs/runtime/stages/code-review-catchup-batch3-resume.md`)
2. Confirm on Opus 4.7 — DO NOT proceed on Sonnet
3. Confirm current main is at `17d8b5cd` (Batch 2 commit) or later
4. Run the scope command to get the actual diff
5. Begin commit-by-commit review starting with `b978302d` Stage 1
