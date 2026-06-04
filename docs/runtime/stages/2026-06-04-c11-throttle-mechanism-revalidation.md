---
slug: c11-throttle-mechanism-revalidation
mode: CLOSED
created: 2026-06-04
closed: 2026-06-05
capital_path: false  # read-only harness outside repo + one result doc; NO production edit
---

## CLOSED (2026-06-05)
Stage 1 harness ran read-only against canonical `gold.db`. All acceptance criteria met:
- Baseline reproduced canonical max-90d-DD **$2,038.84 exactly** (parity anchor `True`).
- All 9 factors measured for mechanisms A/B; factor C logged **NOT-EVALUABLE** (daily-aggregate
  `DailyScenario` has no intraday trade sequence — logged, not omitted, per the acceptance rule).
- Result doc written: `docs/audit/results/2026-06-04-c11-throttle-mechanism-revalidation.md`
  (claim-hygiene PASS: scope / decision / files / verification / limitations).
- **VERDICT: no live-runnable integer participation mechanism clears C11 robustly.**
  `topstep_50k_mnq_auto` C11 remains **NO-GO, live NOT armed.**
- NO production-code edit, NO mechanism wired, NO arming — all blast-radius bounds held.

Stage 2 (capital-path participation-throttle wiring) is **moot at this grid** — the throttle
path is closed. Remaining live-C11 levers (multi-micro base size, intraday mechanism C,
ORB-cap/bracket-parity audit `9b3fc530`, larger account) are tracked in the result doc's
`## Next` and are separate, operator-GO-gated threads — NOT this stage.

# C11 throttle — live-mechanism re-validation harness (Stage 1, read-only)

## Task
Build a read-only re-validation harness (outside the repo, like the prior
`C:\Users\joshd\c11_matrix\throttle_validate.py`) that evaluates THREE integer
participation mechanisms — A (deterministic alternating skip), B (Bernoulli
p=0.5, seeded+logged), C (daily participation budget) — across ALL 9 factors,
and emits a factor matrix + an evidence-labelled recommended mechanism. The
chosen mechanism is an OUTPUT, not an input. No production code edit, no
mechanism committed, no arming.

Design: `docs/plans/2026-06-04-c11-throttle-live-mechanism-design.md`.

## The 9 factors (all required before any hard call)
1. C11 clearance (max-90d-DD vs $1,600 AND $2,000; zero breach-days).
2. Edge retention (full-history total R ≥85% AND holdout-window R).
3. WF stability (≥3/5 anchored steps; trigger/recover band width).
4. Throttle-aware survival (parity-proven MC; op-pass, MLL-breach, p95 DD).
5. Gate↔live reproducibility (identical INTEGER decisions; parity WITHOUT shared
   RNG — hard gate).
6. No-lookahead (day-t decision uses ≤t-1 only; permutation-verified).
7. Trade-drop distribution (pnl of skipped vs taken; winner-drop check).
8. Variance penalty (small-sample variance on a 1-micro book).
9. DD-basis sensitivity (holds under daily-close-peak AND HWM/eod-trailing).

## Acceptance
- Harness runs read-only against canonical `gold.db`; baseline reproduces the
  known $2,038.84 max-90d-DD exactly (parity anchor).
- For EACH mechanism (A/B/C): all 9 factors measured and reported (no factor
  silently skipped — a skipped factor must be logged as such, not omitted).
- factor=off / disabled path reproduces canonical `simulate_survival` exactly
  (MC parity anchor) before any mechanism is injected.
- Output result doc under `docs/audit/results/2026-06-04-c11-throttle-mechanism-revalidation.md`
  with the factor matrix, an evidence-labelled recommendation, and a Limitations
  section (claim-hygiene compliant: scope / decision / files / limitations).
- NO production-code edit; NO mechanism wired; NO live arming.

## scope_lock
- (new, outside repo) C:\Users\joshd\c11_matrix\throttle_mechanism_revalidate.py
- (new) docs/audit/results/2026-06-04-c11-throttle-mechanism-revalidation.md
- (this stage file)

## Blast Radius
- Harness is OUTSIDE the repo (c11_matrix), read-only against gold.db — no
  pipeline/ or trading_app/ edit, no schema, no capital path, no broker, no
  live_config, no lane/contract mutation.
- Reads: gold.db (read-only), canonical account_survival / prop_profiles logic
  (imported read-only or re-derived with a proven parity anchor, never mutated).
- Writes: one result MD under docs/audit/results/ + this stage file. Nothing else.
- The chosen mechanism is reported, NOT implemented. Stage 2 (capital path,
  Tier B) is a separate, unapproved stage gated on operator GO + adversarial audit.
