# C11 Throttle — Corrected Live Mechanism + Re-Validation (Design)

**Status:** DESIGN — approved to build Stage 1 (read-only harness). No production edit, no
mechanism pre-committed, no live arming. Supersedes the size-scale assumption in
`docs/plans/2026-06-04-c11-throttle-implementation.md` (see its "Second-pass review findings").
**Date:** 2026-06-04
**Predecessors:** hypothesis `docs/audit/hypotheses/2026-06-04-c11-equity-drawdown-throttle.yaml`;
validation `docs/audit/results/2026-06-04-c11-throttle-validation.md` (pnl-scale model — NOT
the live realization).

---

## Purpose

The validated throttle halves *daily participation* by scaling pnl 0.5×. The live book emits
a fixed 1 micro per signal (`event.contracts` defaults to 1 across `position_tracker.py`,
`trade_journal.py`; resolved at `session_orchestrator.py` 2194/2544/2602). The canonical
execution resolver `prop_profiles.resolve_execution_order` FAILS CLOSED on any non-integer
quantity ("truncating would under-size and rounding would over-size"). So a 0.5× size scale is
not merely unimplemented — it is forbidden by canonical doctrine at 1 micro. The validated
$1,459 max-90d-DD therefore describes a model the bot cannot execute. This design fixes that:
choose a live mechanism that reduces participation by ~half using only INTEGER quantities, and
re-validate THAT mechanism so the number we arm on is the number the bot actually produces.

## Grounded facts (verified against source 2026-06-04)

- Live size origin: `event.contracts = 1` everywhere; multi-contract is modelled
  (`TradingBookEntry.contracts`) but = 1 for `topstep_50k_mnq_auto`.
- Resolver `prop_profiles.resolve_execution_order:197` raises `ValueError` on indivisible qty.
- Live DD authority: `account_hwm_tracker._dd_used:530` = `max(0, hwm − last_equity)`, HWM-based,
  with a None/NaN poll-failure guard (`:546-559`).
- Gate DD: daily cumulative-pnl running peak → `_max_observed_rolling_drawdown:749`; i.i.d.
  bootstrap in `simulate_survival:608`.
- No prior NO-GO on equity-drawdown participation control (blueprint + graveyard clean).

## Candidate mechanisms (NO pigeonholing — data picks)

- **A — Deterministic alternating skip.** While engaged, take every other eligible signal.
  Exactly half over a run; fully causal; reproducible with NO shared RNG between gate and live.
  Risk: phase-dependence — which trades drop is deterministic-but-arbitrary.
- **B — Probabilistic skip (Bernoulli p=0.5).** Take each engaged signal with prob 0.5.
  Unbiased over which trades drop. Risk: needs a seeded+logged draw shared by gate and live,
  else divergence; higher small-sample variance on a 1-micro book.
- **C — Daily participation budget.** While engaged, cap engaged-day participation (e.g. max 1
  trade/day or stop after first loss). Maps onto the daily aggregate already validated. Risk:
  changes intra-day shape, not just count.
- (Multi-contract books only: a true integer size scale becomes valid at ≥2 contracts — out of
  scope for `topstep_50k_mnq_auto` today; harness notes it but does not select on it.)

**The chosen mechanism is an OUTPUT of the harness, not an input.** Lean is Take A
(exactly-half, reproducible, parity-safe with no RNG) but the 9-factor matrix decides.

## The 9 factors the harness MUST measure per mechanism (before any hard call)

1. C11 clearance — max-90d-DD vs $1,600 strict AND $2,000 MLL; zero breach-days preserved.
2. Edge retention — full-history total R (≥85% floor) AND holdout-window R (2025 was costly).
3. WF stability — clears C11 in ≥3/5 anchored steps; trigger/recover band width (knife-edge).
4. Throttle-aware survival — parity-proven MC (factor-off reproduces canonical exactly),
   op-pass, MLL-breach prob, p95 DD — for the INTEGER mechanism.
5. Gate↔live reproducibility — identical integer decisions on both sides; can it hold parity
   WITHOUT shared RNG? (A/C yes; B only with seeded+logged draw). HARD GATE, not a tiebreaker.
6. No-lookahead — day-t decision uses only ≤t-1 drawdown; verified by permutation.
7. Trade-drop distribution — pnl of skipped vs taken trades; does the mechanism systematically
   drop winners (Take A phase risk)?
8. Variance penalty — small-sample variance on a 1-micro book (especially B).
9. DD-basis sensitivity — result holds under BOTH daily-close-peak and HWM/eod-trailing basis,
   so the basis decision is evidenced not assumed.

## DD-basis fork (now decidable, honestly)

Because the live realization acts per-signal on integer counts, the cleanest shared basis is a
daily-close cumulative-pnl peak computed identically on both sides. This DOES require the live
side to track that peak as new state — correcting the prior plan's "no new persisted state"
claim. Factor 9 tests whether the verdict survives the alternative HWM/eod-trailing basis; the
basis is pinned by evidence before any live edit.

## Layer placement & one-way dependency

New canonical participation-policy module in `trading_app/` (one source imported by both gate
and live). Both consumers already live in `trading_app/`; pipeline→trading_app one-way rule
unaffected. Throttle decision applied to `event.contracts` BEFORE `_resolve_execution_order`,
honoring the fail-closed integer law.

## Staged build (Stage 1 only is approved)

- **Stage 1 (APPROVED — read-only, no production edit):** re-validation harness outside the
  repo (like the prior `c11_matrix` harness), runs A/B/C across all 9 factors on canonical
  read-only data; emits a factor matrix + a recommended mechanism with evidence labels. NO
  source change, NO mechanism committed, NO arming. Output: a result doc under
  `docs/audit/results/`.
- **Stage 2 (NOT approved — capital path, Tier B):** implement the chosen mechanism as the
  canonical policy module + profile binding + gate + live + drift parity check + tests. Gated
  on: Stage 1 verdict → operator GO → adversarial-audit gate.
- **Stage 3 (NOT approved):** arming is a separate explicit operator decision after Stage 2 +
  audit.

## Verification gates before Stage 2 is "done" (carry-forward)

Baseline byte-identical when disabled; chosen mechanism clears both C11 gates with WF
stability; gate/live produce identical INTEGER decisions on the same drawdown path; poll-failure
fail-safe (do not silently un-throttle on a degraded equity poll); no-lookahead green;
`check_drift.py` full pass incl. new parity guard; adversarial audit (capital path).

## Rollback

Ships disabled; one-line enable-flag flip reverts gate + live together; additive module +
guarded call-sites → clean `git revert`. New daily-peak tracked state is documented and cleared
on rollback.

## Live status

STILL BLOCKED for live. This design arms nothing. Every hard call (mechanism, DD basis, arming)
is deferred to evidence + operator GO + audit.
