# Daily-Loss Dollar Cap — Canonical Spec

**Status:** SHIPPED 2026-05-26 (commits `c9ba1b92` feature + `89f8e97b` audit fix).
**Canonical home for:** `AccountProfile.daily_loss_dollars` (`trading_app/prop_profiles.py`)
and `RiskLimits.max_daily_loss_dollars` (`trading_app/risk_manager.py`).

> **Provenance.** This spec is the durable canonical record for the
> dollar-denominated daily-loss circuit breaker. It was relocated here from the
> now-swept implementation stage `docs/runtime/stages/2026-05-26-daily-loss-dollar-cap-wiring.md`
> (deleted in `5768c882` during a closed-stage sweep, which left two production
> `@canonical-source` annotations dangling). Per the stage-gate canonical
> anti-pattern rule, canonical content must live in `docs/specs/` — not in a
> `docs/runtime/stages/` baton that is subject to sweeps. Content below is the
> verbatim grounding from the original stage.

---

## Purpose

Wire an explicit dollar-denominated daily-loss circuit breaker for live
profiles, replacing the silent 5.0R magic default with an intentional,
calibrated dollar belt.

## Grounding (Topstep canonical resources)

- `topstep_50k_mnq_auto` is an Express Funded Account (XFA).
- Per `docs/research-input/topstep/topstep_dll_article.md`, XFA accounts have
  **NO mandatory broker Daily Loss Limit** — it is an opt-in safety net Topstep
  deliberately removed from auto-application. So `PropFirmAccount`
  `daily_loss_limit=None` for the topstep 50K tier is CORRECT, not a bug.
- The binding broker guard is the **$2,000 Maximum Loss Limit** (trailing DD),
  already wired fail-closed via `AccountHWMTracker`
  (`session_orchestrator.py:805-826`) and the cumulative equity DD breaker
  (`session_orchestrator.py:543`, `max_equity_dd_r = -$2000/avg_risk`).
- The per-DAY software circuit breaker (`Portfolio.max_daily_loss_r`, default
  5.0R) is a SELF-IMPOSED discipline belt, not a broker rule.

## Decision (operator, 2026-05-26)

A TRUE dollar-denominated breaker, halting at exactly **-$450 realized loss/day**
per account, summing real per-trade `pnl_dollars` — NOT an R-approximation.
Rationale: R drifts per-trade (each trade's R uses its own risk, not the
portfolio average), so an R-derived cap lands ~$400–650 not exactly the target.
The orchestrator already computes per-trade `pnl_dollars`
(`session_orchestrator.py:2057`) and the engine has `cost_spec` + contracts +
`risk_points`, so a dollar breaker is wireable end-to-end with crash recovery.

## Grounding for the $450 figure (official + empirical)

- **Carver 2015 Table 20** (`docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md:72-83`):
  for a prop account with ~$2K trailing DD, "worst daily loss each month" must
  be ≪ the DD → ≤25% vol target. $450 = 22.5% of the $2,000 per-account MLL,
  under Carver's ceiling.
- **EMPIRICAL:** 100k-day Monte Carlo on the REAL 2026 per-trade `risk_dollars`
  distribution from `orb_outcomes` (NOT the stored `validated_setups` median,
  which understates 2026 risk ~2x because MNQ price ~doubled over the 6yr
  backtest). 2026 per-account daily P&L: avg losing day -$156, 5%-tile -$288,
  1%-tile -$431, worst -$1018. $450 fires ~1% of days/account — a genuine tail,
  not ordinary variance. ($400 fires 1.5%/acct ≈ 11% household with copies=2 =
  strangles edge; $500 = exactly at Carver ceiling.)
- **PER-ACCOUNT semantics:** engine tracks ONE account's contracts;
  `CopyOrderRouter` mirrors to shadows. Each Topstep XFA has its own independent
  $2K MLL, so the breaker is correctly per-account. copies=2 → both halt
  together when primary hits the cap. A $431 per-account day = 22% of THAT
  account's MLL (compliant).
- **METHODOLOGY NOTE (operator catch):** risk limits MUST be calibrated on the
  live (current-year) risk distribution actually traded, never the historical
  backtest median. Stored `median_risk_dollars` is a 6yr average across a period
  where price was ~half today's.

## Implementation (true $ breaker, end-to-end)

1. `AccountProfile` gains `daily_loss_dollars: float | None = None`; set on
   `topstep_50k_mnq_auto` only (None = unchanged for all others).
2. `RiskLimits` gains `max_daily_loss_dollars: float | None = None`.
3. `RiskManager` tracks `daily_pnl_dollars`; `can_enter` Check 1 ALSO halts on
   dollar breach; `on_trade_exit` accepts `pnl_dollars`.
4. `ExecutionEngine` tracks `daily_pnl_dollars`, computes per-trade `pnl_dollars`
   (mirrors `orchestrator:2057`), passes to `on_trade_exit`.
5. `SessionOrchestrator` wires `RiskLimits.max_daily_loss_dollars` from profile;
   persists `daily_pnl_dollars` to `safety_state` symmetric to `daily_pnl_r`.
6. `SessionSafetyState` persists `daily_pnl_dollars` (crash recovery).
7. Drift check: declared `daily_loss_dollars` must be `< tier.max_dd` (MLL).

## Edge cases hardened (capital work)

- `daily_loss_dollars=None` → entire dollar path inert; R path unchanged.
- `pnl_dollars` uncomputable (`risk_points` None) → fail-closed: do NOT silently
  skip the dollar accrual; log + use R-equivalent fallback OR refuse entry.
- Crash mid-day → `safety_state` restores `daily_pnl_dollars` AND `daily_pnl_r`;
  both must match the same `trading_day` stamp or reset.
- EOD rollover → `daily_pnl_dollars` resets with `daily_pnl_r` (same daily_reset).
- Partial fills / multi-contract → `pnl_dollars` uses `event.contracts`.
- Signal-only → breaker still accrues (preview), never routes orders.

## Ship + audit record

SHIPPED 2026-05-26. Commits `c9ba1b92` (feature) + `89f8e97b` (audit fix).
$450/account dollar daily-loss breaker, true-dollar end-to-end.

Adversarial-audit gate: FIRST PASS FAIL (evidence-auditor caught CRITICAL —
breaker wired only into session-end SCRATCH path, dead for live `_exit_trade`
exits). Fixed in `89f8e97b`: `_exit_trade` now computes + forwards `pnl_dollars`
for all win/loss/early-exit/hold-timeout/ib-opposed outcomes. RE-AUDIT: PASS
(double-accrual structurally impossible — EXITED trades pruned before scratch
path; sign correct; regression test would have caught the original bug).
Verification: 224 unit + 235 orchestrator pass; drift 165/0; ruff clean.

**DEFERRED** (audit CONDITIONAL, separate stage): daily-loss halt blocks new
entries but does not auto-flatten open positions. Same behavior as existing
R-cap; broker bracket + $2K MLL HWM tracker are the live guards.
Flatten-on-daily-halt is a design decision affecting BOTH caps — needs its own
stage.

## Doctrine references

- `docs/research-input/topstep/topstep_dll_article.md` — XFA has no mandatory DLL.
- `.claude/rules/institutional-rigor.md` § 4 — reuse canonical median-risk
  conversion, don't re-encode.
