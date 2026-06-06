# Income Unlock — max_contracts Clamp Lift + D-3 Sizing-Seam Close (SCOPE)

**Date:** 2026-06-06
**Mode:** DESIGN / SCOPE only — Tier B capital path, NO implementation without operator GO
**Why:** the bot's realistic banked cash at 1 micro is ~$2,500/yr (withdrawal-bound).
The entire income gap to ~$10k–17.5k/acct (×5 accts ≈ $50k–86k) is the
`max_contracts=1` clamp + the D-3 sim↔live sizing seam that gates lifting it.

---

## The architecture (verified 2026-06-06)

Three coupled sites, two layers:

| # | Site | Role | Current |
|---|---|---|---|
| A | `execution_engine.py:259` `_compute_contracts(..., max_contracts)` | LIVE sizer: vol-scales from equity (Carver Ch.9), then **clamps to `max_contracts`** (:294-301) | clamps to `strategy.max_contracts` |
| B | `portfolio.py:86` `max_contracts: int = 1` | the clamp VALUE | **1** |
| C | `account_survival.py:407` `contracts_per_trade = 1` | SURVIVAL GATE: projects 90d DD assuming 1 micro | **1** |

**The D-3 seam = the gap between A and C.** The live engine (A) would size
vol-scaled up to `max_contracts` (B). The survival gate (C) projects DD on a
HARDCODED 1 micro. Raise B to 5 without touching C and the gate certifies "DD
safe at 1 micro" while the engine trades 5 → **DD protection computed on the
wrong size = silent capital under-protection.** That is why the clamp can't just
be raised.

## The scaling-plan ceiling (the real per-account bound)

From `xfa_scaling_chart.png` (article 8284223, verified 2026-06-06) — max LOTS
held at once, balance-gated:

| Tier | Max lots | Requires balance > |
|---|---|---|
| 50k | **5** | $2,000 |
| 100k | **10** | $3,000 |
| 150k | **15** | $4,500 |

So `max_contracts` should NOT be a flat number — it must be `min(vol_scaled,
scaling_plan_lots(balance), survival_safe_lots)`. The clamp lift is really
"replace the flat 1 with the firm's balance-gated lot ladder, AND make the
survival gate project DD at the SAME lot count the engine will trade."

## Proposed stages (upstream-before-downstream)

**Stage 1 — Close the D-3 seam (survival gate sizes like the engine).**
Make `account_survival.py` project DD at the contract count the live engine WOULD
trade (vol-scaled, clamped to the scaling-plan lot ladder), not a hardcoded 1.
Files: `account_survival.py` (+ tests). NO behavior change to live trading yet
(survival is a gate, not an executor) — this is the SAFE first move. After this,
the gate's DD number is honest for any target lot count.
- Drift/guard: a new check asserting the survival gate's per-trade contract count
  equals the engine's sizing logic (no parallel re-encode — delegate to the same
  vol-sizer or a shared helper). Closes the "two sources of contract count" leak.

**Stage 2 — Wire the scaling-plan lot ladder as canonical.**
Add `SCALING_PLAN_LOTS` (5/10/15 balance-gated) to a canonical config module
(NOT hardcoded in the engine). `max_contracts` resolves from it per current
balance. Files: a config module + `portfolio.py`/`prop_profiles.py`.
- Respects `self-funded-sizing-doctrine.md`: prop lot caps bind ONLY prop
  survival + prop execution, NEVER self-funded earning capacity.

**Stage 3 — Lift the clamp (capital flip — operator GO + adversarial audit).**
Raise `max_contracts` default off 1 to the resolved lot ladder. Re-run survival
(now honest from Stage 1). Only after gate PASS at the new size + independent
adversarial audit + operator `--live` GO.

**Stage 4 — Re-run the income model at the true sustained lot count.**
Feed the real balance↔lots↔withdrawal coupling into `report_max_takehome.py`
(currently flagged unmodeled). Produces the defensible per-acct + 5× ceiling.

## What each stage unlocks (income)

- Stage 1 alone: $0 income, but makes every larger-size number TRUSTWORTHY.
- Stages 1–3: per-acct $2,500 → **$10k (50k) / $13k (100k) / $17.5k (150k)**
  withdrawal-bound (deep-window honest numbers TBD in Stage 4; the 1.15yr-window
  figures above are OPTIMISTIC and will come down).
- × 5 copy-traded accounts (sanctioned, ≤$750K BP): **~$50k–86k/yr ceiling.**

## Hard gates / risks (report, do not bypass)

- **Bracket audit 9b3fc530** — must be CLOSED before any multi-contract arming.
- **Edge-at-scale slippage UNMEASURED** at 5–15 lots — first-order at this size;
  must measure before trusting the ceiling.
- **MLL→$0 after first payout** shrinks balance → drops the lot ladder → the
  15-lot ceiling is NOT sustained post-withdrawal. Stage 4 must model this.
- **150k profile creation** is Tier B (no `topstep_150k_mnq_auto` exists yet).
- Self-funded doctrine: prop lot caps must not leak into personal-capital sizing.

## Decision

Stage 1 (close the D-3 seam) is the correct, SAFE first move — it touches the
survival GATE only, no live execution, and makes every downstream income number
honest. It is Tier B (touches `account_survival.py`, a canonical module) so it
needs operator GO to implement, but it creates ZERO new capital exposure.
