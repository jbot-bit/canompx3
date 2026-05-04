# Portfolio Audit — 38 validated vs 6 live allocator gap

**Date:** 2026-04-18
**Scope:** `active_validated_setups` (38) vs `topstep_50k_mnq_auto` live profile (6 lanes)
**Goal:** identify allocator-unlock candidates without new discovery or OOS consumption
**No OOS consumed.** No new queries against Mode A sacred window.

---

## Setup

**Live profile:** `topstep_50k_mnq_auto` at `trading_app/prop_profiles.py:420`
- `max_slots = 7`
- `copies = 2` (scale target: 5)
- `allowed_instruments = frozenset({"MNQ"})`
- `allowed_sessions` (7): CME_PRECLOSE, COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN, SINGAPORE_OPEN, TOKYO_OPEN, US_DATA_1000
- Lanes DYNAMIC via `docs/runtime/lane_allocation.json` (rebalanced 2026-04-18)

**Live 6 lanes (per lane_allocation.json 2026-04-18):**

| # | strategy_id | annual_r | trailing_expr | session_regime |
|---|---|---:|---:|---|
| 1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 46.1 | +0.1892 | HOT |
| 2 | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 44.0 | +0.2407 | HOT |
| 3 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 41.0 | +0.1756 | HOT |
| 4 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 29.0 | +0.1200 | HOT |
| 5 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 20.3 | +0.0934 | HOT |
| 6 | MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 18.8 | +0.0792 | FLAT |

Total annual R (6 lanes × 1 copy) = 199.2 R/yr. At `copies=2` = 398.4 R/yr per 2 accounts.

**Slot utilization:** 6 of 7 = **1 empty slot** (CME_PRECLOSE is the only allowed session with zero live lane).

---

## Exclusion breakdown (32 non-live + 2 paused = 34 excluded of 38 total)

| Bucket | Count | Reason | Unlockable? |
|---|---:|---|:---:|
| NON_FAMILY_HEAD | 15 | `is_family_head=False` — same strategy family as a live lane, deliberate dedup | **NO** (by design) |
| ALLOCATOR_NOT_SELECTED | 15 | Profile-eligible (MNQ + allowed session + family-head) but not picked by ranker | Maybe |
| PAUSED | 2 | MES CME_PRECLOSE — session regime COLD, AND instrument not allowed on this profile anyway | **NO** |

### Bucket 1: NON_FAMILY_HEAD (15) — DELIBERATELY DEDUPLICATED

These are same-session, same-direction variants of live lanes that the `is_family_head` gate correctly excludes to prevent double-counting the same alpha. Example: live L6 is US_DATA_1000 O15 RR1.5 ORB_G5; `VWAP_MID_ALIGNED_O15` at RR1.5 on same session has `is_family_head=False` because the family's current head is ORB_G5.

**Not unlockable.** Deploying any of these would add correlated positions on sessions already covered.

Notable one to flag: `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (ExpR=0.21, N=701, was L6 in C12 review). The allocator swapped away from it to ORB_G5_O15 on 2026-04-18 rebalance, making VWAP now non-head. Worth checking in a follow-up whether the swap reduced true EV — trailing_expr=0.0792 (ORB_G5) vs full-IS ExpR=0.21 (VWAP) is a sizable gap. NOT an unlock candidate; potential allocator swap-check item.

### Bucket 2: ALLOCATOR_NOT_SELECTED (15) — profile-eligible candidates

These pass the profile's instrument+session+family-head gates but lost to another lane in the same session's ranker. Grouped by session:

| Session | Non-live family-head candidates | Live lane (same session) |
|---|---|---|
| COMEX_SETTLE | RR1.0 × {ORB_G5, COST_LT12, OVNRNG_100, X_MES_ATR60} (4 lanes, ExpR 0.089-0.173) | RR1.5 ORB_G5 |
| EUROPE_FLOW | RR1.0 × {COST_LT12, OVNRNG_100}, RR1.5 CROSS_SGP_MOMENTUM (3 lanes, ExpR 0.092-0.118) | RR1.5 ORB_G5 |
| NYSE_OPEN | RR1.0 × {ORB_G5, X_MES_ATR60}, RR1.5 COST_LT12 (3 lanes, ExpR 0.089-0.137) | RR1.0 COST_LT12 |
| SINGAPORE_OPEN | O30 RR1.5 ATR_P50 (1 lane, ExpR 0.125) | O15 RR1.5 ATR_P50 |
| TOKYO_OPEN | RR2.0 ORB_G5 (1 lane, ExpR 0.087) | RR1.5 COST_LT12 |
| US_DATA_1000 | O5 RR1.0 X_MES_ATR60, O15 RR2.0 VWAP_MID_ALIGNED (2 lanes, ExpR 0.1-0.176) | O15 RR1.5 ORB_G5 |
| **CME_PRECLOSE** | **O5 RR1.0 X_MES_ATR60** (1 lane, ExpR=0.170, N=596) | **NONE — slot empty** |

**14 of 15 are same-session as an existing live lane.** Adding any of them triggers the rho-correlation gate (same session = high intra-day return correlation). The allocator's `correlation-survivor` gate is doing its job — supply is 7-9 per month in 2025 (per `docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md`) and the 6 picked are the top per-session representatives.

**The 1 exception: CME_PRECLOSE.** This session has ZERO live lanes despite being in `allowed_sessions`. The only candidate (`MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60`, ExpR=0.170, N=596) has strong stats but was excluded. Same reason as the paused MES CME_PRECLOSE lanes: **"Session regime COLD (-0.0917)"** — the allocator's session-regime gate blocks the session globally based on recent 12-month performance across all CME_PRECLOSE outcomes.

---

## Unlock ranking — "deployable if constraint relaxed"

| Rank | Lane | ExpR | Expected R/yr if deployed | Constraint to relax | P(justified) | Verdict |
|---|---|---:|---:|---|:---:|:---:|
| 1 | MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | 0.170 | ~25-35 R/yr (est from N=596 over ~7yr) | Session-regime COLD gate (or override) | **LOW** | DO NOT DEPLOY |
| 2 | MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 (swap for current L6) | 0.21 all-IS / 0.08 trailing | ~neutral vs current L6 (ORB_G5 trailing 0.079) | Allocator's swap decision (require trailing-window review) | MED | DEFER — allocator-swap audit |
| 3+ | Other 13 ALLOCATOR_NOT_SELECTED lanes | 0.087-0.176 | same-session duplicates | Correlation gate (rho > threshold) | **LOW** | DO NOT DEPLOY |

### Rank 1: CME_PRECLOSE X_MES_ATR60 — **DO NOT DEPLOY** under current regime

- Strong full-IS stats (ExpR=0.170, N=596, is_family_head=True)
- But **session regime is COLD**: trailing 12-month mean R = -0.0917 across ALL CME_PRECLOSE outcomes
- Both MES CME_PRECLOSE lanes (ORB_G8 and COST_LT08) are PAUSED for the same reason
- Deploying the MNQ lane now = expected negative drift until regime turns HOT
- **Correct action:** leave it paused; re-audit after regime turn signal

The allocator is doing its job. The empty slot 7 is empty because there is no non-cold-regime candidate to fill it. Alpha supply is the bottleneck, not the allocator.

### Rank 2: VWAP swap for L6 — **DEFER** pending allocator swap review

On 2026-04-18 the allocator swapped L6 from VWAP_MID_ALIGNED_O15 (ExpR=0.21 all-IS, N=701) to ORB_G5_O15 (ExpR=0.079 trailing-12mo). The former has BH K=36,000 validation and was the lane the C12 Shiryaev-Roberts review classified as KEEP (outperformance +0.47 R/trade trailing-30 vs IS baseline).

Either the trailing 12-month ExpR for VWAP degraded recently (swap justified), or the allocator's trailing window is catching noise. Worth an audit — but it's a swap question, not an unlock question. The slot is filled either way.

### Ranks 3+: same-session duplicates — **DO NOT DEPLOY**

Each would trigger rho > threshold against the existing live lane on its session. This is the allocator working as designed. The scarcity audit confirmed rho-survivor supply is 5-9 per month — already at or near the budget ceiling.

---

## CONCLUSION: unlock count = **0-1 lanes** (depending on regime-gate override)

The allocator is correctly picking 6 of 38 validated lanes. The non-live 32 split into:

- **15 deliberate family-dedup** (correct exclusion, by design)
- **14 same-session duplicates** that would trigger correlation gate (correct exclusion)
- **1 regime-cold candidate** (CME_PRECLOSE X_MES_ATR60) — technically deployable but currently expected-negative
- **2 paused MES lanes** on a different profile

**The 32-lane gap is NOT a deployment failure.** It is the allocator working as designed: family dedup + session dedup + regime gating correctly reduce 38 candidates to ~6-7 deployable under current regime conditions.

---

## What this unlocks instead — scaling copies

The REAL immediate-capital-unlock path sitting in `trading_app/prop_profiles.py`:

```
copies = 2  # Start with 1-2 Express, scale to 5 after proving loop
```

- Current: 2 Express accounts × 6 lanes × 199 R/yr = **398 R/yr aggregate**
- Scale target: 5 Express accounts × 6 lanes × 199 R/yr = **995 R/yr aggregate**
- **2.5x unlock** with zero new research, zero OOS consumption
- Gated on "proving loop" — live forward track record on the 2 existing Express accounts

This is the highest EV unlock currently available. It does not require new discovery, new infra, or new OOS. It requires continued clean forward execution on the existing 2 accounts to clear the scale-to-5 gate.

## What would unlock a 7th deployed lane

Since the allocator is correctly picking 6 of 7 slots, unlocking the 7th requires either:
1. **CME_PRECLOSE regime turns HOT** (wait-and-monitor — no action)
2. **New non-correlated alpha** on a different session or instrument — but VWAP + HTF families both killed in past 24hr, so current feature surface is exhausted for MNQ
3. **Instrument diversification** — MES CME_PRECLOSE has 2 validated lanes already. Activating `topstep_50k_mes_auto` profile (currently `active=False`) would add a separate 1-lane account once the MES session regime turns HOT. Not an MNQ-profile unlock, but a separate-account unlock.

---

## Action items

1. **Scale copies 2→5 on `topstep_50k_mnq_auto`** per the canonical scale-target note — this is the 2.5x unlock. Requires forward track-record demonstration (not in scope here). User decides when the "proving loop" criterion is met.
2. **Audit 2026-04-18 L6 swap** (VWAP→ORB_G5) as a separate task — confirm allocator's trailing-window basis and verify the swap is EV-positive vs ExecR=0.21 IS baseline.
3. **Monitor CME_PRECLOSE regime** — if it turns HOT, the MNQ X_MES_ATR60 lane becomes deployable (unlock count → 1). Similarly for MES lanes on the separate profile.
4. **No MNQ profile capacity unlock available today** from the 38-lane validated set. Discovery is the bottleneck, not deployment logic.

## What this audit did NOT do

- Did not consume any Mode A OOS data
- Did not run new discovery
- Did not propose new pre-regs
- Did not evaluate MGC discovery (separate task — F16 on prior surface map)
- Did not audit the 2026-04-18 allocator rebalance swap of L6 (flagged for follow-up)
