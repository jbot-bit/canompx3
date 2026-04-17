# VWAP_BP_ALIGNED on MNQ CME_PRECLOSE — C8 Re-check

**Date:** 2026-04-18
**Pre-registration:** `docs/audit/hypotheses/2026-04-13-mnq-vwap-cme-preclose.yaml`
**Prior status:** BLOCKED at Criterion 8 (N_OOS=28 < 30 as of 2026-04-13)
**Re-check trigger:** hypothesis file said "Expected to clear ~2026-04-20"
**Verdict:** **DEAD — OOS direction reversal on all 3 RR cells, not a thin-OOS issue.**

## Scope (pre-reg locked, not widened)

- Instrument: MNQ
- Session: CME_PRECLOSE
- Aperture: O5
- Entry: E2, CB=1
- Stop multiplier: 1.0 (matches validated_setups convention for VWAP filter family)
- Filter: `VWAP_BP_ALIGNED` — break_price (orb_high for longs, orb_low for shorts) aligned vs `orb_CME_PRECLOSE_vwap`
- RR cells: 1.0 / 1.5 / 2.0 (exact pre-reg)
- IS window: `trading_day < 2026-01-01` (Mode A sacred)
- OOS window: `trading_day >= 2026-01-01` (accumulated to 2026-04-16 as-of)

## Results

| RR | N_IS | ExpR_IS | WR_IS | t_IS | N_OOS | ExpR_OOS | WR_OOS | t_OOS | dir_match | eff_ratio | C8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| 1.0 | 805 | +0.1669 | 0.637 | +5.35 | 29 | **−0.1557** | 0.448 | −0.88 | **FAIL** | −0.933 | **FAIL** |
| 1.5 | 713 | +0.1682 | 0.513 | +3.93 | 29 | **−0.3542** | 0.276 | −1.79 | **FAIL** | −2.106 | **FAIL** |
| 2.0 | 642 | +0.1621 | 0.427 | +3.04 | 28 | **−0.4962** | 0.179 | −2.39 | **FAIL** | −3.062 | **FAIL** |

OOS range: 2026-01-08 → 2026-04-16 (67 trading days, 28-29 fires per RR cell).

## Failure mode analysis

**Not a thin-OOS issue.**
- N_OOS sits at 28-29, still 1-2 below the pre-registered N_OOS ≥ 30 gate.
- But the DIRECTION has flipped from strongly positive IS (t=3.04 to 5.35) to strongly negative OOS (t=−0.88 to −2.39) on ALL three RR cells.
- Effect ratio is NEGATIVE on all three, between −0.93 and −3.06 — the magnitude of OOS reversal exceeds the IS signal on RR1.5 and RR2.0.
- WR_IS 51-64% vs WR_OOS 18-45% — WR drop of 15-25 percentage points per RR cell, proportional to RR (higher RR, worse OOS WR).

**This is a classic Mode A sacred-holdout kill pattern, same archetype as the IBS NO-GO entry (2026-04-13).** The filter survives IS K=425 audit (Apr 13) but reverses forward. Waiting for N_OOS to cross 30 would not rescue — the additional fires are moving AGAINST the IS sign.

## Criterion 8 semantics applied

Per `docs/institutional/pre_registered_criteria.md` Criterion 8 as referenced in this pre-reg file:
- N_OOS ≥ 30 (thin-OOS gate): **FAIL** on all 3 (still 29/29/28).
- dir_match (sign(IS) == sign(OOS)): **FAIL** on all 3.
- eff_ratio ≥ 0.40 of IS effect: **FAIL** on all 3 (all negative).

A passing cell would need ALL three. None pass any.

## Verdict

**DEAD.** This family does not clear Criterion 8 now, and will not clear it with additional data collection under current market regime — the OOS divergence is directional, not variance-driven.

## Doctrine actions

- Do NOT re-run this pre-reg.
- Do NOT widen scope (aperture, entry model, stop mult, etc.) — that would be rescue-tuning.
- Mark the Apr 13 pre-reg file as `verdict: DEAD 2026-04-18 OOS reversal` (no file mutation required; this result doc is the canonical kill note).
- **VWAP_MID_ALIGNED on MNQ US_DATA_1000 (deployed, 3 lanes) is UNAFFECTED.** Different session, different aperture, different VWAP reference. Those 3 lanes retain their OOS +0.15 to +0.21 per existing `validated_setups` fitness.
- Queue implication: user's original #1 (VWAP_BP CME_PRECLOSE re-check) is closed as DEAD. The queue collapses to #2 (MES wide-IB distinctness audit) as the highest active item.

## Reopen criteria (if any)

A fundamentally different mechanism would be required:
- Not a different stop_multiplier, RR, CB, or aperture on the same filter
- Not "wait for more OOS data" — the direction is flipped, more data won't fix it
- Would need a mechanism-level finding explaining WHY VWAP-BP-aligned days reversed post-holdout on this specific session (regime change evidence at the session level)

None of the above are on the queue. Closed.
