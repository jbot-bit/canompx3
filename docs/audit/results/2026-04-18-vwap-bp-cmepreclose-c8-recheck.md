# VWAP_BP_ALIGNED on MNQ CME_PRECLOSE — C8 Re-check (v2)

**Date:** 2026-04-18
**Pre-registration:** `docs/audit/hypotheses/2026-04-13-mnq-vwap-cme-preclose.yaml` (written 2026-04-13)
**Governing doctrine at review time:** `docs/institutional/pre_registered_criteria.md` Amendment 2.7 (Mode A sacred holdout 2026-01-01)
**Relationship between the two:** the Apr 13 pre-reg file predates Amendment 2.7 (which was committed 2026-04-09 forward). The pre-reg's own `kill_criteria` block does not include OOS direction match or `eff_ratio ≥ 0.40`; those are Amendment 2.7 criteria applied retroactively as the governing standard at this review.
**Prior run (on branch `research/overnight-2026-04-18`):** documented the same verdict but cited the kill as if it flowed from the pre-reg's own terms, which it does not. See code review findings for Task 1 basis correction.

**Verdict:** **DEAD under Amendment 2.7 Criterion 8. Pre-reg's own kill_criterion 3 (N_OOS < 30) is still "retry when data accumulates" by one sample per cell (N=29/29/28), but Amendment 2.7 governs.**

---

## Scope (pre-reg locked)

- Instrument: MNQ
- Session: CME_PRECLOSE
- Aperture: O5
- Entry: E2, CB=1
- Stop multiplier: 1.0 (matches `validated_setups` convention for the VWAP filter family)
- Filter: `VWAP_BP_ALIGNED` — break_price (`orb_high` for longs, `orb_low` for shorts) aligned vs `orb_CME_PRECLOSE_vwap`
- RR cells: 1.0 / 1.5 / 2.0 (exact pre-reg)
- IS window: `trading_day < 2026-01-01` (Mode A sacred boundary)
- OOS window: `trading_day >= 2026-01-01` (accumulated through 2026-04-16)
- Canonical DB: `pipeline.paths.GOLD_DB_PATH` (= `C:\Users\joshd\canompx3\gold.db`)

---

## Results

| RR | N_IS | ExpR_IS | t_IS | p_IS | N_OOS | ExpR_OOS | dir_match | eff_ratio | max_OOS_day |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|
| 1.0 | 805 | +0.1669 | +5.35 | 0.00000 | 29 | −0.1557 | FAIL | −0.933 | 2026-04-16 |
| 1.5 | 713 | +0.1682 | +3.93 | 0.00009 | 29 | −0.3542 | FAIL | −2.106 | 2026-04-16 |
| 2.0 | 642 | +0.1621 | +3.04 | 0.00249 | 28 | −0.4962 | FAIL | −3.062 | 2026-04-16 |

## Pre-reg kill criteria (as literally written in `2026-04-13-mnq-vwap-cme-preclose.yaml`)

| Criterion | Per-cell result | Kill triggered? |
|---|---|:---:|
| 1. BH-FDR q=0.05 fails at K=3 | raw p = 0 / 9e-5 / 2.5e-3; BH thresholds = 0.0167 / 0.0333 / 0.0500 — ALL pass | NO |
| 2. WFE < 0.50 | Not computed in this re-check (would require 5-fold expanding window; deferred pending doctrine decision) | UNKNOWN |
| 3. N_OOS < 30 (time-structural, retry when data accumulates) | N_OOS = 29/29/28 — ALL below 30 | **YES — retry pending** |

**By the pre-reg's own terms:** verdict is "RETRY PENDING N_OOS ACCUMULATION" — one more eligible trade per cell and N_OOS hits 30.

## Amendment 2.7 governing criterion

Amendment 2.7 of `docs/institutional/pre_registered_criteria.md` (committed 2026-04-09 forward) makes Criterion 8 BINDING with:

> "Under Mode A: `--holdout-date 2026-01-01` required for discovery; OOS ExpR ≥ 0 AND OOS ExpR ≥ 0.40 × IS ExpR on the 2026-01-01 → current window"

Per-cell Amendment 2.7 check:

| RR | OOS ExpR ≥ 0 | OOS ExpR ≥ 0.40 × IS ExpR | C8 verdict |
|---:|:---:|:---:|:---:|
| 1.0 | FAIL (−0.1557) | FAIL (ratio −0.93) | **FAIL** |
| 1.5 | FAIL (−0.3542) | FAIL (ratio −2.11) | **FAIL** |
| 2.0 | FAIL (−0.4962) | FAIL (ratio −3.06) | **FAIL** |

All three cells fail Criterion 8 under Amendment 2.7. None return positive OOS; all have negative eff_ratio.

## Doctrine basis for kill

- **Pre-reg's own kill_criterion 3** would return "retry when data accumulates."
- **Amendment 2.7 governing criterion at review time** returns DEAD on all 3 cells.
- These two are not in contradiction: the pre-reg file predates Amendment 2.7, which adds a direction + ratio requirement to the N_OOS ≥ 30 gate. Under the newer (governing) doctrine, the pre-reg is retired.

**Kill verdict: DEAD under Amendment 2.7 Criterion 8.** The underlying OOS direction flip on all three cells (IS t=3.04-5.35 → OOS t=−0.88/−1.79/−2.39) is directional, not variance-driven — waiting for N_OOS to reach 30 would not rescue. This is the same archetype as the IBS NO-GO entry (Mar 2026) and mirrors the WIDE_REL CME_PRECLOSE pattern observed elsewhere in the shelf.

## Downstream actions

- The Apr 13 pre-reg file is retired as DEAD; no re-run authorized, no widening-scope rescue.
- **Do not** remove the pre-reg YAML file — keep it as part of the audit trail.
- The pre-reg's `kill_criteria` block could optionally be amended to cite Amendment 2.7 as governing, but that's a documentation hygiene item, not a research action.
- `VWAP_MID_ALIGNED` on MNQ US_DATA_1000 (3 deployed validated lanes) is **unaffected** — different session, different VWAP reference (`orb_mid` not `break_price`). Those 3 lanes retain their OOS +0.15 to +0.21 per existing `validated_setups`.

## Reopen criteria

Would require a fundamentally new mechanism claim explaining WHY the OOS reversal happened on this specific session. Not on queue. Not a priority.

## Verification log

- Canonical DB path confirmed via `pipeline.paths.GOLD_DB_PATH` (not hardcoded).
- Filter logic cross-checked against `trading_app/config.py::VWAPBreakDirectionFilter.matches_row()` L2463-2486 (break_price branch). SQL predicate reproduces it exactly.
- Daily-features triple-join respected: `ON trading_day AND symbol AND orb_minutes`.
- Mode A holdout boundary literal: `trading_day < DATE '2026-01-01'` for IS, `>= DATE '2026-01-01'` for OOS.
- BH-FDR computed at K=3 per the pre-reg's own K declaration.
- Amendment 2.7 thresholds (OOS ExpR ≥ 0, OOS/IS ratio ≥ 0.40) applied per the governing doctrine at review time.
