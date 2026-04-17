# MES + MNQ Wide-Relative-IB Distinctness Audit (v2)

**Date:** 2026-04-18
**Trigger:** NO-GO registry reopen flag (`docs/STRATEGY_BLUEPRINT.md` line 287).
**Question to answer:** Is `WIDE_REL_1.0X_20D` (`orb_size / 20d-trailing-same-session-mean >= 1.0`) a disguised version of `ORB_G5` (`orb_size >= 5.0`), or a genuinely distinct filter?
**Prior branch (`research/overnight-2026-04-18`):** produced a v1 distinctness audit using the wrong rho metric (continuous variable rho, not fire correlation) and peeked at 2026 OOS data to shape the pre-reg scope. Both errors are corrected in this v2.

**Verdict:** **Distinctness SURVIVES on Rule 7's correct metric.** Pre-reg scope narrowed to the 2 (instrument, session) combos with IS-significant conditional lift after BH-FDR: MNQ CME_PRECLOSE and MNQ TOKYO_OPEN. MES cells did NOT pass the conditional-lift test at p<0.05, contradicting the NO-GO registry's "MES-only" O5+O15-pooled finding (likely an aperture-pooling effect this O5-only audit doesn't reproduce).

---

## Corrections vs v1

| v1 error | v2 correction |
|---|---|
| Reported `rho(abs_size, rel_width)` = 0.71-0.75 as the Rule 7 result, concluded "fails tautology" | Use Rule 7's actual metric `|corr(WIDE_REL fire, ORB_G5 fire)|` on boolean fires; all 8 combos 0.13-0.55 (below 0.70) |
| Invented a "conditional-lift rescue" for a tautology flag that correct metric doesn't produce | No rescue needed; conditional lift is now SUPPORTING evidence not a rescue |
| Conditional lift reported as raw point estimates without t-test | Welch two-sample t-test + BH-FDR at K=8; only 2 of 8 combos significant |
| Peeked at 2026 OOS across 5 cells then excluded CME_PRECLOSE from pre-reg based on OOS direction | No OOS queried in this audit; pre-reg scope set by IS distinctness only |
| Mechanism narrative "session-local size regime signal" unsupported by literature | Explicit citation: Fitschen 2013 Ch 3 grounds intraday trend-follow core premise; extension to size-regime is PLAUSIBLE INFERENCE, not direct theory. Therefore t ≥ 3.79 (Criterion 4 without-theory threshold) applied |

---

## Scope (audit-level)

- 2 instruments (MES, MNQ) × 4 sessions (CME_PRECLOSE, TOKYO_OPEN, COMEX_SETTLE, EUROPE_FLOW)
- ORB aperture O5 (NO-GO mentions O15 also; deferred)
- WIDE_REL formula: `orb_{session}_size / rolling20(orb_{session}_size) >= 1.0`, rolling window is `ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING` (strict lag, zero look-ahead)
- ORB_G5 baseline: `orb_{session}_size >= 5.0` points absolute
- IS window only: `trading_day < 2026-01-01`. No 2026+ queries in this audit.

---

## Test 1 — Rule 7 tautology (fire correlation)

Canonical rule per `.claude/rules/backtesting-methodology.md` Rule 7:
> "Before claiming a new feature as additive to a deployed filter, compute: `|corr(new_feature_fire, deployed_filter_fire)| > 0.70` → flag TAUTOLOGY, exclude from survivors."

This measures Pearson correlation between BOOLEAN fire indicators (0/1), not continuous source variables.

| Instrument | Session | N_days | P(WIDE) | P(G5) | fire_rho | Rule 7 verdict |
|---|---|---:|---:|---:|---:|:---:|
| MES | CME_PRECLOSE | 1726 | 0.402 | 0.495 | +0.549 | distinct |
| MES | TOKYO_OPEN | 1793 | 0.413 | 0.200 | +0.429 | distinct |
| MES | COMEX_SETTLE | 1726 | 0.387 | 0.364 | +0.546 | distinct |
| MES | EUROPE_FLOW | 1792 | 0.407 | 0.218 | +0.464 | distinct |
| MNQ | CME_PRECLOSE | 1726 | 0.397 | 0.976 | +0.128 | distinct |
| MNQ | TOKYO_OPEN | 1793 | 0.407 | 0.932 | +0.210 | distinct |
| MNQ | COMEX_SETTLE | 1726 | 0.390 | 0.954 | +0.165 | distinct |
| MNQ | EUROPE_FLOW | 1789 | 0.414 | 0.925 | +0.232 | distinct |

**All 8 combos pass Rule 7.** fire_rho ranges 0.13-0.55, well below the 0.70 tautology threshold. Simple Rule 7 tautology flag does NOT trigger. No rescue needed.

Interesting side-observation: MNQ fires G5 at 93-98% rate across all sessions (MNQ's absolute ORB size is typically >= 5 points). MES fires G5 at 20-50%. This means on MNQ the G5 filter is nearly always on; the interesting variable is WIDE_REL alone. On MES, both filters are selective in different ways.

---

## Test 2 — Conditional lift within G5 (supporting evidence, not a rescue)

Question: among ORB_G5-fires only, does requiring WIDE_REL=True add positive ExpR delta?

Method: Welch two-sample t-test on ExpR(G5 fires with WIDE=T) vs ExpR(G5 fires with WIDE=F). E2 CB1 RR1.0 stop_mult=1.0, IS-only.

| Instrument | Session | N_wide+G5 | ExpR_wide | N_G5_only | ExpR_G5only | Δ | t_Welch | p_Welch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MES | CME_PRECLOSE | 397 | +0.1529 | 236 | +0.0178 | +0.1351 | +1.85 | 0.0656 |
| MES | TOKYO_OPEN | 271 | +0.0818 | 45 | −0.0174 | +0.0992 | +0.69 | 0.4908 |
| MES | COMEX_SETTLE | 432 | +0.0651 | 144 | +0.0500 | +0.0150 | +0.18 | 0.8598 |
| MES | EUROPE_FLOW | 305 | −0.0763 | 58 | −0.1477 | +0.0714 | +0.56 | 0.5781 |
| MNQ | CME_PRECLOSE | 497 | +0.1960 | 913 | +0.0572 | **+0.1387** | **+2.74** | **0.0063** |
| MNQ | TOKYO_OPEN | 690 | +0.1313 | 909 | +0.0193 | **+0.1120** | **+2.53** | **0.0114** |
| MNQ | COMEX_SETTLE | 635 | +0.0746 | 934 | +0.0819 | −0.0073 | −0.16 | 0.8761 |
| MNQ | EUROPE_FLOW | 703 | +0.0388 | 879 | +0.0701 | −0.0313 | −0.69 | 0.4883 |

### BH-FDR at K=8 (family of conditional-lift tests)

Sorted p-values with BH thresholds at q=0.05:

| rank | combo | p_Welch | BH threshold = (rank/8)×0.05 | BH-pass |
|---:|---|---:|---:|:---:|
| 1 | MNQ CME_PRECLOSE | 0.0063 | 0.00625 | **PASS** (0.0063 ≤ 0.00625 is marginal; 0.0063 = 0.00630 vs threshold 0.00625 is technically FAIL at 4-digit precision; using 5-digit t=2.74 → p=0.00633 → marginal) |
| 2 | MNQ TOKYO_OPEN | 0.0114 | 0.0125 | **PASS** |
| 3 | MES CME_PRECLOSE | 0.0656 | 0.01875 | FAIL |
| 4 | MES EUROPE_FLOW | 0.5781 | 0.025 | FAIL |
| 5-8 | rest | > 0.48 | > 0.03 | FAIL |

**BH-FDR K=8 survivors:** MNQ CME_PRECLOSE (borderline), MNQ TOKYO_OPEN (clean).

### Corrections vs v1 narrative

- v1 claimed "WIDE adds strong conditional lift on 4 of 8 combos (MES CME_PRECLOSE, MES TOKYO_OPEN, MNQ CME_PRECLOSE, MNQ TOKYO_OPEN)" based on eyeballing deltas. Proper Welch t-test at α=0.05 with BH-FDR at K=8 reduces this to 2 combos, BOTH MNQ. MES does NOT pass the conditional-lift test at p<0.05 on any session in O5.
- v1's claim "NO-GO's 'MES-only' narrative is wrong" was partially right but misstated. Correct read: the NO-GO cited O5+O15 pooled and a different method; at O5-only and with proper BH-FDR at K=8, **conditional lift is MNQ-specific, not MES-specific**. The NO-GO may be correct at O15 or with different methodology (can't verify without the original script).

---

## Test 3 — Alt-tautology vs daily vol regime

Checked correlation of WIDE_REL with `atr_20_pct` (daily vol regime feature, canonical):

- MES CME_PRECLOSE: rho(WIDE_REL, atr_20_pct) = −0.001 (essentially zero)
- MES CME_PRECLOSE: rho(abs_size, atr_20_pct) = +0.405 (moderate)

WIDE_REL is uncorrelated with daily vol. The ratio by construction divides out the shared vol factor. This is a mechanically different axis from `X_MES_ATR60` / daily-vol filters.

---

## Distinctness verdict

**SURVIVES.** Three supporting strands:

1. **Rule 7 (correct metric):** fire rho 0.13-0.55 on all 8 combos, well below 0.70. No tautology flag.
2. **Conditional lift (BH-FDR K=8 Welch t-test):** 2 combos pass at q=0.05 — MNQ CME_PRECLOSE (borderline) and MNQ TOKYO_OPEN (clean).
3. **Alt-tautology (vs vol state):** WIDE_REL uncorrelated with `atr_20_pct`. Not a disguised vol-regime filter.

**Pre-reg scope (no OOS peek):** the 2 IS-significant combos × 3 RR = **K=6**. See pre-reg YAML for locked surface.

**NOT included in pre-reg scope and why:**
- MES cells: conditional lift not significant at p<0.05 on any session in O5. NO-GO's MES-O5+O15 pooled claim does not reproduce at O5-only. O15 deferred to Stage 2 IF Stage 1 passes.
- MNQ COMEX_SETTLE / EUROPE_FLOW: conditional delta near zero or negative.
- CME_PRECLOSE on MES: conditional lift marginal at p=0.066 but does not clear BH-FDR K=8 threshold.

---

## Mechanism note and Criterion 4 threshold selection

### Literature support

`docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` grounds the CORE ORB premise:
- Fitschen Ch 3 Table 3.8 (stock indices hourly) and 3.9 (commodities hourly) show intraday trend-follow beats baseline for both asset classes
- Mechanism per Fitschen: trader-emotion-driven herd behavior drives intraday momentum on stocks; value fundamentals drive daily trend on commodities. Both converge to intraday trend-follow on both asset classes.

### Does Fitschen directly support "wider-than-normal ORB signals stronger trend-follow"?

**Honest answer: No, not directly.** Fitschen grounds the BASELINE intraday trend-follow premise but does not specifically test "ORB size relative to recent norm" as a conviction signal. The inference from Fitschen's emotion-driven mechanism is:
- Wider-than-normal ORB = more emotion/herding on this session = stronger trend-follow
- This is a one-step mechanism extension, plausible but not directly tested in the extract.

### Criterion 4 threshold

Per `pre_registered_criteria.md` Criterion 4:
> "Require t ≥ 3.00 (Harvey-Liu-Zhu 2015) for strategies with strong pre-registered economic theory support. Require t ≥ 3.79 (Chordia et al 2018) for strategies without such theoretical support."

The mechanism extension above is not "strong theory support" — it's plausible inference. **Apply the without-theory threshold: t ≥ 3.79.** (v1 used t ≥ 3.00 on the basis of an unsupported "session-local size regime" narrative. Corrected.)

---

## OOS-peek disclosure (audit trail hygiene)

The prior branch `research/overnight-2026-04-18` queried 2026+ OOS data on 5 cells as part of a scope-filtering exercise. Results observed:
- MES CME_PRECLOSE WIDE+G5: OOS ExpR = −0.313 (N=24)
- MNQ CME_PRECLOSE WIDE+G5: OOS ExpR = −0.009 (N=27)
- MNQ TOKYO_OPEN WIDE+G5: OOS ExpR = +0.088 (N=37)
- MES TOKYO_OPEN WIDE+G5: OOS ExpR = +0.075 (N=29)
- MES COMEX_SETTLE WIDE+G5: OOS ExpR = +0.097 (N=27)

**This v2 audit ignores those observations for scope-setting purposes.** Scope is set purely from IS distinctness (Tests 1-3 above). The prior OOS knowledge is disclosed here so the replay outcome can be honestly interpreted:
- If Stage 1 kills MNQ CME_PRECLOSE on OOS (expected given prior observation), the kill is a real pre-reg outcome, not a scope-filtering decision.
- If Stage 1 passes MNQ TOKYO_OPEN on OOS, the pass aligns with prior peek — still a valid outcome because TOKYO_OPEN is the dominant signal even without OOS.

This is the least-bad path given the prior peek. A fully clean pre-registration would require a Chinese wall that doesn't exist here.

---

## Next step (after this v2 commit)

Write `docs/audit/hypotheses/2026-04-18-mnq-wide-rel-ib-v2.yaml` (K=6) and `docs/plans/2026-04-18-mnq-wide-rel-ib-v2-design.md`. Replay script NOT authored in this commit — awaiting user authorization.
