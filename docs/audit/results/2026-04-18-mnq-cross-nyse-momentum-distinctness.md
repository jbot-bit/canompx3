# MNQ CROSS_NYSE_MOMENTUM Distinctness Audit

**Date:** 2026-04-18
**Candidate filter:** `CROSS_NYSE_MOMENTUM` (existing filter class in `trading_app/config.py:2976`, prior_session="NYSE_OPEN")
**Mechanism class:** 4-state cross-session momentum per `CrossSessionMomentumFilter` (`config.py:2558`), validated via deployed `CROSS_SGP_MOMENTUM` on MNQ EUROPE_FLOW (3 lanes).
**Scope under audit:** MNQ US_DATA_1000 + MNQ NYSE_CLOSE × RR {1.0, 1.5, 2.0} × O5, E2 CB=1 stop_mult=1.0
**Verdict:** **DISTINCTNESS SURVIVES.** Rule 7 fire correlation near zero on US_DATA_1000; NYSE_CLOSE has no deployed filters so no tautology check applies.

---

## Prior-work disclosure (informational — not scope-shaping)

This family is NOT "newly discovered." It was scouted at Phase A on 2026-04-13 per `trading_app/config.py:2578-2580` docstring and `docs/plans/2026-04-11-cross-session-state-round3-memo.md`:

- **Apr 13 Phase A screening:** 3/3 BH FDR at K=18 on MNQ US_DATA_1000 × CROSS_NYSE_MOMENTUM for RR {1.0, 1.5, 2.0}
- **Round-2 matrix on 4 sibling states** (all significant):
  - TAKE_WIN_ALIGN: IS +0.140 / OOS +0.481 (eff_ratio 3.4×)
  - VETO_WIN_OPP: IS +0.185 / OOS +0.340 (eff_ratio 1.8×)
  - VETO_LOSS_ALIGN: IS +0.147 / OOS +0.407 (eff_ratio 2.8×)
  - TAKE_LOSS_OPP: IS +0.129 / OOS +0.285 (eff_ratio 2.2×)
- **Round-3 Pack A was designed** (memo §Round-3 Pack A) but never executed
- **`experimental_strategies` table: zero rows.** Formal validator pipeline never ran this family
- **OOS ratios 1.8-3.4× IS:** `LEAKAGE_SUSPECT` per quant-audit-protocol §T3. Memo predates Amendment 2.7 Mode A (2026-01-01 holdout boundary); OOS window likely different

**Treatment in this pre-reg:** the memo evidence is flagged in the audit trail. Scope is set from current IS evidence + Rule 7 distinctness, NOT from memo lifts. This avoids memo-driven scope shaping while preserving institutional transparency.

---

## Test 1 — Rule 7 tautology (fire correlation)

Canonical Rule 7 per `.claude/rules/backtesting-methodology.md`:
> "`|corr(new_feature_fire, deployed_filter_fire)| > 0.70` → flag TAUTOLOGY, exclude from survivors."

Metric: Pearson correlation on BOOLEAN fires (not continuous variable rho — correction from v1 wide-rel-IB).

### MNQ US_DATA_1000 × CROSS_NYSE_MOMENTUM at O5

Deployed filters on this lane:
| Deployed filter | Aperture | Aperture match to O5? | Fire correlation needed? |
|---|---|:---:|:---:|
| VWAP_MID_ALIGNED | O15 | no | No (different aperture) |
| ORB_G5 | O15 | no | No |
| X_MES_ATR60 | O5 | **yes** | **YES** |

Fire rho vs X_MES_ATR60:
- MNQ US_DATA_1000 O5 CROSS_NYSE_MOMENTUM vs X_MES_ATR60: **fire_rho = −0.025** (N=1784)
- Well below 0.70 threshold. **Distinct.** ✓

### MNQ NYSE_CLOSE × CROSS_NYSE_MOMENTUM at O5

Deployed filters on this lane: **NONE** (zero validated_setups for MNQ NYSE_CLOSE).

No tautology check applies. Baseline = RAW unfiltered MNQ NYSE_CLOSE E2 O5 trades.

---

## Test 2 — Fire rate (Rule 8.1 check)

Rule 8.1: `fire_rate < 5%` or `fire_rate > 95%` → flag `extreme_fire`, exclude.

4-state distribution per lane (all MNQ, O5, full trading history):

### MNQ US_DATA_1000

| State | N | % |
|---|---:|---:|
| TAKE_WIN_ALIGN | 637 | 35.6% |
| TAKE_LOSS_OPP | 564 | 31.5% |
| VETO_LOSS_ALIGN | 262 | 14.6% |
| VETO_WIN_OPP | 326 | 18.2% |
| **Total** | **1789** | 100% |
| **TAKE (filter fires)** | **1201** | **67.1%** |

Fire rate 67.1% — between 5% and 95%. No extreme-fire flag. ✓

### MNQ NYSE_CLOSE

| State | N | % |
|---|---:|---:|
| TAKE_WIN_ALIGN | 447 | 29.7% |
| TAKE_LOSS_OPP | 393 | 26.1% |
| VETO_LOSS_ALIGN | 314 | 20.9% |
| VETO_WIN_OPP | 350 | 23.3% |
| **Total** | **1504** | 100% |
| **TAKE (filter fires)** | **840** | **55.9%** |

Fire rate 55.9%. Also within bounds. ✓

---

## Test 3 — IS expectancy scoping (no OOS peek)

For each cell, `delta_IS = ExpR(TAKE state only) - ExpR(all 4-state days)`, `trading_day < 2026-01-01`:

| Session | RR | N_TAKE | N_VETO | ExpR_TAKE | ExpR_VETO | ExpR_all | **delta_IS** |
|---|---:|---:|---:|---:|---:|---:|---:|
| US_DATA_1000 | 1.0 | 1145 | 556 | +0.0871 | +0.0857 | +0.0867 | **+0.0005** |
| US_DATA_1000 | 1.5 | 1124 | 550 | +0.1194 | +0.0377 | +0.0926 | **+0.0269** |
| US_DATA_1000 | 2.0 | 1098 | 541 | +0.1444 | −0.0172 | +0.0910 | **+0.0533** |
| NYSE_CLOSE | 1.0 | 445 | 360 | +0.1956 | −0.0545 | +0.0838 | **+0.1119** |
| NYSE_CLOSE | 1.5 | 325 | 287 | +0.1174 | −0.1549 | −0.0103 | **+0.1277** |
| NYSE_CLOSE | 2.0 | 271 | 244 | +0.0185 | −0.3746 | −0.1678 | **+0.1863** |

**Observations:**
- NYSE_CLOSE shows larger IS deltas than US_DATA_1000 across all RR (opposite of the Apr 11 memo's "strongest branch" narrative on US_DATA_1000). Possible reasons: (a) current-data state differs from Apr 13 snapshot, (b) Apr 13 scan used different OOS window, (c) NYSE_CLOSE benchmark baseline is worse, inflating the delta.
- US_DATA_1000 RR1.0 delta is essentially zero (+0.0005). Signal grows with RR target.
- NYSE_CLOSE RR2.0 has N_TAKE=271 — thinnest cell. Cross-ref trust accordingly.
- All 6 cells show positive TAKE delta in IS. That's a coherent sign. BUT proper statistical test (Welch t-test + BH-FDR K=6) is Stage 1 replay work, not scoping.

**These IS numbers are scoping checks only. Pre-reg DOES NOT use them to narrow cells — all 6 go into Stage 1.**

---

## Test 4 — Look-ahead freedom

Cross-session momentum filter reads prior-session data. Verify prior session STRICTLY PRECEDES current session within same trading day:

| Current | Current time (Brisbane) | Prior (=NYSE_OPEN) | Prior time (Brisbane) | Prior-before-current? |
|---|---|---|---|:---:|
| US_DATA_1000 | ~01:00 | NYSE_OPEN | ~00:30 | **YES** ✓ |
| NYSE_CLOSE | ~07:00 | NYSE_OPEN | ~00:30 | **YES** ✓ |

Same trading day (Brisbane 09:00 → next 09:00). Both current sessions fall after NYSE_OPEN's break time. No look-ahead. ✓

Filter implementation (`config.py:2604-2608`) reads `orb_NYSE_OPEN_break_dir` and `orb_NYSE_OPEN_high/low` — all computed at NYSE_OPEN's ORB close time. Known before current session's break. ✓

---

## Verdict

**DISTINCTNESS SURVIVES.** Three clean supporting strands:

1. **Rule 7 (correct boolean fire metric):** fire rho = −0.025 on MNQ US_DATA_1000 vs X_MES_ATR60. NYSE_CLOSE has no deployed alternatives. Both lanes pass.
2. **Fire rate bounds (Rule 8.1):** 67.1% and 55.9% — within 5%-95% window.
3. **Look-ahead free:** prior session (NYSE_OPEN) strictly precedes both current sessions (US_DATA_1000, NYSE_CLOSE) within same Brisbane trading day.

**IS scoping preview (for audit trail, not scope-narrowing):**
- All 6 cells show positive TAKE-state delta vs all-4-state-day baseline
- NYSE_CLOSE cells have larger deltas than US_DATA_1000 — opposite of Apr 11 memo finding (memo was US_DATA_1000-focused)
- Thinnest cell: MNQ NYSE_CLOSE RR2.0 (N_TAKE=271)

**Pre-reg scope preserved at K=6 per user directive.** No post-hoc cell exclusion from scoping IS numbers.

**Mechanism citation path:**
- Fitschen 2013 Ch 3 grounds intraday trend-follow core premise
- `CROSS_SGP_MOMENTUM` deployed on MNQ EUROPE_FLOW = empirical proof-of-concept for the 4-state cross-session momentum class
- This pre-reg applies the SAME mechanism class to different sessions on the same instrument
- Still applies **t ≥ 3.79 (without-theory) threshold** per v2 precedent — mechanism extension to new session pairings is inference not direct test
