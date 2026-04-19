# Portfolio Audit — Adversarial Re-Open from First Principles

**Date:** 2026-04-18 (re-opened 2026-04-18 evening)
**Supersedes:** portions of `2026-04-18-portfolio-audit-38-validated-vs-6-live.md` (prior verdict: "allocator is working, unlock = 0-1 lanes"). Prior audit used `active_validated_setups` and `lane_allocation.json` as proof — both derived layers. Re-verified from canonical `orb_outcomes`/`daily_features` + `trading_app/lane_allocator.py` source.
**No OOS consumed.** All discovery queries are IS or pre-existing trailing windows already consumed by prior allocator runs.

---

## 1. EXECUTIVE VERDICT

### What is real
- Discovery surface for MNQ binary filters is genuinely exhausted on the existing feature set (VWAP + HTF prev-week + HTF prev-month all killed in past 24h at family level)
- `copies=2 → 5` scaling IS a real ~2.5× profit multiplier gated only on a proving-loop criterion
- 6 live lanes do diversify across 6 different sessions — true structural portfolio property
- Allocator's correlation-survivor + DD-budget + hysteresis machinery is sophisticated and mostly sound

### What is overstated (by my prior audit)
- **"Allocator is working, unlock = 0-1 lanes" was too quick.** It rested on `is_family_head` as a dedup flag and on "same-session implies correlated" — neither verified against canonical source. Re-verification shows:
  - `is_family_head` is NOT a binding allocator gate (some LIVE lanes have `is_family_head=False`); it's an `edge_families` labelling artefact
  - Same-session pairs are NOT automatically correlated (measured MNQ COMEX_SETTLE RR1.5 ORB_G5 vs RR1.0 OVNRNG_100 rho = 0.636 < 0.70 threshold — the gate would allow both)
- **"32 lanes correctly excluded" is wrong.** The real gate structure is: rank by `_effective_annual_r`, greedy select under rho<0.70 + DD budget. Many of the 32 are excluded by ANNUAL_R RANKING (low frequency × moderate ExpR losing to mid-ExpR × high frequency), not by correlation.

### What is blocked for valid reasons
- DD-budget binding (topstep $2,500 cap) — real constraint per scarcity audit
- Hysteresis 1.2× replacement threshold — real anti-churn discipline
- Rho > 0.70 correlation gate — real diversification logic
- Mode A OOS sacred — real methodology gate

### What is blocked for questionable reasons
- **Session-regime gate uses UNFILTERED data** (`lane_allocator.py:371-398` `_compute_session_regime`). Computes trailing 6-month mean R on `rr_target=1.0 confirm_bars=1 orb_minutes=5 NO FILTER` pooled across the session. A validated filtered lane (e.g., `X_MES_ATR60`) can have strong filtered-signal edge WHILE the unfiltered session shows a cold aggregate. The gate mis-excludes filter-conditioned edges on "cold" sessions. **This is the biggest blocker audit finding.**
- **6-month regime window is noise-prone** (`REGIME_WINDOW_MONTHS = 6`). Carver-canonical Sharpe estimation uses 10+ years. 6 months can flip HOT/COLD on a single adverse cluster.
- **`annual_r_estimate` ranking objective rewards frequency over per-trade edge.** Low-N high-ExpR validated lanes (CME_PRECLOSE X_MES_ATR60 at ExpR=+0.170 but ~60-80 trades/yr) lose to mid-ExpR high-frequency lanes. This is defensible but not obviously optimal.
- **2026-04-18 L6 swap from VWAP_MID_ALIGNED_O15 (BH K=36,000 validated, ExpR=0.21 all-IS, N=701) to ORB_G5_O15 (trailing_expr=0.079)** — the allocator's trailing-12mo window may have caught regime-transient noise on VWAP and swapped for a filter variant with inherently lower per-trade edge. **Potentially the single largest EV leak in the current book.**

### Biggest honest missed opportunity
**Audit the L6 VWAP→ORB_G5 swap.** If the swap is reversible (VWAP's trailing dip was noise), restoring L6 to VWAP could recover ~0.10-0.13 R/trade on ~100 trades/yr = **~10-13 R/yr per contract per copy**. At copies=2, that's 20-26 R/yr unlocked from fixing ONE allocator decision.

---

## 2. PROFIT / IMPROVEMENT MAP

| # | Opportunity | Type | Evidence | EV impact | Confidence | Verdict |
|---|---|---|---|---:|:---:|---|
| 1 | Restore L6 to VWAP_MID_ALIGNED_O15 (if trailing-12mo dip was noise, not regime shift) | implementation | VWAP BH K=36K validated; all-IS ExpR=0.21 N=701; trailing_expr=0.079 → 62% apparent drop in ≤12 months suggests noise OR regime shift; C12 review 2 weeks ago had trailing-30 at +0.47 (KEEP verdict) | +20-26 R/yr/copy (if noise) or +0 (if real shift) | MED | AUDIT REQUIRED |
| 2 | Scale `copies` 2→5 on topstep_50k_mnq_auto | implementation / capital | Profile annotation: `"scale to 5 after proving loop"`; current 2 copies yielding 398 R/yr aggregate | **+600 R/yr aggregate (2.5× unlock)** | HIGH | READY pending proving-loop gate clarification |
| 3 | Re-scope session-regime gate to be FILTERED (per-strategy), not unfiltered-pooled | blocker removal | `lane_allocator.py:371` uses unfiltered session pnl_r; validated filters can have positive edge on "cold" sessions; MNQ X_MES_ATR60 CME_PRECLOSE ExpR=+0.170 (N=596) currently shares the SAME "Session regime COLD" classification that paused MES lanes | +5-10 R/yr if CME_PRECLOSE unlocks; also protects future filtered lanes from mis-gating | MED-HIGH | CODE FIX |
| 4 | Activate `topstep_50k_mes_auto` profile (`active=False` currently) | implementation | Profile is defined at `prop_profiles.py:491-514` with 1 validated MES CME_PRECLOSE ORB_G8 lane; only gated by `active=False` flag | +8-12 R/yr (1 lane × ~$50 avg dollar-R × session frequency) | MED | DECISION — user-gated |
| 5 | Audit `_effective_annual_r` ranking objective | research correction | Current: annual_r × 3mo-decay-penalty. Alternative: Sharpe or per-trade ExpR. Rewards frequency over edge quality. Testable by re-running allocation with different objective, comparing forward equity curves in OOS-clean 2026 data | Unknown — could be 0 or significant | LOW | RESEARCH TASK |
| 6 | Extend regime window from 6mo to 12-24mo | blocker weakening | `REGIME_WINDOW_MONTHS=6`; Carver recommends 10+ yrs for Sharpe; shorter windows catch noise | +0-5 R/yr via reduced false-COLD gating; downside is slower response to real shifts | LOW-MED | RESEARCH TASK |
| 7 | Rho-aware multi-lane-per-session (allow 2 lanes if rho<0.70) | implementation | Allocator ALREADY supports this via correlation_matrix; measured COMEX_SETTLE RR1.5 ORB_G5 vs RR1.0 OVNRNG_100 rho=0.636 (N=23 overlap, noisy). Verify on all 38 with proper lane-level daily-pnl join | +5-15 R/yr if 2-3 diversified same-session pairs unlock; correlation-gate may already be doing this — verify | LOW | ALREADY IN CODE — verify actual pairs |
| 8 | Activate MGC profile (none currently defined) | implementation | Zero MGC validated lanes under Mode A (discovery never run for MGC instrument specifically); prior surface map ranked MGC discovery (F16) but deployability flagged `False` due to short real-micro horizon | +5-20 R/yr if MGC has alpha; 0 if not | LOW | DISCOVERY TASK (1 session) |
| 9 | Half-Kelly sizing on deployed lanes | portfolio | Referenced in prior memory as "valid unlock but gated by F-1 XFA dormant" per `portfolio_dedup_nogo.md`; not currently enabled; standard Kelly yields 10-15% dollar-R uplift at same DD | +40-60 R/yr-equivalent (dollar) | MED | DECISION — F-1 dependency |
| 10 | Audit 2026-04-18 allocation swap (rebalance_date) for all lanes, not just L6 | research correction | JSON rebalance date 2026-04-18; check diffs vs prior allocation; L6 is the obvious swap but others may have swapped too | 0-30 R/yr depending on findings | MED | AUDIT TASK |

**Total plausible unlock (probability-weighted): 50-150 R/yr aggregate at current copies, scaling to 125-375 R/yr at copies=5.**

---

## 3. BLOCKER AUDIT

| # | Blocker | Owner layer | Rationale | Evidence | Still supported? | Cost of keeping | Risk of removing | Verdict |
|---|---|---|---|---|:---:|---|---|:---:|
| B1 | Session-regime gate is UNFILTERED | `lane_allocator.py:371` `_compute_session_regime` | "Unfiltered session signal avoids cherry-picking filter-specific windows" | Code reads `WHERE rr_target=1.0 AND confirm_bars=1 AND orb_minutes=5` with NO filter join | **NO** | Mis-excludes filtered lanes on "cold" sessions; potential -5 to -15 R/yr | Regime signal becomes per-lane, slightly more noise, but matches the actual deployed signal | **WEAKEN** — add per-lane filtered regime check alongside unfiltered, deploy if EITHER is warm |
| B2 | `REGIME_WINDOW_MONTHS = 6` | `lane_allocator.py` constant | Fast response to real regime shifts | 6 months is short for stable estimation | **PARTIALLY** — fast-response has value but 6mo is noisy | False-COLD gating on noise clusters | Slower response to real shifts | **RE-SCOPE** — test 9-12mo window in side-by-side |
| B3 | `annual_r_estimate` ranking | `lane_allocator.py:490-507` `_effective_annual_r` | Maximizes total R per period (economically intuitive) | Correct for maximizing gross R; potentially suboptimal for risk-adjusted return | **PARTIALLY** | Sharpe-superior lanes with low frequency lose | Switching to Sharpe might underweight high-edge lanes | **RE-SCOPE** — compare forward equity curves with both objectives in a sandbox |
| B4 | `CORRELATION_REJECT_RHO = 0.70` | `lane_allocator.py:506` | Standard diversification threshold | Hard threshold; could be 0.60 or 0.80 | YES | May reject lanes with rho 0.65-0.70 that are still usefully diversifying | Looser threshold reduces diversification | **KEEP** but verify empirical rho distribution |
| B5 | DD budget `max_dd = $2,500` (topstep) | `trading_app/prop_tiers.py` `ACCOUNT_TIERS` | Prop firm rule — non-negotiable | Canonical firm constraint | **YES** | None | None (firm-imposed) | **KEEP** |
| B6 | Hysteresis 1.2× replacement | `lane_allocator.py` `build_allocation` | Anti-churn — don't thrash lanes on marginal improvements | Stability is real | **YES** | Slower lane rotation | Churn costs | **KEEP** |
| B7 | `topstep_50k_mnq_auto.copies = 2` | `prop_profiles.py:424` | "Scale to 5 after proving loop" — prove loop first | No operationalized "proving loop" criterion in code or docs | **UNVERIFIED** | At current 2 copies, 60% of potential aggregate R unrealized | Scaling before loop is proven = risk of DD breach | **UNVERIFIED** — user must define "proving loop" graduation criterion |
| B8 | `allowed_instruments=frozenset({"MNQ"})` for topstep profile | `prop_profiles.py:439` | Profile-specific MNQ focus | Design choice | **YES** — design choice | MES/MGC excluded from THIS profile | None (separate profiles exist) | **KEEP** (separate profile for MES) |
| B9 | `topstep_50k_mes_auto.active=False` | `prop_profiles.py:498` | "User activates when TopStep Express account is ready" | User-gated on account availability | **UNVERIFIED** | 1 validated MES lane sitting unused | Account risk + setup time | **DECISION — user** |
| B10 | `is_family_head` dedup flag | `validated_setups` schema | Presumed allocator dedup gate | My prior audit assumed this. **FALSE** — the flag doesn't bind the allocator; some LIVE lanes have `is_family_head=False` | **NO** — flag exists for `edge_families` table, not allocator | None if it's informational only | None | **RE-SCOPE my audit's interpretation** — it's not a deployment gate |

---

## 4. MISSED OPPORTUNITIES

### M1: L6 swap reversal audit
- **Missing:** check whether VWAP_MID_ALIGNED_O15's trailing-12mo dip (ExpR 0.079 vs all-IS 0.21) is regime shift or noise
- **Matters:** if noise, restoring L6 recovers ~10-13 R/yr/copy. Across 2 copies = 20-26 R/yr.
- **Why missed:** prior audit acknowledged the swap but labeled it "deferred follow-up." Re-classify as PRIMARY action.
- **Next test:** compare VWAP_MID_O15 vs ORB_G5_O15 per-year breakdown 2019-2025; check if VWAP has any year with mean_R < 0 (genuine regime breakdown) or if 2025 is a noise cluster. Per-year IS breakdown, NO OOS consumption.

### M2: Session-regime gate fix
- **Missing:** per-lane filtered regime check
- **Matters:** potentially unlocks CME_PRECLOSE slot + future-proofs for next filter/family validations
- **Why missed:** prior audit treated regime-COLD tag as authoritative; didn't re-read the gate's implementation
- **Next test:** patch `_compute_session_regime` to accept an optional filter, apply filter's `matches_df` before averaging pnl_r. Add test case. No OOS consumption.

### M3: Rho audit of all 15 ALLOCATOR_NOT_SELECTED lanes
- **Missing:** actual pairwise rho (using `_load_lane_daily_pnl`) between each excluded lane and the live 6
- **Matters:** if some pairs have rho < 0.70 AND acceptable DD budget, those are UNLOCKS hidden in the current allocator run
- **Why missed:** prior audit assumed same-session = correlated without measuring
- **Next test:** run `compute_pairwise_correlation([live 6] + [excluded 15])` and identify lanes with rho < 0.70 against all 6 lives. No OOS.

### M4: `copies` proving-loop criterion
- **Missing:** operational definition of "proving loop" graduation
- **Matters:** the 2.5× scale unlock is the single highest-EV action available, and it's gated on an undefined criterion
- **Why missed:** prior audit noted the gate but did not challenge its absence
- **Next test:** user declares acceptance criterion (N live trades, R/yr threshold, zero-breach window, etc.). Then verify against current live journal.

### M5: Carver continuous sizing / Half-Kelly
- **Missing:** continuous size scaling (forecast combiner) not enabled
- **Matters:** standard dollar-R uplift 10-15% at same DD
- **Why missed:** Phase D D-0 was killed on COMEX_SETTLE OVNRNG_100 (`2026-04-18-phase-d-d0-backtest.md`); extrapolation: "Carver is dead." But D-0 was one cell. Other lanes untested.
- **Next test:** Phase D D-1 pre-reg on a DIFFERENT lane (e.g., TOKYO_OPEN COST_LT12) with different signal (not rel_vol). Explicit differentiation from D-0 kill.

### M6: Allocator rebalance swap audit
- **Missing:** full diff of 2026-04-18 rebalance vs prior (2026-04-13 swap rebuild?)
- **Matters:** L6 is the obvious swap but other lanes may have churned too — need to know the full set of changes
- **Why missed:** prior audit focused on 38 vs 6, not on what changed between rebalances
- **Next test:** `git log docs/runtime/lane_allocation.json` + diff each commit's lane set

---

## 5. FALSE POSITIVES / FALSE NEGATIVES

### Called good too fast (possible false positives)
- **FP1:** "Allocator is working correctly, unlock = 0-1 lanes" (my prior audit). Real answer is closer to "allocator logic is sound but 2-4 specific decisions may be leaking EV." My prior audit used the derived `is_family_head` flag and `same-session = correlated` heuristic as proof without re-verifying from canonical layers. Institutional error.
- **FP2:** The 2026-04-18 rebalance swap L6 VWAP→ORB_G5 was accepted silently by the allocator + system. No alerting fired despite swapping away from a BH K=36K validated lane. If this is a false positive (VWAP still has positive edge), the allocator swapped for noise.

### Killed too fast (possible false negatives)
- **FN1:** VWAP family DOCTRINE-CLOSED per the 2026-04-18 comprehensive scan — but the SPECIFIC L6 cell IS the validated survivor; it was tautology-flagged in the scan against ITSELF. Killing the family should NOT kill the already-validated cell. Yet the allocator swapped L6 OUT on the same day. The scan's family-kill verdict may have been (incorrectly) interpreted as a validation-kill for L6.
- **FN2:** HTF prev-week and prev-month FAMILY KILLs (2026-04-18) are at family level. Do any individual HTF cells have standalone validation evidence? The kill report shows 0/14 pass BH_family — suggests no cell survived even on its own lane-BH. Likely a genuine family-dead result, not a false-kill. But worth checking per-cell BH_lane for completeness.
- **FN3:** MES CME_PRECLOSE ORB_G8 paused "Session regime COLD (-0.0917)". If the regime gate is unfiltered (per B1), and the FILTERED lane ExpR is +0.196 N=194 (from active_validated_setups), the pause may be based on a wrong signal. A filtered-regime check could release the pause.
- **FN4:** Phase D D-0 Carver sizing KILL on COMEX_SETTLE OVNRNG_100 was cell-specific (Sharpe uplift +7.33% < 10% threshold). Extrapolating to "Carver is dead" for all lanes is premature. Other lanes untested.

---

## 6. NEXT ACTIONS (ordered by truth importance × EV × speed)

### Immediate (this session or next)
1. **Audit L6 swap** (M1): per-year breakdown of VWAP_MID_O15 vs ORB_G5_O15 on MNQ US_DATA_1000 O15 RR1.5. No OOS. If VWAP has ≥5/7 years mean_R > 0 and 2025 is just 1 weak year, push for L6 swap-back to VWAP via manual allocation override. **Highest EV per hour of work. ~20 R/yr unlock if noise.**

2. **Rho audit** (M3): run `compute_pairwise_correlation` over [live 6 + 15 ALLOCATOR_NOT_SELECTED]. Identify any same-session pair with rho < 0.70 that was excluded by annual_r ranking despite correlation gate passing. Deploy if DD budget allows. **Medium EV, <1 hour.**

3. **Session-regime gate patch** (M2 / B1): 1-day code change. `_compute_session_regime(instrument, orb_label, filter_key=None)`. If `filter_key` provided, apply filter's `matches_df` before averaging. Add drift check. Re-run allocation; see if CME_PRECLOSE slot fills. **Medium-high EV, 1 day of work.**

### Near-term (this week)
4. **Operationalize the `copies=2→5` proving-loop gate** (M4 / B7). User defines acceptance criterion. Verify against live_journal. Scale when met. **Highest aggregate EV — 2.5× capital unlock.**

5. **Rebalance swap diff audit** (M6): full `git log docs/runtime/lane_allocation.json` → identify all swap decisions, not just L6. Audit each.

6. **Blockers B2 + B3 re-scope** (regime window, ranking objective): backtest alternative windows and objectives on IS data only. No OOS.

### Deferred (next week+)
7. **Phase D D-1 pre-reg** (M5): Carver sizing on a different lane with different mechanism. Distinct from D-0 COMEX_SETTLE kill.
8. **MES profile activation** (B9 / Opp 4): user-gated.
9. **MGC discovery run** (Opp 8): full discovery with Mode A holdout, as standalone pre-reg.

### Standing (continuous)
10. **Monitor CME_PRECLOSE session regime** for HOT turn (if B1 is not weakened). Automatic unlock of MNQ X_MES_ATR60 + MES paused lanes.

---

## Final adversarial stance

My prior audit concluded "allocator is working, 0-1 unlocks." This re-open produced:

- **3 real blocker-audit findings** (B1 unfiltered regime gate, B2 short regime window, B7 undefined proving-loop criterion)
- **1 plausible major EV leak** (L6 swap, 20-26 R/yr if noise)
- **1 audit-gap** (rho wasn't actually measured on excluded same-session pairs)
- **1 misinterpretation** (is_family_head is not an allocator gate)
- **6+ deferred opportunities** in sizing, MES profile, MGC discovery, regime-window tuning, ranking objective, rebalance-diff audit

None of these kill the allocator — it's still sophisticated and mostly sound. But "0-1 unlocks" understated the real landscape. Honest estimate: **50-150 R/yr recoverable across 3-6 actions**, scaling to **125-375 R/yr at copies=5**, with **the copies scale itself being the single largest unlock at 2.5× aggregate**.

Prior audit should be marked "superseded for conclusions on allocator correctness"; its data tables and exclusion reasoning remain valid but the verdict was too quick.
