# Deployment-throughput leverage — plan for next session

**Date:** 2026-05-12
**Author:** allocator-gate audit follow-up
**Status:** PLAN ONLY — no execution this session.
**Origin artifact:** `scripts/tools/allocator_gate_audit.py` + `docs/runtime/stages/allocator-gate-audit.md`. Re-run with `python scripts/tools/allocator_gate_audit.py --all-profiles --json out.json` to refresh the numbers below.

---

## Question this plan answers

**"How do I trade more of my validated strategies on the brokers?"** Validated shelf is 844 (active deployable). Currently 3 lanes deployed. The audit identifies WHERE the attrition happens and which gates are levers.

## Headline attrition (verified 2026-05-12, `topstep_50k_mnq_auto`, rebalance_date=today)

```
Gate 0  ENTRY                       844
Gate 1  E2_SAFETY                    19   (post-entry filter classes — genuine look-ahead)
Gate 3  REGIME_COLD                  45   (session regime gate, structural)
Gate 4  CHORDIA                     772   (834 of 844 strategies have NO strict-replay audit)
Gate 5  PROFILE_WHITELIST             0   (the 8 chordia-passers happen to be MNQ + allowed sessions)
Gate 6  NON_DEPLOYABLE                0
Gate 7-10 SELECTION_LOOP              5   (correlation gate prunes same-session siblings)
SELECTED                              3
Invariant: 19+45+772+0+0+5+3 = 844 OK
```

**Conservation invariant holds.** All 844 strategies are accounted for at exactly one gate.

## Three levers, ranked by deployable-count impact

### Lever 1 — Chordia audit coverage (BIGGEST)

- Audit log: `docs/runtime/chordia_audit_log.yaml` — **12 entries, all MNQ**.
  - 6 PASS_CHORDIA, 2 PASS_PROTOCOL_A (deploy-allowed) → 8 deployable
  - 1 PARK, 3 FAIL_BOTH → 4 blocked
- 832 strategies have `verdict = MISSING` (no audit ever run). Of those:
  - 13 MGC (12 LONDON_METALS, 1 CME_REOPEN) — **zero MGC audits exist**
  - 48 MES (35 CME_PRECLOSE, 13 other) — **zero MES audits exist**
  - 771 MNQ — only 12 of ~783 MNQ strategies have been audited (~1.5%)

**Action:** expand `chordia_audit_log.yaml`. Each new entry unlocks one strategy from Gate-4-MISSING. Empirical pass rate so far: 8/12 = 67% on MNQ.

**Sub-finding (CME_PRECLOSE warning):** even if you audit all 35 MES CME_PRECLOSE strategies, Gate 3 (REGIME_COLD) currently blocks that entire session (ExpR=-0.0997, COLD). MES would unlock only when the CME_PRECLOSE regime turns HOT. So **prioritize MNQ audits first**, then MGC LONDON_METALS audits second (different session, different regime). Do not invest in MES CME_PRECLOSE audits while the regime is COLD.

**Open question for next session:** what is the operational cost of one Chordia audit? Is there a batch-runner script? `research/chordia_revalidation_deployed_2026_05_01.py` runs Pathway-B K=1 per lane against canonical layers — it may be the template for a batch-audit script. If so, a one-shot script could process N candidates and append to `chordia_audit_log.yaml` in a single pre-registered run.

### Lever 2 — Profile activation for MGC and MES

- `ACCOUNT_PROFILES` has only one `active=True` profile right now: `topstep_50k_mnq_auto` (MNQ only).
- Inactive profiles exist (per memory `topstep_scaling_corrected_apr15.md`): apex_*, bulenox, mffu_rapid, multi-instrument variants. Each is gated by `p.active=False`.
- MGC validated strategies have ZERO deployment path until an MGC-instrument profile is `active=True`. Same for MES.

**Action when ready to scale beyond MNQ:**
1. Pick the next firm/account per the existing scaling memory (`topstep_scaling_corrected_apr15.md`).
2. Flip `active=True` for one MGC profile and one MES profile.
3. Re-run audit to verify: Gate 5 should now allow MGC/MES through the whitelist.

**Compounding constraint with Lever 1:** without MGC/MES Chordia audits, activating a profile yields 0 new lanes — the lanes hit Gate 4 instead of Gate 5. **Lever 1 is prerequisite to Lever 2** for MGC/MES.

### Lever 3 — Fresh `rebalance_lanes --all-profiles` run

- Audit cross-check on `docs/runtime/lane_allocation.json`:
  - Committed: 3 RR1.5 variants (COMEX_SETTLE OVNRNG_100, US_DATA_1000 VWAP, NYSE_OPEN COST_LT12)
  - Fresh audit selects: the 3 RR1.0 variants of the same (instrument, session, filter) triplets
  - **All 3 currently-deployed lanes have `sr_status=ALARM`** and are halved by `SR_ALARM_DISCOUNT=0.50`
  - Their RR1.0 siblings are un-alarmed, rank #1 / #3 / #4
- Per memory `live_lanes_2026_05_03_three_alarm_one_pause.md`: the 2026-05-20 tripwire was the intended switchover.
- The audit shows the switch would happen *today* if `rebalance_lanes --all-profiles` were rerun.

**Action:** decide whether to honor the 2026-05-20 tripwire (paper-trade builds N≥50 first) or flip earlier. Either way, a fresh rebalance produces no NEW lanes — it just swaps RR1.5 for RR1.0. **No throughput gain from this lever**, only correctness.

## What is NOT a lever (do not pull)

- **Loosen Gate 3 (REGIME_COLD)** — gate has a backtest receipt (+630R vs -799R for the per-strategy-pause alternative). Hard-coded in `_classify_status`.
- **Loosen Gate 1 (E2_SAFETY)** — 19 strategies use post-entry filter shapes (`PD_CLEAR_*` and similar). Genuine look-ahead. See `trading_app/config.py` E2_EXCLUDED_FILTER_PREFIXES.
- **Raise `max_slots`** — already 7 for `topstep_50k_mnq_auto`; binding constraint is correlation, not slots.
- **Lower `RHO_REJECT_THRESHOLD = 0.70`** — would let same-session siblings co-deploy, which is correlation-stacking the same trade.
- **"Fix" Gate 4 (CHORDIA)** — the gate is correctly fail-closed (`feedback_chordia_missing_is_not_backlog.md`). The fix is to RUN audits, not to bypass them.

## Cross-finding gaps and minor issues surfaced by the audit

These are not the throughput question itself but came out of the sweep:

1. **Performance:** `compute_lane_scores` calls `_compute_session_regime` once per strategy. Regime is shared across all strategies in the same (instrument, session). Caching `(instrument, session) → regime_expr` would cut the strategy_discovery cross_atr injection log from ~30× to 1× per session and likely 5-10× the rebalance wall-clock. Tier-3 perf observation; not a correctness bug.

2. **Duckdb segfault on interpreter teardown (Windows):** intermittent SIGSEGV after main() returns; exit code 139 despite successful JSON write. Pattern is known. If this hits CI, add explicit `con.close()` everywhere + `gc.collect()` + `os._exit(0)` at end of main. Not worth fixing for an interactive diagnostic.

3. **`sr_distribution` shows 841 UNKNOWN, 3 ALARM** — `load_sr_state` returns empty for any strategy not in `sr_state.json`. Either the SR monitor is only running on the 3 deployed lanes (likely intentional — cheap to compute only what's live) OR this is silent miss-coverage. **Verify next session:** is `sr_state.json` meant to cover ALL 844 strategies or only deployed lanes? Memory entry `feedback_sr_monitor_peak_vs_current_misread.md` says `sr_state.json` is the monitor's *state*, so coverage = monitored lanes only. If we want SR liveness on the 8 Chordia-passers to inform pre-deployment ranking, the SR monitor would need to expand its watch set.

4. **`lane_allocation.json` operational staleness** — file's `profile_id` is `topstep_50k_mnq_auto` but committed strategy_ids are stale (all 3 lanes flagged ALARM by the SR monitor, but only 1 is paused). The audit cross-check makes this visible per-rebalance; consider auto-running the audit in `rebalance_lanes.py` as a verify step before writing the JSON.

5. **MGC sessions:** 12 of 13 MGC validated are LONDON_METALS. The LONDON_METALS regime can be checked independently — if it's HOT, MGC has a real deployment path the moment Chordia audits exist + a profile is active. Next session: query session regime for LONDON_METALS to confirm.

## Recommended next-session sequence

1. **Verify LONDON_METALS regime** is HOT (single SQL, < 1 min).
2. **Decide Chordia audit batch size and process** — review `research/chordia_revalidation_deployed_2026_05_01.py` as a batch-template; if viable, run a pre-registered batch of ~20-50 MNQ candidates ranked by `annual_r_estimate`.
3. **Decide on the 2026-05-20 tripwire vs early flip** for the 3 SR-alarmed lanes.
4. **(Conditional on #1 + #2):** if MGC LONDON_METALS regime is HOT and MGC audits are runnable, batch-audit the 13 MGC strategies and activate one MGC profile.
5. **Defer** any allocator code changes. The allocator is functioning correctly given its inputs.

## Files to keep / move forward

- `scripts/tools/allocator_gate_audit.py` — re-runnable diagnostic, zero production deps, exit-0 verified.
- `docs/runtime/stages/allocator-gate-audit.md` — original stage doc, can be removed once stage is closed.
- `docs/plans/2026-05-12-deployment-throughput-leverage.md` — this file. The decision record for the question.

## Acceptance for "is this plan complete"

A future session can answer "how do I trade more strategies on the brokers" by:
1. Re-running `allocator_gate_audit.py --all-profiles --json out.json`.
2. Reading this plan and the JSON together.
3. Picking one of the three levers, with the trade-offs already documented.

No new investigation needed before action.
