# Phase 3.1 — SR Monitor Stream Source Audit + Path Walk

**Generated:** 2026-04-19
**Scope:** 4 SR-ALARMED strategies on `topstep_50k_mnq_auto` as of C11/C12 refresh today
**Canonical sources:** `orb_outcomes` (stream), `validated_setups.expectancy_r` (stored baseline), Phase 2.1 Mode A output (`docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`) for Mode A baseline
**SR params:** `TARGET_ARL_DAYS=60`, `DEFAULT_DELTA=-1.0σ`, `DEFAULT_VARIANCE_RATIO=1.0`, `threshold=31.96`

## Question

Original audit (the MAX-EV extraction pass earlier today) framed 2 of 4 SR-alarmed strategies as "likely false blockers" based on aggregate gap analysis (stored baseline vs canonical OOS ExpR). The audit appendix (D.3 branch-point) correctly flagged this as insufficient — SR is path-dependent, not aggregate-gap-dependent. Additionally, audit's Phase 2.1 Mode A refresh revealed ALL 38 active lanes have Mode-B-inflated stored baselines. Question:

**If we substitute the Mode A baseline for the Mode-B-contaminated stored baseline, do the SR alarms still fire?**

## Stream source confirmed

`trading_app/sr_monitor.py` flow for the 4 alarmed (no paper_trades in DB):
1. `paper_trades` empty → branch 3 (`_load_canonical_forward_trades`)
2. Stream = `_load_strategy_outcomes(..., start_date=HOLDOUT_SACRED_FROM)` → canonical `orb_outcomes` ≥ `2026-01-01`, filtered via `strategy_fitness._load_strategy_outcomes` (which delegates to canonical filter logic)
3. Baseline `mu0` = `validated_setups.expectancy_r` via `_load_reference_stats()` → `deployable_validated_relation`
4. `sigma` = `_compute_std_r(win_rate, rr_target, mu0)` from stored stats

**Stream source: canonical `orb_outcomes` Mode A OOS (good).**
**Baseline source: stored `expectancy_r` (Mode-B contaminated per Phase 2.1 → systematically inflated for 9 lanes, material drift for all 38).**

## Path-walk results — decisive

Re-ran SR monitor on the canonical stream with **stored (Mode-B) baseline** vs **Mode A baseline** from Phase 2.1:

| Strategy | N | baseline | mu0 | σ | SR@alarm | alarm@ trade | Verdict |
|---|---:|---|---:|---:|---:|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 66 | stored_ModeB | 0.1119 | 1.20 | 39.10 | 27 | ALARM |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 66 | **mode_A** | **0.0922** | 1.20 | **35.83** | **27** | **STILL ALARM** |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 71 | stored_ModeB | 0.0870 | 1.04 | 0.82 | — | CONTINUE |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 71 | **mode_A** | **0.0694** | 1.03 | **0.78** | **—** | **CONTINUE (never alarmed in my reconstruction)** |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 54 | stored_ModeB | 0.1087 | 1.20 | 37.16 | 15 | ALARM |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 54 | **mode_A** | **0.2046** | 1.23 | **48.84** | **15** | **STILL ALARM — MORE alarmed** |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 70 | stored_ModeB | 0.1293 | 1.21 | 37.72 | 31 | ALARM |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 70 | **mode_A** | **0.1036** | 1.20 | **34.33** | **31** | **STILL ALARM** |

## Findings

### F-1. Mode A refresh does NOT auto-resolve any of the 4 alarms.
Direct contradiction of my original audit's "2 false blocker" frame. The hypothesis that Mode-B baseline inflation explained the alarms is FALSIFIED by path-walk on 3 of 4 strategies. Mode A substitution keeps all alarm verdicts identical for COMEX_SETTLE, SINGAPORE_OPEN, TOKYO_OPEN.

### F-2. SINGAPORE_OPEN Mode A ExpR is HIGHER than stored (0.205 vs 0.109).
Opposite direction from the general drift pattern. The Mode A correction makes the SR alarm MORE justified, not less — the stream mean (OOS) is now `~+0.06` vs a higher baseline `0.205`, a larger gap. My original gap-analysis finding "mild decay" was based on the wrong (Mode-B) baseline. Corrected verdict: **severe decay**.

### F-3. NYSE_OPEN COST_LT12 RR1.0 does NOT alarm in my reconstruction under either baseline.
SR final statistic 0.82 (stored) / 0.78 (Mode A) vs threshold 31.96 — never alarmed in path-walk on 71 canonical OOS trades. But C11/C12 refresh TODAY reported it as blocked. **Discrepancy needs investigation.** Candidate causes:
- The `deployable_validated_relation` used by `_load_reference_stats` may exclude some deployable rows that `_load_strategy_outcomes` still returns
- Monitor state may be persisted from an earlier snapshot with different filter semantics
- Stream subset differs: my reconstruction uses `_load_strategy_outcomes(filter_type='COST_LT12')`; monitor may apply additional direction/entry constraints

**Open sub-issue:** why is this strategy blocked if path-walk says CONTINUE?

### F-4. Alarm-trade positions (COMEX=27, SGP=15, TOK=31) reflect drawdown-streak timing.
Path-walk identifies where in the OOS stream the adverse run accumulated past threshold. SGP alarmed earliest (trade 15 of 54) — most severe. COMEX and TOKYO alarmed mid-stream (27/66, 31/70). These are real path dynamics, not aggregate drift.

### F-5. Aggregate-gap analysis is insufficient proxy for SR verdict.
Confirmed: my original "OOS > stored → false alarm" reasoning on NYSE_OPEN and TOKYO_OPEN failed on TOKYO (aggregate +0.024 gap but still alarmed in path-walk because adverse run clustering). SR needs path-walk, not gap.

## Corrections to record

- **Original audit claim "2 false SR blockers" is WRONG.** 0 of 4 are false by path-walk; 1 of 4 (NYSE_OPEN) has an unreconciled state discrepancy.
- **SINGAPORE_OPEN "mild decay" claim is WRONG.** Corrected Mode A baseline shows severe decay.
- **D.3 branch-point claim in the plan appendix is VALIDATED.** Path-walk was the right test; aggregate gap was not.

## Implications for campaign plan

1. **Phase 3.2 rationale survives but for different reason.** Originally framed as "allocator should use Mode A baseline to avoid false SR alarms." Now: Mode A baseline doesn't change alarm verdicts (alarms are path-correct). BUT allocator should still use Mode A baseline for EV-based lane selection — Phase 2.1 showed 38/38 stored values differ from Mode A, average N-compression −45%, mean ΔExpR varies ±0.06. Capital allocation against inflated baselines over-allocates systematically. Phase 3.2 still warranted, just not for SR-alarm-resolution.

2. **The 4 alarmed strategies should remain blocked.** C11/C12 refresh today correctly surfaced them. No reactivation recommendation.

3. **NYSE_OPEN COST_LT12 RR1.0 needs reconciliation.** My path-walk says CONTINUE; current monitor state says ALARM. Possible state-file staleness, possible stream-construction divergence. Sub-audit required.

## Open items

- **O-1. Run full `sr_monitor.run_monitor(apply_pauses=False)`** to see the LIVE SR statistic values for the 4 strategies, to compare against my reconstruction. The earlier C12 refresh output only showed aggregate verdicts.
- **O-2. Compare `_load_strategy_outcomes` stream against `orb_outcomes` direct query** for NYSE_OPEN COST_LT12 to find divergence source.
- **O-3. Check `deployable_validated_relation` filter definition** — does it exclude any of the 4 alarmed strategies by status?

## Methodology notes

- **No live data touched.** All computation is re-construction of the monitor's logic against canonical DB reads (read_only=True).
- **Filter delegation via strategy_fitness._load_strategy_outcomes** — confirmed canonical per `.claude/rules/research-truth-protocol.md` § Canonical filter delegation.
- **No MEMORY cited as truth** — Mode A baseline values come from today's Phase 2.1 output file (commit `122af101` on `research/campaign-2026-04-19-phase-2`), which is canonical-integrity verified (reads `validated_setups.filter_type`, delegates to `filter_signal`, respects `HOLDOUT_SACRED_FROM`).
- **Holdout policy: 2026-01-01 sacred boundary respected.** Stream window `trading_day >= 2026-01-01` per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.

---

**Verdict:** Phase 3.1 reaches a stronger conclusion than the original audit: the 4 SR alarms are ~path-correct even after Mode-B baseline contamination is corrected. Only NYSE_OPEN COST_LT12 has a stream/state discrepancy worth investigating. Campaign plan Phase 3.2 rationale requires re-framing (from "false-alarm fix" to "EV-allocation correctness"), not cancellation.
