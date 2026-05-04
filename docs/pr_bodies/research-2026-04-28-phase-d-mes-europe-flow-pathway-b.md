## Scope

Phase D D1 of `docs/plans/2026-04-28-edge-extraction-phased-plan.md`. Pre-registered Pathway B K=1 confirmatory test for **B-MES-EUR** (highest-EV PATHWAY_B_ELIGIBLE candidate from Phase B). **Read-only research; zero capital path touched.** No write to `validated_setups`, `lane_allocation`, `live_config`, `prop_profiles`, or any deployed-trading file.

## Verdict

**PARK_PENDING_OOS_POWER** — NOT CANDIDATE_READY, NOT deployable, NOT a promotion candidate.

Cell parks until `N_OOS_on >= 50` (estimate Q3-2026 at current trade rates). Pre-reg locked → no post-hoc rescue when OOS accrues.

## Claims (Mode A IS, locked schema)

| Gate | Computed | Verdict |
|---|---|---|
| C5 DSR_PB (K=1, theory-cited) | 0.9845 | PASS (>= 0.95) |
| C7 N_IS_on | 186 | PASS (>= 100) |
| C9 era stability | worst era +0.031 (2025) | PASS (no era < -0.05 with N>=20) |
| Sharpe_ann_IS | +0.8874 | PASS (Amendment 3.0 c2b positive direction) |
| KILL_RAWP | welch_p = 0.000602 | PASS |
| KILL_T | \|t\| = 3.4698 | PASS |
| KILL_BASELINE_SANITY | \|delta - 0.2459\| = 0.000048 | PASS |
| C6 WFE | N_OOS<50 | GATE_INACTIVE_LOWPOWER |
| C8 dir-match | OOS power = 0.106 | GATE_INACTIVE_LOWPOWER |

Block-bootstrap p (block=5, B=10000) = **0.000500** — independently corroborates parametric Welch p. Δ_IS = +0.2459 reproduces Phase B exact.

## Files changed (incremental scope only — vs `research/mnq-unfiltered-high-rr-family`)

- `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml` — pre-reg (locked schema, theory citation, kill criteria numeric)
- `docs/audit/results/2026-04-28-mes-europe-flow-pathway-b-v1-result.md` — verdict
- `research/phase_d_d1_mes_europe_flow_pathway_b.py` — runner (imports `phase_b_candidate_evidence_v1` for canonical math; no re-encoding)
- `HANDOFF.md`

## Evidence

```
$ python research/phase_d_d1_mes_europe_flow_pathway_b.py | tail -3
Pressure test (RULE 13): PASS
VERDICT: PARK_PENDING_OOS_POWER
All non-conditional gates PASS (C5 DSR_PB=0.9845, C7 N=186, C9 era stable, Sharpe_ann_IS=+0.8874).
C6/C8 are GATE_INACTIVE_LOWPOWER because N_OOS_on=9 < 50 power floor (Amendment 3.2).

$ python pipeline/check_drift.py | tail -3
NO DRIFT DETECTED: 114 checks passed [OK], 0 skipped (DB unavailable), 8 advisory

claim_hygiene: PASS
Pre-commit gauntlet: 8/8 PASS on every commit.
```

Independent SQL verification (committed in pre-reg under `baseline.independent_sql_baseline`):
delta_IS = +0.2459 exact match. 8-trade discrepancy with Phase B's 186 reconciled to `> 80` strict cutoff vs `>= 80` (resolved in pre-reg lock).

## Disconfirming Checks

- **KILL_BASELINE_SANITY** guards against runner drift vs locked Phase B numbers (|reproduced - 0.2459| > 0.001 → KILL).
- **RULE 13 pressure test** in runner: feature locked to safe-feature set; banned predictors structurally rejected before any computation.
- **Era stability**: 7/7 IS years positive, worst year 2025 = +0.0310 (well above -0.05 floor).
- **OOS power = 0.106** — explicitly UNVERIFIED, NOT deployable. Per `memory/feedback_oos_power_floor.md`: UNVERIFIED ≠ DEAD.
- **No data snooping**: this PR's pre-reg locks the test schema; any future re-run on accrued OOS uses the same kill criteria — no post-hoc threshold relaxation permitted.
- **No look-ahead**: feature `overnight_range_pct` is § 6.1 safe (RULE 1.2 valid domain — overnight_* features valid for ORB sessions starting >= 17:00 BNE; EUROPE_FLOW starts 18:00). Direction segmentation via `orb_EUROPE_FLOW_break_dir` is post-entry per RULE 6.3 exception.

## What this PR does NOT do (per pre-reg `execution_gate.forbidden_now`)

- No write to `validated_setups`, `edge_families`, `lane_allocation`, `live_config`, `prop_profiles`
- No paper trade simulation
- No CPCV (deferred to future amendment if power floor remains active in Q3-2026)
- **No capital deployment** — Phase E + capital-review skill + explicit user GO required before any allocator change

## Grounding

- `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (Pathway B K=1) + Amendment 3.2 (OOS power floor)
- `.claude/rules/backtesting-methodology.md` Rules 1, 3, 4, 6, 9, 10, 12, 13, 16
- `.claude/rules/research-truth-protocol.md` (canonical layers only — `bars_1m`, `daily_features`, `orb_outcomes`)
- `memory/feedback_oos_power_floor.md` (UNVERIFIED ≠ DEAD; never kill on underpowered OOS)
- Theory citations:
  - `docs/institutional/literature/chan_2013_ch7_intraday_momentum.md`
  - `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`
  - Filter mechanism is treated as within-class refinement on the Fitschen ORB strategy class (Aronson Ch11 extract not yet present in `docs/institutional/literature/`; not fabricated)

## Stacked-base note (if reviewer sees this)

This PR's base `research/mnq-unfiltered-high-rr-family` carries 5 prior-session commits ahead of `origin/main` (Phase A + Phase B + scratch-eod-mtm Stages 5/5b/8). If the reviewer sees those in the diff, that's because the base has not yet merged. Two options:

- **STACK**: merge `research/mnq-unfiltered-high-rr-family` first; this PR's diff then collapses to 4 files.
- **RETARGET to main**: this PR's effective scope grows to include the 5 base commits; review them as well.

The pre-reg's K=1 + theory citation framing depends on Phase B's evidence committed in the base, so the lineage is required either way.
