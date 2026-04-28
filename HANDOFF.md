# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session (2026-04-29 — PARKed Pathway-B Additivity Triage)

### What landed this session

**Additivity audit for D2 + D4 PARK candidates** (research/2026-04-29-parked-pathway-b-additivity-audit.py)
- Result: `docs/audit/results/2026-04-29-parked-pathway-b-additivity-triage.md`
- Verdict matrix:
  - **D2 B-MES-EUR** → **PASS_ADDITIVITY** (worst_rho +0.32, N=387). Optionality preserved; Phase E admission still gated on OOS power floor (~Q3-2026).
  - **D4 B-MNQ-COX** → **FAIL_ADDITIVITY** (worst_rho **+0.81** vs deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`, subset 100%, N=407). Confirms HANDOFF flag (+0.7733) and exceeds canonical `RHO_REJECT_THRESHOLD = 0.70`. **D4 cannot deploy as additive even if OOS power clears.** Two recovery paths: (1) re-spec as a same-cell *replacement* of the deployed lane (head-to-head pre-reg required); (2) park indefinitely until the deployed COMEX_SETTLE ORB_G5 anchor decays out.
- Brute-force OOS accrual on D4's existing predicate is wasted under the current canonical threshold.

**Production-code change (config.py only):** Two `_HYPOTHESIS_SCOPED_FILTERS` registrations + `strict_gt: bool = False` opt-in field added to `OvernightRangeFilter` and `GARCHForecastVolPctFilter`.
- `OVNRNG_PCT_GT80` (`overnight_range_pct > 80` strict) — for D2 audit.
- `GARCH_VOL_PCT_GT70` (`garch_forecast_vol_pct > 70` strict, direction="high") — for D4 audit.
- Both filters NOT in `BASE_GRID_FILTERS`, NOT routed in `get_filters_for_grid()` for any (instrument, session). Reachable only via canonical `ALL_FILTERS` lookup or Phase-4 hypothesis-injection. Verified by 36-cell test sweep in `TestStrictGtVariants`.
- Default `strict_gt=False` preserves existing `>=` semantics on all 5 prior `OvernightRangeFilter`/`GARCHForecastVolPctFilter`-shaped instances (regression-tested).

### Decision gate

D2 stays in PARK queue for OOS accrual (additivity-clean).
D4 is now **dual-gated**: OOS power floor + RULE 7 additivity. Existing predicate cannot pass both. Decision moves from "wait for N" to "respec or park". Recommend respec as replacement-lane Pathway-B if user wants D4 alive; otherwise close.

D-0 v2 (PARK_ABSOLUTE_FLOOR_FAIL) excluded from this triage — different failure class. B-MNQ-EUR does not exist as pre-reg (verified 2026-04-29).

### Verification

- `tests/test_trading_app/test_config.py`: 203/203 pass (186 prior + 17 new strict-gt regression tests).
- `pipeline/check_drift.py`: 20 pre-existing violations from local `certifi`/`annotated_types`/`click` env gaps; verified unchanged pre/post edit via `git stash` baseline. No new violations introduced.
- Audit runner exit 0 on canonical `gold.db`. D2 N=387, D4 N=407 — both above noise floor.
- Read-only against canonical layers; no writes to `validated_setups`, `lane_allocation.json`, or live config.

---

## Prior Session (2026-04-28 — Phase D D4 B-MNQ-COX Pathway B K=1 — PARK_PENDING_OOS_POWER)

### What landed this session

**Phase D D4 — B-MNQ-COX Pathway B K=1** (`research/2026-04-28-phase-d-mnq-comex-settle-pathway-b`,
commits `f7e2c921` pre-reg + `733883ed` runner + this verdict commit)
- Pre-reg: `docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml`
- Runner: `research/phase_d_d4_mnq_comex_settle_pathway_b.py`
- Result: `docs/audit/results/2026-04-28-mnq-comex-settle-pathway-b-v1-result.md`
- **Verdict: PARK_PENDING_OOS_POWER** — all 7 KILL criteria PASS, all non-conditional
  C5/C7/C9/Sharpe gates PASS, but C6/C8 are GATE_INACTIVE_LOWPOWER (N_OOS_on=17,
  power=0.106). Per Amendment 3.2: UNVERIFIED ≠ KILL. Cell parks until N_OOS_on ≥ 50
  (~Q3-2026).
- Numbers: N_IS_on=199, ExpR_IS_on=+0.2453, ΔIS=+0.2286, Welch t=3.18, p=0.00161,
  bootstrap p=0.00190, Sharpe_ann_IS=+1.692, DSR_PB=0.9998, dir_match=True
  (ΔOOS=+0.2538), 6/6 years positive IS_on.
- Pre-reg locked → no post-hoc threshold rescue when OOS accrues.
- **RULE 7 flag preserved:** lane-correlation +0.7733 vs deployed COMEX_SETTLE
  ORB_G5. CANDIDATE_READY/PARK verdict is necessary, not sufficient, for Phase E
  admission. Portfolio additivity audit required before any capital deployment.

### Decision gate

D4 (B-MNQ-COX) → PARK_PENDING_OOS_POWER. Real-money exposure remains unchanged
because (1) OOS is underpowered for C6/C8 verification, AND (2) RULE 7 portfolio
overlap with deployed COMEX_SETTLE ORB_G5 is +0.773 — additivity unproven.

D2 (B-MES-LON) and D3 (B-MNQ-NYC) still pending user GO. Each candidate carries
the same likely outcome (PARK_PENDING_OOS_POWER given identical N_OOS power-floor
situation).

### Verification

- Pre-commit gauntlet (8/8) PASSED on both pre-reg and runner commits.
- Independent SQL reproduction of Phase B numbers confirmed exact match before
  pre-reg commit (delta_IS=0.2286 to 4dp).
- Bootstrap p computed on B=10000, block=5 (Phipson-Smyth correction applied).
- KILL_BASELINE_SANITY PASSED: |reproduced - expected| = 0.000014 R.

---

## Prior Session (2026-04-28 — Phase D D-0 v2 clean re-derivation — PARK_ABSOLUTE_FLOOR_FAIL)

### What landed this session

**Phase D D-0 v2 clean re-derivation (MNQ COMEX_SETTLE E2 CB1 OVNRNG_100 RR1.5):**
- Pre-reg: `docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` (SHA `823b0127`, PR #170 merged)
- Runner: `research/phase_d_d0_v2_backtest.py` (new, scratch-policy: realized-eod, garch predictor, bootstrap p)
- Result: `docs/audit/results/2026-04-28-phase-d-d0-v2-garch-backtest.md`
- **Verdict: PARK_ABSOLUTE_FLOOR_FAIL**
  - Relative Sharpe uplift: **+16.48%** ≥ 15% threshold → PASS
  - Absolute Sharpe diff: **+0.0283** < 0.05 floor → FAIL
  - Bootstrap p: **0.50** > 0.05 → FAIL (underpowered)
  - H1 requires ALL THREE gates; 2/3 fail → PARK, not KILL
- Action-queue item `phase_d_d0_clean_rederivation` closed with override note: gate eval deferred to 2026-05-15 or later; do NOT promote tainted D-0 v1 or D-0 v2 PARK to deployment.
- `phase_d_d0_backtest.py` remains TAINTED (do not modify); v2 script lives alongside.

### Decision gate

D-0 v2 is PARKED — clean predictor does not rescue the regime-amplification thesis with statistical confidence at IS N. **2026-05-15 gate eval has no clean D-0 v2 PASS basis.** Daily shadow accumulation should remain parked.

D2/D3/D4 pre-regs remain pending user GO (see prior session below).

### Verification

- PR #170 merged, 5029 tests passed, 28 skipped (CI SUCCESS).
- `python pipeline/check_drift.py`: run on prior session; no pipeline/ or trading_app/ files touched this session.

---

## Cross-session loss notice (2026-04-28 — crg-calibration session)

**Phase D YAML uncommitted WIP lost.** During calibration session on PR #171 (`crg-calibration` branch), I stashed `docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` (single-char ` M` modification on top of committed `2c8e52ab9889`) with label `phase-d-yaml-WIP-from-other-session-preserved-for-crg-calibration` to switch branches. Stash later disappeared from `git stash list` and from `git reflog stash` — no command in my session removed it. Suspected cause: parallel-session interference or background hook reaping. **Committed version on `prereg-phase-d-d0-v2-garch` is intact (519 lines, blob `2c8e52ab9889`, branch tip `823b0127`).** If the WIP edits were load-bearing, fsck unreachable objects (~5,600) may still contain the blob for ~30 days — search by yaml content. Otherwise re-edit from committed baseline. (Note: PR #170 D-0 v2 work has since landed on main, so the lost WIP may have been superseded by that work.)

---

## Prior Session (2026-04-28 — doctrine fix + Phase D dispatch — branch `triage-e2-lookahead-9-candidates`)

### What landed this session

**Root-cause doctrine fix for the 2026-04-21 postmortem § 5.1 retro-audit:**
- `.claude/rules/backtesting-methodology.md § 6.1`: removed `rel_vol_{s}` from safe list (the description contradicted `pipeline/build_daily_features.py:1600-1660` which computes numerator as `break_bar_volume`).
- `.claude/rules/backtesting-methodology.md § 6.3`: added `rel_vol_{s}` to E2 banned list with canonical-source cite + Chan Ch 1 p.4 cite.
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md § 9`: closure note for 2026-04-28 follow-up. Documents two found drifts: rel_vol § 6.1 wording, and `late-fill-only` annotation as statistically unsafe for signal discovery (Chan p.4 selection bias).
- `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md`: added "Re-derivation dispatch" section. Priority 1 = Phase D D-0 clean re-derivation with HARD DEADLINE 2026-05-15 (predictor swap: `garch_forecast_vol_pct` first, `atr_20_pct` second). Priority 2 = historical predictor-tainted scripts deferred unless downstream citation appears.
- Memory: `phase_d_daily_runbook.md` flagged D-0 locks as research-provisional pending clean re-derivation.
- Stage file: `docs/runtime/stages/triage-e2-lookahead-9-candidates.md`.

### Decision gate

Phase D D2/D3/D4 still pending user GO. **Phase D D-0 clean re-derivation must land before 2026-05-15** to avoid the gate eval consuming contaminated baseline. Recipe is in registry dispatch § Priority 1 — pre-reg amendment + predictor swap, K=1 framing preserved.

### Verification

- `python pipeline/check_drift.py`: 114 passed, 0 skipped, 10 advisory (0 violations on check 124).
- `pytest tests/test_pipeline/test_check_drift_e2_lookahead.py`: 10/10 pass.
- No `pipeline/`, `trading_app/`, schema, or canonical-config files touched.

---

## Prior Session (2026-04-28 — drift check 124: 9 e2-lookahead-policy annotations)

### What landed this session

**Drift check 124 clearance** — 9 scripts surfaced by `check_e2_lookahead_research_contamination()` annotated:
- TAINTED (7): `phase_d_d0_backtest.py`, `break_delay_filtered.py`, `break_delay_nuggets.py`, `l1_europe_flow_pre_break_context_scan.py`, `mnq_comex_unfiltered_overlay_v1.py`, `mnq_l1_europe_flow_prebreak_context_v1.py`, `shadow_htf_mes_europe_flow_long_skip.py` (reclassified after Opus audit)
- CLEARED (1): `audit_sizing_substrate_diagnostic.py` (implements the gate itself)
- NOT-PREDICTOR (1): `output/confluence_program/phase0_run.py`
- Registry rows 19-27 added; row 27 reclassified `not-predictor → tainted` after real-data audit on MES EUROPE_FLOW O15 E2 IS (N=1719) showed 42.6% trades have `entry_ts < break_ts` → `break_dir='long'` selector is post-entry on those rows
- Drift check 124 now passes (0 remaining unannotated)
- Pre-existing test failure `test_pulse_integration::test_text_output_is_scannable` (61>60 lines) confirmed pre-existing

### Decision gate

Phase D D2/D3/D4 still pending user GO (HANDOFF below).
`phase_d_d0_backtest.py` is TAINTED — D-0 pre-reg result needs clean re-derivation before D-0 claim can be cited (high-EV next step: re-derive on `garch_forecast_vol_pct` per Carver Ch 9-10).
`shadow_htf` ledger continues recording (zero-capital observational); re-pre-register required before any deployment use.

---

## Prior Session (2026-04-28 — Phase D D1 + cleanup + hardening)

- **Tool:** Claude Code (autonomous)
- **Date:** 2026-04-28
- **Pickup point:** Phase D D1 verdict landed (B-MES-EUR PARK_PENDING_OOS_POWER); D2/D3/D4 pending user GO

### What landed this session

1. **Stage hygiene cleanup** (`chore/close-landed-stages-2026-04-28`, commits `55c458d7` + `2a474ea7`)
   - Closed 2 stage files for already-merged PRs #158 + #152
   - Added drift check 121 `check_stage_file_landed_drift` (advisory) — surfaces stages
     where `updated:` is >7d old AND ≥3 commits reference the slug, exactly the class
     of bug that caused this session's "thought we sorted this already" confusion.

2. **Phase D D1 — B-MES-EUR Pathway B K=1** (`research/2026-04-28-phase-d-mes-europe-flow-pathway-b`,
   commits `d58e5ce2` + the verdict commit)
   - Pre-reg: `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml`
   - Runner: `research/phase_d_d1_mes_europe_flow_pathway_b.py`
   - Result: `docs/audit/results/2026-04-28-mes-europe-flow-pathway-b-v1-result.md`
   - **Verdict: PARK_PENDING_OOS_POWER** — all 7 KILL criteria PASS, all non-conditional
     C5/C7/C9/Sharpe gates PASS, but C6/C8 are GATE_INACTIVE_LOWPOWER (N_OOS_on=9, power=0.106).
     Per Amendment 3.2: UNVERIFIED ≠ KILL. Cell parks until N_OOS_on ≥ 50 (~Q3-2026).
   - Pre-reg locked → no post-hoc threshold rescue when OOS accrues.
   - DSR_PB = 0.9845, Welch p = 0.000602, bootstrap p = 0.000500, |t| = 3.47, era stable 7/7.

### Decision gate at session-end

D1 (B-MES-EUR) → PARK. D2/D3/D4 still pending user GO before pre-regs are written:
  - **D2** B-MES-LON: MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80 (Chan Ch7 + Fitschen Ch3)
  - **D3** B-MNQ-NYC: MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80 (Chan Ch7 + Fitschen Ch3)
  - **D4** B-MNQ-COX: MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70 (Carver Ch9-10) —
    CAVEAT: B6 lane-correlation +0.773 vs deployed COMEX_SETTLE ORB_G5 — likely portfolio overlap, run anyway but flag for Phase E.

All 4 candidates expected to land at PARK_PENDING_OOS_POWER given identical N_OOS power-floor situation.
Real promotion to Phase E requires N_OOS_on ≥ 50 (Q3-2026 timeframe at current trade rates).

### Decision gate at session-end

Phase B (verification) returned **4 of 4 candidates as PATHWAY_B_ELIGIBLE** — they fail Pathway A discovery DSR (K_family=1850-2700) but pass Pathway B K=1 DSR with theory-citation per Phase 0 Amendment 3.0. All 4 are mechanism-grounded (Chan Ch7 + Fitschen Ch3 for ovn_range_pct_GT80; Carver Ch9-10 for garch_vol_pct_GT70). C8 dir-match and C6 WFE are UNVERIFIED for all 4 (N_OOS = 9-17, below the 50 power floor) — this is the legitimate Phase 0 verdict, not failure.

**Decision needed:** approve Phase D (write Pathway B K=1 pre-regs for D1-D4) or redirect.

### Plan progress (full plan: `docs/plans/2026-04-28-edge-extraction-phased-plan.md`)

- ✅ **Phase A** — Contamination sweep (commit `96bba7a7`)
  - E2 break-bar look-ahead fix in `research/comprehensive_deployed_lane_scan.py`
  - Mode A revalidation of all 59 active validated_setups
  - 18-script E2 LA contamination registry written
  - PR #48 "monotonic-up universal" memory entry flagged TAINTED
  - RULE 16 added to backtesting-methodology-failure-log
- ✅ **Phase B** — Per-candidate verification (this session)
  - DSR Eq.2 + Eq.9 effective-N per cell (Pathway A FAIL, Pathway B PASS)
  - Per-year era stability per cell (all 4 PASS C9)
  - Lane-correlation matrix vs deployed 6 lanes (all |corr| ≤ 0.36, additive)
  - Result: `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md`
- ⏸ **Phase C** — Instrument-family discipline (pending — autonomous)
- ⏸ **Phase D** — Pathway B K=1 pre-regs (REQUIRES YOUR GO)
- ⏸ **Phase E** — Capital integration (REQUIRES YOUR GO + capital-review skill)
- ⏸ **Phase F** — Adjacent edge hunts (deferred until D returns)

### Phase B candidates (all PATHWAY_B_ELIGIBLE)

| ID | Cell | N_IS | ExpR_IS | SR_ann | DSR_PB | Mechanism |
|---|---|---|---|---|---|---|
| B-MES-EUR | MES EUROPE_FLOW O15 RR1.0 long + ovn_range_pct_GT80 | 186 | +0.143 | 0.89 | 0.985 | Chan Ch7 + Fitschen Ch3 |
| B-MES-LON | MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80 | 183 | +0.242 | 0.94 | 0.993 | Chan Ch7 + Fitschen Ch3 |
| B-MNQ-NYC | MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80 | 160 | +0.219 | 1.53 | 0.999 | Chan Ch7 + Fitschen Ch3 |
| B-MNQ-COX | MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70 | 199 | +0.245 | 1.69 | 0.999 | Carver Ch9-10 |

### KILLED in this session

- 3 prior-scan "candidates" using `rel_vol_HIGH_Q3` (E2 break-bar look-ahead, 40-43% post-entry data verified)
- 11 prior-scan OOS-flipped survivors (the OOS flips were the look-ahead artifact's signature; 0 flips on clean scan)
- `dow_thu` / `is_monday` / `is_friday` survivors (no mechanism per Aronson EBTA Ch6)
- PR #48 "monotonic-up universal" TAINTED (rel_vol regression on E2 = post-entry data)

### Files added/changed this session

- `research/phase_b_candidate_evidence_v1.py` (new — Phase B evidence script)
- `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md` (new — verdict per candidate)
- `HANDOFF.md` (this update)
- Memory: `MEMORY.md` flag for PR #48 + registry pointer

### How to pick up

1. Read `docs/plans/2026-04-28-edge-extraction-phased-plan.md` (master plan)
2. Read `docs/audit/results/2026-04-28-phase-b-candidate-evidence.md` (Phase B verdict)
3. User decision: approve Phase D pre-reg writing for B-MES-EUR (highest EV) or redirect
4. If approved: write `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml` per `docs/prompts/prereg-writer-prompt.md`
5. Phase C (instrument-family discipline) can run in parallel — autonomous, no capital touched

### Prior commit context (earlier today)

- **Commit:** 96bba7a7 — research(phase-a): contamination sweep + scan hardening + phased plan

## Session decisions (2026-04-28 — docs lifecycle hardening)

- Added `docs/INDEX.md` as the daily docs front door with a strict 9-entry entrypoint list (authority, workflow, runtime, active plans).
- Applied plan lifecycle metadata frontmatter to all `docs/plans/**/*.md` files: `status`, `owner`, `last_reviewed`, `superseded_by`.
- Restructured plans into date-bounded active/archive topology:
  - active plans now live under `docs/plans/active/2026-04/` (8 files)
  - inactive plans moved under `docs/plans/archive/{2026-02,2026-03,2026-04,undated}/`
  - `*.tasks.json` moved out of root into date-bucketed archive folders
- Added `scripts/tools/list_stale_active_docs.py` to report stale active plan docs by `last_reviewed` age threshold (default 14 days).
- Updated `docs/governance/document_authority.md` with a binding lifecycle policy for plan metadata, active/archive placement, and stale-check command.
- Follow-up refinement: stale-doc scanner now scans only `docs/plans/active/` (not full plan tree) and `docs/INDEX.md` now includes a concise Claude Code daily flow to keep usage project-specific and lightweight.

## Session decisions (2026-04-27 — orphan-branch recovery)

- **Recovered registry hygiene from `codex/control-plane-unify`** (original `9550a668`, 2026-04-24). Manually applied the additive `context/registry.py` subset of the 18-file orphaned commit: registers existing canonical control-plane infrastructure (`docs/runtime/action-queue.yaml`, `pipeline/work_queue.py`, `scripts/tools/work_queue.py`, `pipeline/system_authority.py`, `pipeline/system_context.py`) into FALLBACK_READ_SET + 2 task manifests' canonical_files/doctrine_files; adds `## Control-Plane Truth/Notes/Plane` H2 sections to the 3 markdown render functions. All referenced files VERIFIED present in main. Re-rendered `docs/context/*.md` via `scripts/tools/render_context_catalog.py`. Other 17 files in original commit had heavy CRLF + functional drift; not recovered (separate decision).
- **`codex/control-plane-unify` branch retained locally** — original audit's "DROP — superseded" verdict was wrong on 2 of 3 cited grounds; branch carries unrecovered substantive content; revisit when needed.

## Session decisions (2026-04-27 — sizing-substrate audit closure)

- **Stage-1 sizing-substrate diagnostic SUBSTRATE_WEAK.** 2/6 lanes PASS (EUROPE_FLOW + TOKYO_OPEN, both via rel_vol_session, both UNSTABLE per Carver Ch.7 fn78 → stage2_eligible=False). Tier-A 0/18 STRONG NEGATIVE: deployed binary filters carry no continuous predictive substrate. Pre-reg `docs/audit/hypotheses/2026-04-27-sizing-substrate-prereg.yaml`; result `docs/audit/results/2026-04-27-sizing-substrate-diagnostic.md`.
- **Institutional code+quant audit (evidence-auditor) returned PASS_WITH_RISKS.** Verdict upheld; 5 load-bearing findings. Closures in this session:
  - Finding A (MED, ATR_P50 vol_norm = raw identity): identity test added; effective unique cells = 42 (not K=48) documented in result MD header.
  - Finding B (LOW, COST_LT12 inline formula): anti-drift equivalence test asserts `derive_features` matches canonical `CostRatioFilter`.
  - Finding C (MED rule violation, pooled-finding YAML absent): front-matter added (`flip_rate_pct: 67.0` lane-level, `heterogeneity_ack: true`).
  - Finding D (MED mandatory rule, RULE 13 pressure-test absent): `feature_temporal_validity` extended for RULE 1.1 (hard-banned: `double_break`, `mae_r`, `mfe_r`, `outcome`, `pnl_r`) + RULE 6.3 (E2 break-bar suffixes via `entry_model=E2` gate, canonical authority `trading_app/config.py:3540-3568`); 4 pressure tests + E1 control test added.
  - Finding E (INFO, monotonicity gate `inc or dec`): verified correct, no fix.
  - Findings F/G (INFO): accurate label / pre-specified design, no fix.
  - **Finding H (LOW, deferred):** pre-reg YAML uses `testing_mode: diagnostic_descriptive` which is not in the canonical {`family`, `individual`} set per `pre_registered_criteria.md`/`research-truth-protocol.md`. Doctrine-formalization gap; doesn't affect operative methodology. Future doctrine review to either (a) add `diagnostic_descriptive` as a third canonical value or (b) revise the pre-reg with `corrigendum.md` subdoc.
- **Cross-walk note:** `rel_vol_session` is the SAME column (`daily_features.rel_vol_{ORB_LABEL}`) used by PR #51's universal-scope monotonic-up regression (β₁=+0.278/+0.330/+0.300; t=+9.6/+11.8/+7.5 across MNQ/MES/MGC at 5m E2 RR1.5). Stage-1's 2 PASS cells are lane-level cuts of that universal base, attenuated by stricter cell gates + Carver fn78. Theory grounded in `fitschen_2013_path_of_least_resistance.md` + `chan_2013_ch7_intraday_momentum.md` (already extracted). Hong-Stein attention cite NOT REQUIRED and would be invented.
- **Doctrine entry — `mechanism_priors.md` §7 PARKED:** continuous-sizing substrate of deployed binary filters. Reopen gate: AFML Ch.19 sigmoid bet-sizer (NOT in `resources/`) + per-lane fresh mechanism citation + new pre-reg.
- **Follow-on routing:** do NOT author Path-β `rel_vol_session`-conditioner pre-reg (duplicates PR #51 at lane scope, breaches cumulative-trial Bailey MinBTL bound). Route to existing PR #51 candidate-activation plan in flight per `memory/amendment_3_2_and_cpcv_parked_apr21.md`.
- **Branch finalization:** `chore/freshness-bumps` was scope-bled with 17 unrelated sizing commits. Split via non-destructive label move: `git branch research/sizing-substrate-stage1-2026-04-27` at HEAD, then `git reset --hard 993daccb` on the freshness branch. All sizing commits preserved on the new branch. 8/8 pre-commit checks green on every commit. Neither branch pushed.
- **Test count:** 29 → 35 (all pass).

## Session decisions (2026-04-27 — orphan-branch recovery)

- **Recovered `_stage_file_is_closed()` helper from `codex/stage-hygiene-active-detection`** (original `eb40b35a`, 2026-04-24, "fix(context): ignore closed runtime stage files"). Adds 2-line call in `_parse_stage_file` + new function — closed/completed stage files (status field set OR `## Execution Outcome` H2 present) drop out of `_list_active_stages`, freeing edits previously blocked by stale closed stages. Pre-checks PASS: zero callers in main, additive only, integration via `_list_active_stages` → `stage-gate-guard.py` + `pipeline/system_brief.py` is sound. Manual cherry-pick (file-level `git checkout` had massive CRLF noise); applied just the substantive 2-line + 1-function diff. Branched from `origin/main` as `recover/stage-hygiene-active-detection`. Original commit's 6 stage-doc frontmatter touches dropped (audit-confirmed those files are gone from main).
- **`codex/stage-hygiene-active-detection` branch retained locally** until recovery PR is merged.

## Session decisions (2026-04-27 — PR #124 re-extraction + cleanup)

- **PR #147 merged (170b6085)** — re-extract of abandoned PR #124 onto fresh main. Original PR #124 (`chore/pass-three-drift`, commit `6c279810`) was based on `cebefd92` (pre-PR-#126 + pre-PR-#130-renormalization); CI failed on the wall-clock-time-bomb test fix landed by PR #126, and local rebase developed 4 production-code conflicts (`pipeline/check_drift.py`, `trading_app/live/alert_engine.py`, `trading_app/live/webhook_server.py`, `tests/test_pipeline/test_work_queue.py`) plus 79 CRLF-only files from missing PR #130 renormalization on its base. PR #124 closed with cross-link.
- **Drift check #120 live** — `check_magic_number_rationale(trading_app_dir)` AST-walks `trading_app/live/` for UPPER_SNAKE_CASE numeric constants `abs(value) > 10` and requires `Rationale:` / `rationale` (case-insensitive) within ±10 lines OR membership in `RATIONALE_WHITELIST` (initially empty). Cites Robert Carver, *Systematic Trading*, Ch. 4 (parameter-justification discipline). 9 existing constants retagged to satisfy.
- **Criterion 11 / 12 control state refresh** — both `valid=False reason=db identity mismatch` cleared via `python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto`. C11 now operational 83.1% / age=0d / paths=10000; C12 5 CONTINUE / 1 ALARM (L4 NYSE_OPEN COST_LT12 SR=33.27 vs threshold=31.96 — pre-existing, expected blocked-lane state).
- **Worktree + branch cleanup** — `canompx3-pt` worktree force-removed (held 79 CRLF-only files + 1 stale HANDOFF auto-update stub, no real WIP); `canompx3-pr124` temp worktree removed after PR #147 merge; local branches `chore/pass-three-drift{,-v2}` deleted; remote branches `chore/pass-three-drift{,-v2}` deleted on origin. 3 obsolete stashes dropped (PR #124 cleanup wash, PR #138 mutex already-landed, PR #135 pre-merge HANDOFF state).
- **PR #99 (Codex DRAFT) standing-instruction acknowledgment** — DEFERRED. Hook treated retry as duplicate-attempt. Standing instruction in HANDOFF still holds: leave for Codex.

## Session decisions (2026-04-26 evening + late-evening sweep)

- **Env hardening landed** (PR #136, merged) — venv pinned to `uv.lock`, `_env_drift_lines()` reports drift at session-start, `requirements.txt` regenerated from lock.
- **Same-worktree mutex landed** (PR #138, merged `e306284b`) — atomic `O_EXCL` lock at `<git-dir>/.claude.pid` prevents two Claudes in one worktree (closes the `cannot lock ref 'HEAD'` race). 4 scenarios tested.
- **Architecture decision: per-session clones REJECTED.** Researched against 2026 industry consensus (Anthropic, Augment Code, Penligent, Upsun) — worktrees + runtime isolation is the recommended pattern. `_discover_git_common_root()` in `pipeline/paths.py:33-65` already implements canonical worktree-shared-DB.
- **Phase 1B / 1C DEFERRED** — cross-worktree push lock and runtime isolation audit, both low-priority/advisory. Not blocking.
- **Hook-output trim DEFERRED** — needs production-code edits to `scripts/tools/task_route_packet.py`; properly stage-gated.
- **Recovery PRs #133/#134 CLOSED** — stash content already preserved at `archive/stash-2026-04-26-l1-europe-flow-pr1a-v2` and `archive/stash-2026-04-26-l6-wip-pre-correction`. PRs closed with archive-tag citations; `recover/l1-europe-flow-prereg` and `recover/l6-us-data-2026-diagnostic` deleted local + remote.
- **Evidence-auditor obligation discharged** — the stage `open-pr-review-debt.md` listed 7 deferred PRs. As of 2026-04-26: #72/#74/#59/#8 CLOSED, #12 MERGED, #36 CLOSED (re-extracted as PR #135 which MERGED). #99 remains DRAFT (Codex). All 6 non-DRAFT entries resolved historically without auditor pass — recorded as discharged, not pending.
- **Conservative zombie sweep** — deleted 4 `[gone]`-flagged local branches (`chore/handover-2026-04-26-pr125-126`, `ralph/burndown-v5.3-rebased`, `ralph/crit-high-burndown-v5.3-high-tier`, `research/pr48-sizer-rule-oos-backtest`). `ralph/crit-high-burndown-v5.2` retained because `canompx3-ralph-burndown` worktree still uses it.

## Stages closed this session (work landed, file deleted)

| Stage file | Marker commit |
|---|---|
| hwm-stage1-gap1-none-reason-contract-guard | `8a40a142` (#129) |
| hwm-stage2-tracker-integrity | `8a40a142` (#129) |
| hwm-stage3-orchestrator-pre-session | `8a40a142` (#129) |
| hwm-stage4-inactivity-monitor | `8a40a142` (#129) |
| hwm-warning-tier-notify-dispatch | `8a40a142` (#129) |
| ralph-iter-174-f4-bracket-naked | `87dffa38` |
| osr-debt-frame-audit | `eaa4e055` (#106) |
| debt-burndown-import-side-effects | `2e9f4145` (#128) |
| live-overnight-resilience-hardening | `e02c529d` (audit-verified `2f45a3e8`) |
| ralph-iter-179-pass-one-hardening | `0d54d52e` (#107) |
| ralph-iter-181-r4-signals-rotation | `36276381` (#110) |
| ralph-iter-185-f7-fill-poller-timeout | `36276381` (#110) |
| hardening-three-fixes | `290006e5` (#137) |
| open-pr-review-debt | (this PR — obligation discharged historically) |
| pass-three-magic-number-drift-check | `170b6085` (#147 — re-extracted from abandoned #124) |

## Active stages (1 remaining)

1. **`pr48-mgc-shadow-only-overlay-contract.md`** (IMPLEMENTATION) — operator observation pending; do not action without observation result.

## Next Steps — Active

1. **PR #99** (DRAFT, Codex's `wt-codex-mnq-hiroi-scan`) — research(mnq) MNQ COMEX geometry hardening + COMEX family stamp. Standing instruction: comment + leave for Codex.

2. **Action queue** — fully closed as of 2026-04-26. `cross_asset_session_chronology_spec` marked CLOSED (spec frozen at `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` via PR #101 / `87ebc885`); re-opens when a future cross-asset prereg explicitly cites and gates on it. No P1 ready items remain.

3. **Operator action** — MGC shadow observation; result unblocks `mes_q45_exec_bridge` action-queue item.

## Parallel session activity (DO NOT TOUCH)

- **`canompx3`** — `chore/brain-skill` (orphan branch with merged-content commits + ralph-loop session committing here). Ralph terminal active 2026-04-27 — do not write to this worktree.
- **`canompx3-ralph-burndown`** — `ralph/crit-high-burndown-v5.2` (branch `[gone]` on remote; HEAD `6740d938` predates merged Ralph PRs #107/#110). Stale; do not write to it. Local branch retained because of this worktree.
- **`canompx3-ghost`** — checked out on `main` (this handoff PR's worktree); idle after this session.
- **`canompx3-pt`** — REMOVED 2026-04-27 after PR #124 closed.
- **MGC shadow observation** — operator action; observation result unblocks `mes_q45_exec_bridge` action-queue item.

## Blockers / Warnings

- Memory entries calling F-1 "dormant" are FACTUALLY WRONG — F-1 is **active** and fail-closing in signal-only. Day-1 XFA seeds `_topstep_xfa_eod_balance = $0.0` (canonical per `topstep_scaling_plan.py:51-53`); 6 deployed MNQ lanes ALLOW through `risk_mgr.can_enter()`. Verified empirically (E2E probe in PR #100/B6 work).
- `build_live_portfolio()` is **DEPRECATED** — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only `--profile` path works.
- Drift checks pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (MGC — 1 day with `!= 3 daily_features` rows). Neither blocking.
- Working-tree debt: `requirements.txt` regenerated by `uv export` may be dirty in `canompx3-ghost` (main) worktree. Separate concern; needs its own commit + uv-lock-vs-export reconciliation.
- ~53 zombie local branches remain after 2026-04-26 conservative sweep (4 deleted). Most have merged PRs but no `[gone]` flag because origin still hosts the branch. Aggressive sweep deferred — would benefit from `clean_gone` command + cross-reference of `git branch --list` against `gh pr list --state all`.

## Durable References

- `docs/runtime/action-queue.yaml` — P1/P2 action queue
- `docs/runtime/decision-ledger.md` — durable decisions
- `docs/runtime/debt-ledger.md` — known debt
- `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` — HWM Stages 1-4 design (landed in #129)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` — O-SR grounding (still pending: cusum_monitor → SR Eq 10)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` — ORB premise


