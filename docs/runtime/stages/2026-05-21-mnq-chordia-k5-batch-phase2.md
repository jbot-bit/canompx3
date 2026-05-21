---
task: Phase 2 MNQ Chordia K=3 batch audit (post-PD_* drop, post-CME_PRECLOSE-shadow drop)
mode: IMPLEMENTATION
scope_lock:
  - docs/audit/hypotheses/2026-05-21-mnq-tokyoopen-orbvol4k-o15-chordia-unlock-v1.yaml
  - docs/audit/hypotheses/2026-05-21-mnq-comexsettle-atrp50-chordia-unlock-v1.yaml
  - docs/audit/hypotheses/2026-05-21-mnq-usdata1000-ovnrng50-chordia-unlock-v1.yaml
  - docs/audit/results/2026-05-21-mnq-tokyoopen-orbvol4k-o15-chordia-unlock-v1.md
  - docs/audit/results/2026-05-21-mnq-tokyoopen-orbvol4k-o15-chordia-unlock-v1.csv
  - docs/audit/results/2026-05-21-mnq-tokyoopen-orbvol4k-o15-chordia-unlock-v1_summary.csv
  - docs/audit/results/2026-05-21-mnq-comexsettle-atrp50-chordia-unlock-v1.md
  - docs/audit/results/2026-05-21-mnq-comexsettle-atrp50-chordia-unlock-v1.csv
  - docs/audit/results/2026-05-21-mnq-comexsettle-atrp50-chordia-unlock-v1_summary.csv
  - docs/audit/results/2026-05-21-mnq-usdata1000-ovnrng50-chordia-unlock-v1.md
  - docs/audit/results/2026-05-21-mnq-usdata1000-ovnrng50-chordia-unlock-v1.csv
  - docs/audit/results/2026-05-21-mnq-usdata1000-ovnrng50-chordia-unlock-v1_summary.csv
  - docs/runtime/chordia_audit_log.yaml
---

## Blast Radius

- docs/audit/hypotheses/ — 3 NEW prereg YAMLs (~220 lines each, ~660 total). Author-owned per multi-terminal-shared-file-hygiene exception.
- docs/audit/results/ — 3 NEW result MDs + 3 row-level .csv + 3 .summary.csv from canonical strict-unlock runner.
- docs/runtime/chordia_audit_log.yaml — CONDITIONAL APPEND only for PASS_CHORDIA / PASS_PROTOCOL_A verdicts; multi-terminal three-check protocol applies (PreToolUse guard fires automatically).
- Zero edits to pipeline/ or trading_app/.
- Reads: gold.db (read-only via canonical runner), validated_setups (PROVENANCE_ONLY, not as truth layer).
- Writes: docs/audit/hypotheses/, docs/audit/results/, docs/runtime/chordia_audit_log.yaml (conditional).

## Stage 1 — Pre-flight verifications (COMPLETE)

### E2-SAFETY (canonical source: trading_app/config.py:4045-4057)
- `E2_EXCLUDED_FILTER_PREFIXES = ('VOL_RV', 'ATR70_VOL')` — break-bar look-ahead
- `E2_EXCLUDED_FILTER_SUBSTRINGS = ('_CONT', '_FAST', 'NOMON_CONT')` — break-bar look-ahead
- `E2_DEPLOYMENT_UNSAFE_FILTER_PREFIXES = E2_DIRECTION_SELECTOR_FILTER_PREFIXES = ('PD_',)` — direction selector via break_dir
- **All PD_* candidates (incl. PDR_R125) DROPPED — DEPLOYMENT-UNSAFE on E2.**

### CME_PRECLOSE PARK-shadow analysis (2026-05-21)
3 audited siblings on MNQ CME_PRECLOSE E2 RR1.0 CB1 (chordia_audit_log.yaml):

| strategy_id | t_stat | verdict |
|---|---|---|
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_NO_FILTER | 4.691 | PARK (2026-05-15) |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4 | 4.677 | PARK (2026-05-15) |
| MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60 | 4.211 | PARK (2026-05-02) |

NO_FILTER cleared the strict t≥3.79 hurdle but PARKed at deployment stage on OOS power/sign. **All filter variants on this lane inherit the parent OOS problem.** The two original "surviving" Phase 2 candidates (X_MES_ATR60_O15 and ORB_VOL_8K_O15) are O15 children of this same cluster. **Expected verdict ratio for the K=5-as-designed: ~0 PASS_CHORDIA / 2 likely PARK / 3 unknown.** Both DROPPED.

Authority: `memory/feedback_mid_tier_session_not_hidden_gem_nyse_close_park.md` ("running N-filter Chordia pre-regs on a Sharpe-X baseline session burns N× MinBTL trial budget on what is fundamentally one underlying question") + `memory/feedback_max_profit_grow_chordia_inventory_not_force_slots.md` + `memory/feedback_chordia_candidate_selection_population_vs_sample_frame.md`.

### Replacement-candidate selection (maximally-diverse K=3)
Greedy diversity from 55-row eligible pool (MNQ active, status=active, sample_size≥50, t_naive≥3.79, E2-safe, not-yet-audited, **excluding CME_PRECLOSE cluster**). Diversity axes: distinct `orb_label` + distinct filter family. Selected top-3 by t_naive subject to that constraint:

| # | strategy_id | session | family | orb | rr | N | t_naive | deployment_scope |
|---|---|---|---|---|---|---|---|---|
| 1 | MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O15 | TOKYO_OPEN | ORB_VOL | 15m | 1.0 | 706 | 4.78 | deployable |
| 2 | MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50 | COMEX_SETTLE | ATR_P | 5m | 1.0 | 833 | 4.61 | deployable |
| 3 | MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50 | US_DATA_1000 | OVNRNG | 5m | 1.0 | 1221 | 4.23 | deployable |

3 distinct sessions, 3 distinct filter families, all E2-deployment-safe, all above strict Chordia hurdle, none in current audit log.

Prior-art on each session:
- TOKYO_OPEN: 1 prior audit FAIL_BOTH (COST_LT12 at t=3.27, 2026-05-01). Zero ORB_VOL audits on this session — genuinely new question.
- COMEX_SETTLE: 5 PASS_CHORDIA + 1 PARK (X_MGC_ATR70_O15). Established producing zone. Zero ATR_P* audits on this session.
- US_DATA_1000: 5 PASS_CHORDIA + 2 FAIL_CHORDIA + 1 PARK. Established producing zone. Zero OVNRNG audits on this session.

## Stage 2 — Author K=3 (each prereg gated by estimate_k_budget.py before save)

Each prereg uses the heavyweight Chordia template shape (per `2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.yaml`):
- `execution.entrypoint: research/chordia_strict_unlock_v1.py`
- `total_expected_trials: 1` (one-cell exact-lane replay)
- `chordia_threshold_basis: "Criterion 4 no-theory strict threshold (t >= 3.79)"`
- `kill_criteria` includes t<3.79, ExpR<=0, N_IS<100, OOS sign opposition at N>=30
- `decision_rule.{continue_if, park_if, kill_if}`

K-budget gate: `.venv/Scripts/python.exe scripts/tools/estimate_k_budget.py --hypothesis <yaml>` — must return PASS before commit.

## Stage 3 — Run canonical runner

```powershell
.venv/Scripts/python.exe research/chordia_strict_unlock_v1.py --hypothesis-file <each yaml>
```

Per candidate. Each run emits the result MD + row-level CSV + summary CSV listed in scope_lock.

## Stage 4 — Manual audit-log append (CONDITIONAL on PASS)

For each verdict that resolves to PASS_CHORDIA (t≥3.79) or PASS_PROTOCOL_A (3.00≤t<3.79 with theory-grant — N/A here as no theory claimed): append to `docs/runtime/chordia_audit_log.yaml::audits[]`.

Shared-state hygiene three-check protocol (per `.claude/rules/multi-terminal-shared-file-hygiene.md`) fires automatically via PreToolUse `shared-state-commit-guard.py`. PARK / FAIL_CHORDIA / FAIL_BOTH verdicts also get appended (per the audit log doctrine — every audit run is recorded).

## Stage 5 — Verify + commit

```bash
python pipeline/check_drift.py    # all guardrails pass
git add docs/audit/hypotheses/2026-05-21-mnq-*.yaml \
        docs/audit/results/2026-05-21-mnq-*.{md,csv} \
        docs/runtime/chordia_audit_log.yaml \
        docs/runtime/stages/2026-05-21-mnq-chordia-k5-batch-phase2.md
git commit -m "research(mnq-chordia-phase2): K=3 diverse strict-unlock batch"
```

Acceptance:
- estimate_k_budget PASS on all 3 preregs
- Canonical runner exits 0 on all 3
- check_drift.py passes (zero violations on the new prereg + audit-log delta)
- chordia_audit_log.yaml schema valid (loader smoke-test via `from trading_app.chordia import load_chordia_audit_log; load_chordia_audit_log()`)
