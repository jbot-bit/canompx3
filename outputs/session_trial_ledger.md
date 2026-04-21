# Session Trial Ledger

Cap: `300` total parameterized trials for this session.

| timestamp | phase | hypothesis_id | trial_count_delta | running_total | cap | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-21T02:46:50Z | preflight | session_init | 0 | 0 | 300 | Initialized ledger on fresh branch before any hypothesis parameterization. |
| 2026-04-21T02:53:33Z | pcc-5 | shared_reason_pattern_root_cause | 0 | 0 | 300 | Diagnostic-only posture contribution; no hypothesis parameterization. |
| 2026-04-21T02:54:42Z | pcc-3 | theory_state_elevation_audit | 0 | 0 | 300 | Retrospective theory-state elevation audit for the six live lanes; no parameterization. |
| 2026-04-21T02:56:03Z | pcc-4 | grandfathering_eligibility_audit | 0 | 0 | 300 | Grandfathering map for the six live lanes; no hypothesis parameterization. |
| 2026-04-21T02:57:25Z | pcc-6 | mode_a_readiness_scan | 0 | 0 | 300 | Relative Mode A rerun priority scan for the six live lanes; no hypothesis parameterization. |
| 2026-04-21T02:59:55Z | pcc-2 | dsr_driver_recheck | 0 | 0 | 300 | Recomputed DSR drivers match the Phase B lineage within rounding; no arithmetic drift. |
| 2026-04-21T03:10:17Z | pcc-1 | phase_b_calibration_probe | 0 | 0 | 300 | Synthetic calibration probe confirms the Phase B gate logic can emit KEEP for a strong clean-holdout candidate and separately degrade weak, contaminated, and SR-alarmed cases. |
| 2026-04-21T03:15:33Z | phase-1 | orthogonal_pov_map | 0 | 0 | 300 | Ranked the mandated candidate families, selected the top 3 posture-blocker-proof directions, and fenced off the dropped families for this session. |
| 2026-04-21T03:21:51Z | phase-2 | pre_registered_top3 | 0 | 0 | 300 | Locked the top-3 orthogonal hunt directions into committed family hypotheses before any candidate extraction or metric inspection. |
| 2026-04-21T03:36:11Z | phase-3 | mgc_regime_ortho_v1 | 12 | 12 | 300 | First locked family executed end-to-end on canonical data. Result: DEAD at Phase 3 — zero BH-FDR family survivors, destruction-shuffle p=0.3466, rng-null p=0.3586, positive control passed, negative control passed. |
| 2026-04-21T03:38:39Z | phase-3-block | mes_session_boundary_v1 | 0 | 12 | 300 | Aborted before data contact. Locked `ASIA_RANGE_ATR_Q67_HIGH` predicate is timing-invalid for `BRISBANE_1025`, `TOKYO_OPEN`, and `SINGAPORE_OPEN` under `.claude/rules/backtesting-methodology.md` and `pipeline/build_daily_features.py`. |
| 2026-04-21T03:50:00Z | phase-3 | mgc_microstructure_v1 | 8 | 20 | 300 | Third locked family executed end-to-end on canonical data. Result: DEAD at Phase 3 — zero BH-FDR family survivors, destruction-shuffle p=0.4263, rng-null p=0.3944, positive control passed, negative control passed. Features were timing-valid at ORB close, but only `US_DATA_830` had scoped rows; `BRISBANE_1025` remained zero-coverage. |
