# E2 Break-Bar Look-Ahead Contamination Registry

**Date:** 2026-04-28
**Author:** Claude (autonomous audit pass)
**Severity:** HIGH — research integrity sweep
**Origin postmortem:** `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md`
**Canonical authority:** `trading_app/config.py:3857-3865` (`E2_EXCLUDED_FILTER_PREFIXES`, `E2_EXCLUDED_FILTER_SUBSTRINGS`)
**Literature grounding:** `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` p.4

## Scope

Audit all 73 `research/*.py` scripts that load `orb_outcomes` with `entry_model='E2'` and identify those that reference E2-look-ahead break-bar columns (`rel_vol`, `break_bar_volume`, `break_bar_continues`, `break_delay_min`) as predictors. The 2026-04-21 postmortem established that on E2 entries, ~41% of trades have `entry_ts < break_ts` (i.e., the canonical "break bar" defined by close-outside-ORB is a LATER bar than the E2 range-cross entry), so any feature derived from break-bar properties is post-entry data on those rows.

## Verdict

**18 of 73 E2-using research scripts are contaminated.** Conclusions drawn from these scripts MUST be re-derived on a clean feature set before being cited in deployment, doctrine, or new research baselines.

| # | Script | Features used | Status | Downstream impact |
|---|---|---|---|---|
| 1 | `comprehensive_deployed_lane_scan.py` | rel_vol, bb_volume, break_delay, break_bar_continues | **FIXED 2026-04-28** | Old result superseded; clean re-run is canonical |
| 2 | `participation_optimum_universality_v1.py` | rel_vol | TAINTED — re-run required | PR #48 "monotonic-up universal" memory entry |
| 3 | `participation_optimum_mes_universality_v1.py` | rel_vol | TAINTED | PR #48 MES universality |
| 4 | `participation_optimum_mgc_universality_v1.py` | rel_vol | TAINTED | PR #48 MGC universality |
| 5 | `participation_shape_cross_instrument_v1.py` | rel_vol | TAINTED | PR #48 cross-instrument |
| 6 | `pr48_participation_shape_oos_replication_v1.py` | rel_vol | TAINTED | PR #48 OOS replication |
| 7 | `q1_h04_mechanism_shape_validation_v1.py` | rel_vol, break_delay_min | TAINTED | "shape validation" claim |
| 8 | `close_h2_book_path_c.py` | rel_vol | TAINTED | H2 close-out path |
| 9 | `h2_exploitation_audit.py` | rel_vol | TAINTED | H2 audit |
| 10 | `audit_comex_settle_orb_g5_failure_pocket.py` | rel_vol, break_bar_continues, break_delay_min | TAINTED | COMEX_SETTLE failure-pocket diagnosis |
| 11 | `rel_vol_is_only_quantile_sensitivity.py` | rel_vol | TAINTED — was a META-audit | Was attempting to validate rel_vol-on-E2; conclusion suspect |
| 12 | `rel_vol_mechanism_decomposition.py` | rel_vol | TAINTED — was a META-audit | Same |
| 13 | `research_vol_regime_wf.py` | rel_vol | TAINTED — claims "zero lookahead guaranteed" L15, contradicts canonical | Vol-regime walk-forward |
| 14 | `stress_test_rel_vol_finding.py` | rel_vol | TAINTED — meta-audit of rel_vol | Stress test |
| 15 | `stress_test_rel_vol_finding_v2.py` | rel_vol | TAINTED | Stress test v2 |
| 16 | `t0_t8_audit_volume_cells.py` | rel_vol, break_bar_volume | TAINTED | T0-T8 audit on volume cells |
| 17 | `vwap_comprehensive_family_scan.py` | rel_vol (referenced in narrative; verify if used as predictor) | NEEDS REVIEW | VWAP family |
| 18 | `research_mgc_e2_microstructure_pilot.py` | break_delay (window-sizing only, not predictor) | LIKELY CLEAN — verify | MGC microstructure pilot |

## Additional candidates (surfaced 2026-04-28 by drift check 124)

9 additional scripts surfaced by `check_e2_lookahead_research_contamination()`. Annotated 2026-04-28.

| # | Script | Features used | Status | Policy annotation |
|---|---|---|---|---|
| 19 | `phase_d_d0_backtest.py` | rel_vol as size-scaling predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 20 | `break_delay_filtered.py` | break_delay_min as predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 21 | `break_delay_nuggets.py` | break_delay_min as predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 22 | `l1_europe_flow_pre_break_context_scan.py` | rel_vol as predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 23 | `mnq_comex_unfiltered_overlay_v1.py` | rel_vol as predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 24 | `mnq_l1_europe_flow_prebreak_context_v1.py` | rel_vol as predictor | TAINTED | `# e2-lookahead-policy: tainted` |
| 25 | `audit_sizing_substrate_diagnostic.py` | break_bar suffixes in guard list (not predictor) | CLEARED | `# e2-lookahead-policy: cleared` — implements the gate |
| 26 | `output/confluence_program/phase0_run.py` | rel_vol in eligibility matrix analysis | NOT-PREDICTOR | `# e2-lookahead-policy: not-predictor` — audits the feature |
| 27 | `shadow_htf_mes_europe_flow_long_skip.py` | `break_dir='long'` as fire-day SELECTOR predicate (+ break_bar_volume/rel_vol as ledger context) | TAINTED (reclassified 2026-04-28 via real-data audit; was NOT-PREDICTOR) | `# e2-lookahead-policy: tainted` — measured 42.6% of MES EUROPE_FLOW O15 E2 RR1.5 IS trades have `entry_ts < break_ts` (N=1719). Fire-day selection by `break_dir='long'` is post-entry on those rows. Zero-capital observational ledger so deployment risk = 0; methodologically dirty per postmortem § 5.2 + RULE 6.3. Re-pre-register with pre-break direction proxy required before any deployment use. |

## Confirmed downstream contamination

**PR #48 cluster (memory note `participation_optimum_universality_apr20.md`):**
- "monotonic-up universal MNQ/MES/MGC; 5 CANDIDATE_READYs across 15m/5m MNQ" — built entirely on `rel_vol` regression (`pnl_r = β₀ + β₁·rel_vol + β₂·rel_vol²`). On E2 this is 41% post-entry data on each trade. **The "monotonic up" curvature signature is what break-bar volume looks like ex post — high-volume break bars correlate with continuation BECAUSE the entry already succeeded against an in-progress move.**
- The 5 CANDIDATE_READYs from PR #50/#51 are NOT contaminated by this — those used the high-RR cross-family scan which doesn't include rel_vol as a predictor. Verified by inspection of `mnq_unfiltered_high_rr_family_v1.py`.

**`bb_volume_ratio_*` cluster:**
- Same root cause: `break_bar_volume / orb_volume` numerator is post-entry on E2.
- Original "bb_volume_ratio_HIGH MNQ CME_REOPEN long RR1.5/2.0" finding from 2026-04-15 dirty scan — INVALID.

## Caveats / limitations

- **NOT every script using rel_vol is wrong on every entry model.** E1 and E3 enter AFTER the break bar closes; for them, break-bar properties are knowable. The contamination only applies on E2.
- **`research_mgc_e2_microstructure_pilot.py`** uses `break_delay` only as a *window-sizing constant* (50min window covers max 39min observed delay), not as a predictor — likely clean. Re-verify before clearing.
- **`vwap_comprehensive_family_scan.py`** mentions `rel_vol` in narrative but does not appear to use it as a predictor of pnl_r; needs targeted re-verification.
- **The clean re-run** (post-fix) of `comprehensive_deployed_lane_scan.py` is the canonical baseline. Any prior conclusion that contradicts it should be revisited via the clean scan first.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python -c "..."  # see prompt log of 2026-04-28 audit run
```

Heuristic: regex-grep for `entry_model.*E2` AND `rel_vol|break_bar_volume|break_bar_continues|break_delay`. Filter out comment-only references. Manual review of each hit.

## Action items

1. **Memory update:** flag `participation_optimum_universality_apr20.md` and `recent_findings.md` PR #48 entry as `TAINTED — re-run required after E2 LA fix`.
2. **Failure-log entry:** append to `.claude/rules/backtesting-methodology-failure-log.md` with this date slug and registry path.
3. **Re-run plan:** for each TAINTED script that influences live deployment, run a clean replacement using `ovn_range_pct`, `garch_forecast_vol_pct`, `atr_20_pct`, `prev_day_range`, `gap_open_points` (Rule 6.1 safe list) instead of `rel_vol`. Keep the script's hypothesis and pre-reg structure; only swap the predictor.
4. **Drift check (FUTURE):** add a check that flags any new `research/*.py` script combining `entry_model='E2'` with `rel_vol|break_bar_*|break_delay` references — surface as PASS/WARN at pre-commit.

## Re-derivation dispatch (added 2026-04-28 follow-up commit)

Replaces the generic Action Item §3 wording for the **specifically time-sensitive** items. Action Item §3 still governs the rest.

### Priority 1 — Phase D D-0 clean re-derivation (HARD DEADLINE 2026-05-15)

- **Tainted script:** `research/phase_d_d0_backtest.py` (registry row 19).
- **Tainted predictor:** `rel_vol_COMEX_SETTLE` as size-scaling bucket variable (P33/P67 splits → 0.5×/1.0×/1.5× size_mult).
- **External deadline:** 2026-05-15 — the daily-shadow gate eval fires once on this date per `phase_d_volume_sizing_pilot_d1_review.py`. Any deployment decision based on the D-0 verdict will consume the *current* contaminated locks unless re-derivation lands first.
- **Replacement predictor — pick ONE pre-break candidate from RULE 6.1 safe list** (do not stack — keep the K=1 Pathway B framing intact):
  - First choice: `garch_forecast_vol_pct` — closest economic match to `rel_vol` (volatility-regime conditioning, made at prior close).
  - Second choice: `atr_20_pct` (20-day rolling, prior close).
  - Tertiary: `ovn_range_pct` (overnight 09:00–17:00 Brisbane, valid for COMEX_SETTLE per RULE 1.2 valid-domain table since COMEX_SETTLE starts 04:30 next day).
- **Pre-reg amendment required.** Original at `docs/audit/hypotheses/2026-04-18-phase-d-d0-rel-vol-sizing-mnq-comex-settle.yaml`. Amendment substitutes the predictor; preserves H1 metric (Sharpe uplift % ≥ 15% CONTINUE / < 10% KILL), kill criteria, K=1 framing. Pre-reg writer prompt at `docs/prompts/prereg-writer-prompt.md` enforces gate.
- **Inheritance:** the 28-day shadow-append ledger from 2026-04-17 → present is recomputed against the new predictor. The shadow-append script remains valid (it appends qualifying trading days, not bucket-conditional outcomes).
- **Verdict treatment:** if the contaminated D-0 verdict and the clean re-derivation agree (both PASS or both KILL), the clean re-derivation is the canonical record. If they disagree, the clean re-derivation governs and the D-0 verdict is documented as artifact.

### Priority 2 — historical predictor-tainted scripts (no external deadline)

Registry rows 20–24 and 27: `break_delay_filtered.py`, `break_delay_nuggets.py`, `l1_europe_flow_pre_break_context_scan.py`, `mnq_comex_unfiltered_overlay_v1.py`, `mnq_l1_europe_flow_prebreak_context_v1.py`, `shadow_htf_mes_europe_flow_long_skip.py`.

Re-run only IF the script's prior conclusion is downstream-cited in memory, doctrine, or a deployed-lane decision. Shadow_htf is zero-capital observational (deployment risk = 0); the others have no downstream citation beyond the registry itself (verified via `git grep` 2026-04-28). Status: re-derivation deferred until citation appears or deployment-use is proposed; tainted annotation is sufficient absent that. If shadow_htf moves toward deployment use, re-pre-register with a pre-break direction proxy.

## Not done by this registry

- Did NOT re-run any of the 18 contaminated scripts on clean features
- Did NOT compute the magnitude of bias on prior PR #48 cells (estimated 5-30% inflation per the 2026-04-27 v1 high-RR CORRECTION precedent — but the qualitative shape of "monotonic up" may itself be artifactual, not just a magnitude inflation)
- Did NOT update `MEMORY.md` index pointers (separate commit)
- Did NOT modify drift-check enforcement
