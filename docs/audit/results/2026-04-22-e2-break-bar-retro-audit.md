---
title: "E2 Break-Bar Look-Ahead — Research-Script Retro-Audit Surface"
date: 2026-04-22
type: audit_surface_map
follow_up_to: docs/postmortems/2026-04-21-e2-break-bar-lookahead.md
source_doctrine: .claude/rules/backtesting-methodology.md § RULE 6.1
canonical_code: trading_app/config.py:3540-3568 (E2_EXCLUDED_FILTER_PREFIXES/SUBSTRINGS)
status: surface_mapped_only_no_rerun
---

# E2 break-bar retro-audit — classification of § 5.1 surface

## 0. Scope and non-scope

**In scope:** triage the 17 research scripts listed in `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md § 5.1` into verdicts.

**Out of scope:**

- Re-running any contaminated scan. Each rerun is a discrete follow-up task per file (entry_model gating + `entry_ts >= break_ts` filter + redo statistics).
- Editing the scripts themselves. Retro-audit maps surface; it does not modify producers.
- Changing doctrine. `RULE 6.1` was already narrowed by PR #67.

**Verdict categories:**

| Category | Meaning |
|---|---|
| `CONTAMINATED` | Script fires E2 trades AND uses one of `break_delay_min` / `break_bar_continues` / `break_bar_volume` / `break_dir` / `break_ts` as a filter predicate, feature, or segmentation key. Any survivor is suspect. |
| `PARTIAL_CONTAMINATED` | Script parameterizes `entry_model` from a keeper/preset row. E2 rows in the sweep are contaminated; E1/E3 rows are safe. Re-audit per-row. |
| `SAFE_E1_ONLY_LIKELY` | Script filters `entry_model = 'E1'` (or E3) before consuming break-bar columns. Fast-grep evidence consistent; one-line manual verification recommended before declaring final. |
| `SAFE_INTERNAL_BREAK_DIR` | Script computes its OWN `break_dir` label from E2 fill direction (pandas-derived, not `daily_features.break_dir`). Not a look-ahead because the label derives from the already-taken trade. |
| `SAFE_DESCRIPTIVE_ONLY` | Column appears only in `SELECT` output for logging/display, never in `WHERE` / `.loc[]` / `.query()` / `.groupby()`. |
| `MANUAL_REVIEW` | Evidence not decisive from fast-grep. Needs a human read of the specific usage site. |

## 1. Verdict table — 17 § 5.1 scripts

| # | File | Verdict | Evidence line(s) | What makes it the verdict |
|---|---|---|---|---|
| 1 | `q1_h04_mechanism_shape_validation_v1.py` | `MANUAL_REVIEW` | file contains `delay_col = f"orb_{sess}_break_delay_min"` assignment (L68) + `entry_model` filter reference | Variable is assigned but downstream use unclear from fast-grep; the assignment suggests delay is used as a feature but could also be unused/leftover |
| 2 | `shadow_htf_mes_europe_flow_long_skip.py` | `CONTAMINATED` | `WHERE ... AND d.orb_EUROPE_FLOW_break_dir = ?` with `entry_model='E2'` (YAML semantic `break_dir='long'`) | E2 trade outcomes filtered by post-break `break_dir` predicate |
| 3 | `volume_confluence_scan.py` | `CONTAMINATED` | `break_delay_LT2` / `break_delay_GT10` feature branches (L128-133); `bb_volume_ratio_*` feature (L118-122); `mask_dir = (df["break_dir"] == direction)` (L189) | No `entry_model` filter → covers E2; break-bar columns consumed as anchor/partner composite features |
| 4 | `t0_t8_audit_volume_cells.py` | `CONTAMINATED` | Two `entry_model='E2'` filter sites + `break_bar_volume` ratio used as composite feature + `break_dir` segmentation | E2-scoped scan with explicit break-bar-feature consumption |
| 5 | `update_forward_gate_tracker.py` | `PARTIAL_CONTAMINATED` | L82/L102: `AND o.entry_model='{row["entry_model"]}'` — entry_model comes from keeper row | E2 rows in the keeper manifest produce contaminated forward-gate estimates; E1/E3 rows unaffected |
| 6 | `research_wf_stress_keepers.py` | `PARTIAL_CONTAMINATED` | L96/L118: same parameterized `entry_model='{row[...]}'` pattern | Same: per-row; E2 rows contaminated |
| 7 | `research_wide_non_leadlag_composite.py` | `SAFE_E1_ONLY_LIKELY` | `E1=1` refs, `entry_model` filter present, `break_dir` consumed as predicate | Fast-grep shows E1-only selection; confirm by reading the single `entry_model` filter site before final sign-off |
| 8 | `research_universal_hypothesis_pool.py` | `CONTAMINATED` | L84: `SELECT o.entry_model, o.confirm_bars` → no entry_model filter; L179: `groupby(["entry_model", "confirm_bars", "rr_target"])` | E2 slice included in groupby; any break-bar column used as composite predicate within that groupby contaminates the E2 cell |
| 9 | `research_shinies_overlay_stack_presets.py` | `PARTIAL_CONTAMINATED` | L72/L96: parameterized `entry_model='{row[...]}'` | E2 preset rows contaminated |
| 10 | `research_shinies_universal_overlays.py` | `PARTIAL_CONTAMINATED` | L119/L155: parameterized `entry_model='{row[...]}'` | Same |
| 11 | `research_shinies_bqs_overlay_tests.py` | `SAFE_E1_ONLY_LIKELY` | `E1=2` refs | Fast-grep shows only E1 references; verify one entry_model line before final |
| 12 | `research_round_number_proximity.py` | `SAFE_E1_ONLY_LIKELY` | `E1=1` ref | Same; verify |
| 13 | `research_mgc_e2_microstructure_pilot.py` | `SAFE_INTERNAL_BREAK_DIR` | L88-90: `sample_df["break_dir"]` is a **pandas-derived label** from `model_entry_price` vs `orb_high_col`; L430: `break_delay_min` appears in `SELECT` output only | Internal label is NOT the contaminated `daily_features.break_dir`; it is derived from the already-taken E2 trade. SELECT-only usage for `break_delay_min`. |
| 14 | `research_false_breakout_bqs_tests.py` | `SAFE_E1_ONLY_LIKELY` | `E1=1` ref | Verify one site |
| 15 | `comprehensive_deployed_lane_scan.py` | `CONTAMINATED` | L215: `AND d.orb_{session}_break_dir IN ('long','short')` with `entry_model='E2'`; L85/L101: `break_bar_volume` as ratio feature | E2 lane scan with break-bar-feature consumption |
| 16 | `break_delay_nuggets.py` | `CONTAMINATED` | L39: `delay_col = f"orb_{sess}_break_delay_min"`; L51: `entry_model = 'E2' AND confirm_bars = 1` | E2-scoped scan using `break_delay_min` as the scan feature. By script name and construction this is a pure E2 break-bar predicate scan. |
| 17 | `break_delay_filtered.py` | `CONTAMINATED` | L215-219: `entry_model = 'E2'` + `break_bar_continues = true AND break_delay_min IS NOT NULL` predicate | Archetype leak — E2 outcomes filtered by post-break columns. |

### 1a. SAFE_E1_ONLY_LIKELY — verified in this PR

| File | entry_model filter evidence | Final |
|---|---|---|
| `research_wide_non_leadlag_composite.py` | L97: `AND o.entry_model IN ('E0','E1')` | **SAFE** (E0/E1 only) |
| `research_shinies_bqs_overlay_tests.py` | L32-80: all preset rows are `entry_model: 'E0'` or `'E1'`; L134 filters on preset row | **SAFE** (E0/E1 only) |
| `research_round_number_proximity.py` | L188: `AND o.entry_model = 'E1'` | **SAFE** (E1 only) |
| `research_false_breakout_bqs_tests.py` | L48: `AND o.entry_model='E1'` | **SAFE** (E1 only) |

All 4 previously-LIKELY rows are now confirmed **SAFE**.

## 2. Rollup

| Verdict | N | % |
|---|---|---|
| `CONTAMINATED` | 7 | 41% |
| `PARTIAL_CONTAMINATED` | 3 | 18% |
| `SAFE` (E1/E0 only, verified) | 4 | 24% |
| `SAFE_INTERNAL_BREAK_DIR` | 1 | 6% |
| `MANUAL_REVIEW` | 1 | 6% |
| `SAFE_DESCRIPTIVE_ONLY` | 1 (#13 has descriptive SELECT-only for `break_delay_min`) | — |

## 3. Bonus surface — CLASSIFIED in this PR (§ 5.1 was incomplete)

PR #67 § 5.1 enumerated scripts matching `break_delay_min` / `break_bar_continues` / `break_bar_volume`. It did **not** enumerate scripts that use `break_dir` or `break_ts` **as E2 predicates** — but § 8 Binding Decision #2 explicitly bans `break_dir` and `break_ts` as E2 predictors. § 5.1 should have enumerated all five banned columns, not three.

Classification of the 10 bonus-surface scripts:

| # | File | Verdict | Evidence |
|---|---|---|---|
| B1 | `close_h2_book_path_c.py` | **CONTAMINATED** | E2 ref × 3; `break_dir` predicate × 3 (L67, L114, L202); no entry_ts gate |
| B2 | `carry_encoding_exploration.py` | **CONTAMINATED** | E2 × 2; `break_dir` × 13 refs; predicate `AND d.orb_{ts}_break_dir IS NOT NULL` (L104); no gate |
| B3 | `garch_all_findings_consolidated_stress.py` | **CONTAMINATED** | E2 × 1; predicate `AND d.orb_{sess}_break_dir='{direction}'` with `entry_model='E2'` (L71); no gate |
| B4 | `garch_additive_sizing_audit.py` | **PARTIAL_CONTAMINATED** | parameterized entry_model likely includes E2; `break_dir` predicate (L106) |
| B5 | `garch_all_sessions_universality.py` | **CONTAMINATED** | E2 × 1; predicate with `entry_model='E2'` (L94); no gate |
| B6 | `garch_broad_exact_role_exhaustion.py` | **PARTIAL_CONTAMINATED** | parameterized entry_model; `break_dir` × 5 predicate uses |
| B7 | `garch_comex_settle_institutional_battery.py` | **CONTAMINATED — LOAD-BEARING** | E2 × 1; `break_dir` × 5 predicate uses (L239); **this is the institutional-grounded battery for the L3 deployed lane** |
| B8 | `garch_normalized_sizing_audit.py` | **PARTIAL_CONTAMINATED** | parameterized entry_model |
| B9 | `garch_partner_state_provenance_audit.py` | **PARTIAL_CONTAMINATED** | parameterized entry_model |
| B10 | `audit_comex_settle_orb_g5_failure_pocket.py` | **CONTAMINATED — LOAD-BEARING** | E2 × 3; `break_dir` × 12; `break_delay_min` × 3; `break_bar_continues` × 3; no gate. **This is PR #73's COMEX_SETTLE ORB_G5 failure-pocket audit — backs the L3 "real behavioral edge p=0.012" claim.** |

Bonus surface rollup: **5 CONTAMINATED (2 load-bearing), 4 PARTIAL_CONTAMINATED**.

## 3a. Archive sweep

`research/archive/*.py` — 18 files reference banned columns; 17 predate E2 (A0/E0/E1 era, low priority). Only one has the E2 + `break_dir` predicate pattern:

| File | Verdict |
|---|---|
| `research/archive/research_nested_orb_stacking.py` | **CONTAMINATED (archive)** — E2 × 1, `break_dir` predicate × 3 |

Other archive hits (17 files) are A0/E0/E1-era scripts with no `'E2'` literal — treat as retrospectively retired, re-audit only if resurrected.

## 3b. Deployed-book cite-chase — filter-definition verification

The institutional question: **does any live deployed lane's operational filter or its canonical validation consume a banned break-bar column?** Answered by verifying canonical filter definitions, not citation proximity.

### Live deployed lanes (`docs/runtime/lane_allocation.json` 2026-04-18)

| # | Lane strategy_id | annual_r | Filter class | Reads (canonical) |
|---|---|---|---|---|
| L1 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 46.1 | `ORB_G5` (size) | `orb_{s}_size` |
| L2 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 44.0 | `ATR_P50` (vol pct) | `atr_20_pct` |
| L3 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 41.0 | `ORB_G5` (size) | `orb_{s}_size` |
| L4 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 29.0 | `COST_LT12` (cost) | `cost_risk_pct` |
| L5 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 18.8 | `ORB_G5` (size) | `orb_{s}_size` |
| L6 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 20.3 | `COST_LT12` (cost) | `cost_risk_pct` |

### Filter-definition verification (canonical)

- `trading_app/config.py:2740` — `"G5": OrbSizeFilter(filter_type="ORB_G5", description="ORB size >= 5 points", min_size=5.0)` — pure size threshold.
- `cost_risk_pct` and `atr_20_pct` are derived from canonical pre-entry data per `.claude/rules/backtesting-methodology.md § RULE 6.1`.
- **No deployed lane's operational filter consumes `break_delay_min`, `break_bar_continues`, `break_bar_volume`, `break_dir`, or `break_ts`**.

### Validation-script verification

PR #71's L3 `p=0.012` computation path (the "real behavioral edge" claim):

- Executed by `research/audit_l1_orb_g5_arithmetic_only_check.py` (new read-only script, **not** on the contaminated list).
- Canonical source declared in the script docstring L42-48: `orb_outcomes ⨝ daily_features` on `(trading_day, symbol, orb_minutes)`, `trading_day < 2026-01-01` (Mode A IS), `entry_ts IS NOT NULL AND pnl_r IS NOT NULL`.
- Split predicate: `orb_{s}_size >= 5.0` (canonical pre-entry).
- Test: fire-vs-nofire two-proportion z on WR + Welch on ExpR.
- **No break-bar column anywhere in the computation.**

### Lane-by-lane exposure verdict

| Lane | Operational filter exposure | Validation-script exposure | Verdict |
|---|---|---|---|
| L1 EUROPE_FLOW | None — `ORB_G5` reads `orb_{s}_size` | None — PR #71's `audit_l1_orb_g5_arithmetic_only_check.py` is clean | **NO EXPOSURE** |
| L2 SINGAPORE_OPEN | None — `ATR_P50` reads `atr_20_pct` | PR #74 audit (separate; see its own pre-reg) | **NO EXPOSURE** from this audit |
| L3 COMEX_SETTLE | None — `ORB_G5` reads `orb_{s}_size` | None — PR #71 used the same clean script as L1 | **NO EXPOSURE** |
| L4 NYSE_OPEN | None — `COST_LT12` reads `cost_risk_pct` | Cost-gate by construction | **NO EXPOSURE** |
| L5 US_DATA_1000 | None — `ORB_G5` reads `orb_{s}_size` | PR #71 flagged underpowered (N_nofire=4); separate issue | **NO EXPOSURE from break-bar** |
| L6 TOKYO_OPEN | None — `COST_LT12` reads `cost_risk_pct` | Cost-gate by construction | **NO EXPOSURE** |

### Self-retraction — earlier draft error

An earlier draft of this § 3b asserted L3 had **HIGH EXPOSURE** because two contaminated scripts (`garch_comex_settle_institutional_battery.py`, `audit_comex_settle_orb_g5_failure_pocket.py`) appeared in docs cited in L3's lineage. That claim was built on **citation proximity, not dependency**. On verification:

- `ORB_G5` is a canonical size filter (`trading_app/config.py:2740`). It has no break-bar pathway by construction.
- PR #71's L3 validation executed through `audit_l1_orb_g5_arithmetic_only_check.py`, not through either of the scripts I flagged. My causal chain was also **backwards in time** — `audit_comex_settle_orb_g5_failure_pocket.py` (PR #73) landed **after** PR #71 and cannot be upstream of it.
- The flagged scripts may produce contaminated **side-analyses** on COMEX data (e.g., partner-state provenance, carry encoding, break-bar-segmented institutional batteries). Those side-analyses should still be regated or retired — but they do **not** define or validate any deployed lane.

The retraction is load-bearing: treating L3's `p=0.012` as "SUSPECT" based on citation proximity would have triggered a downstream rebase of the 2026-04-21 handover narrative on a non-finding. That is exactly the "storytelling bias" failure mode `quant-agent-identity.md § Seven Sins` warns against.

### Cite-chase — revised deliverable

No live deployed lane has operational-filter or canonical-validation exposure to the break-bar contamination surface. Contaminated side-analysis scripts remain in the § 5 work-item queue as **research-claim hygiene** items, not as capital-at-risk items.

## 4. Rerun recipe (for any CONTAMINATED / PARTIAL_CONTAMINATED script)

Per PR #67 § 8 Binding Decisions + `.claude/rules/backtesting-methodology.md § RULE 6.1`:

1. Identify E2 trades in the scan: `entry_model = 'E2'`.
2. For E2 trades, either:
   - **(a) Drop the scan** if the break-bar column was the primary edge claim (the claim is a look-ahead artefact), OR
   - **(b) Add trade-time knowability gate**: `entry_ts >= break_ts` — entry has to already know the break-bar is closed. This usually shrinks the sample materially, and the scan may become underpowered. Report effective N before and after.
3. Re-run statistics (WR, ExpR, p-value, bootstrap null) on the gated sample only.
4. Compare post-gate numbers vs pre-gate numbers. Delta is the look-ahead bias.
5. If the script produced a validated_setups insert or a canonical finding, **the original finding is marked `SUSPECT` until the gated rerun lands**. Do not propagate the original finding in new work.

## 5. Work-item list for next terminal

No P0 capital-at-risk item exists from this audit (see § 3b retraction). All remaining work is research-claim hygiene — regate or retire contaminated side-analyses so their outputs cannot be cited in future deployment proposals.

Each row is a discrete PR. Priority reflects **claim-load**: how many downstream docs cite the script's output.

| Priority | File | Work |
|---|---|---|
| P1 | `comprehensive_deployed_lane_scan.py` (#15) | Highest citation load (4 downstream docs). Rerun with canonical `StrategyFilter` delegation per `research-truth-protocol.md § Canonical filter delegation`. |
| P1 | `garch_comex_settle_institutional_battery.py` (B7) | Institutional battery cited in 3 docs. Gated rerun or retirement. Coordinate with Codex (active COMEX research). |
| P1 | `audit_comex_settle_orb_g5_failure_pocket.py` (B10, PR #73, Codex) | Defer to Codex — they own this script's lineage. Audit needs their read of whether `break_dir` usage is filter-predicate or descriptive-segmentation. |
| P1 | `break_delay_filtered.py` (#17) | Archetype leak — E2 + `break_bar_continues=true AND break_delay_min IS NOT NULL`. Rerun with gate or retire. |
| P1 | `break_delay_nuggets.py` (#16) | Full E2 scan on `break_delay_min` bins. Rerun or retire. |
| P1 | `shadow_htf_mes_europe_flow_long_skip.py` (#2) | Shadow ledger — reframe predicate to pre-break-safe direction proxy or drop shadow. |
| P1 | `close_h2_book_path_c.py` (B1) | H2 book path C — regate. |
| P1 | `carry_encoding_exploration.py` (B2) | 13 break_dir refs + E2 — regate and re-examine carry-encoding findings. |
| P1 | `garch_all_findings_consolidated_stress.py` (B3) | Stress consolidation — break-bar-surviving findings suspect. |
| P1 | `garch_all_sessions_universality.py` (B5) | Universality claim across sessions — gated rerun. |
| P1 | `research_universal_hypothesis_pool.py` (#8) | Pool-scoped; bifurcate E2 vs E1/E3 cells. |
| P1 | `t0_t8_audit_volume_cells.py` (#4) | T0-T8 volume-cell decisions — cite-chase downstream. |
| P1 | `volume_confluence_scan.py` (#3) | Composite anchor/partner features — break-bar branches are banned for E2. |
| P2 | B4, B6, B8, B9 (`garch_*` PARTIAL) | Per-row emission: split output by entry_model; E1/E3 rows unchanged, E2 rows regated. |
| P2 | #5, #6, #9, #10 (`update_forward_gate_tracker`, `research_wf_stress_keepers`, two `shinies` presets/overlays) | Per-row; E2 preset rows regated, E1/E3 unchanged. |
| P3 | `q1_h04_mechanism_shape_validation_v1.py` (#1) | Read the `delay_col` downstream usage and reclassify (MANUAL_REVIEW). |
| P3 | `research/archive/research_nested_orb_stacking.py` | Archive file with E2 + break_dir predicate. Low urgency unless resurrected. |
| — | `research_mgc_e2_microstructure_pilot.py` (#13) | **No rerun needed**. Internal break_dir + descriptive break_delay_min in SELECT. |
| — | #7, #11, #12, #14 (SAFE, E0/E1 only, verified § 1a) | **No rerun needed**. |

## 7. What was NOT done in this PR

- No script was edited.
- No `validated_setups` row was touched.
- No claim in `HANDOFF.md` or `memory/*` was revised — this audit produces a surface map + a capital-exposure assessment, not a new edge finding.
- No `research_provenance` annotations were inserted or updated in code.
- The § 5.4 pre-commit static check (research scripts must carry `# E2-SAFE:` / `# E1/E3-ONLY:` comment near any `break_*` reference) is NOT implemented here — it is a separate hardening PR.
- L3's gated-rerun is NOT executed in this PR. It is the P0 next step.

## 8. Provenance

- Classification pass: 2026-04-22, `research/e2-break-bar-retro-audit` branch.
- Evidence base: `grep -n -B2 -A2 "break_delay_min|break_bar_continues|break_bar_volume|break_dir|break_ts"` + per-file `entry_model` + `'E2'` / `'E1'` / `'E3'` literal context.
- No database queries were run for this audit — this is purely a source-code surface map.
- Drift check: `pipeline/check_drift.py` → 106/106 PASSED on `origin/main` (branch base).

---

## 9. Top-line institutional summary

- **Surface mapped:** 17 § 5.1 scripts + 10 bonus-surface scripts + 1 archive script. Total 28 retro-audit items.
- **CONTAMINATED:** 12 (7 original + 5 bonus). **PARTIAL_CONTAMINATED:** 7 (3 original + 4 bonus). **SAFE:** 5 verified + 1 internal-break-dir + 1 descriptive-SELECT-only. **MANUAL_REVIEW:** 1.
- **Deployed-book exposure:** **NONE**. Filter-definition verification (§ 3b) confirms all 6 live lanes use canonical size / vol-pct / cost filters (`trading_app/config.py:2740` for ORB_G5) — no deployed lane's filter or validation consumes a banned break-bar column. PR #71's L3 `p=0.012` was computed by `audit_l1_orb_g5_arithmetic_only_check.py`, which is clean and cites `config.py:2740` as its canonical source.
- **Self-retraction (§ 3b):** an earlier draft of this audit asserted HIGH EXPOSURE for L3 based on citation proximity between contaminated scripts and L3-adjacent docs. On filter-definition + time-ordering verification, that claim was wrong: `ORB_G5` is a canonical size filter, PR #71's validation path does not touch any contaminated script, and `audit_comex_settle_orb_g5_failure_pocket.py` (PR #73) is downstream of PR #71 by merge order — it could not have been upstream evidence for PR #71's `p=0.012`.
- **Where edge actually lives (if anywhere):** For research-claim hygiene: not in break-bar columns — they are post-break and tautologically "predictive" for E2. Candidates for real E2 edge live in upstream variables (pre-break order-flow, pre-break VWAP position relative to ORB midpoint, pre-break vol-regime compression). Retrofit any contaminated scan with those upstream variables as the predictor, not break-bar columns.
- **Highest EV next move:** research-claim hygiene, not capital-at-risk. Priority-ordered in § 5 by downstream citation load, starting with `comprehensive_deployed_lane_scan.py` (4 citing docs) and `garch_comex_settle_institutional_battery.py` (3 citing docs). Gated reruns use the canonical `StrategyFilter` delegation per `research-truth-protocol.md § Canonical filter delegation`, not inline SQL.

**Load-bearing reminder:** when regating any contaminated script, use `research.filter_utils.filter_signal` to delegate to `trading_app.config.ALL_FILTERS[key].matches_df`. Do NOT re-encode filter logic. Parallel filter implementations WILL drift (as caught 2026-04-18 by the VWAP scan code review).

---

## 10. Process self-audit (institutional discipline)

This PR was re-evaluated under `quant-audit-protocol.md` self-audit framework BEFORE first push and AGAIN after the first draft of § 3b. Both iterations are preserved in the commit history. The self-retraction in § 3b landed because:

1. The first draft confused **citation proximity** with **dependency**.
2. Verification against canonical code (`trading_app/config.py:2740`) and PR #71's actual computation path (`audit_l1_orb_g5_arithmetic_only_check.py`) took ~2 minutes and was skipped in the first draft.
3. Time-ordering between PR #71 (earlier) and PR #73 (later) was inverted.

This audit now complies with `integrity-guardian.md § 5 Evidence Over Assertion — Generation Is Not Validation`: every verdict in § 3b cites a canonical file + line, not a narrative inference.
