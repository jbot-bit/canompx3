# Backtesting Methodology — Mandatory Rules

**Authority:** governs every backtest, every discovery scan, every research-claim test. Complementary to `quant-audit-protocol.md`, `research-truth-protocol.md`, `RESEARCH_RULES.md`, `docs/institutional/pre_registered_criteria.md`.

**Origin:** Learned the hard way 2026-04-15 — institutional-grade scans produced 176 false BH-FDR survivors from look-ahead bias until feature-temporal gates were installed, then produced only 13 honest global survivors. See `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`.

**These rules are NOT optional. Skipping any one produces either fake edge or missed edge.**

---

## RULE 1: Feature temporal alignment — the single biggest source of silent bias

Every feature you use in a backtest has a **validity domain** — the set of ORB sessions / decision-points where that feature is KNOWABLE without reading future bars.

### 1.1 Hard-coded look-ahead features (BANNED at entry time)

From `.claude/rules/quant-audit-protocol.md` KNOWN FAILURE PATTERNS:

| Feature | Why banned |
|---------|-----------|
| `double_break` | look-ahead (scans full session AFTER entry) |
| `mae_r`, `mfe_r` | intra-trade retrospective |
| `outcome`, `pnl_r` | post-trade |

### 1.2 Window-derived features with CONDITIONAL validity

These features are computed over a time window. They LEAK FUTURE DATA for ORB sessions that fire DURING or BEFORE the window closes.

**Reference:** `pipeline/build_daily_features.py:445-454` (overnight_* warning) and `SESSION_WINDOWS` at line 83-87.

**Valid-domain gate table** (Brisbane local times):

| Feature class | Source window | VALID ONLY for ORB starting ≥ |
|---------------|---------------|-------------------------------|
| `overnight_high/low/range/range_pct` | 09:00-17:00 Brisbane | 17:00 (LONDON_METALS and later) |
| `overnight_took_pdh`, `overnight_took_pdl` | 09:00-17:00 Brisbane | 17:00 |
| `session_asia_high/low` | 09:00-17:00 Brisbane | 17:00 |
| `session_london_high/low` | 18:00-23:00 Brisbane | 23:00 (US_DATA_830 and later) |
| `session_ny_high/low` | 23:00-02:00 Brisbane (crosses midnight) | 02:00 next day (COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE) |
| `pre_1000_high/low`, `took_pdh_before_1000`, `took_pdl_before_1000` | start of day to 10:00 | 10:00 — TOKYO_OPEN only |

**ORB start times (Brisbane, reference):**
CME_REOPEN ~08:00, BRISBANE_0925 09:25, TOKYO_OPEN 10:00, BRISBANE_1025 10:25, SINGAPORE_OPEN 11:00, LONDON_METALS 17:00, EUROPE_FLOW 18:00, BRISBANE_1955 19:55, US_DATA_830 23:30, NYSE_OPEN 00:30 (next day), US_DATA_1000 01:00, COMEX_SETTLE 04:30, CME_PRECLOSE 06:00, NYSE_CLOSE 07:00.

### 1.3 Implementation

Every new scan must implement `_valid_session_features(orb_session)` and `_overnight_lookhead_clean(orb_session)` guards.

Canonical implementation: `research/comprehensive_deployed_lane_scan.py::_valid_session_features()` and `::_overnight_lookhead_clean()`. **Copy these or import; do not re-derive.**

### 1.4 Smell test

After any scan, inspect top survivors:
- If every top survivor uses `session_{x}_*` or `overnight_*` features on early ORB sessions → look-ahead contamination.
- If |t| > 10 or Δ_IS > ±0.6 → investigate temporal alignment before celebrating.
- Real signals on 20+ years of clean data produce |t| 3-5, not |t| 15-17.

---

## RULE 2: Filter application timing — apply BEFORE the backtest, not after

When backtesting a strategy with a filter (e.g., ORB_G5 on MNQ EUROPE_FLOW):
1. Filter fires at **session start** using only info knowable before ORB ends.
2. If filter=pass → strategy takes the trade per entry rules.
3. If filter=fail → no trade that day (skipped in the P&L ledger).

**Overlay candidate testing (NEW feature on existing filtered lane):**
- **Pass 1 — unfiltered:** test new feature on full lane universe (all days, no deployed filter). Measures raw signal.
- **Pass 2 — filtered:** test new feature ONLY on days where deployed filter fires. Measures the overlay's residual edge after the filter.
- Valid overlay MUST pass both.

Canonical implementation: `research/comprehensive_deployed_lane_scan.py::test_cell(pass_type='unfiltered' | 'filtered')`.

---

## RULE 3: IS/OOS discipline — sacred holdout

- `HOLDOUT_SACRED_FROM = 2026-01-01` per `trading_app/holdout_policy.py`.
- IS window: `trading_day < 2026-01-01`.
- OOS window: `2026-01-01 ≤ trading_day < 2026-04-07` (or current cutoff).
- **Never tune parameters against OOS.** Never re-run with different thresholds to "rescue" an OOS failure.

### 3.1 Direction match (`dir_match`) requirement

- For OOS validation, `sign(delta_IS) == sign(delta_OOS)` must hold.
- A cell with delta_IS = +0.3 and delta_OOS = -0.1 FAILS dir_match regardless of significance.
- Report both deltas always. Never hide mismatch.

### 3.2 OOS sample size

- If `N_on_OOS < 5` → cannot compute delta_OOS reliably. Report as NaN, dir_match as False.
- If `N_on_OOS < 30` → statistical power on OOS is very low. Treat as directional-only evidence, not confirmatory.

---

## RULE 4: Multiple testing correction — K is per-family, not one number

Every cell has its own p, N, t. But "the K" depends on which hypothesis family the cell belongs to.

**Report survivors at MULTIPLE K framings. Don't pick one and call it done.**

| Framing | K definition | When this is the right cut |
|---------|-------------|-----------------------------|
| `K_global` | total cells in the scan | Big-picture single-scan claim |
| `K_family` | cells within feature family (volume, volatility, timing, ...) | Per-feature-class claim |
| `K_lane` | cells within (session, apt, rr, instrument) | Lane-specific overlay claim |
| `K_session` | cells within session across instruments | Session-level claim |
| `K_instrument` | cells within instrument across everything | Instrument-level claim |
| `K_feature` | cells within a single feature name across lanes | "Is this feature universal?" |

Canonical implementation: `research/comprehensive_deployed_lane_scan.py::bh_fdr_multi_framing()`.

### 4.1 Promotion gate

A cell is a legitimate discovery candidate if it passes BH-FDR at K_family OR K_lane AND meets:
- `dir_match == true`
- `|t| ≥ 3.0` (or Chordia strict `t ≥ 3.79` without prior theory)
- `N_on ≥ 50` (deployable sample power)
- Not flagged as `tautology` / `extreme_fire` / `arithmetic_only`

K_global at very large K (e.g., K>10,000) is strong confirmatory evidence but not required — the feature-family is the natural hypothesis unit per Harvey-Liu 2015.

### 4.2 MinBTL budget

Per Bailey et al 2013, `MinBTL = 2·ln(N_trials) / E[max_N]²`. With the 7+ year IS window, N_trials up to ~300 on clean MNQ data or ~2000 on proxy-extended. Pre-commit before running.

---

## RULE 5: Comprehensive scope — no hand-picking

A discovery scan enumerates ALL combos of its declared axes. Not a hand-picked subset.

### 5.1 Default axes for ORB discovery

- **Sessions:** all 12 active from `pipeline.dst.SESSION_CATALOG` — CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, EUROPE_FLOW, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE, CME_PRECLOSE, NYSE_CLOSE, BRISBANE_1025
- **Instruments:** all 3 active — MNQ, MES, MGC
- **Apertures:** 5, 15, 30 minutes
- **RRs:** 1.0, 1.5, 2.0

Full combo count: 12 × 3 × 3 × 3 = 324 lane-spec combos.

### 5.2 Data-availability auto-skips

Some combos have zero data (MGC on equity-hours sessions). Script naturally skips with `N=0` — log coverage: `combos_tested / combos_attempted`. Don't silently drop and claim full coverage.

### 5.3 Narrowing scope

Only allowed with explicit pre-registration justifying the restriction:
- Literature-grounded (e.g., Fitschen Ch 3 scope narrowing)
- Data-horizon restriction
- Deployment-matching (Alpha scope for overlay on deployed lanes)

**Violation caught 2026-04-15:** previous comprehensive scan hand-picked 29 of 324 lane combos with no pre-reg justification. Missed ~295 combos. Fixed.

---

## RULE 6: Trade-time knowability — feature audit before use

Before adding any feature to a backtest, answer in writing:
1. At what Brisbane time is this feature's underlying data FIRST COMPLETE?
2. Does every ORB session it's applied to fire AFTER that time?
3. Is the feature computation pure (no post-entry data)?

If you can't answer (1)-(3) with certainty, do NOT use the feature. Write a provenance note in the scan script.

### 6.1 Safe (always trade-time-knowable) features

- `atr_20`, `atr_20_pct`, `atr_vel_ratio`, `atr_vel_regime` (20-day rolling, prior close)
- `prev_day_high`, `prev_day_low`, `prev_day_close`, `prev_day_range`, `prev_day_direction` (yesterday's close — always prior)
- `gap_open_points`, `gap_type` (known at session open)
- `garch_forecast_vol`, `garch_atr_ratio`, `garch_forecast_vol_pct` (forecast made at prior close)
- `is_nfp_day`, `is_opex_day`, `is_friday`, `is_monday`, `day_of_week` (calendar, priori)
- `orb_{s}_size`, `orb_{s}_high`, `orb_{s}_low`, `orb_{s}_break_dir` (known at ORB end, before entry for E2)
- `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` (known at break-bar close, before E2 CB1 entry)
- `orb_{s}_vwap`, `orb_{s}_pre_velocity` (computed over pre-ORB interval)
- `orb_{s}_compression_z`, `orb_{s}_compression_tier` (computed pre-ORB)
- `rel_vol_{s}` (ORB volume vs session-avg historical — known at ORB end)

### 6.2 Conditionally valid (use gate)

- `session_asia_*`, `session_london_*`, `session_ny_*` — RULE 1.2 gate required
- `overnight_*` — RULE 1.2 gate required
- `pre_1000_*`, `took_pdh_before_1000`, `took_pdl_before_1000` — TOKYO_OPEN only (window ends at 10:00 before ORB)

### 6.3 Banned (look-ahead)

- `double_break`, `*_mae_r`, `*_mfe_r`, `*_outcome`, `pnl_r`, any `*_fill_price`-derived feature used as a PREDICTOR

---

## RULE 7: Tautology check (T0)

Before claiming a new feature as additive to a deployed filter, compute:
`|corr(new_feature_fire, deployed_filter_fire)| > 0.70` → flag **TAUTOLOGY**, exclude from survivors.

Known tautology example: `cost_risk_pct` ∝ `1 / orb_size_pts` — perfect inverse correlation with ORB_G5.

Canonical implementation: `research/comprehensive_deployed_lane_scan.py::t0_correlation()`.

---

## RULE 8: Fire-rate and ARITHMETIC_ONLY guards

### 8.1 Extreme fire rate

- `fire_rate < 5%` or `fire_rate > 95%` → flag `extreme_fire`, exclude from survivors.
- Exception: pre-registered rare-event features (NFP days) where rare firing is the point. In that case Chordia-strict `t ≥ 3.79` applies.

### 8.2 ARITHMETIC_ONLY

- `|wr_spread| < 3%` AND `|Δ_IS| > 0.10` → flag `arithmetic_only`.
- This is a cost-screen (bigger trades net more per win, not a better WR predictor). Deploy as cost-gate class, not "edge."

---

## RULE 9: Data source discipline

- Canonical layers ONLY: `bars_1m`, `daily_features`, `orb_outcomes`.
- Banned for discovery: `validated_setups`, `edge_families`, `live_config`, any LLM / doc-derived claim.
- **Join rule:** `daily_features` has 3 rows per (trading_day, symbol) — one per `orb_minutes`. Triple-join required (see `daily-features-joins.md`).
- **CTE guard:** when reading non-ORB-specific columns from `daily_features` in a CTE, add `WHERE orb_minutes = 5` to deduplicate. Missing this creates 3x N inflation and sqrt(3)=1.73x t-stat inflation.

---

## RULE 10: Pre-registration before discovery

Any scan that writes to `experimental_strategies` or `validated_setups` must have a pre-reg file at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml|.md` BEFORE running. File must include:
- Numbered hypotheses with economic-theory citations
- Exact filter / feature dimensions + threshold ranges
- K_budget (pre-committed)
- Kill criteria
- Expected N per cell

Confirmatory audits (T0-T8 on prior survivors) do NOT require new pre-reg — they're not new discovery.

Full requirement: see `.claude/rules/research-truth-protocol.md` § Phase 0 Literature Grounding.

---

## RULE 11: Audit trail

After every scan:
- Commit research code to `research/`
- Commit results MD to `docs/audit/results/`
- If pre-registered: commit pre-reg to `docs/audit/hypotheses/`
- Never delete prior results — append, supersede via new docs that reference prior
- Add a one-line pointer in `MEMORY.md` user-memory index

---

## RULE 12: Red flags — stop and investigate

Any of these means STOP before celebrating:

- |t| > 7 on a feature you haven't seen in prior literature
- Δ_IS > 0.6 R per trade
- Every top survivor references the same feature class (possible pipeline artifact)
- Every top survivor references window-derived features on early ORB sessions (look-ahead)
- BH_global passes but BH_family fails (spurious global)
- OOS direction flips from IS direction
- Fire rate < 5% or > 95% (noise or constant)
- Signal disappears when a control variable is added

---

## RULE 13: Pressure-test every scan

For any new scan: deliberately introduce a known-bad feature (e.g., `orb_{s}_outcome` = look-ahead) and confirm the script flags or filters it. If the pressure test passes through silently, fix the guard before trusting the scan.

Canonical check: run the scan, then re-run with a `mae_r` or `outcome` column injected as a "feature" — it should fail T0 tautology (perfect correlation with pnl_r) or be rejected upstream.

---

## Historical failure log

Keep this list updated with real examples of what went wrong:

- **2026-04-15: DSR over-weighted as hard gate (v1→v2 self-correction).** Initial stress test of rel_vol_HIGH finding used `var_sr=0.047` (default from `dsr.py` docstring, calibrated for `experimental_strategies`) on comprehensive-scan cells. Wrong population. v2 empirical calibration yielded var_sr=0.012 (3.8× smaller). Also K=14,261 was overly punishing — dsr.py line 35 says "DSR is INFORMATIONAL, not a hard gate" because N_eff unknown. v2 reports DSR at 7 N_eff framings (K=5/12/36/72/300/900/14261) and uses 4 CORE hard gates (bootstrap, temporal, exceeds_max_t, per_day) as primary verdict. Lesson: **always verify var_sr from the actual scan cell population; always report DSR at multiple N_eff; don't treat DSR as hard gate**. See `docs/audit/results/2026-04-15-rel-vol-stress-test-v2.md` + `research/stress_test_rel_vol_finding_v2.py`.
- **2026-04-15: Block bootstrap preserved joint structure (bug caught same session).** v1 bootstrap resampled index and applied to BOTH pnl and mask — preserved joint (pnl, mask) structure, mechanically producing p~0.5 regardless of real signal. Fixed: resample pnl via blocks preserving autocorrelation, keep mask FIXED to break signal-outcome link. Proper moving-block bootstrap per Lahiri / Politis-Romano. After fix, bootstrap p=0.0005 on all 5 lanes (genuine signal evidence). See `research/stress_test_rel_vol_finding.py::block_bootstrap_p`.
- **2026-04-15: Aronson Ch 6 cited for volume — WRONG.** Aronson Ch 6 is "Data-Mining Bias," not volume. Two docs had this error (edge-finding-playbook, volume-exploitation-plan). Ch 7 is "Theories of Nonrandom Price Motion" (EMH literature) — doesn't directly support volume-as-confirmation either. Volume-as-confirmation remains TRAINING MEMORY ONLY until Harris/O'Hara acquisition. Fixed both docs; quant-audit-protocol's Aronson Ch 6 reference (for data-mining/overfitting) remains CORRECT.
- **2026-04-15: day_type is LOOK-AHEAD.** `pipeline/build_daily_features.py:510` explicit warning: uses full-day OHLC. Never use as intraday feature. Caught during mechanism-decomposition audit — removed from all stress tests.
- **2026-04-15: T8 cross-instrument twin wrong for MGC.** `t0_t8_audit_prior_day_patterns.py::t8_cross_instrument` defaults to MNQ as MGC's twin. MNQ is equity index, MGC is gold — wrong asset class. Flag any MGC T8 result as methodologically suspect until twin logic fixed (proper: GC parent or SIL same-class if activated).
- **2026-04-15: Look-ahead via session_* + overnight_* features.** First comprehensive scan produced 176 strict survivors, all contaminated by `session_{asia/london/ny}_*` and `overnight_*` features used on ORB sessions firing DURING the reference window. After RULE 1 gates installed, reduced to 19 real survivors. See `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`.
- **2026-04-15: Scope hand-picking.** Initial comprehensive scan tested 29 of 324 lane combos. Missed ~295 combos. Full re-run in same session produced 14,261 cells, 13 BH-global survivors (vs 3 in narrower scan).
- **2026-04: CTE N-inflation** caught in `verify_external_ibs_nr7.py` — t=6.78 dropped to t=3.89 after adding `orb_minutes=5` to CTE. Commit `94546ccf`. Codified in `daily-features-joins.md`.
- **2026-03-24: cost_risk_pct tautology** — |corr| = 1.0 with 1/orb_size_pts. Filed under `quant-audit-protocol.md` KNOWN FAILURE PATTERNS.
- **2026-02: double_break lookahead** — scanned full session after entry. Banned.
- **2026-04-08: E2 canonical window fix** — divergence between backtest and live execution ORB-window calculation was a look-ahead bias risk per Chan Ch 1. Fixed via single canonical `pipeline.dst.orb_utc_window()`. See `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.
- **2026-04-08: Phase 3c rebuild of orb_outcomes** — FK child-DELETE invisible in same transaction requires 2-transaction split. `docs/handoffs/2026-04-08-phase-3c-rebuild-handover.md`.
- **2026-04-19: Cross-scan "replication" without overlap decomposition.** HTF Path A prev-month v1 result doc claimed MES EUROPE_FLOW long RR2.0 wrong-sign pattern "replicated" prev-week v1. Post-commit adversarial audit showed 42.8% of prev-month trade-fires on that lane are already prev-week trade-fires; non-overlap (PM-only) subset collapses from combined t=-3.53 to t=-1.38 (N=195, raw p=0.168). In the same audit, MES TOKYO_OPEN long RR2.0 was the OPPOSITE pattern — overlap t=-1.86 (n.s.), non-overlap t=-3.37 (N=188, p=0.0009), so prev-month contributes genuinely new information there. **Lesson:** two scans over the same feature family can agree from redundancy rather than independence. Always decompose `(scan_A ∧ scan_B)` vs `(scan_B ∧ ¬scan_A)` on per-day fire masks before claiming replication or treating cross-family agreement as additional evidence. Canonical reproduction: `research/htf_path_a_overlap_decomposition.py` → `docs/audit/results/2026-04-19-htf-path-a-overlap-decomposition.md`. Result doc corrected at `docs/audit/results/2026-04-18-htf-path-a-prev-month-v1-scan.md` § "Adversarial-audit addendum — 2026-04-19".
- **2026-04-19: Cross-instrument same-session redundancy (same class as above, different surface).** Comprehensive scan (`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`) reported 5 non-twin `rel_vol_HIGH_Q3` cells at K_global=14,261 BH-global, framed as "5 independent BH-global survivors" / "universal volume confirmation." Cross-lane overlap decomposition on 2026-04-19 found MES COMEX_SETTLE short and MNQ COMEX_SETTLE short have Jaccard 0.491 — 67% overlap-of-min on fire days. MES and MNQ are near-identical equity-index drivers on the same session; their `rel_vol_HIGH_Q3` fires coincide on most days. Effective K_eff ≈ 4, not 5. Nyholt 2004 Meff (4.82) is misleadingly close to 5 because the union-grid correlation dilutes with zero-zero cross-instrument days; Jaccard is the canonical set-overlap read. **Lesson:** when collecting BH-global survivors across instruments, always pairwise-decompose `same-session same-direction cross-instrument pairs` — same underlying economic driver → dependent tests. Canonical reproduction: `research/rel_vol_cross_scan_overlap_decomposition.py` → `docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition.md`. Follow-up pointer appended to comprehensive scan result doc header.
- **2026-04-19: Research filter delegation drift — compute_deployed_filter mis-implemented OVNRNG_100 as ratio.** `research/comprehensive_deployed_lane_scan.py::compute_deployed_filter` re-implemented four canonical filters (OVNRNG_100, VWAP_MID_ALIGNED, ORB_G5, ATR_P50) inline rather than delegating to `research.filter_utils.filter_signal`. The OVNRNG_100 re-implementation was `overnight_range / atr_20 >= 1.0` — a ratio gate — when canonical `OvernightRangeAbsFilter(min_range=100.0)` is absolute `overnight_range >= 100.0`. Verification on MNQ COMEX_SETTLE O5 RR1.5: old ratio fired 25/1698 rows (1.5%); canonical absolute fires 579/1698 rows (34.1%). A 20× gap. Every prior scan output using `compute_deployed_filter(..., "OVNRNG_100")` tested overlays on a near-empty population. Fix: delegate to `filter_signal` per research-truth-protocol.md § Canonical filter delegation. WARN header added to `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` identifying affected "deployed" scope survivor cells. **Lesson:** pre-2026-04-18 research scripts that look inline-correct for canonical filter logic are suspect by default — only the canonical `filter_signal` delegation path is trustworthy. Full filter-delegation audit across 266 research/ files at `docs/audit/results/2026-04-19-research-filter-delegation-audit.md`.
- **2026-04-19: Mode-B grandfathered validated_setups ExpR drift — 38/38 active lanes affected.** Canonical Mode A re-validation of every active `validated_setups` row (script `research/mode_a_revalidation_active_setups.py`) found ALL 38 lanes have material drift from stored `expectancy_r` values under strict Mode A (`trading_day < 2026-01-01`). Mode A N is consistently ~55% of stored N — the gap being 2026 Q1 data that was eligible under Mode B but is sacred OOS under Mode A. ExpR pattern is mixed (some lanes UP, some DOWN). Worst Mode A drops: MNQ EUROPE_FLOW OVNRNG_100 RR1.0 (0.118→0.056), MNQ NYSE_OPEN X_MES_ATR60 RR1.5 (0.132→0.066). **Lesson:** `validated_setups.expectancy_r` for ANY lane with `last_trade_day >= 2026-01-01` is Mode-B grandfathered and must not be cited as a Mode A baseline. Always recompute from canonical `orb_outcomes` + `daily_features` under `trading_day < HOLDOUT_SACRED_FROM`. Full errata at `docs/audit/results/2026-04-19-mode-a-revalidation-of-active-setups.md`.
- **2026-04-19: Quantile-over-full-sample as feature look-ahead (new class, RULE 1 sub-clause).** Feature helpers like `bucket_high(vals, 67)` compute a percentile threshold on the full cell distribution (IS + OOS combined) and then apply it to gate IS fires. IS fire status therefore depends on OOS distribution — a subtle look-ahead at the feature-construction level (distinct from the temporal-window look-ahead that RULE 1.2 covers). Specific instance: `research/comprehensive_deployed_lane_scan.py::bucket_high` and `bucket_low`. Sensitivity check on 2026-04-19 (`research/rel_vol_cross_scan_overlap_decomposition.py --quantile-method is_only`) found the specific `rel_vol_HIGH_Q3` 5-survivor finding is ROBUST under IS-only-quantile correction (Meff = 4.817, max Jaccard = 0.491 — identical to 3 decimals with full-sample). But the methodology itself should be corrected going forward; any new percentile-binned feature must compute its threshold on IS-only data. **RULE 1 addendum:** feature helpers that use `np.nanpercentile` or `pd.quantile` on a cell's full-sample data are conditionally valid — only valid when the quantile is computed on IS subset and then applied to all rows. A feature built with a full-sample quantile and then tested against an IS-only null contains look-ahead.
- **2026-04-19: Lift-vs-unfiltered framing without significance test on the subset.** Phase 2.4 third-pass adversarial reframe (`docs/audit/results/2026-04-19-phase-2-4-cross-session-momentum-mode-a.md`) flagged MES EUROPE_FLOW composite `ORB_G5 AND CROSS_SGP_MOMENTUM` as a "golden egg" based on +0.20 R/trade lift vs unfiltered baseline (unfiltered ExpR=-0.16 N=850 → composite ExpR=+0.046 N=112). Full C1-C12 audit (`docs/audit/results/2026-04-19-mes-europe-flow-g5-sgp-composite-audit.md`) shows composite FAILS C3 (p=0.668) and C4 (t=+0.43) — the composite subset's own ExpR is noise-indistinguishable from zero. The apparent "lift" came from comparing two disjoint populations (composite fires on 112 days; unfiltered is 850 days; they're NOT the same universe). The relevant question is "does the composite have non-zero ExpR?" — not "does it have higher ExpR than trading every break-day?" **Lesson:** a high "lift vs noise-baseline" delta is not evidence when the post-filter subset's ExpR is not significantly different from zero. Always compute Chordia t on the subset itself BEFORE reporting lift-vs-unfiltered. **RULE 8 addendum:** adopt a new RULE 8.3 ARITHMETIC_LIFT — any `lift_vs_unfiltered > 0.10 R` reported without a subset-level t-stat accompanying it should be flagged pending significance verification on the subset. Canonical reproduction: `research/phase_2_4_mes_composite_c1_c12_audit.py` + pre-reg `docs/audit/hypotheses/2026-04-19-mes-europe-flow-g5-sgp-composite-v1.yaml`.

Append to this list when new failure modes are caught.
