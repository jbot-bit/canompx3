---
paths:
  - "research/**"
  - "trading_app/strategy_*"
  - "trading_app/holdout_policy.py"
  - "pipeline/build_daily_features.py"
  - "pipeline/cost_model.py"
  - "docs/audit/**"
  - "docs/institutional/**"
---

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

**3.4 Router/allocator hypotheses require multi-fold evidence.** Single-fold walk-forward (e.g., 50/50 chronological split) is INSUFFICIENT for any router, allocator, session-selector, or regime-conditional signal. Require EITHER (a) ≥3 rolling folds of annual test periods with ≥3 wins and mean ΔSR_ann ≥ +0.30, OR (b) combinatorial-purged CV (LdP 2020 Ch 8) with embargo matching the signal's decision horizon. Single-fold WF remains acceptable for stationary-signal discovery on deployed-lane overlays; NOT for signals whose map can drift across vol regimes. See failure-log entry 2026-04-21 (ovnrng-router-rolling-cv, commit `4dfd3000`) for the motivating incident.

**3.5 Post-hoc criterion creep is post-hoc REJECTION (mirror of post-hoc rescue).** If a new gate (Sharpe threshold, Spearman p, bootstrap CI, etc.) is discovered MID-AUDIT after the pre-reg's locked gate has already been evaluated, the correct institutional response is: (a) the original pre-reg verdict STANDS on its locked gate; (b) the new criterion becomes its own pre-registered follow-on test on fresh data. NEVER retroactively invalidate a legitimate pre-reg pass by applying new criteria to the same data. This is structurally identical to post-hoc rescue (adding new gates to save a failing pre-reg) and violates Bailey-LdP 2014 §3 / Harvey-Liu 2015 §2. See failure-log entry 2026-04-21 (PR #59 sizer re-audit and PR #51 DSR audit both exhibited this pathology).

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

**Moved to `.claude/rules/backtesting-methodology-failure-log.md`** (self-scoped frontmatter — only auto-injects when that file is edited; `Read` on demand otherwise).

Read that file when appending a new entry or investigating a past incident. Every fresh backtesting failure still gets logged there, cited by date slug.

Append new entries to the companion file with: date slug + one-line lesson + canonical reproduction path.

