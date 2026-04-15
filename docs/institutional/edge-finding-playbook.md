# Edge-Finding Playbook — Institutional-Grade Niche Hunting

**Date authored:** 2026-04-15
**Authority:** Complementary to `.claude/rules/backtesting-methodology.md`, `docs/institutional/mechanism_priors.md`, `docs/institutional/pre_registered_criteria.md`.
**Purpose:** Meta-playbook for finding new tradeable edges in the future. Codifies the lessons learned from the 2026-04-15 volume-signal discovery session.

---

## The 12 commandments of edge finding

### 1. ALWAYS scope comprehensively first, narrow second

Enumerate ALL axes (sessions × instruments × apertures × RRs × directions × features). Then filter to what's relevant. **Never** hand-pick a subset and assume it's representative.

**Violation example (caught 2026-04-15):** tested 29 hand-picked lanes instead of all 324. Missed ~295 combos until the comprehensive rerun found the volume universe.

Default scope for ORB discovery: 12 sessions × 3 instruments × 3 apertures × 3 RRs = 324 combos.

### 2. Multi-framing BH-FDR — K is per-family, not one number

Each cell has its own p, N, t. But "the K" depends on which hypothesis family it belongs to.

Report survivors at 6 framings: K_global, K_family, K_lane, K_session, K_instrument, K_feature. Cells that don't pass global may pass at K_family = natural hypothesis boundary.

**Example:** in the 2026-04-15 comprehensive scan, 13 passed K_global (K=14,261) but 127 passed K_family (avg K~2,500). The per-family survivors are still legitimate — Harvey-Liu 2015 treats feature families as the natural hypothesis unit.

### 3. Look-ahead is the biggest silent killer

Every feature has a **validity domain** — the ORB sessions where it's KNOWABLE without reading future bars. Window-derived features (`session_*`, `overnight_*`, `pre_1000_*`) leak future data for ORB sessions firing during or before the window closes.

**Mandatory:** gate every time-windowed feature with `_valid_session_features()` / `_overnight_lookhead_clean()` before using in backtest.

**Smell test:** if top survivors have |t| > 10 or Δ_IS > ±0.6, assume look-ahead until proven otherwise. Real edges on 20+ years of data produce |t| 3-5, not blowouts.

### 4. Single-factor UNIVERSAL beats multi-factor CONFLUENCE

Confluence AND-gates sound appealing but suffer from:
- K explosion (N features → N² combinations)
- Fire rate shrinkage (AND reduces both signal firing rates)
- OOS power collapse (rare fires × thin OOS window = unreliable validation)
- Marginal additional info often < 30% (composite-vs-anchor correlation 0.5-0.7)

**From 2026-04-15 volume confluence scan:** only 1 of 432 cells was strict dir-match survivor. Single-factor `rel_vol_HIGH_Q3` already captured ~80% of the available edge.

**Rule:** prove single-factor universal FIRST. Test confluence only when single factor needs boosting.

### 5. Mechanism > mathematical survivor

A cell that passes BH-FDR but has no mechanism is likely overfitting. Every survivor should have a testable "why" rooted in market structure. Volume finding has clean mechanism: **institutional participation confirmation** (Aronson Ch 6, Carver Ch 9-10, order-flow literature). Level-rejection finding has clean mechanism: **limit orders defending key prior levels** (Dalton, Murphy).

No mechanism + BH survivor = investigate harder OR demote to "monitoring only."

### 6. Direction asymmetry hunting — LONG vs SHORT split

Real markets are asymmetric. Shorts often behave differently from longs (stop-hunt direction, liquidity patterns, retail vs institutional positioning).

**2026-04-15 finding:** 4 of 5 BH-global volume survivors are SHORT direction. That's a real asymmetry worth testing separately. Always report direction-split results.

### 7. Cross-instrument concordance is cheap expansion

If MNQ finds a signal on SESSION X, test MES and MGC on same session. If it holds, you've added a lane (subject to correlation gate). If it doesn't, you've learned the mechanism is instrument-specific.

**Cost:** ~1/3 of original scan. **Benefit:** potentially 2x portfolio diversification.

### 8. OOS window thinness is a structural constraint

Our sacred 2026 OOS is currently 3 months. For features with 15-25% fire rate, that's N_OOS_on ~10-15. WFE ratios become unstable below N=10; T3 gates fail even for real signals.

**Handling:** when T3 fails but T0/T1/T2/T6/T7/T8 all pass, the cell is likely real but data-limited. Deploy SIGNAL-ONLY for 6-12 weeks to accumulate OOS, then re-audit. Don't prematurely kill.

### 9. Exploration vs discovery — different compliance levels

- **Exploratory stress test** (e.g., cross-factor confluence on already-found signal): no YAML pre-reg needed. Findings are informational, don't write to `validated_setups`.
- **Discovery scan** (new single-feature hypothesis writing to `experimental_strategies`/`validated_setups`): full YAML pre-reg with MinBTL, K_budget, kill criteria, literature grounding.

Mis-labeling exploration as discovery = compliance overhead without benefit. Mis-labeling discovery as exploration = pre-registration violation.

### 10. Pressure-test every scan

Deliberately inject a known-bad feature (e.g., `outcome` column, `mae_r`) and confirm your scan flags or rejects it. If it passes silently, your gates are broken. Fix BEFORE trusting any findings.

Canonical self-test: inject `outcome = 'win'` as a "feature" — should fail T0 tautology (perfect correlation with pnl_r > 0) or be rejected as look-ahead.

### 11. Commit everything — including negative results

Dead patterns are institutional memory. `docs/audit/results/` should contain every scan, every T0-T8, every pre-reg, succeeded or failed. Future sessions avoid re-testing dead patterns.

**Example:** 2026-04-15 adversarial FADE scan produced DEAD verdict. Committed. Now future work starts knowing direction-flip is a confirmed dead path.

### 12. Institutional rigor = audit every claim

Before ANY deployment recommendation:
- T0-T8 battery per `quant-audit-protocol.md`
- 12-criterion gate per `pre_registered_criteria.md`
- Signal-only shadow ≥ 2 weeks
- Explicit user approval

No claim survives memory unless it's backed by committed execution output.

---

## The niche-hunting priority ladder (ranked ROI)

For future sessions, approach new-edge search in this order:

### Tier 1 — Still untested in current data

| Priority | Angle | Method | Expected effort |
|----------|-------|--------|-----------------|
| 1 | Direction-asymmetry audit of existing validated signals | Re-run comprehensive scan with explicit LONG vs SHORT splits of volume / level / timing features | 1 day |
| 2 | Cross-instrument concordance checks | For each BH-survivor, test on twin instrument (MES/MGC) for same session | 2 days |
| 3 | Pre-ORB feature classes | `pre_velocity`, `compression_z`, `pit_range_atr` as overlays on deployed lanes | 2 days |
| 4 | Time-of-break decay | Does signal strength decay within session? Split break_delay_min into fine quintiles × feature | 1 day |
| 5 | Calendar-volume interaction | Already tested broadly; zoom on NFP × volume, OPEX × volume with Chordia-strict t ≥ 3.79 | 1 day |

### Tier 2 — Needs new pipeline data (higher leverage, bigger investment)

| Priority | Data | Unlocks | Effort |
|----------|------|---------|--------|
| 1 | **Tick-level aggressor side / cumulative delta** | Directional volume → order-flow signal, likely >2x edge surface | 1-2 weeks Databento MBO ingestion |
| 2 | **Open interest changes** | Distinguish new positioning from rolls → institutional conviction | 2-3 weeks |
| 3 | **Volume profile within ORB** | Consolidation vs trending detection at session start | 1 week |
| 4 | **Pre-session volume** | Confirmation of institutional positioning before break | 2-3 days |
| 5 | **Cross-asset volume concordance** (ES vs NQ, gold vs bonds) | Regime classification | 2-3 weeks external data + compute |

### Tier 3 — Strategy class expansion (Phase E)

| Priority | Class | Mechanism | Blocker |
|----------|-------|-----------|---------|
| 1 | Direct level fade (SC2.1) | Limit at PDH/PDL with rejection | Dalton / Murphy PDFs not acquired |
| 2 | Level-break momentum (SC2.2) | Breakout + volume confirmation | Tier 1 data + level infrastructure |
| 3 | Mean-reversion within range (SC2.3) | INSIDE_PDR + low-vol gate | Market profile framework literature |

### Tier 4 — Multi-signal architecture (Phase D)

- Continuous forecast combiner (Carver Ch 8): combine validated signals into size multiplier
- Vol targeting: inverse-realized-vol sizing to hit Sharpe target
- Forecast diversification: portfolio of uncorrelated size-multipliers

---

## Known edge landscape as of 2026-04-15

### Signals confirmed institutional-grade
- **Volume (rel_vol_HIGH_Q3):** universal, 5 BH-global survivors, 4 CONDITIONAL T0-T8 (only T3 fails due to thin OOS). Deploy path: Phase D size-scaling.
- **ORB_G5** (top-quintile range): deployed on 3 lanes, proven.
- **ATR_P50 / ATR_P70:** vol regime gates, deployed.
- **OVNRNG_100:** deployed COMEX_SETTLE.
- **VWAP_MID_ALIGNED:** deployed US_DATA_1000.
- **break_delay** (signal quality): research-provisional.

### Signals dead or dead-as-deployed
- **ML (V1/V2/V3):** all DEAD per 2026-04-11.
- **Adversarial FADE:** DEAD per 2026-04-15.
- **Level-rejection as DIRECTIONAL TRADE** (fade opposite direction): DEAD.
- **Level-rejection as binary SKIP:** CONDITIONAL — works on some sessions, not universal.
- **Composite confluence (2-way AND) as multiplier:** adds <30% marginal edge. Low priority.

### Signals under investigation
- **rel_vol_LOW_Q1 SKIP:** mirror of HIGH, cross-family survivor. Ready for T0-T8.
- **bb_volume_ratio HIGH/LOW:** volume confluence with rel_vol, mostly OOS-unverified due to thin fire.
- **F5_BELOW_PDL LONG (P3):** confirmed CONDITIONAL on MNQ US_DATA_1000.
- **F3_NEAR_PIVOT / F2_NEAR_PDL SKIPS:** multiple CONDITIONAL survivors, deployment path requires specific lane mapping.

---

## Process template for the NEXT scan

Copy-paste this template when starting new-edge research:

```
## [YYYY-MM-DD] [TOPIC] Research Plan

### Classification
- [ ] EXPLORATORY (no validated_setups writes, no YAML pre-reg required)
- [ ] DISCOVERY (requires YAML pre-reg per Phase 0 criterion 1)

### Pre-committed scope
- Sessions: [list or ALL 12]
- Instruments: [list or ALL 3]
- Apertures: [list or ALL 3]
- RRs: [list or ALL 3]
- Directions: long, short
- Features: [enumerate + family mapping]
- K_budget: [pre-committed upper bound]
- MinBTL: 2·ln(K) / E[max_N]² = [compute before run]

### Scope EXCLUSIONS (pre-committed)
[list what's explicitly NOT in scope and why]

### Kill criteria (STATE BEFORE RUNNING)
[conditions that would abandon the line of research]

### Methodology gates (verify each applied)
- [ ] Look-ahead gates (RULE 1)
- [ ] Two-pass overlay testing (RULE 2)
- [ ] IS/OOS sacred holdout (RULE 3)
- [ ] Multi-framing BH-FDR (RULE 4)
- [ ] Comprehensive scope within axes (RULE 5)
- [ ] Trade-time-knowable features only (RULE 6)
- [ ] T0 tautology (RULE 7)
- [ ] Extreme-fire + ARITHMETIC_ONLY (RULE 8)
- [ ] Red-flag audit (RULE 12)

### Literature grounding
[cite extracted passages from docs/institutional/literature/]

### Deliverables
- Script: `research/<name>.py`
- Results: `docs/audit/results/YYYY-MM-DD-<name>.md`
- Pre-reg (if DISCOVERY): `docs/audit/hypotheses/YYYY-MM-DD-<name>.yaml`
```

---

## Closing meta-rule

**Every finding is provisional.** The system is not proven right, just not-yet-wrong. Volume confirmation passed BH-global tonight; next week's data may change the picture. Keep Shiryaev-Roberts monitors running, keep historical failure log updated, assume next scan will surprise you.

Institutional rigor is a discipline, not a state. Re-apply every session.
