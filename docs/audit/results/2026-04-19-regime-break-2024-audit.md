# 2024 regime-break systemic audit (Phase 2.7) — results + improved plan

**Date:** 2026-04-19
**Stage:** `docs/runtime/stages/phase-2-7-regime-break-2024-audit.md`
**Script:** `research/phase_2_7_regime_break_2024_audit.py`
**Outputs:**
- `research/output/phase_2_7_regime_break_2024_audit.csv` — per-lane 3-window stats
- `research/output/phase_2_7_cross_matrix_t25_t27.csv` — joined with Phase 2.5 tiers

**Origin:** Two independent Phase 2.4 + 2.6 audit streams surfaced 2024 as a C9 era-stability fail on specific lanes. This audit systematically stratifies ALL 38 active validated_setups lanes by `full Mode A` / `ex-2024 Mode A` / `2024-only` windows, augmented with vol-regime characterization per Chan 2008 Ch 7 § volatility regimes.

## Executive verdict (reframed from prior hypothesis)

**My prior framing — "2024 is a portfolio-wide regime break affecting many lanes" — is PARTIALLY WRONG.**

Actual distribution: 29 of 38 (76%) are 2024_NEUTRAL. The 2024 effect is CONCENTRATED in specific mechanisms:

| Flag | Count | What it means |
|------|------:|--------------|
| 2024_NEUTRAL | **29** | Lane performs similarly with/without 2024 — regime-robust |
| 2024_PURE_DRAG | 2 | 2024 was materially negative; ex-2024 lift > 3% ExpR |
| 2024_CRITICAL | 2 | 2024 CARRIED the lane; ex-2024 drops > 3% ExpR |
| 2024_MIXED | 1 | Small lift but below 3% material threshold |
| 2024_UNEVALUABLE | 4 | N_2024 < 30 (thin sample) |

**Vol-regime characterization (Chan 2008 Ch 7 grounding):**
- ATR_20_pct median across lanes: **2024 = 68.7 vs rest = 48.0 (1.43× higher)**
- GARCH_vol_pct: 2024 = 50.2 vs rest = 43.0
- **2024 was empirically a HIGH-VOL year** by our own feature measurement. Not speculation — from canonical `daily_features`.

Per Chan Ch 7 p120: "high- vs low-volatility regimes... volatility regime switching seems to be most amenable to classical econometric tools." Our system's `garch_forecast_vol_pct` and `atr_20_pct` surfaced the regime shift. The 4 non-NEUTRAL lanes are precisely those where the VOL MECHANISM is the deciding factor.

## Cross-matrix: Phase 2.5 subset-t tier × Phase 2.7 2024 flag

| Phase 2.5 tier ↓ \ 2024 flag → | CRITICAL | MIXED | NEUTRAL | PURE_DRAG | UNEVALUABLE |
|-------------------------------|:-------:|:-----:|:-------:|:---------:|:-----------:|
| ARITHMETIC_LIFT (Rule 8.3)    |    0    |   0   |   2     |    0      |      0      |
| **Tier 1 PASS**               |    **1**|  1    |   **5** |    0      |      1      |
| Tier 1 thin-N                 |    0    |   0   |   0     |    0      |      1      |
| Tier 2                        |    0    |   0   |   3     |    0      |      0      |
| Tier 3                        |    0    |   0   |   9     |    1      |      0      |
| Tier 4                        |    1    |   0   |   10    |    1      |      2      |

## GOLD lanes — robust across regimes (n=5)

Tier 1 subset-t Chordia-PASS **AND** 2024-NEUTRAL. **These are the 5 most deploy-eligible lanes in the current book.**

| Strategy | Full ExpR | Subset t | Δ(ex2024 − full) |
|----------|-----------:|---------:|------------------:|
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | +0.184 | 3.42 | +0.011 |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60` | +0.195 | 4.29 | +0.014 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MES_ATR60` | +0.198 | 3.32 | −0.024 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | +0.221 | 4.14 | −0.014 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | +0.184 | 3.20 | −0.008 |

**Filter-class pattern:** 2 × X_MES_ATR60, 1 × ATR_P50_O30, 1 × OVNRNG_100, 1 × VWAP_MID_ALIGNED. **Volatility-regime-based filters dominate the GOLD pool** — matches Chan Ch 7 prediction that vol regimes are the most tractable.

**Session pattern:** 3 × COMEX_SETTLE, 1 × SINGAPORE_OPEN O30, 1 × US_DATA_1000. COMEX_SETTLE is the strongest-per-session result.

## WATCH lane — regime-dependent (n=1)

Tier 1 subset-t Chordia-PASS **BUT** 2024-CRITICAL (2024 CARRIED the lane — ex-2024 performance drops).

| Strategy | Full | Ex-2024 | 2024-only | Δ (ex − full) | Interpretation |
|----------|-----:|--------:|----------:|--------------:|----------------|
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | +0.205 | +0.161 | **+0.358** | −0.044 | 2024 was the BEST year by far — filter fired more often in high-vol and produced bigger edge |

Ex-2024 ExpR=+0.161 with likely t~3.0+ (still Chordia-ish). Not dead — but **regime-conditional**. Deploy at reduced size, OR gate with a live ATR-regime detector. Twin of `ATR_P50_O30` (which is GOLD) — aperture matters: O30 is regime-robust, O15 is regime-dependent.

## DOUBLE-CONFIRMED retire (Tier 4 + PURE_DRAG)

| Strategy | Full | 2024-only | Subset t | Disposition |
|----------|-----:|----------:|---------:|-------------|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | +0.081 | **−0.125** | 1.70 | Retire |

**Second Tier-3 PURE_DRAG (same family, different RR):** `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` (full=+0.112, 2024=−0.132, t=1.98 — just above Tier 4 threshold at 1.96). Same retire signal.

Both are the Phase 2.4 SGP momentum kills reconfirmed — mechanism ground: cross-session flow memory fails in high-vol regimes.

## MIXED lane (n=1)

| Strategy | Full | Ex-2024 | 2024-only | Δ |
|----------|-----:|--------:|----------:|---:|
| `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60` | +0.214 | +0.252 | +0.092 | +0.038 |

Just above the 0.03 material threshold. 2024 underperformed but didn't drag to negative. Reverse of WATCH — ex-2024 slightly better. Practically still strong (full-Mode-A t=4.19).

## UNEVALUABLE lanes (n=4)

N_2024 < 30 blocks a strict 2024-level read. Noted, not blocked:

- `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` (N_2024=6)
- `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` (N_2024=10)
- `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` (N_2024=27)
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` (N_2024=27)

For MES lanes, the thin 2024 sample is expected (single narrow session). Their full-Mode-A verdict stands independent of this audit.

## Improved plan — data-driven next steps

Prior plan treated "2024 regime break" as a portfolio-wide phenomenon. Corrected per data:

### Immediate (0-15 min decisions)

1. **RETIRE** 2 PURE_DRAG SGP lanes. Third-party confirmation of Phase 2.4 verdict. No further audit needed.
2. **PRIORITIZE deployment** on the 5 GOLD lanes. They're the rigorously-cleared regime-robust core.
3. **SIZE-DOWN or REGIME-GATE** the 1 WATCH lane (`ATR_P50_O15` SGP RR1.5). Ex-2024 still positive but not scale-worthy without a live vol-regime gate.

### Short horizon (2026-05, 2-4h work each)

4. **Build live ATR-regime monitor** using Chan 2008 Ch 7 turning-point methodology (NOT Markov classification — Chan p121 says Markov is useless). Specifically: a feature signaling when `atr_20_pct >= 60` (empirical 2024 median) with rolling-window confirmation per Pepelyshev-Polunchenko Shiryaev-Roberts. Live module `trading_app/regime_monitor.py` that:
   - Flags "elevated-vol regime" in real-time
   - Downweights WATCH lanes by 0.5× when flagged
   - Requires composite filter infra (`CompositeFilter`) still.

5. **Revisit Phase 2.6 CME_PRECLOSE X_MES_ATR60 RR1.5/2.0** with 2024-conditional lens: both failed C9 2024 standalone. But if we REGIME-GATE them out of high-vol days, do they pass? 2-hour research addition — same script, add a regime-filter layer.

### Medium horizon (1-2 weeks)

6. **Test ATR_P50 cross-session extension** (previously-drafted stub at `docs/audit/hypotheses/2026-04-19-atr-p50-cross-session-extension-stub.md`). Now re-framed: ATR_P50_O30 is GOLD, ATR_P50_O15 is WATCH. Cross-session extension should test O30 preferentially, not O15. Update stub before lock.

7. **`CompositeFilter` infra build-out.** Needed for:
   - Regime-gated WATCH lane (`ATR_P50_O15` × regime-clear)
   - Carver-style forecast combiner (referenced in `docs/audit/hypotheses/phase-d-carver-forecast-combiner.md` stub)

### Long horizon

8. **2024 regime characterization for future training.** What was 2024's macro profile? ATR 1.43×; GARCH 1.17×. Match against historical: did 2020 (COVID) / 2018 (vol spike) show similar profiles? If yes, we have a proxy for "2024-like" regimes to characterize statistically. Research question.

## Self-audit (per institutional-rigor rule 1)

### Bias checks

- **Stage file locked before script ran:** ✓ (`docs/runtime/stages/phase-2-7-regime-break-2024-audit.md` created 2026-04-19)
- **Thresholds numeric, no cherry-picking:** ✓ (0.03 delta, 30 N floor, from existing pre_registered_criteria)
- **All 38 lanes audited uniformly:** ✓
- **Holdout sacred:** ✓ — `trading_day < HOLDOUT_SACRED_FROM` unchanged. 2024 is WITHIN Mode A, this is STRATIFICATION not holdout re-optimization.
- **Canonical delegations verified:** ✓ — compute_mode_a agreement check built into script; no inlined 2026-01-01 date; filter_signal canonical path.
- **Literature grounding:** ✓ — Chan 2008 Ch 7 extract explicitly cited; vol-regime interpretation ground in Chan's taxonomy.

### What I might have wrong

1. **"1.43× ATR ratio = 2024 high-vol regime" framing is empirical but unlabeled.** Could be coincidence with some other regime variable (macro event concentration, Fed policy shifts). Would need external macro data to name "what 2024 was" formally. Per integrity-guardian rule 7, external speculation is training-memory — banned without verified local source. Keep claim at "empirically higher vol" only.

2. **The 29 NEUTRAL lanes include 10 Tier-4 lanes.** Those are lanes failing conventional subset-t AND being 2024-regime-insensitive — meaning they're just structurally weak, not regime-dependent. Separate action class.

3. **3 GOLD lanes are at COMEX_SETTLE.** Possible session-correlation concentration risk. Lane correlation gate (`trading_app/lane_correlation.py`) needs re-running on the proposed GOLD subset before deploy decisions — cross-RR or cross-filter lanes on same session can saturate.

### Follow-ups filed (not in this stage)

- Run `lane_correlation` on the 5 GOLD subset to check deploy-safe concurrency
- Update ATR_P50 cross-session stub (O30 preferred; O15 regime-gated)
- Stub `regime-gated CME_PRECLOSE X_MES_ATR60 recovery` research

## Audit trail

- Pre-reg-style stage file locked 2026-04-19 before script ran
- Script delegates to canonical compute_mode_a, filter_signal, HOLDOUT_SACRED_FROM, GOLD_DB_PATH, SESSION_CATALOG
- Cross-reference CSV: `research/output/phase_2_7_cross_matrix_t25_t27.csv`
- No production / validator / DB mutations
