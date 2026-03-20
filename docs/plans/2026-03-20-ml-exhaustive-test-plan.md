# ML Exhaustive Test Plan — MNQ E2 Multi-RR

**Date:** 2026-03-20
**Status:** IN PROGRESS (sweep running)
**Terminal:** This terminal (paper trading terminal)
**Other terminal:** Has hybrid config picker + bypass_validated code

## Purpose

Determine whether ML meta-labeling adds genuine value to MNQ ORB breakout trading at ANY RR target, before declaring ML dead. The other terminal tested RR1.0 only and found 0/33 models passing quality gates. That's insufficient — RR2.0 has structurally stronger univariate signal.

## What We Know (Evidence, Not Memory)

### Baseline landscape (from gold.db, queried today)
| RR | ExpR | WR | Sample Size |
|----|------|-----|-------------|
| 1.0 | +0.085 | 54.8% | 15,291 |
| 1.5 | +0.083 | 42.5% | 15,291 |
| 2.0 | +0.077 | 34.4% | 15,291 |
| 2.5 | +0.053 | 28.3% | 15,291 |
| 3.0 | +0.033 | 23.9% | 15,291 |
| 4.0 | -0.010 | 17.9% | DEAD |

### Univariate feature signal (queried today)
| Feature | RR1.0 Spread | RR2.0 Spread | Signal Direction |
|---------|-------------|-------------|-----------------|
| ATR percentile (Q4/Q1) | 2.6x | 2.9x | High vol → better |
| ORB size NYSE_OPEN (Q2/Q4) | 1.9x | **5.6x** | Mid-size → better |
| ATR velocity (Expanding/Stable) | 1.6x | ~1.7x | Expanding → better |
| Gap type (gap_down/inside) | 1.6x | TBD | Gap down → better |

**Key finding: ORB size discrimination is 3x stronger at RR2.0 than RR1.0.**

### Time split boundaries (60/20/20 chronological)
| Split | N | Dates | Avg PnL_R |
|-------|---|-------|-----------|
| TRAIN (60%) | 9,174 | 2021-02 to 2024-02 | +0.069 |
| VAL (20%) | 3,058 | 2024-02 to 2025-02 | +0.122 |
| TEST (20%) | 3,059 | 2025-02 to 2026-03 | +0.094 |

**Regime drift warning:** Val period (+0.122) is hotter than train (+0.069). Test is in between. Thresholds optimized on hot val data may not transfer.

### Feature readiness
- VWAP: 99% populated for NYSE_OPEN/COMEX_SETTLE (memory was wrong at 57%)
- Velocity: same coverage as VWAP
- Lookahead blacklist: 35 items, all audited
- Session guard: wired
- fillna(-999.0): affects ~1% of rows, not a material confound

### Infrastructure gaps
- **No bootstrap code exists** in `trading_app/ml/`. Must be written.
- **No `--bypass-validated` CLI arg** in meta_label.py (function param exists, CLI doesn't expose it)
- **validated_setups at RR2.0:** only 2 entries. Config picker uses hybrid fallback to orb_outcomes.

## Phases

### Phase 0.5: Univariate Feature Audit (DONE — running)
Script: `scripts/tools/ml_exhaustive_sweep.py`
Log: `logs/ml_sweep_univariate.log`

For each RR target (1.0, 1.5, 2.0), compute quartile ExpR for:
- ATR percentile, ATR velocity, gap type, ORB size per session

**Decision:** If NO feature shows meaningful discrimination at a given RR, skip ML training at that RR.

### Phase 1: Multi-RR Flat Sweep (RUNNING)
Logs: `logs/ml_sweep_rr20_flat.log`, `logs/ml_sweep_rr15_flat.log`, `logs/ml_sweep_rr10_flat.log`

Run `train_per_session_meta_label()` with:
- `single_config=True, skip_filter=True, bypass_validated=False`
- `config_selection="max_samples"`
- `per_aperture=False` (flat: one model per session)

Order: RR2.0 first (strongest signal), then RR1.5, then RR1.0.

**Quality gates per session (existing 4-gate pipeline):**
1. honest_delta_r >= 0 (ML must not hurt)
2. CPCV AUC >= 0.50 (better than random in CV)
3. Test AUC >= 0.52 (clearly above random)
4. Skip rate <= 85% (discrimination, not avoidance)

**Decision after Phase 1:**
- 0 sessions pass at ANY RR → go to Phase 1b anyway, then Phase 3 (simple filter baseline)
- 1+ sessions pass → Phase 2 (bootstrap mandatory)

### Phase 1b: Per-Aperture at RR2.0 (RUNNING)
Log: `logs/ml_sweep_rr20_aperture.log`

Same as Phase 1 but `per_aperture=True`. O5 and O15 have different break dynamics.

**Decision:** Compare per-aperture vs flat results. If per-aperture improves, use it.

### Phase 2: Bootstrap Permutation Test (NOT YET BUILT)
**Must write code.** Budget: ~100 lines.

For every session that passes Phase 1 quality gates (or the best-rejected session):
1. Shuffle labels (win/loss) 1000 times
2. Train RF + optimize threshold on each shuffle (same pipeline)
3. Compute null distribution of honest_delta_r
4. p-value = fraction of null deltas >= real delta
5. **Kill: p > 0.05 = ML has no skill, model is threshold artifact**

This is the test that killed negative-baseline ML (p=0.35). Must apply to positive baselines.

**Simplification:** Can do 200 permutations instead of 1000 (p-value precision to ±0.05 at p=0.05, which is sufficient for a go/no-go decision).

### Phase 3: Simple Regime Filter Baseline (DEFERRED — other terminal)
Before deploying any ML, test whether a simple conditional filter captures the same signal:
- Trade only when atr_20_pct > 50 (Q3+Q4)
- At RR1.0: lifts ExpR from +0.085 to +0.107 (skips 46% of trades)
- No ML, no overfitting, no threshold optimization

**If simple filter captures 80%+ of ML's delta, prefer the filter (fewer moving parts).**

### Phase 4: Replay Validation (IF bootstrap passes)
Run paper_trader with ML-filtered portfolio vs raw baseline:
```bash
python -m trading_app.paper_trader --instrument MNQ --raw-baseline --use-ml \
  --rr-target X.0 --start 2025-01-01 --end 2025-12-31
```
Compare trade counts, win rates, ExpR, drawdown.

## Kill Criteria

| Level | Criterion | Action |
|-------|-----------|--------|
| Phase 1 | 0 sessions pass 4-gate at any RR | Phase 3 (simple filter) |
| Phase 2 | All bootstrap p > 0.05 | ML is dead. Trade raw baseline. |
| Phase 2 | Bootstrap p < 0.05 but delta < +0.02R/trade | ML works but marginal. Simple filter may be better. |
| Phase 4 | ML replay worse than raw baseline | Reject ML. Configuration error or overfit. |

## What Could Go Wrong

1. **Regime drift:** Test period (2025-2026) is hotter than training (2021-2024). ML might pass gates on the hot test period but fail in normal markets. Mitigation: bootstrap tests whether skill is real.

2. **Threshold optimization on finite val set:** 36 candidate thresholds × ~260 val trades = mild selection bias. Mitigation: 4 OOS gates on frozen test set.

3. **Config picker at RR2.0:** Only 2 validated_setups at RR2.0. Hybrid fallback to orb_outcomes works (verified: 13,337 samples loaded). But the configs might not match production deployment.

4. **fillna(-999.0):** RF learns "when feature is -999, do X." For 1% of rows this is noise. For CME_REOPEN (57% VWAP coverage in some years) it could be a temporal proxy. Monitor feature importance for -999-related splits.

5. **Multiple testing across RR levels:** Testing 3 RR × 12 sessions = 36 models. White (2000) selection uplift already computed by the pipeline. Report all, not just survivors.

## Files

| File | Purpose |
|------|---------|
| `scripts/tools/ml_exhaustive_sweep.py` | Unattended sweep runner |
| `logs/ml_sweep_master.log` | Master log (start/stop/summary) |
| `logs/ml_sweep_univariate.log` | Feature signal audit |
| `logs/ml_sweep_rr20_flat.log` | RR2.0 flat model results |
| `logs/ml_sweep_rr15_flat.log` | RR1.5 flat model results |
| `logs/ml_sweep_rr10_flat.log` | RR1.0 flat model results |
| `logs/ml_sweep_rr20_aperture.log` | RR2.0 per-aperture results |

## When You Come Back

1. Check `logs/ml_sweep_master.log` for summary
2. Check each `ml_sweep_rr*.log` for per-session results
3. Count sessions passing 4-gate quality check at each RR
4. If any pass: write bootstrap test (Phase 2)
5. If none pass: evaluate simple regime filter (Phase 3)
6. Update this doc with results
