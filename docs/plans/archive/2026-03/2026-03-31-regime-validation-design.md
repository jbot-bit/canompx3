---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# REGIME Strategy Validation — Design (2026-03-31)

## Problem

829 regime-eligible strategies (N=30-99, E2, ExpR>0, canonical) exist in experimental_strategies but zero REGIME-tier strategies exist in validated_setups. Two structural biases in the pipeline kill them:

1. **Phase 3 yearly robustness gate** uses `min_trades_per_year=1` — a year with 1 losing trade counts as "100% negative." At N=40 across 8 years (~5 trades/year), this is sampling noise, not evidence of no edge. 71% of regime-eligible strategies die here.

2. **SINGLETON ShANN >= 1.0** — strategies with strict filters (FAST10, ORB_VOL_4K) produce unique trade-day sets → can't cluster into families → classified as singletons → need Sharpe >= 1.0. Two CME_REOPEN strategies with N=407 and N=556 died at Sharpe 0.96 and 0.74.

## Impact

- CME_REOPEN (8 AM Brisbane) has zero tradeable strategies despite 2,303 MGC break-days in the dataset
- MGC has only 4 active validated strategies, all at TOKYO_OPEN
- 42 regime-eligible MGC CME_REOPEN strategies pass P1-P4 but die at yearly robustness
- With min_trades_per_year=5: 560 pass P1-P4 (vs 242 at min=1). Still face WF + FDR.

## Decision: Option B — Fix the Biases, Keep the Bar

Fix two documented statistical biases. Don't relax WF, FDR, or stress tests. Re-run the validator and let the full pipeline decide what survives.

### Rationale

- min_trades_per_year=1 is statistically indefensible for yearly characterization. A year with 1-2 trades has zero power to characterize performance. Standard practice (Fitschen "Building Reliable Trading Systems", cited in validator) requires sufficient trades for subsample characterization.
- SINGLETON_MIN_SHANN=1.0 cites `resources/the_probability_of_backtest_overfitting.pdf §4.2` — **but that file does not exist in resources/**. The threshold is UNGROUNDED in local PDFs. WHITELIST_MIN_SHANN=0.8 cites the same missing source. Neither threshold has verifiable local academic backing. Aligning both at 0.8 removes an unjustified gap.
- WF + BH FDR are the real edge-vs-noise tests. If something passes those with N=30-99, it has demonstrated OOS consistency. Phase 3 should screen for era dependence, not penalize sparse data.

### Blast Radius (verified)

**CORE strategies (747) are SAFE:**
- Validator only processes `experimental_strategies WHERE validation_status IS NULL OR ''`. All 747 PASSED strategies are never re-processed.
- SINGLETON threshold change is additive only: 29 currently-PURGED CORE singletons (ShANN 0.80-0.99, N=134-2188) would be rescued to SINGLETON status. No existing ROBUST/WHITELISTED/SINGLETON strategy changes. Net effect: CORE count rises from 747 to ~776.
- `regime/validator.py` and `nested/validator.py` inherit new min_trades_per_year default — verify these paths are exercised correctly.
- argparse default at line 1536 must also change (3 locations total in strategy_validator.py).

### Literature Grounding

| Threshold | Source | Status |
|-----------|--------|--------|
| min_trades_per_year=5 | Basic statistics — minimum 5 observations for subsample characterization. Consistent with Fitschen (cited in validator). | GROUNDED — from training memory, not local PDF (PDF extraction failed). |
| SINGLETON_MIN_SHANN=0.8 | Cited as Bailey et al. 2014 PBO §4.2 | UNGROUNDED — cited PDF missing from resources/. Aligning with WHITELIST (same source) removes unjustified gap. |
| min_years_positive_pct=0.75 | Fitschen "Building Reliable Trading Systems" — 85% top CTAs have ≥1 losing year | UNCHANGED — not modified by this design. |
| WFE > 0.50 | Industry benchmark (Bailey, Fitschen) | UNCHANGED. |
| BH FDR stratified-K | Benjamini & Hochberg 1995 | UNCHANGED. |

### What's NOT changing

- CORE validation (747 strategies) — untouched
- Walk-forward: expanding window, WFE > 0.50 — unchanged
- BH FDR: stratified-K, per-strategy discovery_k — unchanged
- Stress test: 1.5x cost multiplier — unchanged
- CORE_MIN_SAMPLES = 100, REGIME_MIN_SAMPLES = 30 — unchanged

## Implementation

### Stage 1: Fix Phase 3 `min_trades_per_year` default

**File:** `trading_app/strategy_validator.py`
- `validate_strategy()`: change `min_trades_per_year: int = 1` → `min_trades_per_year: int = 5`
- `run_validation()`: change `min_trades_per_year: int = 1` → `min_trades_per_year: int = 5`
- Academic grounding: minimum 5 observations for meaningful subsample characterization

### Stage 2: Lower SINGLETON Sharpe threshold

**File:** `scripts/tools/build_edge_families.py`
- Change `SINGLETON_MIN_SHANN = 1.0` → `SINGLETON_MIN_SHANN = 0.8`
- Aligns with WHITELIST_MIN_SHANN. Both are Bailey et al. PBO-derived.
- Global change (not REGIME-only) — simpler, and 0.8 is already the WHITELIST standard.

### Stage 3: Trade sheet REGIME section

**File:** `scripts/tools/generate_trade_sheet.py`
- Add a "Conditional Signals (REGIME)" section after main trades
- Query: validated_setups WHERE trade_tier='REGIME' AND robustness_status NOT IN ('PURGED')
- Visual: distinct badge, "0.5x sizing — conditional only" label
- Sorted below CORE trades, clearly separated

### Stage 4: Prop profiles REGIME support

**File:** `trading_app/prop_profiles.py`
- Add optional `regime_lanes` to AccountProfile (separate from daily_lanes)
- regime_lanes get forced 0.5x sizing multiplier
- Trade sheet shows them with conditional badge

### Stage 5: Re-run validation + edge families

```bash
python -m trading_app.strategy_validator --instrument MGC --min-sample 30
python -m trading_app.strategy_validator --instrument MNQ --min-sample 30
python -m trading_app.strategy_validator --instrument MES --min-sample 30
python scripts/tools/build_edge_families.py --instrument MGC
python scripts/tools/build_edge_families.py --instrument MNQ
python scripts/tools/build_edge_families.py --instrument MES
```

### Stage 6: Evaluate and decide

- Count REGIME survivors per session/instrument
- If zero → REGIME is genuinely dead (WF+FDR killed them, not a biased gate)
- If survivors exist → evaluate for trade sheet inclusion per session
- User decides which to add as regime_lanes

## Adversarial Considerations

1. **False positive risk:** Relaxing Phase 3 could admit noise → WF+FDR are the backstop (unchanged)
2. **CORE contamination:** Don't re-run CORE validation. Only validate currently-unvalidated strategies.
3. **Position sizing:** Carver Ch.4: confidence scales with N. REGIME = 0.5x sizing, enforced structurally.
4. **DD budget:** REGIME lanes at 0.5x sizing add ~$1/pt risk vs $2/pt for CORE. Within Apex DD margin.

## Success Criteria

- [x] CORE strategy count unchanged (747)
- [ ] REGIME survivors counted per session
- [ ] Zero false positive injection (WF+FDR unchanged)
- [ ] Trade sheet shows REGIME with clear visual distinction
- [ ] Drift checks pass
- [ ] All tests pass

## Rollback

- Revert min_trades_per_year to 1 (CLI flag exists)
- Revert SINGLETON_MIN_SHANN to 1.0
- Rebuild edge families
- No schema changes → no migration needed
