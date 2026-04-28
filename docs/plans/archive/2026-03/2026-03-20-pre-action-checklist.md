---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Pre-Action Checklist — What Must Be Right Before We Do Anything

## What We Have (VERIFIED)
- NYSE_OPEN RR1.0: +0.176R OOS (p=0.000001, BH FDR N=55)
- COMEX_SETTLE RR1.0: +0.115R OOS (p=0.00066, BH FDR N=55)
- Both survive 3-tick slippage stress test

## What's BROKEN or INCOMPLETE

### 1. ML Pipeline Config Issues
- [ ] NYSE_OPEN has NO validated_setups entry at RR1.0 → pipeline skips it entirely
- [ ] The orb_outcomes fallback only triggers when NO validated configs exist for the instrument+RR. MNQ has SOME configs at RR1.0 (CME_PRECLOSE, COMEX_SETTLE) so NYSE_OPEN is silently skipped.
- [ ] FIX: Either add NYSE_OPEN RR1.0 to validated_setups manually, or make the fallback work per-SESSION not per-instrument.

### 2. Session Chronological Order (Bloomey finding)
- [ ] Static order wrong for winter (EUROPE_FLOW/LONDON_METALS swap Oct-Mar)
- [ ] Affects cross-session features in features.py AND session_guard.py
- [ ] FIX: Dynamic per-day resolution from dst.py, or special-case the swap

### 3. Research Script vs Pipeline Divergence
- [ ] Research scripts (scripts/*.py) bypass E6 filter, CPCV, quality gates
- [ ] Blacklists now import from canonical source (fixed) but scripts still don't use CPCV
- [ ] Quick ML scripts produce DIFFERENT results than the real pipeline (proven today)
- [ ] RULE: NEVER write standalone ML scripts. Always use meta_label.py.

### 4. New Features Not Populated
- [ ] VWAP and pre_velocity only ~30% populated (recent data only)
- [ ] Need full MNQ daily_features rebuild to populate for all years
- [ ] Without full rebuild, ML can't use these features properly (training data has them missing)

### 5. Noise Floor Status
- [ ] NOISE_EXPR_FLOOR zeroed in config.py (intentional, but live vulnerability)
- [ ] MNQ noise floor confirmed at 0.52R for 31K grid search
- [ ] For single pre-specified tests: noise floor = just p<0.05 (our 2 setups pass)
- [ ] MES noise floor: unknown (null test incomplete)

### 6. Cost Model Uncertainty
- [ ] Flat $3.74/trade for MNQ regardless of session/conditions
- [ ] Stop-market entries likely have 1-3 ticks slippage (modeled at 1)
- [ ] Kill point: 1.87x slippage ($7.00/trade)
- [ ] Only paper trading resolves this

### 7. 2025 Holdout Contamination
- [ ] 55+ tests touched 2025 data this session
- [ ] 2026 data declared sacred (Jan-Mar, ~60 trading days)
- [ ] CME_PRECLOSE pre-registered for 2026 test
- [ ] Any further testing on 2025 data increases the contamination count

### 8. ML Feature Set Issues
- [ ] New features (VWAP, velocity) not in ML top 10 importance
- [ ] Current features that drive ML: atr_vel_ratio, rel_vol, level proximity, compression
- [ ] These are existing features — new features need proper evaluation
- [ ] ML with CPCV on COMEX_SETTLE produced +0.8R honest delta (essentially nothing)

## ORDER OF OPERATIONS (before any trading or ML)

1. **Rebuild daily_features for MNQ** — populate VWAP/velocity for all years
2. **Fix ML config picker** — make fallback work per-session so NYSE_OPEN gets ML treatment
3. **Fix winter session ordering** — dynamic resolution for EF/LM
4. **Run PROPER ML pipeline** on both verified sessions with full features
5. **Bootstrap the pipeline results** (not quick scripts)
6. **Paper trade the RAW baselines** while ML is being fixed
7. **Collect slippage data** from paper trading to validate cost model
8. **April 2026: Run CME_PRECLOSE on sacred 2026 holdout**

## WHAT NOT TO DO
- Do NOT write more quick ML scripts
- Do NOT test anything else on 2025 data
- Do NOT celebrate until paper trading confirms
- Do NOT add features without rebuilding daily_features first
- Do NOT run ML without CPCV and quality gates
