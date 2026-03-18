# Adversarial Audit Fixes — Design Document

**Date:** 2026-03-18
**Source:** Zero-context adversarial code + DB audit
**Status:** IMPLEMENTED — awaiting rebuild + merge

---

## Findings Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | ts_outcome not used in discovery/validation/fitness | CRITICAL | Workstream A |
| 2 | Filter definitions diverge across pipeline/discovery/live | HIGH | Workstream B |
| 3 | Variant harvesting (post-hoc families) | MEDIUM | NO ACTION — already handled by canonical/alias dedup |
| 4 | MGC high-vol conditional masquerading as general | MEDIUM | Workstream C |
| NEW | Cross-asset ATR injection missing in paper_trader + live | HIGH | Workstream B |

---

## Finding #3 Disposition: NO ACTION NEEDED

The audit measured raw `experimental_strategies` (pre-dedup) against family heads (post-dedup) — apples-to-oranges. The actual dedup happens at the canonical/alias boundary:

- Discovery computes `trade_day_hash` and marks canonical vs alias
- Validator skips all non-canonical strategies (`is_canonical = False → continue`)
- FDR correction K-count uses only canonical strategies
- Aliases never enter `validated_setups`

The 4.3-4.4x ratio is aliases in experimental_strategies, not inflation in the validated set.

---

## Workstream A: Canonical Exit Model (Finding #1)

### Problem
- outcome_builder pre-computes `ts_outcome`/`ts_pnl_r` (time-stop adjusted P&L)
- Discovery, fitness, validator all SELECT raw `outcome`/`pnl_r` — ignoring time-stops
- Live execution engine implements time-stops independently
- 1.9M rows have `ts_outcome != outcome`
- We are validating a DIFFERENT system than we trade

### Design: Hard Switch to ts_outcome

Change discovery/fitness queries to `SELECT ts_outcome AS outcome, ts_pnl_r AS pnl_r`. Alias preserves downstream compatibility — `compute_metrics()` continues reading `o["outcome"]` and `o["pnl_r"]` unchanged.

Handle new outcome label: `"time_stop"` → treat as loss in win_rate calculation.

### Files

| File | Change |
|------|--------|
| `trading_app/strategy_discovery.py:1004` | SELECT ts_outcome/ts_pnl_r |
| `trading_app/strategy_discovery.py:472` | Handle "time_stop" in compute_metrics |
| `trading_app/strategy_fitness.py:440` | SELECT ts_outcome/ts_pnl_r |
| `trading_app/strategy_fitness.py` | Handle "time_stop" in fitness metrics |
| `pipeline/check_drift.py` | Add check: no raw outcome SELECT from orb_outcomes |
| `tests/test_strategy_discovery.py` | Test "time_stop" handling |

### Rebuild Required
- Outcomes: NO (ts_outcome already populated)
- Discovery → Validation → Edge Families: YES (all instruments × apertures)

---

## Workstream B: Cross-Asset ATR Bug + Filter Docs (Finding #2 + NEW)

### Problem 1: Cross-Asset ATR Broken in Replay/Live
- Discovery/fitness inject `cross_atr_{source}_pct` via `_inject_cross_asset_atrs()`
- Paper_trader and session_orchestrator do NOT inject these columns
- All 137 MNQ cross-asset strategies (X_MES_ATR70, X_MGC_ATR70) silently reject every trade
- Fail-closed prevents wrong trades but makes strategies untradeable

### Problem 2: rel_vol Computation Divergence
- Pipeline: session-level lookback (last 20 break-days, same session)
- Discovery/fitness: UTC minute-of-day lookback (last 20 same-minute volumes)
- Live: yesterday's daily_features (stale proxy)
- Assessment: discovery is the validation gate; pipeline's value is used only in replay. Divergence is small and not a validation integrity issue.

### Problem 3: "Future Leak" in Replay
- Paper_trader injects same-day daily_features row
- Assessment: PARTIALLY FALSE ALARM. break_speed and rel_vol are contemporaneous (computed from the break event itself). atr_20_pct uses only prior days (no look-ahead in pipeline). The daily_features injection in replay is actually correct.
- Real gap: Live uses yesterday's proxies for break_speed/rel_vol, which is inherent to real-time trading (can't know today's break speed until break happens).

### Design: Extract + Inject

1. Extract cross-asset ATR enrichment to shared utility
2. Call from paper_trader after loading daily_features
3. Call from session_orchestrator after loading daily_features
4. Document rel_vol divergence (no code change)

### Files

| File | Change |
|------|--------|
| `trading_app/feature_enrichment.py` | NEW — extracted cross-asset ATR logic |
| `trading_app/strategy_discovery.py` | Import from feature_enrichment |
| `trading_app/strategy_fitness.py` | Import from feature_enrichment |
| `trading_app/paper_trader.py:301` | Add cross-asset enrichment call |
| `trading_app/live/session_orchestrator.py:262` | Add cross-asset enrichment call |
| `tests/test_paper_trader.py` | Test cross-asset columns present |

### Rebuild Required
- NO. Code-only fix.

---

## Workstream C: MGC ATR Runtime Gate (Finding #4)

### Problem
- MGC WF starts 2022 (high-vol era); low-vol not validated OOS
- MGC CORE strategies deploy unconditionally (no ATR check)
- config.py documents this as a known regime dependency
- `check_regime.py` exists but is NOT wired into live trading

### Design: ATR Gate on MGC CORE Specs

Add `regime_gate="high_vol"` (or equivalent ATR check) to MGC CORE specs in live_config.py. When `atr_20_pct < 50`, skip native MGC strategies.

### Files

| File | Change |
|------|--------|
| `trading_app/live_config.py` | Add ATR gate to MGC CORE specs |
| `TRADING_RULES.md` | Document MGC as regime-conditional |

### Rebuild Required
- NO. Configuration change only.

---

## Execution Order

1. **Workstream B** — Cross-asset bug fix (smallest, fixes broken functionality)
2. **Workstream A** — ts_outcome switch (largest impact, requires rebuild)
3. **Workstream C** — MGC ATR gate (config change, low risk)

---

## Risk Table

| Risk | Workstream | Severity | Mitigation |
|------|-----------|----------|------------|
| Full rebuild changes validated set >30% | A | HIGH | Compare before/after. Investigate large drops. |
| "time_stop" label missed in some code path | A | MEDIUM | grep for outcome string literals |
| Cross-asset enrichment extracted wrong | B | LOW | Same function, just moved |
| MGC ATR gate too aggressive | C | MEDIUM | Check current atr_20_pct distribution |

## Rollback
- A: revert SELECT + re-run rebuild
- B: revert enrichment calls
- C: remove ATR gate from specs
