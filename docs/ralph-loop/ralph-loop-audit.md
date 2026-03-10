# Ralph Loop — Active Audit State

> This file is overwritten each iteration with the current audit findings.
> Historical findings are preserved in `ralph-loop-history.md`.

## RALPH AUDIT — Iteration 16 (Trade Book: generate_trade_sheet.py, live_config.py)
## Date: 2026-03-10
## Infrastructure Gates: 4/4 PASS

| Gate | Result | Detail |
|------|--------|--------|
| `check_drift.py` | PASS | 71 checks passed, 0 skipped, 6 advisory (non-blocking) |
| `audit_behavioral.py` | PASS | All 6 checks clean |
| `pytest test_live_config.py` | PASS | 20/20 passed (0.67s) |
| `ruff check` | PASS | All checks passed |

---

## Target Files This Iteration

- `scripts/tools/generate_trade_sheet.py` (703 lines)
- `trading_app/live_config.py` (731 lines)

---

## Deferred Findings from Prior Iterations (Status Check)

### F1 — rolling_portfolio.py:304 orb_minutes=5 hardcode (STILL DEFERRED)
- Severity: MEDIUM (dormant)
- File: `trading_app/rolling_portfolio.py:304`
- Status: DEFERRED — annotated. Dormant until rolling evaluation extends to multi-aperture.

### F3 — Unannotated magic numbers (PARTIALLY DONE)
- Severity: MEDIUM (batch)
- Remaining locations NOT yet annotated:
  - `portfolio.py:944` — 0.4 trades/strategy/day estimate
  - `strategy_fitness.py:120` — -0.1 Sharpe decline threshold
  - `cost_model.py:153-229` — SESSION_SLIPPAGE_MULT values
- Status: PARTIALLY DONE

### N4 — HOT Tier Thresholds Missing @research-source (STILL DEFERRED)
- Severity: LOW
- File: `trading_app/live_config.py:54-57`
- Status: DEFERRED (HOT tier dormant)

### N5 — Live Portfolio Constructor Magic Numbers (STILL DEFERRED)
- Severity: LOW
- File: `trading_app/live_config.py:354-355,583-584`
- Status: DEFERRED (refactor scope — named constants in config.py needed)

### Iter 9 LOWs (STILL OPEN)
1. Fill price `or` pattern (falsy zero) — `order_router.py:136,140,202,206`
2. PRODUCT_MAP hardcodes instrument list — `contract_resolver.py:22-27`
3. Auth token refresh not logged — `auth.py:42-60`

---

## New Findings This Iteration

### Finding T1 — Dollar gate FAIL-OPEN in trade sheet (generate_trade_sheet.py:134,140)
- Severity: HIGH
- File: `scripts/tools/generate_trade_sheet.py:134,140`
- Evidence:
  ```python
  # Line 134: NULL median_risk_points → passes gate
  if exp_d is None:
      return True, None  # skip gate if data missing (fail-open on missing data)
  # Line 139-140: cost spec failure → passes gate
  except Exception:
      return True, exp_d
  ```
- Root Cause: Trade sheet `_passes_dollar_gate` returns `True` (pass) when data is missing or cost spec fails. The IDENTICAL gate in `live_config.py:372-391` correctly returns `False` (block). Iteration 13 fixed live_config to fail-closed, but generate_trade_sheet.py was not updated. Result: trade sheet could display phantom trades that the live portfolio builder would never select.
- Current Impact: DORMANT — all strategies currently have median_risk_points populated (0 NULL). But after a partial rebuild or data issue, this divergence would show misleading trades.
- Fix: Align with live_config pattern — return `False` on missing data and on exception.
- Blast Radius: `_passes_dollar_gate` called only from `collect_trades` (same file). No external callers.

### Finding T2 — RR lock JOIN divergence (generate_trade_sheet.py:210 vs live_config.py:239)
- Severity: MEDIUM
- File: `scripts/tools/generate_trade_sheet.py:210` vs `trading_app/live_config.py:239`
- Evidence:
  - Trade sheet: `LEFT JOIN family_rr_locks` + `(frl.locked_rr IS NULL OR vs.rr_target = frl.locked_rr)` — shows strategies even without RR lock
  - live_config: `INNER JOIN family_rr_locks` + `AND vs.rr_target = frl.locked_rr` — requires lock
- Root Cause: Trade sheet was designed for "graceful degradation" (show something even without locks). But this means it can show RR variants that live_config would never select. A maintenance hazard — any future family without an RR lock would appear on the trade sheet but not in the actual portfolio.
- Current Impact: DORMANT — all live families have RR locks (480 rows in family_rr_locks, all 26 specs covered).
- Fix: Change LEFT JOIN → INNER JOIN, remove IS NULL fallback.
- Blast Radius: `_load_best_by_expr` called only from `collect_trades` (same file).

### Finding T3 — Missing orb_minutes column in trade sheet query
- Severity: LOW
- File: `scripts/tools/generate_trade_sheet.py:200-226`
- Evidence: Query does not SELECT `vs.orb_minutes`. Aperture is instead parsed from strategy_id string via `_parse_aperture()` (line 104-110). live_config query correctly selects `vs.orb_minutes`.
- Fix: Add `vs.orb_minutes` to SELECT, use directly instead of string parsing.
- Blast Radius: Same file only.

### Finding T4 — Hardcoded G-filter list in display formatter
- Severity: LOW
- File: `scripts/tools/generate_trade_sheet.py:61`
- Evidence: `["G2", "G3", "G4", "G5", "G6", "G8"]` — inline list. Display-only, no trading logic.
- Status: DEFERRED (display formatting, no data impact)

---

## Confirmed Clean

**live_config.py:**
- Seven Sins: CLEAN. Dollar gate fail-closed (fixed iter 13). Fitness gate fail-closed (fixed iter 12). Instrument exclusion working. All tier pathways correct.
- Canonical integrity: CLEAN. Imports from canonical sources (asset_configs, cost_model, paths, dst).
- The 26 specs match what the trade sheet generated (33 trades = 26 specs resolved across 4 instruments minus exclusions/dollar gate/fitness gate).

**generate_trade_sheet.py:**
- Session time resolution: CORRECT. Uses SESSION_CATALOG resolvers directly.
- HTML generation: CLEAN. No trading logic in display layer.
- Dollar gate + RR lock: DIVERGES from live_config (T1, T2).

---

## Summary
- Total new findings: 4 (0 CRIT, 1 HIGH, 1 MEDIUM, 2 LOW)
- CRITICAL: 0, HIGH: 1 (T1), MEDIUM: 1 (T2), LOW: 2 (T3, T4)
- Deferred carry-forward: F1, F3 (partial), N4, N5, 3x iter-9 LOWs
- Infrastructure Gates: 4/4 PASS

**Top eligible fix: T1** — dollar gate fail-open (HIGH, blast radius = 1 file, 2 lines).
**Next: T2** — RR lock JOIN divergence (MEDIUM, same file).

**Next iteration targets:**
- T2 fix (MEDIUM): RR lock LEFT→INNER JOIN alignment
- T3 fix (LOW): Add orb_minutes to query
- Resolve F3 remaining: portfolio.py:944, strategy_fitness.py:120
