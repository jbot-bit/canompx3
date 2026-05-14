# Pyright cleanup: `pipeline/build_daily_features.py`

**Owner:** next session
**Created:** 2026-05-15 (Brisbane)
**Parent plan:** "2026-05-15 punch list" Stage 1b (split from Stage 1 after actual error count exceeded estimate)
**Predecessor commits on `main`:**
- `b98df2ec` — Stage 1a: `_scalar()` helper for `build_bars_5m.py` (8 errors → 0)
- `7871b4b4` — Stop-hook schema fix (`completion-notify.py`)
- `8d7ec08c` — Stage 1a stage-doc closeout

---

## Goal

Drive `uv run pyright pipeline/build_daily_features.py` from **62 errors → 0** without changing runtime behavior. The CI pyright job at `.github/workflows/ci.yml:39-41` runs `uv run pyright` with `continue-on-error: true`; this plan does NOT change that flag — promoting to hard gate is a separate future track.

## Why this exists

The "2026-05-15 punch list" Stage 1 plan estimated ~36 errors at lines 323–334. Live count is **62**, spanning categories the plan didn't enumerate (ExtensionArray vs ndarray, Series-cell-update on None, GARCH literal-arg). Per `feedback_closeout_verify_against_canonical.md`, plan numerics don't bind; live count does. Split to its own plan.

## Constraints (NON-NEGOTIABLE)

1. **No runtime behavior change.** This file is a canonical pipeline writer; `daily_features` feeds discovery, validators, allocator. Any logic change risks contaminating canonical data. Per `institutional-rigor.md § 4` ("never re-encode canonical logic") and § 6 ("no silent failures"), every coercion must be at type-narrowing site only.
2. **Public symbol surface must stay byte-identical.** Many production importers exist (see "Import surface" section). Names, signatures, return types of the following must NOT change:
   - `_wilders_rsi(closes: np.ndarray, period: int = 14) -> float | None`
   - `compute_trading_day_utc_range`, `compute_trading_day`
   - `_orb_utc_window`, `_break_detection_window`
   - `_apply_htf_level_fields`, `_htf_week_key`, `_htf_prior_month_key`
   - `_load_postpass_seed_rows`, `_load_htf_seed_rows`
   - `build_daily_features`, `detect_double_break`
   - `VALID_ORB_MINUTES`, `ACTIVE_ORB_MINUTES`, `COMPRESSION_SESSIONS`, `GARCH_MIN_PRIOR_CLOSES`, `GARCH_PCT_MIN_PRIOR_VALUES`
3. **No `# type: ignore` without justification.** Each suppression must carry a one-line comment naming the underlying type-system limitation (pandas-stubs gap, ExtensionArray polymorphism, arch library Literal-args). Bare suppressions are forbidden.
4. **Reuse `_scalar()` helper from `build_bars_5m.py` for new lookups, or add an analogous one here.** Per `institutional-rigor.md § 4`, the canonical pattern is now Stage 1a's helper. If `build_bars_5m.py` exports `_scalar`, import it; otherwise add a sibling `_scalar()` here matching the shape exactly (do NOT re-derive a different shape).
5. **Stage doc required.** Write `docs/runtime/stages/pyright-build-daily-features-typing.md` BEFORE editing. `build_daily_features.py` is NEVER_TRIVIAL (hook-enforced in `.claude/hooks/stage-gate-guard.py:94-97`).
6. **Discovery-marker required.** Write `.claude/scratch/discovery-marker.json` with `valid_until` 1–2 hours in the future before the first edit (this plan + pyright output IS the discovery artifact, but the hook checks file presence).

## Error inventory (live counts from `uv run pyright`, 2026-05-15)

Total: **62 errors**. Six categories:

### Category A — `reportOptionalSubscript` (6 sites)
Lines `1715, 1805, 1823, 1839, 1855, 1872`. All are `con.execute(...).fetchone()[0]` on DuckDB COUNT queries — same shape as Stage 1a.

**Fix:** Import the Stage-1a helper or add a local `_scalar()` and call it. Behavior identical, raises `ValueError` on the invariant-violation path.

### Category B — `reportOptionalOperand` (7 sites, lines around 532, 536, 537)
`classify_day_type()` at lines 521–550. Function guards `if None in (...): return None` at line 527, but pyright cannot narrow `daily_high - daily_low` because the `in` check on a tuple doesn't propagate Optional narrowing per-arg.

**Fix:** Replace the tuple-`in` check with explicit per-arg checks. Idiom:
```python
if daily_open is None or daily_high is None or daily_low is None or daily_close is None or atr_20 is None:
    return None
```
This narrows each name to non-None for the rest of the function. **Zero runtime behavior change** — `None in (x, y, z)` and explicit `or` chain are equivalent when all args are `Optional[float]`.

### Category C — pandas Scalar narrowing (~14 sites, lines 323–325, 334, 343, 1031, 1032, 1255)
`bar.close`, `bar.open`, `bar.volume` from `df.itertuples()` are typed as `pandas.Scalar` (a union). `float(scalar)` and `int(scalar)` reject the union even though every concrete branch supports the cast. Same applies to `bar.ts_utc.to_pydatetime()` — `ts_utc` could be 13 unrelated types per the stub.

**Fix at each call site:** narrow via `cast` or by asserting at the row builder boundary. Preferred shape:
```python
from typing import cast
import pandas as pd

ts = cast(pd.Timestamp, bar.ts_utc)
bar_ts = ts.to_pydatetime()
close = float(cast(float, bar.close))  # iff bar.close is genuinely float at runtime
```

**Why `cast` here, not at the dataframe construction:** the `itertuples()` namedtuple type annotations are governed by pandas-stubs, not user code. The narrowing has to happen at the consumer.

**Site 1255** (VWAP computation, ExtensionArray):
```python
vwap_prices = pre_bars["close"].values.astype(float)
vwap_vols = pre_bars["volume"].values.astype(float)
row[f"orb_{label}_vwap"] = round(float((vwap_prices * vwap_vols).sum() / vwap_vols.sum()), 4)
```
`.values` returns `np.ndarray | ExtensionArray`. Then `.astype(float)` returns the same union. Pyright can't prove `*` and `.sum()` work on `ExtensionArray`.

**Fix:** narrow with `np.asarray(...)`:
```python
vwap_prices = np.asarray(pre_bars["close"].values, dtype=float)
vwap_vols = np.asarray(pre_bars["volume"].values, dtype=float)
```
This is functionally identical to `.values.astype(float)` for numeric series and gives pyright the concrete `ndarray` type.

### Category D — Series-cell update on TypedDict-typed None field (6 sites, lines 1050–1065)
`result = {"outcome": None, "mae_r": None, "mfe_r": None}` — pyright infers `dict[str, None]`, so `result["outcome"] = "loss"` fails.

**Fix:** add an explicit type annotation on the dict:
```python
from typing import Any
result: dict[str, Any] = {"outcome": None, "mae_r": None, "mfe_r": None}
```
Or, the cleaner long-term fix per `institutional-rigor.md § 5` ("dead fields are lies"): a TypedDict total=False, since these three keys are documented as the function's return contract:
```python
class _OutcomeResult(TypedDict, total=False):
    outcome: str | None
    mae_r: float | None
    mfe_r: float | None
result: _OutcomeResult = {"outcome": None, "mae_r": None, "mfe_r": None}
```
**Prefer the TypedDict.** Self-documenting and catches future typo-in-key bugs.

### Category E — ExtensionArray vs ndarray on numpy `.searchsorted` and `_wilders_rsi` (3 sites, lines 922, 1366, 1367, 1377, 1378)
Two distinct issues:

**E1 — `_wilders_rsi(closes)` call at line 922.** Function signature is `_wilders_rsi(closes: np.ndarray, period: int = 14) -> float | None`. Caller passes `df["close"].astype(float).values` which is `np.ndarray | ExtensionArray | Categorical`. **The function signature is part of the public surface — DO NOT CHANGE IT** (constraint #2).

**Fix:** narrow at call site with `np.asarray`:
```python
closes_arr = np.asarray(df["close"].astype(float).values, dtype=float)
return _wilders_rsi(closes_arr, period=14)
```

**E2 — `np.searchsorted(all_ts, ...)` at lines 1366/1367.** `all_ts = all_bars_5m_df["ts_utc"].values` is the same union.

**Fix:** at lines 1356–1357, narrow once at construction:
```python
bars_5m_ts = (
    np.asarray(all_bars_5m_df["ts_utc"].values) if not all_bars_5m_df.empty else np.array([])
)
bars_5m_closes = (
    np.asarray(all_bars_5m_df["close"].astype(float).values, dtype=float)
    if not all_bars_5m_df.empty else np.array([])
)
```
This is **a single point of narrowing** instead of casts at every consumer.

Then re-check line 1366: `np.searchsorted(all_ts, ...)` — `all_ts` is local at that scope, defined elsewhere. Read the surrounding context (line 1340-1360) and decide whether `all_ts` is a separate variable that also needs narrowing. **Do NOT rename `all_ts`** without grep-confirming it's used only in this scope (the production-callers grep is on the public surface, but local variables can still leak across function boundaries via closure — verify first).

### Category F — `arch_model()` Literal arg + tail (3 sites, lines 829, 834, 958)
`arch_model(log_returns, vol="Garch", ..., dist="Normal", ...)` — the `arch` library's stubs declare `vol: Literal["GARCH", "ARCH", ...]` and `dist: Literal["normal", "gaussian", ...]`. The code passes title-case strings; library accepts both at runtime but stubs are case-sensitive.

**Fix:** change strings to match the stub-declared literals: `vol="GARCH"`, `dist="normal"`. Verify at runtime with one bar of test data that the GARCH forecast is unchanged (arch normalizes case internally, but always-verify-execution per `integrity-guardian.md § 5`).

**Line 834:** `cond_var**0.5` where `cond_var` is `Scalar`. Coerce: `float(cond_var) ** 0.5`.

**Line 958:** `return round(rsi, 4)` where `rsi` is `floating[Any]`. The function return type is `float | None`. Coerce: `return float(round(rsi, 4))`.

## Verification gates

After EACH category lands (don't batch all six):

1. `uv run pyright pipeline/build_daily_features.py` — error count goes down monotonically, no new errors elsewhere.
2. `python pipeline/check_drift.py` — all checks pass.
3. `pytest tests/test_pipeline/ -x` — green; specifically `test_build_daily_features*` and `test_classify_day_type*` if present.

After ALL categories land:
4. **Behavioral verification (REQUIRED).** Rebuild one day of `daily_features` for one instrument on a scratch DB copy and diff every column vs `main` baseline. Goal: prove zero-byte change in any computed column.
   ```bash
   # On scratch DB (NOT gold.db):
   DUCKDB_PATH=/tmp/df_typing_test.db python pipeline/build_daily_features.py \
       --instrument MGC --start 2026-05-13 --end 2026-05-13 --orb-minutes 5
   # Compare to baseline produced from main HEAD pre-fix.
   ```
   If any column differs → revert and re-investigate. This is the canonical fail-safe per `integrity-guardian.md § 7`.

5. `uv run pyright pipeline/build_daily_features.py` final → **0 errors**.

6. **Self-review pass** per `institutional-rigor.md § 1`. Single commit per category (6 commits expected), each citing the category letter from this plan.

## Stage classification

**IMPLEMENTATION.** NEVER_TRIVIAL per `stage-gate-guard.py`. Stage file required.

**NOT** capital-class (`adversarial-audit-gate.md` does NOT apply): this file is the pipeline writer, not in `trading_app/live/`. No `evidence-auditor` gate required, BUT see § "Optional independent review" below.

## Import surface (full list — do NOT change these symbols)

Confirmed by grep on `pipeline/`, `trading_app/`, `scripts/`, `research/`, `tests/`:

| Symbol | Importers | Type |
|---|---|---|
| `VALID_ORB_MINUTES` | check_drift, run_pipeline, multiple tests | constant tuple |
| `ACTIVE_ORB_MINUTES` | multiple research scripts | constant tuple |
| `GARCH_MIN_PRIOR_CLOSES`, `GARCH_PCT_MIN_PRIOR_VALUES` | check_drift | int constants |
| `COMPRESSION_SESSIONS` | research scripts | tuple |
| `compute_trading_day_utc_range` | 20+ files | function |
| `compute_trading_day` | research scripts | function |
| `_orb_utc_window` | tests, predecessor reference in dst.py | function (PROMOTED to `pipeline.dst.orb_utc_window` — verify whether this private alias still has external callers, per `feedback_canonical_value_unit_verification.md`) |
| `_break_detection_window` | research/_alt_strategy_utils | function |
| `_apply_htf_level_fields`, `_htf_week_key`, `_htf_prior_month_key` | research scripts | functions |
| `_load_htf_seed_rows`, `_load_postpass_seed_rows` | research scripts | functions |
| `_wilders_rsi` | research scripts | function — signature MUST stay `(np.ndarray, int) -> float \| None` |
| `build_daily_features` | run_pipeline, research | main entry |
| `detect_double_break` | research | function |

## Optional independent review

`build_daily_features.py` is downstream of capital decisions (validators read `daily_features`). A typing-only refactor that preserves behavior should be safe, but the file's blast radius is wide. **Optional but recommended:** after all 6 categories land, dispatch `evidence-auditor` (NOT required by `adversarial-audit-gate.md` since the file isn't in `trading_app/live/`, but the auditor's independent context will catch any accidental behavior change the implementer's tests didn't probe).

## Out of scope (DO NOT do in this plan)

- Pyright cleanup elsewhere in `pipeline/` (separate plan if needed)
- Promoting CI pyright to a hard gate (separate plan — 356-error full-repo cleanup)
- Punch-list Stages 2–6 (data backfill, capital fixes, dashboard UX — separate threads)
- Refactoring `build_daily_features.py` structurally — typing-only

## Predecessor context for next session

When you start: read this file end-to-end, then:

1. Confirm `main` is at `8d7ec08c` or later: `git log --oneline -5`.
2. Read the actual pyright output: `uv run pyright pipeline/build_daily_features.py 2>&1 | tail -80`.
3. Verify the live count matches **62**. If different (library upgrade, etc.), revise this plan first — do NOT proceed against stale numerics.
4. Write the stage doc and discovery-marker (per Constraints 5 + 6).
5. Land categories A → F **one commit per category**, verifying after each.
6. Final behavioral verification (gate 4) before declaring done.

If the file has been modified since `8d7ec08c`, re-read the error inventory before starting — line numbers will drift.

## Files NOT to touch in this plan

- Anything under `trading_app/`
- `.github/workflows/ci.yml` (CI gate promotion is a separate plan)
- `pipeline/check_drift.py` (separate plan if drift check needed for new pattern)
- Any other `pipeline/*.py` file (scope-locked to `build_daily_features.py`)
