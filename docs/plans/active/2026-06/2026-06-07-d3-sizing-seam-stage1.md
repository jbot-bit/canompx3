---
status: active
owner: cross-tool
last_reviewed: 2026-06-07
superseded_by: ""
---

# D-3 Sizing-Seam Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the account-survival gate project drawdown at the contract count the live execution engine *would* trade (vol-scaled, capped), instead of a hardcoded 1 micro — closing the "D-3 seam" — without changing any live verdict today (cap stays 1) and without touching the executor.

**Architecture:** Add an opt-in `SizingContext` argument to `_load_lane_trade_paths`. When `None` (every existing caller — correlation, allocation, daily-pnl), behavior is byte-identical to today. ONLY the survival-private caller `_load_profile_daily_scenarios` passes a `SizingContext` (built from `build_profile_portfolio`), which makes the per-trade loop vol-size each trade via the canonical `portfolio.py` sizer, clamp via canonical `max_lots_for_xfa`, and scale ALL contract-derived `TradePath` fields so the downstream scenario math (`_scenario_from_trade_paths`) stays coherent.

**Tech Stack:** Python, duckdb (read-only gold.db), pytest. Canonical modules: `trading_app/account_survival.py`, `trading_app/portfolio.py` (sizer), `trading_app/topstep_scaling_plan.py` (`max_lots_for_xfa`), `trading_app/paper_trader.py` (`_get_median_atr_20`).

**Source spec (audited 4×):** `docs/plans/active/2026-06/2026-06-07-d3-sizing-seam-stage1-design.md`

> **CAPITAL-PATH DISCIPLINE:** This modifies `trading_app/account_survival.py`, a canonical capital-survival gate. Every task is TDD (RED first). NEVER raise `max_contracts`. NEVER arm. After each task run drift (`python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory`). Commit per task.

---

## File Structure

- **Modify** `trading_app/account_survival.py`:
  - new dataclass `SizingContext` (transports equity/risk_pct/per-strategy cap/account_size — does NOT re-encode any ladder).
  - new helper `_lane_atr_by_day(con, instrument, orb_minutes, days) -> dict[date, float]` (sim-local batched ATR query).
  - `_load_lane_trade_paths(...)` gains `size_model: SizingContext | None = None`; sizing block runs only when non-None.
  - `_load_profile_daily_scenarios(...)` builds and passes the `SizingContext`.
  - `_assert_single_micro_sizing` → repurposed/renamed to a parity assertion (see Task 6).
  - comment marking `SurvivalRules.contracts_per_trade_micro` vestigial.
- **Modify** `pipeline/check_drift.py`: new capital-class `requires_db` check binding sim↔engine sizing source.
- **Test** `tests/test_trading_app/test_account_survival.py`: all new assertions.

> **Note — express flag:** measurement confirmed BOTH 50k profiles report `is_express_funded=True` (so `SurvivalRules.starting_balance=0.0`). The sizer MUST use `Portfolio.account_equity` notional, NOT `starting_balance`. This is the load-bearing equity-source rule (Task 3).

---

### Task 1: `SizingContext` dataclass (transport only, no ladder logic)

**Files:**
- Modify: `trading_app/account_survival.py` (near the `TradePath`/`DailyScenario` dataclasses, ~line 189)
- Test: `tests/test_trading_app/test_account_survival.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sizing_context_carries_engine_inputs_and_resolves_cap():
    from trading_app.account_survival import SizingContext
    ctx = SizingContext(
        account_equity=25000.0,
        risk_per_trade_pct=2.0,
        account_size=50000,
        max_contracts_by_strategy={"A": 3, "B": 1},
    )
    # transports the engine's notional equity, never a starting balance
    assert ctx.account_equity == 25000.0
    assert ctx.risk_per_trade_pct == 2.0
    assert ctx.account_size == 50000
    # per-strategy cap lookup; unknown strategy fails closed to 1 (never unbounded)
    assert ctx.max_contracts_for("A") == 3
    assert ctx.max_contracts_for("B") == 1
    assert ctx.max_contracts_for("UNKNOWN") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_sizing_context_carries_engine_inputs_and_resolves_cap -v`
Expected: FAIL — `ImportError: cannot import name 'SizingContext'`.

- [ ] **Step 3: Write minimal implementation**

Add after the `DailyScenario` dataclass in `trading_app/account_survival.py`:

```python
@dataclasses.dataclass(frozen=True)
class SizingContext:
    """Transports the LIVE engine's sizing inputs into the survival sim so the
    gate projects DD at the contract count the engine WOULD trade.

    Transport only — it does NOT re-encode any lot ladder. `account_equity` is
    the Portfolio NOTIONAL (the value execution_engine.py:267 sizes from), NEVER
    SurvivalRules.starting_balance (which is 0.0 for express-funded XFA accounts
    and would zero the sizer → a false PASS).
    """

    account_equity: float
    risk_per_trade_pct: float
    account_size: int
    max_contracts_by_strategy: dict[str, int]

    def max_contracts_for(self, strategy_id: str) -> int:
        # Fail closed to 1 for an unknown lane — never size an unrecognised lane up.
        return int(self.max_contracts_by_strategy.get(strategy_id, 1))
```

Ensure `import dataclasses` is present at the top of the module (it is used by `TradePath`/`DailyScenario`; confirm and add if missing).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_sizing_context_carries_engine_inputs_and_resolves_cap -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "feat(survival): add SizingContext transport for D-3 seam Stage 1"
```

---

### Task 2: Sim-local ATR-by-day helper (own batched query, not feat_dicts)

**Files:**
- Modify: `trading_app/account_survival.py` (new module-level helper near `_load_lane_trade_paths`, ~line 388)
- Test: `tests/test_trading_app/test_account_survival.py`

> **Why a new query:** `_load_strategy_outcomes` (strategy_fitness.py:454) DISCARDS its `feat_dicts` before returning, and is shared by 6 other callers — enriching it would contaminate them. So the sim queries `daily_features` itself. `atr_20` is per-day (0 cross-aperture variance, measured), so `orb_minutes=5` dedups.

- [ ] **Step 1: Write the failing test**

```python
def test_lane_atr_by_day_returns_per_day_atr_map(tmp_path):
    import duckdb
    from trading_app.account_survival import _lane_atr_by_day
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE daily_features (symbol VARCHAR, orb_minutes INT, trading_day DATE, atr_20 DOUBLE)"
    )
    con.execute(
        "INSERT INTO daily_features VALUES "
        "('MNQ',5,DATE '2026-01-02',12.5),"
        "('MNQ',15,DATE '2026-01-02',12.5),"   # other aperture, deduped by orb_minutes=5
        "('MNQ',5,DATE '2026-01-05',NULL)"     # NULL atr → omitted from map
    )
    m = _lane_atr_by_day(con, "MNQ", 5, {date(2026, 1, 2), date(2026, 1, 5)})
    assert m[date(2026, 1, 2)] == 12.5
    assert date(2026, 1, 5) not in m   # NULL not mapped; caller falls back to vol_scalar=1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_lane_atr_by_day_returns_per_day_atr_map -v`
Expected: FAIL — `ImportError: cannot import name '_lane_atr_by_day'`.

- [ ] **Step 3: Write minimal implementation**

Add near `_load_lane_trade_paths` in `trading_app/account_survival.py`:

```python
def _lane_atr_by_day(con, instrument: str, orb_minutes: int, days: set) -> dict:
    """Per-day atr_20 for an instrument over the given trade days (one batched query).

    Uses orb_minutes=5 because atr_20 is a per-day, non-aperture value (verified:
    0 symbol-days with cross-aperture variance). Days with NULL atr_20 are omitted
    — the caller falls back to vol_scalar=1.0 (engine parity, execution_engine.py:280).
    """
    if not days:
        return {}
    rows = con.execute(
        """SELECT trading_day, atr_20 FROM daily_features
           WHERE symbol = ? AND orb_minutes = ? AND atr_20 IS NOT NULL""",
        [instrument, orb_minutes],
    ).fetchall()
    wanted = set(days)
    return {r[0]: float(r[1]) for r in rows if r[0] in wanted}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_lane_atr_by_day_returns_per_day_atr_map -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "feat(survival): sim-local per-day ATR helper for D-3 seam"
```

---

### Task 3: Gate the sizing block in `_load_lane_trade_paths` (None = byte-identical today)

**Files:**
- Modify: `trading_app/account_survival.py:389-459` (`_load_lane_trade_paths`)
- Test: `tests/test_trading_app/test_account_survival.py`

> **Equity rule (CRITICAL):** the sizer is fed `size_model.account_equity` (notional, >0). NEVER `starting_balance`. Guard: if `account_equity <= 0`, raise — a zero-equity DD projection must never be produced silently.

- [ ] **Step 1: Write the failing test (cap-scaling wired + all fields scale)**

```python
def test_load_lane_trade_paths_scales_all_fields_when_size_model_given(monkeypatch):
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths

    base = [
        asv.TradePath(
            trading_day=date(2026, 1, 2), strategy_id="L1",
            entry_ts=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
            exit_ts=datetime(2026, 1, 2, 11, 0, tzinfo=UTC),
            pnl_dollars=40.0, mae_dollars=20.0, mfe_dollars=60.0,
            lots=1, contracts=1, instrument="MNQ",
        )
    ]
    # Stub the data layer so the test is deterministic and offline.
    monkeypatch.setattr(asv, "_load_strategy_snapshot",
                        lambda con, sid: {"instrument": "MNQ", "orb_label": "X", "orb_minutes": 5,
                                          "entry_model": "m", "rr_target": 2.0, "confirm_bars": 1,
                                          "filter_type": "NO_FILTER", "stop_multiplier": 1.0})
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})          # no ATR → vol_scalar 1.0
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})          # ditto

    ctx = SizingContext(account_equity=25000.0, risk_per_trade_pct=2.0,
                        account_size=50000, max_contracts_by_strategy={"L1": 2})
    # vol_scalar=1.0 + cap=2 with ample equity → engine sizer returns >=2, clamp to 2.
    scaled = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
    t = scaled[0]
    assert t.contracts == 2
    assert t.pnl_dollars == 80.0   # 40 * 2
    assert t.mae_dollars == 40.0   # 20 * 2
    assert t.mfe_dollars == 120.0  # 60 * 2


def test_load_lane_trade_paths_unchanged_when_size_model_none(monkeypatch):
    import trading_app.account_survival as asv
    from trading_app.account_survival import _load_lane_trade_paths
    base = [asv.TradePath(trading_day=date(2026, 1, 2), strategy_id="L1",
                          entry_ts=None, exit_ts=None, pnl_dollars=40.0, mae_dollars=20.0,
                          mfe_dollars=60.0, lots=1, contracts=1, instrument="MNQ")]
    monkeypatch.setattr(asv, "_load_strategy_snapshot",
                        lambda con, sid: {"instrument": "MNQ", "orb_label": "X", "orb_minutes": 5,
                                          "entry_model": "m", "rr_target": 2.0, "confirm_bars": 1,
                                          "filter_type": "NO_FILTER", "stop_multiplier": 1.0})
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    out = _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3))  # no size_model
    assert out[0].contracts == 1
    assert out[0].pnl_dollars == 40.0   # untouched


def test_load_lane_trade_paths_fails_closed_on_nonpositive_equity(monkeypatch):
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext, _load_lane_trade_paths
    monkeypatch.setattr(asv, "_load_strategy_snapshot",
                        lambda con, sid: {"instrument": "MNQ", "orb_label": "X", "orb_minutes": 5,
                                          "entry_model": "m", "rr_target": 2.0, "confirm_bars": 1,
                                          "filter_type": "NO_FILTER", "stop_multiplier": 1.0})
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: [])
    ctx = SizingContext(account_equity=0.0, risk_per_trade_pct=2.0,
                        account_size=50000, max_contracts_by_strategy={})
    import pytest
    with pytest.raises(ValueError, match="account_equity"):
        _load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
```

> **Refactor note for Step 3:** the current `_load_lane_trade_paths` inlines outcome loading + TradePath construction. To make the tests above stubbable, extract the existing outcome→TradePath loop (lines ~399-458) into a helper `_build_trade_paths_from_outcomes(con, params, *, as_of_date, effective_stop_multiplier, max_orb_size_pts)` that returns the unscaled `list[TradePath]` (pure move, no behavior change — covered by the existing suite). Also add `_lane_median_atr(con, instrument, days)` delegating per-day to `paper_trader._get_median_atr_20` (Task 5 hardens it; a thin stub returning `{}` is acceptable here so RED is about the gate, not the median).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "load_lane_trade_paths_scales or size_model_none or nonpositive_equity" -v`
Expected: FAIL — `size_model` is not a parameter / helpers not defined.

- [ ] **Step 3: Write minimal implementation**

In `trading_app/account_survival.py`:

1. Extract the existing outcome→TradePath loop into `_build_trade_paths_from_outcomes(...)` returning unscaled `list[TradePath]` (pure move).
2. Change the signature:

```python
def _load_lane_trade_paths(
    con,
    strategy_id: str,
    *,
    as_of_date: date,
    effective_stop_multiplier: float | None = None,
    max_orb_size_pts: float | None = None,
    size_model: "SizingContext | None" = None,
) -> list[TradePath]:
    params = _load_strategy_snapshot(con, strategy_id)
    trades = _build_trade_paths_from_outcomes(
        con, params, as_of_date=as_of_date,
        effective_stop_multiplier=effective_stop_multiplier,
        max_orb_size_pts=max_orb_size_pts,
    )
    if size_model is None:
        return trades   # byte-identical to today — every non-survival caller

    if size_model.account_equity <= 0:
        raise ValueError(
            f"SizingContext.account_equity must be > 0 (got {size_model.account_equity}); "
            "a zero-equity drawdown projection is never a valid survival PASS"
        )

    instrument = params["instrument"]
    cost_spec = get_cost_spec(instrument)
    cap = size_model.max_contracts_for(strategy_id)
    days = {t.trading_day for t in trades}
    atr_by_day = _lane_atr_by_day(con, instrument, params["orb_minutes"], days)
    median_by_day = _lane_median_atr(con, instrument, days)

    scaled: list[TradePath] = []
    for t in trades:
        risk_points = abs(t.mae_dollars) / cost_spec.point_value if cost_spec.point_value else 0.0
        atr = atr_by_day.get(t.trading_day)
        med = median_by_day.get(t.trading_day)
        if atr and med and atr > 0 and med > 0:
            vol_scalar = compute_vol_scalar(atr, med)
        else:
            vol_scalar = 1.0
            logger.warning("survival sizing: vol_scalar=1.0 fallback for %s %s (atr=%s med=%s)",
                           instrument, t.trading_day, atr, med)
        n = compute_position_size_vol_scaled(
            size_model.account_equity, size_model.risk_per_trade_pct, risk_points, cost_spec, vol_scalar
        )
        n = min(n, cap, max_lots_for_xfa(size_model.account_size, size_model.account_equity))
        n = max(n, 0)
        scaled.append(dataclasses.replace(
            t,
            contracts=t.contracts * n if t.contracts else n,
            pnl_dollars=t.pnl_dollars * n,
            mae_dollars=t.mae_dollars * n,
            mfe_dollars=t.mfe_dollars * n,
            lots=lots_for_position(instrument, (t.contracts or 1) * n),
        ))
    return scaled
```

> The risk_points derivation above reuses `mae_dollars` (already dollarized at 1 contract) ÷ point value as the per-trade risk in points — equivalent to the engine's `abs(entry-stop)` because `mae_dollars` for a stopped trade equals stop distance × point value. If the existing loop already retains entry/stop, prefer `abs(entry-stop)` directly. Confirm against `_build_trade_paths_from_outcomes` at implementation time and use whichever is exact.

Ensure imports at top: `compute_vol_scalar, compute_position_size_vol_scaled` from `trading_app.portfolio`; `max_lots_for_xfa, lots_for_position` already imported (`:51`); `get_cost_spec` already used (`:417`).

- [ ] **Step 4: Run tests to verify they pass + existing suite green**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "load_lane_trade_paths or scenario_from_trade_paths" -v`
Expected: PASS (new) and the existing `_scenario_from_trade_paths`/lane tests still PASS (proves the extract was a pure move).

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "feat(survival): opt-in size_model scaling in _load_lane_trade_paths (None=unchanged)"
```

---

### Task 4: Wire `SizingContext` into `_load_profile_daily_scenarios` (survival-private only)

**Files:**
- Modify: `trading_app/account_survival.py:541-581` (`_load_profile_daily_scenarios`)
- Test: `tests/test_trading_app/test_account_survival.py`

- [ ] **Step 1: Write the failing test (blast-radius separation + measured behavior)**

```python
def test_profile_scenarios_pass_size_model_but_daily_pnl_does_not(monkeypatch):
    """Survival path scales; correlation/allocation path (_load_lane_daily_pnl) stays raw."""
    import trading_app.account_survival as asv
    captured = {}

    real = asv._load_lane_trade_paths
    def spy(con, sid, *, as_of_date, effective_stop_multiplier=None, max_orb_size_pts=None, size_model=None):
        captured.setdefault(sid, []).append(size_model)
        return real(con, sid, as_of_date=as_of_date,
                    effective_stop_multiplier=effective_stop_multiplier,
                    max_orb_size_pts=max_orb_size_pts, size_model=size_model)
    monkeypatch.setattr(asv, "_load_lane_trade_paths", spy)

    asv._load_profile_daily_scenarios("topstep_50k_mnq_auto", as_of_date=date(2026, 6, 1))
    # survival path passed a non-None size_model for every lane
    assert all(any(sm is not None for sm in v) for v in captured.values())

    captured.clear()
    asv._load_lane_daily_pnl(  # correlation/allocation entry — must pass size_model=None
        __import__("duckdb").connect(str(asv.GOLD_DB_PATH), read_only=True),
        next(iter(captured), None) or _first_lane("topstep_50k_mnq_auto"),
    )
```

> **Note:** `_load_lane_daily_pnl` takes a lane dict; the real call path in `lane_correlation.py:190` builds it. For this test, prefer asserting at the unit level: call `_load_lane_daily_pnl` and assert (via the same spy) that it invoked `_load_lane_trade_paths` with `size_model=None`. Replace the placeholder `_first_lane` with a `get_profile_lane_definitions(...)[0]` lookup at implementation time.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_profile_scenarios_pass_size_model_but_daily_pnl_does_not -v`
Expected: FAIL — `_load_profile_daily_scenarios` does not yet build/pass a `size_model`.

- [ ] **Step 3: Write minimal implementation**

In `_load_profile_daily_scenarios`, after `profile = get_profile(profile_id)` and `lane_defs = ...`, build the context once and pass it into the loop's `_load_lane_trade_paths` call:

```python
    from trading_app.portfolio import build_profile_portfolio
    _pf = build_profile_portfolio(profile_id=profile_id)
    size_model = SizingContext(
        account_equity=_pf.account_equity,
        risk_per_trade_pct=_pf.risk_per_trade_pct,
        account_size=profile.account_size,
        max_contracts_by_strategy={s.strategy_id: s.max_contracts for s in _pf.strategies},
    )
```

Then in the existing `for lane in lane_defs:` loop, add `size_model=size_model` to the `_load_lane_trade_paths(...)` call (`:568`). Leave `_load_lane_daily_pnl` (`:368`) untouched — it passes no `size_model`.

- [ ] **Step 4: Run test + the byte-identical pin**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "profile_scenarios or load_lane_daily_pnl" -v`
Expected: PASS. `_load_lane_daily_pnl` proven to receive `size_model=None`.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "feat(survival): wire SizingContext into survival-private scenario builder"
```

---

### Task 5: Canonical trailing median provider (`_lane_median_atr`)

**Files:**
- Modify: `trading_app/account_survival.py` (replace the Task-3 stub `_lane_median_atr`)
- Test: `tests/test_trading_app/test_account_survival.py`

> Delegates to `paper_trader._get_median_atr_20` (canonical, already TRAILING: `paper_trader.py:143` `trading_day < ?`). No parallel SQL.

- [ ] **Step 1: Write the failing test (delegation + look-ahead)**

```python
def test_lane_median_atr_delegates_to_canonical_trailing_helper(monkeypatch):
    import trading_app.account_survival as asv
    calls = []
    def fake_median(con, instrument, trading_day, lookback_days=252):
        calls.append((instrument, trading_day))
        return 10.0
    monkeypatch.setattr("trading_app.paper_trader._get_median_atr_20", fake_median)
    m = asv._lane_median_atr(None, "MNQ", {date(2026, 1, 2), date(2026, 1, 3)})
    assert m == {date(2026, 1, 2): 10.0, date(2026, 1, 3): 10.0}
    # must call the CANONICAL trailing helper once per day (no re-encoded SQL)
    assert set(d for _, d in calls) == {date(2026, 1, 2), date(2026, 1, 3)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_lane_median_atr_delegates_to_canonical_trailing_helper -v`
Expected: FAIL — stub returns `{}` / no delegation.

- [ ] **Step 3: Write minimal implementation**

Replace the stub:

```python
def _lane_median_atr(con, instrument: str, days: set) -> dict:
    """Trailing 252d median atr_20 as-of each day, via the CANONICAL helper
    (paper_trader._get_median_atr_20 — already trailing: trading_day < ?).
    """
    from trading_app.paper_trader import _get_median_atr_20
    out: dict = {}
    for d in days:
        med = _get_median_atr_20(con, instrument, d)
        if med:
            out[d] = float(med)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py::test_lane_median_atr_delegates_to_canonical_trailing_helper -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "feat(survival): canonical trailing-median provider for D-3 sizing"
```

---

### Task 5b: Fallback + look-ahead + structural-loss tests (spec oracles c, e, f)

**Files:**
- Test: `tests/test_trading_app/test_account_survival.py`
- (Implementation already landed in Tasks 3 + 5; this task adds the missing dedicated pins.)

> These three oracles were implied by Tasks 3/5 but had no explicit test. Add them now.

- [ ] **Step 1: Write the failing/confirming tests**

```python
def test_d3_null_atr_day_falls_back_to_scalar_one_not_fail(monkeypatch):
    """Oracle (c): a priced day with NULL atr_20 uses vol_scalar=1.0 (engine parity)
    — it is NOT a structural failure; sizing still produces a valid count."""
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext
    base = [asv.TradePath(trading_day=date(2026, 1, 2), strategy_id="L1",
                          entry_ts=None, exit_ts=None, pnl_dollars=40.0, mae_dollars=20.0,
                          mfe_dollars=60.0, lots=1, contracts=1, instrument="MNQ")]
    monkeypatch.setattr(asv, "_load_strategy_snapshot",
                        lambda con, sid: {"instrument": "MNQ", "orb_label": "X", "orb_minutes": 5,
                                          "entry_model": "m", "rr_target": 2.0, "confirm_bars": 1,
                                          "filter_type": "NO_FILTER", "stop_multiplier": 1.0})
    monkeypatch.setattr(asv, "_build_trade_paths_from_outcomes", lambda *a, **k: list(base))
    monkeypatch.setattr(asv, "_lane_atr_by_day", lambda *a, **k: {})   # NULL/missing atr
    monkeypatch.setattr(asv, "_lane_median_atr", lambda *a, **k: {})
    ctx = SizingContext(account_equity=25000.0, risk_per_trade_pct=2.0,
                        account_size=50000, max_contracts_by_strategy={"L1": 2})
    out = asv._load_lane_trade_paths(None, "L1", as_of_date=date(2026, 1, 3), size_model=ctx)
    # vol_scalar=1.0 fallback → still sizes (>=1), does NOT raise, does NOT zero
    assert out[0].contracts >= 1
    assert out[0].pnl_dollars >= 40.0


def test_d3_median_is_trailing_not_full_history(monkeypatch):
    """Oracle (e): the median provider must be point-in-time (a trade on day D
    must not see ATR on/after D). Verified by delegating to the canonical
    _get_median_atr_20, whose SQL uses `trading_day < ?` (paper_trader.py:143)."""
    import trading_app.account_survival as asv
    seen_days = []
    def fake_median(con, instrument, trading_day, lookback_days=252):
        seen_days.append(trading_day)
        return 10.0
    monkeypatch.setattr("trading_app.paper_trader._get_median_atr_20", fake_median)
    asv._lane_median_atr(None, "MNQ", {date(2026, 1, 10)})
    # the helper passes the trade day to the canonical trailing fn (which filters < day)
    assert seen_days == [date(2026, 1, 10)]


def test_d3_structural_atr_loss_fails_closed():
    """Oracle (f): if ATR cannot be obtained structurally (query raises), the
    sizing path must not silently fall back — it must propagate so the gate
    fails closed. Simulate by pointing the helper at a table-less connection."""
    import duckdb, pytest
    from trading_app.account_survival import _lane_atr_by_day
    con = duckdb.connect(":memory:")  # no daily_features table
    with pytest.raises(duckdb.CatalogException):
        _lane_atr_by_day(con, "MNQ", 5, {date(2026, 1, 2)})
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "null_atr_day or median_is_trailing or structural_atr_loss" -v`
Expected: PASS for (c)/(e); (f) PASS if `_lane_atr_by_day` lets the query error propagate (it must NOT swallow it — confirm no try/except around the query in Task 2; if one was added, remove it so structural loss is loud).

- [ ] **Step 3: (If (f) fails) remove any swallow in `_lane_atr_by_day`**

Ensure the `con.execute(...)` in `_lane_atr_by_day` is NOT wrapped in a bare `except` — a missing table / query error must propagate (structural loss = fail closed). NULL handling is via `WHERE atr_20 IS NOT NULL` + the membership filter, not exception swallowing.

- [ ] **Step 4: Re-run to confirm all three pass**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "null_atr_day or median_is_trailing or structural_atr_loss" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trading_app/test_account_survival.py
git commit -m "test(survival): pin D-3 NULL-atr fallback, trailing-median, structural fail-closed"
```

---

### Task 6: Repurpose `_assert_single_micro_sizing` → sizing-parity assertion

**Files:**
- Modify: `trading_app/account_survival.py:836-869` (`_assert_single_micro_sizing`)
- Modify: existing tests at `tests/test_trading_app/test_account_survival.py:910-955` (rewrite)
- Test: `tests/test_trading_app/test_account_survival.py`

> The old guard FORBIDS `max_contracts != 1` (handcuff). Now the honest sim itself fails the gate at unsafe size (measured: operational_pass_prob 0.99→0.10 at cap=2). The guard's remaining job is narrower: confirm the sim CAN size like the engine (build succeeds, equity > 0) — fail closed if it cannot prove parity. It NO LONGER forbids cap > 1.

- [ ] **Step 1: Rewrite the failing tests**

```python
def test_sizing_parity_ok_when_portfolio_builds_with_positive_equity(monkeypatch):
    import trading_app.account_survival as asv
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is True

def test_sizing_parity_fails_closed_on_builder_error(monkeypatch):
    import trading_app.account_survival as asv
    def boom(*a, **k):
        raise RuntimeError("no portfolio")
    monkeypatch.setattr(asv, "build_profile_portfolio", boom)
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is False
    assert "parity" in msg.lower()

def test_sizing_parity_fails_closed_on_nonpositive_equity(monkeypatch):
    import trading_app.account_survival as asv
    class _PF:
        account_equity = 0.0
        risk_per_trade_pct = 2.0
        strategies = []
    monkeypatch.setattr(asv, "build_profile_portfolio", lambda **k: _PF())
    ok, msg = asv._assert_sizing_parity("topstep_50k_mnq_auto")
    assert ok is False
    assert "equity" in msg.lower()

def test_evaluate_profile_survival_gate_fails_closed_on_parity_violation(monkeypatch):
    # COPY the body of the existing
    # test_evaluate_profile_survival_gate_fails_closed_on_sizing_parity_violation
    # (tests/test_trading_app/test_account_survival.py:982-1010) VERBATIM, changing
    # exactly ONE line: the patched symbol name. That existing test patches the guard
    # and a fake_load returning clean scenarios, then asserts gate_pass is False.
    monkeypatch.setattr(asv, "_assert_sizing_parity", lambda pid: (False, "forced"))
    # ... rest identical to the existing harness (fake _current_survival_canonical_inputs,
    #     fake_load with clean DailyScenarios, call evaluate_profile_survival, assert
    #     summary.gate_pass is False). Do not invent a new harness — reuse the proven one.
```

> The existing test at `:982` already exercises the exact gate branch (it even guards the
> `log` NameError regression). Copy it, rename `_assert_single_micro_sizing` →
> `_assert_sizing_parity` on the `monkeypatch.setattr` line, and delete the old
> `test_single_micro_sizing_*` trio (:910/:927/:945) which assert the now-removed
> forbid-cap>1 behavior — they are replaced by the parity tests above.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "sizing_parity or parity_violation" -v`
Expected: FAIL — `_assert_sizing_parity` not defined.

- [ ] **Step 3: Write minimal implementation**

Replace `_assert_single_micro_sizing` with:

```python
def _assert_sizing_parity(profile_id: str) -> tuple[bool, str]:
    """D-3 sizing-parity guard. The survival sim now sizes like the live engine
    (vol-scaled, capped). This guard no longer FORBIDS max_contracts > 1 — the
    honest sim fails the operational gate at unsafe size on its own. It fails
    CLOSED only when parity cannot be PROVEN: the portfolio can't be built, or
    the equity that would feed the sizer is non-positive (a $0-equity DD
    projection is never a valid PASS — express-funded starting_balance is 0.0).
    """
    try:
        portfolio = build_profile_portfolio(profile_id=profile_id)
    except Exception as e:  # fail closed — cannot prove parity
        return (False, f"sizing-parity unprovable: build_profile_portfolio failed: {e}")
    if portfolio.account_equity <= 0:
        return (False,
                f"sizing-parity FAILED: portfolio account_equity={portfolio.account_equity} "
                "<= 0 would zero the sizer and project DD=$0 (false PASS)")
    return (True, f"sizing parity OK ({len(portfolio.strategies)} lanes; "
                  f"equity={portfolio.account_equity})")
```

Update the single caller in `evaluate_profile_survival` (`:906`): rename `_assert_single_micro_sizing` → `_assert_sizing_parity` and keep the `sizing_parity_ok` AND into `gate_pass` (`:909`) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "sizing_parity or parity_violation or evaluate_profile_survival" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py tests/test_trading_app/test_account_survival.py
git commit -m "refactor(survival): D-3 guard forbid-cap>1 -> prove-sizing-parity (equity>0)"
```

---

### Task 7: Mark `contracts_per_trade_micro` vestigial + drift check (sim↔engine same sizer)

**Files:**
- Modify: `trading_app/account_survival.py:130` (comment only)
- Modify: `pipeline/check_drift.py` (new capital-class `requires_db` check)
- Test: `tests/test_pipeline/` (new test for the drift check)

- [ ] **Step 1: Write the failing test for the drift check**

```python
def test_check_survival_engine_sizer_parity_detects_fork(tmp_path):
    from pipeline.check_drift import check_survival_engine_sizer_parity
    ok, msg = check_survival_engine_sizer_parity()
    # canonical: both the survival sim and the engine import the sizer from
    # trading_app.portfolio; the check passes on a clean tree.
    assert ok, msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline/test_check_survival_sizer_parity.py -v`
Expected: FAIL — `check_survival_engine_sizer_parity` not defined.

- [ ] **Step 3: Write minimal implementation**

In `pipeline/check_drift.py`, add a check asserting both `trading_app/account_survival.py` and `trading_app/execution_engine.py` reference `compute_position_size_vol_scaled` from `trading_app.portfolio` (AST/import scan — catches a future re-fork of the seam). Register it in the `CHECKS` tuple right after the existing C11 sizing-parity check, with `requires_db=False` (it is a source-import scan) and a clear blocking label. Add `contracts_per_trade_micro` comment in `account_survival.py:130`:

```python
    contracts_per_trade_micro: int  # VESTIGIAL: set but never read by simulate_survival;
    # DD is pre-scaled at TradePath construction. Do not treat as a sizing knob (D-3 Stage 1).
```

- [ ] **Step 4: Run test + drift**

Run: `python -m pytest tests/test_pipeline/test_check_survival_sizer_parity.py -v`
Expected: PASS.
Run: `python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory`
Expected: SUMMARY clean (0 fail), new check listed PASS.

- [ ] **Step 5: Commit**

```bash
git add trading_app/account_survival.py pipeline/check_drift.py tests/test_pipeline/test_check_survival_sizer_parity.py
git commit -m "feat(drift): bind survival sim <-> engine to one canonical sizer + flag vestigial field"
```

---

### Task 8: Measured-behavior integration assertions + full verification

**Files:**
- Test: `tests/test_trading_app/test_account_survival.py`

> These pin the MEASURED relationships (2026-06-07 run): rolling_dd ≈ ×n linear; operational_pass_prob monotone ↓; cap=1 byte-identical; max_open_lots steps at lot boundary not per-contract.

- [ ] **Step 1: Write the failing integration tests**

```python
def test_d3_cap1_scenarios_byte_identical_to_unscaled(monkeypatch):
    """cap=1 via size_model must equal today's unscaled scenarios field-for-field."""
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext
    pid, as_of = "topstep_50k_mnq_auto", date(2026, 6, 1)
    base = asv._load_profile_daily_scenarios(pid, as_of_date=as_of)  # size_model wired (cap=1 today)
    # Force a raw-baseline by patching the cap to 1 explicitly and comparing fields:
    scen = base[0]
    assert scen.total_pnl_dollars == scen.total_pnl_dollars  # smoke; replace with golden compare
    # Golden compare: snapshot total_pnl/min_delta/max_delta/max_open_lots for a fixed seed/profile
    # against a committed fixture built at cap=1 (the current production value).

def test_d3_rolling_dd_scales_linearly_and_pass_prob_drops():
    """MEASURED 2026-06-07: rolling_dd x2.00 at cap=2; operational_pass_prob 0.99->0.10."""
    import trading_app.account_survival as asv
    from trading_app.account_survival import SizingContext
    pid, as_of = "topstep_50k_mnq_auto", date(2026, 6, 1)
    pf = asv.build_profile_portfolio(profile_id=pid)

    def scenarios_at(cap):
        # build via the same survival-private path but override caps to `cap`
        ctx = SizingContext(account_equity=pf.account_equity, risk_per_trade_pct=pf.risk_per_trade_pct,
                            account_size=asv.get_profile(pid).account_size,
                            max_contracts_by_strategy={s.strategy_id: cap for s in pf.strategies})
        # reuse a helper that maps lanes->scaled scenarios (extract in Task 4 if needed)
        return asv._scenarios_for_context(pid, as_of, ctx)

    s1, s2 = scenarios_at(1), scenarios_at(2)
    dd1 = asv._max_observed_rolling_drawdown(s1, horizon_days=90)
    dd2 = asv._max_observed_rolling_drawdown(s2, horizon_days=90)
    assert 1.8 * dd1 <= dd2 <= 2.2 * dd1   # linear ~x2 (measured x2.00), tolerance for multi-seed
    rules = asv._build_rules(asv.get_profile(pid))
    p1 = asv.simulate_survival(s1, rules, horizon_days=90, n_paths=10000, seed=7)["operational_pass_probability"]
    p2 = asv.simulate_survival(s2, rules, horizon_days=90, n_paths=10000, seed=7)["operational_pass_probability"]
    assert p2 < p1            # monotone decreasing (measured 0.99 -> 0.10)
    assert p1 > 0.90 and p2 < 0.50   # cap=2 flips operational gate False on the live profile
```

> Implementation note: if a clean `_scenarios_for_context(profile_id, as_of, ctx)` seam does not already exist after Task 4, extract one (the lane→scaled-scenario mapping) so this test does not duplicate logic. Build the `cap=1` golden fixture from the current production scenario values and commit it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -k "d3_cap1 or rolling_dd_scales" -v`
Expected: FAIL — `_scenarios_for_context` / golden fixture not present.

- [ ] **Step 3: Write minimal implementation**

Extract `_scenarios_for_context(profile_id, as_of_date, size_model) -> list[DailyScenario]` from the Task-4 loop body (pure refactor); build and commit the cap=1 golden fixture from current production values.

- [ ] **Step 4: Run full survival suite + drift**

Run: `python -m pytest tests/test_trading_app/test_account_survival.py -q`
Expected: ALL PASS.
Run: `python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory 2>&1 | grep -iE "survival|sizing|parity|SUMMARY"`
Expected: SUMMARY clean, sizing/parity checks PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_trading_app/test_account_survival.py trading_app/account_survival.py
git commit -m "test(survival): pin measured D-3 behavior (linear rolling_dd, monotone pass-prob)"
```

---

## Final Verification (after all tasks)

- [ ] Run full survival + pipeline suites:
  `python -m pytest tests/test_trading_app/test_account_survival.py tests/test_pipeline/test_check_survival_sizer_parity.py -q`
- [ ] Run drift: `python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory`
- [ ] Confirm NO live verdict change: a fresh survival run on `topstep_50k_mnq_auto` (cap=1 today) reconciles to the known live DD ≈ $1,535.22 vs $1,800 budget and `gate_pass` unchanged.
- [ ] Ruff: `ruff check trading_app/account_survival.py pipeline/check_drift.py`
- [ ] **Do NOT push.** Capital path — operator pushes / arms only from the front-end.

## Out of scope (later stages, still gated)
- Stage 2: scaling-plan lot ladder as canonical config (raise the cap).
- Stage 3: lift `max_contracts` off 1 (capital flip — operator `--live` GO + adversarial audit + bracket audit `9b3fc530`).
- Stage 4: income re-model with the per-day growing-balance ladder (resolve the `cap_balance=notional` Stage-1 simplification).
