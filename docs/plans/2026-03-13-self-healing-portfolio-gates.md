# Self-Healing Portfolio Gates — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add seasonal gating, weight overrides, and auto-recovery to `LiveStrategySpec` so demoted strategies self-heal when rolling eval shows improvement.

**Architecture:** Three new optional fields on the frozen dataclass, with build-time logic in the existing core spec loop. Recovery piggybacks on the rolling eval match already loaded — zero extra DB queries.

**Tech Stack:** Python dataclass, DuckDB (existing), pytest

**Design doc:** `docs/plans/2026-03-13-self-healing-portfolio-gates-design.md`

---

### Task 1: Add new fields to LiveStrategySpec

**Files:**
- Modify: `trading_app/live_config.py:41-57` (LiveStrategySpec dataclass)
- Test: `tests/test_trading_app/test_live_config.py`

**Step 1: Write failing tests for new fields**

Add to `TestLiveStrategySpec` class in `tests/test_trading_app/test_live_config.py`:

```python
def test_new_fields_default_none(self):
    """New gate fields default to None (no gate active)."""
    spec = LiveStrategySpec("fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None)
    assert spec.active_months is None
    assert spec.weight_override is None
    assert spec.recovery_expr_threshold is None

def test_active_months_field(self):
    spec = LiveStrategySpec(
        "fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None,
        active_months=frozenset({11, 12, 1, 2}),
    )
    assert spec.active_months == frozenset({11, 12, 1, 2})
    assert 1 in spec.active_months
    assert 6 not in spec.active_months

def test_weight_override_field(self):
    spec = LiveStrategySpec(
        "fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None,
        weight_override=0.5,
    )
    assert spec.weight_override == 0.5

def test_recovery_threshold_field(self):
    spec = LiveStrategySpec(
        "fam1", "core", "TOKYO_OPEN", "E1", "ORB_G4", None,
        weight_override=0.5,
        recovery_expr_threshold=0.25,
    )
    assert spec.recovery_expr_threshold == 0.25
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_app/test_live_config.py::TestLiveStrategySpec -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'active_months'`

**Step 3: Add fields to dataclass**

In `trading_app/live_config.py`, add after `exclude_instruments`:

```python
    # Seasonal gate: only trade in these calendar months (1=Jan..12=Dec).
    # None = all months. Outside active months -> weight=0, variant not loaded.
    active_months: frozenset[int] | None = None
    # Manual weight demotion (e.g. 0.5 for decay, 0.0 to disable).
    # None = use default (1.0 for core, fitness-dependent for regime).
    weight_override: float | None = None
    # Auto-recovery: if rolling eval ExpR exceeds this threshold,
    # ignore weight_override and promote back to weight=1.0.
    # None = no auto-recovery. Only fires when source="rolling".
    recovery_expr_threshold: float | None = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trading_app/test_live_config.py::TestLiveStrategySpec -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add trading_app/live_config.py tests/test_trading_app/test_live_config.py
git commit -m "feat: add active_months, weight_override, recovery_expr_threshold to LiveStrategySpec"
```

---

### Task 2: Implement seasonal gate with `as_of_date` parameter

**Files:**
- Modify: `trading_app/live_config.py` — add `as_of_date: date | None = None` param to `build_live_portfolio`, implement seasonal gate in core spec loop
- Test: `tests/test_trading_app/test_live_config.py`

**Design ref:** `as_of_date` avoids `datetime.now()` ambiguity at Brisbane/UTC date boundary.
If None, defaults to `date.today()`. Month resolution: `as_of_date.month`.
Seasonal is a hard skip — no variant loaded, no DB query. Recovery cannot override seasonality.

**Step 1: Write failing tests for seasonal skip**

Add to `tests/test_trading_app/test_live_config.py`:

```python
from datetime import date
from unittest.mock import patch
from trading_app.live_config import build_live_portfolio

class TestSeasonalGate:
    def _seed_strategy(self, db_path, sid="MGC_CME_REOPEN_E2_RR1.5_CB1_ORB_G4_FAST10"):
        """Seed a valid CME_REOPEN FAST10 strategy into all required tables."""
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status, orb_minutes,
                fdr_significant)
            VALUES (?, 'MGC', 'CME_REOPEN', 'E2', 1.5, 1, 'ORB_G4_FAST10',
                    0.35, 0.54, 108, 1.1, 3.7, 'active', 5, TRUE)
        """, [sid])
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
                entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
                win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
            VALUES (?, 'MGC', 'CME_REOPEN', 'E2', 1.5, 1, 'ORB_G4_FAST10',
                    0.35, 0.54, 108, 1.1, 3.7, 6.0)
        """, [sid])
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'CME_REOPEN', 'ORB_G4_FAST10', 'E2', 5, 1, 1.5, 'ONLY_RR',
                    1.1, 3.7, 108, 0.35, 20.0)
        """)
        con.close()

    def test_out_of_season_skips_strategy(self, live_config_db):
        """Strategy with active_months skipped when as_of_date month not in set."""
        self._seed_strategy(live_config_db)
        test_spec = LiveStrategySpec(
            "CME_REOPEN_E2_ORB_G4_FAST10", "core", "CME_REOPEN", "E2",
            "ORB_G4_FAST10", None,
            active_months=frozenset({11, 12, 1, 2}),
        )
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db, instrument="MGC",
                as_of_date=date(2026, 6, 15),  # June = out of season
            )
        assert len(portfolio.strategies) == 0
        assert any("SEASONAL" in n for n in notes)

    def test_in_season_loads_strategy(self, live_config_db):
        """Strategy with active_months loads when as_of_date month IS in set."""
        self._seed_strategy(live_config_db)
        test_spec = LiveStrategySpec(
            "CME_REOPEN_E2_ORB_G4_FAST10", "core", "CME_REOPEN", "E2",
            "ORB_G4_FAST10", None,
            active_months=frozenset({11, 12, 1, 2}),
        )
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db, instrument="MGC",
                as_of_date=date(2026, 1, 15),  # January = in season
            )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 1.0

    def test_as_of_date_defaults_to_today(self, live_config_db):
        """When as_of_date is None, defaults to date.today()."""
        self._seed_strategy(live_config_db)
        test_spec = LiveStrategySpec(
            "CME_REOPEN_E2_ORB_G4_FAST10", "core", "CME_REOPEN", "E2",
            "ORB_G4_FAST10", None,
            # No active_months — always loads
        )
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db, instrument="MGC",
                # as_of_date=None (default)
            )
        assert len(portfolio.strategies) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_app/test_live_config.py::TestSeasonalGate -v`
Expected: FAIL — `as_of_date` parameter not accepted

**Step 3: Add `as_of_date` parameter and seasonal gate**

In `build_live_portfolio` signature (line 433), add `as_of_date: date | None = None` parameter.
Add `from datetime import date` import at top of file.
At the start of `build_live_portfolio` body:
```python
    if as_of_date is None:
        as_of_date = date.today()
```

In the core spec loop (line 470), after `exclude_instruments` check and before rolling eval match lookup:
```python
            # --- Seasonal gate (hard skip — no variant loaded) ---
            if spec.active_months is not None:
                if as_of_date.month not in spec.active_months:
                    notes.append(
                        f"SEASONAL: {spec.family_id} -- month {as_of_date.month} "
                        f"not in {sorted(spec.active_months)}"
                    )
                    continue
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trading_app/test_live_config.py::TestSeasonalGate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add trading_app/live_config.py tests/test_trading_app/test_live_config.py
git commit -m "feat: seasonal gate with as_of_date parameter — no bare datetime.now()"
```

---

### Task 3: Implement weight override + auto-recovery using `rolling_avg_expectancy_r`

**Files:**
- Modify: `trading_app/live_config.py` — weight resolution in core spec loop
- Test: `tests/test_trading_app/test_live_config.py`

**Design ref:** Recovery checks `match["rolling_avg_expectancy_r"]` (family rolling average across
all windows), NOT `match["expectancy_r"]` (latest single window). This was threaded into
`rolling_portfolio.py` in Task 1. Recovery only fires when `source == "rolling"` — baseline
fallback cannot trigger recovery. Institutional grounding: half-Kelly (Chan Ch.8) for 0.5
default demotion; Man AHL staged lifecycle for recovery-must-pass-same-gates.

**Step 1: Write failing tests**

Add to `tests/test_trading_app/test_live_config.py`:

```python
class TestWeightOverrideAndRecovery:
    def _seed_strategy(self, db_path, strategy_id="MGC_TEST_E2_RR1.0_CB1_VOL_RV12_N20",
                       expr=0.30, median_risk=5.0):
        """Seed a valid CME_PRECLOSE VOL strategy."""
        con = duckdb.connect(str(db_path))
        con.execute("""
            INSERT INTO validated_setups (strategy_id, instrument, orb_label, entry_model,
                rr_target, confirm_bars, filter_type, expectancy_r, win_rate,
                sample_size, sharpe_ratio, max_drawdown_r, status, orb_minutes,
                fdr_significant)
            VALUES (?, 'MGC', 'CME_PRECLOSE', 'E2', 1.0, 1, 'VOL_RV12_N20',
                    ?, 0.71, 142, 2.29, 4.0, 'active', 15, TRUE)
        """, [strategy_id, expr])
        con.execute("""
            INSERT INTO experimental_strategies (strategy_id, instrument, orb_label,
                entry_model, rr_target, confirm_bars, filter_type, expectancy_r,
                win_rate, sample_size, sharpe_ratio, max_drawdown_r, median_risk_points)
            VALUES (?, 'MGC', 'CME_PRECLOSE', 'E2', 1.0, 1, 'VOL_RV12_N20',
                    ?, 0.71, 142, 2.29, 4.0, ?)
        """, [strategy_id, expr, median_risk])
        con.execute("""
            INSERT INTO family_rr_locks (instrument, orb_label, filter_type, entry_model,
                orb_minutes, confirm_bars, locked_rr, method,
                sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
            VALUES ('MGC', 'CME_PRECLOSE', 'VOL_RV12_N20', 'E2', 15, 1, 1.0, 'ONLY_RR',
                    2.29, 4.0, 142, ?, 30.0)
        """, [expr])
        con.close()

    def test_weight_override_applied(self, live_config_db):
        """Strategy with weight_override uses that weight instead of 1.0."""
        self._seed_strategy(live_config_db)
        test_spec = LiveStrategySpec(
            "CME_PRECLOSE_E2_VOL_RV12_N20", "core", "CME_PRECLOSE", "E2",
            "VOL_RV12_N20", None,
            weight_override=0.5,
        )
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            portfolio, notes = build_live_portfolio(
                db_path=live_config_db, instrument="MGC"
            )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5
        assert any("DEMOTED" in n for n in notes)

    def test_recovery_promotes_back_using_rolling_avg(self, live_config_db):
        """Rolling avg ExpR (family average) above threshold promotes to weight=1.0."""
        self._seed_strategy(live_config_db, expr=0.30)
        test_spec = LiveStrategySpec(
            "CME_PRECLOSE_E2_VOL_RV12_N20", "core", "CME_PRECLOSE", "E2",
            "VOL_RV12_N20", None,
            weight_override=0.5,
            recovery_expr_threshold=0.25,
        )
        mock_rolling = [{
            "strategy_id": "MGC_TEST_E2_RR1.0_CB1_VOL_RV12_N20",
            "instrument": "MGC", "orb_label": "CME_PRECLOSE",
            "entry_model": "E2", "filter_type": "VOL_RV12_N20",
            "rr_target": 1.0, "confirm_bars": 1, "orb_minutes": 15,
            "expectancy_r": 0.30, "win_rate": 0.71,
            "sample_size": 142, "sharpe_ratio": 2.29,
            "max_drawdown_r": 4.0, "median_risk_points": 5.0,
            "stop_multiplier": 1.0,
            "rolling_avg_expectancy_r": 0.28,  # family avg across windows
            "rolling_weighted_stability": 0.85,
        }]
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            with patch("trading_app.live_config.load_rolling_validated_strategies",
                       return_value=mock_rolling):
                portfolio, notes = build_live_portfolio(
                    db_path=live_config_db, instrument="MGC"
                )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 1.0
        assert any("RECOVERED" in n for n in notes)

    def test_recovery_does_not_fire_below_threshold(self, live_config_db):
        """Rolling avg ExpR below threshold keeps weight_override."""
        self._seed_strategy(live_config_db, expr=0.15)
        test_spec = LiveStrategySpec(
            "CME_PRECLOSE_E2_VOL_RV12_N20", "core", "CME_PRECLOSE", "E2",
            "VOL_RV12_N20", None,
            weight_override=0.5,
            recovery_expr_threshold=0.25,
        )
        mock_rolling = [{
            "strategy_id": "MGC_TEST_E2_RR1.0_CB1_VOL_RV12_N20",
            "instrument": "MGC", "orb_label": "CME_PRECLOSE",
            "entry_model": "E2", "filter_type": "VOL_RV12_N20",
            "rr_target": 1.0, "confirm_bars": 1, "orb_minutes": 15,
            "expectancy_r": 0.15, "win_rate": 0.60,
            "sample_size": 142, "sharpe_ratio": 1.0,
            "max_drawdown_r": 5.0, "median_risk_points": 5.0,
            "stop_multiplier": 1.0,
            "rolling_avg_expectancy_r": 0.18,  # below 0.25 threshold
            "rolling_weighted_stability": 0.70,
        }]
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            with patch("trading_app.live_config.load_rolling_validated_strategies",
                       return_value=mock_rolling):
                portfolio, notes = build_live_portfolio(
                    db_path=live_config_db, instrument="MGC"
                )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5
        assert any("DEMOTED" in n for n in notes)

    def test_recovery_only_from_rolling_source(self, live_config_db):
        """Recovery does NOT fire when match comes from baseline (not rolling)."""
        self._seed_strategy(live_config_db, expr=0.30)
        test_spec = LiveStrategySpec(
            "CME_PRECLOSE_E2_VOL_RV12_N20", "core", "CME_PRECLOSE", "E2",
            "VOL_RV12_N20", None,
            weight_override=0.5,
            recovery_expr_threshold=0.25,
        )
        # Empty rolling → falls back to baseline (no rolling_avg_expectancy_r)
        with patch("trading_app.live_config.LIVE_PORTFOLIO", [test_spec]):
            with patch("trading_app.live_config.load_rolling_validated_strategies",
                       return_value=[]):
                portfolio, notes = build_live_portfolio(
                    db_path=live_config_db, instrument="MGC"
                )
        assert len(portfolio.strategies) == 1
        assert portfolio.strategies[0].weight == 0.5
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trading_app/test_live_config.py::TestWeightOverrideAndRecovery -v`
Expected: FAIL — no DEMOTED/RECOVERED logic exists

**Step 3: Implement weight override + recovery**

In `build_live_portfolio`, after dollar gate passes and before `strategies.append(...)`:

```python
            # --- Weight resolution (Carver forecast scaling / Chan half-Kelly) ---
            weight = 1.0
            weight_note = ""
            if spec.weight_override is not None:
                weight = spec.weight_override
                weight_note = f"DEMOTED (weight={weight})"

                # Auto-recovery: family rolling avg ExpR shows edge recovered?
                if (
                    spec.recovery_expr_threshold is not None
                    and source == "rolling"
                    and match.get("rolling_avg_expectancy_r", 0.0)
                    >= spec.recovery_expr_threshold
                ):
                    weight = 1.0
                    rolling_avg = match["rolling_avg_expectancy_r"]
                    weight_note = (
                        f"RECOVERED (rolling_avg_ExpR={rolling_avg:+.3f} "
                        f">= {spec.recovery_expr_threshold})"
                    )
```

Update `strategies.append(...)` to use `weight=weight` instead of `weight=1.0`.
Update notes line to include weight_note.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trading_app/test_live_config.py::TestWeightOverrideAndRecovery -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add trading_app/live_config.py tests/test_trading_app/test_live_config.py
git commit -m "feat: weight override + recovery using rolling_avg_expectancy_r (family avg)"
```

---

### Task 4: Apply gates to target strategies + CLI demoted section

**Files:**
- Modify: `trading_app/live_config.py` — LIVE_PORTFOLIO spec list (lines 154-166) + CLI main() output

**Design ref:** MGC FAST10 gets `active_months` ONLY — no `weight_override` or `recovery_expr_threshold`
(seasonal gate is the full restriction; recovery does not apply because seasonal is a hard skip
before variant lookup). MNQ VOL gets `weight_override=0.5` + `recovery_expr_threshold=0.25`.

CLI adds a third section for fractional-weight strategies (0 < weight < 1).

**Step 1: Update MGC CME_REOPEN G4_FAST10 spec**

Change lines 153-162 from:
```python
    # BH FDR exclusion: MNQ p_adj > 0.05; MGC survives. @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "CME_REOPEN_E2_ORB_G4_FAST10",
        "core",
        "CME_REOPEN",
        "E2",
        "ORB_G4_FAST10",
        None,
        exclude_instruments=frozenset({"MNQ"}),
    ),
```

To:
```python
    # Adversarial audit 2026-03-13: SUSPICIOUS. Seasonal Q4+Q1 clustering (68% wins Nov-Feb),
    # cost fragility at G4 minimum. No recovery — seasonal gate is a hard skip.
    # No academic support for month-based gating (purely empirical — held to higher bar).
    # BH FDR exclusion: MNQ p_adj > 0.05; MGC survives. @research-source bloomey_audit 2026-03-12
    LiveStrategySpec(
        "CME_REOPEN_E2_ORB_G4_FAST10",
        "core",
        "CME_REOPEN",
        "E2",
        "ORB_G4_FAST10",
        None,
        exclude_instruments=frozenset({"MNQ"}),
        active_months=frozenset({11, 12, 1, 2}),
    ),
```

**Step 2: Update MNQ CME_PRECLOSE VOL_RV12_N20 spec**

Change line 166:
```python
    LiveStrategySpec("CME_PRECLOSE_E2_VOL_RV12_N20", "core", "CME_PRECLOSE", "E2", "VOL_RV12_N20", None),
```

To:
```python
    # Adversarial audit 2026-03-13: LEGIT BUT DECAYING. Monotonic decay 0.54->0.30R/yr (2022-2025).
    # Half weight (Chan half-Kelly under parameter uncertainty) until rolling avg ExpR > 0.25
    # (midpoint of 2024-2025 decay range). Man AHL: recovery requires same gates, not lower bar.
    LiveStrategySpec(
        "CME_PRECLOSE_E2_VOL_RV12_N20",
        "core",
        "CME_PRECLOSE",
        "E2",
        "VOL_RV12_N20",
        None,
        weight_override=0.5,
        recovery_expr_threshold=0.25,
    ),
```

**Step 3: Add CLI demoted section**

In `main()` (line ~748-760), between the `active` section and the `gated` section, add:

```python
    demoted = [s for s in portfolio.strategies if 0 < s.weight < 1]

    if demoted:
        print(f"\nDemoted strategies (0 < weight < 1): {len(demoted)}")
        print(f"  {'Strategy':<50} {'Weight':>6}  {'ExpR':>6}  {'Exp$/trade':>10}")
        print(f"  {'-' * 50} {'------':>6}  {'------':>6}  {'----------':>10}")
        for s in demoted:
            print(
                f"  {s.strategy_id:<50} {s.weight:>6.2f}  {s.expectancy_r:>+6.3f}  "
                f"{_exp_dollars(s):>10}"
            )
```

Update `active` filter to exclude demoted: `active = [s for s in portfolio.strategies if s.weight >= 1.0]`

**Step 4: Run full test suite + drift checks**

Run: `pytest tests/test_trading_app/test_live_config.py -v && python pipeline/check_drift.py`
Expected: ALL PASS

**Step 5: Verify CLI output**

Run: `python -m trading_app.live_config --instrument MGC`
Expected: SEASONAL skip for CME_REOPEN_E2_ORB_G4_FAST10 (March is out of season)

Run: `python -m trading_app.live_config --instrument MNQ`
Expected: Demoted section shows CME_PRECLOSE_E2_VOL_RV12_N20 at weight=0.5 (or RECOVERED if rolling avg > 0.25)

**Step 6: Commit**

```bash
git add trading_app/live_config.py
git commit -m "feat: apply seasonal + decay gates to MGC FAST10 and MNQ VOL + CLI demoted section"
```

---

### Task 5: Final verification

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass (no regressions)

**Step 2: Run drift checks**

Run: `python pipeline/check_drift.py`
Expected: All pass

**Step 3: Run behavioral audit**

Run: `python scripts/tools/audit_behavioral.py`
Expected: No new violations

**Step 4: Verify live portfolio for all instruments**

Run: `python -m trading_app.live_config --instrument MGC && python -m trading_app.live_config --instrument MNQ && python -m trading_app.live_config --instrument MES && python -m trading_app.live_config --instrument M2K`
Expected: All build successfully. MGC shows SEASONAL skip for FAST10. MNQ shows weight status for VOL.

**Step 5: Commit any cleanup**

If any minor fixes needed, commit them.
