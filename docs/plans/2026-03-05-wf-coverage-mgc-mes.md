# Walk-Forward Coverage: MGC & MES

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Run walk-forward validation on ~250 MGC and ~96 MES untested strategies. Kill anything that fails.

**Architecture:** Re-run strategy_validator WITHOUT `--no-walkforward` flag. This re-validates all strategies including WF testing. Strategies failing WF are not promoted. Then rebuild downstream chain (edge families, RR locks, ML retrain).

**Motivation:** Bloomberg director review (Mar 5 2026) identified WF coverage as the #1 risk — 74% of MGC and 74% of MES strategies have no out-of-sample walk-forward validation.

---

### Task 1: Run WF validator for MGC

```bash
python trading_app/strategy_validator.py --instrument MGC --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
```

Note: NO `--no-walkforward` flag — WF enabled by default.
Expected: Some strategies will fail WF and be removed from validated_setups.

### Task 2: Run WF validator for MES

```bash
python trading_app/strategy_validator.py --instrument MES --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75
```

### Task 3: Downstream rebuild

```bash
python scripts/migrations/retire_e3_strategies.py
python scripts/tools/build_edge_families.py --instrument MGC
python scripts/tools/build_edge_families.py --instrument MES
python scripts/tools/select_family_rr.py
```

### Task 4: ML retrain (affected instruments only)

```bash
python -m trading_app.ml.meta_label --instrument MGC --single-config --per-aperture --skip-filter
python -m trading_app.ml.meta_label --instrument MES --single-config --per-aperture --skip-filter
```

### Task 5: Health check + drift check

```bash
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
python pipeline/health_check.py
```

### Task 6: Compare pre/post WF strategy counts

Query validated_setups to see how many strategies survived WF for each instrument/session.
