#!/usr/bin/env python3
"""Phase 2.8 — multi-year regime stratification across 38 active lanes.

Extends Phase 2.7 from single-year (ex-2024 only) to multi-year (ex-2020,
ex-2022, ex-2024). Tests whether the "2024 regime break" framing is actually
a recurring high-vol-year pattern (2020 COVID, 2022 rate-hike, 2024 = all
elevated ATR per Phase 2.7 caveat (a) verification).

Canonical delegations mirror Phase 2.7:
  - compute_mode_a, load_active_setups, direction_from_execution_spec
    from research.mode_a_revalidation_active_setups
  - filter_signal via compute_mode_a (CrossAssetATRFilter handled)
  - HOLDOUT_SACRED_FROM from trading_app.holdout_policy
  - GOLD_DB_PATH from pipeline.paths
  - SESSION_CATALOG from pipeline.dst

Outputs:
  - research/output/phase_2_8_multi_year_regime_stratification.csv
  - stdout summary + pattern classification
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    compute_mode_a,
    direction_from_execution_spec,
    load_active_setups,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
_VALID_SESSIONS = frozenset(SESSION_CATALOG.keys())

# Years to stratify.
#
# Initial scope (v1, shipped 2026-04-19): (2020, 2022, 2024) — the 3 elevated-
# ATR years identified in Phase 2.7 caveat (a) verification. This tests
# whether 2024-specific failures recur across all known high-vol years.
#
# Adversarial review surfaced a gap: 2019/2021/2023/2025 not tested. For a
# fully-comprehensive year-by-year stratification, use COMPREHENSIVE_YEARS
# (includes low-vol control years + 2025 edge-of-Mode-A). Follow-up runs
# should use the comprehensive set. The v1 finding ("0 RECURRING across 2020
# + 2022 + 2024") is honest but limited to the 3 high-vol years.
STRATIFY_YEARS: tuple[int, ...] = (2020, 2022, 2024)
COMPREHENSIVE_YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

# Flag thresholds per Phase 2.7
FLAG_DELTA_THRESHOLD: float = 0.03
FLAG_UNEVALUABLE_MIN_N: int = 30


def subset_t(expr: float | None, sd: float | None, n: int) -> float:
    if n < 2 or expr is None or sd is None or sd == 0:
        return float("nan")
    return float(expr) / (float(sd) / math.sqrt(n))


def _window_stats(
    con: duckdb.DuckDBPyConnection,
    spec: dict[str, Any],
    *,
    exclude_year: int | None = None,
    only_year: int | None = None,
) -> dict[str, Any]:
    """Compute per-window subset-filter stats for a lane.

    - exclude_year=None AND only_year=None → full Mode A
    - exclude_year=Y → Mode A minus calendar year Y
    - only_year=Y → only calendar year Y (within Mode A)
    """
    sess = spec["orb_label"]
    if sess not in _VALID_SESSIONS:
        raise ValueError(f"orb_label {sess!r} not in SESSION_CATALOG")
    direction = direction_from_execution_spec(spec.get("execution_spec"))

    where_clauses = [
        "o.symbol = ?", "o.orb_label = ?", "o.orb_minutes = ?",
        "o.entry_model = ?", "o.confirm_bars = ?", "o.rr_target = ?",
        f"d.orb_{sess}_break_dir = ?",
        "o.pnl_r IS NOT NULL",
    ]
    params: list[Any] = [
        spec["instrument"], sess, spec["orb_minutes"],
        spec["entry_model"], spec["confirm_bars"], spec["rr_target"],
        direction,
    ]

    if only_year is not None:
        where_clauses.append("o.trading_day >= ? AND o.trading_day < ?")
        params.extend([date(only_year, 1, 1), date(only_year + 1, 1, 1)])
    else:
        where_clauses.append("o.trading_day < ?")
        params.append(HOLDOUT_SACRED_FROM)
        if exclude_year is not None:
            where_clauses.append(
                "NOT (o.trading_day >= ? AND o.trading_day < ?)"
            )
            params.extend([date(exclude_year, 1, 1), date(exclude_year + 1, 1, 1)])

    sql = f"""
        SELECT o.trading_day, o.pnl_r, o.outcome, o.symbol, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE {" AND ".join(where_clauses)}
        ORDER BY o.trading_day
    """
    df = con.execute(sql, params).df()
    if len(df) == 0:
        return {"n_universe": 0, "n_on": 0, "expr": None, "sd": None, "t": None}

    from trading_app.config import ALL_FILTERS, CrossAssetATRFilter
    from research.filter_utils import filter_signal

    filter_type = spec.get("filter_type")
    if not filter_type or filter_type == "UNFILTERED":
        fire = np.ones(len(df), dtype=bool)
    else:
        filt_obj = ALL_FILTERS.get(filter_type)
        if filt_obj is not None and isinstance(filt_obj, CrossAssetATRFilter):
            source = filt_obj.source_instrument
            if source != spec["instrument"]:
                src_rows = con.execute(
                    "SELECT trading_day, atr_20_pct FROM daily_features "
                    "WHERE symbol = ? AND orb_minutes = 5 "
                    "AND atr_20_pct IS NOT NULL",
                    [source],
                ).fetchall()
                src_map = {
                    (td.date() if hasattr(td, "date") else td): float(pct)
                    for td, pct in src_rows
                }
                df[f"cross_atr_{source}_pct"] = df["trading_day"].apply(
                    lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
                )
        try:
            fire = np.asarray(filter_signal(df, filter_type, sess)).astype(bool)
        except Exception as e:  # noqa: BLE001
            return {"n_universe": len(df), "n_on": 0, "expr": None,
                    "sd": None, "t": None, "error": str(e)}

    df_on = df[fire].reset_index(drop=True)
    n_on = len(df_on)
    if n_on == 0:
        return {"n_universe": len(df), "n_on": 0, "expr": None, "sd": None, "t": None}
    pnl = df_on["pnl_r"].astype(float).to_numpy()
    expr = float(pnl.mean())
    sd = float(pnl.std(ddof=1)) if n_on > 1 else None
    t = subset_t(expr, sd, n_on)
    return {"n_universe": len(df), "n_on": n_on, "expr": expr, "sd": sd,
            "t": t if not math.isnan(t) else None}


def classify_pattern(year_deltas: dict[int, float | None], year_expr_pure: dict[int, float | None],
                     year_n: dict[int, int]) -> str:
    """Classify a lane by cross-year regime pattern.

    Recurring if ≥2 years show material drag (ex-year lift > 0.03 + the year
    itself had ExpR ≤ -0.05).
    """
    drag_years = []
    for y in STRATIFY_YEARS:
        dyr = year_deltas.get(y)
        eyr = year_expr_pure.get(y)
        nyr = year_n.get(y, 0)
        if nyr < FLAG_UNEVALUABLE_MIN_N:
            continue
        if dyr is not None and eyr is not None:
            if dyr > FLAG_DELTA_THRESHOLD and eyr <= -0.05:
                drag_years.append(y)
    if len(drag_years) >= 2:
        return f"RECURRING_VOL_DRAG ({','.join(str(y) for y in drag_years)})"
    if len(drag_years) == 1:
        return f"SINGLE_YEAR_DRAG ({drag_years[0]})"
    # No drag candidates — either neutral or regime-insensitive
    # Check UNEVALUABLE: all 3 years below N=30
    all_thin = all(year_n.get(y, 0) < FLAG_UNEVALUABLE_MIN_N for y in STRATIFY_YEARS)
    if all_thin:
        return "UNEVALUABLE"
    return "VOL_NEUTRAL"


def audit_lane(con: duckdb.DuckDBPyConnection, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        _, expr_full_canonical, *_ = compute_mode_a(con, spec)
    except Exception as e:  # noqa: BLE001
        return {
            "strategy_id": spec["strategy_id"], "error": f"compute_mode_a: {e}",
            "pattern": "ERROR",
        }

    full = _window_stats(con, spec)
    if full.get("expr") is None:
        return {
            "strategy_id": spec["strategy_id"], "error": "full window empty",
            "pattern": "ERROR",
        }
    # Sanity check: window-helper should agree with compute_mode_a
    if expr_full_canonical is not None and abs(full["expr"] - expr_full_canonical) > 1e-4:
        return {
            "strategy_id": spec["strategy_id"],
            "error": f"window mismatch: {full['expr']} vs {expr_full_canonical}",
            "pattern": "ERROR",
        }

    row: dict[str, Any] = {
        "strategy_id": spec["strategy_id"],
        "instrument": spec["instrument"],
        "session": spec["orb_label"],
        "rr_target": spec["rr_target"],
        "filter_type": spec.get("filter_type"),
        "full_n_on": full["n_on"],
        "full_expr": full["expr"],
        "full_t": full["t"],
    }

    year_deltas: dict[int, float | None] = {}
    year_expr_pure: dict[int, float | None] = {}
    year_n: dict[int, int] = {}

    for y in STRATIFY_YEARS:
        ex = _window_stats(con, spec, exclude_year=y)
        only = _window_stats(con, spec, only_year=y)
        row[f"ex{y}_n"] = ex["n_on"]
        row[f"ex{y}_expr"] = ex["expr"]
        row[f"ex{y}_t"] = ex["t"]
        row[f"y{y}_n"] = only["n_on"]
        row[f"y{y}_expr"] = only["expr"]
        if ex["expr"] is not None and full["expr"] is not None:
            delta = ex["expr"] - full["expr"]
            row[f"delta{y}"] = delta
            year_deltas[y] = delta
        year_expr_pure[y] = only["expr"]
        year_n[y] = only["n_on"]

    row["pattern"] = classify_pattern(year_deltas, year_expr_pure, year_n)
    return row


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="Use COMPREHENSIVE_YEARS (2019-2025) instead of default high-vol trio",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix to append to output CSV filename (e.g. '_comprehensive')",
    )
    args = parser.parse_args()

    # Swap STRATIFY_YEARS if comprehensive flag set
    global STRATIFY_YEARS
    if args.comprehensive:
        STRATIFY_YEARS = COMPREHENSIVE_YEARS
        print(f"Using COMPREHENSIVE_YEARS: {STRATIFY_YEARS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        setups = load_active_setups(con)
        print(f"Loaded {len(setups)} active validated_setups lanes")
        rows = []
        for i, spec in enumerate(setups, 1):
            print(f"  [{i}/{len(setups)}] {spec['strategy_id']}")
            rows.append(audit_lane(con, spec))
    finally:
        con.close()

    df = pd.DataFrame(rows)
    suffix = args.output_suffix or ("_comprehensive" if args.comprehensive else "")
    csv_path = OUTPUT_DIR / f"phase_2_8_multi_year_regime_stratification{suffix}.csv"
    df.to_csv(csv_path, index=False)

    print()
    print("=" * 70)
    print("PHASE 2.8 — MULTI-YEAR REGIME STRATIFICATION")
    print(f"Stratification years: {STRATIFY_YEARS}")
    print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM}")
    print("=" * 70)
    print()

    pattern_counts = df["pattern"].value_counts().sort_index()
    print("Pattern breakdown:")
    for pat, cnt in pattern_counts.items():
        print(f"  {pat:<45} {cnt}")
    print()

    # Recurring-vol-drag detail
    recurring = df[df["pattern"].astype(str).str.startswith("RECURRING_VOL_DRAG")].copy()
    if len(recurring):
        print(f"RECURRING_VOL_DRAG detail (n={len(recurring)}):")
        cols = ["strategy_id", "full_expr", "y2020_expr", "y2022_expr",
                "y2024_expr", "pattern"]
        print(recurring[cols].to_string(index=False))
        print()

    single = df[df["pattern"].astype(str).str.startswith("SINGLE_YEAR_DRAG")].copy()
    if len(single):
        print(f"SINGLE_YEAR_DRAG detail (n={len(single)}):")
        cols = ["strategy_id", "full_expr", "y2020_expr", "y2022_expr",
                "y2024_expr", "pattern"]
        print(single[cols].to_string(index=False))
        print()

    try:
        print(f"Written: {csv_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
