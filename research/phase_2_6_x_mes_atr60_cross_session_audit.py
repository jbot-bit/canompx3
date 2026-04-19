#!/usr/bin/env python3
"""Phase 2.6 — X_MES_ATR60 cross-session extension K=6 audit.

Pre-reg: docs/audit/hypotheses/2026-04-19-x-mes-atr60-cross-session-extension-v1.yaml
Origin: Phase 2.5 portfolio subset-t sweep Tier-1 finding.

Tests X_MES_ATR60 on 6 currently-untested MNQ long cells. Applies C1-C9 gates
per cell + BH-FDR at K=6 family.

Canonical delegations:
  - compute_mode_a, C4_T_WITH_THEORY, C7_MIN_N, C9_ERA_THRESHOLD,
    C9_MIN_N_PER_ERA from research.mode_a_revalidation_active_setups
  - filter_signal via compute_mode_a (which handles X_MES_ATR60 cross-asset
    injection per _inject_cross_asset_atrs pattern)
  - HOLDOUT_SACRED_FROM from trading_app.holdout_policy
  - GOLD_DB_PATH from pipeline.paths
  - SESSION_CATALOG whitelist from pipeline.dst

Outputs:
  - research/output/phase_2_6_x_mes_atr60_cross_session_audit.csv
  - stdout summary + BH-FDR verdict
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data_era import micro_launch_day  # noqa: E402
from pipeline.dst import SESSION_CATALOG  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from research.mode_a_revalidation_active_setups import (  # noqa: E402
    C4_T_WITH_THEORY,
    C7_MIN_N,
    C9_ERA_THRESHOLD,
    C9_MIN_N_PER_ERA,
    compute_mode_a,
)
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"

# C6 WFE threshold
# @research-source docs/institutional/pre_registered_criteria.md § Criterion 6
# @research-source docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md
# @revalidated-for 2026-04-19
C6_WFE_THRESHOLD: float = 0.50

# Pre-reg cell surface (MUST match pre-reg yaml K=6 exactly)
CELLS: list[dict[str, Any]] = [
    {"id": 1, "session": "CME_PRECLOSE", "rr": 1.5},
    {"id": 2, "session": "CME_PRECLOSE", "rr": 2.0},
    {"id": 3, "session": "US_DATA_830", "rr": 1.0},
    {"id": 4, "session": "US_DATA_830", "rr": 1.5},
    {"id": 5, "session": "US_DATA_830", "rr": 2.0},
    {"id": 6, "session": "US_DATA_1000", "rr": 2.0},
]

INSTRUMENT = "MNQ"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
FILTER_TYPE = "X_MES_ATR60"
EXECUTION_SPEC = '{"direction": "long"}'


def _spec(cell: dict[str, Any]) -> dict[str, Any]:
    sess = cell["session"]
    if sess not in SESSION_CATALOG:
        raise ValueError(f"session {sess!r} not in canonical SESSION_CATALOG")
    return {
        "strategy_id": f"{INSTRUMENT}_{sess}_{ENTRY_MODEL}_RR{cell['rr']}_CB{CONFIRM_BARS}_{FILTER_TYPE}",
        "instrument": INSTRUMENT,
        "orb_label": sess,
        "orb_minutes": ORB_MINUTES,
        "entry_model": ENTRY_MODEL,
        "confirm_bars": CONFIRM_BARS,
        "rr_target": cell["rr"],
        "filter_type": FILTER_TYPE,
        "execution_spec": EXECUTION_SPEC,
    }


def subset_t(expr: float | None, sd: float | None, n: int) -> float:
    if n < 2 or expr is None or sd is None or sd == 0:
        return float("nan")
    return float(expr) / (float(sd) / math.sqrt(n))


def t_to_two_sided_p(t: float, df: int) -> float:
    """Two-sided p from t. Uses scipy t-dist CDF."""
    if math.isnan(t) or df < 1:
        return float("nan")
    from scipy import stats
    return float(2 * (1 - stats.t.cdf(abs(t), df=df)))


def bh_fdr(p_vals: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg at q. Returns pass/fail per p-value in original order.

    Handles NaN p-values by treating as FAIL (survives nothing).
    """
    indexed = [(i, p) for i, p in enumerate(p_vals) if not math.isnan(p)]
    indexed.sort(key=lambda x: x[1])
    n = len(indexed)
    if n == 0:
        return [False] * len(p_vals)
    passes = [False] * len(p_vals)
    # Largest k such that p_(k) <= k/n * q
    k_cut = -1
    for k, (_orig, p) in enumerate(indexed, start=1):
        if p <= (k / n) * q:
            k_cut = k
    for rank, (orig, _p) in enumerate(indexed, start=1):
        if rank <= k_cut:
            passes[orig] = True
    return passes


def walk_forward(df: pd.DataFrame, fire: np.ndarray) -> tuple[float, int]:
    df = df.copy()
    df["_year"] = pd.to_datetime(df["trading_day"]).dt.year
    years = sorted(df["_year"].unique())
    if len(years) < 3:
        return float("nan"), 0
    pnl = df["pnl_r"].to_numpy()
    folds: list[tuple[float, float]] = []
    for y in years[1:]:
        train_mask = (df["_year"].values < y) & fire
        test_mask = (df["_year"].values == y) & fire
        train = pnl[train_mask]
        test = pnl[test_mask]
        if len(train) < 10 or len(test) < 5:
            continue
        train_sh = (train.mean() / train.std(ddof=1)) if train.std(ddof=1) > 0 else 0.0
        test_sh = (test.mean() / test.std(ddof=1)) if test.std(ddof=1) > 0 else 0.0
        folds.append((float(train_sh), float(test_sh)))
    if not folds:
        return float("nan"), 0
    mean_train = np.mean([f[0] for f in folds])
    mean_test = np.mean([f[1] for f in folds])
    wfe = (mean_test / mean_train) if mean_train > 0 else float("nan")
    return float(wfe), len(folds)


def audit_cell(con: duckdb.DuckDBPyConnection, cell: dict[str, Any]) -> dict[str, Any]:
    spec = _spec(cell)
    try:
        n, expr, sharpe_ann, wr, year_break, sd = compute_mode_a(con, spec)
    except Exception as e:  # noqa: BLE001
        return {
            "cell_id": cell["id"], "strategy_id": spec["strategy_id"],
            "error": f"compute_mode_a: {e}",
            "n": 0, "expr": None, "verdict": "ERROR",
        }

    t = subset_t(expr, sd, n)
    p = t_to_two_sided_p(t, df=max(n - 1, 1))

    c3 = bool(not math.isnan(p) and p < 0.05)
    c4 = bool(not math.isnan(t) and abs(t) >= C4_T_WITH_THEORY)
    c7 = bool(n >= C7_MIN_N)

    # C9 era stability
    c9_fail_years = [
        y for y, yb in year_break.items()
        if yb["n"] >= C9_MIN_N_PER_ERA and yb["expr"] is not None
        and yb["expr"] < C9_ERA_THRESHOLD
    ]
    c9 = len(c9_fail_years) == 0 if year_break else False

    # C10: first trade on-filter after MICRO launch
    micro = micro_launch_day(INSTRUMENT)
    # (derive first-trade from compute_mode_a year_break keys — cheap heuristic)
    first_year = min(year_break.keys()) if year_break else None
    c10 = bool(first_year is not None and first_year >= micro.year)

    # C6 walk-forward — needs raw df, re-query
    # Simpler: compute WFE by requerying via fetch of filtered outcomes
    sess = cell["session"]
    df_full = con.execute(
        f"""
        SELECT o.trading_day, o.pnl_r, d.*
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ? AND o.orb_label = ? AND o.orb_minutes = ?
          AND o.entry_model = ? AND o.confirm_bars = ? AND o.rr_target = ?
          AND d.orb_{sess}_break_dir = 'long'
          AND o.pnl_r IS NOT NULL AND o.trading_day < ?
        ORDER BY o.trading_day
        """,
        [INSTRUMENT, sess, ORB_MINUTES, ENTRY_MODEL, CONFIRM_BARS, cell["rr"], HOLDOUT_SACRED_FROM],
    ).df()

    if len(df_full) == 0:
        wfe, n_folds = float("nan"), 0
    else:
        # Re-derive fire mask — must mirror compute_mode_a's canonical
        # X_MES_ATR60 injection path. Cheap approach: load MES atr_20_pct
        # and inject on df, then filter_signal.
        from research.filter_utils import filter_signal
        # Inject cross-asset ATR (mirrors mode_a_revalidation:248-266)
        src_rows = con.execute(
            "SELECT trading_day, atr_20_pct FROM daily_features "
            "WHERE symbol = 'MES' AND orb_minutes = 5 AND atr_20_pct IS NOT NULL"
        ).fetchall()
        from datetime import date as _date
        src_map: dict[_date, float] = {}
        for td, pct in src_rows:
            k = td.date() if hasattr(td, "date") else td
            src_map[k] = float(pct)
        df_full["cross_atr_MES_pct"] = df_full["trading_day"].apply(
            lambda d: src_map.get(d.date() if hasattr(d, "date") else d)
        )
        fire = np.asarray(filter_signal(df_full, FILTER_TYPE, sess)).astype(bool)
        wfe, n_folds = walk_forward(df_full, fire)

    c6 = bool(not math.isnan(wfe) and wfe >= C6_WFE_THRESHOLD)

    # Verdict: PASS_CELL if all of C3 C4 C6 C7 C9 C10 PASS
    cell_pass = c3 and c4 and c6 and c7 and c9 and c10

    return {
        "cell_id": cell["id"],
        "strategy_id": spec["strategy_id"],
        "session": sess,
        "rr": cell["rr"],
        "n": n,
        "expr": expr,
        "sd": sd,
        "wr": wr,
        "sharpe_ann": sharpe_ann,
        "subset_t": t if not math.isnan(t) else None,
        "p_two_sided": p if not math.isnan(p) else None,
        "wfe": wfe if not math.isnan(wfe) else None,
        "n_wfe_folds": n_folds,
        "c3_p_lt_05": c3,
        "c4_chordia_t": c4,
        "c6_wfe": c6,
        "c7_n_deployable": c7,
        "c9_era_stability": c9,
        "c9_fail_years": c9_fail_years,
        "c10_micro_only": c10,
        "cell_pass": cell_pass,
        "verdict": "PASS_CELL" if cell_pass else "FAIL_CELL",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        rows = [audit_cell(con, c) for c in CELLS]
    finally:
        con.close()

    # BH-FDR at K=6 on p_two_sided
    p_vals = [r.get("p_two_sided") for r in rows]
    p_for_bh = [p if p is not None else float("nan") for p in p_vals]
    bh_pass = bh_fdr(p_for_bh, q=0.05)
    for r, bh in zip(rows, bh_pass):
        r["bh_fdr_k6_pass"] = bh

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "phase_2_6_x_mes_atr60_cross_session_audit.csv"
    df.to_csv(csv_path, index=False)

    print("=" * 70)
    print("PHASE 2.6 — X_MES_ATR60 CROSS-SESSION EXTENSION K=6 AUDIT")
    print(f"Mode A cutoff: trading_day < {HOLDOUT_SACRED_FROM}")
    print("=" * 70)
    print()
    for r in rows:
        if "error" in r:
            print(f"[{r['cell_id']}] {r['strategy_id']}: ERROR {r['error']}")
            continue
        marks = "".join("P" if r[k] else "F" for k in
                        ["c3_p_lt_05", "c4_chordia_t", "c6_wfe", "c7_n_deployable",
                         "c9_era_stability", "c10_micro_only"])
        bh = "BH_PASS" if r["bh_fdr_k6_pass"] else "BH_FAIL"
        wfe_val = r.get("wfe")
        wfe_str = f"{wfe_val:+.3f}" if wfe_val is not None else "—"
        expr_val = r.get("expr")
        expr_str = f"{expr_val:+.4f}" if expr_val is not None else "—"
        t_val = r.get("subset_t")
        t_str = f"{t_val:+.3f}" if t_val is not None else "—"
        p_val = r.get("p_two_sided")
        p_str = f"{p_val:.4f}" if p_val is not None else "—"
        print(f"[{r['cell_id']}] {r['strategy_id']}")
        print(f"    N={r['n']:>4} ExpR={expr_str} t={t_str} p={p_str}")
        print(f"    WFE={wfe_str} ({r['n_wfe_folds']} folds)")
        print(f"    C3-C4-C6-C7-C9-C10 = {marks}  {bh}")
        print(f"    VERDICT: {r['verdict']}")
        if r["c9_fail_years"]:
            print(f"    C9 fail-years: {r['c9_fail_years']}")
        print()

    n_pass = sum(1 for r in rows if r.get("cell_pass"))
    n_bh_pass = sum(1 for r in rows if r.get("bh_fdr_k6_pass"))
    print(f"Per-cell PASS (all C3-C9): {n_pass} / {len(rows)}")
    print(f"Family BH-FDR K=6 q=0.05 PASS: {n_bh_pass} / {len(rows)}")
    print()
    try:
        print(f"Written: {csv_path.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
