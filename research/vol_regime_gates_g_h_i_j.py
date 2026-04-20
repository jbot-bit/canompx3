"""
Stages G + H + I + J for vol-regime confluence portfolio scan v1.

Runs per survivor cell (6 cells from scan v1):
  G: T4 sensitivity ±20% on thresholds
  H: T6 null floor (block bootstrap, validated in Stage L)
  I: T7 per-year stability + 2019-exclude sensitivity
  J: OVN_P75 reframe (rolling 252d percentile) on OVN-based survivors

Design-first:
  - Each stage is a separate function with explicit kill criteria from pre-reg.
  - Canonical filter delegation via research.filter_utils.filter_signal.
  - Triple-join on (trading_day, symbol, orb_minutes), CTE orb_minutes=5 guard.
  - No ad-hoc thresholds.

Outputs:
  research/output/vol_regime_gates_results.json
  research/output/vol_regime_gates_summary.md (written by Stage M)
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

HOLDOUT = pd.Timestamp(HOLDOUT_SACRED_FROM)
OOS_END = pd.Timestamp("2026-04-16")
SEED = 20260420

# The 6 survivors from scan v1 that enter Stages G/H/I/J
# (cell_id, orb_label, orb_minutes, rr, base_filter_key, variant_type, base_is_expr)
SURVIVORS = [
    (5, "COMEX_SETTLE", 5, 1.5, "ORB_G5", "ovn_only", 0.0915),
    (6, "COMEX_SETTLE", 5, 1.5, "ORB_G5", "xmes_only", 0.0915),
    (7, "COMEX_SETTLE", 5, 1.5, "ORB_G5", "ovn_or_xmes", 0.0915),
    (1, "EUROPE_FLOW", 5, 1.5, "ORB_G5", "ovn_only", None),
    (10, "NYSE_OPEN", 5, 1.0, "COST_LT12", "xmes_only", None),
    (13, "TOKYO_OPEN", 5, 1.5, "COST_LT12", "xmes_only", None),
]

ERAS = [
    (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"), "2019-2020"),
    (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"), "2021-2022"),
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), "2023"),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"), "2024-2025"),
]


def load_lane(con, orb_label, orb_minutes, rr) -> pd.DataFrame:
    q = """
    WITH mnq_feat AS (
      SELECT trading_day, symbol, orb_minutes,
             overnight_range, orb_COMEX_SETTLE_size, orb_EUROPE_FLOW_size,
             orb_NYSE_OPEN_size, orb_US_DATA_1000_size,
             orb_SINGAPORE_OPEN_size, orb_TOKYO_OPEN_size,
             atr_20, atr_20_pct,
             session_london_high, session_london_low
      FROM daily_features WHERE symbol='MNQ'
    ),
    mes_atr AS (
      SELECT trading_day, atr_20_pct AS mes_atr_20_pct
      FROM daily_features WHERE symbol='MES' AND orb_minutes=5
    )
    SELECT o.trading_day, o.symbol, o.pnl_r,
           m.overnight_range, m.orb_COMEX_SETTLE_size, m.orb_EUROPE_FLOW_size,
           m.orb_NYSE_OPEN_size, m.orb_US_DATA_1000_size,
           m.orb_SINGAPORE_OPEN_size, m.orb_TOKYO_OPEN_size,
           m.atr_20, m.atr_20_pct,
           (m.session_london_high - m.session_london_low) AS session_london_range,
           x.mes_atr_20_pct
    FROM orb_outcomes o
    JOIN mnq_feat m ON o.trading_day=m.trading_day AND o.symbol=m.symbol AND o.orb_minutes=m.orb_minutes
    LEFT JOIN mes_atr x ON o.trading_day=x.trading_day
    WHERE o.symbol='MNQ' AND o.orb_label=? AND o.orb_minutes=?
      AND o.entry_model='E2' AND o.confirm_bars=1 AND o.rr_target=?
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = con.execute(q, [orb_label, orb_minutes, rr]).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    return df


def variant_mask(df, variant, ovn_thr=100.0, xmes_thr=60.0, lon_thr=100.0) -> np.ndarray:
    if variant == "ovn_only":
        return (df["overnight_range"].fillna(-np.inf) >= ovn_thr).to_numpy().astype(int)
    if variant == "xmes_only":
        return (df["mes_atr_20_pct"].fillna(-np.inf) >= xmes_thr).to_numpy().astype(int)
    if variant == "ovn_or_xmes":
        ovn = (df["overnight_range"].fillna(-np.inf) >= ovn_thr).to_numpy()
        xmes = (df["mes_atr_20_pct"].fillna(-np.inf) >= xmes_thr).to_numpy()
        return (ovn | xmes).astype(int)
    if variant == "london_range_only":
        return (df["session_london_range"].fillna(-np.inf) >= lon_thr).to_numpy().astype(int)
    raise ValueError(variant)


def _t_stat(pnl: np.ndarray) -> tuple[int, float, float]:
    n = len(pnl)
    if n < 2:
        return n, float(pnl.mean()) if n else 0.0, 0.0
    m = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    t = m / sd * np.sqrt(n) if sd > 0 else 0.0
    return n, m, float(t)


# ==================== STAGE G: T4 Sensitivity ====================
def stage_g_sensitivity(df, base_mask, variant):
    """3x3 grid on thresholds ±20%. Sign must not flip anywhere."""
    is_mask = (df["trading_day"] < HOLDOUT).to_numpy()

    # Build threshold grid per variant. All entries use 3 slots: (ovn, xmes, lon).
    if variant == "ovn_only":
        grid = [(t, 60.0, 100.0) for t in (80.0, 100.0, 120.0)]
    elif variant == "xmes_only":
        grid = [(100.0, t, 100.0) for t in (54.0, 60.0, 66.0)]
    elif variant == "ovn_or_xmes":
        grid = [(o, x, 100.0) for o in (80.0, 100.0, 120.0) for x in (54.0, 60.0, 66.0)]
    elif variant == "london_range_only":
        grid = [(100.0, 60.0, t) for t in (80.0, 100.0, 120.0)]
    else:
        return {"error": variant}

    results = []
    for ovn_t, xmes_t, lon_t in grid:
        if variant == "ovn_or_xmes":
            vm = ((df["overnight_range"].fillna(-np.inf) >= ovn_t).to_numpy() |
                  (df["mes_atr_20_pct"].fillna(-np.inf) >= xmes_t).to_numpy()).astype(int)
        elif variant == "ovn_only":
            vm = (df["overnight_range"].fillna(-np.inf) >= ovn_t).to_numpy().astype(int)
        elif variant == "xmes_only":
            vm = (df["mes_atr_20_pct"].fillna(-np.inf) >= xmes_t).to_numpy().astype(int)
        elif variant == "london_range_only":
            vm = (df["session_london_range"].fillna(-np.inf) >= lon_t).to_numpy().astype(int)

        confluence = (base_mask & vm).astype(bool)
        pnl = df.loc[is_mask & confluence, "pnl_r"].to_numpy()
        n, expr, t = _t_stat(pnl)
        results.append({
            "ovn_thr": ovn_t, "xmes_thr": xmes_t, "lon_thr": lon_t,
            "n": n, "expr": round(expr, 4), "t": round(t, 2),
        })

    # sign flip check — exclude N<30 cells (underpowered) per RULE 3.2
    signs = {np.sign(r["expr"]) for r in results if r["n"] >= 30}
    sign_flip = len(signs) > 1
    return {
        "grid": results,
        "sign_flip": sign_flip,
        "pass": not sign_flip,
    }


# ==================== STAGE H: T6 Null Bootstrap ====================
def moving_block_bootstrap(pnl_base, variant_in_base, n_perms, seed):
    rng = np.random.default_rng(seed)
    n = len(pnl_base)
    block_size = max(1, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block_size))
    observed = float(pnl_base[variant_in_base].mean()) if variant_in_base.sum() > 0 else 0.0
    null_exprs = np.empty(n_perms)
    for i in range(n_perms):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        resampled = np.concatenate([pnl_base[s:s + block_size] for s in starts])[:n]
        null_exprs[i] = float(resampled[variant_in_base].mean())
    p_boot = float((null_exprs >= observed).sum() + 1) / (n_perms + 1)
    return {
        "observed_variant_expr": observed,
        "null_mean": float(null_exprs.mean()),
        "null_p95": float(np.percentile(null_exprs, 95)),
        "null_p05": float(np.percentile(null_exprs, 5)),
        "p_boot": p_boot,
        "n_perms": n_perms,
        "block_size": block_size,
        "pass": p_boot < 0.05,
    }


def stage_h_null_bootstrap(df, base_mask, var_mask):
    is_mask = (df["trading_day"] < HOLDOUT).to_numpy()
    base_is = is_mask & (base_mask == 1).astype(bool)
    pnl_base = df.loc[base_is, "pnl_r"].to_numpy()
    variant_in_base = (var_mask == 1).astype(bool)[base_is]
    return moving_block_bootstrap(pnl_base, variant_in_base, n_perms=5000, seed=SEED)


# ==================== STAGE I: T7 Per-Year + 2019 Exclude ====================
def stage_i_per_year_and_2019_exclude(df, confluence_mask):
    is_mask = (df["trading_day"] < HOLDOUT).to_numpy()
    is_df = df[is_mask].copy()
    confluence_is = confluence_mask[is_mask].astype(bool)
    is_fire = is_df[confluence_is].copy()
    is_fire["year"] = is_fire["trading_day"].dt.year

    yearly = {}
    positive_years, negative_years = 0, 0
    for y, g in is_fire.groupby("year"):
        n, m, t = _t_stat(g["pnl_r"].to_numpy())
        yearly[int(y)] = {"n": n, "expr": round(m, 4), "t": round(t, 2)}
        if n >= 30:
            if m > 0:
                positive_years += 1
            elif m < 0:
                negative_years += 1

    # 2019-exclude rerun (IS starting 2020-01-01)
    exclude_mask = (is_df["trading_day"] >= pd.Timestamp("2020-01-01")) & confluence_is
    pnl_ex = is_df.loc[exclude_mask, "pnl_r"].to_numpy()
    n_ex, expr_ex, t_ex = _t_stat(pnl_ex)

    # Full IS comparison
    pnl_full = is_df.loc[confluence_is, "pnl_r"].to_numpy()
    n_full, expr_full, t_full = _t_stat(pnl_full)

    expr_delta = expr_ex - expr_full
    collapses_without_2019 = (t_ex < 2.0 and t_full >= 3.0)

    return {
        "yearly": yearly,
        "positive_years": positive_years,
        "negative_years": negative_years,
        "full_is_n": n_full,
        "full_is_expr": round(expr_full, 4),
        "full_is_t": round(t_full, 2),
        "exclude_2019_n": n_ex,
        "exclude_2019_expr": round(expr_ex, 4),
        "exclude_2019_t": round(t_ex, 2),
        "delta_excluding_2019": round(expr_delta, 4),
        "collapses_without_2019": collapses_without_2019,
        "pass_no_2019_dependence": not collapses_without_2019,
    }


# ==================== STAGE J: OVN_P75 Reframe ====================
def stage_j_ovn_p75_reframe(df, base_mask, variant, xmes_thr=60.0):
    """
    Substitute rolling-252d percentile >= 75 for OVN >= 100 absolute.
    Only applicable to ovn_only and ovn_or_xmes variants.
    Uses IS-only quantile per 2026-04-19 sub-clause (feature look-ahead).

    Rolling percentile: for each day, compute where today's overnight_range
    ranks among the PRIOR 252 days' overnight_range values (IS only).
    """
    if variant not in ("ovn_only", "ovn_or_xmes"):
        return {"skipped": f"variant '{variant}' does not use OVN"}

    is_mask = (df["trading_day"] < HOLDOUT).to_numpy()
    is_df = df[is_mask].copy().reset_index(drop=True)
    ovn = is_df["overnight_range"].to_numpy()

    # Rolling 252-day percentile (IS-only, per 2026-04-19 RULE 1 addendum)
    window = 252
    ovn_p75_mask = np.zeros(len(ovn), dtype=int)
    for i in range(window, len(ovn)):
        prior = ovn[i - window:i]
        valid = prior[~np.isnan(prior)]
        if len(valid) < 50:
            continue
        today = ovn[i]
        if np.isnan(today):
            continue
        threshold = np.percentile(valid, 75)
        ovn_p75_mask[i] = int(today >= threshold)

    # Build variant
    if variant == "ovn_only":
        new_variant = ovn_p75_mask
    else:  # ovn_or_xmes
        xmes = (is_df["mes_atr_20_pct"].fillna(-np.inf) >= xmes_thr).to_numpy().astype(int)
        new_variant = (ovn_p75_mask | xmes).astype(int)

    base_is = (base_mask[is_mask] == 1).astype(bool)
    conf = (base_is & (new_variant == 1)).astype(bool)
    pnl_p75 = is_df.loc[conf, "pnl_r"].to_numpy()
    n, expr, t = _t_stat(pnl_p75)

    fire_rate = float(ovn_p75_mask[252:].mean()) if len(ovn_p75_mask) > 252 else 0.0

    return {
        "variant": variant,
        "p75_fire_rate": round(fire_rate, 3),
        "n": n,
        "expr": round(expr, 4),
        "t": round(t, 2),
        "note": (
            "IS-only rolling 252d percentile >= 75 substituted for OVN >= 100. "
            "First 252 days skipped (warmup). Compare to scan v1 OVN=100 absolute."
        ),
    }


# ==================== MAIN ====================
def run_cell(con, cell_id, orb_label, orb_minutes, rr, base_filter, variant, _base_is):
    df = load_lane(con, orb_label, orb_minutes, rr)
    # Canonical base filter mask
    base_mask = filter_signal(df, base_filter, orb_label)
    # Variant mask (scan v1 thresholds)
    var_mask = variant_mask(df, variant)
    confluence = (base_mask & var_mask).astype(int)

    return {
        "cell_id": cell_id,
        "lane": f"MNQ_{orb_label}_E2_RR{rr}_CB1_{base_filter}",
        "variant": variant,
        "G_sensitivity": stage_g_sensitivity(df, base_mask, variant),
        "H_null_bootstrap": stage_h_null_bootstrap(df, base_mask, var_mask),
        "I_per_year_and_2019": stage_i_per_year_and_2019_exclude(df, confluence),
        "J_ovn_p75": stage_j_ovn_p75_reframe(df, base_mask, variant),
    }


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print(f"Stages G + H + I + J on {len(SURVIVORS)} survivor cells.")
    print()
    results = []
    for row in SURVIVORS:
        r = run_cell(con, *row)
        results.append(r)

        g = r["G_sensitivity"]
        h = r["H_null_bootstrap"]
        i = r["I_per_year_and_2019"]
        j = r["J_ovn_p75"]

        print(f"Cell {r['cell_id']:>2} {r['lane']} [{r['variant']}]")
        print(f"  G sensitivity: sign_flip={g['sign_flip']}  PASS={g['pass']}")
        print(f"    grid min n={min(c['n'] for c in g['grid'])}  min t={min(c['t'] for c in g['grid']):+.2f}  max t={max(c['t'] for c in g['grid']):+.2f}")
        print(f"  H null bootstrap: observed={h['observed_variant_expr']:+.4f}  null_mean={h['null_mean']:+.4f}  null_p95={h['null_p95']:+.4f}  p_boot={h['p_boot']:.4f}  PASS={h['pass']}")
        print(f"  I per-year: pos={i['positive_years']}  neg={i['negative_years']}  full_t={i['full_is_t']:+.2f}  excl-2019_t={i['exclude_2019_t']:+.2f}  delta={i['delta_excluding_2019']:+.4f}  PASS={i['pass_no_2019_dependence']}")
        if "skipped" in j:
            print(f"  J OVN_P75: {j['skipped']}")
        else:
            print(f"  J OVN_P75: n={j['n']}  expr={j['expr']:+.4f}  t={j['t']:+.2f}  fire_rate={j['p75_fire_rate']:.2f}")
        print()

    # Persist
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "vol_regime_gates_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Output: {out}")

    # Summary verdict per cell
    print()
    print("=== CELL-LEVEL VERDICT (G + H + I pass => confluence is real on this cell) ===")
    for r in results:
        g_pass = r["G_sensitivity"]["pass"]
        h_pass = r["H_null_bootstrap"]["pass"]
        i_pass = r["I_per_year_and_2019"]["pass_no_2019_dependence"]
        all_pass = g_pass and h_pass and i_pass
        print(f"  Cell {r['cell_id']:>2} {r['lane']} [{r['variant']}]: "
              f"G={'P' if g_pass else 'F'} H={'P' if h_pass else 'F'} I={'P' if i_pass else 'F'}  "
              f"{'CONFIRMED' if all_pass else 'FAILS AT LEAST ONE GATE'}")


if __name__ == "__main__":
    main()
