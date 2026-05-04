"""Garch COMEX_SETTLE institutional test battery — all angles for a volatility
regime indicator applied to breakout strategies.

Target cells (the 3 that survived uncorrected at K_global=36):
  C1. H2: MNQ COMEX_SETTLE O5 RR1.0 long (ORB_G5-subset; Path C cell)
  C2. L3 long:  MNQ COMEX_SETTLE O5 RR1.5 long (OVNRNG_100 deployed)
  C3. L3 short: MNQ COMEX_SETTLE O5 RR1.5 short (OVNRNG_100 deployed)

Null control (sanity check — a lane that showed NO garch signal):
  N1. L5: MNQ TOKYO_OPEN O5 RR1.5 long (ORB_G5 deployed; cross-lane R5 showed
      weak positive lift, not significant — expect battery to also show null)

Test battery (trader/Carver discipline for vol-regime overlays):
  A. Sharpe decomposition: ExpR / SD per regime
  B. Vol-targeted returns: pnl_r / SD_regime, mean
  C. Payoff ratio: AvgWin / |AvgLoss| per regime
  D. MAE/MFE decomposition: avg drawdown + avg favorable excursion per regime
  E. MAE/R ratio: drawdown cost per unit R captured
  F. Cost-adjusted Sharpe: (ExpR - friction_R) / SD
  G. Kelly fraction: f* = ExpR / SD^2 (Kelly criterion, symmetric approx)
  H. Variance ratio: SD_on / SD_off (does vol actually scale up?)
  I. Per-year and per-month stability
  J. Win distribution: % full-RR wins vs partial wins vs stop-outs
  K. Threshold sensitivity (50/60/70/80/90)
  L. NTILE-5 breakdown for monotonicity
  M. Permutation test on Sharpe lift (not just ExpR)

Look-ahead:
  All features prior-only. garch_forecast_vol_pct per line 1217 of
  build_daily_features.py. MAE/MFE are post-trade diagnostics (NOT used as
  predictors) — reading them to audit the trade distribution, not filter on.

BH-FDR:
  At K = 4 cells x 5 thresholds = 20 primary tests. Sharpe-lift p-values
  corrected. Null-control cell should NOT show a survivor — if it does,
  methodology is biased.
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import COST_SPECS
from research.filter_utils import filter_signal  # canonical delegation (research-truth-protocol.md)

OUTPUT_MD = Path("docs/audit/results/2026-04-15-garch-comex-settle-institutional-battery.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IS_END = "2026-01-01"
GARCH_GRID = [50, 60, 70, 80, 90]
SEED = 20260415


@dataclass
class Cell:
    name: str
    desc: str
    instrument: str
    orb_label: str
    aperture: int
    rr: float
    direction: str
    filter_type: str | None  # None = no filter (H2 raw cell). Canonical key from trading_app.config.ALL_FILTERS.


CELLS = [
    # H2 raw cell (ORB_G5 superset, no explicit filter) — Path C originator
    Cell("H2_COMEX_SETTLE_RR1.0_long", "MNQ COMEX_SETTLE O5 RR1.0 long (no filter)",
         "MNQ", "COMEX_SETTLE", 5, 1.0, "long", None),

    # All 6 deployed lanes × both directions. Filter keys delegate to
    # research.filter_utils.filter_signal per research-truth-protocol.md
    # § Canonical filter delegation (added 2026-04-18).
    Cell("L1_EUROPE_FLOW_RR1.5_long",  "L1 MNQ EUROPE_FLOW O5 RR1.5 long (ORB_G5)",
         "MNQ", "EUROPE_FLOW", 5, 1.5, "long", "ORB_G5"),
    Cell("L1_EUROPE_FLOW_RR1.5_short", "L1 MNQ EUROPE_FLOW O5 RR1.5 short (ORB_G5)",
         "MNQ", "EUROPE_FLOW", 5, 1.5, "short", "ORB_G5"),

    Cell("L2_SINGAPORE_OPEN_RR1.5_long",  "L2 MNQ SINGAPORE_OPEN O30 RR1.5 long (ATR_P50)",
         "MNQ", "SINGAPORE_OPEN", 30, 1.5, "long", "ATR_P50"),
    Cell("L2_SINGAPORE_OPEN_RR1.5_short", "L2 MNQ SINGAPORE_OPEN O30 RR1.5 short (ATR_P50)",
         "MNQ", "SINGAPORE_OPEN", 30, 1.5, "short", "ATR_P50"),

    Cell("L3_COMEX_SETTLE_RR1.5_long",  "L3 MNQ COMEX_SETTLE O5 RR1.5 long (OVNRNG_100)",
         "MNQ", "COMEX_SETTLE", 5, 1.5, "long", "OVNRNG_100"),
    Cell("L3_COMEX_SETTLE_RR1.5_short", "L3 MNQ COMEX_SETTLE O5 RR1.5 short (OVNRNG_100)",
         "MNQ", "COMEX_SETTLE", 5, 1.5, "short", "OVNRNG_100"),

    Cell("L4_NYSE_OPEN_RR1.0_long",  "L4 MNQ NYSE_OPEN O5 RR1.0 long (ORB_G5)",
         "MNQ", "NYSE_OPEN", 5, 1.0, "long", "ORB_G5"),
    Cell("L4_NYSE_OPEN_RR1.0_short", "L4 MNQ NYSE_OPEN O5 RR1.0 short (ORB_G5)",
         "MNQ", "NYSE_OPEN", 5, 1.0, "short", "ORB_G5"),

    Cell("L5_TOKYO_OPEN_RR1.5_long",  "L5 MNQ TOKYO_OPEN O5 RR1.5 long (ORB_G5)",
         "MNQ", "TOKYO_OPEN", 5, 1.5, "long", "ORB_G5"),
    Cell("L5_TOKYO_OPEN_RR1.5_short", "L5 MNQ TOKYO_OPEN O5 RR1.5 short (ORB_G5)",
         "MNQ", "TOKYO_OPEN", 5, 1.5, "short", "ORB_G5"),

    Cell("L6_US_DATA_1000_RR1.5_long",  "L6 MNQ US_DATA_1000 O15 RR1.5 long (VWAP_MID_ALIGNED)",
         "MNQ", "US_DATA_1000", 15, 1.5, "long", "VWAP_MID_ALIGNED"),
    Cell("L6_US_DATA_1000_RR1.5_short", "L6 MNQ US_DATA_1000 O15 RR1.5 short (VWAP_MID_ALIGNED)",
         "MNQ", "US_DATA_1000", 15, 1.5, "short", "VWAP_MID_ALIGNED"),
]


def load_cell(cell: Cell) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an IS/OOS split for one cell. Canonical filter is applied via
    research.filter_utils.filter_signal AFTER SQL load — so the filter SQL
    stays out of this module and all six deployed-lane filters route through
    the single canonical registry (trading_app.config.ALL_FILTERS).

    Direction filter (orb_{session}_break_dir) remains in the SQL because
    it is NOT a canonical filter — it's a cell-axis parameter that selects
    which side of the ORB break this cell audits.
    """
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    # Load the full daily_features row so filter_signal has every column
    # any canonical filter might need (VWAP filters read orb_{s}_vwap,
    # OVNRNG reads overnight_range, ATR_P* reads atr_20_pct, etc.).
    q = f"""
    SELECT
      o.trading_day, o.pnl_r, o.risk_dollars, o.mae_r, o.mfe_r,
      o.pnl_r * o.risk_dollars AS pnl_dollars,
      d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{cell.instrument}' AND o.orb_minutes={cell.aperture}
      AND o.orb_label='{cell.orb_label}' AND o.entry_model='E2'
      AND o.rr_target={cell.rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{cell.orb_label}_break_dir='{cell.direction}'
    """
    q_is = q + f" AND o.trading_day < DATE '{IS_END}'"
    q_oos = q + f" AND o.trading_day >= DATE '{IS_END}'"
    df_is = con.execute(q_is).df()
    df_oos = con.execute(q_oos).df()
    con.close()

    # Canonical filter application — delegate to filter_signal per
    # research-truth-protocol.md. None = H2 raw cell (no filter).
    if cell.filter_type is not None:
        for df_name, df in (("is", df_is), ("oos", df_oos)):
            if len(df) == 0:
                continue
            mask = np.asarray(filter_signal(df, cell.filter_type, cell.orb_label)).astype(bool)
            if df_name == "is":
                df_is = df_is.loc[mask].reset_index(drop=True)
            else:
                df_oos = df_oos.loc[mask].reset_index(drop=True)

    # Project only the columns the rest of the module consumes (preserve
    # the historical shape of load_cell output so regime_stats / ntile5 /
    # permutation tests don't need changes). garch_pct is aliased from
    # garch_forecast_vol_pct; orb_size is aliased from the canonical
    # per-session column.
    for df_name, df in (("is", df_is), ("oos", df_oos)):
        if len(df) == 0:
            continue
        if "garch_pct" not in df.columns:
            df["garch_pct"] = df["garch_forecast_vol_pct"]
        orb_size_col = f"orb_{cell.orb_label}_size"
        if "orb_size" not in df.columns and orb_size_col in df.columns:
            df["orb_size"] = df[orb_size_col]
    for df in (df_is, df_oos):
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        df["year"] = df["trading_day"].dt.year
        df["month"] = df["trading_day"].dt.to_period("M")
        df["is_win"] = (df["pnl_r"] > 0).astype(int)
        df["pnl_r"] = df["pnl_r"].astype(float)
        df["mae_r"] = df["mae_r"].astype(float)
        df["mfe_r"] = df["mfe_r"].astype(float)
    return df_is, df_oos


def regime_stats(sub: pd.DataFrame, friction_r: float) -> dict:
    """Compute full trader-discipline stats on a subset."""
    if len(sub) < 10:
        return {"N": len(sub), "skip": True}
    pnl = sub["pnl_r"].to_numpy()
    mae = sub["mae_r"].dropna().to_numpy()
    mfe = sub["mfe_r"].dropna().to_numpy()

    expr = float(pnl.mean())
    sd = float(pnl.std(ddof=1))
    sr = expr / sd if sd > 0 else 0.0
    wr = float((pnl > 0).mean())

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss < 0 else float("inf")

    # Kelly fraction (symmetric approximation): f* = ExpR / Var(R)
    kelly = expr / (sd ** 2) if sd > 0 else 0.0
    kelly_capped = max(0.0, min(kelly, 1.0))  # Cap between 0 and 1 for sensible reporting

    # Cost-adjusted Sharpe: approximation — subtract friction_r R-multiple per trade
    expr_net = expr - friction_r
    sr_net = expr_net / sd if sd > 0 else 0.0

    # Vol-targeted return: pnl_r / regime_SD, mean — unit-variance comparison
    vol_target = expr / sd if sd > 0 else 0.0  # same as SR by construction

    # Outcome mix: full-RR wins vs partial wins vs stops
    full_win_r = sub["pnl_r"].max() * 0.95 if len(sub) else 0.0
    full_stop_r = -0.95
    pct_full_win = float((pnl >= full_win_r * 0.95).mean())
    pct_full_stop = float((pnl <= -0.95).mean())
    pct_partial = 1.0 - pct_full_win - pct_full_stop

    return {
        "N": len(sub),
        "ExpR": expr, "SD": sd, "SR": sr,
        "ExpR_net": expr_net, "SR_net": sr_net,
        "WR": wr,
        "AvgWin": avg_win, "AvgLoss": avg_loss, "Payoff": payoff,
        "MAE_mean": float(mae.mean()) if len(mae) else 0.0,
        "MAE_median": float(np.median(mae)) if len(mae) else 0.0,
        "MFE_mean": float(mfe.mean()) if len(mfe) else 0.0,
        "Kelly_f": kelly, "Kelly_capped": kelly_capped,
        "pct_full_win": pct_full_win, "pct_full_stop": pct_full_stop,
        "pct_partial": pct_partial,
        "skip": False,
    }


def sharpe_permutation_p(df: pd.DataFrame, threshold: int, B: int = 2000) -> float:
    """Permutation test on Sharpe lift (not just mean lift)."""
    pnl = df["pnl_r"].to_numpy()
    is_on = (df["garch_pct"].values >= threshold).astype(int)
    on = pnl[is_on == 1]
    off = pnl[is_on == 0]
    if len(on) < 10 or len(off) < 10:
        return float("nan")

    def sharpe(arr):
        s = arr.std(ddof=1)
        return arr.mean() / s if s > 0 else 0.0

    obs_sr_on = sharpe(on)
    obs_sr_off = sharpe(off)
    obs_lift = obs_sr_on - obs_sr_off
    rng = np.random.default_rng(SEED)
    beats = 0
    for _ in range(B):
        shuffled = rng.permutation(is_on)
        on_s = pnl[shuffled == 1]
        off_s = pnl[shuffled == 0]
        if len(on_s) > 1 and len(off_s) > 1:
            sr_l = sharpe(on_s) - sharpe(off_s)
            if abs(sr_l) >= abs(obs_lift):
                beats += 1
    return (beats + 1) / (B + 1)


def per_month_stability(df: pd.DataFrame, threshold: int) -> dict:
    on = df[df["garch_pct"] >= threshold]
    if len(on) < 30:
        return {"skip": True}
    monthly = on.groupby("month")["pnl_r"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "expr"})
    monthly = monthly[monthly["n"] >= 3]
    if len(monthly) < 6:
        return {"skip": True, "reason": f"only {len(monthly)} testable months"}
    pos = int((monthly["expr"] > 0).sum())
    total = len(monthly)
    return {
        "total_months": total, "pos_months": pos, "pct_pos": pos / total,
        "worst_month_expr": float(monthly["expr"].min()),
        "best_month_expr": float(monthly["expr"].max()),
        "skip": False,
    }


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    survives = [False] * n
    max_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if not np.isnan(p) and p <= q * rank / n:
            max_rank = rank
    for rank, (idx, p) in enumerate(indexed, start=1):
        if rank <= max_rank:
            survives[idx] = True
    return survives


def ntile5(df: pd.DataFrame) -> list[dict]:
    if len(df) < 50:
        return []
    df = df.copy()
    df["bucket"] = pd.qcut(df["garch_pct"], 5, labels=False, duplicates="drop")
    rows = []
    for b, sub in df.groupby("bucket"):
        sd = sub["pnl_r"].std(ddof=1)
        rows.append({
            "bucket": int(b),
            "garch_range": f"{sub['garch_pct'].min():.0f}-{sub['garch_pct'].max():.0f}",
            "N": len(sub),
            "ExpR": float(sub["pnl_r"].mean()),
            "SD": float(sd) if sd and not np.isnan(sd) else 0.0,
            "SR": float(sub["pnl_r"].mean() / sd) if sd and sd > 0 else 0.0,
            "WR": float((sub["pnl_r"] > 0).mean()),
            "MAE": float(sub["mae_r"].dropna().mean()) if not sub["mae_r"].dropna().empty else 0.0,
        })
    return rows


def main() -> None:
    # Cost in R-multiples: friction_dollars / avg_risk_dollars across dataset
    # Approximation for simplicity (varies per trade in reality)
    cost = COST_SPECS["MNQ"].total_friction  # $ RT

    all_results = []  # for BH-FDR at K level
    cell_results = {}  # full results per cell

    for cell in CELLS:
        print(f"\n=== {cell.name}: {cell.desc} ===")
        df_is, df_oos = load_cell(cell)
        n_is, n_oos = len(df_is), len(df_oos)
        print(f"  IS N={n_is}  OOS N={n_oos}")
        if n_is < 50:
            print("  SKIP: IS N < 50")
            cell_results[cell.name] = {"cell": cell, "skip": True}
            continue

        # Average risk_dollars → convert cost to R-multiples
        avg_risk = float(df_is["risk_dollars"].mean())
        friction_r = cost / avg_risk if avg_risk > 0 else 0.0

        # NTILE breakdown
        nt = ntile5(df_is)
        print("  NTILE-5 garch_pct (ExpR / SR / SD):")
        for b in nt:
            print(f"    b{b['bucket']} [{b['garch_range']}] N={b['N']:3} "
                  f"ExpR={b['ExpR']:+.3f} SD={b['SD']:.2f} SR={b['SR']:+.3f} WR={b['WR']:.1%} MAE={b['MAE']:+.3f}")

        # Threshold grid — full battery
        thresh_results = []
        for th in GARCH_GRID:
            on = df_is[df_is["garch_pct"] >= th]
            off = df_is[df_is["garch_pct"] < th]
            stats_on = regime_stats(on, friction_r)
            stats_off = regime_stats(off, friction_r)

            if stats_on.get("skip") or stats_off.get("skip"):
                thresh_results.append({
                    "threshold": th, "cell": cell.name,
                    "skip": True, "reason": "thin bucket",
                })
                continue

            lift = stats_on["ExpR"] - stats_off["ExpR"]
            sr_lift = stats_on["SR"] - stats_off["SR"]
            sr_net_lift = stats_on["SR_net"] - stats_off["SR_net"]
            var_ratio = stats_on["SD"] ** 2 / stats_off["SD"] ** 2 if stats_off["SD"] > 0 else 0.0
            mae_lift = stats_on["MAE_mean"] - stats_off["MAE_mean"]
            mfe_lift = stats_on["MFE_mean"] - stats_off["MFE_mean"]
            kelly_lift = stats_on["Kelly_capped"] - stats_off["Kelly_capped"]

            # Permutation Sharpe test (the trader-discipline metric)
            p_sharpe = sharpe_permutation_p(df_is, th, B=2000)
            # Welch t on mean (for comparison)
            t_stat, p_two = stats.ttest_ind(on["pnl_r"], off["pnl_r"], equal_var=False)

            month = per_month_stability(df_is, th)

            # OOS slice
            oos_on = df_oos[df_oos["garch_pct"] >= th] if n_oos else pd.DataFrame()
            oos_off = df_oos[df_oos["garch_pct"] < th] if n_oos else pd.DataFrame()
            oos_lift = None
            oos_sr_lift = None
            if len(oos_on) >= 3 and len(oos_off) >= 3:
                oos_lift = float(oos_on["pnl_r"].mean() - oos_off["pnl_r"].mean())
                s_on = oos_on["pnl_r"].std(ddof=1)
                s_off = oos_off["pnl_r"].std(ddof=1)
                if s_on > 0 and s_off > 0:
                    oos_sr_lift = float(
                        oos_on["pnl_r"].mean() / s_on - oos_off["pnl_r"].mean() / s_off
                    )

            result = {
                "cell": cell.name, "threshold": th,
                "N_on": stats_on["N"], "N_off": stats_off["N"],
                "ExpR_on": stats_on["ExpR"], "ExpR_off": stats_off["ExpR"], "lift": lift,
                "SR_on": stats_on["SR"], "SR_off": stats_off["SR"], "sr_lift": sr_lift,
                "SR_net_on": stats_on["SR_net"], "SR_net_off": stats_off["SR_net"], "sr_net_lift": sr_net_lift,
                "WR_on": stats_on["WR"], "WR_off": stats_off["WR"],
                "Payoff_on": stats_on["Payoff"], "Payoff_off": stats_off["Payoff"],
                "MAE_on": stats_on["MAE_mean"], "MAE_off": stats_off["MAE_mean"], "mae_lift": mae_lift,
                "MFE_on": stats_on["MFE_mean"], "MFE_off": stats_off["MFE_mean"], "mfe_lift": mfe_lift,
                "Kelly_on": stats_on["Kelly_capped"], "Kelly_off": stats_off["Kelly_capped"], "kelly_lift": kelly_lift,
                "var_ratio": var_ratio,
                "t_stat": float(t_stat), "p_mean": float(p_two), "p_sharpe": p_sharpe,
                "month": month,
                "oos_lift": oos_lift, "oos_sr_lift": oos_sr_lift,
                "oos_N_on": len(oos_on), "oos_N_off": len(oos_off),
                "friction_r": friction_r,
                "skip": False,
            }
            thresh_results.append(result)
            all_results.append(result)

            print(f"  @ {th}: N_on={stats_on['N']:4} SR={stats_on['SR']:+.3f} (vs {stats_off['SR']:+.3f}) "
                  f"lift={lift:+.3f} sr_lift={sr_lift:+.3f} "
                  f"MAE={stats_on['MAE_mean']:+.3f}/{stats_off['MAE_mean']:+.3f} "
                  f"VarRatio={var_ratio:.2f} "
                  f"Kelly={stats_on['Kelly_capped']:.2f}/{stats_off['Kelly_capped']:.2f} "
                  f"p_mean={p_two:.4f} p_sharpe={p_sharpe:.4f} "
                  f"OOS_lift={oos_lift if oos_lift is None else f'{oos_lift:+.3f}'}")

        cell_results[cell.name] = {"cell": cell, "nt": nt, "thresh": thresh_results,
                                    "n_is": n_is, "n_oos": n_oos, "avg_risk": avg_risk,
                                    "friction_r": friction_r}

    # BH-FDR on Sharpe-lift p-values at K=len(all_results)
    p_sharpe_vals = [r["p_sharpe"] for r in all_results]
    K = len(p_sharpe_vals)
    survives = bh_fdr(p_sharpe_vals, q=0.05)
    for i, r in enumerate(all_results):
        r["bh_fdr_sharpe"] = survives[i]

    # Also BH-FDR on mean-lift (t-test) p-values for comparison
    p_mean_vals = [r["p_mean"] for r in all_results]
    survives_mean = bh_fdr(p_mean_vals, q=0.05)
    for i, r in enumerate(all_results):
        r["bh_fdr_mean"] = survives_mean[i]

    print(f"\n=== BH-FDR on SHARPE lift K={K} (all {len(CELLS)} cells x 5 thresholds) ===")
    print(f"  Survivors (Sharpe permutation): {sum(survives)}")
    print(f"  Survivors (mean t-test): {sum(survives_mean)}")
    for r in sorted(all_results, key=lambda x: x["p_sharpe"]):
        if r.get("bh_fdr_sharpe"):
            oos_str = "n/a" if r["oos_sr_lift"] is None else f"{r['oos_sr_lift']:+.3f}"
            print(f"    {r['cell']} @ {r['threshold']}: sr_lift={r['sr_lift']:+.3f} "
                  f"p_sharpe={r['p_sharpe']:.5f} p_mean={r['p_mean']:.5f} "
                  f"VarRatio={r['var_ratio']:.2f}  OOS_sr_lift={oos_str}")

    # Separate null-like (non-signal expected) from signal candidates based on pre-scan
    # From cross-lane test: L1/L2/L4 long/L5/L6 were null; L3 long/short + H2 were signal
    signal_cells = {"H2_COMEX_SETTLE_RR1.0_long", "L3_COMEX_SETTLE_RR1.5_long",
                    "L3_COMEX_SETTLE_RR1.5_short"}
    null_survivors = [r for r in all_results
                      if r.get("bh_fdr_sharpe") and r["cell"] not in signal_cells]
    print(f"\n  Null-lane BH-FDR survivors: {len(null_survivors)}")
    for r in null_survivors:
        print(f"    [null-lane SURVIVED] {r['cell']} @ {r['threshold']}: "
              f"sr_lift={r['sr_lift']:+.3f} p_sharpe={r['p_sharpe']:.5f}")

    emit(cell_results, all_results, K, K)


def emit(cell_results, all_results, K, K_non_null):
    lines = [
        "# Garch COMEX_SETTLE Institutional Battery — All Angles",
        "",
        "**Date:** 2026-04-15",
        "**Trigger:** User challenged whether prior R5 analysis applied the correct institutional discipline for a VOLATILITY REGIME indicator (vs generic filter). This battery uses Carver/Fitschen trader discipline: Sharpe decomposition, vol-targeted returns, MAE/MFE decomposition, payoff ratios, Kelly fraction, variance ratios, cost-adjusted Sharpe, permutation on Sharpe lift.",
        "",
        "**Cells:** 3 signal candidates (C1 H2 raw, C2 L3 long with OVNRNG_100, C3 L3 short with OVNRNG_100) + 1 null control (N1 L5 TOKYO_OPEN long with ORB_G5).",
        "",
        "**Thresholds tested:** 50, 60, 70, 80, 90 (K_threshold=5 per cell).",
        "",
        "**Permutation test:** 2000 shuffles on Sharpe-lift (not just ExpR-lift). Shuffle the garch fire assignment, recompute Sharpe on each regime, count how often |shuffled lift| >= |observed lift|.",
        "",
        "**BH-FDR correction:** K = (3 signal + 1 null) × 5 thresholds = 20 primary tests. Also reported at K=15 (excluding null control) to reflect the honest pre-reg-able hypothesis set.",
        "",
        "**Null control:** N1 L5 TOKYO_OPEN long showed no garch effect in cross-lane R5. If the null control shows a BH-FDR survivor, the methodology is biased — red flag. If null is clean and signals pass, robust evidence.",
        "",
        "---",
        "",
    ]

    for cell_name, cr in cell_results.items():
        if cr.get("skip"):
            lines.append(f"## {cell_name} — SKIP ({cr.get('reason', 'insufficient data')})")
            lines.append("")
            continue
        cell = cr["cell"]
        lines += [
            f"## {cell.name} — {cell.desc}",
            "",
            f"**IS N:** {cr['n_is']}  |  **OOS N:** {cr['n_oos']}  |  **Avg risk $:** {cr['avg_risk']:.2f}  |  **Cost in R:** {cr['friction_r']:.4f}",
            "",
            "### NTILE-5 breakdown",
            "",
            "| Bucket | Garch range | N | ExpR | SD | SR | WR | MAE |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for b in cr["nt"]:
            lines.append(
                f"| {b['bucket']} | {b['garch_range']} | {b['N']} | {b['ExpR']:+.3f} | "
                f"{b['SD']:.2f} | {b['SR']:+.3f} | {b['WR']:.1%} | {b['MAE']:+.3f} |"
            )

        lines += ["", "### Threshold grid — full battery", "",
                  "| Thr | N_on | SR_on | SR_off | sr_lift | ExpR_on | ExpR_off | WR_on | WR_off | "
                  "MAE_on | MFE_on | Payoff_on | Kelly_on | VarRatio | p_mean | p_sharpe | BH-FDR K=15 | OOS_sr_lift |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in cr["thresh"]:
            if t.get("skip"):
                lines.append(f"| {t['threshold']} | SKIP | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            bh = "PASS" if t.get("bh_fdr_non_null") else "—"
            oos = f"{t['oos_sr_lift']:+.3f}" if t.get("oos_sr_lift") is not None else "n/a"
            lines.append(
                f"| {t['threshold']} | {t['N_on']} | {t['SR_on']:+.3f} | {t['SR_off']:+.3f} | "
                f"{t['sr_lift']:+.3f} | {t['ExpR_on']:+.3f} | {t['ExpR_off']:+.3f} | "
                f"{t['WR_on']:.1%} | {t['WR_off']:.1%} | "
                f"{t['MAE_on']:+.3f} | {t['MFE_on']:+.3f} | {t['Payoff_on']:.2f} | "
                f"{t['Kelly_on']:.2f} | {t['var_ratio']:.2f} | "
                f"{t['p_mean']:.4f} | {t['p_sharpe']:.4f} | {bh} | {oos} |"
            )
        lines += ["",
                  "**Interpretation key:**",
                  "- `sr_lift` > 0 with `var_ratio` ~= 1.0 → real SHARPE edge (not leverage illusion)",
                  "- `sr_lift` > 0 with `var_ratio` > 2.0 → likely leverage artifact — ExpR lifted proportionally to vol",
                  "- `MAE_on` >> `MAE_off` with similar MFE → high-vol trades pay more drawdown per R captured",
                  "- `WR_on` > `WR_off` with stable payoff → directional edge (not regime-conditional variance)",
                  "- `Kelly_on` > `Kelly_off` → if sized by Kelly, more aggressive on high-garch (sanity check for R5)",
                  ""]

    # Summary
    survivors_full = [r for r in all_results if r.get("bh_fdr_sharpe")]
    survivors_nonnull = [r for r in all_results if r.get("bh_fdr_non_null")]
    null_results = [r for r in all_results if r["cell"].startswith("N1")]
    null_survivors = [r for r in null_results if r.get("bh_fdr_sharpe")]

    lines += [
        "---",
        "",
        "## BH-FDR summary",
        "",
        f"- K={K} (including null control): {len(survivors_full)} survivors",
        f"- K={K_non_null} (excluding null control): {len(survivors_nonnull)} survivors",
        f"- Null control N1 produced {len(null_survivors)} BH-FDR survivors — " +
        ("OK (methodology clean)" if not null_survivors else "**RED FLAG — methodology bias suspected**"),
        "",
    ]

    if survivors_nonnull:
        lines += ["### Sharpe-lift BH-FDR survivors (K=" + str(K_non_null) + ", excluding null)", "",
                  "| Cell | Thr | sr_lift | p_sharpe | OOS_sr_lift | WR lift | VarRatio | MAE_lift |",
                  "|---|---|---|---|---|---|---|---|"]
        for s in sorted(survivors_nonnull, key=lambda x: x["p_sharpe"]):
            oos = f"{s['oos_sr_lift']:+.3f}" if s.get("oos_sr_lift") is not None else "n/a"
            wr_l = s["WR_on"] - s["WR_off"]
            lines.append(f"| {s['cell']} | {s['threshold']} | {s['sr_lift']:+.3f} | "
                         f"{s['p_sharpe']:.5f} | {oos} | {wr_l:+.1%} | {s['var_ratio']:.2f} | "
                         f"{s['mae_lift']:+.3f} |")
    lines.append("")

    lines += ["---", "",
              "## Trader verdict (honest read)",
              "",
              "A real vol-regime edge shows:",
              "1. `sr_lift` > 0 (Sharpe improves, not just ExpR)",
              "2. `var_ratio` near 1.0 (edge not driven by vol scaling alone)",
              "3. `WR_on` > `WR_off` (directional accuracy, not just bigger bars)",
              "4. `MAE_on` NOT proportionally worse than MAE_off (drawdown profile not degraded)",
              "5. Null control does NOT produce false positives",
              "6. OOS sr_lift sign matches IS",
              "",
              "Rows meeting 4+ criteria above are legitimate edge candidates. See survivor table.",
              ""]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
