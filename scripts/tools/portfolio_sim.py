"""Portfolio-level simulation: 4 configs, real trade data from orb_outcomes.

READ-ONLY analysis. Does not modify any database or config.
Applies filters against daily_features to determine eligible trade days,
then computes portfolio-level ExpR, Sharpe, max DD, utilization.
"""

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", "{:.4f}".format)


def load_data():
    """Load all MNQ E2 CB1 trades joined with daily_features + MES ATR."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    trades = con.execute("""
        SELECT
            o.trading_day, o.orb_label, o.orb_minutes, o.rr_target, o.confirm_bars,
            o.entry_model, o.pnl_r, o.outcome, o.risk_dollars,
            d.atr_20_pct,
            d.orb_COMEX_SETTLE_size AS comex_size,
            d.orb_NYSE_OPEN_size AS nyse_open_size,
            d.orb_NYSE_CLOSE_size AS nyse_close_size,
            d.orb_SINGAPORE_OPEN_size AS sing_size,
            d.orb_CME_PRECLOSE_size AS cme_pre_size,
            d.orb_US_DATA_1000_size AS usdata_size,
            d.rel_vol_COMEX_SETTLE AS rv_comex,
            d.rel_vol_NYSE_OPEN AS rv_nyse_open,
            d.rel_vol_NYSE_CLOSE AS rv_nyse_close,
            d.rel_vol_SINGAPORE_OPEN AS rv_sing,
            d.rel_vol_CME_PRECLOSE AS rv_cme_pre,
            d.rel_vol_US_DATA_1000 AS rv_usdata
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = 'MNQ'
        AND o.entry_model = 'E2'
        AND o.confirm_bars = 1
    """).fetchdf()

    mes_atr = con.execute("""
        SELECT DISTINCT trading_day, atr_20_pct AS mes_atr_pct
        FROM daily_features
        WHERE symbol = 'MES' AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
    """).fetchdf()

    trades = trades.merge(mes_atr, on="trading_day", how="left")
    con.close()
    return trades


def add_filter_cols(trades):
    """Add session-specific rel_vol and orb_size columns."""
    rv_map = {
        "COMEX_SETTLE": "rv_comex",
        "NYSE_OPEN": "rv_nyse_open",
        "NYSE_CLOSE": "rv_nyse_close",
        "SINGAPORE_OPEN": "rv_sing",
        "CME_PRECLOSE": "rv_cme_pre",
        "US_DATA_1000": "rv_usdata",
    }
    sz_map = {
        "COMEX_SETTLE": "comex_size",
        "NYSE_OPEN": "nyse_open_size",
        "NYSE_CLOSE": "nyse_close_size",
        "SINGAPORE_OPEN": "sing_size",
        "CME_PRECLOSE": "cme_pre_size",
        "US_DATA_1000": "usdata_size",
    }
    trades["rel_vol"] = trades.apply(lambda r: r[rv_map[r["orb_label"]]] if r["orb_label"] in rv_map else None, axis=1)
    trades["orb_size"] = trades.apply(lambda r: r[sz_map[r["orb_label"]]] if r["orb_label"] in sz_map else None, axis=1)
    return trades


# --- Filter functions ---
def filt_orb_g8(df):
    return df["orb_size"].notna() & (df["orb_size"] >= 8.0)


def filt_atr70_vol(df):
    return df["atr_20_pct"].notna() & (df["atr_20_pct"] >= 70) & df["rel_vol"].notna() & (df["rel_vol"] >= 1.2)


def filt_vol_rv12(df):
    return df["rel_vol"].notna() & (df["rel_vol"] >= 1.2)


def filt_x_mes_atr60(df):
    return df["mes_atr_pct"].notna() & (df["mes_atr_pct"] >= 60)


# --- 4 portfolio configurations ---
# Each lane: (session, orb_minutes, rr_target, filter_fn, stop_mult, filter_label)
CONFIGS = {
    "A_CURRENT": {
        "NYSE_CLOSE": ("NYSE_CLOSE", 15, 1.0, filt_vol_rv12, 1.0, "VOL_RV12_N20"),
        "SINGAPORE_OPEN": ("SINGAPORE_OPEN", 15, 4.0, filt_orb_g8, 1.0, "ORB_G8"),
        "COMEX_SETTLE": ("COMEX_SETTLE", 5, 1.0, filt_orb_g8, 1.0, "ORB_G8"),
        "NYSE_OPEN": ("NYSE_OPEN", 15, 1.0, filt_x_mes_atr60, 1.0, "X_MES_ATR60"),
        "US_DATA_1000": ("US_DATA_1000", 5, 1.0, filt_x_mes_atr60, 0.75, "X_MES_ATR60"),
    },
    "B_ALL_ATR70": {
        "NYSE_CLOSE": ("NYSE_CLOSE", 5, 1.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "SINGAPORE_OPEN": ("SINGAPORE_OPEN", 15, 4.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "COMEX_SETTLE": ("COMEX_SETTLE", 5, 2.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "NYSE_OPEN": ("NYSE_OPEN", 5, 1.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "US_DATA_1000": ("US_DATA_1000", 5, 1.0, filt_x_mes_atr60, 0.75, "X_MES_ATR60"),
    },
    "C_MIXED": {
        "NYSE_CLOSE": ("NYSE_CLOSE", 15, 1.0, filt_vol_rv12, 1.0, "VOL_RV12_N20"),
        "SINGAPORE_OPEN": ("SINGAPORE_OPEN", 15, 4.0, filt_orb_g8, 1.0, "ORB_G8"),
        "COMEX_SETTLE": ("COMEX_SETTLE", 5, 2.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "NYSE_OPEN": ("NYSE_OPEN", 5, 1.0, filt_atr70_vol, 1.0, "ATR70_VOL"),
        "US_DATA_1000": ("US_DATA_1000", 5, 1.0, filt_x_mes_atr60, 0.75, "X_MES_ATR60"),
    },
    "D_CURRENT+CME_PRE": {
        "NYSE_CLOSE": ("NYSE_CLOSE", 15, 1.0, filt_vol_rv12, 1.0, "VOL_RV12_N20"),
        "SINGAPORE_OPEN": ("SINGAPORE_OPEN", 15, 4.0, filt_orb_g8, 1.0, "ORB_G8"),
        "COMEX_SETTLE": ("COMEX_SETTLE", 5, 1.0, filt_orb_g8, 1.0, "ORB_G8"),
        "NYSE_OPEN": ("NYSE_OPEN", 15, 1.0, filt_x_mes_atr60, 1.0, "X_MES_ATR60"),
        "US_DATA_1000": ("US_DATA_1000", 5, 1.0, filt_x_mes_atr60, 0.75, "X_MES_ATR60"),
        "CME_PRECLOSE": ("CME_PRECLOSE", 15, 1.0, filt_atr70_vol, 0.75, "ATR70_VOL"),
    },
}


def simulate(config, trades_df, label):
    """Simulate a portfolio config. Returns dict of metrics."""
    all_lane_trades = {}

    for lane_name, (session, orb_min, rr, filt_fn, _stop_mult, _filt_label) in config.items():
        mask = (
            (trades_df["orb_label"] == session)
            & (trades_df["orb_minutes"] == orb_min)
            & (trades_df["rr_target"] == rr)
            & filt_fn(trades_df)
        )
        lt = trades_df[mask][["trading_day", "orb_label", "pnl_r", "atr_20_pct"]].copy()
        lt["lane"] = lane_name
        all_lane_trades[lane_name] = lt

    portfolio = pd.concat(all_lane_trades.values(), ignore_index=True)
    if len(portfolio) == 0:
        return None

    portfolio = portfolio.sort_values("trading_day")

    # Core metrics
    total_trades = len(portfolio)
    day_range = (pd.to_datetime(portfolio.trading_day.max()) - pd.to_datetime(portfolio.trading_day.min())).days
    years = day_range / 365.25
    trades_per_year = total_trades / years if years > 0 else 0
    portfolio_expr = portfolio["pnl_r"].mean()
    total_r = portfolio["pnl_r"].sum()

    # Daily PnL for Sharpe
    daily_pnl = portfolio.groupby("trading_day")["pnl_r"].sum()
    sharpe_daily = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0
    sharpe_ann = sharpe_daily * np.sqrt(252)

    # Max drawdown in R (trade-by-trade)
    cum_r = portfolio["pnl_r"].cumsum()
    max_dd = (cum_r - cum_r.cummax()).min()

    # Worst month / quarter
    portfolio["month"] = pd.to_datetime(portfolio["trading_day"]).dt.to_period("M")
    portfolio["quarter"] = pd.to_datetime(portfolio["trading_day"]).dt.to_period("Q")
    worst_month = portfolio.groupby("month")["pnl_r"].sum().min()
    worst_quarter = portfolio.groupby("quarter")["pnl_r"].sum().min()

    # Utilization
    all_days = trades_df["trading_day"].nunique()
    days_active = portfolio["trading_day"].nunique()
    utilization = days_active / all_days * 100

    # Lane stats
    lane_stats = {}
    for ln, lt in all_lane_trades.items():
        if len(lt) > 0:
            lane_stats[ln] = {
                "n": len(lt),
                "expr": lt["pnl_r"].mean(),
                "total_r": lt["pnl_r"].sum(),
                "wr": (lt["pnl_r"] > 0).mean(),
            }

    # Lane day sets (for overlap)
    lane_days = {ln: set(lt["trading_day"].values) for ln, lt in all_lane_trades.items() if len(lt) > 0}

    # ATR regime split
    high_atr = portfolio[portfolio["atr_20_pct"] >= 70]
    low_atr = portfolio[(portfolio["atr_20_pct"] < 70) & portfolio["atr_20_pct"].notna()]

    return {
        "label": label,
        "total_trades": total_trades,
        "trades_per_year": trades_per_year,
        "portfolio_expr": portfolio_expr,
        "total_r": total_r,
        "sharpe_ann": sharpe_ann,
        "max_dd_r": max_dd,
        "worst_month_r": worst_month,
        "worst_quarter_r": worst_quarter,
        "utilization_pct": utilization,
        "lane_stats": lane_stats,
        "lane_days": lane_days,
        "high_atr_n": len(high_atr),
        "high_atr_expr": high_atr["pnl_r"].mean() if len(high_atr) > 0 else 0,
        "high_atr_total": high_atr["pnl_r"].sum() if len(high_atr) > 0 else 0,
        "low_atr_n": len(low_atr),
        "low_atr_expr": low_atr["pnl_r"].mean() if len(low_atr) > 0 else 0,
        "low_atr_total": low_atr["pnl_r"].sum() if len(low_atr) > 0 else 0,
    }


def print_overlap_matrix(r, title):
    """Print pairwise day-overlap matrix."""
    lanes = list(r["lane_days"].keys())
    print(f"\n--- {title} ---")
    header = f"{'':20s}"
    for lane in lanes:
        header += f"{lane[:12]:>14s}"
    print(header)

    for l1 in lanes:
        row = f"{l1[:20]:20s}"
        for l2 in lanes:
            if l1 == l2:
                row += f"{'N=' + str(len(r['lane_days'][l1])):>14s}"
            else:
                overlap = len(r["lane_days"][l1] & r["lane_days"][l2])
                smaller = min(len(r["lane_days"][l1]), len(r["lane_days"][l2]))
                pct = overlap / smaller * 100 if smaller > 0 else 0
                row += f"{pct:13.1f}%"
        print(row)


def main():
    print("Loading trade data...")
    trades = load_data()
    trades = add_filter_cols(trades)
    print(f"Loaded {len(trades):,} trades ({trades.trading_day.min()} to {trades.trading_day.max()})")
    print(f"MES ATR coverage: {trades.mes_atr_pct.notna().sum():,}/{len(trades):,}")

    # Run all configs
    results = {}
    for name, config in CONFIGS.items():
        results[name] = simulate(config, trades, name)

    # ====================================================================
    # TABLE 1: CONFIG COMPARISON
    # ====================================================================
    print("\n" + "=" * 110)
    print("TABLE 1: PORTFOLIO CONFIGURATION COMPARISON (real trade data)")
    print("=" * 110)

    header = f"{'Metric':30s}"
    for name in results:
        header += f"  {name:>18s}"
    print(header)
    print("-" * 110)

    rows = [
        ("Total Trades", "total_trades", lambda v: f"{v:,.0f}"),
        ("Trades/Year", "trades_per_year", lambda v: f"{v:,.1f}"),
        ("Portfolio ExpR", "portfolio_expr", lambda v: f"{v:+.4f}"),
        ("Total R (cumulative)", "total_r", lambda v: f"{v:+.1f}"),
        ("Sharpe (annualized)", "sharpe_ann", lambda v: f"{v:.4f}"),
        ("Max Drawdown (R)", "max_dd_r", lambda v: f"{v:.2f}"),
        ("Worst Month (R)", "worst_month_r", lambda v: f"{v:.2f}"),
        ("Worst Quarter (R)", "worst_quarter_r", lambda v: f"{v:.2f}"),
        ("Utilization %", "utilization_pct", lambda v: f"{v:.1f}%"),
    ]

    for label, key, fmt in rows:
        line = f"{label:30s}"
        for r in results.values():
            line += f"  {fmt(r[key]):>18s}"
        print(line)

    # ====================================================================
    # TABLE 2: PER-LANE BREAKDOWN
    # ====================================================================
    print("\n" + "=" * 110)
    print("TABLE 2: PER-LANE BREAKDOWN")
    print("=" * 110)

    for cname, r in results.items():
        print(f"\n--- {cname} ---")
        for lane, s in sorted(r["lane_stats"].items()):
            print(f"  {lane:20s}: N={s['n']:5d}  WR={s['wr']:.1%}  ExpR={s['expr']:+.4f}  TotalR={s['total_r']:+.1f}")

    # ====================================================================
    # TABLE 3: DAY OVERLAP MATRICES
    # ====================================================================
    print("\n" + "=" * 110)
    print("TABLE 3: TRADE-DAY OVERLAP (% of smaller set)")
    print("=" * 110)

    print_overlap_matrix(results["A_CURRENT"], "Config A (CURRENT)")
    print_overlap_matrix(results["B_ALL_ATR70"], "Config B (ALL ATR70)")

    # ====================================================================
    # TABLE 4: STRESS TEST — ATR REGIME SPLIT
    # ====================================================================
    print("\n" + "=" * 110)
    print("TABLE 4: STRESS TEST -- PERFORMANCE BY ATR REGIME")
    print("=" * 110)

    header = f"{'Regime':25s}"
    for name in results:
        header += f"  {name:>18s}"
    print(header)
    print("-" * 110)

    for regime, nk, ek, tk in [
        ("ATR >= 70 (N)", "high_atr_n", "high_atr_expr", "high_atr_total"),
        ("ATR >= 70 (ExpR)", "high_atr_n", "high_atr_expr", "high_atr_total"),
        ("ATR >= 70 (TotalR)", "high_atr_n", "high_atr_expr", "high_atr_total"),
        ("ATR < 70 (N)", "low_atr_n", "low_atr_expr", "low_atr_total"),
        ("ATR < 70 (ExpR)", "low_atr_n", "low_atr_expr", "low_atr_total"),
        ("ATR < 70 (TotalR)", "low_atr_n", "low_atr_expr", "low_atr_total"),
    ]:
        line = f"{regime:25s}"
        for r in results.values():
            if "N" in regime:
                line += f"  {r[nk]:>18,}"
            elif "ExpR" in regime:
                line += f"  {r[ek]:>+18.4f}"
            else:
                line += f"  {r[tk]:>+18.1f}"
        print(line)

    # Cleaner stress test summary
    a = results["A_CURRENT"]
    print(
        f"\n  Current portfolio (A) on LOW ATR days: N={a['low_atr_n']:,}  ExpR={a['low_atr_expr']:+.4f}  TotalR={a['low_atr_total']:+.1f}"
    )
    if a["low_atr_expr"] > 0:
        print("  --> POSITIVE. Current filters earn real money on non-ATR70 days.")
    else:
        print("  --> NEGATIVE/ZERO. Current filters LOSE money on non-ATR70 days.")

    b = results["B_ALL_ATR70"]
    print(f"\n  Config B on LOW ATR days: N={b['low_atr_n']:,}  ExpR={b['low_atr_expr']:+.4f}")
    print(f"  (These {b['low_atr_n']} trades are US_DATA_1000 via X_MES_ATR60, not ATR70)")

    # ====================================================================
    # TABLE 5: COMPOSITE FILTERS CHECK
    # ====================================================================
    print("\n" + "=" * 110)
    print("TABLE 5: COMPOSITE FILTERS IN VALIDATED_SETUPS")
    print("=" * 110)

    con2 = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    composites = con2.execute("""
        SELECT filter_type, COUNT(*) as n, AVG(expectancy_r) as avg_expr
        FROM validated_setups
        WHERE instrument = 'MNQ'
        AND (filter_type LIKE '%ATR%' AND filter_type LIKE '%G%')
        GROUP BY filter_type
    """).fetchdf()
    if len(composites) > 0:
        print(composites.to_string(index=False))
    else:
        print("  No ATR+G composite filters exist. GAP in discovery grid.")

    # Check compound filters
    compound = con2.execute("""
        SELECT filter_type, COUNT(*) as n, AVG(expectancy_r) as avg_expr
        FROM validated_setups
        WHERE instrument = 'MNQ'
        AND (filter_type LIKE '%CONT%' OR filter_type LIKE '%FAST%' OR filter_type LIKE '%NOMON%')
        GROUP BY filter_type
        ORDER BY avg_expr DESC
    """).fetchdf()
    if len(compound) > 0:
        print(f"\n  Compound G-filters found ({len(compound)} types):")
        print(compound.to_string(index=False))
    con2.close()

    # ====================================================================
    # DECISION MATRIX
    # ====================================================================
    print("\n" + "=" * 110)
    print("DECISION MATRIX")
    print("=" * 110)

    a = results["A_CURRENT"]
    b = results["B_ALL_ATR70"]
    c = results["C_MIXED"]
    d = results["D_CURRENT+CME_PRE"]

    print(f"\n  {'':40s} {'A_CURRENT':>12s} {'B_ALL_ATR70':>12s} {'C_MIXED':>12s} {'D_+CME_PRE':>12s}")
    print(
        f"  {'Sharpe (ann)':40s} {a['sharpe_ann']:12.4f} {b['sharpe_ann']:12.4f} {c['sharpe_ann']:12.4f} {d['sharpe_ann']:12.4f}"
    )
    print(
        f"  {'Max DD (R)':40s} {a['max_dd_r']:12.2f} {b['max_dd_r']:12.2f} {c['max_dd_r']:12.2f} {d['max_dd_r']:12.2f}"
    )
    print(
        f"  {'Portfolio ExpR':40s} {a['portfolio_expr']:12.4f} {b['portfolio_expr']:12.4f} {c['portfolio_expr']:12.4f} {d['portfolio_expr']:12.4f}"
    )
    print(f"  {'Total R':40s} {a['total_r']:12.1f} {b['total_r']:12.1f} {c['total_r']:12.1f} {d['total_r']:12.1f}")
    print(
        f"  {'Utilization':40s} {a['utilization_pct']:11.1f}% {b['utilization_pct']:11.1f}% {c['utilization_pct']:11.1f}% {d['utilization_pct']:11.1f}%"
    )

    # Decision rules
    print("\n  --- Decision Rules ---")

    # B vs A
    b_sharpe_better = b["sharpe_ann"] > a["sharpe_ann"]
    b_dd_ratio = b["max_dd_r"] / a["max_dd_r"] if a["max_dd_r"] != 0 else 1
    print(
        f"\n  B Sharpe vs A: {b['sharpe_ann']:.4f} vs {a['sharpe_ann']:.4f} -> {'BETTER' if b_sharpe_better else 'WORSE'}"
    )
    print(f"  B MaxDD / A MaxDD: {b_dd_ratio:.2f}x")

    if b_sharpe_better and b_dd_ratio <= 1.5:
        print("  RULE 1: B Sharpe better AND DD < 1.5x A -> PROMOTE B")
    elif b_sharpe_better:
        print(f"  RULE 1: B Sharpe better BUT DD = {b_dd_ratio:.2f}x A")
    else:
        print("  RULE 1: B Sharpe NOT better -> DO NOT PROMOTE B")

    if b_dd_ratio > 2.0:
        print("  RULE 2: B MaxDD > 2x A -> CONCENTRATION RISK FATAL")
    else:
        print(f"  RULE 2: B MaxDD {b_dd_ratio:.2f}x A -> concentration acceptable")

    # C vs A
    c_sharpe_better = c["sharpe_ann"] > a["sharpe_ann"]
    c_dd_ratio = c["max_dd_r"] / a["max_dd_r"] if a["max_dd_r"] != 0 else 1
    if b["sharpe_ann"] != a["sharpe_ann"]:
        c_captures = (c["sharpe_ann"] - a["sharpe_ann"]) / (b["sharpe_ann"] - a["sharpe_ann"]) * 100
    else:
        c_captures = 0
    print(
        f"\n  C Sharpe vs A: {c['sharpe_ann']:.4f} vs {a['sharpe_ann']:.4f} -> {'BETTER' if c_sharpe_better else 'WORSE'}"
    )
    print(f"  C captures {c_captures:.0f}% of B-vs-A Sharpe gain")
    print(f"  C MaxDD / A MaxDD: {c_dd_ratio:.2f}x")

    # D vs A
    d_expr_delta = d["portfolio_expr"] - a["portfolio_expr"]
    d_dd_pct = (d["max_dd_r"] - a["max_dd_r"]) / abs(a["max_dd_r"]) * 100 if a["max_dd_r"] != 0 else 0
    print(f"\n  D ExpR delta vs A: {d_expr_delta:+.4f}")
    print(f"  D MaxDD change: {d_dd_pct:+.1f}%")
    if d_expr_delta > 0.02 and d_dd_pct < 20:
        print("  RULE 4: D adds >0.02 ExpR, DD < 20% increase -> ADD CME_PRECLOSE")
    elif d_expr_delta > 0:
        print(f"  RULE 4: D adds {d_expr_delta:+.4f} ExpR (< 0.02 threshold)")

    # Current portfolio low-ATR verdict
    print(f"\n  A low-ATR ExpR: {a['low_atr_expr']:+.4f}")
    if a["low_atr_expr"] > 0:
        print("  VERDICT: Current portfolio EARNS on low-ATR days. Real diversification value exists.")
    else:
        print("  VERDICT: Current portfolio LOSES on low-ATR days. ATR70 concentration is cutting dead weight.")


if __name__ == "__main__":
    main()
