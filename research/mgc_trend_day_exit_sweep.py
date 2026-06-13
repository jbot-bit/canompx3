"""research/mgc_trend_day_exit_sweep.py

Phase A2 — MGC Trend-Day Runner EXIT × FILTER sweep (read-only, NO prereg, NO LOCK).

Phase A (mgc_trend_day_tail_descriptive.py) tested ONE exit (arm@1R, 50% giveback
trail) and concluded PARK. That was a thin search: a 50%-giveback trail is the
WORST harvester for a fat right tail — it caps every runner at half its peak. The
MFE ceiling proved the raw material is there (oracle TREND CLOSE %≥3R = 19.5%, max
49R). The real question is exit GEOMETRY × pre-entry CONDITIONING, not whether one
bad trail works.

This sweep computes, in ONE bar-replay pass per trade (the replay is the only
expensive step), a panel of REALIZABLE exits:

  EXIT FAMILY                          variants
  ─────────────────────────────────  ─────────────────────────────────────────
  fixed_target_{R}                     hold to a fixed R-multiple target (3/4/5/6R),
                                       hard stop live → captures the '4R+' framing
  trail_{arm}_{giveback}               give-back-of-peak trail, tight givebacks
                                       (arm 0.5/1R × giveback 10/20/30%)
  breakeven_hold                       move stop to breakeven at +1R, then hold to
                                       session close (no giveback bleed)
  scaleout_1R_runner                   bank half at +1R, run the rest to close
  hold_to_close                        pure hold-to-close (baseline, = Phase A close)
  mfe_ceiling                          NON-REALIZABLE upper bound (benchmark only)

Then it sweeps pre-entry-SAFE filters (interactions + ORB structure + session×dir)
against each exit and reports the realizable tail (sum-R, %≥3R, P90/P99, Sharpe,
gain-to-pain), year-by-year (RULE 12).

All exits NET of friction except mfe_ceiling. All filters pre-entry-safe
(backtesting-methodology § 6.1). day_type kept ONLY as the look-ahead oracle
upper-bound comparator.

@research-source: docs/plans/2026-06-14-mgc-trend-day-runner-phase-a.md
@entry-models: E1, E2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.build_daily_features import compute_trading_day_utc_range
from pipeline.cost_model import COST_SPECS, CostSpec, pnl_points_to_r
from pipeline.paths import GOLD_DB_PATH
from research.research_trend_day_mfe import load_bars_1m_for_day, load_outcomes

INSTRUMENT = "MGC"
IS_CUTOFF = pd.Timestamp("2026-01-01")
TESTABLE_SESSIONS = [
    "TOKYO_OPEN", "LONDON_METALS", "EUROPE_FLOW", "SINGAPORE_OPEN", "US_DATA_830",
    "NYSE_OPEN", "US_DATA_1000", "COMEX_SETTLE", "CME_REOPEN",
]
ORB_MINUTES = [5, 15, 30]
MIN_CELL_N = 30

FIXED_TARGETS_R = [3.0, 4.0, 5.0, 6.0]
TRAIL_VARIANTS = [(0.5, 0.10), (0.5, 0.20), (1.0, 0.10), (1.0, 0.20), (1.0, 0.30)]


# ---------------------------------------------------------------------------
# ONE-PASS multi-exit replay — all exits from a single per-trade bar walk
# ---------------------------------------------------------------------------
def compute_all_exits(
    bars_1m: pd.DataFrame,  # entry_ts < ts_utc <= trading_day_end, ORDERED
    entry_price: float,
    stop_price: float,
    break_dir: int,
    cost_spec: CostSpec,
) -> dict:
    """Walk the post-entry 1m bars ONCE; return realizable R for every exit
    variant plus the MFE ceiling. NET of friction except mfe_ceiling.

    Causal discipline (sub-bar look-ahead guard): within a 1m bar the high/low
    order is unknown. The hard stop and any target/trail are tested against state
    established by PRIOR bars before the current bar can ratchet the peak — so a
    bar's own low can't trip a trail its own high just raised. Adverse leg wins
    ties (bias DOWN — honest for a go/no-go gate).
    """
    out: dict = {}
    hard_stop_pts = abs(entry_price - stop_price)
    if bars_1m.empty or hard_stop_pts <= 0:
        keys = ["mfe_ceiling", "hold_to_close", "breakeven_hold", "scaleout_1R_runner"]
        keys += [f"fixed_{int(t)}R" for t in FIXED_TARGETS_R]
        keys += [f"trail_a{a}_g{int(g*100)}" for a, g in TRAIL_VARIANTS]
        return {k: None for k in keys}

    highs = bars_1m["high"].values
    lows = bars_1m["low"].values
    closes = bars_1m["close"].values
    n = len(bars_1m)

    friction_pts = cost_spec.total_friction / cost_spec.point_value

    def net_r(gross_pts: float) -> float:
        return round(pnl_points_to_r(cost_spec, entry_price, stop_price, gross_pts - friction_pts), 4)

    def gross_r(gross_pts: float) -> float:
        return round(pnl_points_to_r(cost_spec, entry_price, stop_price, gross_pts), 4)

    # Pre-compute per-bar favorable high/low excursions (points).
    if break_dir == 1:
        fav_high = highs - entry_price
        fav_low = lows - entry_price
    else:
        fav_high = entry_price - lows
        fav_low = entry_price - highs

    # --- MFE ceiling (non-realizable; friction in denom only) ---
    out["mfe_ceiling"] = gross_r(max(float(np.max(fav_high)), 0.0))

    # --- hold_to_close (realizable): rides to the close UNLESS the hard stop is
    #     hit intra-session first (you cannot hold through a stop-out). ---
    close_pts = (closes[-1] - entry_price) if break_dir == 1 else (entry_price - closes[-1])
    hold_done = None  # set to -hard_stop the first bar the stop is hit

    # --- fixed R-targets: first bar to reach target wins; hard stop first ---
    target_pts = {t: t * hard_stop_pts for t in FIXED_TARGETS_R}
    fixed_done = {t: None for t in FIXED_TARGETS_R}

    # --- trails ---
    trail_done = {(a, g): None for a, g in TRAIL_VARIANTS}
    trail_peak = {(a, g): 0.0 for a, g in TRAIL_VARIANTS}
    trail_armed = {(a, g): False for a, g in TRAIL_VARIANTS}

    # --- breakeven_hold: stop -> breakeven once peak >= 1R; then hold to close ---
    be_armed = False
    be_done = None

    # --- scaleout: bank 0.5 unit at +1R, run other 0.5 to close (or stop) ---
    scale_banked = None  # R banked on the first half (None until +1R reached)
    scale_done = None

    arm_1r_pts = 1.0 * hard_stop_pts

    for i in range(n):
        fh = fav_high[i]
        fl = fav_low[i]

        # ===== hard stop (adverse leg first) =====
        stopped = (-fl) >= hard_stop_pts

        # ----- hold_to_close still dies on the hard stop -----
        if hold_done is None and stopped:
            hold_done = net_r(-hard_stop_pts)

        # ----- fixed targets: target hit this bar? (favorable high reaches it) -----
        for t in FIXED_TARGETS_R:
            if fixed_done[t] is None:
                if stopped and fh < target_pts[t]:
                    fixed_done[t] = net_r(-hard_stop_pts)
                elif fh >= target_pts[t]:
                    fixed_done[t] = net_r(target_pts[t])
                elif stopped:
                    fixed_done[t] = net_r(-hard_stop_pts)

        # ----- trails: test against PRIOR peak, then ratchet -----
        for key in TRAIL_VARIANTS:
            if trail_done[key] is not None:
                continue
            a, g = key
            if stopped:
                trail_done[key] = net_r(-hard_stop_pts)
                continue
            if trail_armed[key]:
                lvl = trail_peak[key] * (1.0 - g)
                if fl <= lvl:
                    trail_done[key] = net_r(lvl)
                    continue
            if fh > trail_peak[key]:
                trail_peak[key] = fh
            if not trail_armed[key] and trail_peak[key] >= a * hard_stop_pts:
                trail_armed[key] = True

        # ----- breakeven_hold -----
        if be_done is None:
            if not be_armed:
                if stopped:
                    be_done = net_r(-hard_stop_pts)
                elif fh >= arm_1r_pts:
                    be_armed = True
            else:
                # stop is now at breakeven (0 pts). Exit at 0 if price returns to entry.
                if fl <= 0.0:
                    be_done = net_r(0.0)

        # ----- scaleout: first half banks +1R, second half rides -----
        if scale_banked is None:
            if stopped:
                scale_done = net_r(-hard_stop_pts)  # both halves stopped before +1R
                scale_banked = "stopped"
            elif fh >= arm_1r_pts:
                scale_banked = net_r(1.0 * hard_stop_pts)  # banked first half at +1R
        elif scale_banked != "stopped" and scale_done is None:
            # second half rides; dies at breakeven-or-stop? Use original hard stop on 2nd half.
            if stopped:
                # 2nd half stopped at -1R; blend 0.5*banked + 0.5*(-1R)
                scale_done = round(0.5 * float(scale_banked) + 0.5 * net_r(-hard_stop_pts), 4)

    # ---- resolve unfinished exits at session close ----
    close_r = net_r(close_pts)
    out["hold_to_close"] = hold_done if hold_done is not None else close_r

    for t in FIXED_TARGETS_R:
        out[f"fixed_{int(t)}R"] = fixed_done[t] if fixed_done[t] is not None else close_r

    for key in TRAIL_VARIANTS:
        a, g = key
        out[f"trail_a{a}_g{int(g*100)}"] = trail_done[key] if trail_done[key] is not None else close_r

    if be_done is not None:
        out["breakeven_hold"] = be_done
    elif be_armed:
        out["breakeven_hold"] = close_r  # armed, never returned to BE -> rode to close
    else:
        out["breakeven_hold"] = close_r  # never armed -> = hold to close

    if scale_done is not None:
        out["scaleout_1R_runner"] = scale_done
    elif scale_banked is not None and scale_banked != "stopped":
        # first half banked +1R, second half rode to close
        out["scaleout_1R_runner"] = round(0.5 * float(scale_banked) + 0.5 * close_r, 4)
    else:
        out["scaleout_1R_runner"] = close_r  # never reached +1R -> both ride to close

    return out


EXIT_KEYS = (
    ["mfe_ceiling", "hold_to_close", "breakeven_hold", "scaleout_1R_runner"]
    + [f"fixed_{int(t)}R" for t in FIXED_TARGETS_R]
    + [f"trail_a{a}_g{int(g*100)}" for a, g in TRAIL_VARIANTS]
)
REALIZABLE_EXITS = [k for k in EXIT_KEYS if k != "mfe_ceiling"]


# ---------------------------------------------------------------------------
def run_replay(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    cost_spec = COST_SPECS[INSTRUMENT]
    grouped = df.groupby("trading_day")
    n_days = len(grouped)
    print(f"  One-pass multi-exit replay: {len(df):,} outcomes / {n_days:,} days...")
    t0 = time.time()
    results, done = [], 0
    for di, (td, day_df) in enumerate(grouped):
        day_bars = load_bars_1m_for_day(con, INSTRUMENT, td)
        _, td_end = compute_trading_day_utc_range(td)
        td_end_ts = pd.Timestamp(td_end)
        for ridx, row in day_df.iterrows():
            ets = pd.Timestamp(row["entry_ts"])
            post = day_bars[(day_bars["ts_utc"] > ets) & (day_bars["ts_utc"] <= td_end_ts)]
            ex = compute_all_exits(post, row["entry_price"], row["stop_price"], row["break_dir"], cost_spec)
            ex["idx"] = ridx
            results.append(ex)
        done += len(day_df)
        if (di + 1) % 200 == 0 or (di + 1) == n_days:
            print(f"    [{di+1:,}/{n_days:,}] {done:,}/{len(df):,} ({done/len(df)*100:.0f}%) — {time.time()-t0:.0f}s")
    ex_df = pd.DataFrame(results).set_index("idx")
    for c in ex_df.columns:
        df[c] = ex_df[c]
    print(f"  Replay complete: {time.time()-t0:.0f}s\n")
    return df


# ---------------------------------------------------------------------------
def tail_stats(r: pd.Series) -> dict:
    a = np.asarray(r.dropna().to_numpy(), dtype=float)
    n = len(a)
    if n == 0:
        return {k: np.nan for k in ["n", "mean", "p90", "p99", "max", "pct_ge3r", "pct_ge4r", "sum_r", "sharpe", "gain_to_pain"]}
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else np.nan
    gains = float(a[a > 0].sum())
    pains = float(-a[a < 0].sum())
    return {
        "n": n, "mean": round(mean, 4),
        "p90": round(float(np.quantile(a, 0.90)), 3), "p99": round(float(np.quantile(a, 0.99)), 3),
        "max": round(float(np.max(a)), 2),
        "pct_ge3r": round(float((a >= 3.0).mean()) * 100, 2),
        "pct_ge4r": round(float((a >= 4.0).mean()) * 100, 2),
        "sum_r": round(float(np.sum(a)), 1),
        "sharpe": round(mean / std, 4) if std and std > 0 else np.nan,
        "gain_to_pain": round(gains / pains, 3) if pains > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Pre-entry-SAFE filters (interactions + ORB structure + session×dir)
# ---------------------------------------------------------------------------
def build_filters(df: pd.DataFrame) -> dict:
    """Return {filter_name: boolean mask}. ALL pre-entry-safe (§6.1).
    NaN-safe: a filter's required column NaN -> that row excluded (mask False)."""
    f: dict = {}
    f["ALL"] = pd.Series(True, index=df.index)

    # --- single safe proxies (kept for baseline) ---
    if "atr_20_pct" in df:
        thr = df["atr_20_pct"].dropna().quantile(2 / 3)
        f["ATR_HI"] = df["atr_20_pct"].notna() & (df["atr_20_pct"] >= thr)
        # Fitschen Ch6 p.100-101: HIGH-vol EXCLUSION raised commodity gain-to-pain
        # 1.23->2.11. Test ATR as an exclusion gate (skip the most volatile third),
        # NOT only as an inclusion prior.
        f["ATR_NOT_HI"] = df["atr_20_pct"].notna() & (df["atr_20_pct"] < thr)
    if "garch_atr_ratio" in df:
        f["GARCH_GT1"] = df["garch_atr_ratio"].notna() & (df["garch_atr_ratio"] > 1.0)
    if "is_nfp_day" in df:
        f["NFP"] = df["is_nfp_day"] == True  # noqa: E712

    # --- Fitschen Ch6 p.99 longer-term TREND-ALIGNMENT filter (his strongest
    #     commodity filter: +40% profit/trade). Take the breakout only when it
    #     aligns with the prior-day direction. break_dir 1=long/-1=short;
    #     prev_day_direction 'bull'/'bear'. ---
    if {"prev_day_direction", "break_dir"} <= set(df.columns):
        pdir = df["prev_day_direction"].map({"bull": 1, "bear": -1})
        f["TREND_ALIGNED"] = pdir.notna() & (pdir == df["break_dir"])
        f["TREND_COUNTER"] = pdir.notna() & (pdir != df["break_dir"])  # contrast arm

    # --- INTERACTIONS (the thin part of Phase A) ---
    if {"atr_20_pct", "gap_open_points"} <= set(df.columns):
        athr = df["atr_20_pct"].dropna().quantile(2 / 3)
        gthr = df["gap_open_points"].abs().dropna().quantile(2 / 3)
        f["ATRHI_x_GAPHI"] = (df["atr_20_pct"] >= athr) & (df["gap_open_points"].abs() >= gthr)
    if {"atr_20_pct", "garch_atr_ratio"} <= set(df.columns):
        athr = df["atr_20_pct"].dropna().quantile(2 / 3)
        f["ATRHI_x_GARCHGT1"] = (df["atr_20_pct"] >= athr) & (df["garch_atr_ratio"] > 1.0)
    if {"garch_atr_ratio", "gap_open_points"} <= set(df.columns):
        gthr = df["gap_open_points"].abs().dropna().quantile(2 / 3)
        f["GARCHGT1_x_GAPHI"] = (df["garch_atr_ratio"] > 1.0) & (df["gap_open_points"].abs() >= gthr)

    # --- ORB-STRUCTURE (breakout's own character) ---
    if "orb_size_pts" in df:  # session-specific ORB size extracted per-row (below)
        sthr = df["orb_size_pts"].dropna().quantile(2 / 3)
        f["ORBSIZE_HI"] = df["orb_size_pts"].notna() & (df["orb_size_pts"] >= sthr)
        slo = df["orb_size_pts"].dropna().quantile(1 / 3)
        f["ORBSIZE_LO"] = df["orb_size_pts"].notna() & (df["orb_size_pts"] <= slo)
    if "orb_compression_z" in df:
        cthr = df["orb_compression_z"].dropna().quantile(1 / 3)  # tight pre-ORB = compressed
        f["ORB_COMPRESSED"] = df["orb_compression_z"].notna() & (df["orb_compression_z"] <= cthr)
    if "orb_pre_velocity" in df:
        vthr = df["orb_pre_velocity"].abs().dropna().quantile(2 / 3)
        f["PREVEL_HI"] = df["orb_pre_velocity"].notna() & (df["orb_pre_velocity"].abs() >= vthr)

    return f


def extract_orb_structure(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """Pull session-specific orb_{S}_size / compression_z / pre_velocity and gap,
    extract the per-row session value. Triple-join on (trading_day, orb_minutes)."""
    sessions = sorted(set(df["orb_label"].unique()))
    size_cols = ", ".join(f'orb_{s}_size' for s in sessions if _col_exists(con, f'orb_{s}_size'))
    comp_cols = ", ".join(f'orb_{s}_compression_z' for s in sessions if _col_exists(con, f'orb_{s}_compression_z'))
    vel_cols = ", ".join(f'orb_{s}_pre_velocity' for s in sessions if _col_exists(con, f'orb_{s}_pre_velocity'))
    extra = ", ".join(c for c in [size_cols, comp_cols, vel_cols] if c)
    sql = f"""
        SELECT trading_day, orb_minutes, day_type, prev_day_direction, atr_20_pct,
               garch_atr_ratio, is_nfp_day, gap_open_points, day_of_week{(', ' + extra) if extra else ''}
        FROM daily_features WHERE symbol = ?
    """
    feats = con.execute(sql, [INSTRUMENT]).fetchdf()
    feats["trading_day"] = pd.to_datetime(feats["trading_day"])
    df = df.merge(feats, on=["trading_day", "orb_minutes"], how="left")

    # Extract per-row session-specific ORB structure into flat columns.
    for base in ["size", "compression_z", "pre_velocity"]:
        target = "orb_size_pts" if base == "size" else f"orb_{base}"
        df[target] = np.nan
        for s in sessions:
            col = f"orb_{s}_{base}"
            if col in df.columns:
                m = df["orb_label"] == s
                df.loc[m, target] = df.loc[m, col]
    return df


def _col_exists(con: duckdb.DuckDBPyConnection, col: str) -> bool:
    try:
        con.execute(f"SELECT {col} FROM daily_features LIMIT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
def sweep(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Exit × filter grid against the realizable tail. Year-by-year for the best."""
    filters = build_filters(df)
    df = df.copy()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year

    rows = []
    for fname, mask in filters.items():
        sub = df[mask.fillna(False)]
        if len(sub) < MIN_CELL_N:
            continue
        for exit_key in EXIT_KEYS:
            s = tail_stats(sub[exit_key])
            if s["n"] < MIN_CELL_N:
                continue
            # year stability: fraction of years with positive sum_r (realizable only)
            yrs = sub.groupby("year")[exit_key].apply(lambda x: float(np.nansum(x.values)))
            pos_years = int((yrs > 0).sum())
            n_years = int(yrs.notna().sum())
            rows.append({
                "filter": fname, "exit": exit_key, "fire_rate": round(len(sub) / len(df) * 100, 1),
                **s, "pos_years": pos_years, "n_years": n_years,
            })
    grid = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out / "mgc_exit_sweep_grid.csv", index=False)

    # --- report: best realizable cells by sum_r (exclude ceiling) ---
    real = grid[grid["exit"] != "mfe_ceiling"].copy()
    print("  " + "=" * 92)
    print("  TOP REALIZABLE CELLS by sum_R (exit × filter) — full grid in mgc_exit_sweep_grid.csv")
    print("  " + "=" * 92)
    print(f"  {'filter':18s} {'exit':18s} {'fire%':>5s} {'N':>7s} {'mean':>7s} {'%≥3R':>6s} {'%≥4R':>6s} {'sumR':>9s} {'Shrp':>6s} {'G/P':>5s} {'yrs+':>5s}")
    top = real.sort_values("sum_r", ascending=False).head(25)
    for _, r in top.iterrows():
        print(f"  {r['filter']:18s} {r['exit']:18s} {r['fire_rate']:>5.1f} {r['n']:>7} {r['mean']:>7.3f} "
              f"{r['pct_ge3r']:>5.1f}% {r['pct_ge4r']:>5.1f}% {r['sum_r']:>9.1f} {r['sharpe']:>6.2f} {r['gain_to_pain']:>5.2f} {r['pos_years']}/{r['n_years']}")
    print()

    # --- best exit on ALL (no filter) — is any exit positive on the full book? ---
    print("  " + "=" * 92)
    print("  EXIT COMPARISON on ALL trades (no filter) — which geometry harvests the tail?")
    print("  " + "=" * 92)
    allf = grid[grid["filter"] == "ALL"].sort_values("sum_r", ascending=False)
    print(f"  {'exit':18s} {'N':>7s} {'mean':>7s} {'P90':>6s} {'P99':>6s} {'max':>7s} {'%≥3R':>6s} {'%≥4R':>6s} {'sumR':>9s} {'Shrp':>6s} {'yrs+':>5s}")
    for _, r in allf.iterrows():
        tag = " (CEILING)" if r["exit"] == "mfe_ceiling" else ""
        print(f"  {r['exit']:18s} {r['n']:>7} {r['mean']:>7.3f} {r['p90']:>6.2f} {r['p99']:>6.2f} {r['max']:>7.2f} "
              f"{r['pct_ge3r']:>5.1f}% {r['pct_ge4r']:>5.1f}% {r['sum_r']:>9.1f} {r['sharpe']:>6.2f} {r['pos_years']}/{r['n_years']}{tag}")
    print()
    print(f"  Full grid ({len(grid)} cells) saved: {out / 'mgc_exit_sweep_grid.csv'}\n")
    return grid


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="MGC trend-day EXIT × FILTER sweep (Phase A2)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default="research/output/")
    args = ap.parse_args()

    print("=== MGC Trend-Day EXIT × FILTER Sweep (Phase A2) ===")
    print(f"DB: {GOLD_DB_PATH} | {INSTRUMENT} | IS < {IS_CUTOFF.date()}")
    print(f"Exits: {len(REALIZABLE_EXITS)} realizable + MFE ceiling\n")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    try:
        df = load_outcomes(con, INSTRUMENT, limit=args.limit)
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        df = df[df["trading_day"] < IS_CUTOFF]
        df = df[df["orb_label"].isin(TESTABLE_SESSIONS)]
        df = df[df["orb_minutes"].isin(ORB_MINUTES)]
        print(f"  Outcomes (IS, testable, E1/E2): {len(df):,}")
        if df.empty:
            print("  No outcomes — abort.")
            return

        df = run_replay(con, df)
        df = extract_orb_structure(con, df)
        sweep(df, args.output_dir)

        # persist per-trade exits for downstream Phase B selection
        out = Path(args.output_dir)
        keep = ["trading_day", "orb_label", "orb_minutes", "entry_model", "rr_target",
                "day_type", "prev_day_direction", "atr_20_pct", "garch_atr_ratio",
                "is_nfp_day", "gap_open_points", "orb_size_pts", "orb_compression_z",
                "orb_pre_velocity", "day_of_week"] + EXIT_KEYS
        keep = [c for c in keep if c in df.columns]
        df[keep].to_csv(out / "mgc_exit_sweep_raw.csv", index=False)
        print(f"  Per-trade exits saved: {out / 'mgc_exit_sweep_raw.csv'}")
    finally:
        con.close()

    print("=" * 60)
    print("PHASE A2 COMPLETE — exit × filter sweep. Read the grid before any verdict.")
    print("=" * 60)


if __name__ == "__main__":
    main()
