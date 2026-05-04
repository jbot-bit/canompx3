"""research/research_breakeven_stop.py

Breakeven Stop Research: Move stop to entry after condition is met.
NO partial exit — full position preserved, full target upside kept.

Two trigger types tested:
  A. TIME-BASED: After X minutes, if MTM > threshold, move stop to entry
  B. PRICE-BASED: Once price reaches X·R favorable, move stop to entry

Why this is different from partial profit:
  - Partial profit EXITS 50% early, clipping winner tails (destroyed -30K R)
  - Breakeven stop keeps 100% of position, just tightens the stop
  - Preserves ALL upside while protecting against late reversals
  - No extra friction (same 1 entry + 1 exit)

Lookahead guards:
  - ALL E1+E2 outcomes (not filtered to validated)
  - Parameter grid with BH FDR
  - Ambiguous bar = breakeven stop wins (conservative)

@research-source: docs/plans/2026-03-04-m25-audit-improvements-plan.md (Phase 3 extension)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS, risk_in_dollars
from pipeline.paths import GOLD_DB_PATH
from research.lib.stats import bh_fdr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
DB_PATH = GOLD_DB_PATH
MIN_TRADES = 30

# Grid A: time-based breakeven
CHECK_MINUTES = [10, 15, 20, 30, 45, 60]
MTM_THRESHOLDS = [0.0, 0.25, 0.5]  # min MTM in R to trigger

# Grid B: price-based breakeven (move to BE after reaching X·R)
PRICE_TRIGGERS = [0.5, 0.75, 1.0, 1.25, 1.5]  # in R-multiples


# ---------------------------------------------------------------------------
# Bar-by-bar simulation
# ---------------------------------------------------------------------------
def sim_be_time(
    highs,
    lows,
    closes,
    timestamps,
    direction,
    entry_price,
    stop_price,
    target_price,
    risk_points,
    check_min,
    mtm_thresh,
    entry_ts,
):
    """Time-based breakeven: after check_min minutes, if MTM > thresh, move stop to entry."""
    n = len(highs)
    if n == 0:
        return None

    d = direction
    ep, sp, tp, rp = entry_price, stop_price, target_price, risk_points
    check_ts = entry_ts + np.timedelta64(check_min, "m")

    # Find check bar index
    check_idx = int(np.searchsorted(timestamps, check_ts)) - 1

    # Check if trade already exited before check time
    for b in range(min(check_idx + 1, n)):
        if d == 1:
            if highs[b] >= tp:
                return 0.0  # target hit before check -> no change
            if lows[b] <= sp:
                return 0.0  # stop hit before check -> no change
        else:
            if lows[b] <= tp:
                return 0.0
            if highs[b] >= sp:
                return 0.0

    if check_idx < 0 or check_idx >= n:
        return 0.0  # check time outside session

    # MTM at check time
    mtm_points = (closes[check_idx] - ep) * d
    mtm_r = mtm_points / rp

    if mtm_r < mtm_thresh:
        return 0.0  # not triggered

    # TRIGGERED: simulate remainder with stop = entry_price (breakeven)
    for b in range(check_idx + 1, n):
        if d == 1:
            hit_target = highs[b] >= tp
            hit_be = lows[b] <= ep
        else:
            hit_target = lows[b] <= tp
            hit_be = highs[b] >= ep

        if hit_be and hit_target:
            # Ambiguous -> breakeven wins (conservative)
            exit_pnl = 0.0
            break
        elif hit_be:
            exit_pnl = 0.0  # breakeven
            break
        elif hit_target:
            exit_pnl = (tp - ep) * d  # target
            break
    else:
        # EOD
        exit_pnl = (closes[-1] - ep) * d

    return exit_pnl


def sim_be_price(
    highs,
    lows,
    closes,
    direction,
    entry_price,
    stop_price,
    target_price,
    risk_points,
    price_trigger_r,
):
    """Price-based breakeven: once price reaches trigger_r, move stop to entry."""
    n = len(highs)
    if n == 0:
        return None

    d = direction
    ep, sp, tp, rp = entry_price, stop_price, target_price, risk_points
    trigger_price = ep + price_trigger_r * rp * d

    # Scan for trigger, stop, or target
    for b in range(n):
        if d == 1:
            hit_trigger = highs[b] >= trigger_price
            hit_stop = lows[b] <= sp
            hit_target = highs[b] >= tp
        else:
            hit_trigger = lows[b] <= trigger_price
            hit_stop = highs[b] >= sp
            hit_target = lows[b] <= tp

        # Stop before trigger -> no change
        if hit_stop and not hit_trigger:
            return 0.0

        # Target before trigger -> no change (already a winner)
        if hit_target and not hit_trigger:
            return 0.0

        if hit_trigger:
            # Check: if stop ALSO hit in same bar, stop wins (no trigger)
            if hit_stop:
                return 0.0

            # TRIGGERED: simulate remainder with stop = entry_price
            for b2 in range(b + 1, n):
                if d == 1:
                    h_target = highs[b2] >= tp
                    h_be = lows[b2] <= ep
                else:
                    h_target = lows[b2] <= tp
                    h_be = highs[b2] >= ep

                if h_be and h_target:
                    exit_pnl = 0.0  # ambiguous -> BE wins
                    break
                elif h_be:
                    exit_pnl = 0.0
                    break
                elif h_target:
                    exit_pnl = (tp - ep) * d
                    break
            else:
                exit_pnl = (closes[-1] - ep) * d

            return exit_pnl

    # Nothing hit -> no change (EOD same as baseline)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    results_time = []
    results_price = []

    for instrument in INSTRUMENTS:
        cost = COST_SPECS[instrument]
        print(f"\nProcessing {instrument}...")

        # Load bars
        bars = con.execute(
            "SELECT ts_utc, high, low, close FROM bars_1m WHERE symbol = ? ORDER BY ts_utc",
            [instrument],
        ).fetchdf()
        bars["ts_utc"] = pd.to_datetime(bars["ts_utc"]).dt.tz_localize(None)
        bars_ts = bars["ts_utc"].values.astype("datetime64[us]")
        bars_h = bars["high"].values
        bars_l = bars["low"].values
        bars_c = bars["close"].values

        # Load outcomes
        ph = ",".join(["?"] * len(RR_TARGETS))
        outcomes = con.execute(
            f"SELECT trading_day, orb_label, rr_target, "
            f"  entry_ts, entry_price, stop_price, target_price, pnl_r, "
            f"  CASE WHEN target_price > entry_price THEN 1 ELSE -1 END AS direction "
            f"FROM orb_outcomes "
            f"WHERE symbol = ? AND entry_model IN ('E1','E2') "
            f"  AND confirm_bars = 1 AND orb_minutes = 5 "
            f"  AND rr_target IN ({ph}) "
            f"  AND entry_ts IS NOT NULL AND entry_price IS NOT NULL "
            f"  AND stop_price IS NOT NULL AND target_price IS NOT NULL",
            [instrument] + RR_TARGETS,
        ).fetchdf()
        outcomes["entry_ts"] = pd.to_datetime(outcomes["entry_ts"]).dt.tz_localize(None)
        outcomes["risk_points"] = (outcomes["entry_price"] - outcomes["stop_price"]).abs()
        outcomes["td_end"] = pd.to_datetime(outcomes["trading_day"]) + pd.Timedelta(hours=23)
        print(f"  {len(outcomes):,} outcomes")

        t0 = time.time()
        for i, row in enumerate(outcomes.itertuples(index=False)):
            if row.risk_points <= 0:
                continue

            entry_ts = np.datetime64(row.entry_ts, "us")
            td_end = np.datetime64(row.td_end, "us")
            si = int(np.searchsorted(bars_ts, entry_ts))
            ei = int(np.searchsorted(bars_ts, td_end))
            if si >= ei:
                continue

            h = bars_h[si:ei]
            l = bars_l[si:ei]
            c = bars_c[si:ei]
            ts = bars_ts[si:ei]
            d = int(row.direction)
            ep = float(row.entry_price)
            sp = float(row.stop_price)
            tp = float(row.target_price)
            rp = float(row.risk_points)
            baseline_r = float(row.pnl_r) if row.pnl_r is not None else 0.0

            risk_dollars = risk_in_dollars(cost, ep, sp)
            if risk_dollars <= 0:
                continue

            # --- Grid A: time-based ---
            for cm in CHECK_MINUTES:
                for mt in MTM_THRESHOLDS:
                    exit_pnl = sim_be_time(
                        h,
                        l,
                        c,
                        ts,
                        d,
                        ep,
                        sp,
                        tp,
                        rp,
                        cm,
                        mt,
                        entry_ts,
                    )
                    if exit_pnl is None or exit_pnl == 0.0:
                        delta = 0.0
                        be_r = baseline_r
                    else:
                        pnl_dollars = exit_pnl * cost.point_value - cost.total_friction
                        be_r = pnl_dollars / risk_dollars
                        delta = be_r - baseline_r

                    results_time.append(
                        {
                            "trading_day": row.trading_day,
                            "instrument": instrument,
                            "session": row.orb_label,
                            "rr_target": row.rr_target,
                            "check_min": cm,
                            "mtm_thresh": mt,
                            "baseline_r": baseline_r,
                            "be_r": be_r,
                            "delta_r": delta,
                        }
                    )

            # --- Grid B: price-based ---
            for ptr in PRICE_TRIGGERS:
                if ptr >= row.rr_target:
                    continue  # trigger above target is meaningless

                exit_pnl = sim_be_price(
                    h,
                    l,
                    c,
                    d,
                    ep,
                    sp,
                    tp,
                    rp,
                    ptr,
                )
                if exit_pnl is None or exit_pnl == 0.0:
                    delta = 0.0
                    be_r = baseline_r
                else:
                    pnl_dollars = exit_pnl * cost.point_value - cost.total_friction
                    be_r = pnl_dollars / risk_dollars
                    delta = be_r - baseline_r

                results_price.append(
                    {
                        "trading_day": row.trading_day,
                        "instrument": instrument,
                        "session": row.orb_label,
                        "rr_target": row.rr_target,
                        "price_trigger_r": ptr,
                        "baseline_r": baseline_r,
                        "be_r": be_r,
                        "delta_r": delta,
                    }
                )

            if (i + 1) % 50000 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                rem = (len(outcomes) - i - 1) / rate
                print(f"  {i + 1:,}/{len(outcomes):,} ({el:.0f}s, ~{rem:.0f}s remaining)")

        print(f"  Done in {time.time() - t0:.0f}s")

    con.close()

    # --- Analyze time-based ---
    print("\n" + "=" * 80)
    print("GRID A: TIME-BASED BREAKEVEN STOP")
    print("=" * 80)
    analyze_grid(
        pd.DataFrame(results_time),
        ["instrument", "session", "rr_target", "check_min", "mtm_thresh"],
        agg_cols=["check_min", "mtm_thresh"],
    )

    # --- Analyze price-based ---
    print("\n" + "=" * 80)
    print("GRID B: PRICE-BASED BREAKEVEN STOP")
    print("=" * 80)
    analyze_grid(
        pd.DataFrame(results_price),
        ["instrument", "session", "rr_target", "price_trigger_r"],
        agg_cols=["price_trigger_r"],
    )


def analyze_grid(df, group_cols, agg_cols):
    """Run paired t-tests + BH FDR on a results grid."""
    if df.empty:
        print("No results.")
        return

    rows = []
    for key, grp in df.groupby(group_cols):
        vals = dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
        n = len(grp)
        if n < MIN_TRADES:
            continue

        daily = (
            grp.groupby("trading_day")
            .agg(
                b=("baseline_r", "sum"),
                p=("be_r", "sum"),
            )
            .reset_index()
        )
        daily["d"] = daily["p"] - daily["b"]
        non_zero = daily["d"].values[daily["d"].values != 0]
        if len(non_zero) < 10:
            continue

        t_stat, p_two = stats.ttest_1samp(non_zero, 0.0)
        p_val = float(p_two / 2) if t_stat > 0 else 1.0 - float(p_two / 2)

        total_delta = grp["delta_r"].sum()
        grp2 = grp.copy()
        grp2["year"] = pd.to_datetime(grp2["trading_day"]).dt.year
        yearly = grp2.groupby("year")["delta_r"].sum()

        vals.update(
            {
                "n_trades": n,
                "total_delta_r": round(total_delta, 2),
                "per_trade": round(total_delta / n, 4),
                "t_stat": round(t_stat, 3),
                "p_val": round(p_val, 6),
                "n_years": len(yearly),
                "years_pos": int((yearly > 0).sum()),
            }
        )
        rows.append(vals)

    sdf = pd.DataFrame(rows)
    if sdf.empty:
        print("No cells with enough trades.")
        return

    # BH FDR
    rejected = bh_fdr(sdf["p_val"].values, q=0.05)
    sdf["bh_pass"] = [i in rejected for i in range(len(sdf))]

    n_tests = len(sdf)
    n_fdr = sdf["bh_pass"].sum()
    print(f"Total tests: {n_tests}, BH FDR survivors: {n_fdr}")

    # Aggregate by parameter combo
    print(f"\n{'---' * 25}")
    print("BY PARAMETER COMBO:")
    for combo_vals, sub in sdf.groupby(agg_cols):
        if not isinstance(combo_vals, tuple):
            combo_vals = (combo_vals,)
        label = " ".join(f"{c}={v}" for c, v in zip(agg_cols, combo_vals))
        td = sub["total_delta_r"].sum()
        nf = sub["bh_pass"].sum()
        avg_pt = sub["per_trade"].mean()
        n_pos = (sub["total_delta_r"] > 0).sum()
        print(
            f"  {label}: delta={td:+10,.0f}R  avg/trade={avg_pt:+.4f}R  "
            f"FDR={nf}/{len(sub)}  positive={n_pos}/{len(sub)}"
        )

    # FDR survivors
    surv = sdf[sdf["bh_pass"]].sort_values("total_delta_r", ascending=False)
    if len(surv) > 0:
        print(f"\n{'---' * 25}")
        print(f"BH FDR SURVIVORS ({len(surv)}):")
        for _, r in surv.head(40).iterrows():
            params = " ".join(f"{c}={r[c]}" for c in agg_cols)
            print(
                f"  {r['instrument']} {r['session']} RR{r['rr_target']} | "
                f"{params} | N={int(r['n_trades']):,} "
                f"delta={r['total_delta_r']:+,.1f}R "
                f"({r['per_trade']:+.4f}R/trade) "
                f"p={r['p_val']:.6f} yrs+={r['years_pos']}/{r['n_years']}"
            )

    # Top positive even if not FDR
    pos = sdf[sdf["total_delta_r"] > 0].sort_values("total_delta_r", ascending=False)
    if len(pos) > 0:
        print(f"\n{'---' * 25}")
        print(f"TOP POSITIVE ({len(pos)} combos with positive delta):")
        for _, r in pos.head(20).iterrows():
            params = " ".join(f"{c}={r[c]}" for c in agg_cols)
            fdr_mark = " [FDR]" if r.get("bh_pass", False) else ""
            print(
                f"  {r['instrument']} {r['session']} RR{r['rr_target']} | "
                f"{params} | N={int(r['n_trades']):,} "
                f"delta={r['total_delta_r']:+,.1f}R "
                f"({r['per_trade']:+.4f}R/trade) "
                f"p={r['p_val']:.4f} yrs+={r['years_pos']}/{r['n_years']}{fdr_mark}"
            )

    # Save
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    tag = "_".join(agg_cols)
    sdf.to_csv(out / f"breakeven_stop_{tag}.csv", index=False)
    print(f"\nSaved to: {out / f'breakeven_stop_{tag}.csv'}")


if __name__ == "__main__":
    main()
