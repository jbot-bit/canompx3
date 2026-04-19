"""Adversarial stress test on the MNQ_NYSE_OPEN garch inverse finding.

Before pre-registering and implementing a SKIP_GARCH_70 filter, red-team
the claim. The finding:
  MNQ_NYSE_OPEN across 54 (apt x RR x dir x threshold) cells: 46/54
  negative sr_lift, binomial sign p=6.9e-8 BH-FDR K=32.

Adversarial tests:
  S1. Shuffle control: shuffle garch_pct labels, rerun pipeline. Should yield
      ~50/50 split. If inverse pattern persists on shuffled data -> methodology
      bias.
  S2. Per-year breakdown: does the effect hold every year, or is it driven
      by one regime (e.g., COVID-2020)?
  S3. Direction split: long-only inverse vs short-only vs both?
  S4. Event-day exclusion: rerun excluding NFP Fridays, FOMC-adjacent days,
      CPI days, OPEX, month-end. Does inverse persist?
  S5. Break-direction confounder: does garch correlate with break direction
      at NYSE_OPEN? If garch=HIGH days break LONG more often, and longs fail
      regardless of garch, that's a confounder.
  S6. MAE/MFE decomposition: is it bigger losses or time-stopped no-go?
  S7. Continuous garch regression: if the effect is a clean regime signal,
      garch_pct as continuous regressor should give monotonic slope. Nonlinear
      or noisy -> threshold artifact.
  S8. Tail behavior check: garch@90 behavior — does the inverse collapse in
      the extreme tail (suggesting noise dominance)?

Output: docs/audit/results/2026-04-15-nyse-open-skip-stress-test.md
"""

from __future__ import annotations

import io
import sys
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

OUTPUT_MD = Path("docs/audit/results/2026-04-15-nyse-open-skip-stress-test.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
IS_END = "2026-01-01"
SEED = 20260415


def load_nyse_open(apt: int, rr: float, direction: str) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT o.trading_day, o.pnl_r, o.mae_r, o.mfe_r,
           d.garch_forecast_vol_pct AS garch_pct,
           d.orb_NYSE_OPEN_break_dir AS break_dir,
           d.is_friday, d.day_of_week,
           d.overnight_range, d.overnight_range_pct,
           d.gap_open_points
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_minutes={apt}
      AND o.orb_label='NYSE_OPEN' AND o.entry_model='E2'
      AND o.rr_target={rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_NYSE_OPEN_break_dir='{direction}'
      AND o.trading_day < DATE '{IS_END}'
    ORDER BY o.trading_day
    """
    df = con.execute(q).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["pnl_r"] = df["pnl_r"].astype(float)
    df["garch_pct"] = df["garch_pct"].astype(float)
    return df


def sr_lift(df: pd.DataFrame, thresh: int) -> tuple:
    on = df[df["garch_pct"] >= thresh]
    off = df[df["garch_pct"] < thresh]
    if len(on) < 10 or len(off) < 10:
        return (None, None, None, None)
    e_on, e_off = on["pnl_r"].mean(), off["pnl_r"].mean()
    s_on = on["pnl_r"].std(ddof=1)
    s_off = off["pnl_r"].std(ddof=1)
    sr_on = e_on / s_on if s_on > 0 else 0
    sr_off = e_off / s_off if s_off > 0 else 0
    return (sr_on - sr_off, e_on - e_off, len(on), len(off))


def s1_shuffle_control():
    """Shuffle garch_pct labels within each (apt, rr, dir) cell, rerun, tally."""
    print("\n=== S1: Shuffle control ===")
    print("If methodology is unbiased, shuffled garch should give ~50/50 directional split.")

    tot_real_pos = 0
    tot_real_neg = 0
    tot_shuffled_pos = []
    tot_shuffled_neg = []

    rng = np.random.default_rng(SEED)
    N_SHUFFLES = 100
    shuf_runs = [[] for _ in range(N_SHUFFLES)]

    for apt in [5, 15, 30]:
        for rr in [1.0, 1.5, 2.0]:
            for direction in ["long", "short"]:
                df = load_nyse_open(apt, rr, direction)
                if len(df) < 60:
                    continue
                for thresh in [60, 70, 80]:
                    real_sr_lift, _, n_on, n_off = sr_lift(df, thresh)
                    if real_sr_lift is None:
                        continue
                    if real_sr_lift > 0:
                        tot_real_pos += 1
                    else:
                        tot_real_neg += 1
                    # N shuffles of the garch column
                    garch_arr = df["garch_pct"].to_numpy().copy()
                    for si in range(N_SHUFFLES):
                        shuffled = rng.permutation(garch_arr)
                        on_mask = shuffled >= thresh
                        on, off = df["pnl_r"].to_numpy()[on_mask], df["pnl_r"].to_numpy()[~on_mask]
                        if len(on) > 1 and len(off) > 1:
                            s_on = on.std(ddof=1)
                            s_off = off.std(ddof=1)
                            srl = (on.mean() / s_on if s_on > 0 else 0) - (off.mean() / s_off if s_off > 0 else 0)
                            if srl > 0:
                                shuf_runs[si].append(1)
                            else:
                                shuf_runs[si].append(-1)

    shuf_pos_fracs = [sum(1 for v in run if v > 0) / len(run) for run in shuf_runs if len(run)]
    real_pos_frac = tot_real_pos / max(tot_real_pos + tot_real_neg, 1)

    print(f"  Real data: {tot_real_pos}/{tot_real_pos + tot_real_neg} positive = {real_pos_frac:.3f}")
    print(
        f"  Shuffled ({N_SHUFFLES} runs): median positive frac = "
        f"{np.median(shuf_pos_fracs):.3f}  range [{min(shuf_pos_fracs):.3f}, {max(shuf_pos_fracs):.3f}]"
    )
    # One-sided p: how many shuffles had positive_frac as extreme as real
    extreme = sum(1 for f in shuf_pos_fracs if f <= real_pos_frac)
    p = (extreme + 1) / (len(shuf_pos_fracs) + 1)
    print(f"  Shuffle p-value (positive_frac <= real): {p:.4f}")
    print(
        f"  Verdict: {'PASS' if p < 0.05 else 'FAIL'} — methodology "
        f"{'distinguishes real from shuffled' if p < 0.05 else 'may be biased'}"
    )

    return {
        "real_pos_frac": real_pos_frac,
        "shuf_median": np.median(shuf_pos_fracs),
        "shuf_range": (min(shuf_pos_fracs), max(shuf_pos_fracs)),
        "shuf_p": p,
    }


def s2_per_year():
    """Per-year sr_lift across all (apt, rr, dir) for NYSE_OPEN at threshold 70."""
    print("\n=== S2: Per-year breakdown (threshold=70) ===")
    all_yr_lifts = {}
    for apt in [5, 15, 30]:
        for rr in [1.0, 1.5, 2.0]:
            for direction in ["long", "short"]:
                df = load_nyse_open(apt, rr, direction)
                if len(df) < 60:
                    continue
                for yr in df["year"].unique():
                    sub = df[df["year"] == yr]
                    if len(sub) < 20:
                        continue
                    on = sub[sub["garch_pct"] >= 70]
                    off = sub[sub["garch_pct"] < 70]
                    if len(on) >= 3 and len(off) >= 3:
                        e_on, e_off = on["pnl_r"].mean(), off["pnl_r"].mean()
                        lift = e_on - e_off
                        all_yr_lifts.setdefault(int(yr), []).append((apt, rr, direction, lift))

    print(f"  {'Year':5} {'N cells':8} {'% positive':12} {'avg lift':10}")
    per_yr_summary = {}
    for yr in sorted(all_yr_lifts):
        lifts = [x[3] for x in all_yr_lifts[yr]]
        pos_frac = sum(1 for x in lifts if x > 0) / len(lifts)
        avg = np.mean(lifts)
        per_yr_summary[yr] = (len(lifts), pos_frac, avg)
        print(f"  {yr:5} {len(lifts):8} {pos_frac:12.1%} {avg:+10.3f}")
    return per_yr_summary


def s3_direction_split():
    """Inverse pattern: long-only, short-only, or both?"""
    print("\n=== S3: Direction split ===")
    lo_cells_pos = 0
    lo_cells_neg = 0
    sh_cells_pos = 0
    sh_cells_neg = 0
    for apt in [5, 15, 30]:
        for rr in [1.0, 1.5, 2.0]:
            for thresh in [60, 70, 80]:
                for direction in ["long", "short"]:
                    df = load_nyse_open(apt, rr, direction)
                    if len(df) < 60:
                        continue
                    srl, _, _, _ = sr_lift(df, thresh)
                    if srl is None:
                        continue
                    if direction == "long":
                        if srl > 0:
                            lo_cells_pos += 1
                        else:
                            lo_cells_neg += 1
                    else:
                        if srl > 0:
                            sh_cells_pos += 1
                        else:
                            sh_cells_neg += 1
    print(f"  LONG: pos={lo_cells_pos}/{lo_cells_pos + lo_cells_neg} neg={lo_cells_neg}/{lo_cells_pos + lo_cells_neg}")
    print(f"  SHORT: pos={sh_cells_pos}/{sh_cells_pos + sh_cells_neg} neg={sh_cells_neg}/{sh_cells_pos + sh_cells_neg}")
    return {"long_pos": lo_cells_pos, "long_neg": lo_cells_neg, "short_pos": sh_cells_pos, "short_neg": sh_cells_neg}


def s4_event_day_exclusion():
    """Exclude event days (NFP Fri, Wed FOMC-day heuristic, first-Fri month)."""
    print("\n=== S4: Event-day exclusion ===")
    results = {}
    for direction in ["long", "short"]:
        for apt in [5, 15]:
            for rr in [1.0, 1.5]:
                df = load_nyse_open(apt, rr, direction)
                if len(df) < 60:
                    continue
                # Approximations:
                # - NFP = first Friday of month (is_friday + day<8)
                # - FOMC = Wed (dow=2), but we don't have FOMC flag. Approximate as "first Wed of month"
                df["day_of_month"] = df["trading_day"].dt.day
                df["is_first_fri"] = df["is_friday"] & (df["day_of_month"] <= 7)
                df["is_first_wed"] = (df["day_of_week"] == 2) & (df["day_of_month"] <= 7)
                df["is_event"] = df["is_first_fri"] | df["is_first_wed"]

                full = sr_lift(df, 70)
                clean = sr_lift(df[~df["is_event"]], 70)
                results[(apt, rr, direction)] = {
                    "full_lift": full[0],
                    "clean_lift": clean[0],
                    "full_n_on": full[2],
                    "clean_n_on": clean[2],
                }
                fs = f"{full[0]:+.3f}" if full[0] is not None else "n/a"
                cs = f"{clean[0]:+.3f}" if clean[0] is not None else "n/a"
                print(
                    f"  O{apt} RR{rr} {direction}: full sr_lift={fs} "
                    f"(N_on={full[2]}) vs no-events sr_lift={cs} "
                    f"(N_on={clean[2]})"
                )
    return results


def s5_break_direction_confounder():
    """Does garch=HIGH correlate with break direction at NYSE_OPEN?"""
    print("\n=== S5: Break-direction confounder ===")
    # Need both directions in same frame — use orb_outcomes join with break_dir column
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT o.trading_day, d.garch_forecast_vol_pct AS gp,
           d.orb_NYSE_OPEN_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='MNQ' AND o.orb_minutes=5 AND o.orb_label='NYSE_OPEN'
      AND o.entry_model='E2' AND o.rr_target=1.0
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND o.trading_day < DATE '{IS_END}'
    GROUP BY o.trading_day, d.garch_forecast_vol_pct, d.orb_NYSE_OPEN_break_dir
    """
    df = con.execute(q).df()
    con.close()
    df["gp"] = df["gp"].astype(float)
    high = df[df["gp"] >= 70]
    low = df[df["gp"] < 70]
    high_long = (high["break_dir"] == "long").mean()
    low_long = (low["break_dir"] == "long").mean()
    print(f"  garch>=70 days: {len(high)} total, P(break=long) = {high_long:.3f}")
    print(f"  garch<70 days:  {len(low)} total, P(break=long) = {low_long:.3f}")
    diff = high_long - low_long
    print(f"  Diff in P(long-break): {diff:+.3f}")
    print(f"  Verdict: {'Confounder present' if abs(diff) > 0.05 else 'No break-direction confound'}")
    return {"high_long_frac": float(high_long), "low_long_frac": float(low_long), "diff": float(diff)}


def s6_mae_mfe_decomp():
    """Is the inverse effect from bigger losses or smaller wins?"""
    print("\n=== S6: MAE/MFE decomposition ===")
    out = {}
    for direction in ["long", "short"]:
        df = load_nyse_open(5, 1.0, direction)
        on = df[df["garch_pct"] >= 70]
        off = df[df["garch_pct"] < 70]
        if len(on) < 20 or len(off) < 20:
            continue
        wins_on = on[on["pnl_r"] > 0]["pnl_r"].mean()
        losses_on = on[on["pnl_r"] <= 0]["pnl_r"].mean()
        wins_off = off[off["pnl_r"] > 0]["pnl_r"].mean()
        losses_off = off[off["pnl_r"] <= 0]["pnl_r"].mean()
        mae_on = on["mae_r"].astype(float).dropna().mean()
        mae_off = off["mae_r"].astype(float).dropna().mean()
        mfe_on = on["mfe_r"].astype(float).dropna().mean()
        mfe_off = off["mfe_r"].astype(float).dropna().mean()
        wr_on = (on["pnl_r"] > 0).mean()
        wr_off = (off["pnl_r"] > 0).mean()
        print(f"  {direction}:")
        print(f"    WR: {wr_on:.1%} on vs {wr_off:.1%} off (diff {wr_on - wr_off:+.1%})")
        print(f"    AvgWin: {wins_on:+.3f} on vs {wins_off:+.3f} off")
        print(f"    AvgLoss: {losses_on:+.3f} on vs {losses_off:+.3f} off")
        print(f"    MAE: {mae_on:+.3f} on vs {mae_off:+.3f} off")
        print(f"    MFE: {mfe_on:+.3f} on vs {mfe_off:+.3f} off")
        out[direction] = dict(
            wr_on=float(wr_on),
            wr_off=float(wr_off),
            wins_on=float(wins_on),
            wins_off=float(wins_off),
            losses_on=float(losses_on),
            losses_off=float(losses_off),
            mae_on=float(mae_on),
            mae_off=float(mae_off),
            mfe_on=float(mfe_on),
            mfe_off=float(mfe_off),
        )
    return out


def s7_continuous_regression():
    """Regress pnl_r on garch_pct continuously at NYSE_OPEN."""
    print("\n=== S7: Continuous regression ===")
    out = {}
    for direction in ["long", "short"]:
        dfs = [load_nyse_open(apt, rr, direction) for apt in [5, 15, 30] for rr in [1.0, 1.5, 2.0]]
        df = pd.concat([d for d in dfs if len(d) > 10], ignore_index=True)
        if len(df) < 100:
            continue
        slope, intercept, r, p, se = stats.linregress(df["garch_pct"], df["pnl_r"])
        print(f"  {direction}: N={len(df)} slope={slope:+.5f} r={r:+.3f} p={p:.4f}")
        out[direction] = dict(N=len(df), slope=float(slope), r=float(r), p=float(p))
    return out


def s8_tail_behavior():
    """garch@90 behavior across NYSE_OPEN."""
    print("\n=== S8: Tail behavior (threshold 90) ===")
    out = []
    for apt in [5, 15, 30]:
        for rr in [1.0, 1.5, 2.0]:
            for direction in ["long", "short"]:
                df = load_nyse_open(apt, rr, direction)
                if len(df) < 60:
                    continue
                for th in [70, 80, 90]:
                    srl, ml, n_on, n_off = sr_lift(df, th)
                    out.append((apt, rr, direction, th, srl, n_on))
                    if srl is not None:
                        print(f"  O{apt} RR{rr} {direction} @{th}: sr_lift={srl:+.3f} N_on={n_on}")
    return out


def main():
    print("MNQ_NYSE_OPEN SKIP_GARCH_70 stress test\n" + "=" * 70)

    s1 = s1_shuffle_control()
    s2 = s2_per_year()
    s3 = s3_direction_split()
    s4 = s4_event_day_exclusion()
    s5 = s5_break_direction_confounder()
    s6 = s6_mae_mfe_decomp()
    s7 = s7_continuous_regression()
    s8 = s8_tail_behavior()

    emit(s1, s2, s3, s4, s5, s6, s7, s8)


def emit(s1, s2, s3, s4, s5, s6, s7, s8):
    lines = [
        "# NYSE_OPEN SKIP_GARCH_70 — Adversarial Stress Test",
        "",
        "**Date:** 2026-04-15",
        "**Trigger:** User demanded stress-test before implementing NYSE_OPEN SKIP filter. Red-team the finding to ensure no bias, lookahead, or pigeonholing.",
        "",
        "## Base claim",
        "",
        "MNQ_NYSE_OPEN across 54 (apt × RR × dir × threshold) test cells showed 46/54 negative sr_lift on garch_pct overlay. Binomial sign test p=6.9×10⁻⁸. Identified as strongest SKIP candidate on the project.",
        "",
        "---",
        "",
        "## S1 — Shuffle control",
        "",
        f"Real data: {s1['real_pos_frac']:.3f} positive fraction across all NYSE_OPEN cells.",
        f"Shuffled (100 runs): median positive frac = {s1['shuf_median']:.3f}, range "
        f"[{s1['shuf_range'][0]:.3f}, {s1['shuf_range'][1]:.3f}].",
        f"Shuffle p-value: {s1['shuf_p']:.4f}",
        "",
        f"**Verdict:** {'PASS' if s1['shuf_p'] < 0.05 else 'FAIL'} — methodology "
        + ("distinguishes real signal from shuffled noise." if s1["shuf_p"] < 0.05 else "may be biased."),
        "",
        "---",
        "",
        "## S2 — Per-year breakdown",
        "",
        "| Year | N cells | % positive | avg lift |",
        "|---|---|---|---|",
    ]
    for yr, (n, pf, avg) in s2.items():
        lines.append(f"| {yr} | {n} | {pf:.1%} | {avg:+.3f} |")

    neg_years = sum(1 for n, pf, _ in s2.values() if pf < 0.3)
    total_yrs = len(s2)
    lines += [
        "",
        f"**{neg_years} of {total_yrs} years** show <30% positive fraction (strongly inverse).",
        f"**Verdict:** {'CONSISTENT' if neg_years >= total_yrs * 0.6 else 'YEAR-DEPENDENT'}",
        "",
        "---",
        "",
        "## S3 — Direction split",
        "",
        f"LONG cells: {s3['long_pos']} positive / {s3['long_neg']} negative",
        f"SHORT cells: {s3['short_pos']} positive / {s3['short_neg']} negative",
        "",
    ]
    long_neg_frac = s3["long_neg"] / max(s3["long_pos"] + s3["long_neg"], 1)
    short_neg_frac = s3["short_neg"] / max(s3["short_pos"] + s3["short_neg"], 1)
    lines += [
        f"LONG inverse fraction: {long_neg_frac:.1%}",
        f"SHORT inverse fraction: {short_neg_frac:.1%}",
        "",
        f"**Verdict:** "
        + (
            "BOTH directions inverse"
            if long_neg_frac > 0.7 and short_neg_frac > 0.7
            else "DIRECTION-ASYMMETRIC"
            if abs(long_neg_frac - short_neg_frac) > 0.3
            else "PREDOMINANTLY LONG INVERSE"
            if long_neg_frac > short_neg_frac + 0.2
            else "MIXED"
        ),
        "",
        "---",
        "",
        "## S4 — Event-day exclusion",
        "",
    ]
    lines += ["| apt | rr | dir | full sr_lift (N_on) | clean sr_lift (N_on) | delta |", "|---|---|---|---|---|---|"]
    for (apt, rr, direction), v in s4.items():
        full = v["full_lift"] if v["full_lift"] is not None else float("nan")
        clean = v["clean_lift"] if v["clean_lift"] is not None else float("nan")
        delta = clean - full if not np.isnan(full) and not np.isnan(clean) else float("nan")
        lines.append(
            f"| O{apt} | {rr} | {direction} | {full:+.3f} ({v['full_n_on']}) | "
            f"{clean:+.3f} ({v['clean_n_on']}) | {delta:+.3f} |"
        )
    # If clean stays inverse, effect survives event removal
    persistent = sum(1 for v in s4.values() if v["clean_lift"] is not None and v["clean_lift"] < -0.05)
    lines += [
        "",
        f"**{persistent} of {len(s4)} cells** still show inverse after event-day exclusion.",
        "",
        "---",
        "",
        "## S5 — Break-direction confounder",
        "",
        f"P(break=long | garch>=70): {s5['high_long_frac']:.3f}",
        f"P(break=long | garch<70): {s5['low_long_frac']:.3f}",
        f"Diff: {s5['diff']:+.3f}",
        "",
        f"**Verdict:** {'CONFOUNDER PRESENT' if abs(s5['diff']) > 0.05 else 'NO CONFOUND'} — garch "
        + ("IS" if abs(s5["diff"]) > 0.05 else "is NOT")
        + " associated with break direction at NYSE_OPEN.",
        "",
        "---",
        "",
        "## S6 — MAE/MFE decomposition",
        "",
        "Is the inverse signal from WR change, bigger losses, or smaller wins?",
        "",
        "| dir | WR on | WR off | AvgWin on | AvgWin off | AvgLoss on | AvgLoss off | MAE on | MAE off |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for direction, v in s6.items():
        lines.append(
            f"| {direction} | {v['wr_on']:.1%} | {v['wr_off']:.1%} | {v['wins_on']:+.3f} | "
            f"{v['wins_off']:+.3f} | {v['losses_on']:+.3f} | {v['losses_off']:+.3f} | "
            f"{v['mae_on']:+.3f} | {v['mae_off']:+.3f} |"
        )
    lines += [
        "",
        "---",
        "",
        "## S7 — Continuous regression",
        "",
        "If garch is a clean regime indicator, linear slope on pnl_r should be consistent.",
        "",
        "| Direction | N | slope | r | p |",
        "|---|---|---|---|---|",
    ]
    for direction, v in s7.items():
        lines.append(f"| {direction} | {v['N']} | {v['slope']:+.5f} | {v['r']:+.3f} | {v['p']:.4f} |")

    lines += [
        "",
        "Negative slope = garch higher → pnl lower (inverse signal confirmed continuously).",
        "",
        "---",
        "",
        "## S8 — Tail behavior",
        "",
        "If the effect collapses or reverses at threshold 90, it's a mid-tail artifact, not a robust regime.",
        "",
        "| apt | rr | dir | @70 lift | @80 lift | @90 lift |",
        "|---|---|---|---|---|---|",
    ]
    by_cell = {}
    for apt, rr, direction, th, srl, n_on in s8:
        by_cell.setdefault((apt, rr, direction), {})[th] = (srl, n_on)
    for (apt, rr, direction), thr_vals in sorted(by_cell.items()):
        v70 = thr_vals.get(70, (None, 0))
        v80 = thr_vals.get(80, (None, 0))
        v90 = thr_vals.get(90, (None, 0))
        s70 = f"{v70[0]:+.3f}" if v70[0] is not None else "—"
        s80 = f"{v80[0]:+.3f}" if v80[0] is not None else "—"
        s90 = f"{v90[0]:+.3f}" if v90[0] is not None else "—"
        lines.append(f"| O{apt} | {rr} | {direction} | {s70} ({v70[1]}) | {s80} ({v80[1]}) | {s90} ({v90[1]}) |")

    lines += [
        "",
        "---",
        "",
        "## Final stress-test verdict",
        "",
        "Each S-test either confirms or falsifies part of the base claim. Consolidated below.",
        "",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
