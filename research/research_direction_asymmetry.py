"""
SIDEWAYS FINDING HUNT: Break Direction Asymmetry
=================================================
Primary hypothesis we were NOT testing: does LONG vs SHORT ORB break direction
predict outcome quality? We know DIR_LONG is in the grid for 1000. But has anyone
looked at SHORT being catastrophically bad at OTHER sessions?

The compressed spring finding came from scanning sideways. This does the same:
scan LONG vs SHORT outcomes across all instruments × sessions, looking for
subgroups that are systematically bad in EITHER direction.

Methodology:
- Uses E1/CB1/RR2.5 (broad filter, maximize N per group)
- G4+ size filter only (no small ORB noise)
- BH FDR correction across all tests
- Reports both catastrophically BAD (avoid signal) and unexpectedly GOOD subgroups

Run:
    python research/research_direction_asymmetry.py

Output:
    research/output/direction_asymmetry_results.csv
    research/output/direction_asymmetry_summary.txt
"""

import sys
from pathlib import Path
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_enabled_sessions

# Config — broad net to catch anything
ENTRY_MODEL = "E1"
CONFIRM_BARS = 1
RR_TARGET = 2.5
ORB_MINUTES = 5
MIN_ORB_G = 4.0  # G4+ only

MIN_N_VALID = 30  # RESEARCH_RULES.md: <30 = INVALID

OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENTS = ["MGC", "MES", "MNQ"]


def query_direction_outcomes(con, instrument: str, session: str) -> pd.DataFrame:
    """Fetch E1/CB1/RR2.5 G4+ outcomes with break direction for one (instrument, session)."""
    dir_col = f"orb_{session}_break_dir"
    size_col = f"orb_{session}_size"

    query = f"""
        SELECT
            o.pnl_r,
            o.outcome,
            d.{dir_col}    AS break_dir,
            d.{size_col}   AS orb_size,
            d.us_dst,
            o.trading_day
        FROM orb_outcomes o
        JOIN daily_features d
            ON  o.trading_day  = d.trading_day
            AND o.symbol       = d.symbol
            AND o.orb_minutes  = d.orb_minutes
        WHERE
            o.symbol        = '{instrument}'
            AND o.orb_label    = '{session}'
            AND o.orb_minutes  = {ORB_MINUTES}
            AND o.entry_model  = '{ENTRY_MODEL}'
            AND o.confirm_bars = {CONFIRM_BARS}
            AND o.rr_target    = {RR_TARGET}
            AND o.pnl_r IS NOT NULL
            AND d.{size_col} >= {MIN_ORB_G}
            AND d.{dir_col} IS NOT NULL
    """
    try:
        return con.execute(query).df()
    except Exception as e:
        print(f"  SKIP {instrument}/{session}: {e}")
        return pd.DataFrame()


def t_test_vs_zero(values: np.ndarray) -> tuple[float, float]:
    """One-sample t-test: is mean significantly different from 0?"""
    if len(values) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(values, 0.0)
    return float(t), float(p)


def bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    p_bh = [float("nan")] * n
    prev = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(pairs), 1):
        adjusted = p * n / (n - rank + 1)
        adjusted = min(adjusted, prev)
        p_bh[orig_idx] = adjusted
        prev = adjusted
    return p_bh


def classify(avg_r: float, p: float, n: int) -> str:
    if n < MIN_N_VALID:
        return "INVALID"
    if p < 0.005:
        strength = "STRONG"
    elif p < 0.01:
        strength = "SOLID"
    elif p < 0.05:
        strength = "WEAK"
    else:
        return "NOISE"
    direction = "POSITIVE" if avg_r > 0 else "NEGATIVE"
    return f"{strength}_{direction}"


def main():
    print(f"Direction Asymmetry Hunt")
    print(f"Config: {ENTRY_MODEL}/CB{CONFIRM_BARS}/RR{RR_TARGET}/G{MIN_ORB_G}+")
    print(f"DB: {GOLD_DB_PATH}")
    print()

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    rows = []

    for instrument in INSTRUMENTS:
        sessions = get_enabled_sessions(instrument)
        print(f"=== {instrument} ({len(sessions)} sessions) ===")

        for session in sessions:
            df = query_direction_outcomes(con, instrument, session)
            if df.empty:
                continue

            # Overall baseline (for comparison)
            overall_n = len(df)
            overall_avg = df["pnl_r"].mean()

            # Split by direction
            for direction in ("long", "short"):
                sub = df[df["break_dir"] == direction]["pnl_r"].values
                n = len(sub)
                if n < MIN_N_VALID:
                    print(f"  {session} {direction}: N={n} (INVALID, <{MIN_N_VALID})")
                    continue

                avg_r = float(np.mean(sub))
                wr = float(np.mean(sub > 0))
                t, p = t_test_vs_zero(sub)
                delta_vs_overall = avg_r - overall_avg

                rows.append({
                    "instrument": instrument,
                    "session": session,
                    "direction": direction,
                    "n": n,
                    "n_overall": overall_n,
                    "avg_r": round(avg_r, 4),
                    "wr": round(wr, 4),
                    "t_stat": round(t, 4),
                    "p_raw": round(p, 6),
                    "delta_vs_overall": round(delta_vs_overall, 4),
                    "overall_avg_r": round(overall_avg, 4),
                })
                label = "**" if abs(avg_r) > 0.20 and n >= 50 else "  "
                print(f"  {label} {session} {direction:5s}: N={n:4d} avgR={avg_r:+.3f} WR={wr:.1%} p={p:.4f} d={delta_vs_overall:+.3f}")

        print()

    con.close()

    if not rows:
        print("No results.")
        return

    results = pd.DataFrame(rows)

    # BH correction across all tests
    raw_ps = results["p_raw"].tolist()
    p_bh = bh_correct(raw_ps)
    results["p_bh"] = [round(x, 6) for x in p_bh]
    results["bh_sig"] = results["p_bh"] < 0.10
    results["classification"] = results.apply(
        lambda r: classify(r["avg_r"], r["p_bh"], r["n"]), axis=1
    )

    # Save CSV
    results_path = OUTPUT_DIR / "direction_asymmetry_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    # Summary report
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("DIRECTION ASYMMETRY: BH-SIGNIFICANT FINDINGS (q < 0.10)")
    summary_lines.append(f"Config: {ENTRY_MODEL}/CB{CONFIRM_BARS}/RR{RR_TARGET}/G{MIN_ORB_G}+")
    summary_lines.append("=" * 70)

    sig = results[results["bh_sig"]].sort_values("avg_r")
    if sig.empty:
        summary_lines.append("NO BH-significant findings at q=0.10")
    else:
        for _, row in sig.iterrows():
            marker = "AVOID" if row["avg_r"] < 0 else "BOOST"
            summary_lines.append(
                f"{marker}  {row['instrument']} {row['session']} {row['direction']:5s} | "
                f"N={row['n']:4d} | avgR={row['avg_r']:+.3f} | WR={row['wr']:.1%} | "
                f"p_bh={row['p_bh']:.4f} | Δ={row['delta_vs_overall']:+.3f}"
            )

    summary_lines.append("")
    summary_lines.append("-" * 70)
    summary_lines.append("ALL SUBGROUPS (sorted by avgR, N >= 30)")
    summary_lines.append("-" * 70)
    for _, row in results.sort_values("avg_r").iterrows():
        bh_tag = "** BH-SIG **" if row["bh_sig"] else ""
        summary_lines.append(
            f"  {row['instrument']:3s} {row['session']:15s} {row['direction']:5s} | "
            f"N={row['n']:4d} | avgR={row['avg_r']:+.3f} | WR={row['wr']:.1%} | "
            f"p={row['p_raw']:.4f} | p_bh={row['p_bh']:.4f} {bh_tag}"
        )

    summary_lines.append("")
    summary_lines.append("-" * 70)
    summary_lines.append("WORST SUBGROUPS (avgR < -0.20, sorted by avgR)")
    summary_lines.append("-" * 70)
    worst = results[results["avg_r"] < -0.20].sort_values("avg_r")
    if worst.empty:
        summary_lines.append("  None found.")
    else:
        for _, row in worst.iterrows():
            bh_tag = "BH-SIG" if row["bh_sig"] else "not-sig"
            summary_lines.append(
                f"  {row['instrument']} {row['session']} {row['direction']:5s}: "
                f"avgR={row['avg_r']:+.3f} WR={row['wr']:.1%} N={row['n']} ({bh_tag})"
            )

    summary_lines.append("")
    summary_lines.append("-" * 70)
    summary_lines.append("ASYMMETRY (LONG - SHORT avgR, sessions with both >= 30 trades)")
    summary_lines.append("-" * 70)
    for (inst, sess), grp in results.groupby(["instrument", "session"]):
        if len(grp) < 2:
            continue
        long_row = grp[grp["direction"] == "long"]
        short_row = grp[grp["direction"] == "short"]
        if long_row.empty or short_row.empty:
            continue
        long_r = long_row.iloc[0]["avg_r"]
        short_r = short_row.iloc[0]["avg_r"]
        asymmetry = long_r - short_r
        marker = " **" if abs(asymmetry) > 0.30 else ""
        summary_lines.append(
            f"  {inst:3s} {sess:15s}: LONG={long_r:+.3f} SHORT={short_r:+.3f} "
            f"asymmetry={asymmetry:+.3f}{marker}"
        )

    summary_text = "\n".join(summary_lines)
    print()
    print(summary_text)

    summary_path = OUTPUT_DIR / "direction_asymmetry_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
