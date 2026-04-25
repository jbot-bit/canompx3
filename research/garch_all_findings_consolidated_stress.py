"""Consolidated stress test across ALL 32 (instrument, session) garch families.

Not just NYSE_OPEN — full garch narrative.

For each (inst, sess) family, applies the stress-test battery:
  A. Shuffle control — how many shuffle-runs produce directional pattern as
     extreme as observed?
  B. Per-year consistency — does the directional pattern hold across years?
  C. Long-vs-short split — is the effect direction-symmetric or asymmetric?
  D. Continuous regression — linear slope of pnl_r on garch_pct.

Classifies each family as:
  - SURVIVES_POSITIVE: real positive regime signal (deploy as sizer-candidate family)
  - SURVIVES_INVERSE: real inverse (skip-candidate family)
  - LONG_ONLY_INVERSE: inverse only on longs (NYSE_OPEN pattern)
  - SHORT_ONLY_X: directional asymmetry
  - NOISE: within shuffle envelope, no consistent structure

Output: docs/audit/results/2026-04-15-garch-consolidated-stress.md
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

OUTPUT_MD = Path("docs/audit/results/2026-04-15-garch-consolidated-stress.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
IS_END = "2026-01-01"
SEED = 20260415

SESSIONS = [
    "CME_REOPEN",
    "TOKYO_OPEN",
    "SINGAPORE_OPEN",
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
    "BRISBANE_1025",
]


def load_family(con, inst: str, sess: str, apt: int, rr: float, direction: str) -> pd.DataFrame:
    q = f"""
    SELECT o.trading_day, o.pnl_r, d.garch_forecast_vol_pct AS gp
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day=d.trading_day AND o.symbol=d.symbol AND o.orb_minutes=d.orb_minutes
    WHERE o.symbol='{inst}' AND o.orb_minutes={apt} AND o.orb_label='{sess}'
      AND o.entry_model='E2' AND o.rr_target={rr} AND o.pnl_r IS NOT NULL
      AND d.garch_forecast_vol_pct IS NOT NULL
      AND d.orb_{sess}_break_dir='{direction}'
      AND o.trading_day < DATE '{IS_END}'
    """
    try:
        df = con.execute(q).df()
    except Exception:
        return pd.DataFrame()
    if len(df) > 0:
        df["trading_day"] = pd.to_datetime(df["trading_day"])
        df["year"] = df["trading_day"].dt.year
        df["pnl_r"] = df["pnl_r"].astype(float)
        df["gp"] = df["gp"].astype(float)
    return df


def sr_lift(df: pd.DataFrame, thresh: int):
    on = df[df["gp"] >= thresh]
    off = df[df["gp"] < thresh]
    if len(on) < 10 or len(off) < 10:
        return None
    e_on, e_off = on["pnl_r"].mean(), off["pnl_r"].mean()
    s_on = on["pnl_r"].std(ddof=1)
    s_off = off["pnl_r"].std(ddof=1)
    sr_on = e_on / s_on if s_on > 0 else 0
    sr_off = e_off / s_off if s_off > 0 else 0
    return sr_on - sr_off


def stress_family(con, inst: str, sess: str, n_shuffles: int = 50) -> dict:
    """Per-family: shuffle control + direction split + per-year + regression."""
    cells = []
    dfs_by_dir = {"long": [], "short": []}

    rng = np.random.default_rng(SEED)

    for apt in [5, 15, 30]:
        for rr in [1.0, 1.5, 2.0]:
            for direction in ["long", "short"]:
                df = load_family(con, inst, sess, apt, rr, direction)
                if len(df) < 60:
                    continue
                dfs_by_dir[direction].append(df.copy())
                for thresh in [60, 70, 80]:
                    srl = sr_lift(df, thresh)
                    if srl is None:
                        continue
                    cells.append(
                        {
                            "apt": apt,
                            "rr": rr,
                            "dir": direction,
                            "thresh": thresh,
                            "sr_lift": srl,
                        }
                    )

    if not cells:
        return {"inst": inst, "sess": sess, "skip": True}

    n_cells = len(cells)
    n_pos = sum(1 for c in cells if c["sr_lift"] > 0)
    pos_frac = n_pos / n_cells

    # Long-vs-short split
    long_cells = [c for c in cells if c["dir"] == "long"]
    short_cells = [c for c in cells if c["dir"] == "short"]
    long_pos = sum(1 for c in long_cells if c["sr_lift"] > 0)
    short_pos = sum(1 for c in short_cells if c["sr_lift"] > 0)
    long_pos_frac = long_pos / max(len(long_cells), 1)
    short_pos_frac = short_pos / max(len(short_cells), 1)

    # Shuffle control — for each (apt, rr, dir), shuffle garch, rerun, count pos
    shuf_pos_fracs = []
    for si in range(n_shuffles):
        shuf_pos = 0
        shuf_total = 0
        for direction in ["long", "short"]:
            for df in dfs_by_dir[direction]:
                pnl = df["pnl_r"].to_numpy()
                gp = df["gp"].to_numpy().copy()
                rng.shuffle(gp)
                for thresh in [60, 70, 80]:
                    on_mask = gp >= thresh
                    on, off = pnl[on_mask], pnl[~on_mask]
                    if len(on) >= 10 and len(off) >= 10:
                        s_on = on.std(ddof=1)
                        s_off = off.std(ddof=1)
                        srl = (on.mean() / s_on if s_on > 0 else 0) - (off.mean() / s_off if s_off > 0 else 0)
                        shuf_total += 1
                        if srl > 0:
                            shuf_pos += 1
        if shuf_total > 0:
            shuf_pos_fracs.append(shuf_pos / shuf_total)

    shuf_med = float(np.median(shuf_pos_fracs)) if shuf_pos_fracs else 0.5
    shuf_lo = float(np.percentile(shuf_pos_fracs, 5)) if shuf_pos_fracs else 0.3
    shuf_hi = float(np.percentile(shuf_pos_fracs, 95)) if shuf_pos_fracs else 0.7

    # Shuffle envelope outside check
    if pos_frac > shuf_hi:
        envelope = "ABOVE_POS"
    elif pos_frac < shuf_lo:
        envelope = "BELOW_NEG"
    else:
        envelope = "WITHIN"

    # Continuous regression per direction
    regr = {}
    for direction in ["long", "short"]:
        if dfs_by_dir[direction]:
            combined = pd.concat(dfs_by_dir[direction], ignore_index=True)
            if len(combined) >= 50:
                slope, intercept, r, p, _ = stats.linregress(combined["gp"], combined["pnl_r"])
                regr[direction] = {"slope": float(slope), "p": float(p), "N": len(combined), "r": float(r)}
            else:
                regr[direction] = None
        else:
            regr[direction] = None

    # Per-year consistency (at threshold 70)
    yr_fracs = {}
    for direction in ["long", "short"]:
        for df in dfs_by_dir[direction]:
            for yr in df["year"].unique():
                sub = df[df["year"] == yr]
                on = sub[sub["gp"] >= 70]
                off = sub[sub["gp"] < 70]
                if len(on) >= 3 and len(off) >= 3:
                    s_on = on["pnl_r"].std(ddof=1)
                    s_off = off["pnl_r"].std(ddof=1)
                    yr_lift = (on["pnl_r"].mean() / s_on if s_on > 0 else 0) - (
                        off["pnl_r"].mean() / s_off if s_off > 0 else 0
                    )
                    yr_fracs.setdefault(int(yr), []).append(yr_lift > 0)
    yr_summary = {y: sum(vs) / len(vs) for y, vs in yr_fracs.items()}

    # Classification
    cls = classify(envelope, long_pos_frac, short_pos_frac, regr, yr_summary)

    return {
        "inst": inst,
        "sess": sess,
        "n_cells": n_cells,
        "n_pos": n_pos,
        "pos_frac": pos_frac,
        "long_pos": long_pos,
        "long_total": len(long_cells),
        "long_pos_frac": long_pos_frac,
        "short_pos": short_pos,
        "short_total": len(short_cells),
        "short_pos_frac": short_pos_frac,
        "shuf_median": shuf_med,
        "shuf_lo": shuf_lo,
        "shuf_hi": shuf_hi,
        "envelope": envelope,
        "regr": regr,
        "yr_summary": yr_summary,
        "classification": cls,
        "skip": False,
    }


def classify(envelope, long_f, short_f, regr, yr_summary) -> str:
    # Stress-test classification
    yr_pos_fracs = list(yr_summary.values())
    yr_consistency = np.mean([f > 0.5 for f in yr_pos_fracs]) if yr_pos_fracs else 0.5

    if envelope == "ABOVE_POS" and long_f >= 0.7 and short_f >= 0.7 and yr_consistency >= 0.6:
        return "SURVIVES_POSITIVE_BOTH_DIRS"
    if envelope == "ABOVE_POS" and long_f >= 0.7 and short_f < 0.7:
        return "SURVIVES_POSITIVE_LONG_ONLY"
    if envelope == "ABOVE_POS" and short_f >= 0.7 and long_f < 0.7:
        return "SURVIVES_POSITIVE_SHORT_ONLY"
    if envelope == "ABOVE_POS":
        return "POSITIVE_INCONSISTENT"
    if envelope == "BELOW_NEG" and long_f <= 0.3 and short_f <= 0.3 and yr_consistency <= 0.4:
        return "SURVIVES_INVERSE_BOTH_DIRS"
    if envelope == "BELOW_NEG" and long_f <= 0.3 and short_f > 0.3:
        return "SURVIVES_INVERSE_LONG_ONLY"
    if envelope == "BELOW_NEG" and short_f <= 0.3 and long_f > 0.3:
        return "SURVIVES_INVERSE_SHORT_ONLY"
    if envelope == "BELOW_NEG":
        return "INVERSE_INCONSISTENT"
    return "NOISE_WITHIN_ENVELOPE"


def main():
    print("Consolidated stress test — all 32 garch families")
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    results = []
    for inst in ["MNQ", "MES", "MGC"]:
        for sess in SESSIONS:
            print(f"  {inst} {sess}...", end=" ", flush=True)
            r = stress_family(con, inst, sess)
            if r.get("skip"):
                print("SKIP")
                continue
            print(f"pos={r['pos_frac']:.2f} env={r['envelope']} cls={r['classification']}")
            results.append(r)
    con.close()

    print(f"\n=== CLASSIFICATION SUMMARY ===")
    cls_counts = {}
    for r in results:
        cls_counts[r["classification"]] = cls_counts.get(r["classification"], 0) + 1
    for cls, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n}")

    emit(results, cls_counts)


def emit(results, cls_counts):
    lines = [
        "# Garch Findings — Consolidated Stress Test (All 32 Families)",
        "",
        "**Date:** 2026-04-15",
        "**Trigger:** User demanded stress test on ALL garch findings, not just NYSE_OPEN. Answers: which families survive the stress battery vs which are within shuffle-envelope noise.",
        "",
        "**Tests per family:**",
        "- A. Shuffle control (50 shuffles) — is observed positive fraction outside shuffle envelope?",
        "- B. Per-year consistency — does the directional pattern hold year over year?",
        "- C. Long-vs-short split — direction-symmetric or asymmetric?",
        "- D. Continuous regression — linear slope of pnl_r on garch_pct.",
        "",
        "**Classifications:**",
        "- SURVIVES_POSITIVE_BOTH_DIRS: clean positive signal both directions (deploy sizer candidate)",
        "- SURVIVES_POSITIVE_LONG/SHORT_ONLY: positive only one direction",
        "- POSITIVE_INCONSISTENT: outside envelope but year/direction mixed",
        "- SURVIVES_INVERSE_BOTH_DIRS: clean inverse both directions (deploy SKIP)",
        "- SURVIVES_INVERSE_LONG/SHORT_ONLY: inverse only one direction",
        "- INVERSE_INCONSISTENT: outside envelope but mixed",
        "- NOISE_WITHIN_ENVELOPE: indistinguishable from shuffle noise",
        "",
        "---",
        "",
        "## Classification summary",
        "",
        "| Classification | Count |",
        "|---|---|",
    ]
    for cls, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cls} | {n} |")

    lines += [
        "",
        "---",
        "",
        "## Per-family full stress-test grid",
        "",
        "| Inst | Session | Cells | Pos% | Long Pos% | Short Pos% | Shuf envelope | Envelope | Regr long p | Regr short p | Yr consistency | Class |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda x: (x["classification"], x["inst"], x["sess"])):
        regr_l = r["regr"].get("long")
        regr_s = r["regr"].get("short")
        regr_l_p = f"{regr_l['p']:.3f}" if regr_l else "n/a"
        regr_s_p = f"{regr_s['p']:.3f}" if regr_s else "n/a"
        yr_cons = f"{sum(1 for v in r['yr_summary'].values() if v > 0.5)}/{len(r['yr_summary'])}"
        lines.append(
            f"| {r['inst']} | {r['sess']} | {r['n_cells']} | {r['pos_frac']:.1%} | "
            f"{r['long_pos_frac']:.1%} | {r['short_pos_frac']:.1%} | "
            f"[{r['shuf_lo']:.2f}, {r['shuf_hi']:.2f}] | {r['envelope']} | "
            f"{regr_l_p} | {regr_s_p} | {yr_cons} | {r['classification']} |"
        )

    lines += ["", "---", "", "## Deployable findings (only classifications that survive)", ""]
    deployable = [r for r in results if r["classification"].startswith("SURVIVES")]
    if not deployable:
        lines.append("_No family survives the full battery unambiguously._")
    else:
        lines += ["| Inst | Session | Class | Long Pos% | Short Pos% | Action |", "|---|---|---|---|---|---|"]
        for r in deployable:
            action = ""
            if "POSITIVE_BOTH_DIRS" in r["classification"]:
                action = "SIZER family — pre-reg both directions"
            elif "POSITIVE_LONG" in r["classification"]:
                action = "SIZER long-only"
            elif "POSITIVE_SHORT" in r["classification"]:
                action = "SIZER short-only"
            elif "INVERSE_BOTH_DIRS" in r["classification"]:
                action = "SKIP family — both directions"
            elif "INVERSE_LONG" in r["classification"]:
                action = "SKIP long-only"
            elif "INVERSE_SHORT" in r["classification"]:
                action = "SKIP short-only"
            lines.append(
                f"| {r['inst']} | {r['sess']} | {r['classification']} | "
                f"{r['long_pos_frac']:.1%} | {r['short_pos_frac']:.1%} | {action} |"
            )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")


if __name__ == "__main__":
    main()
