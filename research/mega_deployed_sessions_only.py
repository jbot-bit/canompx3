"""Scoped mega-exploration — deployed MNQ lanes only.

Narrows the 2026-04-15 mega (K=6252, cells on all 12 sessions × 3 instruments
× all RRs × both directions × all apertures × 14 signals) down to the sessions
and parameters actually traded live.

Deployed lanes per docs/runtime/lane_allocation.json (2026-04-13):
  MNQ EUROPE_FLOW RR1.5 ORB_G5 (E2 O5)
  MNQ SINGAPORE_OPEN RR1.5 ATR_P50 (E2 O30)
  MNQ COMEX_SETTLE RR1.5 OVNRNG_100 (E2 O5)
  MNQ NYSE_OPEN RR1.0 ORB_G5 (E2 O5)
  MNQ TOKYO_OPEN RR1.5 ORB_G5 (E2 O5)
  MNQ US_DATA_1000 RR1.5 VWAP_MID_ALIGNED (E2 O5)

Scope:
- 6 sessions × MNQ only × 14 signals × 2 directions × 1 aperture (O5 mostly,
  SINGAPORE_OPEN also O30)
- RR: both deployed (1.0/1.5) tested to allow overlay at multiple RRs
- Aperture: O5 for 5 lanes, O30 for SINGAPORE_OPEN (the deployed aperture)
- K estimate: 6 × 1 × 14 × 2 × ~2 × 2 = ~672 cells (far below mega's 6252)

Output:
  docs/audit/results/2026-04-15-mega-deployed-sessions-only.md

Purpose: find level-based SKIP/TAKE filters that would apply AS OVERLAY on top
of existing deployed lanes (execution-time skip on days when level signal fires).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM  # noqa: E402

SEED = 20260415

# Deployed config per lane_allocation.json
DEPLOYED_LANES = [
    # (session, rr, aperture)
    ("EUROPE_FLOW", 1.5, 5),
    ("SINGAPORE_OPEN", 1.5, 30),
    ("COMEX_SETTLE", 1.5, 5),
    ("NYSE_OPEN", 1.0, 5),
    ("TOKYO_OPEN", 1.5, 5),
    ("US_DATA_1000", 1.5, 5),
]

# Also test at RR 1.0, 2.0 variants for each (overlay could suggest new RR)
RR_VARIANTS = [1.0, 1.5, 2.0]
DIRECTIONS = ("long", "short")
THETAS = (0.15, 0.30, 0.50)

OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()

OUTPUT_MD = Path("docs/audit/results/2026-04-15-mega-deployed-sessions-only.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_deployed_sessions() -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    parts = []
    # For each deployed-lane session+aperture combo, pull data
    unique_sessions_apt = {(s, a) for s, _, a in DEPLOYED_LANES}
    for session, aperture in unique_sessions_apt:
        parts.append(
            f"""
        SELECT
            o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
            o.entry_model, o.rr_target, o.outcome, o.pnl_r,
            d.atr_20, d.prev_day_high, d.prev_day_low, d.prev_day_close,
            d.gap_type, d.gap_open_points,
            (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
            d.orb_{session}_break_dir AS break_dir
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
          AND o.symbol = d.symbol
          AND o.orb_minutes = d.orb_minutes
        WHERE o.orb_label = '{session}'
          AND o.symbol = 'MNQ'
          AND o.orb_minutes = {aperture}
          AND o.entry_model = 'E2'
          AND o.rr_target IN (1.0, 1.5, 2.0)
          AND o.pnl_r IS NOT NULL
          AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
          AND d.prev_day_high IS NOT NULL
          AND d.prev_day_low IS NOT NULL
          AND d.prev_day_close IS NOT NULL
          AND d.orb_{session}_break_dir IN ('long','short')
        """
        )
    query = " UNION ALL ".join(parts)
    df = con.execute(query).df()
    con.close()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = df["atr_20"].astype(float)
    mid = df["orb_mid"].astype(float)
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    pdc = df["prev_day_close"].astype(float)
    pivot = (pdh + pdl + pdc) / 3.0

    for theta in THETAS:
        tag = f"{int(theta * 100):02d}"
        df[f"F1_NEAR_PDH_{tag}"] = (np.abs(mid - pdh) / atr < theta).astype(int)
        df[f"F2_NEAR_PDL_{tag}"] = (np.abs(mid - pdl) / atr < theta).astype(int)
        df[f"F3_NEAR_PIVOT_{tag}"] = (np.abs(mid - pivot) / atr < theta).astype(int)

    df["F4_ABOVE_PDH"] = (mid > pdh).astype(int)
    df["F5_BELOW_PDL"] = (mid < pdl).astype(int)
    df["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).astype(int)
    df["F7_GAP_UP"] = (df["gap_type"] == "gap_up").astype(int)
    df["F8_GAP_DOWN"] = (df["gap_type"] == "gap_down").astype(int)
    return df


FEATURES = (
    [f"F1_NEAR_PDH_{int(t * 100):02d}" for t in THETAS]
    + [f"F2_NEAR_PDL_{int(t * 100):02d}" for t in THETAS]
    + [f"F3_NEAR_PIVOT_{int(t * 100):02d}" for t in THETAS]
    + ["F4_ABOVE_PDH", "F5_BELOW_PDL", "F6_INSIDE_PDR", "F7_GAP_UP", "F8_GAP_DOWN"]
)


def welch_t(is_df: pd.DataFrame, signal: str) -> tuple[float, float, int, int, float, float]:
    """Return (t, p, n_on, n_off, expr_on, expr_off) using Welch's t-test."""
    on = is_df[is_df[signal] == 1]["pnl_r"]
    off = is_df[is_df[signal] == 0]["pnl_r"]
    if len(on) < 30 or len(off) < 30:
        return (float("nan"), float("nan"), len(on), len(off), float("nan"), float("nan"))
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return (float(t), float(p), len(on), len(off), float(on.mean()), float(off.mean()))


def scan_session_rr_direction(
    df_all: pd.DataFrame,
    session: str,
    aperture: int,
    rr: float,
    direction: str,
) -> list[dict]:
    sub = df_all[
        (df_all["orb_label"] == session)
        & (df_all["orb_minutes"] == aperture)
        & (df_all["rr_target"] == rr)
        & (df_all["break_dir"] == direction)
    ]
    if len(sub) == 0:
        return []
    is_df = sub[sub["is_is"]]
    oos_df = sub[sub["is_oos"]]
    rows = []
    for signal in FEATURES:
        t_is, p_is, n_on_is, n_off_is, expr_on_is, expr_off_is = welch_t(is_df, signal)
        if np.isnan(t_is) or n_on_is < 30:
            continue
        t_oos, p_oos, n_on_oos, n_off_oos, expr_on_oos, expr_off_oos = welch_t(oos_df, signal)
        delta_is = expr_on_is - expr_off_is
        delta_oos = expr_on_oos - expr_off_oos if not np.isnan(expr_on_oos) else float("nan")
        dir_match = not np.isnan(delta_oos) and (np.sign(delta_is) == np.sign(delta_oos))
        fire_rate = n_on_is / (n_on_is + n_off_is)
        rows.append(
            {
                "session": session,
                "aperture": aperture,
                "rr": rr,
                "direction": direction,
                "signal": signal,
                "n_on_is": n_on_is,
                "n_off_is": n_off_is,
                "n_on_oos": n_on_oos,
                "expr_on_is": expr_on_is,
                "expr_off_is": expr_off_is,
                "delta_is": delta_is,
                "expr_on_oos": expr_on_oos,
                "delta_oos": delta_oos,
                "t_is": t_is,
                "p_is": p_is,
                "dir_match": dir_match,
                "fire_rate": fire_rate,
                "deployed_match": (session, rr, aperture) in {(s, r, a) for s, r, a in DEPLOYED_LANES},
            }
        )
    return rows


def main():
    print("[mega-deployed] loading data...")
    df = load_deployed_sessions()
    print(f"  rows loaded: {len(df):,}")
    df = compute_features(df)
    print(f"  features computed: {len(FEATURES)}")

    all_rows = []
    unique_sessions_apt = {(s, a) for s, _, a in DEPLOYED_LANES}
    for session, aperture in unique_sessions_apt:
        for rr in RR_VARIANTS:
            for direction in DIRECTIONS:
                rows = scan_session_rr_direction(df, session, aperture, rr, direction)
                all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    print(f"  total cells scanned: {len(res)}")

    # BH FDR within deployed sessions
    res_sorted = res.sort_values("p_is").reset_index(drop=True)
    K = len(res_sorted)
    res_sorted["bh_rank"] = res_sorted.index + 1
    res_sorted["bh_crit"] = 0.05 * res_sorted["bh_rank"] / K
    res_sorted["bh_pass"] = res_sorted["p_is"] <= res_sorted["bh_crit"]

    # Strict filter: |t|>=3, dir_match, deployed_match
    survivors_strict = res_sorted[
        (res_sorted["t_is"].abs() >= 3.0) & (res_sorted["dir_match"]) & (res_sorted["deployed_match"])
    ].copy()
    survivors_bh = res_sorted[res_sorted["bh_pass"] & res_sorted["deployed_match"]].copy()

    # Emit report
    lines = [
        "# Mega — Deployed Sessions Only",
        "",
        "**Date:** 2026-04-15",
        "**Scope:** MNQ only, 6 deployed-lane (session, aperture) pairs, RR {1.0,1.5,2.0}, "
        f"both directions, all {len(FEATURES)} level features",
        f"**Total cells scanned:** {K}",
        f"**Rows loaded:** {len(df):,}",
        "",
        "## Deployed-Match Survivors (|t| >= 3.0 AND dir_match)",
        "",
        "These are candidate OVERLAY filters for the exact (session, aperture, RR) of a live lane.",
        "",
        "| Session | Apt | RR | Dir | Signal | N_on | Fire% | ExpR_on | ExpR_off | Δ_IS | Δ_OOS | t | p | BH_pass |",
        "|---------|-----|----|----|--------|------|-------|---------|----------|------|-------|---|---|---------|",
    ]
    for _, r in survivors_strict.iterrows():
        bh = "Y" if bool(r["bh_pass"]) else "."
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['signal']} | {r['n_on_is']} | {r['fire_rate']:.1%} | "
            f"{r['expr_on_is']:+.3f} | {r['expr_off_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.4f} | {bh} |"
        )

    lines += [
        "",
        "## BH-FDR Survivors at Deployed-Match (q=0.05)",
        "",
        "| Session | Apt | RR | Dir | Signal | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p | BH_crit |",
        "|---------|-----|----|----|--------|------|-------|---------|------|-------|---|---|---------|",
    ]
    for _, r in survivors_bh.iterrows():
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['signal']} | {r['n_on_is']} | {r['fire_rate']:.1%} | "
            f"{r['expr_on_is']:+.3f} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.5f} | {r['bh_crit']:.5f} |"
        )

    lines += [
        "",
        "## All Cells (ranked by |t|)",
        "",
        "| Session | Apt | RR | Dir | Signal | N_on | ExpR_on | Δ_IS | Δ_OOS | t | p | deployed | dir_match |",
        "|---------|-----|----|----|--------|------|---------|------|-------|---|---|----------|-----------|",
    ]
    res_ranked = res.copy()
    res_ranked["abs_t"] = res_ranked["t_is"].abs()
    res_ranked = res_ranked.sort_values("abs_t", ascending=False).head(50)
    for _, r in res_ranked.iterrows():
        dep = "Y" if r["deployed_match"] else "."
        dm = "Y" if r["dir_match"] else "."
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['signal']} | {r['n_on_is']} | {r['expr_on_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.4f} | {dep} | {dm} |"
        )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")
    print(f"  Strict survivors (|t|>=3 + dir_match + deployed): {len(survivors_strict)}")
    print(f"  BH-FDR survivors (deployed-match): {len(survivors_bh)}")


if __name__ == "__main__":
    main()
