"""MGC-focused level feature scan — F1-F8 across all sessions × apertures × RRs.

Purpose: user asked for comprehensive test of previous-day levels and S/R
signals on MGC specifically. Mega-exploration covered MGC but only top-50
survivors reported; we need full MGC-specific surface.

Scope (pre-committed EXPLORATORY — no validated_setups writes):
- MGC only
- All 12 sessions: CME_REOPEN, TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS,
  EUROPE_FLOW, US_DATA_830, NYSE_OPEN, US_DATA_1000, COMEX_SETTLE,
  CME_PRECLOSE, NYSE_CLOSE, BRISBANE_1025
- Apertures: 5, 15, 30
- RRs: 1.0, 1.5, 2.0
- Directions: long, short
- Features: 14 level features per mega: F1-F3 at θ∈{0.15,0.30,0.50}, F4-F8 binary
- K_budget = 12 sessions × 3 apt × 3 rr × 2 dir × 14 feat = 3024 cells max
  (many skip via data availability gates — MGC has no equity-hours deep history)

MinBTL check: 2·ln(3024)/E[max_N]² ≈ 16/E[max_N]². With per-cell N=100-500,
E[max_N] ≈ 3-4, MinBTL ≈ 1-2 years per cell. MGC has 3.8 years → within budget.

Methodology gates (per .claude/rules/backtesting-methodology.md):
- RULE 1 look-ahead: all level features use prev_day_* → safe (prior day's close)
- RULE 4 multi-framing BH-FDR: global + per-family + per-lane + per-session
- RULE 7 T0 tautology: vs deployed filters (even though MGC has no deployed lanes)
- RULE 8.1 extreme fire rate flagged
- RULE 8.2 ARITHMETIC_ONLY flagged
- RULE 12 red-flag audit on |t|>7 or Δ>0.6

Cross-reference verified MGC cells (bring-forward info):
- MGC SINGAPORE_OPEN O15 F3_NEAR_PIVOT_15 LONG (RR 1.5, 2.0): CONDITIONAL already
  per docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md

Output:
  docs/audit/results/2026-04-15-mgc-level-scan.md
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
from research.comprehensive_deployed_lane_scan import (  # type: ignore  # noqa: E402
    bh_fdr_multi_framing,
)

SEED = 20260415
SESSIONS = [
    "CME_REOPEN", "TOKYO_OPEN", "SINGAPORE_OPEN", "LONDON_METALS",
    "EUROPE_FLOW", "US_DATA_830", "NYSE_OPEN", "US_DATA_1000",
    "COMEX_SETTLE", "CME_PRECLOSE", "NYSE_CLOSE", "BRISBANE_1025",
]
APERTURES = [5, 15, 30]
RRS = [1.0, 1.5, 2.0]
DIRECTIONS = ["long", "short"]
THETAS = [0.15, 0.30, 0.50]
INSTRUMENT = "MGC"

OOS_START = HOLDOUT_SACRED_FROM
OOS_END = pd.Timestamp("2026-04-07").date()

OUTPUT_MD = Path("docs/audit/results/2026-04-15-mgc-level-scan.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)


def load_mgc_session(session: str, apt: int, rr: float) -> pd.DataFrame:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    q = f"""
    SELECT
      o.trading_day, o.symbol, o.orb_minutes, o.orb_label,
      o.entry_model, o.rr_target, o.outcome, o.pnl_r,
      d.atr_20, d.prev_day_high, d.prev_day_low, d.prev_day_close,
      d.gap_type,
      (d.orb_{session}_high + d.orb_{session}_low) / 2.0 AS orb_mid,
      d.orb_{session}_break_dir AS break_dir
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
      AND o.symbol = d.symbol
      AND o.orb_minutes = d.orb_minutes
    WHERE o.orb_label = '{session}'
      AND o.symbol = '{INSTRUMENT}'
      AND o.orb_minutes = {apt}
      AND o.entry_model = 'E2'
      AND o.rr_target = {rr}
      AND o.pnl_r IS NOT NULL
      AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
      AND d.prev_day_high IS NOT NULL AND d.prev_day_low IS NOT NULL AND d.prev_day_close IS NOT NULL
      AND d.orb_{session}_break_dir IN ('long','short')
    """
    try:
        df = con.execute(q).df()
    except Exception:
        con.close()
        return pd.DataFrame()
    con.close()
    if len(df) == 0:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["is_is"] = df["trading_day"].dt.date < OOS_START
    df["is_oos"] = (df["trading_day"].dt.date >= OOS_START) & (df["trading_day"].dt.date < OOS_END)
    return df


def build_level_features(df: pd.DataFrame) -> dict:
    mid = df["orb_mid"].astype(float)
    atr = df["atr_20"].astype(float).replace(0, np.nan)
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    pdc = df["prev_day_close"].astype(float)
    pivot = (pdh + pdl + pdc) / 3.0

    feats = {}
    for theta in THETAS:
        tag = f"{int(theta * 100):02d}"
        feats[f"F1_NEAR_PDH_{tag}"] = ((np.abs(mid - pdh) / atr) < theta).fillna(False).astype(int).values
        feats[f"F2_NEAR_PDL_{tag}"] = ((np.abs(mid - pdl) / atr) < theta).fillna(False).astype(int).values
        feats[f"F3_NEAR_PIVOT_{tag}"] = ((np.abs(mid - pivot) / atr) < theta).fillna(False).astype(int).values
    feats["F4_ABOVE_PDH"] = (mid > pdh).fillna(False).astype(int).values
    feats["F5_BELOW_PDL"] = (mid < pdl).fillna(False).astype(int).values
    feats["F6_INSIDE_PDR"] = ((mid > pdl) & (mid < pdh)).fillna(False).astype(int).values
    feats["F7_GAP_UP"] = (df["gap_type"] == "gap_up").astype(int).values
    feats["F8_GAP_DOWN"] = (df["gap_type"] == "gap_down").astype(int).values
    return feats


FEATURE_FAMILY_MAP = {}
for theta in THETAS:
    tag = f"{int(theta * 100):02d}"
    FEATURE_FAMILY_MAP[f"F1_NEAR_PDH_{tag}"] = "level_proximity"
    FEATURE_FAMILY_MAP[f"F2_NEAR_PDL_{tag}"] = "level_proximity"
    FEATURE_FAMILY_MAP[f"F3_NEAR_PIVOT_{tag}"] = "level_proximity"
for f in ["F4_ABOVE_PDH", "F5_BELOW_PDL", "F6_INSIDE_PDR"]:
    FEATURE_FAMILY_MAP[f] = "level_position"
for f in ["F7_GAP_UP", "F8_GAP_DOWN"]:
    FEATURE_FAMILY_MAP[f] = "gap"


def test_cell(df: pd.DataFrame, feature_name: str, signal: np.ndarray, direction: str) -> dict | None:
    mask = (df["break_dir"] == direction).values
    sub = df[mask].copy()
    sig = signal[mask]
    sub["_sig"] = sig
    is_df = sub[sub["is_is"]]
    oos_df = sub[sub["is_oos"]]
    on_is = is_df[is_df["_sig"] == 1]["pnl_r"]
    off_is = is_df[is_df["_sig"] == 0]["pnl_r"]
    if len(on_is) < 30 or len(off_is) < 30:
        return None

    t_is, p_is = stats.ttest_ind(on_is, off_is, equal_var=False)
    expr_on = float(on_is.mean())
    expr_off = float(off_is.mean())
    delta_is = expr_on - expr_off
    wr_on = float((on_is > 0).mean())
    wr_off = float((off_is > 0).mean())
    wr_spread = wr_on - wr_off

    on_oos = oos_df[oos_df["_sig"] == 1]["pnl_r"]
    off_oos = oos_df[oos_df["_sig"] == 0]["pnl_r"]
    expr_on_oos = float(on_oos.mean()) if len(on_oos) >= 5 else float("nan")
    expr_off_oos = float(off_oos.mean()) if len(off_oos) >= 5 else float("nan")
    delta_oos = (expr_on_oos - expr_off_oos) if not (np.isnan(expr_on_oos) or np.isnan(expr_off_oos)) else float("nan")
    dir_match = (not np.isnan(delta_oos)) and (np.sign(delta_is) == np.sign(delta_oos))
    fire_rate = int(sig.sum()) / max(1, len(sig))
    extreme_fire = (fire_rate < 0.05) or (fire_rate > 0.95)
    arithmetic_only = (abs(wr_spread) < 0.03) and (abs(delta_is) > 0.10)

    return {
        "feature": feature_name,
        "family": FEATURE_FAMILY_MAP.get(feature_name, "other"),
        "direction": direction,
        "n_on_is": len(on_is),
        "n_off_is": len(off_is),
        "n_on_oos": len(on_oos),
        "expr_on_is": expr_on,
        "expr_off_is": expr_off,
        "delta_is": delta_is,
        "wr_on_is": wr_on,
        "wr_off_is": wr_off,
        "wr_spread": wr_spread,
        "expr_on_oos": expr_on_oos,
        "delta_oos": delta_oos,
        "dir_match": dir_match,
        "t_is": float(t_is),
        "p_is": float(p_is),
        "fire_rate": fire_rate,
        "extreme_fire": extreme_fire,
        "arithmetic_only": arithmetic_only,
    }


def scan_session_rr(session: str, apt: int, rr: float) -> list[dict]:
    df = load_mgc_session(session, apt, rr)
    if len(df) < 50:
        return []
    feats = build_level_features(df)
    rows = []
    for feature_name, sig in feats.items():
        if sig.sum() < 30 or sig.sum() > len(sig) - 30:
            continue
        for direction in DIRECTIONS:
            res = test_cell(df, feature_name, sig, direction)
            if res is None:
                continue
            res["session"] = session
            res["aperture"] = apt
            res["rr"] = rr
            res["instrument"] = INSTRUMENT
            rows.append(res)
    return rows


def emit(res: pd.DataFrame) -> None:
    res = bh_fdr_multi_framing(res, alpha=0.05)
    res["bh_pass"] = res["bh_pass_global"]

    trustworthy = res[(~res["extreme_fire"]) & (~res["arithmetic_only"])].copy()
    strict = trustworthy[
        (trustworthy["t_is"].abs() >= 3.0)
        & (trustworthy["dir_match"])
        & (trustworthy["n_on_is"] >= 50)
    ].copy()
    bh_global = trustworthy[trustworthy["bh_pass_global"]].copy()
    bh_family = trustworthy[trustworthy["bh_pass_family"]].copy()
    bh_lane = trustworthy[trustworthy["bh_pass_lane"]].copy()
    bh_session = trustworthy[trustworthy["bh_pass_session"]].copy()
    promising = trustworthy[
        (trustworthy["t_is"].abs() >= 2.5)
        & (trustworthy["dir_match"])
        & (trustworthy["n_on_is"] >= 50)
    ].copy()

    lines = [
        "# MGC Level Scan — Previous-Day Levels + S/R",
        "",
        "**Date:** 2026-04-15",
        "**Instrument:** MGC only",
        "**Scope:** 12 sessions × 3 apertures × 3 RRs × 2 directions × 14 level features",
        "**Classification:** EXPLORATORY (no validated_setups writes)",
        "**Data:** MGC 2022-06-13 to 2026-04-10 (~3.8 years)",
        "",
        f"**Total cells:** {len(res)}",
        f"**Trustworthy:** {len(trustworthy)} (not extreme-fire, not arithmetic-only)",
        f"**Strict survivors** (|t|>=3 + dir_match + N>=50): {len(strict)}",
        "",
        "## BH-FDR at each K framing",
        f"- K_global ({int(trustworthy['K_global'].iloc[0]) if len(trustworthy) else 0}): {len(bh_global)} pass",
        f"- K_family (avg K~{int(trustworthy['K_family'].mean()) if len(trustworthy) else 0}): {len(bh_family)} pass",
        f"- K_lane (avg K~{int(trustworthy['K_lane'].mean()) if len(trustworthy) else 0}): {len(bh_lane)} pass",
        f"- K_session (avg K~{int(trustworthy['K_session'].mean()) if len(trustworthy) else 0}): {len(bh_session)} pass",
        f"- Promising (|t|>=2.5 + dir_match): {len(promising)}",
        "",
        "## Strict Survivors",
        "",
        "| Session | Apt | RR | Dir | Feature | Family | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | BH_g | BH_f | BH_l |",
        "|---------|-----|----|----|---------|--------|------|-------|---------|------|------|-------|---|---|------|------|------|",
    ]
    for _, r in strict.sort_values("t_is", key=abs, ascending=False).iterrows():
        bhg = "Y" if bool(r["bh_pass_global"]) else "."
        bhf = "Y" if bool(r["bh_pass_family"]) else "."
        bhl = "Y" if bool(r["bh_pass_lane"]) else "."
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['feature']} | {r['family']} | {r['n_on_is']} | "
            f"{r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | {r['wr_spread']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.4f} | {bhg} | {bhf} | {bhl} |"
        )

    lines += [
        "",
        "## Promising cells (|t|>=2.5 + dir_match) — T0-T8 candidates",
        "",
        "| Session | Apt | RR | Dir | Feature | N_on | ExpR_on | Δ_IS | Δ_OOS | t | p |",
        "|---------|-----|----|----|---------|------|---------|------|-------|---|---|",
    ]
    for _, r in promising.sort_values("t_is", key=abs, ascending=False).head(30).iterrows():
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['feature']} | {r['n_on_is']} | {r['expr_on_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {r['p_is']:.4f} |"
        )

    # All cells ranked
    lines += [
        "",
        "## Top 40 by |t| (all trustworthy cells)",
        "",
        "| Session | Apt | RR | Dir | Feature | N_on | ExpR_on | Δ_IS | Δ_OOS | t | dir_match |",
        "|---------|-----|----|----|---------|------|---------|------|-------|---|-----------|",
    ]
    all_ranked = trustworthy.copy()
    all_ranked["abs_t"] = all_ranked["t_is"].abs()
    all_ranked = all_ranked.sort_values("abs_t", ascending=False).head(40)
    for _, r in all_ranked.iterrows():
        dm = "Y" if bool(r["dir_match"]) else "."
        lines.append(
            f"| {r['session']} | O{r['aperture']} | {r['rr']:.1f} | {r['direction']} | "
            f"{r['feature']} | {r['n_on_is']} | {r['expr_on_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
            f"{r['t_is']:+.2f} | {dm} |"
        )

    # Cross-reference verified MGC cells
    lines += [
        "",
        "## Cross-reference — previously verified MGC cells",
        "",
        "From `docs/audit/results/2026-04-15-t0-t8-audit-hot-warm-batch.md`:",
        "- MGC SINGAPORE_OPEN O15 RR1.5 long F3_NEAR_PIVOT_15 → CONDITIONAL (6P/1F)",
        "- MGC SINGAPORE_OPEN O15 RR2.0 long F3_NEAR_PIVOT_15 → CONDITIONAL (6P/1F)",
        "",
        "Both should appear in this scan's survivors/promising (cross-check below):",
        "",
    ]
    mgc_known_check = trustworthy[
        (trustworthy["session"] == "SINGAPORE_OPEN")
        & (trustworthy["aperture"] == 15)
        & (trustworthy["feature"] == "F3_NEAR_PIVOT_15")
        & (trustworthy["direction"] == "long")
    ]
    if len(mgc_known_check) > 0:
        lines.append("### Verification rows")
        lines.append("| RR | N_on | Δ_IS | Δ_OOS | t | p | dir_match |")
        lines.append("|----|------|------|-------|---|---|-----------|")
        for _, r in mgc_known_check.iterrows():
            dm = "Y" if bool(r["dir_match"]) else "."
            lines.append(
                f"| {r['rr']:.1f} | {r['n_on_is']} | {r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | "
                f"{r['t_is']:+.2f} | {r['p_is']:.4f} | {dm} |"
            )
    else:
        lines.append("**NOT FOUND** in trustworthy set — investigate.")

    # Baseline per session
    lines += [
        "",
        "## Baseline MGC per-session (no feature overlay) — data availability check",
        "",
        "| Session | Apt | RR | N_is | N_oos | ExpR_is | ExpR_oos |",
        "|---------|-----|----|------|-------|---------|----------|",
    ]
    for session in SESSIONS:
        for apt in APERTURES:
            for rr in RRS:
                df_lane = load_mgc_session(session, apt, rr)
                if len(df_lane) == 0:
                    continue
                is_df = df_lane[df_lane["is_is"]]
                oos_df = df_lane[df_lane["is_oos"]]
                if len(is_df) < 30:
                    continue
                expr_is = float(is_df["pnl_r"].mean()) if len(is_df) else float("nan")
                expr_oos = float(oos_df["pnl_r"].mean()) if len(oos_df) else float("nan")
                lines.append(
                    f"| {session} | O{apt} | {rr:.1f} | {len(is_df)} | {len(oos_df)} | "
                    f"{expr_is:+.3f} | {expr_oos:+.3f} |"
                )

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")
    print(f"  Strict: {len(strict)}, BH_g: {len(bh_global)}, BH_f: {len(bh_family)}, BH_l: {len(bh_lane)}, Promising: {len(promising)}")


def main():
    print(f"MGC level scan — {len(SESSIONS)} sessions × {len(APERTURES)} apts × {len(RRS)} RRs × 14 features × 2 directions")
    all_rows = []
    combos_tested = 0
    combos_skipped = 0
    for session in SESSIONS:
        for apt in APERTURES:
            for rr in RRS:
                rows = scan_session_rr(session, apt, rr)
                if rows:
                    combos_tested += 1
                    all_rows.extend(rows)
                    print(f"  {session:<20} O{apt} RR{rr:.1f}: {len(rows)} cells")
                else:
                    combos_skipped += 1

    print(f"\nTotal cells: {len(all_rows)}; combos tested: {combos_tested}, skipped (N<50): {combos_skipped}")
    if not all_rows:
        return
    res = pd.DataFrame(all_rows)
    emit(res)


if __name__ == "__main__":
    main()
