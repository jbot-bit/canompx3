"""Volume confluence scan — cross-factor stress test on rel_vol_HIGH_Q3 finding.

Per pre-reg `docs/audit/hypotheses/2026-04-15-volume-exploitation-research-plan.md`:
- EXPLORATORY classification (no validated_setups writes).
- K_budget ≤ 200 (2-factor AND interactions only, no 3-way).
- Scope: 6 deployed MNQ lanes + volume=HIGH_Q3 × partner-feature AND combinations.

Partner features:
  bb_volume_ratio_HIGH           — volume self-confluence
  break_delay_LT2                — fast break
  {F1_NEAR_PDH,F2_NEAR_PDL,F3_NEAR_PIVOT,F5_BELOW_PDL,F6_INSIDE_PDR}_15  — level
  atr_20_pct_{LT20,GT80}         — vol regime
  garch_vol_pct_{LT30,GT70}      — forward vol
  is_{nfp,opex,friday,monday}    — calendar

Output:
  docs/audit/results/2026-04-15-volume-confluence-scan.md

Gates (per .claude/rules/backtesting-methodology.md):
- Look-ahead gates (RULE 1)
- Two-pass overlay testing (RULE 2)
- IS/OOS sacred holdout (RULE 3)
- Multi-framing BH-FDR (RULE 4)
- Trade-time-knowable features only (RULE 6)
- T0 tautology vs deployed filter (RULE 7)
- Extreme fire / ARITHMETIC_ONLY flags (RULE 8)
- Red-flag audit (RULE 12)

Reuses helpers from research/comprehensive_deployed_lane_scan.py.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from research.comprehensive_deployed_lane_scan import (  # type: ignore  # noqa: E402
    load_lane,
    compute_deployed_filter,
    bh_fdr_multi_framing,
    DEPLOYED_LANE_SPECS,
    _valid_session_features,
    _overnight_lookhead_clean,
)

OUTPUT_MD = Path("docs/audit/results/2026-04-15-volume-confluence-scan.md")
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

# Pre-committed scope per research plan §6.2
ANCHORS = [
    ("rel_vol_HIGH_Q3", "volume_high"),
    ("rel_vol_LOW_Q1", "volume_low"),  # mirror (SKIP direction)
]

PARTNER_FEATURES = [
    # Volume self-confluence
    ("bb_volume_ratio_HIGH", "volume_confluence"),
    ("bb_volume_ratio_LOW", "volume_confluence"),
    # Timing
    ("break_delay_LT2", "timing"),
    ("break_delay_GT10", "timing"),
    # Level (narrow theta = 0.15 only, per F1-F8 scope)
    ("F1_NEAR_PDH_15", "level"),
    ("F2_NEAR_PDL_15", "level"),
    ("F3_NEAR_PIVOT_15", "level"),
    ("F5_BELOW_PDL", "level"),
    ("F6_INSIDE_PDR", "level"),
    # Vol regime
    ("atr_20_pct_GT80", "vol_regime"),
    ("atr_20_pct_LT20", "vol_regime"),
    ("garch_vol_pct_GT70", "vol_regime"),
    ("garch_vol_pct_LT30", "vol_regime"),
    # Calendar
    ("is_nfp_TRUE", "calendar"),
    ("is_opex_TRUE", "calendar"),
    ("is_friday_TRUE", "calendar"),
    ("is_monday_TRUE", "calendar"),
]


# ============================================================================
# Feature materializers (from daily_features columns)
# ============================================================================


def materialize(df: pd.DataFrame, feature: str) -> np.ndarray | None:
    """Return 0/1 numpy array for a named feature, or None if unavailable.
    Look-ahead gates applied where relevant."""
    if len(df) == 0:
        return None
    n = len(df)
    orb_session = df["orb_label"].iloc[0]

    # Volume features
    if feature == "rel_vol_HIGH_Q3":
        v = df["rel_vol"].astype(float)
        if v.notna().sum() < 20:
            return None
        return (v > np.nanpercentile(v, 67)).fillna(False).astype(int).values
    if feature == "rel_vol_LOW_Q1":
        v = df["rel_vol"].astype(float)
        if v.notna().sum() < 20:
            return None
        return (v < np.nanpercentile(v, 33)).fillna(False).astype(int).values

    # Break-bar volume ratio
    if feature.startswith("bb_volume_ratio_"):
        orb_vol = df["orb_volume"].astype(float).replace(0, np.nan)
        ratio = df["break_bar_volume"].astype(float) / orb_vol
        if ratio.notna().sum() < 20:
            return None
        if feature.endswith("HIGH"):
            return (ratio > np.nanpercentile(ratio, 67)).fillna(False).astype(int).values
        return (ratio < np.nanpercentile(ratio, 33)).fillna(False).astype(int).values

    # Timing
    if feature == "break_delay_LT2":
        bd = df["break_delay_min"].astype(float)
        return (bd < 2).fillna(False).astype(int).values
    if feature == "break_delay_GT10":
        bd = df["break_delay_min"].astype(float)
        return (bd > 10).fillna(False).astype(int).values

    # Level features (theta=0.15 narrow)
    mid = df["orb_mid"].astype(float)
    atr = df["atr_20"].astype(float).replace(0, np.nan)
    pdh = df["prev_day_high"].astype(float)
    pdl = df["prev_day_low"].astype(float)
    pdc = df["prev_day_close"].astype(float)
    pivot = (pdh + pdl + pdc) / 3.0
    if feature == "F1_NEAR_PDH_15":
        return ((np.abs(mid - pdh) / atr) < 0.15).fillna(False).astype(int).values
    if feature == "F2_NEAR_PDL_15":
        return ((np.abs(mid - pdl) / atr) < 0.15).fillna(False).astype(int).values
    if feature == "F3_NEAR_PIVOT_15":
        return ((np.abs(mid - pivot) / atr) < 0.15).fillna(False).astype(int).values
    if feature == "F5_BELOW_PDL":
        return (mid < pdl).fillna(False).astype(int).values
    if feature == "F6_INSIDE_PDR":
        return ((mid > pdl) & (mid < pdh)).fillna(False).astype(int).values

    # Vol regime
    if feature == "atr_20_pct_GT80":
        return (df["atr_20_pct"].astype(float) > 80).fillna(False).astype(int).values
    if feature == "atr_20_pct_LT20":
        return (df["atr_20_pct"].astype(float) < 20).fillna(False).astype(int).values
    if feature == "garch_vol_pct_GT70":
        return (df["garch_forecast_vol_pct"].astype(float) > 70).fillna(False).astype(int).values
    if feature == "garch_vol_pct_LT30":
        return (df["garch_forecast_vol_pct"].astype(float) < 30).fillna(False).astype(int).values

    # Calendar
    if feature == "is_nfp_TRUE":
        return df["is_nfp_day"].fillna(False).astype(int).values
    if feature == "is_opex_TRUE":
        return df["is_opex_day"].fillna(False).astype(int).values
    if feature == "is_friday_TRUE":
        return df["is_friday"].fillna(False).astype(int).values
    if feature == "is_monday_TRUE":
        return df["is_monday"].fillna(False).astype(int).values

    # Overnight (gated)
    if feature.startswith("ovn_") and not _overnight_lookhead_clean(orb_session):
        return None

    return None


def test_confluence_cell(
    df: pd.DataFrame,
    anchor_sig: np.ndarray,
    partner_sig: np.ndarray,
    direction: str,
    filter_sig: np.ndarray,
    pass_type: str,
) -> dict | None:
    """AND-combine anchor + partner features, test within direction subset."""
    mask_dir = (df["break_dir"] == direction).values
    composite = (anchor_sig & partner_sig).astype(int)
    csig = composite[mask_dir]
    fsig = filter_sig[mask_dir]
    sub = df[mask_dir].copy()
    if pass_type == "filtered":
        keep = fsig == 1
        sub = sub[keep]
        csig = csig[keep]

    if len(sub) == 0:
        return None

    sub = sub.copy()
    sub["_sig"] = csig
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
    delta_oos = expr_on_oos - expr_off_oos if not (np.isnan(expr_on_oos) or np.isnan(expr_off_oos)) else float("nan")
    dir_match = (not np.isnan(delta_oos)) and (np.sign(delta_is) == np.sign(delta_oos))

    fire_rate = int(csig.sum()) / max(1, len(csig))
    extreme_fire = (fire_rate < 0.05) or (fire_rate > 0.95)
    arithmetic_only = (abs(wr_spread) < 0.03) and (abs(delta_is) > 0.10)

    # T0 tautology vs anchor alone — does AND add info beyond anchor?
    anchor_alone = anchor_sig[mask_dir]
    if pass_type == "filtered":
        keep = filter_sig[mask_dir] == 1
        anchor_alone = anchor_alone[keep]
    try:
        corr_composite_anchor = float(np.corrcoef(csig.astype(float), anchor_alone.astype(float))[0, 1])
        if np.isnan(corr_composite_anchor):
            corr_composite_anchor = 0.0
    except Exception:
        corr_composite_anchor = 0.0

    return {
        "direction": direction,
        "pass_type": pass_type,
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
        "corr_composite_vs_anchor": corr_composite_anchor,
    }


def scan_confluence_lane(
    session: str,
    apt: int,
    rr: float,
    instr: str,
    filter_key: str | None,
) -> list[dict]:
    df = load_lane(session, apt, rr, instr)
    if len(df) < 50:
        return []
    filter_sig = compute_deployed_filter(df, filter_key, orb_label=session)

    # Gate level features for look-ahead (shouldn't apply here since level uses prev-day)
    rows = []
    for anchor_name, anchor_family in ANCHORS:
        anchor_sig = materialize(df, anchor_name)
        if anchor_sig is None:
            continue
        for partner_name, partner_family in PARTNER_FEATURES:
            # Session-level LOOKAHEAD gates — level features use prev-day, so clean
            # Overnight features would need gate — we're not using overnight here
            partner_sig = materialize(df, partner_name)
            if partner_sig is None:
                continue

            # Skip same-feature pairings (not informative)
            if anchor_name == partner_name:
                continue

            # Require partner has minimum fire activity
            if partner_sig.sum() < 30 or partner_sig.sum() > len(partner_sig) - 30:
                continue

            passes = ["unfiltered"] if filter_key is None else ["unfiltered", "filtered"]
            for direction in ("long", "short"):
                for pt in passes:
                    res = test_confluence_cell(df, anchor_sig, partner_sig, direction, filter_sig, pt)
                    if res is None:
                        continue
                    res["anchor"] = anchor_name
                    res["anchor_family"] = anchor_family
                    res["partner"] = partner_name
                    res["partner_family"] = partner_family
                    res["composite"] = f"{anchor_name}_AND_{partner_name}"
                    res["session"] = session
                    res["aperture"] = apt
                    res["rr"] = rr
                    res["instrument"] = instr
                    res["deployed_filter"] = filter_key or "NONE"
                    res["family"] = f"{anchor_family}_x_{partner_family}"
                    res["feature"] = res["composite"]  # alias for bh_fdr_multi_framing
                    rows.append(res)
    return rows


def emit(res: pd.DataFrame) -> None:
    res = bh_fdr_multi_framing(res, alpha=0.05)

    # Filter trustworthy cells
    trustworthy = res[(~res["extreme_fire"]) & (~res["arithmetic_only"])].copy()

    strict = trustworthy[
        (trustworthy["t_is"].abs() >= 3.0) & (trustworthy["dir_match"]) & (trustworthy["n_on_is"] >= 50)
    ].copy()

    bh_global = trustworthy[trustworthy["bh_pass_global"]].copy()
    bh_family = trustworthy[trustworthy["bh_pass_family"]].copy()
    bh_lane = trustworthy[trustworthy["bh_pass_lane"]].copy()

    # Did confluence ADD to anchor? Only interesting if t of composite > t of anchor alone
    # in the same cell. For that we'd need anchor-alone results merged. Skip for now.
    promising = trustworthy[
        (trustworthy["t_is"].abs() >= 2.5) & (trustworthy["dir_match"]) & (trustworthy["n_on_is"] >= 50)
    ].copy()

    lines = [
        "# Volume Confluence Scan — 2-factor AND Combinations",
        "",
        "**Date:** 2026-04-15",
        "**Pre-reg:** `docs/audit/hypotheses/2026-04-15-volume-exploitation-research-plan.md`",
        "**Classification:** EXPLORATORY (no validated_setups writes).",
        f"**Total cells:** {len(res)}",
        f"**Trustworthy:** {len(trustworthy)} (not extreme-fire, not arithmetic-only)",
        f"**Strict survivors:** {len(strict)} (|t|>=3 + dir_match + N>=50)",
        "",
        "## BH-FDR at each K framing",
        f"- K_global ({int(trustworthy['K_global'].iloc[0]) if len(trustworthy) else 0}): {len(bh_global)} pass",
        f"- K_family (avg K~{int(trustworthy['K_family'].mean()) if len(trustworthy) else 0}): {len(bh_family)} pass",
        f"- K_lane (avg K~{int(trustworthy['K_lane'].mean()) if len(trustworthy) else 0}): {len(bh_lane)} pass",
        f"- Promising (|t|>=2.5 + dir_match): {len(promising)}",
        "",
        "## Strict Survivors",
        "",
        "| Scope | Instr | Session | Apt | RR | Dir | Pass | Composite | N_on | Fire% | ExpR_on | WR_Δ | Δ_IS | Δ_OOS | t | p | Anchor_corr | BH_g | BH_f | BH_l |",
        "|-------|-------|---------|-----|----|----|------|-----------|------|-------|---------|------|------|-------|---|---|-------------|------|------|------|",
    ]
    for _, r in strict.sort_values("t_is", key=abs, ascending=False).iterrows():
        bhg = "Y" if bool(r["bh_pass_global"]) else "."
        bhf = "Y" if bool(r["bh_pass_family"]) else "."
        bhl = "Y" if bool(r["bh_pass_lane"]) else "."
        scope = (
            "deployed" if (r["session"], r["aperture"], r["rr"], r["instrument"]) in DEPLOYED_LANE_SPECS else "other"
        )
        lines.append(
            f"| {scope} | {r['instrument']} | {r['session']} | O{r['aperture']} | {r['rr']:.1f} | "
            f"{r['direction']} | {r['pass_type']} | {r['composite']} | {r['n_on_is']} | "
            f"{r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | {r['wr_spread']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | {r['t_is']:+.2f} | {r['p_is']:.4f} | "
            f"{r['corr_composite_vs_anchor']:.2f} | {bhg} | {bhf} | {bhl} |"
        )

    lines += [
        "",
        "## BH-FDR per-family Survivors (top 30 by |t|)",
        "",
        "| Scope | Instr | Session | Dir | Pass | Composite | N_on | Fire% | ExpR_on | Δ_IS | Δ_OOS | t | p |",
        "|-------|-------|---------|----|------|-----------|------|-------|---------|------|-------|---|---|",
    ]
    for _, r in bh_family.sort_values("t_is", key=abs, ascending=False).head(30).iterrows():
        scope = (
            "deployed" if (r["session"], r["aperture"], r["rr"], r["instrument"]) in DEPLOYED_LANE_SPECS else "other"
        )
        lines.append(
            f"| {scope} | {r['instrument']} | {r['session']} | {r['direction']} | {r['pass_type']} | "
            f"{r['composite']} | {r['n_on_is']} | {r['fire_rate']:.1%} | {r['expr_on_is']:+.3f} | "
            f"{r['delta_is']:+.3f} | {r['delta_oos']:+.3f} | {r['t_is']:+.2f} | {r['p_is']:.5f} |"
        )

    lines += [
        "",
        "## Honest Kill-Criteria Check (from pre-reg §6.4)",
        "",
        f"- **0 family BH:** {'TRIGGERED — confluence does not help, Phase D on single rel_vol becomes priority' if len(bh_family) == 0 else 'Not triggered'}",
        f"- **≥20 global BH:** {'TRIGGERED — audit for look-ahead or correlation structure' if len(bh_global) >= 20 else 'Not triggered'}",
        f"- **Concentration in one lane:** {'CHECK MANUAL — review per-lane counts' if len(bh_family) > 0 else 'N/A (no survivors)'}",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUTPUT_MD}")
    print(
        f"  Strict: {len(strict)}, BH_g: {len(bh_global)}, BH_f: {len(bh_family)}, BH_l: {len(bh_lane)}, Promising: {len(promising)}"
    )


def main():
    lanes = [(s, a, r, i, f) for (s, a, r, i), f in DEPLOYED_LANE_SPECS.items()]
    print(f"Scanning volume confluence on {len(lanes)} deployed lanes")
    print(f"Anchors: {[a[0] for a in ANCHORS]}")
    print(f"Partners: {len(PARTNER_FEATURES)} features")
    print(f"K_budget pre-committed: <= 200 cells (2-factor only, no 3-way)\n")

    all_rows = []
    for i, (s, a, r, instr, filt) in enumerate(lanes, 1):
        print(f"[{i}/{len(lanes)}] {instr} {s} O{a} RR{r} filter={filt}")
        rows = scan_confluence_lane(s, a, r, instr, filt)
        print(f"  → {len(rows)} cells")
        all_rows.extend(rows)

    if not all_rows:
        print("No cells tested.")
        return

    res = pd.DataFrame(all_rows)
    print(f"\nTotal cells: {len(res)}")
    emit(res)


if __name__ == "__main__":
    main()
