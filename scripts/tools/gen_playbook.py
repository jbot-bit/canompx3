#!/usr/bin/env python
"""Generate MARKET_PLAYBOOK.md from validated_setups snapshot."""

from __future__ import annotations

import datetime

import pandas as pd
from pathlib import Path

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

PROJECT = Path(__file__).parent.parent.parent
CSV = PROJECT / "research" / "output" / "_playbook_data.csv"
OUT = PROJECT / "MARKET_PLAYBOOK.md"

SESSION_ORDER = [
    ("NYSE_OPEN", "00:30", "23:30", "NYSE cash open 9:30 AM ET"),
    ("US_DATA_1000", "01:00", "00:00", "US 10:00 AM data (ISM/CC)"),
    ("COMEX_SETTLE", "04:30", "03:30", "COMEX gold settlement 1:30 PM ET"),
    ("CME_PRECLOSE", "06:45", "05:45", "CME equity pre-settlement 2:45 PM CT"),
    ("NYSE_CLOSE", "07:00", "06:00", "NYSE closing bell 4:00 PM ET"),
    ("CME_REOPEN", "09:00", "08:00", "CME Globex reopen 5:00 PM CT"),
    ("TOKYO_OPEN", "10:00", "10:00", "Tokyo Stock Exchange 9:00 AM JST"),
    ("BRISBANE_1025", "10:25", "10:25", "Fixed 10:25 AM Brisbane"),
    ("SINGAPORE_OPEN", "11:00", "11:00", "SGX/HKEX open 9:00 AM SGT"),
    ("LONDON_METALS", "18:00", "17:00", "London metals AM 8:00 AM London"),
    ("US_DATA_830", "23:30", "22:30", "US econ data 8:30 AM ET"),
]

# Display order matches the Quick Reference table header — do NOT use sorted().
# sorted(ACTIVE_ORB_INSTRUMENTS) gives ['M2K','MES','MGC','MNQ'] which mismatches
# the hardcoded header columns (MGC|MNQ|MES|M2K) and scrambles every cell.
INSTRUMENTS = ["MGC", "MNQ", "MES", "M2K"]
assert set(INSTRUMENTS) == set(ACTIVE_ORB_INSTRUMENTS), (
    f"INSTRUMENTS out of sync: {INSTRUMENTS} vs {list(ACTIVE_ORB_INSTRUMENTS)}"
)

# Minimum expected R per trade to show a strategy in detail.
# Below this threshold the edge is too small to trust given real-world friction.
MIN_EXPR = 0.05


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank score = ExpR × √N × robustness_bonus.

    Rationale for ORB binary outcomes:
    - ExpR is the actual edge per trade (already post-cost).
    - √N makes this a t-stat proxy: larger samples make the mean more trustworthy.
    - all_years_positive bonus (+25%) rewards regime robustness without over-weighting it.
    - Sharpe is NOT used because it structurally penalises high-RR configs (their
      binary variance is a feature, not a flaw).
    """
    robustness = df["all_years_positive"].map({True: 1.25, False: 1.0})
    df = df.copy()
    df["score"] = df["expectancy_r"] * df["sample_size"].pow(0.5) * robustness
    return df


def fmt_row(row) -> str:
    ayp = "★" if row.all_years_positive else ""
    return (
        f"| {row.rr_target} | {row.entry_model} | "
        f"{int(row.confirm_bars)} | {row.filter_type} | "
        f"{int(row.orb_minutes)}m | {int(row.sample_size)} | "
        f"{int(row.years_tested)}y | {row.win_rate:.0%} | "
        f"{row.expectancy_r:+.4f} | {ayp} |"
    )


def main():
    df = pd.read_csv(CSV)
    df = add_score(df)
    today = datetime.date.today().isoformat()
    L = []  # lines

    # ── Header ──────────────────────────────────────────────────────────────
    L.append("# Market Playbook — What Do I Trade?")
    L.append("")
    L.append(
        f"**Auto-generated from gold.db validated_setups.** "
        f"{len(df)} active strategies across {df.instrument.nunique()} instruments, "
        f"{df.orb_label.nunique()} sessions."
    )
    L.append(f"**Snapshot date:** {today}. Entry models: E1, E2 (dominant). E0 purged. E3 retired.")
    L.append("")
    L.append(
        "**HOW TO READ THIS:** Find your Brisbane time below. "
        "Each session shows the top strategies per instrument at each RR level, "
        f"ranked by **score = ExpR × √N × robustness** (★ = profitable every year tested). "
        f"Only strategies with ExpR ≥ {MIN_EXPR} shown — weaker ones are FDR-validated but not worth trading."
    )
    L.append("")
    L.append("---")
    L.append("")

    # ── Quick Reference ──────────────────────────────────────────────────────
    L.append("## Quick Reference — Session Schedule")
    L.append("")
    header_insts = " | ".join(INSTRUMENTS)
    L.append(f"| Brisbane (Winter) | Brisbane (Summer) | Session | Event | {header_insts} |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sess, bw, bs, event in SESSION_ORDER:
        cells = []
        for inst in INSTRUMENTS:
            mask = (df.orb_label == sess) & (df.instrument == inst)
            n = mask.sum()
            rr15 = (mask & (df.rr_target >= 1.5)).sum()
            if n == 0:
                cells.append("-")
            elif rr15 > 0:
                cells.append(f"**{rr15}** ({n})")
            else:
                cells.append(str(n))
        L.append(f"| {bw} | {bs} | {sess} | {event} | {' | '.join(cells)} |")

    L.append("")
    L.append("*Bold = strategies at RR ≥ 1.5 (count at RR1.5+, total in parens). Plain number = RR1.0 only.*")
    L.append("")
    L.append("---")
    L.append("")

    # ── Per-Session Detail ───────────────────────────────────────────────────
    for sess, bw, bs, event in SESSION_ORDER:
        sdf = df[df.orb_label == sess]
        if sdf.empty:
            continue

        L.append(f"## {sess} — {bw} Winter / {bs} Summer")
        L.append(f"*{event}*")
        L.append("")

        for inst in INSTRUMENTS:
            idf = sdf[sdf.instrument == inst].copy()
            if idf.empty:
                continue

            high_rr = idf[(idf.rr_target >= 1.5) & (idf.expectancy_r >= MIN_EXPR)]
            low_rr = idf[idf.rr_target < 1.5]

            L.append(f"### {inst}")

            if high_rr.empty:
                # All validated strategies are below RR1.5 or below quality floor.
                # low_rr may include RR1.0, RR1.25, etc — describe by actual RR levels.
                qual = idf[idf.expectancy_r >= MIN_EXPR]
                if qual.empty:
                    n_all = len(idf)
                    best = idf.sort_values("score", ascending=False).iloc[0]
                    L.append(f"*{n_all} validated strategies, all below quality floor (ExpR < {MIN_EXPR}).*")
                    L.append(
                        f"Best available: {best.entry_model} CB{int(best.confirm_bars)} "
                        f"{best.filter_type} O{int(best.orb_minutes)}m "
                        f"RR{best.rr_target} — "
                        f"N={int(best.sample_size)}, WR={best.win_rate:.0%}, ExpR={best.expectancy_r:+.3f}"
                    )
                else:
                    rr_levels = sorted(low_rr.rr_target.unique())
                    rr_label = "/".join(f"RR{r}" for r in rr_levels) if rr_levels else "low-RR"
                    # Best = highest-scoring strategy that clears the quality floor
                    best = qual.sort_values("score", ascending=False).iloc[0]
                    L.append(f"*{len(low_rr)} strategies at {rr_label} only.*")
                    L.append(
                        f"Best: {best.entry_model} CB{int(best.confirm_bars)} "
                        f"{best.filter_type} O{int(best.orb_minutes)}m "
                        f"RR{best.rr_target} — "
                        f"N={int(best.sample_size)}, {int(best.years_tested)}y, "
                        f"WR={best.win_rate:.0%}, ExpR={best.expectancy_r:+.3f}"
                    )
                L.append("")
                continue

            L.append("")
            L.append("| RR | Entry | CB | Filter | Apt | N | Yrs | WR | ExpR | AllYrs |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")

            for rr in sorted(high_rr.rr_target.unique(), reverse=True):
                rr_df = high_rr[high_rr.rr_target == rr].sort_values("score", ascending=False)
                for _, row in rr_df.head(3).iterrows():
                    L.append(fmt_row(row))

            n_rr10 = len(low_rr)
            if n_rr10 > 0:
                L.append("")
                L.append(f"*Plus {n_rr10} strategies at RR1.0.*")
            L.append("")

        L.append("---")
        L.append("")

    # ── Instrument Summary ───────────────────────────────────────────────────
    L.append("## Instrument Summary")
    L.append("")
    L.append("| Instrument | Total | Sessions | Best Session | Best ExpR | Best Score |")
    L.append("|---|---|---|---|---|---|")
    for inst in INSTRUMENTS:
        idf = df[df.instrument == inst]
        sessions = sorted(idf.orb_label.unique())
        best_expr = idf.sort_values("expectancy_r", ascending=False).iloc[0]
        best_score = idf.sort_values("score", ascending=False).iloc[0]
        sess_str = ", ".join(sessions)
        L.append(
            f"| {inst} | {len(idf)} | {len(sessions)} ({sess_str}) | "
            f"{best_expr.orb_label} | {best_expr.expectancy_r:+.4f} | {best_score.score:.2f} |"
        )

    L.append("")
    L.append("---")
    L.append("")

    # ── Notes ────────────────────────────────────────────────────────────────
    L.append("## Notes")
    L.append("")
    L.append("- **All strategies are post-cost, FDR-corrected, multi-year validated.**")
    L.append("- **E2** (stop-market) is the dominant entry model. E0 purged Feb 2026. E3 soft-retired.")
    L.append(
        "- **Ranking:** `score = ExpR × √N × 1.25 (if all years positive)`. "
        "Sharpe is not used — it structurally penalises high-RR configs whose binary variance is expected."
    )
    L.append(f"- **Quality floor:** ExpR < {MIN_EXPR} strategies are FDR-validated but excluded from detail tables.")
    L.append("- **Apt**: ORB aperture — 5m = 5-minute range, 15m/30m = wider opening range.")
    L.append("- **Yrs**: calendar years in the backtest window.")
    L.append("- **★ (AllYrs)**: every individual calendar year was profitable.")
    L.append(
        "- **Filter types**: NO_FILTER (all days), ORB_G4/G5/G6/G8 (ORB size ≥ N points), "
        "VOL_RV12_N20 (realized vol filter), DIR_LONG (long-only), FAST5/FAST10 (fast-break), "
        "CONT (contiguous), NOMON (exclude Monday), NOTUE (exclude Tuesday)."
    )
    L.append("- **This file is a SNAPSHOT.** Rebuild after any strategy_validator run.")
    L.append("- **To regenerate:** `python scripts/tools/gen_playbook.py` (requires CSV snapshot in research/output/).")
    L.append("- **Cost model**: round-trip friction deducted per instrument (see `pipeline/cost_model.py`).")
    L.append("- **Brisbane winter** = UTC+10 (no DST). US times shift ±1 hr seasonally (ET DST).")
    L.append("")

    text = "\n".join(L)
    OUT.write_text(text, encoding="utf-8")
    print(f"Written {len(L)} lines, {len(text):,} chars to {OUT}")


if __name__ == "__main__":
    main()
