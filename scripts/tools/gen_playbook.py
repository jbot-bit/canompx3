#!/usr/bin/env python
"""Generate MARKET_PLAYBOOK.md from validated_setups snapshot."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

PROJECT = Path(__file__).parent.parent.parent
CSV = PROJECT / "research" / "output" / "_playbook_data.csv"
OUT = PROJECT / "MARKET_PLAYBOOK.md"

SESSION_ORDER = [
    ("NYSE_OPEN",      "00:30", "23:30", "NYSE cash open 9:30 AM ET"),
    ("US_DATA_1000",   "01:00", "00:00", "US 10:00 AM data (ISM/CC)"),
    ("COMEX_SETTLE",   "04:30", "03:30", "COMEX gold settlement 1:30 PM ET"),
    ("CME_PRECLOSE",   "06:45", "05:45", "CME equity pre-settlement 2:45 PM CT"),
    ("NYSE_CLOSE",     "07:00", "06:00", "NYSE closing bell 4:00 PM ET"),
    ("CME_REOPEN",     "09:00", "08:00", "CME Globex reopen 5:00 PM CT"),
    ("TOKYO_OPEN",     "10:00", "10:00", "Tokyo Stock Exchange 9:00 AM JST"),
    ("BRISBANE_1025",  "10:25", "10:25", "Fixed 10:25 AM Brisbane"),
    ("SINGAPORE_OPEN", "11:00", "11:00", "SGX/HKEX open 9:00 AM SGT"),
    ("LONDON_METALS",  "18:00", "17:00", "London metals AM 8:00 AM London"),
    ("US_DATA_830",    "23:30", "22:30", "US econ data 8:30 AM ET"),
]

INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)


def main():
    df = pd.read_csv(CSV)
    L = []  # lines

    # ── Header ──
    L.append("# Market Playbook — What Do I Trade?")
    L.append("")
    L.append(
        f"**Auto-generated from gold.db validated_setups.** "
        f"{len(df)} active strategies across 4 instruments, "
        f"{df.orb_label.nunique()} sessions."
    )
    L.append("**Snapshot date:** 2026-03-04. Entry models: E1, E2 (dominant). E0 purged. E3 retired.")
    L.append("")
    L.append(
        "**HOW TO READ THIS:** Find your Brisbane time below. "
        "Each session shows the **best strategies per instrument at each RR level**. "
        "Only RR >= 1.5 shown in detail. RR1.0 count noted for reference."
    )
    L.append("")
    L.append("---")
    L.append("")

    # ── Quick Reference ──
    L.append("## Quick Reference — Session Schedule")
    L.append("")
    L.append("| Brisbane (Winter) | Brisbane (Summer) | Session | Event | MGC | MNQ | MES | M2K |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sess, bw, bs, event in SESSION_ORDER:
        cells = []
        for inst in INSTRUMENTS:
            mask = (df.orb_label == sess) & (df.instrument == inst)
            n = mask.sum()
            rr15 = ((df.orb_label == sess) & (df.instrument == inst) & (df.rr_target >= 1.5)).sum()
            if n == 0:
                cells.append("-")
            elif rr15 > 0:
                cells.append(f"**{rr15}** ({n})")
            else:
                cells.append(str(n))
        L.append(f"| {bw} | {bs} | {sess} | {event} | {' | '.join(cells)} |")

    L.append("")
    L.append("*Bold = strategies at RR >= 1.5 (count at RR1.5+, total in parens). Plain number = RR1.0 only.*")
    L.append("")
    L.append("---")
    L.append("")

    # ── Per-Session Detail ──
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

            high_rr = idf[idf.rr_target >= 1.5]
            low_rr = idf[idf.rr_target < 1.5]

            L.append(f"### {inst}")

            if high_rr.empty:
                L.append(f"*{len(low_rr)} strategies at RR1.0 only.*")
                best = idf.sort_values("expectancy_r", ascending=False).iloc[0]
                L.append(
                    f"Best: {best.entry_model} CB{best.confirm_bars} "
                    f"{best.filter_type} O{int(best.orb_minutes)} "
                    f"RR{best.rr_target} — "
                    f"N={best.sample_size}, WR={best.win_rate:.0%}, "
                    f"ExpR={best.expectancy_r:+.3f}, Sharpe={best.sharpe_ratio:.3f}"
                )
                L.append("")
                continue

            L.append("")
            L.append("| RR | Entry | CB | Filter | Aperture | N | WR | ExpR | Sharpe | AllYrs+ |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")

            for rr in sorted(high_rr.rr_target.unique(), reverse=True):
                rr_df = high_rr[high_rr.rr_target == rr].sort_values(
                    "expectancy_r", ascending=False
                )
                for _, row in rr_df.head(3).iterrows():
                    ayp = "Y" if row.all_years_positive else ""
                    L.append(
                        f"| {row.rr_target} | {row.entry_model} | "
                        f"{int(row.confirm_bars)} | {row.filter_type} | "
                        f"{int(row.orb_minutes)}m | {int(row.sample_size)} | "
                        f"{row.win_rate:.0%} | {row.expectancy_r:+.4f} | "
                        f"{row.sharpe_ratio:.4f} | {ayp} |"
                    )

            n_rr10 = len(low_rr)
            if n_rr10 > 0:
                L.append("")
                L.append(f"*Plus {n_rr10} strategies at RR1.0.*")
            L.append("")

        L.append("---")
        L.append("")

    # ── Instrument Summary ──
    L.append("## Instrument Summary")
    L.append("")
    L.append("| Instrument | Total | Sessions | Best Session | Best ExpR | Best Sharpe |")
    L.append("|---|---|---|---|---|---|")
    for inst in INSTRUMENTS:
        idf = df[df.instrument == inst]
        n = len(idf)
        sessions = sorted(idf.orb_label.unique())
        best = idf.sort_values("expectancy_r", ascending=False).iloc[0]
        best_s = idf.sort_values("sharpe_ratio", ascending=False).iloc[0]
        sess_str = ", ".join(sessions)
        L.append(
            f"| {inst} | {n} | {len(sessions)} ({sess_str}) | "
            f"{best.orb_label} | {best.expectancy_r:+.4f} | {best_s.sharpe_ratio:.4f} |"
        )

    L.append("")
    L.append("---")
    L.append("")

    # ── Notes ──
    L.append("## Notes")
    L.append("")
    L.append("- **All strategies are post-cost, FDR-corrected, multi-year validated.**")
    L.append("- **E2** (stop-market) is the dominant entry model. E0 purged Feb 2026. E3 soft-retired.")
    L.append("- **Aperture**: 5m = 5-minute ORB, 15m/30m = wider opening range.")
    L.append("- **Filter types**: NO_FILTER (all days), ORB_G4/G5/G6/G8 (ORB size >= N points), VOL_RV12_N20 (realized vol filter), DIR_LONG (long-only), FAST5/FAST10 (fast-break), CONT (contiguous), NOTUE (exclude Tuesday).")
    L.append("- **AllYrs+**: Every calendar year tested was individually profitable.")
    L.append("- **This file is a SNAPSHOT.** Rebuild after any strategy_validator run.")
    L.append("- **To regenerate:** `python scripts/tools/gen_playbook.py` (requires CSV snapshot in research/output/).")
    L.append("- **Cost model**: Round-trip friction deducted per instrument (see cost_model.py).")
    L.append("- **Brisbane winter** = UTC+10 (no DST). US times shift +/-1hr seasonally (ET DST).")
    L.append("")

    text = "\n".join(L)
    OUT.write_text(text, encoding="utf-8")
    print(f"Written {len(L)} lines, {len(text):,} chars to {OUT}")


if __name__ == "__main__":
    main()
