"""H1 signal-only shadow retrospective one-shot.

Reads 2026-01-01 to 2026-04-18 OOS data ONCE for the H1 cell
(MES LONDON_METALS O30 RR1.5 long overnight_range_pct >= 80) per
Mode A single-use-OOS discipline. Writes retrospective results md
and initializes the shadow log csv.

Per pre-reg lock at docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml
this retrospective:
- MUST NOT be re-run after commit
- MUST NOT query data past 2026-04-18
- MUST write per-fire rows + aggregate summary
- Output is the locked one-shot; feeds final verdict at review_date 2026-12-15

Mode A source: trading_app/holdout_policy.py HOLDOUT_SACRED_FROM = 2026-01-01.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH

LOCK_DATE = "2026-04-18"
RETROSPECTIVE_FROM = "2026-01-01"
RETROSPECTIVE_TO = LOCK_DATE  # exclusive; query uses < LOCK_DATE below
UNIVERSE = {
    "instrument": "MES",
    "session": "LONDON_METALS",
    "orb_minutes": 30,
    "entry_model": "E2",
    "confirm_bars": 1,
    "rr_target": 1.5,
    "direction": "long",
    "filter_threshold": 80,  # overnight_range_pct >= 80
}


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    output_md = repo_root / "docs" / "audit" / "results" / "2026-04-18-h1-retrospective-one-shot.md"
    output_csv = repo_root / "docs" / "audit" / "results" / "h1-mes-london-metals-shadow-log.csv"

    if output_md.exists():
        print(f"REFUSE: {output_md.name} already exists — one-shot retrospective must not be re-run.")
        return 2

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Retrospective query — 2026-01-01 to 2026-04-18 exclusive upper bound
    sql = """
    WITH df AS (
      SELECT trading_day, overnight_range_pct
      FROM daily_features
      WHERE symbol = 'MES' AND orb_minutes = 5
        AND trading_day >= DATE '2026-01-01'
        AND trading_day < DATE '2026-04-18'
    ),
    oc AS (
      SELECT
        o.trading_day,
        o.entry_ts,
        o.entry_price,
        o.stop_price,
        o.target_price,
        o.pnl_r,
        o.outcome,
        CASE WHEN o.target_price > o.entry_price THEN 'long' ELSE 'short' END AS direction
      FROM orb_outcomes o
      WHERE o.symbol = 'MES'
        AND o.orb_label = 'LONDON_METALS'
        AND o.orb_minutes = 30
        AND o.entry_model = 'E2'
        AND o.confirm_bars = 1
        AND o.rr_target = 1.5
        AND o.trading_day >= DATE '2026-01-01'
        AND o.trading_day < DATE '2026-04-18'
        AND o.outcome NOT IN ('skip_no_break', 'skip_missing_data')
    )
    SELECT
      oc.trading_day,
      oc.entry_ts,
      df.overnight_range_pct,
      CASE WHEN df.overnight_range_pct >= 80 THEN 1 ELSE 0 END AS on_signal,
      oc.entry_price,
      oc.stop_price,
      oc.target_price,
      oc.direction,
      oc.pnl_r,
      oc.outcome
    FROM oc
    LEFT JOIN df USING (trading_day)
    WHERE oc.direction = 'long'
    ORDER BY oc.trading_day
    """

    rows = con.execute(sql).fetchall()
    con.close()

    # Per-fire stats (on-signal only)
    on_signal_rows = [r for r in rows if r[3] == 1]
    n_on = len(on_signal_rows)
    n_long_total = len(rows)
    if n_on > 0:
        pnl_on = [r[8] for r in on_signal_rows]
        expR_on = sum(pnl_on) / n_on
        wins_on = sum(1 for r in on_signal_rows if r[9] == "win")
        losses_on = sum(1 for r in on_signal_rows if r[9] == "loss")
        wr_on = wins_on / (wins_on + losses_on) if (wins_on + losses_on) > 0 else None
        # Sample SD
        mean = expR_on
        var = sum((x - mean) ** 2 for x in pnl_on) / (n_on - 1) if n_on > 1 else 0.0
        sd_on = var**0.5
    else:
        expR_on = None
        wr_on = None
        sd_on = None
        wins_on = 0
        losses_on = 0

    off_signal_rows = [r for r in rows if r[3] == 0]
    n_off = len(off_signal_rows)
    if n_off > 0:
        pnl_off = [r[8] for r in off_signal_rows]
        expR_off = sum(pnl_off) / n_off
    else:
        expR_off = None

    # IS locked values for comparison (from drift check)
    IS_EXP_R = 0.2158
    IS_SD = 1.1629
    IS_WR = 0.5243
    IS_N = 189

    # Write MD
    lines: list[str] = []
    lines.append("# H1 Retrospective One-Shot — MES LONDON_METALS O30 RR1.5 long overnight_range_pct>=80")
    lines.append("")
    lines.append(f"**Date committed:** {LOCK_DATE}")
    lines.append(f"**Window:** {RETROSPECTIVE_FROM} to {RETROSPECTIVE_TO} (exclusive upper bound)")
    lines.append(
        "**Governing pre-reg:** `docs/audit/hypotheses/2026-04-18-h1-mes-london-metals-signal-only-shadow.yaml`"
    )
    lines.append("**Mode A status:** this is part of the single-shot OOS consumption. Not re-runnable.")
    lines.append("")
    lines.append("## Retrospective aggregate")
    lines.append("")
    lines.append("| Metric | IS (pre-2026, from drift check) | OOS retrospective (2026-01-01 to 2026-04-18) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| N (long entries total) | — | {n_long_total} |")
    lines.append(f"| N (on-signal, overnight_range_pct>=80) | {IS_N} | {n_on} |")
    lines.append(
        f"| ExpR on-signal | +{IS_EXP_R:.4f} | {expR_on:+.4f}"
        if expR_on is not None
        else f"| ExpR on-signal | +{IS_EXP_R:.4f} | — (N_on=0) |"
    )
    if expR_on is not None:
        lines[-1] += " |"
    lines.append(f"| SD on-signal | {IS_SD:.4f} | {sd_on:.4f} |" if sd_on is not None else "| SD on-signal | — | — |")
    lines.append(f"| WR on-signal | {IS_WR:.4f} | {wr_on:.4f} |" if wr_on is not None else "| WR on-signal | — | — |")
    lines.append(f"| Wins / Losses on-signal | — | {wins_on} / {losses_on} |")
    lines.append(
        f"| ExpR off-signal | -0.1069 (from drift check) | {expR_off:+.4f} |"
        if expR_off is not None
        else "| ExpR off-signal | — | — |"
    )
    lines.append("")

    # Retrospective gate status (indicative only — combined verdict at review_date)
    lines.append(
        "## Retrospective gate status (INDICATIVE ONLY — final verdict is combined 2026-01-01 to 2026-12-15 at review_date)"
    )
    lines.append("")
    if expR_on is not None:
        eff_ratio = expR_on / IS_EXP_R if IS_EXP_R != 0 else None
        lines.append(f"- **Primary: OOS ExpR >= 0**: {expR_on:+.4f} — {'PASS' if expR_on >= 0 else 'FAIL'}")
        lines.append(
            f"- **Primary: eff_ratio >= 0.40**: {eff_ratio:+.4f} — {'PASS' if eff_ratio and eff_ratio >= 0.40 else 'FAIL'}"
            if eff_ratio is not None
            else "- **Primary: eff_ratio**: N/A"
        )
        lines.append(f"- **Primary: direction match (sign +)**: {'PASS' if expR_on > 0 else 'FAIL'}")
        lines.append(
            f"- **Secondary: N_OOS >= 30**: {n_on} — {'PASS' if n_on >= 30 else 'FAIL (underpowered at retrospective; forward shadow may close gap by review_date)'}"
        )
    else:
        lines.append("- No on-signal fires in retrospective window — shadow continues forward observation only.")
    lines.append("")
    lines.append(
        "**Interpretation:** this retrospective read is PART of the one-shot OOS consumption. It does NOT constitute a gate evaluation. Combined gate evaluation occurs ONCE at 2026-12-15 review date on the full 2026-01-01 to 2026-12-15 universe per the pre-reg. Retrospective PASSes here do not guarantee final PASS; retrospective FAILs here do not auto-kill (forward shadow may shift the combined result)."
    )
    lines.append("")

    lines.append("## Per-fire log (retrospective window)")
    lines.append("")
    if n_on > 0:
        lines.append("| trading_day | ovn_range_pct | entry_price | stop_price | target_price | pnl_r | outcome |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for r in on_signal_rows:
            lines.append(f"| {r[0]} | {r[2]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[8]:+.4f} | {r[9]} |")
    else:
        lines.append("(no on-signal fires in 2026-01-01 to 2026-04-18)")
    lines.append("")

    lines.append("## CSV shadow log initialized")
    lines.append("")
    lines.append(f"Shadow log csv at `docs/audit/results/h1-mes-london-metals-shadow-log.csv` initialized")
    lines.append(f"with {n_on} retrospective on-signal fire(s) + headers. Daily forward shadow appends from")
    lines.append("2026-04-18 through 2026-12-15.")
    lines.append("")

    lines.append("## Compliance checklist")
    lines.append("")
    lines.append(
        f"- [x] Window upper bound: `trading_day < {RETROSPECTIVE_TO}` enforced in SQL — no data >= 2026-04-18 queried."
    )
    lines.append(
        f"- [x] Window lower bound: `trading_day >= {RETROSPECTIVE_FROM}` enforced — Mode A sacred boundary respected."
    )
    lines.append("- [x] Output written once; script refuses re-run if output md exists.")
    lines.append("- [x] No threshold tuning; `overnight_range_pct >= 80` locked from pre-reg.")
    lines.append("- [x] No gate thresholds overridden; gate values inherited from pre-reg verbatim.")
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_md}")

    # Write CSV (init with retrospective on-signal fires)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trading_day",
                "window_label",
                "overnight_range_pct",
                "on_signal",
                "entry_price",
                "stop_price",
                "target_price",
                "direction",
                "pnl_r",
                "outcome",
                "note",
            ]
        )
        for r in on_signal_rows:
            writer.writerow(
                [
                    r[0].isoformat(),
                    "retrospective",
                    f"{r[2]:.2f}" if r[2] is not None else "",
                    1,
                    f"{r[4]:.2f}",
                    f"{r[5]:.2f}",
                    f"{r[6]:.2f}",
                    r[7],
                    f"{r[8]:+.6f}",
                    r[9],
                    "",
                ]
            )
    print(f"Wrote {output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
