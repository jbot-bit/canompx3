#!/usr/bin/env python3
"""Dalton acceptance filter ON/OFF uplift at anchor level (no-lookahead).

Why this exists:
- Per-strategy-cell filtering can be too sparse.
- This script measures whether Dalton acceptance helps at practical aggregate level:
  symbol x anchor.

Outputs:
- research/output/dalton_filter_anchor_uplift.csv
- research/output/dalton_filter_anchor_uplift.md
"""

from __future__ import annotations

from pathlib import Path
import sys
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.research_dalton_filter_uplift import build_filter_flags


def _max_dd(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    c = s.cumsum()
    p = c.cummax()
    d = p - c
    return float(d.max())


def main() -> int:
    con = duckdb.connect("gold.db", read_only=True)

    flags = build_filter_flags(con)
    if flags.empty:
        print("No Dalton flags built.")
        return 0

    outcomes = con.execute(
        """
        SELECT symbol, trading_day, orb_label AS anchor, pnl_r, entry_ts
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND pnl_r IS NOT NULL
          AND entry_ts IS NOT NULL
          AND symbol IN ('MGC','MES','MNQ')
          AND orb_label IN ('0900','1000','1100')
        """
    ).fetchdf()
    con.close()

    outcomes["entry_ts"] = pd.to_datetime(outcomes["entry_ts"], utc=True)
    m = outcomes.merge(flags, on=["symbol", "trading_day", "anchor"], how="inner")

    # NO-LOOKAHEAD: only trades whose entry happened after B close gate
    m = m[m["entry_ts"] >= m["gate_ts"]].copy()

    rows = []
    for (sym, anch), g in m.groupby(["symbol", "anchor"]):
        g = g.sort_values(["trading_day", "entry_ts"])
        on = g[g["dalton_accept"] == 1]
        off = g[g["dalton_accept"] == 0]

        n_all = len(g)
        n_on = len(on)
        n_off = len(off)
        if n_all < 500 or n_on < 30:
            continue

        rows.append(
            {
                "symbol": sym,
                "anchor": anch,
                "n_all": n_all,
                "n_on": n_on,
                "on_rate": n_on / n_all,
                "avg_r_all": float(g["pnl_r"].mean()),
                "avg_r_on": float(on["pnl_r"].mean()),
                "avg_r_off": float(off["pnl_r"].mean()),
                "uplift_on_vs_all": float(on["pnl_r"].mean() - g["pnl_r"].mean()),
                "uplift_on_vs_off": float(on["pnl_r"].mean() - off["pnl_r"].mean()),
                "wr_all": float((g["pnl_r"] > 0).mean()),
                "wr_on": float((on["pnl_r"] > 0).mean()),
                "wr_off": float((off["pnl_r"] > 0).mean()),
                "max_dd_all": _max_dd(g["pnl_r"]),
                "max_dd_on": _max_dd(on["pnl_r"]),
                "dd_delta_on_minus_all": _max_dd(on["pnl_r"]) - _max_dd(g["pnl_r"]),
            }
        )

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dalton_filter_anchor_uplift.csv"
    md_path = out_dir / "dalton_filter_anchor_uplift.md"

    if not rows:
        md_path.write_text("# Dalton filter anchor uplift\n\nNo anchor-level rows met thresholds.", encoding="utf-8")
        print("No rows.")
        return 0

    df = pd.DataFrame(rows).sort_values("uplift_on_vs_off", ascending=False)
    df.to_csv(csv_path, index=False)

    lines = [
        "# Dalton filter ON/OFF uplift (anchor-level)",
        "",
        "No-lookahead applied (entry_ts >= A/B gate_ts).",
        "",
    ]

    for r in df.itertuples(index=False):
        lines.append(
            f"- {r.symbol} {r.anchor}: N={r.n_all}, ON={r.n_on} ({r.on_rate:.1%}), "
            f"avgR all {r.avg_r_all:+.4f}, ON {r.avg_r_on:+.4f}, OFF {r.avg_r_off:+.4f}, "
            f"Δ(on-off)={r.uplift_on_vs_off:+.4f}, WR on/off {r.wr_on:.1%}/{r.wr_off:.1%}, "
            f"DD all/on {r.max_dd_all:.2f}/{r.max_dd_on:.2f}"
        )

    keep = df[(df["uplift_on_vs_off"] > 0) & (df["dd_delta_on_minus_all"] <= 0)]
    lines.append("")
    lines.append(f"Candidate keepers (uplift>0 and DD not worse): {len(keep)}")
    for r in keep.itertuples(index=False):
        lines.append(f"- {r.symbol} {r.anchor}: Δ={r.uplift_on_vs_off:+.4f}, ΔDD={r.dd_delta_on_minus_all:+.2f}")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
