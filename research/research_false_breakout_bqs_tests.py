#!/usr/bin/env python3
"""Zero-lookahead false-breakout filter tests (fast pass).

Slice tested (current baseline): E1 / CB2 / RR2.5 on orb_minutes=5.
Uses only features knowable at breakout bar close.
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
SYMBOLS = ["MGC", "MES", "M2K", "M6E", "MNQ"]


def _sess_val(row: pd.Series, stem: str):
    return row.get(f"orb_{row['orb_label']}_{stem}", np.nan)


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
    }


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    q = """
    SELECT o.symbol, o.trading_day, o.orb_label, o.pnl_r,
           d.atr_20,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.symbol=o.symbol
     AND d.trading_day=o.trading_day
     AND d.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.entry_model='E1'
      AND o.confirm_bars=2
      AND o.rr_target=2.5
      AND o.pnl_r IS NOT NULL
      AND o.symbol IN ('MGC','MES','M2K','M6E','MNQ')
    """
    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows for test slice.")
        return 0

    # Dynamic session-specific feature extraction
    for stem in ["size", "break_delay_min", "break_bar_continues", "break_bar_volume", "volume", "break_dir"]:
        df[stem] = df.apply(lambda r: _sess_val(r, stem), axis=1)

    # Derived zero-lookahead features
    df["size_atr"] = np.where((df["atr_20"].notna()) & (df["atr_20"] > 0), df["size"] / df["atr_20"], np.nan)
    # session volume is OR window total; normalize to 1-minute proxy
    df["vol_impulse"] = np.where((df["volume"].notna()) & (df["volume"] > 0), df["break_bar_volume"] / (df["volume"] / 5.0), np.nan)
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year

    # Global thresholds (fast pass)
    size_q70 = df["size_atr"].quantile(0.70)
    vol_q60 = df["vol_impulse"].quantile(0.60)

    # Components (all available at/before break bar close)
    df["C_CONT"] = (df["break_bar_continues"] == True)
    df["C_BSP"] = df["break_delay_min"].notna() & (df["break_delay_min"] <= 10)
    df["C_RES"] = df["size_atr"].notna() & (df["size_atr"] >= size_q70)
    df["C_VIS"] = df["vol_impulse"].notna() & (df["vol_impulse"] >= vol_q60)

    df["BQS"] = df[["C_CONT", "C_BSP", "C_RES", "C_VIS"]].sum(axis=1)

    # Rules A-E
    rules = {
        "A_strict_all4": (df["BQS"] == 4),
        "B_score_ge3": (df["BQS"] >= 3),
        "C_momentum": (df["C_BSP"] & df["C_RES"] & df["C_VIS"]),
        "D_veto_bad": ~(~df["C_CONT"] | ((df["break_delay_min"] > 30) & (~df["C_VIS"]))),
        "E_asym_long_ge3": (df["break_dir"] == "long") & (df["BQS"] >= 3),
        "E_asym_short_ge3": (df["break_dir"] == "short") & (df["BQS"] >= 3),
    }

    base = df["pnl_r"]
    rows = []
    for name, m in rules.items():
        on = df.loc[m, "pnl_r"]
        off = df.loc[~m, "pnl_r"]
        if len(on) < 500 or len(off) < 500:
            continue

        s_on = _stats(on)
        s_off = _stats(off)

        # yearly uplift stability
        yp = 0
        yt = 0
        for y, g in df.groupby("year"):
            my = m.loc[g.index]
            oy = g.loc[my, "pnl_r"]
            fy = g.loc[~my, "pnl_r"]
            if len(oy) < 100 or len(fy) < 100:
                continue
            yt += 1
            if oy.mean() - fy.mean() > 0:
                yp += 1

        # quick OOS 2025 if possible
        tr = df[df["year"] <= 2024]
        te = df[df["year"] == 2025]
        mtr = m.loc[tr.index]
        mte = m.loc[te.index]
        tr_on = tr.loc[mtr, "pnl_r"]
        tr_off = tr.loc[~mtr, "pnl_r"]
        te_on = te.loc[mte, "pnl_r"]
        te_off = te.loc[~mte, "pnl_r"]

        tr_up = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 200 and len(tr_off) >= 200 else np.nan
        te_up = float(te_on.mean() - te_off.mean()) if len(te_on) >= 100 and len(te_off) >= 100 else np.nan

        rows.append(
            {
                "rule": name,
                "n_on": s_on["n"],
                "on_rate": s_on["n"] / len(df),
                "avg_on": s_on["avg_r"],
                "avg_off": s_off["avg_r"],
                "uplift_on_off": s_on["avg_r"] - s_off["avg_r"],
                "wr_on": s_on["wr"],
                "wr_off": s_off["wr"],
                "years_pos": yp,
                "years_total": yt,
                "train_uplift": tr_up,
                "test2025_uplift": te_up,
            }
        )

    out = pd.DataFrame(rows).sort_values(["avg_on", "uplift_on_off"], ascending=False)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "false_breakout_bqs_rules.csv"
    p_md = out_dir / "false_breakout_bqs_rules.md"

    if out.empty:
        p_md.write_text("# False-breakout BQS tests\n\nNo rules met sample thresholds.", encoding="utf-8")
        print("No rules met sample thresholds.")
        return 0

    out.to_csv(p_csv, index=False)

    lines = [
        "# Zero-Lookahead False-Breakout Rules (Fast Test)",
        "",
        "Slice: E1/CB2/RR2.5 (5m ORB outcomes)",
        f"Rows tested: {len(df)}",
        f"Global thresholds: size_atr q70={size_q70:.3f}, vol_impulse q60={vol_q60:.3f}",
        "",
        "## Results",
    ]

    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.rule}: N_on={r.n_on}, avg_on={r.avg_on:+.4f}, avg_off={r.avg_off:+.4f}, "
            f"Δ={r.uplift_on_off:+.4f}, WR on/off {r.wr_on:.1%}/{r.wr_off:.1%}, "
            f"years+={r.years_pos}/{r.years_total}, test2025Δ={r.test2025_uplift:+.4f}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
