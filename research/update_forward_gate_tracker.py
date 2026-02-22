#!/usr/bin/env python3
"""Update forward gate tracker using current 2026 forward window.

Forward window: trading_day >= 2026-01-01
Applies frozen presets per strategy id.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import duckdb
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "gold.db"
REG_PATH = ROOT / "research" / "output" / "shinies_registry.csv"
TRACKER_PATH = ROOT / "research" / "output" / "forward_gate_tracker.csv"
STATUS_LATEST_PATH = ROOT / "research" / "output" / "forward_gate_status_latest.md"
STATUS_SNAPSHOTS_DIR = ROOT / "research" / "output"

# frozen presets
PRESET = {
    "A0": "base_plus_both",
    "A1": "base",
    "A2": "base_plus_both",
    "A3": "base_plus_both",
    "B1": "base_plus_both",
    "B2": "base_plus_vol60",
}


def parse_tag(tag: str):
    if not isinstance(tag, str) or "_" not in tag:
        return None
    a, b = tag.split("_", 1)
    if re.fullmatch(r"[A-Za-z0-9]+", a) is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", b) is None:
        return None
    return a, b


def max_dd(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    c = series.cumsum()
    p = c.cummax()
    d = p - c
    return float(d.max())


def load_df(con, row):
    fsym, fsess = parse_tag(str(row["follower"]))
    fs = fsess
    leader_tag = str(row["leader"])
    if str(row.get("id", "")) == "B2" or leader_tag.endswith("fast_le_15"):
        leader = None
    else:
        leader = parse_tag(leader_tag)

    if leader is not None:
        lsym, lsess = leader
        q = f"""
        SELECT o.trading_day,o.pnl_r,o.entry_ts,
               d_f.orb_{fs}_break_dir AS f_dir,
               d_f.orb_{fs}_break_delay_min AS f_delay,
               d_f.orb_{fs}_break_bar_continues AS f_cont,
               d_f.orb_{fs}_size AS f_size,
               d_f.orb_{fs}_volume AS f_vol,
               d_f.orb_{fs}_break_bar_volume AS f_bvol,
               d_f.atr_20 AS f_atr,
               d_l.orb_{lsess}_break_dir AS l_dir,
               d_l.orb_{lsess}_break_ts  AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        JOIN daily_features d_l ON d_l.symbol='{lsym}' AND d_l.trading_day=o.trading_day AND d_l.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}' AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}' AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
        """
    else:
        q = f"""
        SELECT o.trading_day,o.pnl_r,o.entry_ts,
               d_f.orb_{fs}_break_dir AS f_dir,
               d_f.orb_{fs}_break_delay_min AS f_delay,
               d_f.orb_{fs}_break_bar_continues AS f_cont,
               d_f.orb_{fs}_size AS f_size,
               d_f.orb_{fs}_volume AS f_vol,
               d_f.orb_{fs}_break_bar_volume AS f_bvol,
               d_f.atr_20 AS f_atr,
               NULL::VARCHAR AS l_dir,
               NULL::TIMESTAMPTZ AS l_ts
        FROM orb_outcomes o
        JOIN daily_features d_f ON d_f.symbol=o.symbol AND d_f.trading_day=o.trading_day AND d_f.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}' AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}' AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL AND o.entry_ts IS NOT NULL
        """
    df = con.execute(q).fetchdf()
    if df.empty:
        return df
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    if "l_ts" in df.columns:
        df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)
    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)
    return df


def base_mask(df, sid, leader_tag):
    if sid == "B2" or str(leader_tag).endswith("fast_le_15"):
        return df["f_delay"].notna() & (df["f_delay"] <= 15)
    return (
        df["f_dir"].isin(["long", "short"]) &
        df["l_dir"].isin(["long", "short"]) &
        (df["f_dir"] == df["l_dir"]) &
        df["l_ts"].notna() &
        (df["l_ts"] <= df["entry_ts"])
    )


def preset_mask(dfb, preset):
    vq = dfb["f_vol_imp"].quantile(0.60)
    if preset == "base":
        return pd.Series(True, index=dfb.index)
    if preset == "base_plus_fast15":
        return dfb["f_delay"].notna() & (dfb["f_delay"] <= 15)
    if preset == "base_plus_vol60":
        return dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vq)
    if preset == "base_plus_both":
        return (dfb["f_delay"].notna() & (dfb["f_delay"] <= 15) & dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= vq))
    return pd.Series(True, index=dfb.index)


def main():
    reg = pd.read_csv(REG_PATH)
    reg = reg[reg["status"] == "KEEP"].copy()
    tracker = pd.read_csv(TRACKER_PATH)
    if "decision" in tracker.columns:
        tracker["decision"] = tracker["decision"].astype("string")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    for i, t in tracker.iterrows():
        sid = t["id"]
        row = reg[reg["id"] == sid]
        if row.empty:
            continue
        r = row.iloc[0]

        df = load_df(con, r)
        if df.empty:
            continue

        bm = base_mask(df, sid, r["leader"])
        dfb = df[bm].copy()
        if dfb.empty:
            continue

        # forward window
        fw = dfb[dfb["trading_day"] >= pd.Timestamp("2026-01-01")].copy()
        if fw.empty:
            continue

        pm = preset_mask(fw, PRESET.get(sid, "base"))
        on = fw.loc[pm, "pnl_r"].sort_index()
        base = fw["pnl_r"].sort_index()

        n_on = int(len(on))
        avg_on = float(on.mean()) if n_on else np.nan
        uplift = float(on.mean() - base.mean()) if n_on else np.nan

        dd_base = max_dd(base)
        dd_on = max_dd(on) if n_on else np.nan
        dd_delta_pct = ((dd_on - dd_base) / dd_base * 100.0) if dd_base > 0 and pd.notna(dd_on) else np.nan

        target_n = int(t["target_n"])

        decision = "PENDING"
        if n_on >= target_n:
            pass_all = (
                pd.notna(avg_on) and avg_on > 0 and
                pd.notna(uplift) and uplift > 0 and
                (pd.isna(dd_delta_pct) or dd_delta_pct <= 10)
            )
            decision = "PROMOTE" if pass_all else "KILL"

        tracker.loc[i, "forward_n"] = n_on
        tracker.loc[i, "forward_avg_r"] = round(avg_on, 6) if pd.notna(avg_on) else np.nan
        tracker.loc[i, "forward_uplift"] = round(uplift, 6) if pd.notna(uplift) else np.nan
        tracker.loc[i, "forward_dd_delta_pct"] = round(dd_delta_pct, 3) if pd.notna(dd_delta_pct) else np.nan
        tracker.loc[i, "decision"] = decision

    con.close()

    tracker.to_csv(TRACKER_PATH, index=False)

    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date_tag = now.strftime("%Y-%m-%d")

    lines = []
    lines.append("# Forward Gate Status (latest)")
    lines.append("")
    lines.append(f"Updated: {stamp}")
    lines.append("")

    pending = tracker[tracker["decision"] == "PENDING"]
    promote = tracker[tracker["decision"] == "PROMOTE"]
    kill = tracker[tracker["decision"] == "KILL"]
    lines.append(f"- Pending: {len(pending)}")
    lines.append(f"- Promote: {len(promote)}")
    lines.append(f"- Kill: {len(kill)}")
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("| id | n/target | forward_avg_r | forward_uplift | dd_delta_pct | decision |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for _, r in tracker.iterrows():
        n_target = f"{int(r['forward_n'])}/{int(r['target_n'])}"
        favg = "" if pd.isna(r["forward_avg_r"]) else f"{float(r['forward_avg_r']):+.4f}"
        fupl = "" if pd.isna(r["forward_uplift"]) else f"{float(r['forward_uplift']):+.4f}"
        fdd = "" if pd.isna(r["forward_dd_delta_pct"]) else f"{float(r['forward_dd_delta_pct']):+.2f}%"
        lines.append(f"| {r['id']} | {n_target} | {favg} | {fupl} | {fdd} | {r['decision']} |")

    STATUS_LATEST_PATH.write_text("\n".join(lines), encoding="utf-8")

    snap_path = STATUS_SNAPSHOTS_DIR / f"forward_gate_status_{date_tag}.md"
    snap_path.write_text("\n".join(lines), encoding="utf-8")

    print(tracker.to_string(index=False))
    print(f"\nWrote: {STATUS_LATEST_PATH}")
    print(f"Wrote: {snap_path}")


if __name__ == "__main__":
    main()
