#!/usr/bin/env python3
"""Find add-on overlays that improve the current KEEP strategy set.

Goal:
- Identify practical filters that can be layered onto existing validated strategies.
- Evaluate per-strategy and cross-strategy (how many improved).

No-lookahead maintained in all base conditions.
"""

from __future__ import annotations

from pathlib import Path
import re
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"
REG_PATH = PROJECT_ROOT / "research" / "output" / "shinies_registry.csv"


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def parse_pair_tag(tag: str) -> tuple[str, str] | None:
    if not isinstance(tag, str) or "_" not in tag:
        return None
    a, b = tag.split("_", 1)
    if not a or not b:
        return None
    if re.fullmatch(r"[A-Za-z0-9]+", a) is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", b) is None:
        return None
    return a, b


def load_strategy_rows() -> pd.DataFrame:
    reg = pd.read_csv(REG_PATH)
    reg = reg[reg["status"] == "KEEP"].copy()
    return reg


def load_strategy_df(con: duckdb.DuckDBPyConnection, row: pd.Series) -> pd.DataFrame:
    follower = parse_pair_tag(str(row["follower"]))
    if follower is None:
        return pd.DataFrame()
    fsym, fsess = follower
    fs = safe_label(fsess)

    leader_tag = str(row["leader"])
    # Special non-lead-lag rows (e.g., B2 marker)
    if str(row.get("id", "")) == "B2" or leader_tag.endswith("fast_le_15"):
        leader = None
    else:
        leader = parse_pair_tag(leader_tag)

    cols_f = {
        "f_dir": f"orb_{fs}_break_dir",
        "f_ts": f"orb_{fs}_break_ts",
        "f_delay": f"orb_{fs}_break_delay_min",
        "f_cont": f"orb_{fs}_break_bar_continues",
        "f_size": f"orb_{fs}_size",
        "f_vol": f"orb_{fs}_volume",
        "f_bvol": f"orb_{fs}_break_bar_volume",
    }

    if leader is not None:
        lsym, lsess = leader
        ls = safe_label(lsess)
        l_cols = {
            "l_dir": f"orb_{ls}_break_dir",
            "l_ts": f"orb_{ls}_break_ts",
            "l_delay": f"orb_{ls}_break_delay_min",
            "l_cont": f"orb_{ls}_break_bar_continues",
            "l_size": f"orb_{ls}_size",
            "l_vol": f"orb_{ls}_volume",
            "l_bvol": f"orb_{ls}_break_bar_volume",
        }

        q = f"""
        SELECT
          o.trading_day,
          o.pnl_r,
          o.entry_ts,
          d_f.atr_20 AS f_atr,
          d_l.atr_20 AS l_atr,
          d_f.{cols_f['f_dir']} AS f_dir,
          d_f.{cols_f['f_ts']} AS f_ts,
          d_f.{cols_f['f_delay']} AS f_delay,
          d_f.{cols_f['f_cont']} AS f_cont,
          d_f.{cols_f['f_size']} AS f_size,
          d_f.{cols_f['f_vol']} AS f_vol,
          d_f.{cols_f['f_bvol']} AS f_bvol,
          d_l.{l_cols['l_dir']} AS l_dir,
          d_l.{l_cols['l_ts']} AS l_ts,
          d_l.{l_cols['l_delay']} AS l_delay,
          d_l.{l_cols['l_cont']} AS l_cont,
          d_l.{l_cols['l_size']} AS l_size,
          d_l.{l_cols['l_vol']} AS l_vol,
          d_l.{l_cols['l_bvol']} AS l_bvol
        FROM orb_outcomes o
        JOIN daily_features d_f
          ON d_f.symbol=o.symbol
         AND d_f.trading_day=o.trading_day
         AND d_f.orb_minutes=o.orb_minutes
        JOIN daily_features d_l
          ON d_l.symbol='{lsym}'
         AND d_l.trading_day=o.trading_day
         AND d_l.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}'
          AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """
    else:
        q = f"""
        SELECT
          o.trading_day,
          o.pnl_r,
          o.entry_ts,
          d_f.atr_20 AS f_atr,
          NULL::DOUBLE AS l_atr,
          d_f.{cols_f['f_dir']} AS f_dir,
          d_f.{cols_f['f_ts']} AS f_ts,
          d_f.{cols_f['f_delay']} AS f_delay,
          d_f.{cols_f['f_cont']} AS f_cont,
          d_f.{cols_f['f_size']} AS f_size,
          d_f.{cols_f['f_vol']} AS f_vol,
          d_f.{cols_f['f_bvol']} AS f_bvol,
          NULL::VARCHAR AS l_dir,
          NULL::TIMESTAMPTZ AS l_ts,
          NULL::DOUBLE AS l_delay,
          NULL::BOOLEAN AS l_cont,
          NULL::DOUBLE AS l_size,
          NULL::DOUBLE AS l_vol,
          NULL::DOUBLE AS l_bvol
        FROM orb_outcomes o
        JOIN daily_features d_f
          ON d_f.symbol=o.symbol
         AND d_f.trading_day=o.trading_day
         AND d_f.orb_minutes=o.orb_minutes
        WHERE o.orb_minutes=5
          AND o.symbol='{fsym}'
          AND o.orb_label='{fsess}'
          AND o.entry_model='{row['entry_model']}'
          AND o.confirm_bars={int(row['confirm_bars'])}
          AND o.rr_target={float(row['rr_target'])}
          AND o.pnl_r IS NOT NULL
          AND o.entry_ts IS NOT NULL
        """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["f_ts"] = pd.to_datetime(df["f_ts"], utc=True)
    if "l_ts" in df.columns:
        df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    # derived
    df["f_size_atr"] = np.where((df["f_atr"].notna()) & (df["f_atr"] > 0), df["f_size"] / df["f_atr"], np.nan)
    df["f_vol_imp"] = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)
    df["l_size_atr"] = np.where((df["l_atr"].notna()) & (df["l_atr"] > 0), df["l_size"] / df["l_atr"], np.nan)

    return df


def base_mask(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    # B2 special single-asset baseline
    if str(row["id"]) == "B2" or str(row["leader"]).endswith("fast_le_15"):
        return df["f_delay"].notna() & (df["f_delay"] <= 15)

    # default lead-lag same-dir no-lookahead
    return (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )


def evaluate_overlay(dfb: pd.DataFrame, m: pd.Series) -> dict:
    base = dfb["pnl_r"]
    on = dfb.loc[m, "pnl_r"]
    off = dfb.loc[~m, "pnl_r"]

    if len(on) < 25 or len(off) < 25:
        return {}

    # OOS delta vs base in 2025
    te = dfb[dfb["year"] == 2025]
    test_delta = np.nan
    if not te.empty:
        mt = m.loc[te.index]
        te_on = te.loc[mt, "pnl_r"]
        if len(te_on) >= 15:
            test_delta = float(te_on.mean() - te["pnl_r"].mean())

    return {
        "n_base": int(len(base)),
        "n_on": int(len(on)),
        "retain": float(len(on) / len(base)) if len(base) else np.nan,
        "avg_base": float(base.mean()),
        "avg_on": float(on.mean()),
        "delta_avg": float(on.mean() - base.mean()),
        "wr_base": float((base > 0).mean()),
        "wr_on": float((on > 0).mean()),
        "delta_wr": float((on > 0).mean() - (base > 0).mean()),
        "test2025_delta_vs_base": test_delta,
    }


def main() -> int:
    reg = load_strategy_rows()
    con = duckdb.connect(DB_PATH, read_only=True)

    detail_rows = []

    for _, r in reg.iterrows():
        df = load_strategy_df(con, r)
        if df.empty:
            continue

        bm = base_mask(df, r)
        dfb = df[bm].copy()
        if len(dfb) < 80:
            continue

        # local quantiles per strategy baseline
        f_size_q60 = dfb["f_size_atr"].quantile(0.60)
        f_vol_q60 = dfb["f_vol_imp"].quantile(0.60)
        l_size_q60 = dfb["l_size_atr"].quantile(0.60)

        overlays = {
            "OV_f_cont": (dfb["f_cont"] == True),
            "OV_f_fast30": dfb["f_delay"].notna() & (dfb["f_delay"] <= 30),
            "OV_f_fast15": dfb["f_delay"].notna() & (dfb["f_delay"] <= 15),
            "OV_f_size60": dfb["f_size_atr"].notna() & (dfb["f_size_atr"] >= f_size_q60),
            "OV_f_vol60": dfb["f_vol_imp"].notna() & (dfb["f_vol_imp"] >= f_vol_q60),
            "OV_f_cont_fast30": (dfb["f_cont"] == True) & dfb["f_delay"].notna() & (dfb["f_delay"] <= 30),
            "OV_f_fast30_size60": dfb["f_delay"].notna() & (dfb["f_delay"] <= 30) & dfb["f_size_atr"].notna() & (dfb["f_size_atr"] >= f_size_q60),
        }

        if dfb["l_dir"].notna().any():
            overlays.update(
                {
                    "OV_l_cont": (dfb["l_cont"] == True),
                    "OV_l_fast30": dfb["l_delay"].notna() & (dfb["l_delay"] <= 30),
                    "OV_l_size60": dfb["l_size_atr"].notna() & (dfb["l_size_atr"] >= l_size_q60),
                    "OV_l_cont_fast30": (dfb["l_cont"] == True) & dfb["l_delay"].notna() & (dfb["l_delay"] <= 30),
                }
            )

        for oname, omask in overlays.items():
            ev = evaluate_overlay(dfb, omask)
            if not ev:
                continue
            detail_rows.append(
                {
                    "strategy_id": r["id"],
                    "strategy": f"{r['leader']} -> {r['follower']} {r['entry_model']}/CB{int(r['confirm_bars'])}/RR{float(r['rr_target'])}",
                    "overlay": oname,
                    **ev,
                }
            )

    con.close()

    detail = pd.DataFrame(detail_rows)

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_detail = out_dir / "shinies_overlay_detail.csv"
    p_rank = out_dir / "shinies_overlay_rank.csv"
    p_md = out_dir / "shinies_overlay_rank.md"

    if detail.empty:
        p_md.write_text("# Shinies overlay rank\n\nNo rows.", encoding="utf-8")
        print("No rows.")
        return 0

    detail.to_csv(p_detail, index=False)

    # cross-strategy rank
    rank = (
        detail.groupby("overlay", as_index=False)
        .agg(
            strategies=("strategy_id", "nunique"),
            improved_count=("delta_avg", lambda s: int((s > 0).sum())),
            strong_count=("delta_avg", lambda s: int((s > 0.02).sum())),
            avg_delta=("delta_avg", "mean"),
            median_retain=("retain", "median"),
            avg_test_delta=("test2025_delta_vs_base", "mean"),
        )
    )
    rank["improve_ratio"] = rank["improved_count"] / rank["strategies"]
    rank = rank.sort_values(["improve_ratio", "avg_delta", "median_retain"], ascending=[False, False, False])
    rank.to_csv(p_rank, index=False)

    lines = ["# Overlay candidates across KEEP strategies", ""]
    lines.append("Top overlays by how many strategies they improved (delta_avg > 0):")
    for r in rank.head(12).itertuples(index=False):
        lines.append(
            f"- {r.overlay}: improved {r.improved_count}/{r.strategies}, avgΔ={r.avg_delta:+.4f}, median retain={r.median_retain:.2f}, avg testΔ={r.avg_test_delta:+.4f}"
        )

    lines.append("")
    lines.append("Best per strategy:")
    best = detail.sort_values(["strategy_id", "delta_avg"], ascending=[True, False]).groupby("strategy_id", as_index=False).first()
    for r in best.itertuples(index=False):
        lines.append(
            f"- {r.strategy_id}: {r.overlay}, Δ={r.delta_avg:+.4f}, retain={r.retain:.2f}, testΔ={r.test2025_delta_vs_base:+.4f}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_detail}")
    print(f"Saved: {p_rank}")
    print(f"Saved: {p_md}")
    print("\nTop overlay rank:")
    print(rank.head(20).to_string(index=False))
    print("\nBest per strategy:")
    print(best.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
