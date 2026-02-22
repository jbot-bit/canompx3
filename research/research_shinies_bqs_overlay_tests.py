#!/usr/bin/env python3
"""Test BQS false-breakout overlays on current shinies.

Runs for each keeper candidate:
- baseline candidate condition
- baseline + veto_D
- baseline + BQS>=3
- baseline + strict_all4

No-lookahead for lead-lag candidates:
- leader_break_ts <= follower entry_ts
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"

CANDIDATES = [
    {
        "id": "A1",
        "name": "M6E_US_EQUITY_OPEN -> M2K_US_POST_EQUITY | E1/CB5/RR1.5",
        "leader_symbol": "M6E",
        "leader_session": "US_EQUITY_OPEN",
        "follower_symbol": "M2K",
        "follower_session": "US_POST_EQUITY",
        "entry_model": "E1",
        "confirm_bars": 5,
        "rr_target": 1.5,
        "single_asset_fast15": False,
    },
    {
        "id": "A2",
        "name": "MES_US_DATA_OPEN -> M2K_US_DATA_OPEN | E0/CB1/RR1.5",
        "leader_symbol": "MES",
        "leader_session": "US_DATA_OPEN",
        "follower_symbol": "M2K",
        "follower_session": "US_DATA_OPEN",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 1.5,
        "single_asset_fast15": False,
    },
    {
        "id": "A3",
        "name": "MES_1000 -> M2K_US_POST_EQUITY | E1/CB5/RR1.5",
        "leader_symbol": "MES",
        "leader_session": "1000",
        "follower_symbol": "M2K",
        "follower_session": "US_POST_EQUITY",
        "entry_model": "E1",
        "confirm_bars": 5,
        "rr_target": 1.5,
        "single_asset_fast15": False,
    },
    {
        "id": "B1",
        "name": "M2K_1000 -> MES_1000 | E0/CB1/RR2.5",
        "leader_symbol": "M2K",
        "leader_session": "1000",
        "follower_symbol": "MES",
        "follower_session": "1000",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 2.5,
        "single_asset_fast15": False,
    },
    {
        "id": "B2",
        "name": "MES_1000 fast<=15 | E0/CB1/RR2.5",
        "leader_symbol": None,
        "leader_session": None,
        "follower_symbol": "MES",
        "follower_session": "1000",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 2.5,
        "single_asset_fast15": True,
    },
]


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan, "signals_per_year": np.nan}
    yrs = max(1, s.index.get_level_values("year").nunique())
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
        "signals_per_year": float(len(s) / yrs),
    }


def _safe_col(session: str, stem: str) -> str:
    return f"orb_{session}_{stem}"


def _load_candidate_df(con: duckdb.DuckDBPyConnection, c: dict) -> pd.DataFrame:
    fs = c["follower_session"]
    fs_dir = _safe_col(fs, "break_dir")
    fs_delay = _safe_col(fs, "break_delay_min")
    fs_cont = _safe_col(fs, "break_bar_continues")
    fs_size = _safe_col(fs, "size")
    fs_vol = _safe_col(fs, "volume")
    fs_bvol = _safe_col(fs, "break_bar_volume")

    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_ts,
      d.atr_20,
      d.{fs_dir} AS f_dir,
      d.{fs_delay} AS f_delay,
      d.{fs_cont} AS f_cont,
      d.{fs_size} AS f_size,
      d.{fs_vol} AS f_vol,
      d.{fs_bvol} AS f_bvol
    FROM orb_outcomes o
    JOIN daily_features d
      ON d.symbol=o.symbol
     AND d.trading_day=o.trading_day
     AND d.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='{c['follower_symbol']}'
      AND o.orb_label='{c['follower_session']}'
      AND o.entry_model='{c['entry_model']}'
      AND o.confirm_bars={c['confirm_bars']}
      AND o.rr_target={c['rr_target']}
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    if c["leader_symbol"] is not None:
        ls = c["leader_session"]
        ls_dir = _safe_col(ls, "break_dir")
        ls_ts = _safe_col(ls, "break_ts")

        ql = f"""
        SELECT trading_day,
               {ls_dir} AS l_dir,
               {ls_ts}  AS l_ts
        FROM daily_features
        WHERE symbol='{c['leader_symbol']}'
          AND orb_minutes=5
        """
        ld = con.execute(ql).fetchdf()
        ld["trading_day"] = pd.to_datetime(ld["trading_day"])
        ld["l_ts"] = pd.to_datetime(ld["l_ts"], utc=True)
        df = df.merge(ld, on="trading_day", how="left")

    return df


def _base_mask(df: pd.DataFrame, c: dict) -> pd.Series:
    if c["single_asset_fast15"]:
        return df["f_delay"].notna() & (df["f_delay"] <= 15)

    return (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & (df["f_dir"] == df["l_dir"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )


def _bqs_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    size_atr = np.where((df["atr_20"].notna()) & (df["atr_20"] > 0), df["f_size"] / df["atr_20"], np.nan)
    vol_imp = np.where((df["f_vol"].notna()) & (df["f_vol"] > 0), df["f_bvol"] / (df["f_vol"] / 5.0), np.nan)

    size_q70 = pd.Series(size_atr).quantile(0.70)
    vol_q60 = pd.Series(vol_imp).quantile(0.60)

    c_cont = (df["f_cont"] == True)
    c_bsp = df["f_delay"].notna() & (df["f_delay"] <= 10)
    c_res = pd.Series(size_atr).notna() & (pd.Series(size_atr) >= size_q70)
    c_vis = pd.Series(vol_imp).notna() & (pd.Series(vol_imp) >= vol_q60)

    bqs = c_cont.astype(int) + c_bsp.astype(int) + c_res.astype(int) + c_vis.astype(int)

    veto_d = ~(~c_cont | ((df["f_delay"] > 30) & (~c_vis)))

    return {
        "baseline": pd.Series(True, index=df.index),
        "veto_D": veto_d,
        "bqs_ge3": (bqs >= 3),
        "strict_all4": (bqs == 4),
    }


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    summary_rows = []
    detail_rows = []

    for c in CANDIDATES:
        df = _load_candidate_df(con, c)
        if df.empty:
            continue

        base = _base_mask(df, c)
        dfb = df[base].copy()
        if dfb.empty:
            continue

        masks = _bqs_masks(dfb)

        # MultiIndex with year for signals/year calc
        dfb = dfb.set_index([dfb.index, "year"])  # keep original row id + year
        pnl = dfb["pnl_r"]

        base_stats = _stats(pnl)

        for variant, vm in masks.items():
            vm = vm.loc[dfb.index.get_level_values(0)]
            on = pnl[vm.values]
            st = _stats(on)

            # OOS delta vs baseline in 2025
            test = dfb[dfb.index.get_level_values("year") == 2025]
            if not test.empty:
                vm_test = vm.loc[test.index.get_level_values(0)]
                test_on = test[vm_test.values]["pnl_r"]
                test_base = test["pnl_r"]
                test_delta = float(test_on.mean() - test_base.mean()) if len(test_on) >= 20 else np.nan
            else:
                test_delta = np.nan

            summary_rows.append(
                {
                    "candidate_id": c["id"],
                    "candidate": c["name"],
                    "variant": variant,
                    "n": st["n"],
                    "signals_per_year": st["signals_per_year"],
                    "wr": st["wr"],
                    "avg_r": st["avg_r"],
                    "total_r": st["total_r"],
                    "delta_avg_r_vs_base": (st["avg_r"] - base_stats["avg_r"]) if variant != "baseline" else 0.0,
                    "delta_test2025_vs_base": test_delta if variant != "baseline" else 0.0,
                }
            )

            for y, gy in dfb.groupby(level="year"):
                vmy = vm.loc[gy.index.get_level_values(0)]
                oy = gy[vmy.values]["pnl_r"]
                if len(oy) == 0:
                    continue
                detail_rows.append(
                    {
                        "candidate_id": c["id"],
                        "variant": variant,
                        "year": int(y),
                        "n": int(len(oy)),
                        "avg_r": float(oy.mean()),
                        "wr": float((oy > 0).mean()),
                    }
                )

    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_sum = out_dir / "shinies_bqs_overlay_summary.csv"
    p_year = out_dir / "shinies_bqs_overlay_yearly.csv"
    p_md = out_dir / "shinies_bqs_overlay_notes.md"

    s = pd.DataFrame(summary_rows)
    y = pd.DataFrame(detail_rows)

    if s.empty:
        p_md.write_text("# BQS overlay tests\n\nNo rows produced.", encoding="utf-8")
        print("No rows produced.")
        return 0

    s = s.sort_values(["candidate_id", "variant"])
    s.to_csv(p_sum, index=False)
    y.to_csv(p_year, index=False)

    lines = ["# BQS Overlay Tests on Shinies", ""]
    for cid, g in s.groupby("candidate_id"):
        lines.append(f"## {cid}")
        for r in g.itertuples(index=False):
            lines.append(
                f"- {r.variant}: N={r.n}, sig/yr={r.signals_per_year:.1f}, avgR={r.avg_r:+.4f}, "
                f"Δavg={r.delta_avg_r_vs_base:+.4f}, test2025Δ={r.delta_test2025_vs_base:+.4f}"
            )
        lines.append("")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_year}")
    print(f"Saved: {p_md}")
    print("\nSummary:")
    print(s.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
