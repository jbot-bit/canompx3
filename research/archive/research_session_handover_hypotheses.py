#!/usr/bin/env python3
"""Theory-first session handover hypotheses (pre-registered, low overfit).

Four fixed hypotheses only. No wide threshold search.
All signals are zero-lookahead: leader_break_ts must be <= follower entry_ts.

Outputs:
- research/output/session_handover_hypotheses.csv
- research/output/session_handover_hypotheses.md
"""

from __future__ import annotations

from pathlib import Path
import re
import duckdb
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"


def safe_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", label):
        raise ValueError(f"Unsafe label: {label}")
    return label


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
    }


HYPOTHESES = [
    {
        "id": "H1",
        "name": "Europe FX -> US Data carry",
        "leader_symbol": "M6E",
        "leader_session": "LONDON_OPEN",
        "follower_symbol": "M2K",
        "follower_session": "US_DATA_OPEN",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 1.5,
        "mode": "same_dir_fast",
    },
    {
        "id": "H2",
        "name": "US Data -> US Equity momentum carry",
        "leader_symbol": "MES",
        "leader_session": "US_DATA_OPEN",
        "follower_symbol": "MES",
        "follower_session": "US_EQUITY_OPEN",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 3.0,
        "mode": "same_dir_cont",
    },
    {
        "id": "H3",
        "name": "US Data exhaustion -> US Equity reversal",
        "leader_symbol": "MES",
        "leader_session": "US_DATA_OPEN",
        "follower_symbol": "MES",
        "follower_session": "US_EQUITY_OPEN",
        "entry_model": "E0",
        "confirm_bars": 1,
        "rr_target": 3.0,
        "mode": "opp_dir_late_fail",
    },
    {
        "id": "H4",
        "name": "US Equity stretch -> Post-equity mean reversion",
        "leader_symbol": "M2K",
        "leader_session": "US_EQUITY_OPEN",
        "follower_symbol": "M2K",
        "follower_session": "US_POST_EQUITY",
        "entry_model": "E1",
        "confirm_bars": 5,
        "rr_target": 1.5,
        "mode": "opp_dir_stretch",
    },
]


def load_hypothesis_df(con: duckdb.DuckDBPyConnection, h: dict) -> pd.DataFrame:
    ls = safe_label(h["leader_session"])
    fs = safe_label(h["follower_session"])

    q = f"""
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_ts,
      df_f.orb_{fs}_break_dir AS f_dir,
      df_l.orb_{ls}_break_dir AS l_dir,
      df_l.orb_{ls}_break_ts  AS l_ts,
      df_l.orb_{ls}_break_delay_min AS l_delay,
      df_l.orb_{ls}_break_bar_continues AS l_cont,
      df_l.orb_{ls}_size AS l_size,
      df_l.atr_20 AS l_atr
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON df_f.symbol=o.symbol
     AND df_f.trading_day=o.trading_day
     AND df_f.orb_minutes=o.orb_minutes
    JOIN daily_features df_l
      ON df_l.symbol='{h['leader_symbol']}'
     AND df_l.trading_day=o.trading_day
     AND df_l.orb_minutes=o.orb_minutes
    WHERE o.orb_minutes=5
      AND o.symbol='{h['follower_symbol']}'
      AND o.orb_label='{h['follower_session']}'
      AND o.entry_model='{h['entry_model']}'
      AND o.confirm_bars={h['confirm_bars']}
      AND o.rr_target={h['rr_target']}
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["l_ts"] = pd.to_datetime(df["l_ts"], utc=True)

    # no-lookahead validity
    valid = (
        df["f_dir"].isin(["long", "short"])
        & df["l_dir"].isin(["long", "short"])
        & df["l_ts"].notna()
        & (df["l_ts"] <= df["entry_ts"])
    )
    df = df[valid].copy()

    if not df.empty:
        df["l_size_atr"] = np.where((df["l_atr"].notna()) & (df["l_atr"] > 0), df["l_size"] / df["l_atr"], np.nan)

    return df


def condition_mask(df: pd.DataFrame, mode: str) -> pd.Series:
    same = df["l_dir"] == df["f_dir"]
    opp = ~same

    if mode == "same_dir_fast":
        return same & df["l_delay"].notna() & (df["l_delay"] <= 30)

    if mode == "same_dir_cont":
        return same & (df["l_cont"] == True)

    if mode == "opp_dir_late_fail":
        return opp & df["l_delay"].notna() & (df["l_delay"] >= 60) & (df["l_cont"] == False)

    if mode == "opp_dir_stretch":
        q70 = df["l_size_atr"].quantile(0.70)
        return opp & df["l_size_atr"].notna() & (df["l_size_atr"] >= q70) & df["l_delay"].notna() & (df["l_delay"] <= 30)

    raise ValueError(f"Unknown mode: {mode}")


def evaluate(df: pd.DataFrame, m: pd.Series) -> dict:
    on = df.loc[m, "pnl_r"]
    off = df.loc[~m, "pnl_r"]

    s_on = _stats(on)
    s_off = _stats(off)

    # yearly stability
    yp, yt = 0, 0
    for _, gy in df.groupby("year"):
        my = m.loc[gy.index]
        oy = gy.loc[my, "pnl_r"]
        fy = gy.loc[~my, "pnl_r"]
        if len(oy) < 20 or len(fy) < 20:
            continue
        yt += 1
        if oy.mean() - fy.mean() > 0:
            yp += 1

    # OOS split (train <=2024, test=2025)
    tr = df[df["year"] <= 2024]
    te = df[df["year"] == 2025]

    mtr = m.loc[tr.index] if not tr.empty else pd.Series(dtype=bool)
    mte = m.loc[te.index] if not te.empty else pd.Series(dtype=bool)

    tr_on = tr.loc[mtr, "pnl_r"] if not tr.empty else pd.Series(dtype=float)
    tr_off = tr.loc[~mtr, "pnl_r"] if not tr.empty else pd.Series(dtype=float)
    te_on = te.loc[mte, "pnl_r"] if not te.empty else pd.Series(dtype=float)
    te_off = te.loc[~mte, "pnl_r"] if not te.empty else pd.Series(dtype=float)

    train_uplift = float(tr_on.mean() - tr_off.mean()) if len(tr_on) >= 40 and len(tr_off) >= 40 else np.nan
    test_uplift = float(te_on.mean() - te_off.mean()) if len(te_on) >= 30 and len(te_off) >= 30 else np.nan

    years_cov = max(1, int(df["year"].nunique()))

    return {
        "n_base": int(len(df)),
        "n_on": s_on["n"],
        "signals_per_year": s_on["n"] / years_cov,
        "avg_on": s_on["avg_r"],
        "avg_off": s_off["avg_r"],
        "uplift": s_on["avg_r"] - s_off["avg_r"],
        "wr_on": s_on["wr"],
        "wr_off": s_off["wr"],
        "years_pos": yp,
        "years_total": yt,
        "years_pos_ratio": (yp / yt) if yt else np.nan,
        "train_uplift": train_uplift,
        "test2025_uplift": test_uplift,
        "n_test_on": int(len(te_on)),
        "n_test_off": int(len(te_off)),
    }


def verdict(row: dict) -> str:
    if (
        row["avg_on"] is not None
        and row["avg_on"] >= 0.12
        and row["uplift"] >= 0.20
        and pd.notna(row["test2025_uplift"])
        and row["test2025_uplift"] > 0
        and row["n_on"] >= 70
    ):
        return "PROMOTE"

    if (
        row["avg_on"] is not None
        and row["avg_on"] > 0
        and row["uplift"] >= 0.10
        and row["n_on"] >= 40
    ):
        return "WATCH"

    return "KILL"


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    rows = []
    for h in HYPOTHESES:
        df = load_hypothesis_df(con, h)
        if df.empty:
            rows.append({**h, "result": "KILL", "reason": "no data"})
            continue

        m = condition_mask(df, h["mode"])
        r = evaluate(df, m)
        v = verdict(r)
        rows.append({**h, **r, "result": v})

    con.close()

    out = pd.DataFrame(rows)
    out = out.sort_values(["result", "avg_on", "uplift"], ascending=[True, False, False])

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "session_handover_hypotheses.csv"
    p_md = out_dir / "session_handover_hypotheses.md"

    out.to_csv(p_csv, index=False)

    lines = ["# Session Handover Hypotheses", "", "Pre-registered theory-first tests (4 only).", ""]
    for r in out.itertuples(index=False):
        lines.append(
            f"- {r.id} {r.name} [{r.mode}] => {r.result}: avg_on={r.avg_on:+.4f}, uplift={r.uplift:+.4f}, sig/yr={r.signals_per_year:.1f}, years+={r.years_pos}/{r.years_total}, test2025Δ={r.test2025_uplift:+.4f}, N_on={r.n_on}"
        )

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_csv}")
    print(f"Saved: {p_md}")
    print(out.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
