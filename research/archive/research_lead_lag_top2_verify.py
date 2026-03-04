#!/usr/bin/env python3
"""Deep verification for top-2 lead-lag candidates (no-lookahead).

Top-2 pairs from fast screen:
1) MES_0900 -> MNQ_1000
2) MNQ_0900 -> MES_1000

Verification upgrades vs fast screen:
- Uses leader break_ts and follower entry_ts
- Requires leader signal known before follower entry (leader_break_ts <= entry_ts)
- Computes aggregate, year-by-year, and rolling OOS uplift

Outputs:
- research/output/lead_lag_top2_verify_summary.csv
- research/output/lead_lag_top2_verify_yearly.csv
- research/output/lead_lag_top2_verify_oos.csv
- research/output/lead_lag_top2_verify.md
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"

PAIR_SPECS = [
    {
        "name": "MES_0900_to_MNQ_1000",
        "leader_symbol": "MES",
        "leader_session": "0900",
        "follower_symbol": "MNQ",
        "follower_session": "1000",
    },
    {
        "name": "MNQ_0900_to_MES_1000",
        "leader_symbol": "MNQ",
        "leader_session": "0900",
        "follower_symbol": "MES",
        "follower_session": "1000",
    },
]

ENTRY_MODEL = "E1"
CONFIRM_BARS = 2
RR_TARGET = 2.5


def _stats(s: pd.Series) -> dict:
    if s.empty:
        return {"n": 0, "wr": np.nan, "avg_r": np.nan, "total_r": np.nan}
    return {
        "n": int(len(s)),
        "wr": float((s > 0).mean()),
        "avg_r": float(s.mean()),
        "total_r": float(s.sum()),
    }


def _load_pair_df(con: duckdb.DuckDBPyConnection, spec: dict) -> pd.DataFrame:
    ls = spec["leader_session"]
    fs = spec["follower_session"]

    q = f"""
    SELECT
        o.trading_day,
        o.symbol AS follower_symbol,
        o.orb_label AS follower_session,
        o.pnl_r,
        o.entry_ts,
        df_f.orb_{fs}_break_dir AS follower_break_dir,
        df_l.orb_{ls}_break_dir AS leader_break_dir,
        df_l.orb_{ls}_break_ts  AS leader_break_ts
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON o.symbol = df_f.symbol
     AND o.trading_day = df_f.trading_day
     AND o.orb_minutes = df_f.orb_minutes
    JOIN daily_features df_l
      ON df_l.symbol = '{spec['leader_symbol']}'
     AND df_l.trading_day = o.trading_day
     AND df_l.orb_minutes = o.orb_minutes
    WHERE o.orb_minutes = 5
      AND o.symbol = '{spec['follower_symbol']}'
      AND o.orb_label = '{spec['follower_session']}'
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR_TARGET}
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    if df.empty:
        return df

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["leader_break_ts"] = pd.to_datetime(df["leader_break_ts"], utc=True)

    # Valid signal space only
    df = df[
        df["leader_break_dir"].isin(["long", "short"])
        & df["follower_break_dir"].isin(["long", "short"])
        & df["leader_break_ts"].notna()
    ].copy()

    # No-lookahead guard: leader signal must exist before follower entry
    df = df[df["leader_break_ts"] <= df["entry_ts"]].copy()

    # Condition ON when directions align
    df["cond_on"] = df["leader_break_dir"] == df["follower_break_dir"]

    return df


def _aggregate(spec_name: str, df: pd.DataFrame) -> dict:
    on = df[df["cond_on"]]["pnl_r"]
    off = df[~df["cond_on"]]["pnl_r"]
    on_s = _stats(on)
    off_s = _stats(off)

    return {
        "pair": spec_name,
        "n_total": int(len(df)),
        "n_on": on_s["n"],
        "n_off": off_s["n"],
        "on_rate": float(on_s["n"] / len(df)) if len(df) else np.nan,
        "avg_r_on": on_s["avg_r"],
        "avg_r_off": off_s["avg_r"],
        "uplift_on_minus_off": float(on_s["avg_r"] - off_s["avg_r"]) if on_s["n"] and off_s["n"] else np.nan,
        "wr_on": on_s["wr"],
        "wr_off": off_s["wr"],
    }


def _yearly(spec_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in df.groupby("year"):
        on = g[g["cond_on"]]["pnl_r"]
        off = g[~g["cond_on"]]["pnl_r"]
        if len(on) == 0 or len(off) == 0:
            continue
        rows.append(
            {
                "pair": spec_name,
                "year": int(y),
                "n_on": int(len(on)),
                "n_off": int(len(off)),
                "avg_r_on": float(on.mean()),
                "avg_r_off": float(off.mean()),
                "uplift": float(on.mean() - off.mean()),
                "wr_on": float((on > 0).mean()),
                "wr_off": float((off > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _rolling_oos(spec_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = sorted(df["year"].unique().tolist())
    if len(years) < 3:
        return pd.DataFrame()

    for test_year in years[2:]:  # require at least two train years
        train = df[df["year"] < test_year]
        test = df[df["year"] == test_year]

        tr_on = train[train["cond_on"]]["pnl_r"]
        tr_off = train[~train["cond_on"]]["pnl_r"]
        te_on = test[test["cond_on"]]["pnl_r"]
        te_off = test[~test["cond_on"]]["pnl_r"]

        if len(tr_on) < 80 or len(tr_off) < 80:
            continue
        if len(te_on) < 30 or len(te_off) < 30:
            continue

        rows.append(
            {
                "pair": spec_name,
                "test_year": int(test_year),
                "n_train_on": int(len(tr_on)),
                "n_test_on": int(len(te_on)),
                "train_uplift": float(tr_on.mean() - tr_off.mean()),
                "test_uplift": float(te_on.mean() - te_off.mean()),
                "test_avg_on": float(te_on.mean()),
                "test_wr_on": float((te_on > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    con = duckdb.connect(DB_PATH, read_only=True)

    s_rows = []
    y_parts = []
    o_parts = []

    for spec in PAIR_SPECS:
        df = _load_pair_df(con, spec)
        if df.empty:
            continue

        s_rows.append(_aggregate(spec["name"], df))

        y = _yearly(spec["name"], df)
        if not y.empty:
            y_parts.append(y)

        o = _rolling_oos(spec["name"], df)
        if not o.empty:
            o_parts.append(o)

    con.close()

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_s = out_dir / "lead_lag_top2_verify_summary.csv"
    p_y = out_dir / "lead_lag_top2_verify_yearly.csv"
    p_o = out_dir / "lead_lag_top2_verify_oos.csv"
    p_m = out_dir / "lead_lag_top2_verify.md"

    s = pd.DataFrame(s_rows)
    y = pd.concat(y_parts, ignore_index=True) if y_parts else pd.DataFrame()
    o = pd.concat(o_parts, ignore_index=True) if o_parts else pd.DataFrame()

    # Always write output files (even if empty) for deterministic automation
    s.to_csv(p_s, index=False)
    y.to_csv(p_y, index=False)
    o.to_csv(p_o, index=False)

    lines = [
        "# Lead-Lag Top2 Verification (No-Lookahead)",
        "",
        "Pairs:",
        "- MES_0900 -> MNQ_1000",
        "- MNQ_0900 -> MES_1000",
        "",
        "Guard:",
        "- include row only if leader_break_ts <= follower entry_ts",
        "",
    ]

    if not s.empty:
        lines.append("## Aggregate")
        for r in s.itertuples(index=False):
            lines.append(
                f"- {r.pair}: N={r.n_total}, ON={r.n_on}, OFF={r.n_off}, "
                f"avgR on/off {r.avg_r_on:+.4f}/{r.avg_r_off:+.4f}, Δ={r.uplift_on_minus_off:+.4f}, "
                f"WR on/off {r.wr_on:.1%}/{r.wr_off:.1%}"
            )

    if not y.empty:
        lines.append("")
        lines.append("## Yearly uplift")
        for pair in y["pair"].unique():
            yp = y[y["pair"] == pair]
            pos = int((yp["uplift"] > 0).sum())
            tot = int(len(yp))
            lines.append(f"- {pair}: years uplift>0 = {pos}/{tot}")

    if not o.empty:
        lines.append("")
        lines.append("## Rolling OOS")
        for r in o.itertuples(index=False):
            lines.append(
                f"- {r.pair} test {r.test_year}: train Δ={r.train_uplift:+.4f}, test Δ={r.test_uplift:+.4f}, n_test_on={r.n_test_on}"
            )

    lines.append("")
    lines.append("## Decision rule")
    lines.append("- KEEP if aggregate Δ>0, yearly positive majority, and OOS mostly positive.")
    lines.append("- WATCH if mixed but still positive in latest OOS.")
    lines.append("- KILL if OOS negative or unstable.")

    p_m.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_s}")
    print(f"Saved: {p_y}")
    print(f"Saved: {p_o}")
    print(f"Saved: {p_m}")

    if not s.empty:
        print("\nAggregate:")
        print(s.to_string(index=False))
    if not y.empty:
        print("\nYearly:")
        print(y.to_string(index=False))
    if not o.empty:
        print("\nRolling OOS:")
        print(o.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
