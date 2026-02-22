#!/usr/bin/env python3
"""Relay-chain hypothesis test.

Hypothesis:
- Trade follower M2K US_POST_EQUITY (E1/CB5/RR1.5) only when:
  1) M6E US_EQUITY_OPEN direction matches follower direction, and
  2) MES US_DATA_OPEN direction matches follower direction,
  3) both leader break timestamps are known before follower entry.

Compares:
- baseline (no filter)
- single leader filter (M6E only)
- relay filter (M6E + MES)

Outputs:
- research/output/relay_chain_summary.csv
- research/output/relay_chain_yearly.csv
- research/output/relay_chain_oos.csv
- research/output/relay_chain_notes.md
"""

from __future__ import annotations

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "gold.db"


def stats(s: pd.Series) -> dict:
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
    SELECT
      o.trading_day,
      o.pnl_r,
      o.entry_ts,
      df_f.orb_US_POST_EQUITY_break_dir AS f_dir,
      df_m6e.orb_US_EQUITY_OPEN_break_dir AS m6e_dir,
      df_m6e.orb_US_EQUITY_OPEN_break_ts  AS m6e_ts,
      df_mes.orb_US_DATA_OPEN_break_dir   AS mes_dir,
      df_mes.orb_US_DATA_OPEN_break_ts    AS mes_ts
    FROM orb_outcomes o
    JOIN daily_features df_f
      ON df_f.symbol = o.symbol
     AND df_f.trading_day = o.trading_day
     AND df_f.orb_minutes = o.orb_minutes
    JOIN daily_features df_m6e
      ON df_m6e.symbol = 'M6E'
     AND df_m6e.trading_day = o.trading_day
     AND df_m6e.orb_minutes = o.orb_minutes
    JOIN daily_features df_mes
      ON df_mes.symbol = 'MES'
     AND df_mes.trading_day = o.trading_day
     AND df_mes.orb_minutes = o.orb_minutes
    WHERE o.orb_minutes = 5
      AND o.symbol = 'M2K'
      AND o.orb_label = 'US_POST_EQUITY'
      AND o.entry_model = 'E1'
      AND o.confirm_bars = 5
      AND o.rr_target = 1.5
      AND o.pnl_r IS NOT NULL
      AND o.entry_ts IS NOT NULL
    """

    df = con.execute(q).fetchdf()
    con.close()

    if df.empty:
        print("No rows for follower slice.")
        return 0

    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["m6e_ts"] = pd.to_datetime(df["m6e_ts"], utc=True)
    df["mes_ts"] = pd.to_datetime(df["mes_ts"], utc=True)

    valid_f = df["f_dir"].isin(["long", "short"])

    single_on = (
        valid_f
        & df["m6e_dir"].isin(["long", "short"])
        & (df["m6e_dir"] == df["f_dir"])
        & df["m6e_ts"].notna()
        & (df["m6e_ts"] <= df["entry_ts"])
    )

    relay_on = (
        single_on
        & df["mes_dir"].isin(["long", "short"])
        & (df["mes_dir"] == df["f_dir"])
        & df["mes_ts"].notna()
        & (df["mes_ts"] <= df["entry_ts"])
    )

    baseline = df["pnl_r"]
    single = df.loc[single_on, "pnl_r"]
    relay = df.loc[relay_on, "pnl_r"]

    baseline_s = stats(baseline)
    single_s = stats(single)
    relay_s = stats(relay)

    summary = pd.DataFrame(
        [
            {"bucket": "baseline", **baseline_s},
            {"bucket": "single_m6e", **single_s},
            {"bucket": "relay_m6e_mes", **relay_s},
        ]
    )

    # Yearly for each bucket
    y_rows = []
    for y, g in df.groupby("year"):
        b = g["pnl_r"]
        s = g.loc[single_on.loc[g.index], "pnl_r"]
        r = g.loc[relay_on.loc[g.index], "pnl_r"]
        for name, ser in [("baseline", b), ("single_m6e", s), ("relay_m6e_mes", r)]:
            st = stats(ser)
            y_rows.append({"year": int(y), "bucket": name, **st})
    yearly = pd.DataFrame(y_rows)

    # Quick OOS: latest full-ish year (>=150 baseline rows)
    year_counts = df.groupby("year").size().to_dict()
    eligible = [int(y) for y, n in year_counts.items() if n >= 150]
    test_year = max(eligible) if eligible else max(int(y) for y in df["year"].unique())

    tr = df[df["year"] < test_year]
    te = df[df["year"] == test_year]

    def uplift(train_or_test: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
        on = train_or_test.loc[mask.loc[train_or_test.index], "pnl_r"]
        off = train_or_test.loc[~mask.loc[train_or_test.index], "pnl_r"]
        if len(on) == 0 or len(off) == 0:
            return np.nan, int(len(on))
        return float(on.mean() - off.mean()), int(len(on))

    tr_single_up, tr_single_n = uplift(tr, single_on)
    te_single_up, te_single_n = uplift(te, single_on)
    tr_relay_up, tr_relay_n = uplift(tr, relay_on)
    te_relay_up, te_relay_n = uplift(te, relay_on)

    oos = pd.DataFrame(
        [
            {
                "bucket": "single_m6e",
                "test_year": int(test_year),
                "train_uplift": tr_single_up,
                "test_uplift": te_single_up,
                "n_train_on": tr_single_n,
                "n_test_on": te_single_n,
            },
            {
                "bucket": "relay_m6e_mes",
                "test_year": int(test_year),
                "train_uplift": tr_relay_up,
                "test_uplift": te_relay_up,
                "n_train_on": tr_relay_n,
                "n_test_on": te_relay_n,
            },
        ]
    )

    out_dir = PROJECT_ROOT / "research" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    p_sum = out_dir / "relay_chain_summary.csv"
    p_year = out_dir / "relay_chain_yearly.csv"
    p_oos = out_dir / "relay_chain_oos.csv"
    p_md = out_dir / "relay_chain_notes.md"

    summary.to_csv(p_sum, index=False)
    yearly.to_csv(p_year, index=False)
    oos.to_csv(p_oos, index=False)

    lines = [
        "# Relay Chain Hypothesis Results",
        "",
        "Follower slice: M2K US_POST_EQUITY E1/CB5/RR1.5",
        "Single filter: M6E_US_EQUITY_OPEN direction match",
        "Relay filter: Single + MES_US_DATA_OPEN direction match",
        "No-lookahead: leader break_ts <= follower entry_ts",
        "",
        "## Summary",
    ]

    for r in summary.itertuples(index=False):
        lines.append(f"- {r.bucket}: N={r.n}, WR={r.wr:.1%}, avgR={r.avg_r:+.4f}, totalR={r.total_r:+.2f}")

    lines.append("")
    lines.append(f"## Quick OOS (test year {int(test_year)})")
    for r in oos.itertuples(index=False):
        lines.append(
            f"- {r.bucket}: trainΔ={r.train_uplift:+.4f}, testΔ={r.test_uplift:+.4f}, n_test_on={r.n_test_on}"
        )

    # simple verdict
    verdict = "KILL"
    if (
        relay_s["n"] >= 150
        and relay_s["avg_r"] > single_s["avg_r"] > baseline_s["avg_r"]
        and pd.notna(te_relay_up)
        and te_relay_up > 0
    ):
        verdict = "PROMOTE"
    elif (
        relay_s["n"] >= 100
        and relay_s["avg_r"] >= single_s["avg_r"]
        and pd.notna(te_relay_up)
        and te_relay_up >= 0
    ):
        verdict = "HOLD"
    lines.append("")
    lines.append(f"## Verdict: {verdict}")

    p_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {p_sum}")
    print(f"Saved: {p_year}")
    print(f"Saved: {p_oos}")
    print(f"Saved: {p_md}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nOOS:")
    print(oos.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
