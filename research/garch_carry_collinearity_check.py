"""Day-level carry-garch collinearity check.

Single-number gate: if corr(any_prior_win, garch_high) > 0.5 on the pooled
validated-shelf population, park the entire carry family. If < 0.3, the
portfolio-context path is worth pre-registering.

This is the cheapest honest test before committing research budget to any carry
implementation class after W2e.

Output: prints to stdout (no MD file — this is a gate query, not a finding).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import datetime
import duckdb
import numpy as np
import pandas as pd

from pipeline.dst import SESSION_CATALOG, orb_utc_window
from pipeline.paths import GOLD_DB_PATH
from research import garch_broad_exact_role_exhaustion as broad
from research import garch_partner_state_provenance_audit as prov

GARCH_HIGH = 70.0


def main() -> None:
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Step 1: validated shelf target rows
    rows = prov.load_rows(con)
    print(f"Validated shelf rows: {len(rows)}")
    print(f"Instruments: {sorted(rows['instrument'].unique())}")
    print(f"Target sessions: {sorted(rows['orb_label'].unique())}")
    print()

    # Step 2: collect unique (trading_day, symbol, target_session, garch_pct)
    day_records: list[pd.DataFrame] = []
    for _, row in rows.iterrows():
        filter_sql, join_sql = broad.exact_filter_sql(row["filter_type"], row["orb_label"], row["instrument"])
        if filter_sql is None:
            continue
        ts = row["orb_label"]
        q = f"""
        SELECT DISTINCT o.trading_day, o.symbol,
               d.garch_forecast_vol_pct AS gp
        FROM orb_outcomes o
        JOIN daily_features d
          ON o.trading_day = d.trading_day
         AND o.symbol = d.symbol
         AND o.orb_minutes = d.orb_minutes
        {join_sql}
        WHERE o.symbol = '{row["instrument"]}'
          AND o.orb_label = '{ts}'
          AND o.orb_minutes = {row["orb_minutes"]}
          AND o.entry_model = '{row["entry_model"]}'
          AND o.rr_target = {row["rr_target"]}
          AND o.pnl_r IS NOT NULL
          AND d.garch_forecast_vol_pct IS NOT NULL
          AND {filter_sql}
        """
        df = con.execute(q).df()
        if len(df) == 0:
            continue
        df["target_session"] = ts
        df["orb_minutes"] = int(row["orb_minutes"])
        day_records.append(df)

    target_pop = pd.concat(day_records, ignore_index=True)
    target_pop["trading_day"] = pd.to_datetime(target_pop["trading_day"]).dt.date
    target_pop["gp"] = pd.to_numeric(target_pop["gp"], errors="coerce")
    target_days = target_pop.drop_duplicates(subset=["trading_day", "symbol", "target_session"]).copy()
    print(f"Unique (day, symbol, target_session) rows: {len(target_days)}")

    # Step 3: compute target_start_ts per row
    start_ts_list: list[pd.Timestamp | None] = []
    for _, td in target_days.iterrows():
        try:
            raw = orb_utc_window(td["trading_day"], td["target_session"], int(td["orb_minutes"]))[0]
            start_ts_list.append(pd.Timestamp(raw))
        except Exception:
            start_ts_list.append(None)
    target_days["target_start_ts"] = start_ts_list
    target_days = target_days.dropna(subset=["target_start_ts"]).copy()
    print(f"After start_ts resolution: {len(target_days)}")

    # Step 4: load all prior outcomes (E2/CB1/RR1.0/O5)
    priors = con.execute("""
    SELECT trading_day, symbol, orb_label, outcome, exit_ts
    FROM orb_outcomes
    WHERE entry_model = 'E2'
      AND confirm_bars = 1
      AND rr_target = 1.0
      AND orb_minutes = 5
      AND outcome IS NOT NULL
      AND exit_ts IS NOT NULL
    """).df()
    priors["trading_day"] = pd.to_datetime(priors["trading_day"]).dt.date
    priors["exit_ts"] = pd.to_datetime(priors["exit_ts"], utc=True)
    print(f"Prior trade rows (E2/CB1/RR1.0/O5): {len(priors)}")

    # Step 5: for each target row, check if ANY prior session win resolved before start
    # Vectorized via merge + filter
    merged = target_days.merge(priors, on=["trading_day", "symbol"], how="left", suffixes=("", "_prior"))
    # Exclude same-session priors
    merged = merged[merged["orb_label"] != merged["target_session"]].copy()
    # Ensure tz-aware comparison — both exit_ts and target_start_ts should be UTC
    merged["target_start_ts"] = pd.to_datetime(merged["target_start_ts"], utc=True)
    merged["exit_ts"] = pd.to_datetime(merged["exit_ts"], utc=True)
    # Filter: prior resolved before target start
    resolved = merged[merged["exit_ts"] < merged["target_start_ts"]].copy()
    # Filter: prior was a win
    prior_wins = resolved[resolved["outcome"] == "win"]

    # Flag: any prior win per (trading_day, symbol, target_session)
    any_win_flags = (
        prior_wins.groupby(["trading_day", "symbol", "target_session"]).size().reset_index(name="n_prior_wins")
    )
    any_win_flags["any_prior_win"] = 1

    # Join back
    target_days = target_days.merge(
        any_win_flags[["trading_day", "symbol", "target_session", "any_prior_win"]],
        on=["trading_day", "symbol", "target_session"],
        how="left",
    )
    target_days["any_prior_win"] = target_days["any_prior_win"].fillna(0).astype(int)
    target_days["garch_high"] = (target_days["gp"] >= GARCH_HIGH).astype(int)

    rdf = target_days[["trading_day", "symbol", "target_session", "garch_high", "any_prior_win"]].copy()

    print(f"\n{'=' * 60}")
    print(f"DAY-LEVEL CARRY-GARCH COLLINEARITY")
    print(f"{'=' * 60}")
    print(f"Population: {len(rdf)} (day, symbol, target_session) rows")
    print(f"garch_high rate: {rdf['garch_high'].mean():.3f}")
    print(f"any_prior_win rate: {rdf['any_prior_win'].mean():.3f}")

    g = rdf["garch_high"].to_numpy(float)
    p = rdf["any_prior_win"].to_numpy(float)
    corr = float(np.corrcoef(g, p)[0, 1])
    print(f"\n*** corr(any_prior_win, garch_high) = {corr:+.4f} ***")

    if corr > 0.5:
        print("\n>> VERDICT: COLLINEAR (corr > 0.5). Park entire carry family.")
    elif corr > 0.3:
        print(
            "\n>> VERDICT: GREY ZONE (0.3 < corr < 0.5). Carry adds some independent info but is partially redundant with garch."
        )
    else:
        print("\n>> VERDICT: LOW COLLINEARITY (corr < 0.3). Portfolio-context path worth pre-registering.")

    # Per target session
    print(f"\nPer target session:")
    for ts, sub in sorted(rdf.groupby("target_session"), key=lambda x: x[0]):
        if len(sub) < 50:
            print(f"  {ts:20s}  N={len(sub):5d}  (too thin)")
            continue
        sg = sub["garch_high"].to_numpy(float)
        sp = sub["any_prior_win"].to_numpy(float)
        c = float(np.corrcoef(sg, sp)[0, 1])
        grate = sub["garch_high"].mean()
        prate = sub["any_prior_win"].mean()
        both = float(((sub["garch_high"] == 1) & (sub["any_prior_win"] == 1)).mean())
        print(
            f"  {ts:20s}  N={len(sub):5d}  garch_high={grate:.3f}  "
            f"any_prior_win={prate:.3f}  both={both:.3f}  corr={c:+.4f}"
        )

    # Per instrument
    print(f"\nPer instrument:")
    for inst, sub in sorted(rdf.groupby("symbol"), key=lambda x: x[0]):
        if len(sub) < 50:
            print(f"  {inst:10s}  N={len(sub):5d}  (too thin)")
            continue
        sg = sub["garch_high"].to_numpy(float)
        sp = sub["any_prior_win"].to_numpy(float)
        c = float(np.corrcoef(sg, sp)[0, 1])
        print(f"  {inst:10s}  N={len(sub):5d}  corr={c:+.4f}")

    # Contingency table
    print(f"\nContingency table:")
    ct = pd.crosstab(
        rdf["garch_high"].map({0: "garch_low", 1: "garch_high"}),
        rdf["any_prior_win"].map({0: "no_prior_win", 1: "any_prior_win"}),
        margins=True,
    )
    print(ct)

    con.close()


if __name__ == "__main__":
    main()
