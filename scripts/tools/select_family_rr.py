#!/usr/bin/env python
"""Select locked RR per family using JK-MaxSharpe criterion.

JK-MaxSharpe: among RR levels with statistically-equal per-trade Sharpe
(Jobson-Korkie p > 0.05 vs best), pick the one with highest Sharpe.

History of tiebreaker evolution:
- v1 (SharpeDD): lowest MaxDD among JK-equal → 74% locked to RR1.0
  (lower RR = lower DD mechanically). Created phantom trades.
- v2 (JK-MaxExpR): highest ExpR among JK-equal → biased to highest RR
  for positive-skew distributions. Selection bias per Bailey & Lopez
  de Prado (DSR, 2014): ExpR increases mechanically with RR when
  skew > 0, so MAX_EXPR always picks grid edge.
- v3 (JK-MaxSharpe): highest Sharpe among JK-equal → no directional
  bias. Selects best risk-adjusted RR. Sharpe peak depends on actual
  distribution shape, not mechanical RR ordering.

Statistical basis:
- JK rho=0.7 validated: measured rho 0.67-0.77 for adjacent RRs,
  0.47-0.53 for distant pairs (RR1.0 vs RR4.0). 0.7 is conservative.
- 97% of multi-RR families: ALL RRs are JK-equal candidates.
  Tiebreaker is therefore the dominant selection force.
- DSR (Bailey & LdP 2014): select on risk-adjusted metrics to avoid
  "winner's curse" from multiple trials on the same dataset.

Writes results to `family_rr_locks` table in gold.db.
Run after strategy_validator, before build_edge_families.

Usage:
    python scripts/tools/select_family_rr.py [--db-path gold.db] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.stats import jobson_korkie_p as _jobson_korkie_p
from trading_app.config import ENTRY_MODELS, SKIP_ENTRY_MODELS
from trading_app.validated_shelf import deployable_validated_relation

# Jobson-Korkie assumed correlation between same-setup / different-RR return streams.
# Same ORB break, same entry, different exit target -> high correlation.
JK_RHO = 0.7
JK_ALPHA = 0.05

# 6-column family key
FAMILY_COLS = ["instrument", "orb_label", "filter_type", "entry_model", "orb_minutes", "confirm_bars"]


def select_rr_for_family(group: pd.DataFrame) -> dict:
    """Select locked RR for one family using JK-MaxSharpe criterion.

    Among RR levels with statistically-equal Sharpe (JK p > 0.05),
    pick the one with highest Sharpe. Grounded in Bailey & Lopez de
    Prado (DSR, 2014): select on risk-adjusted metrics to avoid
    selection bias from multiple trials.

    Args:
        group: DataFrame rows for one family, all with same 6-col key,
               varying only in rr_target. Must have columns:
               rr_target, sharpe_ratio, max_drawdown_r, sample_size,
               expectancy_r, trades_per_year.

    Returns:
        dict with locked_rr, method, and metrics at the selected RR.
    """
    family_key = {col: group.iloc[0][col] for col in FAMILY_COLS}

    if len(group) == 1:
        row = group.iloc[0]
        return {
            **family_key,
            "locked_rr": float(row["rr_target"]),
            "method": "ONLY_RR",
            "sharpe_at_rr": float(row["sharpe_ratio"]),
            "maxdd_at_rr": float(row["max_drawdown_r"]),
            "n_at_rr": int(row["sample_size"]),
            "expr_at_rr": float(row["expectancy_r"]),
            "tpy_at_rr": float(row["trades_per_year"]),
        }

    # Find best Sharpe
    best_idx = group["sharpe_ratio"].idxmax()
    best_sharpe = group.loc[best_idx, "sharpe_ratio"]
    best_n = group.loc[best_idx, "sample_size"]

    # Jobson-Korkie: find all RRs NOT significantly worse than best
    candidates = []
    for idx, row in group.iterrows():
        p = _jobson_korkie_p(
            best_sharpe,
            row["sharpe_ratio"],
            int(best_n),
            int(row["sample_size"]),
            JK_RHO,
        )
        if p > JK_ALPHA:
            candidates.append(idx)

    # Fallback: if no candidates pass (shouldn't happen since best vs itself = p=1.0)
    if not candidates:
        candidates = [best_idx]

    # Among candidates, pick highest Sharpe (DSR: select on risk-adjusted metric)
    cand_df = group.loc[candidates]
    selected_idx = cand_df["sharpe_ratio"].idxmax()
    sel = group.loc[selected_idx]

    # Determine method — since we pick best Sharpe from JK-equal candidates,
    # and best Sharpe is always JK-equal to itself, this is nearly always MAX_SHARPE.
    # JK_SHARPE only occurs if JK excludes the global best Sharpe (extremely rare).
    if selected_idx == best_idx:
        method = "MAX_SHARPE"  # best Sharpe from JK-equal candidates
    else:
        method = "JK_SHARPE"  # JK excluded global best, picked best from remainder

    return {
        **family_key,
        "locked_rr": float(sel["rr_target"]),
        "method": method,
        "sharpe_at_rr": float(sel["sharpe_ratio"]),
        "maxdd_at_rr": float(sel["max_drawdown_r"]),
        "n_at_rr": int(sel["sample_size"]),
        "expr_at_rr": float(sel["expectancy_r"]),
        "tpy_at_rr": float(sel["trades_per_year"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Select locked RR per family (JK-MaxSharpe)")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing to DB")
    args = parser.parse_args()

    db_path = args.db_path
    print(f"Database: {db_path}")

    active_models = [em for em in ENTRY_MODELS if em not in SKIP_ENTRY_MODELS]
    placeholders = ", ".join(f"'{em}'" for em in active_models)

    con = duckdb.connect(str(db_path), read_only=args.dry_run)
    shelf_relation = deployable_validated_relation(con)
    df = con.execute(f"""
        SELECT instrument, orb_label, filter_type, entry_model,
               orb_minutes, confirm_bars, rr_target,
               sample_size, win_rate, expectancy_r,
               sharpe_ratio, max_drawdown_r, trades_per_year
        FROM {shelf_relation}
        WHERE entry_model IN ({placeholders})
    """).fetchdf()

    print(f"Loaded {len(df)} active strategies across {df.groupby(FAMILY_COLS).ngroups} families")

    # Process each family
    results = []
    for _key, group in df.groupby(FAMILY_COLS):
        result = select_rr_for_family(group)
        results.append(result)

    rdf = pd.DataFrame(results)
    n_total = len(rdf)

    # === Summary ===
    print(f"\n{'=' * 60}")
    print(f"FAMILY RR LOCKS — {n_total} families")
    print(f"{'=' * 60}")

    print("\n--- Method Distribution ---")
    for method, count in rdf["method"].value_counts().items():
        print(f"  {method:12s}: {count:3d} ({count / n_total:.0%})")

    print("\n--- Locked RR Distribution ---")
    for rr in sorted(rdf["locked_rr"].unique()):
        c = (rdf["locked_rr"] == rr).sum()
        print(f"  RR{rr:<4g}: {c:3d} ({c / n_total:.0%})")

    print("\n--- Per-Instrument Breakdown ---")
    for inst in sorted(rdf["instrument"].unique()):
        idf = rdf[rdf["instrument"] == inst]
        rr_dist = idf["locked_rr"].value_counts().sort_index()
        rr_str = ", ".join(f"RR{rr}={n}" for rr, n in rr_dist.items())
        print(f"  {inst}: {len(idf)} families — {rr_str}")

    multi = rdf[rdf["method"] != "ONLY_RR"]
    print(f"\n--- Multi-RR Families: {len(multi)}/{n_total} ---")
    print(f"  MAX_SHARPE (best Sharpe from JK-equal candidates): {(multi['method'] == 'MAX_SHARPE').sum()}")
    print(f"  JK_SHARPE  (JK excluded best, picked from rest):   {(multi['method'] == 'JK_SHARPE').sum()}")

    # Write to DB (transactional — crash between DELETE and INSERT won't empty table)
    if not args.dry_run:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM family_rr_locks")
            con.execute("""
                INSERT INTO family_rr_locks
                    (instrument, orb_label, filter_type, entry_model,
                     orb_minutes, confirm_bars, locked_rr, method,
                     sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr)
                SELECT instrument, orb_label, filter_type, entry_model,
                       orb_minutes, confirm_bars, locked_rr, method,
                       sharpe_at_rr, maxdd_at_rr, n_at_rr, expr_at_rr, tpy_at_rr
                FROM rdf
            """)
            row_count = con.execute("SELECT count(*) FROM family_rr_locks").fetchone()[0]
            assert row_count == n_total, f"Expected {n_total}, got {row_count}"
            con.execute("COMMIT")
            print(f"\nWritten {row_count} rows to family_rr_locks")
        except Exception:
            con.execute("ROLLBACK")
            raise
    else:
        print("\n[DRY RUN — no DB writes]")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
