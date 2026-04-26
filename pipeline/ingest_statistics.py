#!/usr/bin/env python3
"""
Ingest CME exchange statistics into exchange_statistics table.

Reads statistics files (databento .dbn.zst) and extracts per-calendar-date
statistics (session high/low, settlement, OI, volume) for front-month contracts.
Populates the exchange_statistics table. The pit_range_atr column in
daily_features is populated by pipeline/backfill_pit_range_atr.py, which
reads this table and writes the computed ratio back to daily_features.

Idempotent: DELETE+INSERT per instrument. Safe to re-run.

Usage:
    python pipeline/ingest_statistics.py --instrument MNQ
    python pipeline/ingest_statistics.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import databento as db
import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.db_config import configure_connection
from pipeline.ingest_dbn_mgc import choose_front_contract
from pipeline.log import get_logger
from pipeline.paths import GOLD_DB_PATH

logger = get_logger(__name__)

STATS_ROOT = PROJECT_ROOT / "data" / "raw" / "databento" / "statistics"

# Contract symbol patterns per instrument (outright only, no spreads)
OUTRIGHT_PATTERNS: dict[str, list[tuple[re.Pattern, int]]] = {
    "MES": [
        (re.compile(r"^ES[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MES[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MGC": [
        (re.compile(r"^GC[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MGC[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
    "MNQ": [
        (re.compile(r"^NQ[FGHJKMNQUVXZ]\d{1,2}$"), 2),
        (re.compile(r"^MNQ[FGHJKMNQUVXZ]\d{1,2}$"), 3),
    ],
}

# Databento stat_type codes
STAT_TYPES = {
    1: "opening_price",
    2: "indicative_open",
    3: "settlement",
    4: "session_low",
    5: "session_high",
    6: "cleared_volume",
    9: "open_interest",
}


def extract_statistics(instrument: str) -> pd.DataFrame:
    """Extract all stat types per calendar date, front-month by cleared volume.

    Returns DataFrame with columns: cal_date, symbol, session_high, session_low,
    settlement, opening_price, indicative_open, cleared_volume, open_interest,
    total_cleared_volume, front_contract.
    """
    stats_dir = STATS_ROOT / instrument
    if not stats_dir.exists():
        logger.warning(f"No statistics directory for {instrument}: {stats_dir}")
        return pd.DataFrame()

    files = sorted(stats_dir.glob("*.dbn.zst"))
    if not files:
        logger.warning(f"No DBN files in {stats_dir}")
        return pd.DataFrame()

    patterns = OUTRIGHT_PATTERNS[instrument]
    all_rows: list[dict] = []

    for filepath in files:
        try:
            store = db.DBNStore.from_file(str(filepath))
            df = store.to_df()
        except Exception as e:
            logger.warning(f"Failed to read {filepath.name}: {e}")
            continue

        if df.empty:
            continue

        df = df.reset_index()
        df.rename(columns={"ts_event": "ts_utc"}, inplace=True)
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        df["symbol"] = df["symbol"].astype(str)

        # Filter to relevant stat types
        df = df[df["stat_type"].isin(STAT_TYPES.keys())]
        if df.empty:
            continue

        # Filter to outright contracts only
        outright_mask = pd.Series(False, index=df.index)
        for pattern, _ in patterns:
            outright_mask |= df["symbol"].str.match(pattern.pattern)
        df = df[outright_mask]
        if df.empty:
            continue

        df["cal_date"] = df["ts_utc"].dt.date

        for cal_date, day_df in df.groupby("cal_date"):
            row: dict = {"cal_date": cal_date, "symbol": instrument}

            # Select front-month by cleared volume
            vol_rows = day_df[day_df["stat_type"] == 6]
            front_symbol = None
            for pattern, plen in patterns:
                matched_vol = vol_rows[vol_rows["symbol"].str.match(pattern.pattern)]
                if not matched_vol.empty:
                    vol_by_sym = matched_vol.groupby("symbol")["quantity"].sum().to_dict()
                    vol_by_sym = {s: v for s, v in vol_by_sym.items() if 0 < v < 2147483647}
                    if vol_by_sym:
                        front_symbol = choose_front_contract(vol_by_sym, outright_pattern=pattern, prefix_len=plen)
                        if front_symbol:
                            break

            if not front_symbol:
                continue

            row["front_contract"] = front_symbol
            front_df = day_df[day_df["symbol"] == front_symbol]

            for st, name in STAT_TYPES.items():
                st_rows = front_df[front_df["stat_type"] == st]
                if st_rows.empty:
                    row[name] = None
                    continue
                if st in (6, 9):  # volume/OI use quantity
                    val = st_rows["quantity"].iloc[-1]
                    row[name] = int(val) if 0 < val < 2147483647 else None
                elif st == 2:  # indicative open: last update
                    row[name] = float(st_rows.iloc[-1]["price"]) if st_rows.iloc[-1]["price"] > 0 else None
                else:  # price-based: last update
                    val = float(st_rows.iloc[-1]["price"])
                    row[name] = val if val > 0 else None

            # Total cleared volume across ALL outrights
            all_vol = vol_rows[vol_rows["quantity"] < 2147483647]["quantity"]
            row["total_cleared_volume"] = int(all_vol.sum()) if not all_vol.empty and all_vol.sum() > 0 else None

            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    result = result.drop_duplicates(subset=["cal_date"], keep="last")
    result["cal_date"] = pd.to_datetime(result["cal_date"]).dt.date
    return result


def ingest_statistics(instrument: str, db_path: Path | None = None) -> int:
    """Extract and load exchange statistics for one instrument. Returns row count."""
    db_path = db_path or GOLD_DB_PATH

    logger.info(f"Extracting statistics for {instrument}...")
    stats_df = extract_statistics(instrument)

    if stats_df.empty:
        logger.warning(f"No statistics data for {instrument}")
        return 0

    logger.info(f"  {len(stats_df)} calendar dates extracted ({stats_df.cal_date.min()} to {stats_df.cal_date.max()})")

    # Idempotent: DELETE existing rows for this instrument, then INSERT
    con = duckdb.connect(str(db_path))
    configure_connection(con, writing=True)
    try:
        con.execute("DELETE FROM exchange_statistics WHERE symbol = ?", [instrument])
        con.execute(
            """INSERT INTO exchange_statistics
               (cal_date, symbol, session_high, session_low, settlement,
                opening_price, indicative_open, cleared_volume, open_interest,
                total_cleared_volume, front_contract)
               SELECT cal_date, symbol, session_high, session_low, settlement,
                      opening_price, indicative_open, cleared_volume, open_interest,
                      total_cleared_volume, front_contract
               FROM stats_df""",
        )
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM exchange_statistics WHERE symbol = ?", [instrument]).fetchone()[0]  # type: ignore[index]
        logger.info(f"  {count} rows written to exchange_statistics")
        return count
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest CME exchange statistics")
    parser.add_argument("--instrument", type=str, help="Instrument (MES, MGC, MNQ)")
    parser.add_argument("--all", action="store_true", help="Ingest all active instruments")
    args = parser.parse_args()

    if not args.instrument and not args.all:
        parser.error("Specify --instrument or --all")

    instruments = sorted(ACTIVE_ORB_INSTRUMENTS) if args.all else [args.instrument]
    total = 0
    for inst in instruments:
        if inst not in OUTRIGHT_PATTERNS:
            logger.error(f"No outright patterns defined for {inst}")
            continue
        total += ingest_statistics(inst)

    logger.info(f"Done. Total: {total} rows across {len(instruments)} instruments.")


if __name__ == "__main__":
    main()
