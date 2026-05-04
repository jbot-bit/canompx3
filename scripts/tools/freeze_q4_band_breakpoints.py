"""One-shot: compute per-lane IS-frozen P60/P80 rel_vol breakpoints for Q4-band
deployment-shape shadow monitoring.

Pre-reg authority: docs/audit/hypotheses/2026-04-20-q4-band-deployment-shape-v1.yaml
Lock SHA: aa9999a7
Parent universality run: PR #41 merge commit 126ed6b8

Writes: docs/audit/shadow_ledgers/q4_band_frozen_breakpoints.json

Refuses to overwrite if the output file exists — operator must delete it
explicitly to re-run. This preserves the "freeze once, never recalibrate"
discipline from the pre-reg's calibration.lock_policy.

NO CAPITAL ACTION. NO WRITES TO CANONICAL LAYERS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import HOLDOUT_SACRED_FROM

OUTPUT_PATH = Path("docs/audit/shadow_ledgers/q4_band_frozen_breakpoints.json")
PREREG_SHA = "aa9999a7"

# Per-reg § scope.eligibility — 5m E2 CB1 RR1.5, MNQ; sessions discovered
# canonically; 3 flipped cells excluded.
INSTRUMENT = "MNQ"
ORB_MINUTES = 5
ENTRY_MODEL = "E2"
CONFIRM_BARS = 1
RR_TARGET = 1.5
EXCLUDED_CELLS = [
    ("COMEX_SETTLE", "long"),
    ("NYSE_OPEN", "short"),
    ("US_DATA_830", "short"),
]


def _list_canonical_sessions(con: duckdb.DuckDBPyConnection) -> list[str]:
    sql = f"""
    SELECT DISTINCT orb_label
    FROM orb_outcomes
    WHERE symbol = '{INSTRUMENT}'
      AND orb_minutes = {ORB_MINUTES}
      AND entry_model = '{ENTRY_MODEL}'
      AND confirm_bars = {CONFIRM_BARS}
      AND rr_target = {RR_TARGET}
      AND pnl_r IS NOT NULL
    ORDER BY orb_label
    """
    return [row[0] for row in con.sql(sql).fetchall()]


def _load_lane_rel_vol(con: duckdb.DuckDBPyConnection, session: str, direction: str) -> np.ndarray:
    rel_col = f"rel_vol_{session}"
    sql = f"""
    SELECT
      d.{rel_col} AS rel_vol,
      o.entry_price,
      o.stop_price
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = '{INSTRUMENT}'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = {ORB_MINUTES}
      AND o.entry_model = '{ENTRY_MODEL}'
      AND o.confirm_bars = {CONFIRM_BARS}
      AND o.rr_target = {RR_TARGET}
      AND o.pnl_r IS NOT NULL
      AND o.trading_day < '{HOLDOUT_SACRED_FROM}'
    """
    df = con.sql(sql).to_df()
    if len(df) == 0:
        return np.array([], dtype=np.float64)
    entry = df["entry_price"].to_numpy(dtype=np.float64)
    stop = df["stop_price"].to_numpy(dtype=np.float64)
    is_long = entry > stop
    mask = is_long if direction == "long" else ~is_long
    vals = df.loc[mask, "rel_vol"].to_numpy(dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    return vals


def main() -> int:
    if OUTPUT_PATH.exists():
        print(f"FAIL-CLOSED: {OUTPUT_PATH} already exists.")
        print("The freeze-once policy forbids overwriting. If you truly need to")
        print("re-freeze (e.g. superseded pre-reg), delete the file explicitly")
        print("AND cite the new pre-reg lock SHA when re-running.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    try:
        sessions = _list_canonical_sessions(con)
        breakpoints: dict[str, dict] = {}
        for session in sessions:
            for direction in ("long", "short"):
                if (session, direction) in EXCLUDED_CELLS:
                    continue
                vals = _load_lane_rel_vol(con, session, direction)
                if len(vals) < 50:
                    continue  # too thin for quintile freeze
                p60 = float(np.nanpercentile(vals, 60))
                p80 = float(np.nanpercentile(vals, 80))
                lane_id = (
                    f"{INSTRUMENT}_{session}_O{ORB_MINUTES}_{ENTRY_MODEL}_CB{CONFIRM_BARS}_RR{RR_TARGET}_{direction}"
                )
                breakpoints[lane_id] = {
                    "session": session,
                    "direction": direction,
                    "n_is_rows": int(len(vals)),
                    "p60": p60,
                    "p80": p80,
                }
    finally:
        con.close()

    payload = {
        "pre_reg_sha": PREREG_SHA,
        "parent_universality_commit": "126ed6b8",
        "is_window_exclusive_upper": str(HOLDOUT_SACRED_FROM),
        "instrument": INSTRUMENT,
        "orb_minutes": ORB_MINUTES,
        "entry_model": ENTRY_MODEL,
        "confirm_bars": CONFIRM_BARS,
        "rr_target": RR_TARGET,
        "excluded_cells": [{"session": s, "direction": d} for s, d in EXCLUDED_CELLS],
        "recalibration_allowed": False,
        "lanes": breakpoints,
    }
    # Write atomically — tmp then rename
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(OUTPUT_PATH)

    print(f"FROZEN: {len(breakpoints)} qualifying lanes")
    print(f"OUTPUT: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
