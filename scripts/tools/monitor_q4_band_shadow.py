"""Q4-band deployment-shape shadow monitor — append-only, read-only, phase SHADOW_ONLY.

Pre-reg authority: docs/audit/hypotheses/2026-04-20-q4-band-deployment-shape-v1.yaml
Lock SHA: aa9999a7

Reads (all read-only):
  - gold.db (canonical orb_outcomes + daily_features)
  - docs/audit/shadow_ledgers/q4_band_frozen_breakpoints.json (frozen P60/P80)
  - docs/runtime/lane_allocation.json (current DEPLOY set intersection)

Writes only:
  - docs/audit/shadow_ledgers/q4-band-shadow-ledger.csv (append-only, idempotent)

NO CAPITAL ACTION. NO SIZING OVERLAY CALLABLE. NO PHASE ADVANCEMENT LOGIC.

Usage:
    PYTHONPATH=. python scripts/tools/monitor_q4_band_shadow.py [--dry-run]
        [--from-date YYYY-MM-DD] [--to-date YYYY-MM-DD]

  --dry-run:  compute rows but do not write to the ledger
  --from-date / --to-date:  restrict range; defaults to today only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH

PREREG_SHA = "aa9999a7"
LEDGER_VERSION = 1
SHADOW_ACTION_LITERAL = "LOG_ONLY"
PHASE_NAME_DEFAULT = "SHADOW_ONLY"

LEDGER_PATH = Path("docs/audit/shadow_ledgers/q4-band-shadow-ledger.csv")
BREAKPOINTS_PATH = Path("docs/audit/shadow_ledgers/q4_band_frozen_breakpoints.json")
LANE_ALLOCATION_PATH = Path("docs/runtime/lane_allocation.json")
PREREG_PATH = Path("docs/audit/hypotheses/2026-04-20-q4-band-deployment-shape-v1.yaml")

CSV_COLUMNS = [
    "ledger_version",
    "pre_reg_sha",
    "source_commit",
    "written_at_utc",
    "trading_day",
    "lane_id",
    "session",
    "direction",
    "phase_name",
    "eligible_flag",
    "rel_vol_observed",
    "breakpoint_p60",
    "breakpoint_p80",
    "in_q4_band",
    "shadow_action",
    "breach_flag",
    "breach_reason",
    "notes",
]


@dataclass(frozen=True)
class LaneSpec:
    lane_id: str
    session: str
    direction: str
    p60: float
    p80: float


def _current_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "UNKNOWN"


def _load_breakpoints() -> tuple[dict[str, LaneSpec], dict]:
    if not BREAKPOINTS_PATH.exists():
        raise FileNotFoundError(
            f"FAIL-CLOSED: frozen breakpoints file missing at {BREAKPOINTS_PATH}. "
            "Run scripts/tools/freeze_q4_band_breakpoints.py first."
        )
    payload = json.loads(BREAKPOINTS_PATH.read_text(encoding="utf-8"))
    if payload.get("pre_reg_sha") != PREREG_SHA:
        raise RuntimeError(
            f"FAIL-CLOSED: breakpoints file pre_reg_sha "
            f"{payload.get('pre_reg_sha')!r} does not match monitor's expected "
            f"{PREREG_SHA!r}. Aborting."
        )
    if payload.get("recalibration_allowed", None) is True:
        raise RuntimeError("FAIL-CLOSED: breakpoints file has recalibration_allowed=True")
    specs: dict[str, LaneSpec] = {}
    for lane_id, row in payload["lanes"].items():
        specs[lane_id] = LaneSpec(
            lane_id=lane_id,
            session=row["session"],
            direction=row["direction"],
            p60=float(row["p60"]),
            p80=float(row["p80"]),
        )
    return specs, payload


def _load_deployed_sessions() -> set[tuple[str, str]]:
    """Return the set of (instrument, session) pairs the allocator has currently DEPLOYed.

    The allocator format uses `strategy_id`-per-row with `status=DEPLOY`. Per
    the deployment-shape pre-reg §scope.eligibility criterion (b), qualifying
    lanes must exist in the allocator DEPLOY set. The allocator does NOT
    distinguish direction at the entry level (E2 takes both long and short
    daily), so the allocator-side gate is (instrument, session); per-direction
    exclusion is enforced separately via the frozen breakpoints file.
    """
    if not LANE_ALLOCATION_PATH.exists():
        raise FileNotFoundError(f"FAIL-CLOSED: {LANE_ALLOCATION_PATH} missing")
    payload = json.loads(LANE_ALLOCATION_PATH.read_text(encoding="utf-8"))
    out: set[tuple[str, str]] = set()
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        status = lane.get("status", "")
        if status != "DEPLOY":
            continue
        instrument = lane.get("instrument")
        session = lane.get("orb_label")
        if not instrument or not session:
            raise RuntimeError(f"FAIL-CLOSED: lane entry missing instrument/orb_label: {list(lane.keys())}")
        out.add((instrument, session))
    return out


def _existing_ledger_keys() -> set[tuple[str, str]]:
    """Return set of (trading_day, lane_id) tuples already present."""
    if not LEDGER_PATH.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with LEDGER_PATH.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            td = row.get("trading_day", "")
            lid = row.get("lane_id", "")
            if td and lid:
                seen.add((td, lid))
    return seen


def _ensure_ledger_header() -> None:
    if LEDGER_PATH.exists():
        # Validate header matches; never silently overwrite
        with LEDGER_PATH.open("r", encoding="utf-8", newline="") as fh:
            first = fh.readline().rstrip("\r\n")
            expected = ",".join(CSV_COLUMNS)
            if first != expected:
                raise RuntimeError(
                    f"FAIL-CLOSED: existing ledger header does not match schema.\n"
                    f"  expected: {expected}\n  actual:   {first}"
                )
        return
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()


def _append_rows(rows: list[dict]) -> None:
    """Atomic append: write to .tmp + concat."""
    if not rows:
        return
    # Use a direct append with flush+fsync for durability — CSV append is
    # naturally atomic at line granularity on POSIX/NTFS for <=PIPE_BUF lines.
    with LEDGER_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="raise")
        for row in rows:
            writer.writerow(row)
        fh.flush()
        import os

        os.fsync(fh.fileno())


def _query_lane_data(
    con: duckdb.DuckDBPyConnection,
    session: str,
    direction: str,
    from_day: date,
    to_day: date,
) -> list[dict]:
    rel_col = f"rel_vol_{session}"
    sql = f"""
    SELECT
      o.trading_day,
      d.{rel_col} AS rel_vol,
      o.entry_price,
      o.stop_price
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = '{session}'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.confirm_bars = 1
      AND o.rr_target = 1.5
      AND o.pnl_r IS NOT NULL
      AND o.trading_day BETWEEN '{from_day.isoformat()}' AND '{to_day.isoformat()}'
    """
    df = con.sql(sql).to_df()
    if len(df) == 0:
        return []
    entry = df["entry_price"].to_numpy(dtype=np.float64)
    stop = df["stop_price"].to_numpy(dtype=np.float64)
    is_long = entry > stop
    dir_mask = is_long if direction == "long" else ~is_long
    sub = df.loc[dir_mask]
    out = []
    for _, row in sub.iterrows():
        td = row["trading_day"]
        if hasattr(td, "date"):
            td = td.date()
        out.append(
            {
                "trading_day": td.isoformat() if hasattr(td, "isoformat") else str(td),
                "rel_vol": float(row["rel_vol"])
                if row["rel_vol"] is not None and not np.isnan(row["rel_vol"])
                else float("nan"),
            }
        )
    return out


def _compose_row(
    spec: LaneSpec,
    td_iso: str,
    rel_vol: float,
    eligible: bool,
    source_commit: str,
    written_at_utc: str,
) -> dict:
    breach = False
    reason = ""
    in_q4 = 0
    if np.isnan(rel_vol):
        breach = True
        reason = "rel_vol_missing"
    else:
        if spec.p60 <= rel_vol <= spec.p80:
            in_q4 = 1
    return {
        "ledger_version": LEDGER_VERSION,
        "pre_reg_sha": PREREG_SHA,
        "source_commit": source_commit,
        "written_at_utc": written_at_utc,
        "trading_day": td_iso,
        "lane_id": spec.lane_id,
        "session": spec.session,
        "direction": spec.direction,
        "phase_name": PHASE_NAME_DEFAULT,
        "eligible_flag": 1 if eligible else 0,
        "rel_vol_observed": "" if np.isnan(rel_vol) else f"{rel_vol:.6f}",
        "breakpoint_p60": f"{spec.p60:.6f}",
        "breakpoint_p80": f"{spec.p80:.6f}",
        "in_q4_band": in_q4,
        "shadow_action": SHADOW_ACTION_LITERAL,
        "breach_flag": 1 if breach else 0,
        "breach_reason": reason,
        "notes": "",
    }


def run(
    from_date: date | None,
    to_date: date | None,
    dry_run: bool,
) -> int:
    specs, payload = _load_breakpoints()
    deploy_set = _load_deployed_sessions()
    instrument = payload["instrument"]

    # Eligibility = lane is in frozen breakpoints AND its (instrument, session)
    # is currently DEPLOY'd in the allocator. Per-direction exclusion is enforced
    # upstream in the frozen breakpoints file (3 flipped cells excluded there).
    eligible_ids = {lid for lid, spec in specs.items() if (instrument, spec.session) in deploy_set}

    # Default range = today only (Brisbane-local date per pipeline.dst; but
    # canonical orb_outcomes.trading_day is Brisbane-local; use today's UTC date
    # which is the nearest approximation without importing tz libs at runtime).
    today = datetime.now(UTC).date()
    if from_date is None:
        from_date = today
    if to_date is None:
        to_date = today
    if to_date < from_date:
        raise ValueError(f"to_date {to_date} < from_date {from_date}")

    if not dry_run:
        _ensure_ledger_header()

    existing = _existing_ledger_keys() if not dry_run else set()
    source_commit = _current_commit()
    written_at_utc = datetime.now(UTC).isoformat(timespec="seconds")

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    new_rows: list[dict] = []
    try:
        for spec in specs.values():
            rows = _query_lane_data(con, spec.session, spec.direction, from_date, to_date)
            for r in rows:
                key = (r["trading_day"], spec.lane_id)
                if key in existing:
                    continue
                new_rows.append(
                    _compose_row(
                        spec=spec,
                        td_iso=r["trading_day"],
                        rel_vol=r["rel_vol"],
                        eligible=(spec.lane_id in eligible_ids),
                        source_commit=source_commit,
                        written_at_utc=written_at_utc,
                    )
                )
    finally:
        con.close()

    if dry_run:
        print(f"DRY-RUN: would append {len(new_rows)} rows")
        if new_rows:
            print("sample row:", new_rows[0])
        return 0

    _append_rows(new_rows)
    print(f"APPENDED: {len(new_rows)} rows to {LEDGER_PATH}")
    print(f"  eligible lanes: {len(eligible_ids)} / frozen {len(specs)}")
    print(f"  range: {from_date.isoformat()} to {to_date.isoformat()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Q4-band shadow monitor (append-only)")
    parser.add_argument("--dry-run", action="store_true", help="compute rows, don't write")
    parser.add_argument("--from-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None)
    parser.add_argument("--to-date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=None)
    args = parser.parse_args()
    return run(args.from_date, args.to_date, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
