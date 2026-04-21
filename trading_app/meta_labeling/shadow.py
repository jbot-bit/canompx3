"""Append-only shadow monitor for the monotonic allocator baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading_app.live.cusum_monitor import CUSUMMonitor
from trading_app.live.sr_monitor import ShiryaevRobertsMonitor, calibrate_sr_threshold
from trading_app.meta_labeling.dataset import (
    PHASE1_V1_FAMILY,
    DatasetBundle,
    build_family_dataset,
)
from trading_app.meta_labeling.monotonic import (
    TOKYO_OPEN_MONOTONIC_V1,
    apply_monotonic_allocator,
    fit_monotonic_allocator,
)

PREREG_PATH = Path("docs/audit/hypotheses/2026-04-21-meta-label-sizing-v1.yaml")
DEFAULT_LEDGER_PATH = Path("docs/audit/shadow_ledgers/meta-labeling-monotonic-shadow-ledger.csv")
LEDGER_VERSION = 1
PHASE_NAME = "SHADOW_ONLY"
SHADOW_ACTION = "LOG_ONLY"

CSV_COLUMNS = [
    "ledger_version",
    "phase_name",
    "shadow_action",
    "written_at_utc",
    "trading_day",
    "family_id",
    "instrument",
    "orb_label",
    "entry_model",
    "rr_target",
    "confirm_bars",
    "direction",
    "score_bucket",
    "size_scalar",
    "baseline_pnl_r",
    "shadow_pnl_r",
    "baseline_expectancy_r",
    "shadow_expectancy_r_dev",
    "cusum_severity",
    "cusum_alarm",
    "sr_alarm_ratio",
    "sr_alarm",
    "notes",
]


@dataclass(frozen=True)
class ShadowMonitorSummary:
    """Forward shadow summary after sequential monitor updates."""

    baseline_expectancy_r: float
    shadow_expectancy_r_dev: float
    baseline_std_r: float
    shadow_std_r_dev: float
    cusum_threshold: float
    sr_threshold: float


def _family_id(bundle: DatasetBundle) -> str:
    family = bundle.family
    direction = family.direction or "both"
    return (
        f"{family.symbol}_{family.orb_label}_{family.entry_model}"
        f"_RR{family.rr_target}_CB{family.confirm_bars}_O{family.orb_minutes}_{direction}"
    )


def _sample_std(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) < 2:
        return 0.0
    return float(series.std(ddof=1))


def _ensure_ledger_header(ledger_path: Path) -> None:
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8", newline="") as fh:
            header = fh.readline().rstrip("\r\n")
        expected = ",".join(CSV_COLUMNS)
        if header != expected:
            raise RuntimeError(
                "FAIL-CLOSED: existing monotonic shadow ledger header mismatch.\n"
                f"expected: {expected}\nactual:   {header}"
            )
        return
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()


def _existing_keys(ledger_path: Path) -> set[tuple[str, str]]:
    if not ledger_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            td = row.get("trading_day")
            family_id = row.get("family_id")
            if td and family_id:
                keys.add((td, family_id))
    return keys


def _append_rows(ledger_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with ledger_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="raise")
        for row in rows:
            writer.writerow(row)
        fh.flush()
        import os

        os.fsync(fh.fileno())


def build_shadow_rows(
    bundle: DatasetBundle,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
) -> tuple[list[dict[str, object]], ShadowMonitorSummary]:
    """Build shadow-only forward rows from the frozen monotonic allocator."""
    development_scored = apply_monotonic_allocator(
        fit_monotonic_allocator(bundle.development_frame, config=TOKYO_OPEN_MONOTONIC_V1),
        bundle.development_frame,
    )
    frozen_allocator = fit_monotonic_allocator(bundle.development_frame, config=TOKYO_OPEN_MONOTONIC_V1)
    forward_scored = apply_monotonic_allocator(frozen_allocator, bundle.forward_frame)
    forward_scored = forward_scored.sort_values("trading_day").reset_index(drop=True)

    if from_date is not None:
        forward_scored = forward_scored.loc[forward_scored["trading_day"] >= from_date].reset_index(drop=True)
    if to_date is not None:
        forward_scored = forward_scored.loc[forward_scored["trading_day"] <= to_date].reset_index(drop=True)

    baseline_expectancy_r = float(pd.to_numeric(bundle.development_frame["pnl_r"], errors="coerce").mean())
    shadow_expectancy_r_dev = float(pd.to_numeric(development_scored["sized_pnl_r"], errors="coerce").mean())
    baseline_std_r = _sample_std(bundle.development_frame["pnl_r"])
    shadow_std_r_dev = _sample_std(development_scored["sized_pnl_r"])

    cusum = CUSUMMonitor(
        expected_r=shadow_expectancy_r_dev,
        std_r=shadow_std_r_dev if shadow_std_r_dev > 0 else 1.0,
        threshold=4.0,
    )
    sr_threshold = calibrate_sr_threshold(target_arl=60, delta=-1.0, variance_ratio=1.0)
    sr = ShiryaevRobertsMonitor(
        expected_r=shadow_expectancy_r_dev,
        std_r=shadow_std_r_dev if shadow_std_r_dev > 0 else 1.0,
        threshold=sr_threshold,
        delta=-1.0,
        variance_ratio=1.0,
    )

    family = bundle.family
    family_id = _family_id(bundle)
    written_at_utc = datetime.now(UTC).isoformat()
    rows: list[dict[str, object]] = []
    for _, row in forward_scored.iterrows():
        actual_shadow = float(row["sized_pnl_r"])
        cusum_alarm = cusum.update(actual_shadow)
        sr_alarm = sr.update(actual_shadow)
        rows.append(
            {
                "ledger_version": LEDGER_VERSION,
                "phase_name": PHASE_NAME,
                "shadow_action": SHADOW_ACTION,
                "written_at_utc": written_at_utc,
                "trading_day": row["trading_day"].isoformat(),
                "family_id": family_id,
                "instrument": family.symbol,
                "orb_label": family.orb_label,
                "entry_model": family.entry_model,
                "rr_target": family.rr_target,
                "confirm_bars": family.confirm_bars,
                "direction": family.direction or "both",
                "score_bucket": int(row["score_bucket"]) if pd.notna(row["score_bucket"]) else "",
                "size_scalar": round(float(row["size_scalar"]), 6),
                "baseline_pnl_r": round(float(row["pnl_r"]), 6),
                "shadow_pnl_r": round(actual_shadow, 6),
                "baseline_expectancy_r": round(baseline_expectancy_r, 6),
                "shadow_expectancy_r_dev": round(shadow_expectancy_r_dev, 6),
                "cusum_severity": round(float(cusum.drift_severity), 6),
                "cusum_alarm": int(cusum_alarm or cusum.alarm_triggered),
                "sr_alarm_ratio": round(float(sr.alarm_ratio), 6),
                "sr_alarm": int(sr_alarm or sr.alarm_triggered),
                "notes": "forward_monitor_only",
            }
        )

    summary = ShadowMonitorSummary(
        baseline_expectancy_r=baseline_expectancy_r,
        shadow_expectancy_r_dev=shadow_expectancy_r_dev,
        baseline_std_r=baseline_std_r,
        shadow_std_r_dev=shadow_std_r_dev,
        cusum_threshold=4.0,
        sr_threshold=sr_threshold,
    )
    return rows, summary


def record_shadow_ledger(
    *,
    db_path: Path | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    from_date: date | None = None,
    to_date: date | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Append monotonic shadow rows to the append-only ledger."""
    bundle = build_family_dataset(family=PHASE1_V1_FAMILY, db_path=db_path)
    rows, summary = build_shadow_rows(bundle, from_date=from_date, to_date=to_date)

    existing = _existing_keys(ledger_path)
    new_rows = [row for row in rows if (str(row["trading_day"]), str(row["family_id"])) not in existing]
    if not dry_run:
        _ensure_ledger_header(ledger_path)
        _append_rows(ledger_path, new_rows)

    shadow_total = sum(float(row["shadow_pnl_r"]) for row in rows)
    baseline_total = sum(float(row["baseline_pnl_r"]) for row in rows)
    return {
        "pre_reg_path": str(PREREG_PATH),
        "ledger_path": str(ledger_path),
        "dry_run": dry_run,
        "family_id": _family_id(bundle),
        "rows_considered": len(rows),
        "rows_appended": len(new_rows),
        "forward_window": {
            "from_date": from_date.isoformat() if from_date else None,
            "to_date": to_date.isoformat() if to_date else None,
        },
        "summary": asdict(summary),
        "forward_totals": {
            "baseline_total_r": round(baseline_total, 6),
            "shadow_total_r": round(shadow_total, 6),
            "delta_total_r": round(shadow_total - baseline_total, 6),
        },
        "latest_row": new_rows[-1] if new_rows else (rows[-1] if rows else None),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Append-only monotonic allocator shadow monitor")
    parser.add_argument("--db-path", type=Path, default=None, help="Override canonical gold.db path")
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Append-only shadow ledger CSV path",
    )
    parser.add_argument("--from-date", type=date.fromisoformat, default=None)
    parser.add_argument("--to-date", type=date.fromisoformat, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = record_shadow_ledger(
        db_path=args.db_path,
        ledger_path=args.ledger_path,
        from_date=args.from_date,
        to_date=args.to_date,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
