from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from trading_app.meta_labeling.dataset import DatasetBundle, MetaLabelFamilySpec
from trading_app.meta_labeling.feature_contract import MetaLabelFeatureContract
from trading_app.meta_labeling import shadow


def _make_bundle() -> DatasetBundle:
    development_rows = []
    for year, base in [(2021, 1.0), (2022, 2.0), (2023, 3.0), (2024, 4.0), (2025, 5.0)]:
        for offset in range(10):
            strength = base + offset / 20.0
            pnl_r = -0.10 + strength * 0.08
            development_rows.append(
                {
                    "trading_day": date(year, 1 + (offset % 4), 1 + offset),
                    "pnl_r": pnl_r,
                    "target": int(pnl_r > 0),
                    "prev_day_range": 8.0 * strength,
                    "atr_20": 4.0,
                    "atr_vel_ratio": 0.8 + strength / 5.0,
                }
            )
    forward_rows = []
    for idx in range(4):
        strength = 6.0 + idx / 10.0
        pnl_r = -0.10 + strength * 0.08
        forward_rows.append(
            {
                "trading_day": date(2026, 1, 10 + idx),
                "pnl_r": pnl_r,
                "target": int(pnl_r > 0),
                "prev_day_range": 8.0 * strength,
                "atr_20": 4.0,
                "atr_vel_ratio": 0.8 + strength / 5.0,
            }
        )

    development = pd.DataFrame(development_rows)
    forward = pd.DataFrame(forward_rows)
    family = MetaLabelFamilySpec(
        symbol="MNQ",
        orb_label="TOKYO_OPEN",
        entry_model="E2",
        rr_target=1.5,
        confirm_bars=1,
        direction="long",
        orb_minutes=5,
    )
    contract = MetaLabelFeatureContract(
        target_session="TOKYO_OPEN",
        numeric_features=("atr_20", "atr_vel_ratio", "prev_day_range"),
        categorical_features=(),
    )
    return DatasetBundle(
        family=family,
        feature_contract=contract,
        full_frame=pd.concat([development, forward], ignore_index=True),
        development_frame=development,
        forward_frame=forward,
    )


def test_build_shadow_rows_emits_forward_monitor_only_rows(monkeypatch):
    bundle = _make_bundle()
    rows, summary = shadow.build_shadow_rows(bundle)

    assert len(rows) == len(bundle.forward_frame)
    assert all(row["notes"] == "forward_monitor_only" for row in rows)
    assert rows[0]["family_id"].startswith("MNQ_TOKYO_OPEN_E2")
    assert summary.shadow_expectancy_r_dev > 0


def test_record_shadow_ledger_is_idempotent(tmp_path, monkeypatch):
    bundle = _make_bundle()
    ledger_path = tmp_path / "shadow.csv"

    monkeypatch.setattr(shadow, "build_family_dataset", lambda family, db_path=None: bundle)

    first = shadow.record_shadow_ledger(ledger_path=ledger_path)
    second = shadow.record_shadow_ledger(ledger_path=ledger_path)

    assert first["rows_appended"] == len(bundle.forward_frame)
    assert second["rows_appended"] == 0

    with ledger_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == len(bundle.forward_frame)


def test_record_shadow_ledger_honors_dry_run(tmp_path, monkeypatch):
    bundle = _make_bundle()
    ledger_path = tmp_path / "shadow.csv"

    monkeypatch.setattr(shadow, "build_family_dataset", lambda family, db_path=None: bundle)

    report = shadow.record_shadow_ledger(ledger_path=ledger_path, dry_run=True)

    assert report["rows_appended"] == len(bundle.forward_frame)
    assert not ledger_path.exists()
