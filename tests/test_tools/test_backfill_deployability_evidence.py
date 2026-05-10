from __future__ import annotations

from pathlib import Path

import duckdb

from scripts.tools import backfill_deployability_evidence as backfill


def _row(**overrides):
    row = {
        "strategy_id": "SID_A",
        "instrument": "MES",
        "orb_label": "COMEX_SETTLE",
        "orb_minutes": 15,
        "entry_model": "E2",
        "rr_target": 1.5,
        "confirm_bars": 1,
        "filter_type": "OVNRNG_100",
        "validation_pathway": "pooled",
        "slippage_validation_status": None,
        "c8_oos_status": None,
    }
    row.update(overrides)
    return row


def test_backfill_plan_recomputes_c8_and_marks_slippage_event_tail_pending(monkeypatch):
    monkeypatch.setattr(backfill, "_load_rows", lambda *_args, **_kwargs: [_row()])
    monkeypatch.setattr(
        backfill,
        "_evaluate_criterion_8_oos",
        lambda *_args, **_kwargs: {
            "c8_oos_status": "PASSED",
            "reason": None,
            "n_oos": 40,
            "oos_expectancy_r": 0.1,
        },
    )

    plan = backfill.build_backfill_plan(db_path=Path("unused.db"), instruments={"MES"})

    assert plan["summary"]["updates"] == 2
    by_field = {update["field"]: update for update in plan["updates"]}
    assert by_field["c8_oos_status"]["new_value"] == "PASSED"
    assert by_field["slippage_validation_status"]["new_value"] == "PENDING_EVENT_TAIL"


def test_backfill_plan_can_scope_to_strategy_ids(monkeypatch):
    seen = {}

    def fake_load_rows(_db_path, instruments, strategy_ids=None):
        seen["instruments"] = instruments
        seen["strategy_ids"] = strategy_ids
        return [_row(strategy_id="SID_A")]

    monkeypatch.setattr(backfill, "_load_rows", fake_load_rows)
    monkeypatch.setattr(backfill, "_evaluate_criterion_8_oos", lambda *_args, **_kwargs: {"c8_oos_status": "PASSED"})

    plan = backfill.build_backfill_plan(
        db_path=Path("unused.db"),
        instruments={"MNQ"},
        strategy_ids={"SID_A"},
        evidence="c8_oos",
    )

    assert seen == {"instruments": {"MNQ"}, "strategy_ids": {"SID_A"}}
    assert plan["strategy_ids"] == ["SID_A"]
    assert plan["summary"]["updates"] == 1


def test_backfill_does_not_overwrite_existing_values_without_flag(monkeypatch):
    monkeypatch.setattr(
        backfill,
        "_load_rows",
        lambda *_args, **_kwargs: [
            _row(slippage_validation_status="PASSED", c8_oos_status="FAILED_RATIO"),
        ],
    )
    monkeypatch.setattr(backfill, "_evaluate_criterion_8_oos", lambda *_args, **_kwargs: {"c8_oos_status": "PASSED"})

    plan = backfill.build_backfill_plan(db_path=Path("unused.db"), instruments={"MES"})

    assert plan["summary"]["updates"] == 0


def test_apply_backfill_plan_updates_only_planned_fields(tmp_path: Path):
    db_path = tmp_path / "evidence.db"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE validated_setups (
                strategy_id VARCHAR,
                c8_oos_status VARCHAR,
                slippage_validation_status VARCHAR
            )
            """
        )
        con.execute("INSERT INTO validated_setups VALUES ('SID_A', NULL, NULL)")
    finally:
        con.close()

    plan = {
        "updates": [
            {
                "strategy_id": "SID_A",
                "field": "c8_oos_status",
                "new_value": "PASSED",
            },
            {
                "strategy_id": "SID_A",
                "field": "slippage_validation_status",
                "new_value": "PENDING_EVENT_TAIL",
            },
        ]
    }

    assert backfill.apply_backfill_plan(db_path, plan) == 2
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute("SELECT c8_oos_status, slippage_validation_status FROM validated_setups").fetchone()
    finally:
        con.close()
    assert row == ("PASSED", "PENDING_EVENT_TAIL")
