"""
Tests for the REGIME shadow universe builder (record-ALL).

Properties under test:
  1. REGIME band: sample_size 30-99 in; 100 out (delegated to classify_strategy).
  2. RECORD-ALL: every active REGIME lane is included regardless of fitness
     status (FIT / WATCH / DECAY / STALE) — fitness is an attribute, not a gate.
  3. Fitness-error lanes are still included (flagged ERROR), never dropped.
  4. YAML round-trips forward_start.

build_universe is exercised against a temp DB with controlled validated_setups
rows; compute_fitness is monkeypatched to drive the status attribute so we test
the universe mechanics, not the fitness engine.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import duckdb

from scripts.tools import regime_shadow_universe as uni

AS_OF = datetime.date(2026, 6, 10)


class _Score:
    def __init__(self, status, rolling_sample=40, rolling_exp_r=0.4):
        self.fitness_status = status
        self.rolling_sample = rolling_sample
        self.rolling_exp_r = rolling_exp_r


def _make_db(tmp_path: Path, rows: list[dict]) -> Path:
    """validated_setups with the columns build_universe reads."""
    db = tmp_path / "uni_test.db"
    with duckdb.connect(str(db)) as con:
        con.execute(
            """CREATE TABLE validated_setups (
                 strategy_id VARCHAR, instrument VARCHAR, orb_label VARCHAR,
                 orb_minutes INTEGER, rr_target DOUBLE, entry_model VARCHAR,
                 confirm_bars INTEGER, filter_type VARCHAR, sample_size INTEGER,
                 status VARCHAR)"""
        )
        for r in rows:
            con.execute(
                """INSERT INTO validated_setups
                   (strategy_id, instrument, orb_label, orb_minutes, rr_target,
                    entry_model, confirm_bars, filter_type, sample_size, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    r["strategy_id"],
                    r.get("instrument", "MNQ"),
                    "COMEX_SETTLE",
                    5,
                    1.0,
                    "E2",
                    1,
                    "NO_FILTER",
                    r["sample_size"],
                    r.get("status", "active"),
                ],
            )
    return db


def _patch_fitness(monkeypatch, status_by_id):
    def fake(strategy_id, db_path=None, as_of_date=None):
        st = status_by_id.get(strategy_id, "FIT")
        if st == "RAISE":
            raise ValueError("forced fitness failure")
        return _Score(st)

    import trading_app.strategy_fitness as sf

    monkeypatch.setattr(sf, "compute_fitness", fake)


def test_regime_band_99_in_100_out(tmp_path, monkeypatch):
    db = _make_db(
        tmp_path,
        [
            {"strategy_id": "IN_99", "sample_size": 99},
            {"strategy_id": "OUT_100", "sample_size": 100},
            {"strategy_id": "IN_30", "sample_size": 30},
            {"strategy_id": "OUT_29", "sample_size": 29},
        ],
    )
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF)
    ids = {x.strategy_id for x in lanes}
    assert ids == {"IN_99", "IN_30"}, "only sample_size 30-99 are REGIME"


def test_record_all_includes_every_fitness_status(tmp_path, monkeypatch):
    db = _make_db(
        tmp_path,
        [
            {"strategy_id": "L_FIT", "sample_size": 60},
            {"strategy_id": "L_WATCH", "sample_size": 60},
            {"strategy_id": "L_DECAY", "sample_size": 60},
            {"strategy_id": "L_STALE", "sample_size": 60},
        ],
    )
    _patch_fitness(
        monkeypatch,
        {
            "L_FIT": "FIT",
            "L_WATCH": "WATCH",
            "L_DECAY": "DECAY",
            "L_STALE": "STALE",
        },
    )
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF)
    assert len(lanes) == 4
    assert all(x.included for x in lanes), "record-ALL: every REGIME lane included"
    statuses = {x.strategy_id: x.fitness_status for x in lanes}
    assert statuses == {
        "L_FIT": "FIT",
        "L_WATCH": "WATCH",
        "L_DECAY": "DECAY",
        "L_STALE": "STALE",
    }, "fitness recorded as attribute, not gated"


def test_fitness_error_lane_still_included_flagged(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [{"strategy_id": "L_ERR", "sample_size": 60}])
    _patch_fitness(monkeypatch, {"L_ERR": "RAISE"})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF)
    assert len(lanes) == 1
    assert lanes[0].included is True, "fitness-error lane must NOT be dropped"
    assert lanes[0].fitness_status == "ERROR"
    assert "uncomputable" in lanes[0].reason


def test_retired_setups_excluded(tmp_path, monkeypatch):
    db = _make_db(
        tmp_path,
        [
            {"strategy_id": "ACTIVE", "sample_size": 60, "status": "active"},
            {"strategy_id": "RETIRED", "sample_size": 60, "status": "retired"},
        ],
    )
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF)
    assert {x.strategy_id for x in lanes} == {"ACTIVE"}


def test_yaml_round_trips_forward_start(tmp_path, monkeypatch):
    db = _make_db(tmp_path, [{"strategy_id": "L1", "sample_size": 60}])
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))
    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF)

    out = tmp_path / "u.yaml"
    fwd = datetime.date(2026, 6, 3)
    uni.write_universe_yaml(lanes, forward_start=fwd, path=out, as_of_date=AS_OF)

    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["forward_start"] == "2026-06-03"
    assert data["regime_tier"]["min_sample"] == uni.REGIME_MIN_SAMPLE
    assert len(data["lanes"]) == 1
