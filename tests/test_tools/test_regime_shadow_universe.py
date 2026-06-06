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


# ── F1: per-lane first_seen ───────────────────────────────────────────────


def test_new_lane_gets_first_seen_as_of_date(tmp_path, monkeypatch):
    """A lane with no prior first_seen is stamped with as_of_date (newly seen)."""
    db = _make_db(tmp_path, [{"strategy_id": "NEW_LANE", "sample_size": 60}])
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF, prior_first_seen={})
    assert len(lanes) == 1
    assert lanes[0].first_seen == AS_OF, "newly-seen lane starts at as_of_date"


def test_existing_lane_first_seen_preserved_across_refresh(tmp_path, monkeypatch):
    """A lane already present keeps its ORIGINAL first_seen; a later refresh must
    NOT advance it (mirrors the global forward_start preserve discipline)."""
    db = _make_db(tmp_path, [{"strategy_id": "OLD_LANE", "sample_size": 60}])
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    original = datetime.date(2026, 6, 3)
    lanes = uni.build_universe(
        db_path=db,
        as_of_date=datetime.date(2026, 7, 1),  # a LATER run
        prior_first_seen={"OLD_LANE": original},
    )
    assert lanes[0].first_seen == original, "existing lane's first_seen must be preserved, not advanced"


def test_legacy_yaml_without_first_seen_backderives_to_forward_start(tmp_path):
    """Migration no-op: a prior YAML lane lacking first_seen back-derives to the
    persisted forward_start, so max(forward_start, first_seen)==forward_start."""
    uyaml = tmp_path / "u.yaml"
    # Hand-write a LEGACY snapshot: forward_start present, lanes have NO first_seen.
    uyaml.write_text(
        "forward_start: '2026-06-03'\nlanes:\n- strategy_id: LEGACY_LANE\n  sample_size: 60\n",
        encoding="utf-8",
    )
    pfs = uni._load_prior_first_seen(universe_yaml=uyaml)
    assert pfs == {"LEGACY_LANE": datetime.date(2026, 6, 3)}, "legacy lane back-derives to forward_start"


def test_first_seen_round_trips_through_yaml(tmp_path, monkeypatch):
    """write_universe_yaml emits first_seen as an ISO date; _load_prior_first_seen
    reads it back identically."""
    db = _make_db(tmp_path, [{"strategy_id": "RT_LANE", "sample_size": 60}])
    _patch_fitness(monkeypatch, {})
    monkeypatch.setattr(uni, "_active_instruments", lambda: ("MNQ",))

    fseen = datetime.date(2026, 6, 25)
    lanes = uni.build_universe(db_path=db, as_of_date=AS_OF, prior_first_seen={"RT_LANE": fseen})
    out = tmp_path / "u2.yaml"
    uni.write_universe_yaml(lanes, forward_start=datetime.date(2026, 6, 3), path=out, as_of_date=AS_OF)

    import yaml

    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["lanes"][0]["first_seen"] == "2026-06-25", "first_seen serialized as ISO date"
    reloaded = uni._load_prior_first_seen(universe_yaml=out)
    assert reloaded["RT_LANE"] == fseen, "first_seen round-trips through YAML"
