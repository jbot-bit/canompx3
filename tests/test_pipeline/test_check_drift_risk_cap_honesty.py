"""Tests for check_lane_allocation_risk_cap_honesty.

Companion to ``pipeline.check_drift.check_lane_allocation_risk_cap_honesty``.
Pattern mirrors TestLaneAllocationPerProfileLegacyParity: monkeypatch
PROJECT_ROOT to ``tmp_path`` and write fixtures under ``tmp_path/docs/runtime/``.

Invariant under test: when a lane carries ``risk_cap_pts`` it must satisfy
``0 < risk_cap_pts <= p90_orb_pts`` (C11 cap remediation, 2026-06-05).
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_per_profile(tmp_path: Path, profile_id: str, lanes: list[dict]) -> None:
    runtime = tmp_path / "docs" / "runtime" / "lane_allocation"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / f"{profile_id}.json").write_text(json.dumps({"profile_id": profile_id, "lanes": lanes}))


def _patch_root(monkeypatch, tmp_path: Path) -> None:
    from pipeline import check_drift

    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)


def _lane(**over) -> dict:
    base = {"strategy_id": "MNQ_X", "p90_orb_pts": 49.8}
    base.update(over)
    return base


class TestRiskCapHonesty:
    def test_passes_when_no_files(self, tmp_path, monkeypatch):
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_risk_cap_honesty() == []

    def test_passes_when_no_cap_field(self, tmp_path, monkeypatch):
        """Absent risk_cap_pts is the uncapped baseline — not a violation."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane()])
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_risk_cap_honesty() == []

    def test_passes_valid_cap(self, tmp_path, monkeypatch):
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=37.35)])
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_risk_cap_honesty() == []

    def test_passes_cap_equal_to_p90(self, tmp_path, monkeypatch):
        """cap == p90 is the boundary (cap factor 1.0); allowed."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=49.8)])
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_risk_cap_honesty() == []

    def test_fails_cap_above_p90(self, tmp_path, monkeypatch):
        """A cap ABOVE p90 is dishonest (not a reduction)."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=60.0)])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "exceeds p90_orb_pts" in out[0]

    def test_fails_zero_cap(self, tmp_path, monkeypatch):
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=0)])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "must be > 0" in out[0]

    def test_fails_negative_cap(self, tmp_path, monkeypatch):
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=-5.0)])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "must be > 0" in out[0]

    def test_fails_non_numeric_cap(self, tmp_path, monkeypatch):
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts="lots")])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "must be a number" in out[0]

    def test_fails_bool_cap(self, tmp_path, monkeypatch):
        """bool is an int subclass — must be rejected as non-numeric."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [_lane(risk_cap_pts=True)])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "must be a number" in out[0]

    def test_fails_cap_present_p90_missing(self, tmp_path, monkeypatch):
        """Can't verify the invariant without p90 → fail closed."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        _write_per_profile(tmp_path, "p1", [{"strategy_id": "X", "risk_cap_pts": 30.0}])
        _patch_root(monkeypatch, tmp_path)
        out = check_lane_allocation_risk_cap_honesty()
        assert len(out) == 1
        assert "cannot verify honesty invariant" in out[0]

    def test_skips_unparseable_file(self, tmp_path, monkeypatch):
        """Corrupt JSON is covered by sibling integrity checks; skip here."""
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        runtime = tmp_path / "docs" / "runtime" / "lane_allocation"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "p1.json").write_text("{ not json")
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_risk_cap_honesty() == []

    def test_real_canonical_profile_passes(self, tmp_path, monkeypatch):
        """The actual committed topstep_50k_mnq_auto.json must satisfy the invariant
        (cap 37.35<=49.8, 107.4<=143.2, 33.15<=44.2). No monkeypatch — runs against
        the real PROJECT_ROOT so it guards the shipped file.
        """
        from pipeline.check_drift import check_lane_allocation_risk_cap_honesty

        assert check_lane_allocation_risk_cap_honesty() == []
