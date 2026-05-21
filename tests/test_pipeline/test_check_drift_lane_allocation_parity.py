"""Tests for check_lane_allocation_per_profile_legacy_parity.

Companion to ``pipeline.check_drift.check_lane_allocation_per_profile_legacy_parity``.
Pattern mirrors TestLaneAllocationChordiaGate in test_check_drift.py: monkeypatch
PROJECT_ROOT to ``tmp_path`` and write fixtures under
``tmp_path/docs/runtime/``.

Stage: docs/runtime/stages/2026-05-21-multi-profile-lane-allocation-stage-1b-i.md.
Companion: docs/specs/lane_allocation_schema.md § 3 (writer contract), § 5
(this check).
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_legacy(tmp_path: Path, body: dict | str) -> None:
    """Write the legacy single-profile file. ``body`` can be a dict (json.dumps'd)
    or a raw string (for tests that need to control bytes exactly).
    """
    runtime = tmp_path / "docs" / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    legacy = runtime / "lane_allocation.json"
    if isinstance(body, dict):
        legacy.write_text(json.dumps(body))
    else:
        legacy.write_text(body)


def _write_per_profile(tmp_path: Path, profile_id: str, body: dict | str) -> None:
    runtime = tmp_path / "docs" / "runtime" / "lane_allocation"
    runtime.mkdir(parents=True, exist_ok=True)
    target = runtime / f"{profile_id}.json"
    if isinstance(body, dict):
        target.write_text(json.dumps(body))
    else:
        target.write_text(body)


def _patch_root(monkeypatch, tmp_path: Path) -> None:
    from pipeline import check_drift

    monkeypatch.setattr(check_drift, "PROJECT_ROOT", tmp_path)


class TestLaneAllocationPerProfileLegacyParity:
    """check_lane_allocation_per_profile_legacy_parity skip + happy + injection paths."""

    def test_passes_when_new_dir_absent(self, tmp_path, monkeypatch):
        """No new-path directory => skip mode (returns no violations)."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        _write_legacy(tmp_path, {"profile_id": "p1", "lanes": []})
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []

    def test_passes_when_legacy_absent(self, tmp_path, monkeypatch):
        """Per-profile present but no legacy => skip mode."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        _write_per_profile(tmp_path, "p1", {"profile_id": "p1", "lanes": []})
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []

    def test_passes_when_legacy_unparseable(self, tmp_path, monkeypatch):
        """Unparseable legacy returns a BAD-legacy line, not a parity diff."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        _write_legacy(tmp_path, "{not valid json")
        _write_per_profile(tmp_path, "p1", {"profile_id": "p1"})
        _patch_root(monkeypatch, tmp_path)
        violations = check_lane_allocation_per_profile_legacy_parity()
        assert len(violations) == 1
        assert "BAD legacy lane_allocation.json" in violations[0]

    def test_passes_byte_equal(self, tmp_path, monkeypatch):
        """Byte-identical bodies for matching profile_id => clean pass."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        body = {"profile_id": "p1", "lanes": [{"strategy_id": "X"}]}
        body_str = json.dumps(body)
        _write_legacy(tmp_path, body_str)
        _write_per_profile(tmp_path, "p1", body_str)
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []

    def test_passes_when_profile_id_mismatch(self, tmp_path, monkeypatch):
        """Legacy holds profile B, per-profile has A. NOT a parity violation —
        legacy is last-write-wins single-profile; only matching profile_id pairs
        are compared.
        """
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        _write_legacy(tmp_path, {"profile_id": "B", "lanes": ["different"]})
        _write_per_profile(tmp_path, "A", {"profile_id": "A", "lanes": ["A_lanes"]})
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []

    def test_fails_on_byte_diff(self, tmp_path, monkeypatch):
        """Same profile_id, different body bytes => violation."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        legacy_body = {"profile_id": "p1", "lanes": [{"strategy_id": "X"}]}
        new_body = {"profile_id": "p1", "lanes": [{"strategy_id": "Y"}]}
        _write_legacy(tmp_path, json.dumps(legacy_body))
        _write_per_profile(tmp_path, "p1", json.dumps(new_body))
        _patch_root(monkeypatch, tmp_path)

        violations = check_lane_allocation_per_profile_legacy_parity()
        assert len(violations) == 1
        assert "parity drift" in violations[0]
        assert "p1" in violations[0]
        assert "docs/runtime/lane_allocation/p1.json" in violations[0]
        assert "docs/runtime/lane_allocation.json" in violations[0]
        assert "save_allocation" in violations[0]

    def test_fails_on_whitespace_diff_only(self, tmp_path, monkeypatch):
        """Byte-equal is stricter than dict-equal: trailing newline / whitespace
        differences trip the check even though json.loads() of both would yield
        identical dicts. This is the writer-drift class we want to catch.
        """
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        legacy_str = '{"profile_id": "p1", "lanes": []}'
        new_str = legacy_str + "\n"  # trailing newline only
        _write_legacy(tmp_path, legacy_str)
        _write_per_profile(tmp_path, "p1", new_str)
        _patch_root(monkeypatch, tmp_path)

        violations = check_lane_allocation_per_profile_legacy_parity()
        assert len(violations) == 1
        assert "parity drift" in violations[0]

    def test_fails_on_single_byte_mutation(self, tmp_path, monkeypatch):
        """Injection test (per feedback_injection_test_catches_float_repr_class_bug.md):
        mutate one byte of the per-profile file, confirm violation fires.
        """
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        body = '{"profile_id": "p1", "lanes": [], "all_scores_count": 0}'
        _write_legacy(tmp_path, body)
        # Mutate the per-profile body by changing the trailing 0 to a 1.
        mutated = body[:-2] + "1}"
        _write_per_profile(tmp_path, "p1", mutated)
        _patch_root(monkeypatch, tmp_path)

        violations = check_lane_allocation_per_profile_legacy_parity()
        assert len(violations) == 1
        assert "parity drift" in violations[0]

    def test_compares_only_matching_profile_among_many(self, tmp_path, monkeypatch):
        """When new-dir holds many profiles, only the one matching legacy.profile_id
        is compared. The others are silently OK regardless of content.
        """
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        body = {"profile_id": "p1", "lanes": ["matching"]}
        _write_legacy(tmp_path, json.dumps(body))
        _write_per_profile(tmp_path, "p1", json.dumps(body))  # byte-equal pair
        _write_per_profile(tmp_path, "p2", '{"profile_id": "p2", "lanes": ["other"]}')
        _write_per_profile(tmp_path, "p3", '{"profile_id": "p3"}')
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []

    def test_ignores_non_json_files_in_new_dir(self, tmp_path, monkeypatch):
        """Files under new-dir without .json suffix are ignored."""
        from pipeline.check_drift import check_lane_allocation_per_profile_legacy_parity

        _write_legacy(tmp_path, {"profile_id": "p1"})
        runtime = tmp_path / "docs" / "runtime" / "lane_allocation"
        runtime.mkdir(parents=True, exist_ok=True)
        (runtime / "README.md").write_text("not a profile file")
        (runtime / ".gitkeep").write_text("")
        _patch_root(monkeypatch, tmp_path)
        assert check_lane_allocation_per_profile_legacy_parity() == []
