"""Injection tests for Check #168 (fast-lane status roll-up reconstruction parity).

Each test mutates a fixture and confirms the check fires. Negative-control
tests confirm a clean fixture passes. Mirrors the pattern of
test_check_drift_fast_lane_structural_hash_schema_parity.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pipeline.check_drift import check_fast_lane_status_rollup_reconstruction_parity


def _write_rollup(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _baseline_payload() -> dict:
    """A payload that should reconstruct cleanly from the live repo state.

    The check imports the writer module and runs ``build_status_entries()``
    against the live repo; we re-serialize that here to guarantee parity
    pass on the negative-control test.
    """
    from scripts.tools.fast_lane_status import build_status_entries, serialize_rollup

    entries = build_status_entries()
    return yaml.safe_load(serialize_rollup(entries))


def test_check_passes_on_freshly_written_rollup(tmp_path: Path) -> None:
    """Negative control: a payload built from the live writer reconstructs cleanly."""
    payload = _baseline_payload()
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert violations == [], violations


def test_check_fires_on_missing_rollup(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=missing)
    assert any("roll-up not found" in v for v in violations), violations


def test_check_fires_on_tampered_schema_version(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["schema_version"] = 999
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any("BANNER TAMPERED" in v and "schema_version" in v for v in violations), violations


def test_check_fires_on_tampered_do_not_hand_edit(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["do_not_hand_edit"] = False
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any(
        "BANNER TAMPERED" in v and "do_not_hand_edit" in v for v in violations
    ), violations


def test_check_fires_on_tampered_source(tmp_path: Path) -> None:
    payload = _baseline_payload()
    payload["source"] = "scripts/tools/other.py"
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any("BANNER TAMPERED" in v and "source" in v for v in violations), violations


def test_check_fires_on_orphan_entry_in_rollup(tmp_path: Path) -> None:
    """Hand-edited entry that doesn't appear in a fresh reconstruction."""
    payload = _baseline_payload()
    payload["entries"].append(
        {
            "strategy_id": "GHOST_INSERTED_BY_HAND_2026_05_19",
            "current_stage": "ACTIVE_PREREG",
            "age_days": 0,
            "next_action_token": "run_fast_lane",
            "upstream_artifact_path": None,
            "downstream_artifact_path": None,
            "observed_at": {},
        }
    )
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any(
        "ORPHAN_IN_ROLLUP" in v and "GHOST_INSERTED_BY_HAND_2026_05_19" in v
        for v in violations
    ), violations


def test_check_fires_on_missing_entry(tmp_path: Path) -> None:
    """An entry that EXISTS in reconstruction but is dropped from the on-disk roll-up."""
    payload = _baseline_payload()
    if not payload.get("entries"):
        return  # live repo has zero entries — skip (only possible in clean fixture)
    dropped = payload["entries"].pop(0)
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any(
        "MISSING_FROM_ROLLUP" in v and dropped["strategy_id"] in v for v in violations
    ), violations


def test_check_fires_on_field_drift_current_stage(tmp_path: Path) -> None:
    """Hand-edited current_stage doesn't match what reconstruction would compute."""
    payload = _baseline_payload()
    if not payload.get("entries"):
        return
    target = payload["entries"][0]
    # Flip stage to something that the reconstruction would NOT produce.
    original = target["current_stage"]
    target["current_stage"] = "ERROR" if original != "ERROR" else "ACTIVE_PREREG"
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(rollup_path=rollup)
    assert any(
        "FIELD_DRIFT" in v and target["strategy_id"] in v and "current_stage" in v
        for v in violations
    ), violations


def test_check_fires_on_capital_class_write_attempt(tmp_path: Path) -> None:
    """Writer source containing a capital-class path near a write marker fails."""
    fake_writer = tmp_path / "fake_writer.py"
    fake_writer.write_text(
        '''"""docstring mentioning chordia_audit_log.yaml is fine in prose."""
from pathlib import Path
out = Path("docs/runtime/lane_allocation.json")
out.write_text("malicious")
''',
        encoding="utf-8",
    )
    # Use a valid empty rollup so banner+reconstruction don't muddy the signal.
    payload = _baseline_payload()
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(
        rollup_path=rollup, writer_path=fake_writer
    )
    assert any("CAPITAL-CLASS WRITE ATTEMPT" in v for v in violations), violations


def test_capital_class_mention_in_docstring_does_not_fire(tmp_path: Path) -> None:
    """Mere prose mention of capital-class paths in docstrings is allowed."""
    fake_writer = tmp_path / "fake_writer.py"
    fake_writer.write_text(
        '''"""This writer is read-only over chordia_audit_log.yaml,
lane_allocation.json, validated_setups, and trading_app/live/."""
# No write calls — pure docstring mention.
print("hello")
''',
        encoding="utf-8",
    )
    payload = _baseline_payload()
    rollup = tmp_path / "fast_lane_status.yaml"
    _write_rollup(rollup, payload)
    violations = check_fast_lane_status_rollup_reconstruction_parity(
        rollup_path=rollup, writer_path=fake_writer
    )
    capital_class_violations = [v for v in violations if "CAPITAL-CLASS" in v]
    assert capital_class_violations == [], capital_class_violations
