"""Injection test for check_fast_lane_promote_orphans (drift check #157).

Verifies the check catches:
  - ERROR entries (pooling-artifact PROMOTE without revocation sidecar)
  - Cache-out-of-sync (hand-edited or stale promote_queue.yaml)
  - Returns clean on the current healthy state
"""

from __future__ import annotations

import yaml

from pipeline.check_drift import check_fast_lane_promote_orphans
from scripts.research import fast_lane_promote_queue as flpq


def test_drift_check_passes_on_current_state():
    """Real result MDs + revocation sidecar + cache = clean."""
    violations = check_fast_lane_promote_orphans()
    assert violations == [], f"unexpected violations: {violations}"


def test_drift_check_fails_on_hand_edited_cache(tmp_path, monkeypatch):
    """Hand-edit the cache: flip REVOKED -> QUEUED. Check must catch it."""
    # Load current cache, mutate one entry's status, rewrite to tmp,
    # point QUEUE_CACHE at it for the duration of the test.
    real_cache = flpq.QUEUE_CACHE
    data = yaml.safe_load(real_cache.read_text(encoding="utf-8"))
    flipped = False
    for entry in data["entries"]:
        if entry["status"] == "REVOKED":
            entry["status"] = "QUEUED"
            flipped = True
            break
    assert flipped, "test precondition: at least one REVOKED entry in real cache"

    forged = tmp_path / "promote_queue.yaml"
    forged.write_text(yaml.safe_dump(data), encoding="utf-8")

    monkeypatch.setattr(flpq, "QUEUE_CACHE", forged)
    violations = check_fast_lane_promote_orphans()
    assert violations, "drift check did not detect hand-edited cache"
    assert any("cache out of sync" in v for v in violations)
    assert any("REVOKED -> QUEUED" in v or "QUEUED -> REVOKED" in v for v in violations)


def test_drift_check_fails_on_unrevoked_pooling_artifact(tmp_path, monkeypatch):
    """If lane #2's revocation sidecar were deleted, the scanner would
    classify it as ERROR and the drift check must catch it."""
    real_results = flpq.RESULTS_DIR

    # Copy result MDs to tmp, but OMIT the .revocation.md sidecar.
    shadow = tmp_path / "results"
    shadow.mkdir()
    for md in real_results.glob("*fast-lane*.md"):
        if md.name.endswith(".revocation.md"):
            continue
        (shadow / md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(flpq, "RESULTS_DIR", shadow)
    monkeypatch.setattr(flpq, "QUEUE_CACHE", tmp_path / "no_cache.yaml")
    violations = check_fast_lane_promote_orphans()
    assert violations, "drift check did not detect unrevoked pooling artifact"
    assert any("pooling artifact lacks revocation sidecar" in v for v in violations)
    assert any("ORB_VOL_16K" in v for v in violations)
