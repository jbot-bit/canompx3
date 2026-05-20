"""Injection tests for check_fast_lane_graveyard_digest_parity (Check #170).

Four mutation probes on a synthetic digest file that flip a single invariant
and assert the parity check returns a violation naming the broken invariant:

  1. ghost entry added (digest entry with no source backing)  -> ORPHAN_IN_DIGEST
  2. real entry removed (digest entry absent from sources)    -> MISSING_FROM_DIGEST
  3. hash collision (two distinct titles share a hash)        -> HASH_COLLISION
  4. banner stripped (do_not_hand_edit removed)               -> BANNER TAMPERED

Plus clean-state baseline + missing-file fail-closed.

Design grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
  § "New Derived-State Files" § "Suppression Rules" item 1.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pipeline.check_drift import check_fast_lane_graveyard_digest_parity
from scripts.research.fast_lane_graveyard_digest import (
    DIGEST_SCHEMA_VERSION,
    build_digest,
)


def _serialise_digest(payload: dict) -> str:
    """Mirror fast_lane_graveyard_digest._dump_digest_yaml output shape."""
    banner = "do_not_hand_edit: true\n"
    schema = f"schema_version: {payload['schema_version']}\n"
    body_dict = {k: v for k, v in payload.items() if k not in {"do_not_hand_edit", "schema_version"}}
    body = yaml.safe_dump(body_dict, sort_keys=False, default_flow_style=False)
    return banner + schema + body


def _write_digest(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "fast_lane_graveyard_digest.yaml"
    p.write_text(_serialise_digest(payload), encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Clean-state baseline
# ----------------------------------------------------------------------


def test_real_repo_digest_passes():
    """The on-disk digest landed by Stage 2A.2 must pass clean."""
    assert check_fast_lane_graveyard_digest_parity() == []


def test_freshly_built_digest_passes(tmp_path: Path):
    fresh = build_digest()
    target = _write_digest(tmp_path, fresh)
    assert check_fast_lane_graveyard_digest_parity(digest_path=target) == []


# ----------------------------------------------------------------------
# Fail-closed: missing digest file
# ----------------------------------------------------------------------


def test_missing_digest_fails_closed(tmp_path: Path):
    forged = tmp_path / "does-not-exist.yaml"
    violations = check_fast_lane_graveyard_digest_parity(digest_path=forged)
    assert violations
    assert any("digest file missing" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 1: ghost entry (orphan in digest)
# ----------------------------------------------------------------------


def test_ghost_entry_is_caught(tmp_path: Path):
    fresh = build_digest()
    fresh["entries"].append(
        {
            "source_path": "chatgpt_bundle/06_RD_GRAVEYARD.md",
            "title": "GHOST ENTRY NOT IN ANY SOURCE",
            "status": "DEAD",
            "hash_kind": "class",
            "structural_hash": "1234567890abcdef",
            "lane_inputs": {},
        }
    )
    target = _write_digest(tmp_path, fresh)
    violations = check_fast_lane_graveyard_digest_parity(digest_path=target)
    assert violations
    assert any("ORPHAN_IN_DIGEST" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 2: missing real entry (digest under-reports sources)
# ----------------------------------------------------------------------


def test_missing_real_entry_is_caught(tmp_path: Path):
    fresh = build_digest()
    assert fresh["entries"], "real repo must have at least one graveyard entry"
    # Drop the first real entry; rebuild will reinstate it -> MISSING_FROM_DIGEST.
    fresh["entries"] = fresh["entries"][1:]
    target = _write_digest(tmp_path, fresh)
    violations = check_fast_lane_graveyard_digest_parity(digest_path=target)
    assert violations
    assert any("MISSING_FROM_DIGEST" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 3: hash collision (two distinct titles share a hash)
# ----------------------------------------------------------------------


def test_hash_collision_is_caught(tmp_path: Path):
    fresh = build_digest()
    assert fresh["entries"], "real repo must have at least one graveyard entry"
    target_hash = fresh["entries"][0]["structural_hash"]
    fresh["entries"].append(
        {
            "source_path": "chatgpt_bundle/06_RD_GRAVEYARD.md",
            "title": "COLLIDING TITLE DISTINCT FROM FIRST ENTRY",
            "status": "DEAD",
            "hash_kind": "class",
            "structural_hash": target_hash,
            "lane_inputs": {},
        }
    )
    target = _write_digest(tmp_path, fresh)
    violations = check_fast_lane_graveyard_digest_parity(digest_path=target)
    assert violations
    assert any("HASH_COLLISION" in v for v in violations)


# ----------------------------------------------------------------------
# Injection 4: banner stripped
# ----------------------------------------------------------------------


def test_banner_stripped_is_caught(tmp_path: Path):
    fresh = build_digest()
    target = _write_digest(tmp_path, fresh)
    # Strip the banner from the on-disk file directly.
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("do_not_hand_edit: true\n", ""), encoding="utf-8")
    violations = check_fast_lane_graveyard_digest_parity(digest_path=target)
    assert violations
    assert any("BANNER TAMPERED" in v for v in violations)


def test_schema_version_mutation_is_caught(tmp_path: Path):
    fresh = build_digest()
    fresh["schema_version"] = DIGEST_SCHEMA_VERSION + 99
    target = _write_digest(tmp_path, fresh)
    violations = check_fast_lane_graveyard_digest_parity(digest_path=target)
    assert violations
    assert any("schema_version" in v for v in violations)
