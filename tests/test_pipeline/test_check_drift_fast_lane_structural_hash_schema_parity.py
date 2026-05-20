"""Injection tests for check_fast_lane_structural_hash_schema_parity (Check #167).

Four mutation probes that flip a single invariant -- module schema-version,
module input-field list, hash output length, hash determinism -- and assert
the parity check returns a violation that names the broken constant.

Per Stage 2A design doc acceptance criteria § Tests (Stage 2A.1 split
renumbered #168 -> #167):

  1. drop a hash-input field in code             -> module-side drift catch
  2. mutate hash schema version                  -> module-side drift catch
  3. alter hash-output length (truncate / extend) -> formula-side drift catch
  4. scrub the canonical-json normaliser         -> determinism drift catch

Plus a fail-closed clean-state test and a missing-canonical-doc test --
the standard fail-closed surface every parity check exposes (mirrors
``test_check_drift_fast_lane_promote_threshold_parity.py``).

Class anchor: [[canonical-inline-copy-parity-bug-class]] (7th confirmed
instance, 2026-05-20).

Design grounding: ``docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.check_drift import check_fast_lane_structural_hash_schema_parity
from scripts.research import fast_lane_structural_hash as flsh


# ----------------------------------------------------------------------
# Clean-state baseline
# ----------------------------------------------------------------------


def test_clean_state_passes():
    """Real module constants vs real design doc must match."""
    violations = check_fast_lane_structural_hash_schema_parity()
    assert violations == [], f"unexpected parity violations on clean state: {violations}"


# ----------------------------------------------------------------------
# Fail-closed: missing canonical design doc
# ----------------------------------------------------------------------


def test_missing_design_doc_fails_closed(tmp_path: Path):
    """If the canonical Stage 2A design doc is unreachable the check
    returns a single violation rather than silently passing."""
    forged = tmp_path / "does-not-exist.md"
    violations = check_fast_lane_structural_hash_schema_parity(design_doc_path=forged)
    assert violations, "missing canonical design doc must NOT pass silently"
    assert len(violations) == 1
    assert "canonical design doc missing" in violations[0]
    assert str(forged) in violations[0]


# ----------------------------------------------------------------------
# Injection 1: drop a hash-input field from the module
# ----------------------------------------------------------------------


def test_drift_drop_schema_input_field_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Removing an input field from the module's HASH_SCHEMA_INPUTS while
    the doc still lists all nine must produce a violation that names
    HASH_SCHEMA_INPUTS and cites the canonical source."""
    truncated = tuple(name for name in flsh.HASH_SCHEMA_INPUTS if name != "direction")
    monkeypatch.setattr(flsh, "HASH_SCHEMA_INPUTS", truncated)

    violations = check_fast_lane_structural_hash_schema_parity()

    assert violations, "HASH_SCHEMA_INPUTS drop went undetected"
    relevant = [v for v in violations if "HASH_SCHEMA_INPUTS" in v]
    assert relevant, f"no violation mentioned HASH_SCHEMA_INPUTS: {violations}"
    assert "direction" in relevant[0], f"violation should cite the missing field by name: {relevant[0]}"
    assert "canonical-inline-copy-parity-bug-class" in relevant[0], "violation must point to the bug-class anchor"


# ----------------------------------------------------------------------
# Injection 2: mutate HASH_SCHEMA_VERSION
# ----------------------------------------------------------------------


def test_drift_mutate_schema_version_is_caught(monkeypatch: pytest.MonkeyPatch):
    """Bumping the module's HASH_SCHEMA_VERSION without amending the doc
    must produce a violation that names HASH_SCHEMA_VERSION and shows
    both values."""
    monkeypatch.setattr(flsh, "HASH_SCHEMA_VERSION", 999)

    violations = check_fast_lane_structural_hash_schema_parity()

    assert violations, "HASH_SCHEMA_VERSION drift went undetected"
    relevant = [v for v in violations if "HASH_SCHEMA_VERSION" in v]
    assert relevant, f"no violation mentioned HASH_SCHEMA_VERSION: {violations}"
    assert "999" in relevant[0]
    assert "canonical hash_schema_version" in relevant[0]


# ----------------------------------------------------------------------
# Injection 3: alter hash-output length
# ----------------------------------------------------------------------


def test_drift_hash_output_length_is_caught(monkeypatch: pytest.MonkeyPatch):
    """A truncated digest (e.g. [:8] instead of [:16]) silently changes
    every hash. The parity check's hash-output invariant must catch this
    on a fixture-driven probe."""

    def _short_hash(inputs: dict[str, Any]) -> str:
        # Compute the real hash, then truncate -- mirrors what a scrub
        # of `digest[:16]` -> `digest[:8]` would look like at runtime.
        full = compute_structural_hash_original(inputs)
        return full[:8]

    # Save the un-monkeypatched function so the helper can compute against
    # the real implementation while the public API is overridden.
    compute_structural_hash_original = flsh.compute_structural_hash
    monkeypatch.setattr(flsh, "compute_structural_hash", _short_hash)

    violations = check_fast_lane_structural_hash_schema_parity()

    assert violations, "hash-output length drift went undetected"
    relevant = [v for v in violations if "len=" in v or "16" in v]
    assert relevant, f"no violation cited output length: {violations}"
    assert any("expected exactly 16" in v for v in violations), (
        f"violation should state the expected length: {violations}"
    )


# ----------------------------------------------------------------------
# Injection 4: scrub the canonical-json normaliser (non-determinism)
# ----------------------------------------------------------------------


def test_drift_canonical_json_normaliser_scrub_is_caught(
    monkeypatch: pytest.MonkeyPatch,
):
    """If a refactor removes sort_keys=True or otherwise lets dict order
    leak into the hash, two structurally-identical calls can produce
    different output. The parity check's determinism invariant runs the
    function twice and must catch this.

    Probe: monkeypatch a function that returns a different 16-hex string
    on every call (simulating a hash that incorporates wall-clock time or
    a random salt). Whether the underlying source is sort_keys removal or
    a timestamp leak, the symptom -- non-equal outputs for identical
    inputs -- is the same.
    """
    counter = {"n": 0}

    def _nondeterministic_hash(inputs: dict[str, Any]) -> str:
        counter["n"] += 1
        # Produce a valid-shape (16-char lowercase hex) output that differs
        # on each call. This passes the output-length / format probe but
        # fails the determinism probe.
        suffix = f"{counter['n']:08x}"
        return ("a" * 8) + suffix

    monkeypatch.setattr(flsh, "compute_structural_hash", _nondeterministic_hash)

    violations = check_fast_lane_structural_hash_schema_parity()

    assert violations, "canonical-json normaliser scrub went undetected"
    relevant = [v for v in violations if "non-deterministic" in v]
    assert relevant, f"no violation cited non-determinism: {violations}"
    assert "canonical-json" in relevant[0].lower() or "normaliser" in relevant[0].lower(), (
        f"violation should reference the canonical-json normaliser: {relevant[0]}"
    )
