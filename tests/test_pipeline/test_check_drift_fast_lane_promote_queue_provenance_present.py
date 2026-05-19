"""Injection tests for Check #172 (fast_lane_promote_queue_provenance_present).

Stage 2A.3 canonical-inline-copy parity (10th instance):
canonical = ``docs/runtime/stages/2026-05-20-fast-lane-anti-fp-2a3-scanner-bridge-wiring.md`` § Suppression Status Enum;
inline   = ``scripts.research.fast_lane_promote_queue.STATUS_VALUES``.

Each test mutates exactly ONE input -- the on-disk cache OR the inline
STATUS_VALUES tuple -- and asserts Check #172 catches it. Sibling-coverage
doctrine (per ``memory/feedback_regex_alternation_sibling_coverage.md``):
one test per gated constant. The single gated constant here is
``STATUS_VALUES``; mutation tests for each of the 6 SUPPRESSED_* tokens
collapse into one parametrised test that walks the tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.check_drift import (
    check_fast_lane_promote_queue_provenance_present,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "docs" / "runtime" / "promote_queue.yaml"


def _read_cache_payload() -> dict[str, Any]:
    payload = yaml.safe_load(CACHE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_cache_payload(payload: dict[str, Any]) -> None:
    CACHE_PATH.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@pytest.fixture
def restore_cache():
    """Snapshot + restore the live cache so each mutation test is hermetic."""
    snapshot = CACHE_PATH.read_text(encoding="utf-8")
    try:
        yield
    finally:
        CACHE_PATH.write_text(snapshot, encoding="utf-8")


def _first_gated_entry_index(payload: dict[str, Any]) -> int:
    """Return the index of the first entry whose status is gated by the check.

    REVOKED + ERROR are not gated; PARKED + QUEUED + SUPPRESSED_* are. The
    live cache has at least one PARKED entry today; if a future
    refactor drops it, this fixture skips rather than silently passes.
    """
    gated = {
        "QUEUED",
        "ESCALATED",
        "PARKED",
        "REJECTED_OOS_UNPOWERED",
        "SUPPRESSED_BANNED_ENTRY_MODEL",
        "SUPPRESSED_E2_LOOKAHEAD",
        "SUPPRESSED_GRAVEYARD",
        "SUPPRESSED_DUPLICATE_ACTIVE",
        "SUPPRESSED_SIBLING_RETEST",
        "SUPPRESSED_K_OVERRUN",
    }
    for idx, entry in enumerate(payload.get("entries") or []):
        if isinstance(entry, dict) and entry.get("status") in gated:
            return idx
    pytest.skip("no gated entries in current cache; cannot inject")


def test_baseline_cache_passes(restore_cache):
    """The live cache must pass Check #172 in its current state."""
    violations = check_fast_lane_promote_queue_provenance_present()
    assert violations == [], (
        "baseline cache fails Check #172: " f"{violations}"
    )


def test_missing_structural_hash_caught(restore_cache):
    """Stripping structural_hash from a gated entry fails the check."""
    payload = _read_cache_payload()
    idx = _first_gated_entry_index(payload)
    payload["entries"][idx]["structural_hash"] = None
    _write_cache_payload(payload)

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [v for v in violations if "structural_hash" in v]
    assert matching, (
        "Check #172 did not catch a stripped structural_hash; "
        f"got violations: {violations}"
    )


def test_null_k_lineage_field_caught(restore_cache):
    """Removing one required key from k_lineage fails the check."""
    payload = _read_cache_payload()
    idx = _first_gated_entry_index(payload)
    kl = payload["entries"][idx].get("k_lineage")
    assert isinstance(kl, dict), "fixture: k_lineage must be a dict to mutate"
    del kl["K_lane"]
    _write_cache_payload(payload)

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [v for v in violations if "K_lane" in v or "missing required keys" in v]
    assert matching, (
        "Check #172 did not catch a missing k_lineage.K_lane key; "
        f"got violations: {violations}"
    )


def test_missing_n_hat_caught(restore_cache):
    """Removing n_hat from a gated entry fails the check."""
    payload = _read_cache_payload()
    idx = _first_gated_entry_index(payload)
    payload["entries"][idx]["n_hat"] = None
    _write_cache_payload(payload)

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [v for v in violations if "n_hat" in v]
    assert matching, (
        "Check #172 did not catch a stripped n_hat; "
        f"got violations: {violations}"
    )


def test_mutated_rho_hat_caught(restore_cache):
    """Mutating rho_hat_assumed away from 0.5 fails the check."""
    payload = _read_cache_payload()
    idx = _first_gated_entry_index(payload)
    payload["entries"][idx]["k_lineage"]["rho_hat_assumed"] = 0.6
    _write_cache_payload(payload)

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [v for v in violations if "rho_hat_assumed" in v]
    assert matching, (
        "Check #172 did not catch rho_hat_assumed=0.6; "
        f"got violations: {violations}"
    )


def test_tampered_banner_caught(restore_cache):
    """Erasing the canonical 'DERIVED STATE' banner fails the check."""
    snapshot = CACHE_PATH.read_text(encoding="utf-8")
    CACHE_PATH.write_text(
        snapshot.replace("DERIVED STATE - do not hand-edit", "REGULAR STATE"),
        encoding="utf-8",
    )

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [v for v in violations if "BANNER TAMPERED" in v]
    assert matching, (
        "Check #172 did not catch a tampered DERIVED-STATE banner; "
        f"got violations: {violations}"
    )


# ---- STATUS_VALUES sibling-coverage tests --------------------------------
#
# Six SUPPRESSED_* tokens are the canonical-inline-copy alphabet. Mutate
# the inline tuple (one token per parametrisation) and assert the parity
# branch of Check #172 catches it. Per
# feedback_regex_alternation_sibling_coverage.md, every member of the
# alternation must be exercised so a future refactor that drops one token
# is caught by mutation-probe.


SUPPRESSION_TOKENS = (
    "SUPPRESSED_GRAVEYARD",
    "SUPPRESSED_DUPLICATE_ACTIVE",
    "SUPPRESSED_SIBLING_RETEST",
    "SUPPRESSED_BANNED_ENTRY_MODEL",
    "SUPPRESSED_E2_LOOKAHEAD",
    "SUPPRESSED_K_OVERRUN",
)


@pytest.fixture
def restore_status_values():
    """Snapshot + restore the inline STATUS_VALUES so each mutation is hermetic."""
    import scripts.research.fast_lane_promote_queue as scanner_mod

    snapshot = scanner_mod.STATUS_VALUES
    try:
        yield
    finally:
        scanner_mod.STATUS_VALUES = snapshot


@pytest.mark.parametrize("token_to_drop", SUPPRESSION_TOKENS)
def test_dropped_inline_suppression_token_caught(
    token_to_drop: str, restore_status_values
):
    """Dropping any of the 6 inline tokens fails the parity branch."""
    import scripts.research.fast_lane_promote_queue as scanner_mod

    scanner_mod.STATUS_VALUES = tuple(
        t for t in scanner_mod.STATUS_VALUES if t != token_to_drop
    )

    violations = check_fast_lane_promote_queue_provenance_present()

    matching = [
        v
        for v in violations
        if "STATUS_VALUES drift" in v and token_to_drop in v
    ]
    assert matching, (
        f"Check #172 did not catch a dropped inline token {token_to_drop!r}; "
        f"got violations: {violations}"
    )
