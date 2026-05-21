"""Tests for scripts/research/fast_lane_structural_hash.py (Stage 2A.1).

Eight tests = 6 unit + 2 property.

Unit (one per canonical-source delegation OR input-sensitivity dimension):
  1. instrument input changes the hash AND is validated against
     pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS (unknown -> ValueError).
  2. orb_label input changes the hash AND is validated against
     pipeline.dst.SESSION_CATALOG (unknown -> ValueError).
  3. filter_type input changes the hash AND is validated against
     trading_app.config.ALL_FILTERS (unknown -> ValueError).
  4. orb_minutes / rr_target / confirm_bars: every numeric input bin
     produces a distinct hash (sensitivity to the structural axes the
     suppression rules need).
  5. entry_model and direction are case-normalised (``"e1"`` == ``"E1"``,
     ``"long"`` == ``"LONG"``) so callers cannot accidentally split a
     lane into two hashes.
  6. Missing-key / extra-key / wrong-type inputs fail-closed
     (KeyError / ValueError / TypeError) per the public-API contract.

Property:
  7. Dict insertion-order independence -- hash is identical regardless of
     the order keys are inserted into the input dict (canonical_json sorts).
  8. Reproducibility -- two calls with structurally identical inputs
     produce identical 16-char lowercase-hex output across runs.

Design grounding: ``docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md``.
Implementation grounding: ``docs/runtime/stages/2026-05-20-fast-lane-anti-fp-implementation.md``.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.dst import SESSION_CATALOG
from scripts.research.fast_lane_structural_hash import (
    HASH_SCHEMA_INPUTS,
    HASH_SCHEMA_VERSION,
    compute_structural_hash,
)
from trading_app.config import ALL_FILTERS

# ----------------------------------------------------------------------
# Canonical-source-valid fixture builders
# ----------------------------------------------------------------------
#
# Building fixtures from canonical sources (rather than literal "MNQ" /
# "US_DATA_1000" strings) means a future canonical-source amendment that
# renames an instrument or session cannot silently break this test file;
# it would either pick up the new canonical name automatically or fail
# loudly at fixture construction. Matches the parity-check convention.


def _two_canonical_instruments() -> tuple[str, str]:
    """Return two distinct canonical instruments, deterministically ordered."""
    sorted_instruments = sorted(ACTIVE_ORB_INSTRUMENTS)
    assert len(sorted_instruments) >= 2, (
        f"test fixture: ACTIVE_ORB_INSTRUMENTS must contain >= 2 instruments (got {sorted_instruments})"
    )
    return sorted_instruments[0], sorted_instruments[1]


def _two_canonical_orb_labels() -> tuple[str, str]:
    sorted_labels = sorted(SESSION_CATALOG.keys())
    assert len(sorted_labels) >= 2, f"test fixture: SESSION_CATALOG must contain >= 2 labels (got {sorted_labels})"
    return sorted_labels[0], sorted_labels[1]


def _two_canonical_filters() -> tuple[str, str]:
    sorted_filters = sorted(ALL_FILTERS.keys())
    assert len(sorted_filters) >= 2, f"test fixture: ALL_FILTERS must contain >= 2 entries (got {sorted_filters})"
    return sorted_filters[0], sorted_filters[1]


def _base_inputs() -> dict[str, Any]:
    """Canonical-source-valid baseline. Mutate one field per test."""
    instrument_a, _ = _two_canonical_instruments()
    orb_label_a, _ = _two_canonical_orb_labels()
    filter_a, _ = _two_canonical_filters()
    return {
        "instrument": instrument_a,
        "orb_label": orb_label_a,
        "orb_minutes": 5,
        "rr_target": 1.0,
        "entry_model": "E1",
        "confirm_bars": 1,
        "filter_type": filter_a,
        "direction": "LONG",
        "filter_threshold": "",
    }


# ----------------------------------------------------------------------
# Unit 1: instrument
# ----------------------------------------------------------------------


def test_instrument_changes_hash_and_validates_against_canonical():
    """Two different canonical instruments -> two different hashes.

    Also confirms the canonical-source delegation: an instrument not in
    ACTIVE_ORB_INSTRUMENTS raises ValueError, NOT a silently-accepted
    hash collision with a real instrument.
    """
    instrument_a, instrument_b = _two_canonical_instruments()
    base = _base_inputs()
    base["instrument"] = instrument_a
    hash_a = compute_structural_hash(base)

    base["instrument"] = instrument_b
    hash_b = compute_structural_hash(base)

    assert hash_a != hash_b, (
        f"instrument is structurally significant: {instrument_a!r} and "
        f"{instrument_b!r} must hash differently (got {hash_a!r}=={hash_b!r})"
    )

    base["instrument"] = "FAKE_INSTRUMENT_DOES_NOT_EXIST"
    with pytest.raises(ValueError, match="not in canonical"):
        compute_structural_hash(base)


# ----------------------------------------------------------------------
# Unit 2: orb_label
# ----------------------------------------------------------------------


def test_orb_label_changes_hash_and_validates_against_canonical():
    """Two canonical session keys -> two distinct hashes; unknown -> ValueError."""
    label_a, label_b = _two_canonical_orb_labels()
    base = _base_inputs()
    base["orb_label"] = label_a
    hash_a = compute_structural_hash(base)

    base["orb_label"] = label_b
    hash_b = compute_structural_hash(base)

    assert hash_a != hash_b, (
        f"orb_label is structurally significant: {label_a!r} and {label_b!r} "
        f"must hash differently (got {hash_a!r}=={hash_b!r})"
    )

    base["orb_label"] = "NOT_A_REAL_SESSION"
    with pytest.raises(ValueError, match="not in canonical"):
        compute_structural_hash(base)


# ----------------------------------------------------------------------
# Unit 3: filter_type
# ----------------------------------------------------------------------


def test_filter_type_changes_hash_and_validates_against_canonical():
    """Two canonical filter keys -> two distinct hashes; unknown -> ValueError.

    Also confirms the explicit no-filter convention: empty string is
    accepted as a sentinel without consulting ALL_FILTERS.
    """
    filter_a, filter_b = _two_canonical_filters()
    base = _base_inputs()
    base["filter_type"] = filter_a
    hash_a = compute_structural_hash(base)

    base["filter_type"] = filter_b
    hash_b = compute_structural_hash(base)

    assert hash_a != hash_b, (
        f"filter_type is structurally significant: {filter_a!r} and "
        f"{filter_b!r} must hash differently (got {hash_a!r}=={hash_b!r})"
    )

    base["filter_type"] = "FAKE_FILTER_KEY"
    with pytest.raises(ValueError, match="not in canonical"):
        compute_structural_hash(base)

    base["filter_type"] = ""  # explicit no-filter
    h_empty = compute_structural_hash(base)
    assert re.fullmatch(r"[0-9a-f]{16}", h_empty), (
        f"no-filter sentinel must still produce a valid hash, got {h_empty!r}"
    )


# ----------------------------------------------------------------------
# Unit 4: numeric inputs (orb_minutes, rr_target, confirm_bars)
# ----------------------------------------------------------------------


def test_numeric_axes_produce_distinct_hashes():
    """Each numeric input bin produces a distinct hash from baseline.

    Covers all three numeric structural axes the fast-lane scanner uses:
    orb_minutes (5/15/30), rr_target (1.0/1.5/2.0), confirm_bars (0/1/2).
    If any axis silently collapses, two distinct lanes would share a hash
    and de-dup / suppression would over-fire.
    """
    base = _base_inputs()
    h_base = compute_structural_hash(base)

    for key, alt in [
        ("orb_minutes", 30),
        ("rr_target", 2.0),
        ("confirm_bars", 2),
    ]:
        variant = dict(base)
        variant[key] = alt
        h_alt = compute_structural_hash(variant)
        assert h_alt != h_base, (
            f"{key} axis collapsed: base={base[key]!r} -> {h_base!r}, variant={alt!r} -> {h_alt!r} (must differ)"
        )

    # rr_target equivalence: 1.0 == 1 == 1.00 (rounded to 1dp).
    base["rr_target"] = 1
    h_int = compute_structural_hash(base)
    base["rr_target"] = 1.0
    h_float = compute_structural_hash(base)
    base["rr_target"] = 1.00
    h_pad = compute_structural_hash(base)
    assert h_int == h_float == h_pad, (
        f"rr_target rounding broken: 1, 1.0, 1.00 must hash identically (got {h_int!r}, {h_float!r}, {h_pad!r})"
    )


# ----------------------------------------------------------------------
# Unit 5: case normalisation on entry_model + direction
# ----------------------------------------------------------------------


def test_case_normalisation_on_entry_model_and_direction():
    """Lowercase variants of entry_model / direction must collide with
    uppercase canonical -- otherwise the same lane splits into two hashes
    and de-dup misses sibling-retest."""
    base = _base_inputs()
    base["entry_model"] = "E1"
    base["direction"] = "LONG"
    h_canonical = compute_structural_hash(base)

    base["entry_model"] = "e1"
    base["direction"] = "long"
    h_lower = compute_structural_hash(base)

    assert h_canonical == h_lower, (
        f"case normalisation broken on entry_model/direction: {h_canonical!r} (E1/LONG) != {h_lower!r} (e1/long)"
    )

    # Mixed-case + whitespace stripping
    base["entry_model"] = "  E1  "
    base["direction"] = "Long"
    h_mixed = compute_structural_hash(base)
    assert h_canonical == h_mixed, f"whitespace + mixed-case normalisation broken: {h_canonical!r} != {h_mixed!r}"


# ----------------------------------------------------------------------
# Unit 6: missing / extra / wrong-type inputs fail-closed
# ----------------------------------------------------------------------


def test_missing_extra_and_wrong_type_inputs_fail_closed():
    """Public API contract: missing key -> KeyError, extra key -> ValueError,
    wrong type -> TypeError. Silent acceptance of any of these would let
    a caller produce a hash whose meaning drifts from the schema."""
    base = _base_inputs()

    incomplete = dict(base)
    del incomplete["instrument"]
    with pytest.raises(KeyError, match="missing required keys"):
        compute_structural_hash(incomplete)

    extra = dict(base)
    extra["unknown_field"] = "garbage"
    with pytest.raises(ValueError, match="unknown keys"):
        compute_structural_hash(extra)

    wrong_type = dict(base)
    wrong_type["orb_minutes"] = "5"  # str instead of int
    with pytest.raises(TypeError, match="expected int"):
        compute_structural_hash(wrong_type)


# ----------------------------------------------------------------------
# Property 7: dict insertion-order independence
# ----------------------------------------------------------------------


def test_dict_insertion_order_does_not_change_hash():
    """canonical_json must sort keys -- identical content built via two
    different insertion orders produces the same hash. Catches a regression
    where sort_keys=True is silently removed."""
    base = _base_inputs()

    # Reverse-order build using the canonical schema order.
    reverse: dict[str, Any] = {}
    for key in reversed(HASH_SCHEMA_INPUTS):
        reverse[key] = base[key]

    h_forward = compute_structural_hash(base)
    h_reverse = compute_structural_hash(reverse)

    assert h_forward == h_reverse, (
        f"hash depends on dict insertion order: forward-order={h_forward!r}, "
        f"reverse-order={h_reverse!r}. canonical_json normaliser broken "
        "(sort_keys=True scrubbed?)"
    )


# ----------------------------------------------------------------------
# Property 8: reproducibility across calls
# ----------------------------------------------------------------------


def test_reproducibility_across_calls_and_format_invariants():
    """Same inputs, two calls -> bit-identical output. Hash is 16-char
    lowercase hex. Schema version is the current canonical version.

    Catches non-determinism leaks (e.g. accidental inclusion of
    datetime.utcnow() into the hashed payload) and format drift (e.g.
    truncating to a different length, switching to base64)."""
    base = _base_inputs()

    h1 = compute_structural_hash(base)
    h2 = compute_structural_hash(dict(base))  # fresh dict, same contents

    assert h1 == h2, f"hash is non-deterministic across calls: {h1!r} != {h2!r}"
    assert isinstance(h1, str) and len(h1) == 16, (
        f"hash format drift: expected 16-char str, got "
        f"{type(h1).__name__} len={len(h1) if isinstance(h1, str) else 'n/a'}"
    )
    assert re.fullmatch(r"[0-9a-f]{16}", h1), f"hash format drift: expected lowercase hex, got {h1!r}"

    # Schema version sanity (cheap belt-and-braces; the parity check is
    # the real enforcement, but a wildly wrong version here would mean a
    # tester edited the wrong module).
    assert isinstance(HASH_SCHEMA_VERSION, int) and HASH_SCHEMA_VERSION >= 1, (
        f"HASH_SCHEMA_VERSION must be a positive int, got {type(HASH_SCHEMA_VERSION).__name__}={HASH_SCHEMA_VERSION!r}"
    )
