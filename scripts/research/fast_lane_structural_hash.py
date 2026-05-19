"""Canonical structural-hash for fast-lane preregs (Stage 2A.1).

Single source of truth for the 16-hex ``structural_hash`` that identifies a
"lane" — the (instrument, session, ORB minutes, RR, entry model, confirm bars,
filter, direction, filter threshold) tuple a prereg tests. Two preregs with
the same structural_hash test the same lane; that is the de-dup criterion
consumed by the Stage 2A.3 scanner suppression rules.

Design grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-trial-provenance.md
Implementation grounding:
  docs/runtime/stages/2026-05-20-fast-lane-anti-fp-implementation.md

Canonical delegations (per institutional-rigor.md § 10):
- ``pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`` -- valid instruments.
- ``pipeline.dst.SESSION_CATALOG`` -- valid orb_label keys.
- ``trading_app.config.ALL_FILTERS`` -- valid filter_type keys.

The hash excludes data window, K framing, t-stat, and any test-instance fields
deliberately: those vary across the same lane and are not part of structural
identity. Adding a new input field requires bumping HASH_SCHEMA_VERSION and
amending the parity check (#167); see canonical_inline_copies.py registry.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# HASH_SCHEMA_VERSION is the gated constant. Parity check #167 asserts this
# stays in lockstep with the documented schema (the 9 inputs below). Bump on
# any structural change to the hash recipe.
HASH_SCHEMA_VERSION = 1

# The 9 canonical inputs. Order is significant -- it determines key order in
# the canonicalised JSON payload. Adding/removing entries here requires a
# HASH_SCHEMA_VERSION bump and a corresponding update to the canonical-inline-
# copy registry entry's gated_constants.
HASH_SCHEMA_INPUTS: tuple[str, ...] = (
    "instrument",
    "orb_label",
    "orb_minutes",
    "rr_target",
    "entry_model",
    "confirm_bars",
    "filter_type",
    "direction",
    "filter_threshold",
)


def _normalise_instrument(value: Any) -> str:
    """Validate against canonical ACTIVE_ORB_INSTRUMENTS. Fail-closed on miss."""
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # local import: avoid cycle on cold start

    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.instrument: expected str, got {type(value).__name__}"
        )
    upper = value.strip().upper()
    if upper not in set(ACTIVE_ORB_INSTRUMENTS):
        raise ValueError(
            f"structural_hash.instrument: {value!r} not in canonical "
            f"ACTIVE_ORB_INSTRUMENTS {sorted(ACTIVE_ORB_INSTRUMENTS)!r}"
        )
    return upper


def _normalise_orb_label(value: Any) -> str:
    """Validate against canonical SESSION_CATALOG. Fail-closed on miss."""
    from pipeline.dst import SESSION_CATALOG  # local import: avoid cycle on cold start

    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.orb_label: expected str, got {type(value).__name__}"
        )
    upper = value.strip().upper()
    if upper not in SESSION_CATALOG:
        raise ValueError(
            f"structural_hash.orb_label: {value!r} not in canonical "
            f"SESSION_CATALOG keys {sorted(SESSION_CATALOG.keys())!r}"
        )
    return upper


def _normalise_orb_minutes(value: Any) -> int:
    """Accept 5, 15, or 30. Fail-closed on anything else."""
    if isinstance(value, bool):  # bool is int subclass; reject explicitly
        raise TypeError("structural_hash.orb_minutes: bool not accepted; expected int")
    if not isinstance(value, int):
        raise TypeError(
            f"structural_hash.orb_minutes: expected int, got {type(value).__name__}"
        )
    if value not in (5, 15, 30):
        raise ValueError(
            f"structural_hash.orb_minutes: {value!r} not in {{5, 15, 30}}"
        )
    return value


def _normalise_rr_target(value: Any) -> float:
    """Round to 1dp; reject NaN/inf. Accept int or float input."""
    if isinstance(value, bool):
        raise TypeError("structural_hash.rr_target: bool not accepted; expected number")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"structural_hash.rr_target: expected number, got {type(value).__name__}"
        )
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"structural_hash.rr_target: {value!r} is not finite")
    return round(f, 1)


def _normalise_entry_model(value: Any) -> str:
    """E1/E2 active; E0 purged; E3 graveyard-banned. Suppression rule fires
    downstream in the scanner on E0/E3 -- hash itself does not gate."""
    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.entry_model: expected str, got {type(value).__name__}"
        )
    upper = value.strip().upper()
    if upper not in {"E0", "E1", "E2", "E3"}:
        raise ValueError(
            f"structural_hash.entry_model: {value!r} not in {{E0, E1, E2, E3}}"
        )
    return upper


def _normalise_confirm_bars(value: Any) -> int:
    """Accept 0, 1, or 2. Fail-closed on anything else."""
    if isinstance(value, bool):
        raise TypeError(
            "structural_hash.confirm_bars: bool not accepted; expected int"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"structural_hash.confirm_bars: expected int, got {type(value).__name__}"
        )
    if value not in (0, 1, 2):
        raise ValueError(
            f"structural_hash.confirm_bars: {value!r} not in {{0, 1, 2}}"
        )
    return value


def _normalise_filter_type(value: Any) -> str:
    """Validate against canonical ALL_FILTERS. Empty string is the no-filter
    convention used in some preregs -- accept it explicitly."""
    from trading_app.config import ALL_FILTERS  # local import: heavy module

    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.filter_type: expected str, got {type(value).__name__}"
        )
    stripped = value.strip()
    if stripped == "":
        return ""  # explicit no-filter
    if stripped not in ALL_FILTERS:
        raise ValueError(
            f"structural_hash.filter_type: {value!r} not in canonical "
            f"ALL_FILTERS (got {len(ALL_FILTERS)} registered filters)"
        )
    return stripped


def _normalise_direction(value: Any) -> str:
    """Accept LONG, SHORT, or BOTH (case-normalised)."""
    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.direction: expected str, got {type(value).__name__}"
        )
    upper = value.strip().upper()
    if upper not in {"LONG", "SHORT", "BOTH"}:
        raise ValueError(
            f"structural_hash.direction: {value!r} not in {{LONG, SHORT, BOTH}}"
        )
    return upper


def _normalise_filter_threshold(value: Any) -> str:
    """Free-form string of bound values. Empty string accepted (no-filter)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(
            f"structural_hash.filter_threshold: expected str or None, "
            f"got {type(value).__name__}"
        )
    return value.strip()


# Map of input-name -> normaliser. Used by compute_structural_hash; the keys
# MUST match HASH_SCHEMA_INPUTS exactly. Parity check #167 asserts this.
_NORMALISERS = {
    "instrument": _normalise_instrument,
    "orb_label": _normalise_orb_label,
    "orb_minutes": _normalise_orb_minutes,
    "rr_target": _normalise_rr_target,
    "entry_model": _normalise_entry_model,
    "confirm_bars": _normalise_confirm_bars,
    "filter_type": _normalise_filter_type,
    "direction": _normalise_direction,
    "filter_threshold": _normalise_filter_threshold,
}


def compute_structural_hash(inputs: dict[str, Any]) -> str:
    """Compute the canonical 16-hex structural hash for a prereg lane.

    Parameters
    ----------
    inputs : dict
        Mapping with the 9 keys named in HASH_SCHEMA_INPUTS. Missing keys
        raise ``KeyError``; extra keys raise ``ValueError``.

    Returns
    -------
    str
        Lower-case 16-character hex string -- the first 16 hex chars of
        sha256(canonical_json(normalised_inputs)).

    Raises
    ------
    KeyError
        If any HASH_SCHEMA_INPUTS key is missing.
    ValueError
        If any value fails canonical-source validation, or if unknown keys
        are present (fail-closed to prevent silent input drift).
    TypeError
        If a value has the wrong type.
    """
    # Runtime check retained per institutional-rigor.md § 6 (fail-closed on
    # malformed input). Annotation type-narrows to dict, so Pyright flags
    # both the isinstance call and the raise as redundant/unreachable --
    # ignored for fail-closed public-API discipline.
    if not isinstance(inputs, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"compute_structural_hash: inputs must be dict, got {type(inputs).__name__}"
        )

    missing = [k for k in HASH_SCHEMA_INPUTS if k not in inputs]
    if missing:
        raise KeyError(
            f"compute_structural_hash: missing required keys {missing}; "
            f"expected exactly {list(HASH_SCHEMA_INPUTS)}"
        )

    extras = [k for k in inputs if k not in _NORMALISERS]
    if extras:
        raise ValueError(
            f"compute_structural_hash: unknown keys {extras}; expected exactly "
            f"{list(HASH_SCHEMA_INPUTS)}. Add to HASH_SCHEMA_INPUTS and bump "
            f"HASH_SCHEMA_VERSION if a new field is canonical."
        )

    # Normalise in canonical order. Building a fresh dict means insertion
    # order matches HASH_SCHEMA_INPUTS regardless of caller's dict order.
    normalised: dict[str, Any] = {}
    for key in HASH_SCHEMA_INPUTS:
        normalised[key] = _NORMALISERS[key](inputs[key])

    payload = {
        "schema_version": HASH_SCHEMA_VERSION,
        "inputs": normalised,
    }

    # sort_keys=True is belt-and-braces over the explicit ordered build above:
    # if a future refactor reorders HASH_SCHEMA_INPUTS, the hash is still
    # determined by alphabetical key order under sort_keys, so the value moves
    # but only once -- not silently per call.
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest[:16]


__all__ = [
    "HASH_SCHEMA_VERSION",
    "HASH_SCHEMA_INPUTS",
    "compute_structural_hash",
]
