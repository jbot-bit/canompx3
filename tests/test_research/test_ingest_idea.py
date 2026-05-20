"""Tests for ``scripts/research/ingest_idea.py`` (Stage A).

Three tests:

1. **Schema parity** — Stage A emitter output has the same top-level structural
   schema as a 2026-05-18 in-the-wild fast-lane v5.1 pre-reg. Catches drift
   between the emitter and the canonical schema.
2. **Refusal delegation** — Each refusal gate fires on a known-bad input
   (unknown instrument, unknown session, E2 + banned filter, empty mechanism,
   unknown literature slug).
3. **Output round-trips through yaml.safe_load** — guards against the emitter
   accidentally producing invalid YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.research.ingest_idea import (
    IngestRefused,
    build_prereg,
    build_slug,
    build_strategy_id,
    run_gates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FASTLANE_PREREG = (
    REPO_ROOT
    / "docs"
    / "audit"
    / "hypotheses"
    / "2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.yaml"
)

# A valid scope known to pass every gate (matches the canonical baseline).
_VALID_KWARGS: dict = dict(
    instrument="MES",
    session="CME_PRECLOSE",
    orb_minutes=30,
    entry_model="E2",
    confirm_bars=1,
    rr=1.0,
    direction="pooled",
    filter_type="COST_LT15",
    mechanism="cost-fraction floor predicts post-break participation",
    literature_slug="fitschen_2013_path_of_least_resistance",
)


def test_canonical_baseline_exists() -> None:
    """If the baseline file moves, the parity test below loses its anchor."""
    assert CANONICAL_FASTLANE_PREREG.exists(), (
        f"Schema parity baseline missing: {CANONICAL_FASTLANE_PREREG}. "
        "Update test_ingest_idea.py CANONICAL_FASTLANE_PREREG to point at "
        "another 2026-05-18 fast-lane v5.1 pre-reg."
    )


def test_schema_parity_against_canonical_fastlane_prereg() -> None:
    """Emitted YAML has the same top-level keys as a real fast-lane pre-reg."""
    canonical = yaml.safe_load(CANONICAL_FASTLANE_PREREG.read_text(encoding="utf-8"))
    emitted = build_prereg(**_VALID_KWARGS)

    # Top-level structural keys must match. Order doesn't matter; presence does.
    assert set(canonical.keys()) == set(emitted.keys()), (
        f"Top-level key drift. "
        f"canonical-only: {set(canonical.keys()) - set(emitted.keys())}; "
        f"emitted-only: {set(emitted.keys()) - set(canonical.keys())}"
    )

    # Required scope fields per fast_lane_to_heavyweight_bridge._REQUIRED_SCOPE_FIELDS.
    required_scope = {
        "instrument",
        "strategy_id",
        "session",
        "orb_minutes",
        "entry_model",
        "confirm_bars",
        "rr_target",
        "direction",
        "filter_type",
    }
    assert required_scope.issubset(emitted["scope"].keys()), (
        f"missing required scope fields: {required_scope - emitted['scope'].keys()}"
    )

    # metadata.theory_grant MUST be False (per evidence-auditor finding 2:
    # Amendment 3.3 in hypothesis_loader.py requires explicit bool).
    assert emitted["metadata"]["theory_grant"] is False

    # theory_citation must NOT appear in metadata (per
    # feedback_chordia_theory_citation_field_presence_trap.md).
    assert "theory_citation" not in emitted["metadata"]

    # No numeric chordia_gate_threshold (per evidence-auditor finding 5:
    # bridge emits prose only; runner resolves at execution time).
    assert "chordia_gate_threshold" not in emitted["primary_schema"]
    assert isinstance(emitted["primary_schema"]["chordia_threshold_basis"], str)


def test_refusal_unknown_instrument() -> None:
    kwargs = dict(_VALID_KWARGS, instrument="MCL")  # MCL is dead for ORB
    with pytest.raises(IngestRefused, match="instrument"):
        run_gates(**kwargs)


def test_refusal_unknown_session() -> None:
    kwargs = dict(_VALID_KWARGS, session="MIDNIGHT_FAKE")
    with pytest.raises(IngestRefused, match="session"):
        run_gates(**kwargs)


def test_refusal_e2_with_banned_filter_prefix() -> None:
    # VOL_RV20_N20 is registered in ALL_FILTERS and starts with E2_EXCLUDED prefix VOL_RV.
    kwargs = dict(_VALID_KWARGS, entry_model="E2", filter_type="VOL_RV20_N20")
    with pytest.raises(IngestRefused, match="E2_EXCLUDED_FILTER_PREFIXES"):
        run_gates(**kwargs)


def test_refusal_e2_with_banned_filter_substring() -> None:
    # ORB_G4_FAST5 is registered and contains the banned _FAST substring.
    kwargs = dict(_VALID_KWARGS, entry_model="E2", filter_type="ORB_G4_FAST5")
    with pytest.raises(IngestRefused, match="E2_EXCLUDED_FILTER_SUBSTRINGS"):
        run_gates(**kwargs)


def test_refusal_unknown_filter() -> None:
    kwargs = dict(_VALID_KWARGS, filter_type="TOTALLY_FAKE_FILTER_X")
    with pytest.raises(IngestRefused, match="ALL_FILTERS"):
        run_gates(**kwargs)


def test_refusal_empty_mechanism() -> None:
    kwargs = dict(_VALID_KWARGS, mechanism="   ")
    with pytest.raises(IngestRefused, match="mechanism"):
        run_gates(**kwargs)


def test_refusal_unknown_literature_slug() -> None:
    kwargs = dict(_VALID_KWARGS, literature_slug="nonexistent_paper_2099")
    with pytest.raises(IngestRefused, match="literature"):
        run_gates(**kwargs)


def test_refusal_bad_direction() -> None:
    kwargs = dict(_VALID_KWARGS, direction="sideways")
    with pytest.raises(IngestRefused, match="direction"):
        run_gates(**kwargs)


def test_refusal_e0_entry_model() -> None:
    # E0 is not in trading_app.config.ENTRY_MODELS (purged Feb 2026).
    kwargs = dict(_VALID_KWARGS, entry_model="E0")
    with pytest.raises(IngestRefused, match="entry_model"):
        run_gates(**kwargs)


def test_refusal_e3_blocked_for_fastlane() -> None:
    # E3 IS in canonical ENTRY_MODELS but NOT in fast-lane v5.1.
    kwargs = dict(_VALID_KWARGS, entry_model="E3")
    with pytest.raises(IngestRefused, match="fast-lane v5.1"):
        run_gates(**kwargs)


def test_output_is_valid_yaml() -> None:
    """Built dict must round-trip through yaml.safe_dump/safe_load cleanly."""
    prereg = build_prereg(**_VALID_KWARGS)
    text = yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True)
    roundtripped = yaml.safe_load(text)
    assert roundtripped["scope"]["strategy_id"] == prereg["scope"]["strategy_id"]
    assert roundtripped["metadata"]["theory_grant"] is False


def test_strategy_id_omits_o5_suffix() -> None:
    """O5 lanes drop the _O5 suffix per canonical parser default."""
    sid = build_strategy_id(
        instrument="MNQ",
        session="NYSE_OPEN",
        entry_model="E2",
        rr=1.5,
        confirm_bars=1,
        filter_type="ORB_G4",
        orb_minutes=5,
    )
    assert sid == "MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4"


def test_strategy_id_includes_o30_suffix() -> None:
    sid = build_strategy_id(
        instrument="MES",
        session="CME_PRECLOSE",
        entry_model="E2",
        rr=1.0,
        confirm_bars=1,
        filter_type="COST_LT15",
        orb_minutes=30,
    )
    assert sid == "MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_O30"


def test_slug_format_matches_in_wild_naming() -> None:
    """Slug must match the 2026-05-18 in-the-wild naming style."""
    slug = build_slug(
        instrument="MES",
        session="CME_PRECLOSE",
        entry_model="E2",
        rr=1.0,
        confirm_bars=1,
        filter_type="COST_LT15",
        orb_minutes=30,
        direction="pooled",
    )
    assert slug == "mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30"


def test_happy_path_gates_pass() -> None:
    """All gates pass on the valid baseline."""
    run_gates(**_VALID_KWARGS)  # raises on failure
