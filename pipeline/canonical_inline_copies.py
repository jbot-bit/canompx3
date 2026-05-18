"""Canonical→inline copy registry (Layer 2 of 3-layer hardening).

Doctrine surface — no runtime logic, no DB access, no side effects.
Imported by ``pipeline.check_drift`` to drive the meta-check
``check_canonical_inline_copies_have_parity_check`` (Check #159 per the
Stage 2 plan at ``docs/runtime/stages/2026-05-19-stage-2-canonical-inline-copies-meta-registry.md``).

The class-level bug pattern this registry enforces is described in
``memory/feedback_canonical_inline_copy_parity_bug_class.md``: any inline
literal copy of a canonical value silently drifts when the canonical is
amended. The n=3+ doctrine threshold
(``memory/feedback_n3_same_class_doctrine_threshold.md``) mandates
structural enforcement once three same-class instances have surfaced.

What COUNTS as a value-parity inline (TRUE positive):
    A literal Python expression (string, number, set, dict) whose bytes
    match the canonical source AND would not auto-update if the canonical
    source mutates. Test: change the canonical; does the consumer use the
    new value at next import? If yes → re-export, NOT a parity inline.

What DOES NOT count (FALSE positives observed in the May 2026 audit —
see ``memory/feedback_grep_candidate_to_seed_value_parity_required.md``):

    1. Runtime re-export — ``from m import X; LOCAL = list(X)`` (e.g.,
       ``trading_app/config.py:4203`` TRADEABLE_INSTRUMENTS). Comment cites
       the canonical, but value is not inlined.
    2. DDL prose comment — ``# Mirrors experimental_strategies...`` over a
       CREATE TABLE block (e.g., ``trading_app/regime/schema.py:47``).
       Documents schema lineage, no value to mirror.
    3. Deprecation notice — ``# DEPRECATED: see X``; the inline is en route
       to deletion, not parity maintenance.
    4. Docstring narrative — module preamble explaining what the canonical
       is, no Python literal at all.

Adding a new entry requires:
    a) A dedicated parity-check function in ``pipeline.check_drift``
       module globals named exactly ``parity_check_func``.
    b) A test file at ``tests/test_pipeline/<test_slug>.py`` with
       ``>= len(gated_constants)`` test functions (sibling-coverage
       doctrine — see
       ``memory/feedback_regex_alternation_sibling_coverage.md``).
    c) For canonical sources governed by the supersession-banner pattern
       (e.g., ``pre_registered_criteria.md``), the parity check MUST PARSE
       the canonical doc at runtime, not hardcode the expected literal —
       see ``memory/feedback_chordia_threshold_doctrine_supersession_layer_trap.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InlineCopyPair:
    """One registered canonical→inline parity pair.

    Attributes
    ----------
    slug : str
        Stable identifier — drives the test-file convention
        (``tests/test_pipeline/<test_slug>.py``) and shows up in
        meta-check violation messages.
    inline_site : str
        ``relative/path/to/file.py:LINE`` of the consumer that inlines.
    canonical_source : str
        ``relative/path/to/file.py:LINE`` or
        ``relative/path/to/doc.md:LINE`` of the truth source.
    gated_constants : tuple[str, ...]
        Names of the constants/keys covered by this pair. The
        sibling-coverage doctrine requires one test function per name.
    parity_check_func : str
        The function name in ``pipeline.check_drift`` that enforces this
        pair. Meta-check asserts the function exists, is callable, and
        lives in ``vars(check_drift)`` (no ``getattr`` fallback — aliased
        imports must not satisfy the gate).
    test_slug : str
        Filename stem at ``tests/test_pipeline/<test_slug>.py``.
    rationale : str
        Why this pair is value-parity (not re-export, not prose).
    canonical_class : str
        ``"value"`` (literal byte parity), ``"doctrine"`` (parsed from
        doctrine doc — see Chordia note above), or ``"derived"``
        (computed from canonical at check time).
    """

    slug: str
    inline_site: str
    canonical_source: str
    gated_constants: tuple[str, ...]
    parity_check_func: str
    test_slug: str
    rationale: str
    canonical_class: str = "value"
    bug_class_anchor: str = "feedback_canonical_inline_copy_parity_bug_class.md"


# Seed list — verified during 2026-05-19 Stage 2 truth audit.
# Three TRUE positives admitted; two FALSE positives rejected on value-parity grounds:
#
#   REJECTED — trading_app/config.py:4203 TRADEABLE_INSTRUMENTS
#     `from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS` followed by
#     `TRADEABLE_INSTRUMENTS = list(ACTIVE_ORB_INSTRUMENTS)` is a runtime
#     re-export. No literal bytes inlined; canonical mutation propagates at
#     next import. Not a parity candidate.
#
#   REJECTED — trading_app/regime/schema.py:47
#     `# Mirrors experimental_strategies + run_label, start_date, end_date`
#     prose comment over a CREATE TABLE block. Documents schema lineage;
#     no Python literal in scope. Not a parity candidate.
CANONICAL_INLINE_COPIES: list[InlineCopyPair] = [
    InlineCopyPair(
        slug="c8_fail_labels",
        inline_site="pipeline/check_drift.py:9459",
        canonical_source="trading_app/lane_allocator.py:812",
        gated_constants=(
            "FAILED_RATIO",
            "NEGATIVE_OOS_EXPR",
            "NO_OOS_DATA",
            "INSUFFICIENT_N_PATHWAY_B_REJECT",
            "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH",
            "REJECTED",
        ),
        parity_check_func="check_c8_fail_labels_parity",
        test_slug="test_canonical_inline_copies_registry",
        rationale=(
            "Verbatim 6-element string set duplicated at both sites. "
            "Inline at check_drift.py:9459 self-cites canonical "
            "(comment: 'Mirrors apply_c8_gate._C8_FAIL_LABELS'). "
            "If the allocator adds a 7th label (e.g., NO_CANONICAL_BASELINE), "
            "the drift check silently misses it — fail-open allocator-class "
            "bug per feedback_allocator_gate_class_pattern_fail_open.md."
        ),
        canonical_class="value",
    ),
    InlineCopyPair(
        slug="calibrate_null_sigmas",
        inline_site="scripts/tools/calibrate_null_sigma.py:51",
        canonical_source="scripts/tests/run_null_batch.py:42",
        gated_constants=("MGC", "MNQ", "MES"),
        parity_check_func="check_calibrate_null_sigmas_parity",
        test_slug="test_canonical_inline_copies_registry",
        rationale=(
            "calibrate_null_sigma.py:51 CURRENT_SIGMAS={MGC:1.2,MNQ:5.0,MES:1.1} "
            "is a value-parity copy of run_null_batch.py:42 "
            "INSTRUMENT_NULL_PARAMS[*]['sigma']. Inline self-cites canonical "
            "(comment: 'Canonical source: scripts/tests/run_null_batch.py "
            "INSTRUMENT_NULL_PARAMS'). Drift hazard: sigma recalibration of "
            "run_null_batch updates the producer but calibrate's 'current "
            "value' display lies — operator sees false no-change-needed."
        ),
        canonical_class="value",
    ),
    InlineCopyPair(
        slug="criterion_ladder_chordia_thresholds",
        inline_site="scripts/tools/criterion_ladder_check.py:43-44",
        canonical_source="docs/institutional/pre_registered_criteria.md (Criterion 4 row)",
        gated_constants=("T_THRESHOLD_WITH_THEORY", "T_THRESHOLD_NO_THEORY"),
        parity_check_func="check_criterion_ladder_chordia_thresholds_parity",
        test_slug="test_canonical_inline_copies_registry",
        rationale=(
            "criterion_ladder_check.py:43-44 inlines T_THRESHOLD_WITH_THEORY=3.00 "
            "and T_THRESHOLD_NO_THEORY=3.79 as module constants with prose "
            "comment 'Locked thresholds from pre_registered_criteria.md "
            "acceptance matrix (line 262-280)'. The doctrine doc uses the "
            "supersession-banner amendment pattern — Criterion 4 lines 381-388 "
            "already describe banded amendments. Parity check MUST parse the "
            "doc at runtime; hardcoded literal would just relocate the same "
            "inline-copy bug class one layer up. See "
            "feedback_chordia_threshold_doctrine_supersession_layer_trap.md."
        ),
        canonical_class="doctrine",
    ),
]


# Defensive: enforce per-entry uniqueness at module import time. A duplicate
# slug or parity_check_func would silently make the meta-check redundant on
# one entry and blind on another.
def _validate_registry(entries: list[InlineCopyPair]) -> None:
    seen_slugs: set[str] = set()
    seen_funcs: set[str] = set()
    for entry in entries:
        if entry.slug in seen_slugs:
            raise ValueError(
                f"CANONICAL_INLINE_COPIES: duplicate slug {entry.slug!r} "
                "(registry-integrity violation)"
            )
        seen_slugs.add(entry.slug)
        if entry.parity_check_func in seen_funcs:
            raise ValueError(
                f"CANONICAL_INLINE_COPIES: duplicate parity_check_func "
                f"{entry.parity_check_func!r} for slug {entry.slug!r} "
                "(two entries cannot share a parity check — meta-check "
                "would be blind on one of them)"
            )
        seen_funcs.add(entry.parity_check_func)
        if entry.canonical_class not in ("value", "doctrine", "derived"):
            raise ValueError(
                f"CANONICAL_INLINE_COPIES: entry {entry.slug!r} has unknown "
                f"canonical_class={entry.canonical_class!r} "
                "(expected one of: value, doctrine, derived)"
            )
        if not entry.gated_constants:
            raise ValueError(
                f"CANONICAL_INLINE_COPIES: entry {entry.slug!r} has empty "
                "gated_constants (sibling-coverage doctrine requires "
                ">=1 named constant per entry)"
            )


_validate_registry(CANONICAL_INLINE_COPIES)


__all__ = ["InlineCopyPair", "CANONICAL_INLINE_COPIES"]
