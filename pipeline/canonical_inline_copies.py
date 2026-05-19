"""Canonical-inline-copy meta-registry (Layer 2 of 3-layer hardening).

Layer 2 of the canonical-inline-copy-parity-bug-class defense. Lists every
known canonical-source -> inline-copy pair in the repo. Consumed by
``pipeline.check_drift.check_canonical_inline_copies_have_parity_check``
(Check #159), which asserts every registered pair is covered by a dedicated
parity drift check and a sibling-coverage injection-test file.

Background
----------
- Layer 1 (per-pair drift check): catches drift after a canonical source
  amendment. Example: ``check_fast_lane_promote_threshold_parity`` (Check
  #158, landed in commit d88a5465).
- Layer 2 (this module + Check #159): catches an *orphan* parity check
  (registry entry pointing at a missing function or test file) and a
  registered pair lacking sibling-coverage tests.
- Layer 3 (edit-time PreToolUse hook): planned. Catches the literal at birth
  before an inline copy lands without a corresponding registry entry.

Doctrine anchors
----------------
- ``memory/feedback_canonical_inline_copy_parity_bug_class.md`` -- bug-class
  definition, 4 documented instances as of 2026-05-18.
- ``memory/feedback_n3_same_class_doctrine_threshold.md`` -- n>=3+ class
  pattern authorizes mechanical enforcement (registry + edit-time hook).
- ``memory/feedback_grep_candidate_to_seed_value_parity_required.md`` --
  4-shape false-positive taxonomy. Only literal-byte duplication counts.

Inclusion criteria
------------------
A pair belongs in this registry when ALL of the following hold:

1. The inline site stores a numeric or short-string literal whose value is
   a duplicate of a value held by a separately-maintained canonical source
   (YAML pre-reg template, Python constants module, validated_setups
   column, doctrine doc).
2. The inline site is *active* code: production, allocator, scanner,
   doctrine-driven script. Frozen one-shot research scripts (e.g.
   ``research/audit_*``, ``research/phase_d_*``) are EXCLUDED -- their
   value is anchored in time to a result MD and re-running them against
   a moved canonical value would defeat the audit, not honour it.
3. Drift on the value silently affects a live decision (capital, lane
   gating, deployment threshold, screen routing). Comment-only mirrors,
   provenance prose, runtime re-exports, DDL prose, and deprecation
   notices are EXCLUDED per the 4-shape taxonomy.

Grep-audit survivors -- rejection notes
---------------------------------------
Run on 2026-05-19 via::

    grep -rn "# canonical\\|# mirrors\\|# from canonical\\|# cite:" \\
        pipeline/ trading_app/ scripts/ research/

Rejected candidates (kept here for audit-trail; do NOT re-add without
proof of literal-byte duplication on an active path):

- ``pipeline/cost_model.py:105/138/188/200`` ``commission_rt=X``
  cite "canonical TopStep Rithmic" -- IS the canonical (cost_model is
  the source). Comment cites upstream broker rate, not project canon.
- ``scripts/ingestion/*.py`` ``DB_PATH = GOLD_DB_PATH`` -- runtime
  re-export (RHS identifier, not literal).
- ``trading_app/config.py:3447`` "canonical friction model" -- docstring
  prose, no numeric inline.
- ``research/audit_l2_atr_p50_regime_vs_arithmetic.py:63`` --
  ``ATR_P50_MIN = 50.0`` with comment citing the canonical config-module
  ATR_P50 filter. Literal-byte duplicate of
  ``ALL_FILTERS["ATR_P50"].min_pct`` BUT frozen one-shot audit script
  (last modified 2026-04-26, only referenced from result MD
  ``docs/audit/results/2026-04-21-l2-atr-p50-regime-vs-arithmetic-audit.md``).
  Result MD anchors the value -- re-running against a moved canonical
  would defeat the audit, not honour it.
- ``research/phase_d_d5_mnq_comex_settle_d5_both_sides.py:77`` same shape,
  same rejection reason.
- ``pipeline/check_drift.py:9458`` ``# Mirrors apply_c8_gate._C8_FAIL_LABELS``
  -- producer/consumer parity (different bug class, separately tracked
  per ``feedback_producer_consumer_parity_class_bug_2026_05_06.md``).
- Numerous ``Mirrors X`` / ``# canonical`` docstring-prose comments in
  ``trading_app/*.py`` and ``scripts/run_live_session.py`` -- describe
  functional similarity, not literal-value duplication.

Other documented n>=3+ bug-class instances NOT yet registered here
because they do not currently match all three inclusion criteria:

- Cost-specs class (``feedback_doctrine_drift_cost_specs_2026_05_01.md``)
  -- doctrine drift in ``.claude/rules/backtesting-methodology.md`` lagged
  the canonical ``COST_SPECS`` after F-4 fix. No active python inline-copy
  in production: the issue was rule-file lag, which is governed by
  ``check_doctrine_drift_cost_specs`` (existing). If a future production
  module inlines ``COST_SPECS`` numerics, register here.
- Allocator-gate class
  (``feedback_allocator_gate_class_pattern_fail_open.md``) -- different
  pattern: validator writes a column, allocator must read it. Tracked
  separately via ``apply_chordia_gate`` / ``apply_c8_oos_gate``.
- Chordia threshold class
  (``feedback_chordia_theory_citation_field_presence_trap.md``) --
  already covered by ``check_chordia_result_threshold_matches_prereg``;
  the inline isn't a literal-byte duplicate, it's a loader-side gate
  determined by ``theory_citation`` field presence.

When to add a new entry
-----------------------
Three actions, same commit, in this order:

1. Add the per-pair drift check to ``pipeline.check_drift``.
2. Add the sibling-coverage injection-test file at
   ``tests/test_pipeline/<test_slug>.py`` with >= one test function per
   gated constant.
3. Append the ``InlineCopyPair`` to ``CANONICAL_INLINE_COPIES`` below.

Check #159 fails closed if any of those three is missing for a registered
entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InlineCopyPair:
    """One registered canonical-source -> inline-copy pair.

    Attributes
    ----------
    name : str
        Short identifier for the pair. Used in error messages and
        decision-ledger references. Stable -- callers may grep on it.
    inline_site : str
        Repo-relative path to the file holding the inline literal(s).
    canonical_source : str
        Repo-relative path (and optional anchor) of the source of truth.
        Free-form -- may include a line range or YAML key path. Audit-only;
        not parsed by the meta-check.
    gated_constants : tuple[str, ...]
        Tuple of constant names defined at ``inline_site`` that are
        canonical-derived and are asserted by the parity check.
        Sibling-coverage doctrine: ``test_file`` must contain >= one test
        per constant.
    parity_check : str
        Name of the function in ``pipeline.check_drift`` module globals
        that asserts parity. Must be importable and callable.
    test_file : str
        Repo-relative path to the injection-test file. Must exist.
    bug_class_anchor : str
        Repo-relative memory or docs path documenting the bug-class
        instance. Audit-only.
    """

    name: str
    inline_site: str
    canonical_source: str
    gated_constants: tuple[str, ...]
    parity_check: str
    test_file: str
    bug_class_anchor: str
    notes: str = field(default="")


# Registered canonical-inline-copy pairs.
#
# Order is documentation-only -- the meta-check iterates the list, and
# every entry is independently verified. New entries append at the end.
CANONICAL_INLINE_COPIES: list[InlineCopyPair] = [
    InlineCopyPair(
        name="fast_lane_promote_threshold",
        inline_site="scripts/research/fast_lane_promote_queue.py",
        canonical_source=(
            "docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml "
            "(screen: block, lines ~102-145)"
        ),
        gated_constants=(
            "T_KILL_FLOOR",
            "T_PROMOTE_FLOOR",
            "EXPR_FLOOR",
            "N_FLOOR",
            "FIRE_MIN",
            "FIRE_MAX",
        ),
        parity_check="check_fast_lane_promote_threshold_parity",
        test_file=(
            "tests/test_pipeline/"
            "test_check_drift_fast_lane_promote_threshold_parity.py"
        ),
        bug_class_anchor=(
            "memory/feedback_canonical_inline_copy_parity_bug_class.md "
            "(4th confirmed instance, 2026-05-18)"
        ),
        notes=(
            "Six FAST_LANE v5.1 promote-queue scanner thresholds inlined "
            "with prose-comment cite to the canonical YAML template. "
            "Drift check landed in commit d88a5465 (2026-05-19)."
        ),
    ),
    InlineCopyPair(
        name="cherry_pick_ranker_heavyweight_t_threshold",
        inline_site="scripts/research/cherry_pick_ranker.py",
        canonical_source=(
            "docs/institutional/pre_registered_criteria.md "
            "Criterion 4 (no-theory threshold, Chordia 2018 Tier 1 at "
            "literature/chordia_et_al_2018_two_million_strategies.md:20)"
        ),
        gated_constants=("HEAVYWEIGHT_T_THRESHOLD",),
        parity_check="check_cherry_pick_ranker_threshold_parity",
        test_file=(
            "tests/test_pipeline/"
            "test_check_drift_cherry_pick_ranker_threshold_parity.py"
        ),
        bug_class_anchor=(
            "memory/feedback_canonical_inline_copy_parity_bug_class.md "
            "(5th confirmed instance, 2026-05-19)"
        ),
        notes=(
            "Cherry-pick ranker inlines the Chordia strict no-theory "
            "t-threshold (3.79) as HEAVYWEIGHT_T_THRESHOLD. Mirrors "
            "pre_registered_criteria.md Criterion 4 line citing "
            "Chordia et al 2018 verbatim. Parity required so the ranker's "
            "deflation_headroom component does not silently drift if "
            "Criterion 4 is ever amended."
        ),
    ),
    InlineCopyPair(
        name="bridge_methodology_rules_applied",
        inline_site="scripts/research/fast_lane_to_heavyweight_bridge.py",
        canonical_source=(
            ".claude/rules/backtesting-methodology.md (`## RULE N:` headings)"
        ),
        gated_constants=("METHODOLOGY_RULES_APPLIED",),
        parity_check="check_bridge_methodology_rules_parity",
        test_file=(
            "tests/test_pipeline/"
            "test_check_drift_bridge_methodology_rules_parity.py"
        ),
        bug_class_anchor=(
            "memory/feedback_canonical_inline_copy_parity_bug_class.md "
            "(6th confirmed instance, 2026-05-19)"
        ),
        notes=(
            "Bridge embeds methodology-rule slugs as boilerplate in every "
            "generated heavyweight prereg draft. Each slug must map to a "
            "real `## RULE N:` heading in the canonical methodology doc; "
            "otherwise the bridge propagates fake or stale rule citations "
            "into operator-reviewed drafts."
        ),
    ),
    InlineCopyPair(
        name="fast_lane_structural_hash_schema",
        inline_site="scripts/research/fast_lane_structural_hash.py",
        canonical_source=(
            "docs/runtime/stages/"
            "2026-05-20-fast-lane-anti-fp-trial-provenance.md "
            "section `## Hash Schema` (hash_schema_version + 9-field inputs)"
        ),
        gated_constants=(
            "HASH_SCHEMA_VERSION",
            "HASH_SCHEMA_INPUTS",
        ),
        parity_check="check_fast_lane_structural_hash_schema_parity",
        test_file=(
            "tests/test_pipeline/"
            "test_check_drift_fast_lane_structural_hash_schema_parity.py"
        ),
        bug_class_anchor=(
            "memory/feedback_canonical_inline_copy_parity_bug_class.md "
            "(7th confirmed instance, 2026-05-20 — Stage 2A.1)"
        ),
        notes=(
            "Foundational substrate for fast-lane anti-FP trial provenance. "
            "The structural_hash function inlines `HASH_SCHEMA_VERSION = 1` "
            "and the 9-field `HASH_SCHEMA_INPUTS` tuple with a docstring "
            "cite to the canonical `## Hash Schema` block in the Stage 2A "
            "design doc. Schema drift between code and doc would silently "
            "change every hash going forward, breaking de-dup / suppression "
            "rules consumed by 2A.2 (ledger) and 2A.3 (scanner/bridge). "
            "Check #167 parses the doc YAML block and asserts parity."
        ),
    ),
    InlineCopyPair(
        name="fast_lane_trial_ledger_holdout_sentinel",
        inline_site="scripts/research/fast_lane_trial_ledger.py",
        canonical_source=(
            "trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM "
            "(date(2026, 1, 1) — Amendment 2.7 Mode A sacred window)"
        ),
        gated_constants=("HOLDOUT_SACRED_FROM_SENTINEL",),
        parity_check="check_holdout_sentinel_inline_copy_parity",
        test_file=(
            "tests/test_pipeline/"
            "test_check_drift_holdout_sentinel_inline_copy_parity.py"
        ),
        bug_class_anchor=(
            "memory/feedback_canonical_inline_copy_parity_bug_class.md "
            "(8th confirmed instance, 2026-05-20 — Stage 2A.2 follow-up)"
        ),
        notes=(
            "The trial-ledger writer inlines the Mode A holdout boundary as "
            "a string sentinel `HOLDOUT_SACRED_FROM_SENTINEL = \"2026-01-01\"`. "
            "Canonical authority is `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` "
            "(`date(2026, 1, 1)`). Drift would silently keep the ledger stamping "
            "the old boundary on every entry while the rest of the codebase "
            "advanced, corrupting Bailey-Lopez de Prado 2014 sec 3 effective-N "
            "accounting. `HOLDOUT_POLICY_SENTINEL = \"mode_A\"` is NOT registered "
            "because the ledger module is the canonical home of that token — "
            "no upstream constant exists to mirror."
        ),
    ),
]


__all__ = ["InlineCopyPair", "CANONICAL_INLINE_COPIES"]
