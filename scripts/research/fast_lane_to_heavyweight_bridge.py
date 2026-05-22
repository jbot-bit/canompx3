"""FAST_LANE result -> heavyweight Chordia prereg DRAFT generator.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain definition.

Translates a fast-lane v5.1 result MD + its source prereg YAML into a
heavyweight Chordia strict-unlock prereg DRAFT under
``docs/audit/hypotheses/drafts/`` (NOT under ``docs/audit/hypotheses/``).
Drafts skip the LHP validator's theory-citation requirement -- the loader
only walks active preregs, not ``drafts/`` (per
``memory/feedback_lhp_validator_vs_field_presence_trap_n1.md``). Operator
moves the draft into the active dir manually after either:

  (a) accepting the no-theory strict t=3.79 hurdle (recommended default),
      OR
  (b) authoring a literature_citation that grounds an explicit theory_grant
      upgrade (lowers hurdle to t>=3.00 per Criterion 4).

The bridge fills every machine-determinable field. It NEVER writes
``theory_citation`` -- per
``memory/feedback_chordia_theory_citation_field_presence_trap.md`` an empty
string would trip the loader's field-presence check and silently relax the
threshold. ``theory_grant: false`` is always emitted; operator deliberately
adds ``theory_citation`` only when upgrading.

Doctrine grounding
------------------
- Plan: ``C:/Users/joshd/.claude/plans/or-linknin-them-togehr-delegated-gizmo.md``
- Stage file: ``docs/runtime/stages/2026-05-19-fast-lane-to-heavyweight-bridge.md``
- Field-presence trap: ``memory/feedback_chordia_theory_citation_field_presence_trap.md``
- Quarantine pattern: ``memory/feedback_lhp_validator_vs_field_presence_trap_n1.md``
- K-budget MinBTL: ``.claude/rules/hypothesis-prereg-discipline.md``
- OOS power doctrine: ``.claude/rules/backtesting-methodology.md`` RULE 3.3

This script does NOT write to ``chordia_audit_log.yaml``,
``validated_setups``, ``allocation file``, or any file under
``trading_app/live/``. The output is a draft YAML on disk plus an operator
checklist printed to stdout.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESES_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses"
DRAFTS_DIR = HYPOTHESES_DIR / "drafts"
PROMOTE_QUEUE_CACHE = REPO_ROOT / "docs" / "runtime" / "promote_queue.yaml"
GRAVEYARD_DIGEST_PATH = REPO_ROOT / "docs" / "runtime" / "fast_lane_graveyard_digest.yaml"


class BridgeRefused(Exception):
    """Raised when bridge pre-flight refuses to author a heavyweight draft.

    Stage 2A.3+V2 introduces non-OOS refusal triggers:
      - banned entry_model (E0 / E3)
      - E2 + canonical lookahead-class filter
      - structural_hash matches a lane-level graveyard digest entry
      - promote_queue status is not QUEUED
      - missing/malformed V2 provenance (structural_hash, k_lineage, n_hat)
      - K_lane > K_declared_in_prereg

    The OOS-power gate (RULE 3.3) stays where it is in the scanner; this
    bridge refuses anything other than QUEUED (the scanner emits
    REJECTED_OOS_UNPOWERED upstream), so OOS power is not re-checked here.
    Refusal raises BridgeRefused with an operator-readable reason; main()
    catches it and exits 3 with the reason on stderr. No draft is written.
    """


# Strict Chordia threshold prose. The numeric value lives in
# ``cherry_pick_ranker.HEAVYWEIGHT_T_THRESHOLD`` (3.79) under Check #160
# parity; here we emit prose only -- the runner resolves the numeric value
# from doctrine at execution time.
HEAVYWEIGHT_T_THRESHOLD_PROSE: str = "Criterion 4 no-theory strict threshold (t >= 3.79, Chordia 2018 verbatim Tier 1)"

# Required scope fields -- absence raises ValueError. Every field is consumed
# downstream by ``primary_schema.family_cells`` or by the heavyweight Chordia
# runner's strategy resolution. A typo in the source YAML ("filter_ttype" for
# "filter_type") must FAIL LOUDLY, not silently drop and vanish into the
# draft -- evidence-auditor 2026-05-19 finding.
_REQUIRED_SCOPE_FIELDS: tuple[str, ...] = (
    "instrument",
    "strategy_id",
    "session",
    "orb_minutes",
    "entry_model",
    "confirm_bars",
    "rr_target",
    "direction",
    "filter_type",
)

# Optional scope fields -- propagated when present, silently absent when not.
_OPTIONAL_SCOPE_FIELDS: tuple[str, ...] = (
    "filter_source",
    "out_of_scope",
)

# Full allowlist (required + optional). Unknown fields are silently dropped to
# defend against the field-presence trap (theory_citation et al.). Required
# fields raise on absence; unknown fields raise nothing -- the asymmetry is
# deliberate and is the entire point of the fail-closed split.
_ALLOWED_SCOPE_FIELDS: tuple[str, ...] = _REQUIRED_SCOPE_FIELDS + _OPTIONAL_SCOPE_FIELDS

# Methodology rules boilerplate. Each entry MUST correspond to a real RULE
# block in ``.claude/rules/backtesting-methodology.md``. Parity enforced by
# Check #161 (``check_bridge_methodology_rules_parity``): the canonical rule
# file is parsed at check time, and the set below must be a subset of the
# parsed RULE-numbers (the bridge picks a subset; not every methodology rule
# is applicable to every replay).
#
# Order is documentation-only -- the check uses set membership.
METHODOLOGY_RULES_APPLIED: tuple[str, ...] = (
    "rule_1_temporal_alignment",
    "rule_3_is_oos_discipline",
    "rule_4_multi_framing",
    "rule_9_canonical_layers",
    "rule_10_pre_registration",
)


@dataclass(frozen=True)
class FastLaneSource:
    """Parsed fast-lane source: result MD path + source YAML payload."""

    result_md_rel: str
    source_yaml_rel: str
    scope: dict[str, Any]


def _rel_to_repo(path: Path) -> str:
    """Return repo-relative path string with forward slashes."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def locate_source_yaml(result_md: Path, *, hypotheses_dir: Path | None = None) -> Path | None:
    """Find the fast-lane source YAML matching a result MD by filename stem."""
    hypotheses = hypotheses_dir if hypotheses_dir is not None else HYPOTHESES_DIR
    stem = result_md.stem
    candidate = hypotheses / f"{stem}.yaml"
    return candidate if candidate.exists() else None


def load_fast_lane_source(result_md: Path, *, hypotheses_dir: Path | None = None) -> FastLaneSource | None:
    """Load and validate a fast-lane result + source pair.

    Returns None when either the result MD or matching YAML is missing,
    or when the YAML lacks the required ``scope.strategy_id`` field.
    """
    if not result_md.exists():
        return None
    source_yaml = locate_source_yaml(result_md, hypotheses_dir=hypotheses_dir)
    if source_yaml is None:
        return None
    try:
        data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    scope = data.get("scope")
    if not isinstance(scope, dict):
        return None
    if not scope.get("strategy_id"):
        return None
    return FastLaneSource(
        result_md_rel=_rel_to_repo(result_md),
        source_yaml_rel=_rel_to_repo(source_yaml),
        scope=scope,
    )


def _draft_slug(strategy_id: str) -> str:
    """Map a strategy_id to a kebab-case slug for the draft filename."""
    return strategy_id.lower().replace("_", "-").replace(".", "-")


def _lookup_promote_queue_entry(strategy_id: str, *, queue_path: Path | None = None) -> dict[str, Any] | None:
    """Return the promote_queue.yaml entry for ``strategy_id``, or None.

    The bridge needs ``structural_hash`` + ``k_lineage`` to enforce the
    Stage 2A.3 suppression rules at draft-authoring time. The scanner is
    the single source of truth for those values; the bridge never
    re-computes them.
    """
    target = queue_path if queue_path is not None else PROMOTE_QUEUE_CACHE
    if not target.exists():
        return None
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("strategy_id") == strategy_id:
            return entry
    return None


def _graveyard_has_lane_match(structural_hash: str, *, digest_path: Path | None = None) -> dict[str, Any] | None:
    """Return the FIRST lane-level graveyard entry whose structural_hash
    matches, or None.

    Class-level entries (``hash_kind == "class"``) DO NOT refuse the
    bridge -- they are informational matches the scanner already cited.
    """
    target = digest_path if digest_path is not None else GRAVEYARD_DIGEST_PATH
    if not target.exists():
        return None
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    for entry in data.get("entries") or []:
        if (
            isinstance(entry, dict)
            and entry.get("structural_hash") == structural_hash
            and entry.get("hash_kind") == "lane"
        ):
            return entry
    return None


def _bridge_preflight_refuse(
    source: FastLaneSource,
    *,
    queue_path: Path | None = None,
    digest_path: Path | None = None,
) -> None:
    """Apply Stage 2A.3+V2 refusal chain. Raises BridgeRefused on
    any hit; returns None when the source survives every gate.

    Triggers (in order):
      1. ``entry_model in {E0, E3}`` -- banned, fail-closed.
      2. ``entry_model == E2`` AND filter matches
         ``E2_EXCLUDED_FILTER_PREFIXES`` / ``E2_EXCLUDED_FILTER_SUBSTRINGS``
         (canonical from ``trading_app.config`` -- NEVER re-encode here).
      3. ``structural_hash`` matches a lane-level graveyard entry.
      4. promote_queue status is not ``QUEUED``.
      5. V2 provenance is missing or malformed.
      6. ``K_lane > K_declared_in_prereg``.
    """
    scope = source.scope
    entry_model_raw = scope.get("entry_model")
    entry_model = entry_model_raw.upper().strip() if isinstance(entry_model_raw, str) else ""
    filter_type = scope.get("filter_type", "")
    if not isinstance(filter_type, str):
        filter_type = ""

    # Trigger 1: banned entry_model.
    if entry_model in {"E0", "E3"}:
        raise BridgeRefused(
            f"entry_model={entry_model!r} is graveyard-banned "
            "(E0 purged; E3 soft-retired Feb 2026). See trading_app/config.py "
            "ENTRY_MODELS + memory/feedback_e2_lookahead_drift_check_landed.md. "
            f"Source YAML: {source.source_yaml_rel}."
        )

    # Trigger 2: E2 + lookahead-class filter (canonical delegation only).
    if entry_model == "E2":
        # Canonical source per institutional-rigor.md § 10 -- import here,
        # never re-encode the prefixes/substrings tuples in this file.
        from trading_app.config import (
            E2_EXCLUDED_FILTER_PREFIXES,
            E2_EXCLUDED_FILTER_SUBSTRINGS,
        )

        if any(filter_type.startswith(p) for p in E2_EXCLUDED_FILTER_PREFIXES):
            raise BridgeRefused(
                f"entry_model=E2 with filter_type={filter_type!r} matches "
                f"trading_app.config.E2_EXCLUDED_FILTER_PREFIXES "
                f"{E2_EXCLUDED_FILTER_PREFIXES!r}; this is the break-bar "
                "lookahead class (E2 enters intra-bar; bar-close-derived "
                "filter features are post-entry data). "
                f"Source YAML: {source.source_yaml_rel}."
            )
        if any(s in filter_type for s in E2_EXCLUDED_FILTER_SUBSTRINGS):
            raise BridgeRefused(
                f"entry_model=E2 with filter_type={filter_type!r} contains "
                f"trading_app.config.E2_EXCLUDED_FILTER_SUBSTRINGS "
                f"{E2_EXCLUDED_FILTER_SUBSTRINGS!r}; this is the break-bar "
                "lookahead class. "
                f"Source YAML: {source.source_yaml_rel}."
            )

    # Triggers 3 + 4 both need the scanner cache entry.
    cache_entry = _lookup_promote_queue_entry(scope["strategy_id"], queue_path=queue_path)
    if cache_entry is None:
        # No cache entry yet -- the scanner has not seen this result MD.
        # Pre-flight cannot enforce the structural-hash gates here. Per
        # Stage 2A.3 design, the bridge expects the scanner to have run
        # first (it always does in the operational flow). Fail-closed
        # rather than guess.
        raise BridgeRefused(
            f"no promote_queue.yaml cache entry for strategy_id "
            f"{scope['strategy_id']!r}; rerun the scanner "
            "(scripts/research/fast_lane_promote_queue.py --write) before "
            "invoking the bridge so provenance gates can be enforced."
        )

    status = cache_entry.get("status")
    if status != "QUEUED":
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} has "
            f"status={status!r}; expected QUEUED. Bridge only authors drafts "
            "from currently queued Fast Lane survivors. Resolve the upstream "
            "blocker instead of bypassing promote_queue status."
        )

    structural_hash = cache_entry.get("structural_hash")
    if not isinstance(structural_hash, str) or len(structural_hash) != 16:
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} has "
            f"malformed structural_hash={structural_hash!r}; rerun the "
            "scanner with --write to refresh provenance fields."
        )
    try:
        int(structural_hash, 16)
    except ValueError as exc:
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} has "
            f"non-hex structural_hash={structural_hash!r}; rerun the scanner "
            "with --write to refresh provenance fields."
        ) from exc

    # Trigger 3: graveyard lane match.
    grave = _graveyard_has_lane_match(structural_hash, digest_path=digest_path)
    if grave is not None:
        raise BridgeRefused(
            f"structural_hash={structural_hash!r} matches a lane-level "
            "graveyard entry: "
            f"title={grave.get('title')!r}, status={grave.get('status')!r}, "
            f"source={grave.get('source_path')!r}. Bridge refuses to "
            "author a draft for a graveyard lane."
        )

    k_lineage = cache_entry.get("k_lineage") or {}
    if not isinstance(k_lineage, dict) or not k_lineage:
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} is "
            "missing k_lineage; rerun the scanner with --write so V2 trial "
            "budget provenance is present before bridge authoring."
        )
    try:
        k_lane = int(k_lineage["K_lane"])
        k_declared = int(k_lineage["K_declared_in_prereg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} has "
            f"malformed k_lineage={k_lineage!r}; expected K_lane and "
            "K_declared_in_prereg integers."
        ) from exc
    if k_lane > k_declared:
        raise BridgeRefused(
            f"K_lane={k_lane} > K_declared_in_prereg={k_declared} for "
            f"structural_hash={structural_hash!r}; observed trial count "
            "exceeds the preregistered V2 lane-level trial budget. PARK or "
            "pool with siblings rather than authoring another heavyweight draft."
        )
    if cache_entry.get("n_hat") is None:
        raise BridgeRefused(
            f"promote_queue.yaml entry for {scope['strategy_id']!r} is "
            "missing n_hat; rerun the scanner with --write so correlation "
            "haircut provenance is present before bridge authoring."
        )


def build_heavyweight_prereg(source: FastLaneSource, *, today: str) -> dict[str, Any]:
    """Build the heavyweight Chordia prereg dict from a fast-lane source.

    Fields NEVER written:
      - ``metadata.theory_citation`` -- field-presence trap; operator adds
        explicitly only when upgrading to theory_grant.
      - ``hypotheses[].theory_citation`` -- same trap.

    Fields ALWAYS written:
      - ``metadata.theory_grant: false`` -- the bridge cannot author
        literature citations; default to strict no-theory.
      - ``metadata.is_triage_screen: false`` -- this is heavyweight.
      - ``primary_schema.chordia_threshold_basis`` -- prose cite to
        Criterion 4 strict threshold.
      - ``methodology_rules_applied`` -- mapping keyed by METHODOLOGY_RULES_APPLIED.
      - ``upstream_discovery_provenance`` -- points at the fast-lane source.
      - ``execution_gate.allowed_now: false`` -- operator must explicitly
        flip after literature/power review.

    Pre-flight refusal:
      Stage 2A.3 calls ``_bridge_preflight_refuse`` BEFORE missing-required-
      field validation. Refusal raises ``BridgeRefused``; ``main()``
      catches and exits 3 with the reason on stderr. No draft is written.
    """
    # Stage 2A.3+V2 pre-flight: refuse banned entry, E2 lookahead,
    # graveyard lane, non-QUEUED status, malformed provenance, and K
    # overrun before authoring any heavyweight draft.
    _bridge_preflight_refuse(source)

    scope = source.scope
    missing_required = [f for f in _REQUIRED_SCOPE_FIELDS if scope.get(f) in (None, "")]
    if missing_required:
        raise ValueError(
            "fast-lane source scope is missing required field(s): "
            f"{missing_required}. Source YAML: {source.source_yaml_rel}. "
            "Every field is consumed downstream; a typo or omission must "
            "fail loudly rather than vanish silently into the draft "
            "(evidence-auditor 2026-05-19 finding; integrity-guardian "
            "section 3 fail-closed)."
        )
    strategy_id = scope["strategy_id"]

    methodology_application_defaults: dict[str, str] = {
        "rule_1_temporal_alignment": (
            "Inherited from fast-lane source; operator reviews look-ahead "
            "class against backtesting-methodology.md section 1 for the "
            "heavyweight replay before flipping execution_gate.allowed_now."
        ),
        "rule_3_is_oos_discipline": (
            "HOLDOUT_SACRED_FROM=2026-01-01 enforced by canonical loader. "
            "OOS-power floor must be respected per RULE 3.3 "
            "(see research/oos_power.py)."
        ),
        "rule_4_multi_framing": (
            "Pathway B K=1 heavyweight replay; K=1 own framing. Upstream "
            "fast-lane PROMOTE is provenance-only, not verdict evidence."
        ),
        "rule_9_canonical_layers": (
            "Reads only orb_outcomes JOIN daily_features per chordia_strict_unlock_v1.py canonical layer discipline."
        ),
        "rule_10_pre_registration": (
            "Operator must commit this prereg to docs/audit/hypotheses/ "
            "BEFORE invoking the heavyweight runner. Drafts are not "
            "preregistered until moved out of drafts/."
        ),
    }
    methodology_block: dict[str, dict[str, str]] = {
        rule: {"application": methodology_application_defaults[rule]} for rule in METHODOLOGY_RULES_APPLIED
    }

    prereg: dict[str, Any] = {
        "metadata": {
            "theory_grant": False,
            "name": _draft_slug(strategy_id).replace("-", "_") + "_chordia_unlock_v1",
            "purpose": (
                "Heavyweight Chordia strict unlock authored from fast-lane "
                f"v5.1 PROMOTE provenance ({source.result_md_rel}). "
                "Bridge-generated DRAFT: operator must review literature "
                "grounding, OOS power, and era-stability before moving "
                "out of drafts/."
            ),
            "date_locked": f"{today}T00:00:00+10:00",
            "holdout_date": "2026-01-01",
            "total_expected_trials": 1,
            "testing_mode": "individual",
            "research_question_type": "conditional_role",
            "template_version": "chordia_strict_v1",
            "is_triage_screen": False,
            "validation_status_explicit": (
                "DRAFT -- not yet preregistered. theory_grant=false applies "
                "strict no-theory Chordia hurdle (t>=3.79). Move out of "
                "drafts/ only after literature/power/era review."
            ),
        },
        "execution": {
            "mode": "bounded_runner",
            "entrypoint": "research/chordia_strict_unlock_v1.py",
        },
        "authority": {
            "primary": [
                "RESEARCH_RULES.md",
                ".claude/rules/backtesting-methodology.md",
                ".claude/rules/research-truth-protocol.md",
                "docs/institutional/pre_registered_criteria.md",
                "docs/runtime/chordia_audit_log.yaml",
            ],
            "notes": [
                "Authored by scripts/research/fast_lane_to_heavyweight_bridge.py.",
                f"Provenance: fast-lane v5.1 PROMOTE result at {source.result_md_rel}.",
                f"Threshold basis: {HEAVYWEIGHT_T_THRESHOLD_PROSE}.",
                "Mode A holdout remains sacred from 2026-01-01.",
            ],
        },
        "scope": {k: scope[k] for k in _ALLOWED_SCOPE_FIELDS if k in scope},
        "data_policy": {
            "is_window": {
                "description": "trading_day < HOLDOUT_SACRED_FROM",
                "constant_source": "trading_app.holdout_policy.HOLDOUT_SACRED_FROM",
                "locked_boundary": "2026-01-01",
            },
            "oos_window": {
                "description": "trading_day >= HOLDOUT_SACRED_FROM",
                "policy": (
                    "read-only descriptive; OOS power floor per RULE 3.3 "
                    "must be respected before any binary OOS gate fires"
                ),
            },
            "tuning_against_oos": False,
            "canonical_layers_only": True,
            "scratch_policy": "realized-eod",
            "scratch_handling": ("COALESCE(pnl_r, 0.0) -- never WHERE pnl_r IS NOT NULL"),
        },
        "grounding": {
            "filter_grounding_status": {
                "verdict": "UNSUPPORTED",
                "basis": (
                    "Bridge cannot author literature citations. "
                    "theory_grant=false applies strict no-theory threshold. "
                    "Operator may upgrade by adding theory_citation field "
                    "and flipping theory_grant=true (lowers t hurdle "
                    "3.79->3.00 per Criterion 4)."
                ),
            },
        },
        "upstream_discovery_provenance": {
            "role": "PROVENANCE_ONLY",
            "sources": [source.result_md_rel, source.source_yaml_rel],
            "note": (
                "Fast-lane PROMOTE earns the right to heavyweight authoring "
                "but is NOT verdict evidence. Fresh canonical replay against "
                f"strict Chordia threshold ({HEAVYWEIGHT_T_THRESHOLD_PROSE}) "
                "is the only valid verdict source."
            ),
        },
        "primary_schema": {
            "family_cells": [
                {
                    "id": strategy_id,
                    "strategy_id": strategy_id,
                    "instrument": scope.get("instrument"),
                    "session": scope.get("session"),
                    "orb_minutes": scope.get("orb_minutes"),
                    "rr_target": scope.get("rr_target"),
                    "filter": scope.get("filter_type"),
                }
            ],
            "k_family": 1,
            "k_global": 1,
            "k_lane": 1,
            "k_session": 1,
            "chordia_threshold_basis": HEAVYWEIGHT_T_THRESHOLD_PROSE,
            "promotion_gate": ("PASS_CHORDIA only; PASS_PROTOCOL_A not available without theory grant upgrade"),
        },
        "trial_budget": {
            "primary_selection_trials": 1,
            "schema_locked_before_any_metric": True,
            "minbtl_bound": ("MinBTL = 2*ln(1)/1 = 0.0 years; N=1 within Bailey 2013 cap"),
        },
        "total_hypothesis_count": 1,
        "total_expected_trials": 1,
        "budget_check": {
            "max_allowed_clean": 300,
            "max_allowed_proxy": 2000,
            "status": "under budget",
        },
        "methodology_rules_applied": methodology_block,
        "execution_gate": {
            "allowed_now": False,
            "execution_surface": "Windows PowerShell with .venv",
            "operator_must_review_before_flip": [
                "literature grounding (or accept no-theory strict 3.79 hurdle)",
                "OOS power readiness via research/oos_power.py",
                "era-stability replay",
                "K-budget MinBTL re-verification (scripts/tools/estimate_k_budget.py)",
            ],
            "forbidden_now": [
                "any deployment action",
                "writes to chordia_audit_log.yaml until result MD exists",
            ],
        },
    }
    return prereg


def draft_path_for(strategy_id: str, today: str) -> Path:
    """Compose: drafts/<today>-<slug>-chordia-heavyweight-v1.draft.yaml."""
    slug = _draft_slug(strategy_id)
    return DRAFTS_DIR / f"{today}-{slug}-chordia-heavyweight-v1.draft.yaml"


def write_draft(prereg: dict[str, Any], out_path: Path) -> None:
    """Write the draft YAML. Creates drafts/ if needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True, width=80),
        encoding="utf-8",
    )


def format_operator_checklist(draft_path: Path, source: FastLaneSource) -> str:
    """Operator next-step checklist printed after a draft is written."""
    rel = _rel_to_repo(draft_path)
    return (
        f"\nDraft heavyweight prereg written to: {rel}\n"
        f"Source: {source.result_md_rel}\n\n"
        "Operator next-step checklist (per "
        "memory/feedback_chordia_unlock_deployment_gate_audit_checklist.md):\n"
        "  1. Add literature_citation if upgrading to theory_grant=true "
        "(lowers hurdle 3.79->3.00).\n"
        "  2. Verify OOS power tier via research/oos_power.py "
        "(N>=30 + power>=0.80 for binary OOS gate eligibility).\n"
        "  3. Verify era-stability via heavyweight replay run separately.\n"
        "  4. Re-run K-budget gate (scripts/tools/estimate_k_budget.py) "
        "before moving out of drafts/.\n"
        "  5. Move drafts/<file>.draft.yaml -> hypotheses/<file>.yaml, "
        "flip execution_gate.allowed_now=true, then invoke "
        "research/chordia_strict_unlock_v1.py.\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fast_lane_to_heavyweight_bridge",
        description=(
            "Generate a heavyweight Chordia strict-unlock prereg DRAFT from a fast-lane v5.1 PROMOTE result MD."
        ),
    )
    p.add_argument(
        "result_md",
        type=Path,
        help="Path to the fast-lane result MD (docs/audit/results/*.md).",
    )
    p.add_argument(
        "--today",
        type=str,
        default=None,
        help="ISO date stamp for the draft filename + metadata (default: today).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the draft YAML to stdout without writing to disk.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    today = args.today or date.today().isoformat()
    source = load_fast_lane_source(args.result_md)
    if source is None:
        print(
            f"ERROR: could not load fast-lane source from {args.result_md}. "
            "Confirm the result MD exists and a matching source YAML lives "
            "in docs/audit/hypotheses/.",
            file=sys.stderr,
        )
        return 1
    try:
        prereg = build_heavyweight_prereg(source, today=today)
    except BridgeRefused as exc:
        print(
            f"BRIDGE REFUSED: {exc}",
            file=sys.stderr,
        )
        return 3
    if args.dry_run:
        print(yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True, width=80))
        return 0
    out_path = draft_path_for(source.scope["strategy_id"], today)
    write_draft(prereg, out_path)
    print(format_operator_checklist(out_path, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
