"""Stage A — Idea Ingestion Front-Door for the Fast-Lane chain.

Turns a structured CLI invocation + interactive mechanism/literature prompts
into a fast-lane v5.1-compliant pre-reg YAML under
``docs/audit/hypotheses/drafts/``. The output is a NEW input to the
fast-lane chain (not a chain output) — operator promotes it to
``docs/audit/hypotheses/`` after review.

Refusal logic delegates to canonical sources only — never re-encodes:

* Instrument:   ``pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS``
* Session:      ``pipeline.dst.SESSION_CATALOG``
* Entry model:  ``trading_app.config.ENTRY_MODELS`` (further restricted to {E1, E2} for fast-lane v5.1)
* Filter:       ``trading_app.config.ALL_FILTERS``
* E2 look-ahead: ``trading_app.config.E2_EXCLUDED_FILTER_PREFIXES`` / ``E2_EXCLUDED_FILTER_SUBSTRINGS``
* Literature:   ``scripts.research.lhp.literature_index.load_corpus`` + ``citation_exists``

Per evidence-auditor review:
* No ``fcntl`` locking (Unix-only; ledger writer has no lock either)
* No numeric ``chordia_gate_threshold`` field (emit prose only — runner resolves at execution)
* No ``theory_citation_pending_audit_log_approval`` (Amendment 3.3 closed the trap via ``theory_grant: false``)

Exit codes:
* 0 — draft written
* 2 — argparse usage error
* 3 — refusal (gate failed); reason on stderr, no file written
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESES_DIR = REPO_ROOT / "docs" / "audit" / "hypotheses"
DRAFTS_DIR = HYPOTHESES_DIR / "drafts"
LITERATURE_DIR = REPO_ROOT / "docs" / "institutional" / "literature"

# Fast-lane v5.1 restricts to E1/E2 (E0 purged Feb 2026; E3 soft-retired).
# This is NOT a re-encoding of canonical state — it is the fast-lane v5.1
# eligibility set, declared by the lane itself. The canonical ENTRY_MODELS
# dict still contains E3 because E3 is permitted in other lanes (e.g.,
# heavyweight Chordia retests on grandfathered E3 strategies).
_FASTLANE_ENTRY_MODELS: frozenset[str] = frozenset({"E1", "E2"})
_VALID_DIRECTIONS: frozenset[str] = frozenset({"pooled", "long", "short"})
_VALID_ORB_MINUTES: frozenset[int] = frozenset({5, 15, 30})

# Brisbane is UTC+10, no DST. Used for the date_locked stamp so the YAML
# matches the in-the-wild fast-lane v5.1 timestamps.
_BRISBANE_TZ = timezone(timedelta(hours=10))


class IngestRefused(Exception):
    """Raised when a gate refuses to emit. Caught by ``main()`` -> exit 3."""


def _check_instrument(instrument: str) -> None:
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

    if instrument not in ACTIVE_ORB_INSTRUMENTS:
        raise IngestRefused(
            f"instrument={instrument!r} not in "
            f"pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS "
            f"({sorted(ACTIVE_ORB_INSTRUMENTS)})."
        )


def _check_session(session: str) -> None:
    from pipeline.dst import SESSION_CATALOG

    if session not in SESSION_CATALOG:
        raise IngestRefused(
            f"session={session!r} not in pipeline.dst.SESSION_CATALOG ({sorted(SESSION_CATALOG.keys())})."
        )


def _check_entry_model(entry_model: str) -> None:
    from trading_app.config import ENTRY_MODELS

    if entry_model not in ENTRY_MODELS:
        raise IngestRefused(
            f"entry_model={entry_model!r} not in trading_app.config.ENTRY_MODELS ({sorted(ENTRY_MODELS)})."
        )
    if entry_model not in _FASTLANE_ENTRY_MODELS:
        raise IngestRefused(
            f"entry_model={entry_model!r} is canonical but NOT eligible "
            f"for fast-lane v5.1. Allowed: {sorted(_FASTLANE_ENTRY_MODELS)} "
            "(E0 purged Feb 2026; E3 soft-retired)."
        )


def _check_filter_and_e2_lookahead(entry_model: str, filter_type: str) -> None:
    from trading_app.config import (
        ALL_FILTERS,
        E2_EXCLUDED_FILTER_PREFIXES,
        E2_EXCLUDED_FILTER_SUBSTRINGS,
    )

    if filter_type not in ALL_FILTERS:
        raise IngestRefused(
            f"filter_type={filter_type!r} not in "
            f"trading_app.config.ALL_FILTERS ({len(ALL_FILTERS)} filters). "
            "Pick a registered key."
        )

    if entry_model == "E2":
        if any(filter_type.startswith(p) for p in E2_EXCLUDED_FILTER_PREFIXES):
            raise IngestRefused(
                f"entry_model=E2 with filter_type={filter_type!r} matches "
                f"trading_app.config.E2_EXCLUDED_FILTER_PREFIXES "
                f"{E2_EXCLUDED_FILTER_PREFIXES!r}; this is the break-bar "
                "look-ahead class. See "
                "backtesting-methodology.md § 6.3."
            )
        if any(s in filter_type for s in E2_EXCLUDED_FILTER_SUBSTRINGS):
            raise IngestRefused(
                f"entry_model=E2 with filter_type={filter_type!r} contains "
                f"trading_app.config.E2_EXCLUDED_FILTER_SUBSTRINGS "
                f"{E2_EXCLUDED_FILTER_SUBSTRINGS!r}; break-bar look-ahead."
            )


def _check_direction(direction: str) -> None:
    if direction not in _VALID_DIRECTIONS:
        raise IngestRefused(f"direction={direction!r} not in {sorted(_VALID_DIRECTIONS)}.")


def _check_orb_minutes(orb_minutes: int) -> None:
    if orb_minutes not in _VALID_ORB_MINUTES:
        raise IngestRefused(f"orb_minutes={orb_minutes!r} not in {sorted(_VALID_ORB_MINUTES)}.")


def _check_mechanism(mechanism: str) -> None:
    if not mechanism or not mechanism.strip():
        raise IngestRefused(
            "mechanism is empty. Every fast-lane pre-reg must declare an "
            "economic mechanism (one line is fine). See "
            "memory/feedback_literature_before_prereg.md."
        )


def _check_literature(literature_slug: str) -> None:
    from scripts.research.lhp.literature_index import (
        citation_exists,
        load_corpus,
    )

    corpus = load_corpus(LITERATURE_DIR)
    if not citation_exists(corpus, literature_slug):
        slugs = sorted({e.slug for e in corpus})
        raise IngestRefused(
            f"literature={literature_slug!r} does not resolve to any file "
            f"in {LITERATURE_DIR.relative_to(REPO_ROOT)} "
            f"(found {len(slugs)} slugs). Pick one of these or extract a "
            f"new literature file first. Sample: {slugs[:5]}"
        )


def build_strategy_id(
    *,
    instrument: str,
    session: str,
    entry_model: str,
    rr: float,
    confirm_bars: int,
    filter_type: str,
    orb_minutes: int,
) -> str:
    """Build the canonical strategy_id used by the fast-lane chain.

    Format: ``{INSTRUMENT}_{SESSION}_{ENTRY}_RR{rr}_CB{cb}_{FILTER}[_O{apt}]``.

    The aperture suffix ``_O{apt}`` is OMITTED when ``orb_minutes == 5`` to
    match the canonical parser (``trading_app.eligibility.builder.parse_strategy_id``
    which defaults orb_minutes=5 when no suffix is present). For 15/30 the
    suffix is required.
    """
    base = f"{instrument}_{session}_{entry_model}_RR{rr:.1f}_CB{confirm_bars}_{filter_type}"
    if orb_minutes == 5:
        return base
    return f"{base}_O{orb_minutes}"


def build_slug(
    *,
    instrument: str,
    session: str,
    entry_model: str,
    rr: float,
    confirm_bars: int,
    filter_type: str,
    orb_minutes: int,
    direction: str,
) -> str:
    """Build the filesystem slug for the draft YAML.

    Mirrors the naming pattern of in-the-wild 2026-05-18 fast-lane pre-regs:
    e.g. ``mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30``.
    """
    rr_compact = f"rr{int(round(rr * 10))}"
    parts = [
        instrument.lower(),
        session.lower().replace("_", ""),
        entry_model.lower(),
        rr_compact,
        f"cb{confirm_bars}",
        filter_type.lower().replace("_", ""),
        direction,
        f"o{orb_minutes}",
    ]
    return "-".join(parts)


def build_prereg(
    *,
    instrument: str,
    session: str,
    orb_minutes: int,
    entry_model: str,
    confirm_bars: int,
    rr: float,
    direction: str,
    filter_type: str,
    mechanism: str,
    literature_slug: str,
    purpose: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the fast-lane v5.1 pre-reg dict.

    Mirrors the structure of the 2026-05-18 in-the-wild fast-lane pre-regs:
    ``docs/audit/hypotheses/2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.yaml``
    is the schema baseline (tested via ``test_ingest_idea.py::test_schema_parity``).
    """
    if now is None:
        now = datetime.now(_BRISBANE_TZ)
    strategy_id = build_strategy_id(
        instrument=instrument,
        session=session,
        entry_model=entry_model,
        rr=rr,
        confirm_bars=confirm_bars,
        filter_type=filter_type,
        orb_minutes=orb_minutes,
    )
    slug = build_slug(
        instrument=instrument,
        session=session,
        entry_model=entry_model,
        rr=rr,
        confirm_bars=confirm_bars,
        filter_type=filter_type,
        orb_minutes=orb_minutes,
        direction=direction,
    )
    name = slug.replace("-", "_") + "_fast_lane_v1"

    if purpose is None:
        purpose = (
            f"FAST_LANE v5.1 triage screen on {instrument} {session} "
            f"{entry_model} O{orb_minutes} {filter_type} ({direction}). "
            f"Ingested via scripts/research/ingest_idea.py."
        )

    prereg: dict = {
        "metadata": {
            "theory_grant": False,
            "name": name,
            "purpose": purpose,
            "date_locked": now.replace(microsecond=0).isoformat(),
            "holdout_date": "2026-01-01",
            "total_expected_trials": 1,
            "testing_mode": "individual",
            "research_question_type": "conditional_role",
            "template_version": "fast_lane_v5.1",
            "template_path": "docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml",
            "is_triage_screen": True,
            "promotion_target": "heavyweight_chordia_prereg",
            "validation_status_explicit": (
                "NOT_VALIDATED — fast-lane v5.1 triage screen, "
                "not calibrated. PROMOTE means 'worth heavyweight Chordia "
                "review', never deploy. Never capital."
            ),
            "ingested_by": "scripts/research/ingest_idea.py",
            "ingested_at": now.replace(microsecond=0).isoformat(),
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
                "docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml",
                f"docs/institutional/literature/{literature_slug}.md",
            ],
            "notes": [
                "v5.1 triage screen — PROMOTE if t >= 3.0; NEEDS-MORE 2.5-3.0; KILL < 2.5.",
                "PROMOTE authorizes heavyweight Chordia review only, NOT capital.",
            ],
        },
        "scope": {
            "instrument": instrument,
            "strategy_id": strategy_id,
            "session": session,
            "orb_minutes": orb_minutes,
            "entry_model": entry_model,
            "confirm_bars": confirm_bars,
            "rr_target": rr,
            "direction": direction,
            "filter_type": filter_type,
            "filter_source": f"trading_app.config.ALL_FILTERS['{filter_type}']",
            "out_of_scope": [
                f"{filter_type} siblings at other RR/sessions/apertures/entry models/confirm_bars",
                "theory-grant upgrade — operator authors literature claim separately",
                "allocator, sizing, schema, or live-execution changes",
                "any write to validated_setups, chordia_audit_log.yaml, or allocation file",
            ],
        },
        "data_policy": {
            "is_window": {
                "description": "trading_day < HOLDOUT_SACRED_FROM",
                "constant_source": "trading_app.holdout_policy.HOLDOUT_SACRED_FROM",
                "locked_boundary": "2026-01-01",
            },
            "oos_window": {
                "description": "trading_day >= HOLDOUT_SACRED_FROM",
                "policy": ("read-only descriptive; v5.1 holdout-boundary proof asserts max_IS < boundary <= min_OOS"),
            },
            "tuning_against_oos": False,
            "canonical_layers_only": True,
            "scratch_policy": "realized-eod",
            "scratch_handling": "COALESCE(pnl_r, 0.0) — never WHERE pnl_r IS NOT NULL",
            "lookahead_banned_always": [
                "mae_r",
                "mfe_r",
                "outcome",
                "pnl_r",
                "double_break",
            ],
        },
        "grounding": {
            "core_orb_premise": [
                {
                    "source_file": "resources/Building_Reliable_Trading_Systems.pdf",
                    "extract_proxy": "docs/institutional/literature/fitschen_2013_path_of_least_resistance.md",
                    "use": "grounds intraday breakout continuation",
                    "with_theory_claim": False,
                }
            ],
            "filter_grounding_status": {
                "verdict": "UNSUPPORTED",
                "basis": (
                    f"Mechanism narrative provided by operator at ingestion: "
                    f"{mechanism!r}. Cited literature: {literature_slug}. "
                    "v5.1 triage does NOT require theory citation; any "
                    "downstream heavyweight pre-reg must include the T0 "
                    "tautology audit + load-bearing literature extraction "
                    "before any deployment claim."
                ),
            },
        },
        "upstream_discovery_provenance": {
            "role": "PROVENANCE_ONLY",
            "sources": ["Ingested via scripts/research/ingest_idea.py (no prior validated_setups row or scan result)."],
            "note": (
                "Ingestion is the front-door for the fast-lane chain. The "
                "v5.1 .summary.csv emission downstream is the verdict "
                "evidence, not this file."
            ),
        },
        "primary_schema": {
            "family_cells": [
                {
                    "id": strategy_id,
                    "strategy_id": strategy_id,
                    "instrument": instrument,
                    "session": session,
                    "orb_minutes": orb_minutes,
                    "rr_target": rr,
                    "filter": filter_type,
                }
            ],
            "k_family": 1,
            "k_global": 1,
            "k_lane": 1,
            "k_session": 1,
            "chordia_threshold_basis": (
                "v5.1 triage screen: promote_threshold=2.5 + needs_more_band=0.5 (PROMOTE iff t >= 3.0)."
            ),
            "promotion_gate": "FAST_LANE PROMOTE only authorizes heavyweight Chordia review.",
        },
        "hypotheses": [
            {
                "id": 1,
                "name": f"{strategy_id} fast-lane v5.1 triage screen",
                "economic_basis": (
                    f"Operator-ingested mechanism: {mechanism}. "
                    f"Cited literature: docs/institutional/literature/{literature_slug}.md. "
                    "Triage-stage screen on whether the lane clears v5.1 "
                    "PROMOTE gates under canonical Mode A replay."
                ),
                "role": {
                    "kind": "standalone",
                    "parent": f"Exact {strategy_id} canonical lane replay.",
                    "comparator": "v5.1 triage thresholds.",
                    "primary_metric": "selected_trade_mean_r",
                    "promotion_target": "heavyweight_chordia_prereg_authorization_only",
                },
                "filter": {
                    "type": filter_type,
                    "thresholds": ["fire_vs_off"],
                },
                "scope": {
                    "instruments": [instrument],
                    "sessions": [session],
                    "rr_targets": [rr],
                    "entry_models": [entry_model],
                    "confirm_bars": [confirm_bars],
                    "stop_multipliers": [1.0],
                    "orb_minutes": [orb_minutes],
                },
                "expected_trial_count": 1,
                "kill_criteria": [
                    "IS t-stat < 2.5 (KILL — below triage floor)",
                    "IS ExpR <= 0.0 (KILL)",
                    "N_IS_on < 50 (KILL — below v5.1 sample floor)",
                    "Fire-rate outside [0.05, 0.95] (KILL — extreme)",
                    "Holdout boundary proof FALSE (KILL — leakage)",
                    "Canonical filter delegation diverges from trading_app.config",
                ],
                "needs_more_band": [
                    "t in [2.5, 3.0) -> NEEDS-MORE",
                    "Holdout proof FALSE on otherwise-passing cell -> NEEDS-MORE",
                    "Pooled direction lacks per-direction long/short ExpR breakdown -> NEEDS-MORE",
                ],
                "statement": (
                    f"On the fixed {strategy_id} lane using canonical "
                    f"{filter_type} delegation, IS t-stat clears the v5.1 "
                    "PROMOTE floor (3.0) with positive ExpR, fire-rate in "
                    "[0.05, 0.95], holdout-boundary proof TRUE, and "
                    "per-direction sign-check satisfied."
                ),
                "pass_metric": {
                    "metric": "t_IS",
                    "formula": "compute_chordia_t(sharpe_IS, N_IS) on canonical IS fired trades",
                    "threshold_gte": {
                        "t_is": 3.0,
                        "n_is_on": 50,
                        "expr_is": 0.0,
                    },
                    "extra_gates": [
                        "fire_rate in [0.05, 0.95]",
                        "holdout_boundary_proof TRUE",
                        "direction=pooled requires long_ExpR/short_ExpR emitted AND same sign",
                    ],
                },
                "counted_against_trial_budget": True,
            }
        ],
        "trial_budget": {
            "primary_selection_trials": 1,
            "schema_locked_before_any_metric": True,
            "minbtl_bound": "MinBTL = 2*ln(1)/1 = 0.0 years; N=1 within Bailey 2013 cap",
        },
        "total_hypothesis_count": 1,
        "total_expected_trials": 1,
        "budget_check": {
            "max_allowed_clean": 300,
            "max_allowed_proxy": 2000,
            "status": "under budget",
        },
        "decision_rule": {
            "promote_if": (
                "t_IS >= 3.0 AND ExpR_IS > 0 AND N_IS_on >= 50 AND "
                "fire_rate in [0.05, 0.95] AND holdout_boundary_proof AND "
                "(direction != 'pooled' OR (long_ExpR is not null AND "
                "short_ExpR is not null AND sign(long_ExpR) == sign(short_ExpR)))"
            ),
            "needs_more_if": (
                "t_IS in [2.5, 3.0) OR holdout_boundary_proof FALSE on "
                "otherwise-passing cell OR (direction == 'pooled' AND "
                "(long_ExpR is null OR short_ExpR is null OR "
                "sign(long_ExpR) != sign(short_ExpR)))"
            ),
            "kill_if": ("t_IS < 2.5 OR ExpR_IS <= 0 OR fire_rate outside [0.05, 0.95]"),
        },
        "methodology_rules_applied": {
            "rule_1_temporal_alignment": {
                "application": (
                    f"{filter_type} validated trade-time-knowable for "
                    f"{entry_model} per backtesting-methodology.md § 1 + § 6."
                )
            },
            "rule_3_is_oos_discipline": {
                "application": "HOLDOUT_SACRED_FROM=2026-01-01 enforced; OOS descriptive only."
            },
            "rule_4_multi_framing": {"application": "Pathway B K=1 triage screen."},
            "rule_9_canonical_layers": {"application": "Reads only orb_outcomes JOIN daily_features."},
            "rule_10_pre_registration": {
                "application": ("This file IS the pre-reg, written by ingest_idea.py before any fast-lane run.")
            },
        },
        "outputs_required_after_run": [
            f"result markdown at docs/audit/results/{now.strftime('%Y-%m-%d')}-{slug}-fast-lane-v1.md",
            "row-level CSV with IS/OOS split",
            "v5.1 summary.csv with the 19 columns",
            "applied v5.1 verdict (PROMOTE / KILL / NEEDS-MORE)",
        ],
        "execution_gate": {
            "allowed_now": True,
            "execution_surface": "Windows PowerShell with .venv",
            "forbidden_now": [
                "auto-writing PROMOTE rows into chordia_audit_log.yaml",
                "any deployment action",
                "sibling-cell rescue",
            ],
        },
        "not_done_by_this_pre_reg": [
            "Allocator deployment decision",
            "Theory grant assignment",
            "Sibling-RR / sibling-CB / sibling-session / sibling-aperture audits",
            "Heavyweight Chordia replay",
        ],
    }
    return prereg


def run_gates(
    *,
    instrument: str,
    session: str,
    orb_minutes: int,
    entry_model: str,
    confirm_bars: int,
    rr: float,
    direction: str,
    filter_type: str,
    mechanism: str,
    literature_slug: str,
) -> None:
    """Run all refusal gates. Raises IngestRefused on first failure."""
    _check_instrument(instrument)
    _check_session(session)
    _check_orb_minutes(orb_minutes)
    _check_entry_model(entry_model)
    _check_filter_and_e2_lookahead(entry_model, filter_type)
    _check_direction(direction)
    _check_mechanism(mechanism)
    _check_literature(literature_slug)
    if confirm_bars < 1:
        raise IngestRefused(f"confirm_bars={confirm_bars} must be >= 1.")
    if rr <= 0:
        raise IngestRefused(f"rr={rr} must be > 0.")


def write_draft(prereg: dict, draft_path: Path) -> None:
    """Write the pre-reg YAML to ``draft_path``. Parent dir auto-created."""
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True, width=100)
    draft_path.write_text(yaml_text, encoding="utf-8")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Idea Ingestion Front-Door for the Fast-Lane chain (Stage A). "
            "Writes a fast-lane v5.1 pre-reg under "
            "docs/audit/hypotheses/drafts/. Refusal is delegated to canonical "
            "sources (asset_configs, dst, ALL_FILTERS, "
            "E2_EXCLUDED_FILTER_PREFIXES/SUBSTRINGS, literature_index)."
        ),
    )
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--orb-minutes", required=True, type=int)
    parser.add_argument("--entry", required=True, dest="entry_model")
    parser.add_argument("--confirm-bars", required=True, type=int)
    parser.add_argument("--rr", required=True, type=float)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--filter", required=True, dest="filter_type")
    parser.add_argument(
        "--mechanism",
        required=True,
        help="One-line economic mechanism narrative.",
    )
    parser.add_argument(
        "--literature",
        required=True,
        dest="literature_slug",
        help=("Slug of an existing file under docs/institutional/literature/ (without .md extension)."),
    )
    parser.add_argument("--purpose", default=None)
    parser.add_argument(
        "--out-dir",
        default=str(DRAFTS_DIR),
        help=f"Override draft output dir (default: {DRAFTS_DIR}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        run_gates(
            instrument=args.instrument,
            session=args.session,
            orb_minutes=args.orb_minutes,
            entry_model=args.entry_model,
            confirm_bars=args.confirm_bars,
            rr=args.rr,
            direction=args.direction,
            filter_type=args.filter_type,
            mechanism=args.mechanism,
            literature_slug=args.literature_slug,
        )
    except IngestRefused as exc:
        print(f"INGEST_REFUSED: {exc}", file=sys.stderr)
        return 3

    now = datetime.now(_BRISBANE_TZ)
    prereg = build_prereg(
        instrument=args.instrument,
        session=args.session,
        orb_minutes=args.orb_minutes,
        entry_model=args.entry_model,
        confirm_bars=args.confirm_bars,
        rr=args.rr,
        direction=args.direction,
        filter_type=args.filter_type,
        mechanism=args.mechanism,
        literature_slug=args.literature_slug,
        purpose=args.purpose,
        now=now,
    )

    slug = build_slug(
        instrument=args.instrument,
        session=args.session,
        entry_model=args.entry_model,
        rr=args.rr,
        confirm_bars=args.confirm_bars,
        filter_type=args.filter_type,
        orb_minutes=args.orb_minutes,
        direction=args.direction,
    )
    draft_path = Path(args.out_dir) / f"{now.strftime('%Y-%m-%d')}-{slug}-fast-lane-v1.draft.yaml"
    write_draft(prereg, draft_path)
    print(str(draft_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
