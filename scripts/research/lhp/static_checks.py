"""Static checks run against an LLM-emitted hypothesis YAML BEFORE writing it.

Every check that overlaps an existing canonical authority DELEGATES to that
authority (institutional-rigor.md § 4). New logic is added only for two
checks that canonical sources do not already cover:

1. ``check_banned_features`` — RULE 6.3 E2 look-ahead. Delegates to
   ``trading_app.config.is_e2_lookahead_filter`` /
   ``is_e2_deployment_unsafe_filter`` for filter_type membership, and to a
   hard-coded column set for ``filter.column`` references (the column
   blacklist is repeated from ``.claude/rules/backtesting-methodology.md``
   § 6.3 because no canonical Python constant for it exists today).
2. ``check_citations_exist`` — every ``theory_citation`` must match a real
   file in ``docs/institutional/literature/``. Uses
   ``literature_index.citation_exists``.

All other checks call existing canonical authority and translate exceptions
to ``CheckFailure`` records so the orchestrator gets a uniform return shape.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.research.lhp.literature_index import (
    LiteratureEntry,
    citation_exists,
    verify_citation_content,
)


@dataclass(frozen=True)
class CheckFailure:
    code: str
    field: str
    detail: str
    fatal: bool


# Banned-column substrings on E2. From RULE 6.3 + RULE 1.1.
# Filter-type membership is delegated to trading_app.config; columns lack a
# canonical Python constant today so we keep them here. Any expansion is a
# one-line addition tracked by tests.
_E2_BANNED_COLUMN_SUBSTRINGS: tuple[str, ...] = (
    "break_ts",
    "break_delay",
    "break_bar_continues",
    "break_bar_volume",
    "break_dir",  # banned only as predictor; static check is conservative
    "rel_vol_",
    "double_break",
    "_mae_r",
    "_mfe_r",
    "_outcome",
    "pnl_r",
)


def _parse_yaml(yaml_text: str) -> tuple[dict[str, Any] | None, CheckFailure | None]:
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return None, CheckFailure(
            code="YAML_PARSE_ERROR",
            field="<root>",
            detail=f"YAML did not parse: {exc}",
            fatal=True,
        )
    if not isinstance(data, dict):
        return None, CheckFailure(
            code="YAML_NOT_MAPPING",
            field="<root>",
            detail=f"Top-level YAML must be a mapping, got {type(data).__name__}",
            fatal=True,
        )
    return data, None


def check_schema_load(yaml_text: str) -> list[CheckFailure]:
    """Round-trip through ``hypothesis_loader.load_hypothesis_metadata``.

    The canonical loader enforces top-level keys, metadata keys,
    ``total_expected_trials > 0`` (excluding ``bool``), ``holdout_date``
    parseable, ``research_question_type`` / ``testing_mode`` enums, plus
    ``filter.type`` registered in ``ALL_FILTERS``. We write the candidate
    YAML to a temp file and call the loader to reuse all of that.
    """
    # Import locally so unit tests that mock the LLM do not require the
    # full trading_app package to be importable in every test scope.
    from trading_app.hypothesis_loader import (
        HypothesisLoaderError,
        load_hypothesis_metadata,
    )

    parsed, parse_fail = _parse_yaml(yaml_text)
    if parse_fail is not None:
        return [parse_fail]
    assert parsed is not None

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml", delete=False) as fp:
        fp.write(yaml_text)
        tmp_path = Path(fp.name)
    try:
        try:
            load_hypothesis_metadata(tmp_path)
        except HypothesisLoaderError as exc:
            return [
                CheckFailure(
                    code="SCHEMA_LOAD_FAILED",
                    field="<root>",
                    detail=str(exc),
                    fatal=True,
                )
            ]
    finally:
        tmp_path.unlink(missing_ok=True)
    return []


def check_banned_features(parsed: dict[str, Any]) -> list[CheckFailure]:
    """RULE 6.3 — banned E2 look-ahead features.

    Delegates filter-type membership to
    ``trading_app.config.is_e2_deployment_unsafe_filter``. Column substring
    check uses the local ``_E2_BANNED_COLUMN_SUBSTRINGS`` set (no canonical
    Python constant exists for these column names today).
    """
    from trading_app.config import is_e2_deployment_unsafe_filter

    failures: list[CheckFailure] = []
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return failures  # schema check will catch this

    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        scope = hyp.get("scope")
        if not isinstance(scope, dict):
            continue
        ems = scope.get("entry_models", [])
        if not isinstance(ems, list) or "E2" not in ems:
            continue  # only E2 is gated by RULE 6.3
        filt = hyp.get("filter")
        if not isinstance(filt, dict):
            continue
        ftype = filt.get("type", "")
        fcol = filt.get("column", "")
        field_prefix = f"hypotheses[{idx}].filter"
        if isinstance(ftype, str) and ftype and is_e2_deployment_unsafe_filter(ftype):
            failures.append(
                CheckFailure(
                    code="BANNED_FEATURE",
                    field=f"{field_prefix}.type",
                    detail=(
                        f"filter.type {ftype!r} is E2 look-ahead per "
                        "trading_app.config.is_e2_deployment_unsafe_filter "
                        "(RULE 6.3, backtesting-methodology.md)."
                    ),
                    fatal=True,
                )
            )
        if isinstance(fcol, str):
            lowered = fcol.lower()
            for needle in _E2_BANNED_COLUMN_SUBSTRINGS:
                if needle in lowered:
                    failures.append(
                        CheckFailure(
                            code="BANNED_FEATURE",
                            field=f"{field_prefix}.column",
                            detail=(
                                f"filter.column {fcol!r} contains banned "
                                f"substring {needle!r} (E2 look-ahead per "
                                "RULE 6.3)."
                            ),
                            fatal=True,
                        )
                    )
                    break  # one failure per column is enough
    return failures


def check_holdout_date(parsed: dict[str, Any]) -> list[CheckFailure]:
    """Delegate to ``holdout_policy.enforce_holdout_date``.

    The canonical helper raises ``ValueError`` when ``holdout_date`` exceeds
    ``HOLDOUT_SACRED_FROM``. Any earlier date is accepted by the helper —
    but the LLM proposer specifically wants Mode A (sacred boundary), so we
    additionally reject ``holdout_date < HOLDOUT_SACRED_FROM`` as a
    pre-registration mismatch.
    """
    from trading_app.holdout_policy import (
        HOLDOUT_SACRED_FROM,
        enforce_holdout_date,
    )

    meta = parsed.get("metadata", {})
    if not isinstance(meta, dict):
        return []  # schema check covers this
    raw = meta.get("holdout_date")
    if isinstance(raw, datetime):
        cmp = raw.date()
    elif isinstance(raw, date):
        cmp = raw
    elif isinstance(raw, str):
        try:
            cmp = date.fromisoformat(raw)
        except ValueError:
            return [
                CheckFailure(
                    code="WRONG_HOLDOUT",
                    field="metadata.holdout_date",
                    detail=f"holdout_date {raw!r} not parseable as ISO date",
                    fatal=True,
                )
            ]
    else:
        return [
            CheckFailure(
                code="WRONG_HOLDOUT",
                field="metadata.holdout_date",
                detail=f"holdout_date must be a date or ISO string, got {type(raw).__name__}",
                fatal=True,
            )
        ]
    try:
        enforce_holdout_date(cmp)
    except ValueError as exc:
        return [
            CheckFailure(
                code="WRONG_HOLDOUT",
                field="metadata.holdout_date",
                detail=str(exc),
                fatal=True,
            )
        ]
    if cmp != HOLDOUT_SACRED_FROM:
        return [
            CheckFailure(
                code="WRONG_HOLDOUT",
                field="metadata.holdout_date",
                detail=(
                    f"holdout_date {cmp.isoformat()} != Mode A sacred "
                    f"{HOLDOUT_SACRED_FROM.isoformat()} "
                    "(pre_registered_criteria.md Amendment 2.7)"
                ),
                fatal=True,
            )
        ]
    return []


def check_minbtl_budget(parsed: dict[str, Any]) -> list[CheckFailure]:
    """Delegate to ``hypothesis_loader.enforce_minbtl_bound``.

    Strict Bailey E=1.0 bound for MNQ 6.65yr is ~28 — that is a default the
    proposer aims for but not a hard fatal: the operational ceiling per
    Criterion 2 Amendment 2.8 is 300 (clean) or 2000 (proxy). The loader
    enforces the hard ceiling; we add a non-fatal warning when the strict
    bound is exceeded.
    """
    from trading_app.hypothesis_loader import (
        HypothesisLoaderError,
        enforce_minbtl_bound,
    )

    meta = parsed.get("metadata", {})
    if not isinstance(meta, dict):
        return []
    declared = meta.get("total_expected_trials")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
        return []  # schema check covers shape

    on_proxy = meta.get("data_source_mode") == "proxy"
    # enforce_minbtl_bound reads metadata.data_source_mode under meta["metadata"],
    # because it expects the load_hypothesis_metadata output shape. Build the
    # minimal envelope here.
    envelope = {"total_expected_trials": declared, "metadata": meta}
    try:
        verdict, reason = enforce_minbtl_bound(envelope, on_proxy_data=on_proxy)
    except HypothesisLoaderError as exc:
        return [
            CheckFailure(
                code="MINBTL_BUDGET_EXCEEDED",
                field="metadata.total_expected_trials",
                detail=str(exc),
                fatal=True,
            )
        ]
    failures: list[CheckFailure] = []
    if verdict == "REJECTED":
        failures.append(
            CheckFailure(
                code="MINBTL_BUDGET_EXCEEDED",
                field="metadata.total_expected_trials",
                detail=reason or "MinBTL bound exceeded",
                fatal=True,
            )
        )
    # Strict Bailey E=1.0 warning: 28 trials for MNQ 6.65yr clean.
    if not on_proxy and declared > 28:
        failures.append(
            CheckFailure(
                code="MINBTL_LOOSE_OPERATIONAL",
                field="metadata.total_expected_trials",
                detail=(
                    f"total_expected_trials={declared} exceeds strict Bailey "
                    "E=1.0 bound (28 for MNQ 6.65yr clean). Operational "
                    "ceiling 300 still respected, but DSR will be lower."
                ),
                fatal=False,
            )
        )
    return failures


def check_citations_exist(parsed: dict[str, Any], corpus: Sequence[LiteratureEntry]) -> list[CheckFailure]:
    """Every ``theory_citation`` must match a real corpus entry."""
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    failures: list[CheckFailure] = []
    any_real = False
    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        cite = hyp.get("theory_citation")
        if not isinstance(cite, str) or not cite.strip():
            failures.append(
                CheckFailure(
                    code="CITATION_MISSING",
                    field=f"hypotheses[{idx}].theory_citation",
                    detail="theory_citation is required and must be non-empty",
                    fatal=True,
                )
            )
            continue
        if citation_exists(corpus, cite):
            any_real = True
        else:
            failures.append(
                CheckFailure(
                    code="CITATION_NOT_FOUND",
                    field=f"hypotheses[{idx}].theory_citation",
                    detail=(
                        f"theory_citation {cite!r} does not match any file in "
                        "docs/institutional/literature/. Either cite an "
                        "existing extract or write a new extract first."
                    ),
                    fatal=True,
                )
            )
    if hypotheses and not any_real:
        failures.append(
            CheckFailure(
                code="CITATION_NOT_FOUND",
                field="hypotheses",
                detail="Zero hypotheses cite a real on-disk literature file",
                fatal=True,
            )
        )
    return failures


def check_instruments_active(parsed: dict[str, Any]) -> list[CheckFailure]:
    """Delegate to ``pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS``."""
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    failures: list[CheckFailure] = []
    active = set(ACTIVE_ORB_INSTRUMENTS)
    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        scope = hyp.get("scope")
        if not isinstance(scope, dict):
            continue
        for instr in scope.get("instruments", []) or []:
            if instr not in active:
                failures.append(
                    CheckFailure(
                        code="INSTRUMENT_INACTIVE",
                        field=f"hypotheses[{idx}].scope.instruments",
                        detail=(
                            f"instrument {instr!r} not in ACTIVE_ORB_INSTRUMENTS "
                            f"({sorted(active)}). MCL/SIL/M6E/MBT/M2K are dead "
                            "for ORB."
                        ),
                        fatal=True,
                    )
                )
    return failures


def check_sessions_valid(parsed: dict[str, Any]) -> list[CheckFailure]:
    """Delegate to ``pipeline.dst.SESSION_CATALOG``."""
    from pipeline.dst import SESSION_CATALOG

    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    failures: list[CheckFailure] = []
    catalog = set(SESSION_CATALOG.keys())
    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        scope = hyp.get("scope")
        if not isinstance(scope, dict):
            continue
        for sess in scope.get("sessions", []) or []:
            if sess not in catalog:
                failures.append(
                    CheckFailure(
                        code="SESSION_NOT_IN_CATALOG",
                        field=f"hypotheses[{idx}].scope.sessions",
                        detail=(f"session {sess!r} not in SESSION_CATALOG. Known sessions: {sorted(catalog)}"),
                        fatal=True,
                    )
                )
    return failures


def check_scratch_policy(parsed: dict[str, Any]) -> list[CheckFailure]:
    """C13 BINDING — scratch_policy must be declared and = realized-eod for capital decisions.

    Per ``feedback_chordia_unlock_deployment_gate_audit_checklist.md`` and
    ``pre_registered_criteria.md`` C13: every new pre-reg must declare
    ``scratch_policy`` at the top level. ``realized-eod`` is the required
    value when results may inform capital decisions; ``include-zero`` or
    ``drop`` must be explicitly justified.

    Origin: yesterday's 3 LLM-drafted pre-regs all omitted this field. The
    runner did not block, the rejections came on Criterion 8, and IS
    baselines were potentially inflated 10-45% by ``WHERE pnl_r IS NOT NULL``
    silent drops.
    """
    failures: list[CheckFailure] = []
    raw = parsed.get("scratch_policy")
    if raw is None:
        failures.append(
            CheckFailure(
                code="SCRATCH_POLICY_MISSING",
                field="scratch_policy",
                detail=(
                    "scratch_policy block is required (pre_registered_criteria.md C13 BINDING). "
                    "Add `scratch_policy: {policy: realized-eod, justification: ...}` at the top level."
                ),
                fatal=True,
            )
        )
        return failures
    if isinstance(raw, str):
        policy = raw
        justification: Any = ""
    elif isinstance(raw, dict):
        policy = raw.get("policy", "")
        justification = raw.get("justification", "")
    else:
        return [
            CheckFailure(
                code="SCRATCH_POLICY_MALFORMED",
                field="scratch_policy",
                detail=f"scratch_policy must be a string or mapping, got {type(raw).__name__}",
                fatal=True,
            )
        ]
    if not isinstance(policy, str) or not policy.strip():
        failures.append(
            CheckFailure(
                code="SCRATCH_POLICY_MISSING",
                field="scratch_policy.policy",
                detail="scratch_policy.policy must be a non-empty string",
                fatal=True,
            )
        )
        return failures
    allowed = {"realized-eod", "include-zero", "drop"}
    if policy not in allowed:
        failures.append(
            CheckFailure(
                code="SCRATCH_POLICY_INVALID",
                field="scratch_policy.policy",
                detail=f"scratch_policy.policy={policy!r} not in {sorted(allowed)}",
                fatal=True,
            )
        )
        return failures
    if policy != "realized-eod" and not (isinstance(justification, str) and justification.strip()):
        failures.append(
            CheckFailure(
                code="SCRATCH_POLICY_UNJUSTIFIED",
                field="scratch_policy.justification",
                detail=(
                    f"scratch_policy.policy={policy!r} (non-default) requires "
                    "justification: text explaining why realized-eod is inappropriate"
                ),
                fatal=True,
            )
        )
    return failures


def check_oos_power_floor(parsed: dict[str, Any]) -> list[CheckFailure]:
    """RULE 3.3 — any binary OOS kill criterion requires a power-floor declaration.

    If ``kill_criteria`` mentions ``OOS ExpR``, ``dir_match``, or any binary
    OOS gate, the pre-reg MUST declare an OOS power floor (under
    ``oos_power_floor`` at metadata-level OR per-hypothesis) referencing the
    tier (CAN_REFUTE / DIRECTIONAL_ONLY / STATISTICALLY_USELESS) the result
    needs to clear before applying the gate.

    Reference incident: 2026-04-20 ``bull_short_avoidance`` would have been
    falsely killed on a STATISTICALLY_USELESS OOS (power 7.9%) without this
    rule.
    """
    failures: list[CheckFailure] = []
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []

    binary_oos_keywords = ("oos expr", "oos_expr", "dir_match", "sign-flip", "p_oos", "oos ratio")

    def has_binary_oos_kill(hyp: dict[str, Any]) -> bool:
        kc = hyp.get("kill_criteria") or []
        if not isinstance(kc, list):
            kc = [kc]
        for entry in kc:
            entry_str = str(entry).lower() if entry is not None else ""
            if any(kw in entry_str for kw in binary_oos_keywords):
                return True
        return False

    meta_pf = (
        parsed.get("oos_power_floor") or parsed.get("metadata", {}).get("oos_power_floor")
        if isinstance(parsed.get("metadata"), dict)
        else parsed.get("oos_power_floor")
    )
    meta_has_pf = bool(meta_pf)

    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        if not has_binary_oos_kill(hyp):
            continue
        per_hyp_pf = hyp.get("oos_power_floor")
        if per_hyp_pf or meta_has_pf:
            continue
        failures.append(
            CheckFailure(
                code="OOS_POWER_FLOOR_MISSING",
                field=f"hypotheses[{idx}].oos_power_floor",
                detail=(
                    "kill_criteria references a binary OOS gate (ExpR<0, dir_match, sign-flip) "
                    "but no oos_power_floor is declared. Per backtesting-methodology.md RULE 3.3, "
                    "binary OOS gates require a power-tier declaration (CAN_REFUTE / "
                    "DIRECTIONAL_ONLY / STATISTICALLY_USELESS) computed via "
                    "research.oos_power.oos_ttest_power() before kill applies."
                ),
                fatal=True,
            )
        )
    return failures


def check_sensitivity_test(parsed: dict[str, Any]) -> list[CheckFailure]:
    """RESEARCH_RULES.md § Sensitivity — parameterised filters need ±N variants.

    If a hypothesis declares a filter with a numeric threshold (ATR_P30,
    ORB_VOL_16K, OVNRNG_100, etc.), the pre-reg must declare at least 2
    additional threshold variants under ``sensitivity_test.axes`` so a
    curve-fit can be detected.
    """
    failures: list[CheckFailure] = []
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    import re as _re

    has_threshold = _re.compile(r"\d")

    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        filt = hyp.get("filter")
        if not isinstance(filt, dict):
            continue
        ftype = str(filt.get("type") or "")
        if not has_threshold.search(ftype):
            continue  # no numeric threshold to be sensitive about
        sens = hyp.get("sensitivity_test") or filt.get("sensitivity_test") or {}
        if isinstance(sens, dict):
            axes = sens.get("axes") or sens.get("variants") or []
        else:
            axes = sens
        if not isinstance(axes, list) or len(axes) < 2:
            failures.append(
                CheckFailure(
                    code="SENSITIVITY_TEST_MISSING",
                    field=f"hypotheses[{idx}].sensitivity_test.axes",
                    detail=(
                        f"filter.type={ftype!r} has a numeric threshold but pre-reg declares "
                        f"fewer than 2 sensitivity variants. Per RESEARCH_RULES.md "
                        "(ATR/RSI/MA thresholds especially prone to curve-fitting), declare "
                        "at least 2 ±N% variants under sensitivity_test.axes."
                    ),
                    fatal=True,
                )
            )
    return failures


def check_prior_art_block(parsed: dict[str, Any]) -> list[CheckFailure]:
    """Require a ``prior_art`` block citing the neighbor-scan output.

    The proposer pipeline writes neighbour scan results into the draft;
    this check confirms the LLM didn't strip the block. A pre-reg without
    prior_art commitments can re-litigate a documented KILL without
    acknowledging it.
    """
    pa = (
        parsed.get("prior_art") or parsed.get("metadata", {}).get("prior_art")
        if isinstance(parsed.get("metadata"), dict)
        else parsed.get("prior_art")
    )
    if pa is None:
        return [
            CheckFailure(
                code="PRIOR_ART_MISSING",
                field="prior_art",
                detail=(
                    "prior_art block is required. Should contain `family_health`, "
                    "`siblings_killed`, `siblings_blocked_by_graveyard`, and a list "
                    "of relevant prior audit verdicts (auto-generated by proposer)."
                ),
                fatal=True,
            )
        ]
    if isinstance(pa, dict):
        # Must declare family_health at minimum.
        if not pa.get("family_health"):
            return [
                CheckFailure(
                    code="PRIOR_ART_INCOMPLETE",
                    field="prior_art.family_health",
                    detail="prior_art.family_health required (CLEAN/MIXED/HOSTILE)",
                    fatal=True,
                )
            ]
    return []


def check_citation_content(parsed: dict[str, Any], corpus: Sequence[LiteratureEntry]) -> list[CheckFailure]:
    """Verify each theory_citation's content actually addresses the economic_basis.

    Uses ``literature_index.verify_citation_content`` (lexical token overlap)
    to catch fabricated cites and topic-mismatched cites. Cannot catch
    same-vocabulary category errors (e.g. Carver-sizing → entry-filter) on
    its own; the proposer-side prompt + the ``passage_quote`` field
    (declared but not enforced as of v1) close that gap.
    """
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    failures: list[CheckFailure] = []
    for idx, hyp in enumerate(hypotheses):
        if not isinstance(hyp, dict):
            continue
        cite = hyp.get("theory_citation")
        basis = hyp.get("economic_basis")
        if not isinstance(cite, str) or not cite.strip():
            continue  # check_citations_exist will handle missing cites
        if not isinstance(basis, str) or not basis.strip():
            failures.append(
                CheckFailure(
                    code="ECONOMIC_BASIS_MISSING",
                    field=f"hypotheses[{idx}].economic_basis",
                    detail="economic_basis text required to verify citation content",
                    fatal=True,
                )
            )
            continue
        result = verify_citation_content(corpus, cite, basis)
        if not result["passes"]:
            failures.append(
                CheckFailure(
                    code="CITATION_CONTENT_MISMATCH",
                    field=f"hypotheses[{idx}].theory_citation",
                    detail=(
                        f"{result['reason']}. "
                        f"Cited files: {result['cited_files']}. "
                        "Either rewrite economic_basis to actually align with the cited "
                        "passages, or cite a different file."
                    ),
                    fatal=True,
                )
            )
    return failures


def run_all(yaml_text: str, corpus: Sequence[LiteratureEntry]) -> tuple[dict[str, Any] | None, list[CheckFailure]]:
    """Run every check. Schema-load runs first; on parse failure we stop.

    Returns ``(parsed, failures)``. ``parsed`` is None if the YAML did not
    parse at all (in which case ``failures`` contains the parse error).
    Otherwise ``parsed`` is the dict from ``yaml.safe_load`` and ``failures``
    is the concatenated list across every check (fatal AND non-fatal).
    """
    failures: list[CheckFailure] = []
    parsed, parse_fail = _parse_yaml(yaml_text)
    if parse_fail is not None:
        failures.append(parse_fail)
        return None, failures
    assert parsed is not None

    failures.extend(check_schema_load(yaml_text))
    # If schema-load fails fatally, downstream checks may crash on missing
    # structure — short-circuit.
    if any(f.fatal for f in failures):
        return parsed, failures

    failures.extend(check_banned_features(parsed))
    failures.extend(check_holdout_date(parsed))
    failures.extend(check_minbtl_budget(parsed))
    failures.extend(check_citations_exist(parsed, corpus))
    failures.extend(check_citation_content(parsed, corpus))
    failures.extend(check_instruments_active(parsed))
    failures.extend(check_sessions_valid(parsed))
    failures.extend(check_scratch_policy(parsed))
    failures.extend(check_oos_power_floor(parsed))
    failures.extend(check_sensitivity_test(parsed))
    failures.extend(check_prior_art_block(parsed))
    return parsed, failures


__all__ = [
    "CheckFailure",
    "check_banned_features",
    "check_citation_content",
    "check_citations_exist",
    "check_holdout_date",
    "check_instruments_active",
    "check_minbtl_budget",
    "check_oos_power_floor",
    "check_prior_art_block",
    "check_schema_load",
    "check_scratch_policy",
    "check_sensitivity_test",
    "check_sessions_valid",
    "run_all",
]
