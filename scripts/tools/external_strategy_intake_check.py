#!/usr/bin/env python3
"""Validate external strategy intake records before preregistration.

The gate is intentionally small: it checks that imported ideas carry enough
provenance, bias accounting, and trial-budget discipline before they can feed
the existing preregistration workflow.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DECISIONS = {
    "BIN",
    "DOC_ONLY",
    "PREREG_CANDIDATE",
    "INFRA_CANDIDATE",
    "ADJACENT_STACK_CANDIDATE",
}

ALLOWED_ROLES = {
    "standalone",
    "filter",
    "veto",
    "sizing_overlay",
    "allocator_input",
    "execution_aid",
    "visualization_aid",
    "dead_end",
}

ALLOWED_REPO_COVERAGE = {
    "already_covered",
    "covered_as_process_gap",
    "killed",
    "adjacent",
    "genuinely_new",
    "unknown",
}

PREREG_ALLOWED_ROLES = {
    "standalone",
    "filter",
    "veto",
    "sizing_overlay",
    "allocator_input",
}

LIST_FIELDS = {
    "packaging_removed",
    "bias_risks",
    "negative_evidence",
    "golden_nuggets",
    "evidence_refs",
}

PINE_SOURCE_TYPES = {"pine", "pine_script", "tradingview", "tradingview_pine"}
PINE_RISK_FLAGS = {
    "request_security",
    "repaint",
    "broker_emulator",
    "bar_magnifier",
    "data_vendor_mismatch",
}

FORBIDDEN_EVIDENCE_PREFIXES = ("memory/",)
FORBIDDEN_EVIDENCE_EXACT = {"HANDOFF.md"}
FORBIDDEN_EVIDENCE_FRAGMENTS = ("screenshot", "screenshots", "strategy-tester")
FORBIDDEN_EVIDENCE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

OOS_LEAKAGE_RE = re.compile(
    r"\b("
    r"tune again|"
    r"iterate on oos|"
    r"iterative oos|"
    r"reuse oos|"
    r"optimize oos|"
    r"optimi[sz]e (?:on )?holdout|"
    r"use holdout to select|"
    r"select (?:the )?best .*holdout|"
    r"peek at (?:the )?holdout"
    r")\b",
    re.I,
)
NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
THRESHOLD_RE = re.compile(
    r"(<=|>=|<|>|=|\bat least\b|\bat most\b|\bbelow\b|\babove\b|\bno more than\b|\bno less than\b)",
    re.I,
)

# Grounded in docs/institutional/pre_registered_criteria.md Criterion 2,
# which applies Bailey et al. 2013 MinBTL discipline from
# docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md.
MAX_CLEAN_TRIALS = 300
MAX_PROXY_TRIALS = 2000


def _load_yaml(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [f"{path}: invalid YAML: {exc}"]
    except OSError as exc:
        return None, [f"{path}: cannot read file: {exc}"]
    if not isinstance(loaded, dict):
        return None, [f"{path}: top-level YAML must be a mapping"]
    return loaded, []


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def _require_mapping(record: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return {}
    return value


def _require_nonblank(record: dict[str, Any], key: str, errors: list[str]) -> None:
    if _is_blank(record.get(key)):
        errors.append(f"{key} is required")


def _require_nonempty_list(record: dict[str, Any], key: str, errors: list[str]) -> None:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{key} must be a non-empty list")


def _require_source(source: dict[str, Any], errors: list[str]) -> None:
    for key in ("title", "url_or_path", "source_type", "reviewed_date", "authority_level"):
        if _is_blank(source.get(key)):
            errors.append(f"source.{key} is required")


def _validate_decision(record: dict[str, Any], errors: list[str]) -> str:
    decision = record.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"decision must be one of {sorted(ALLOWED_DECISIONS)}")
        return ""
    return str(decision)


def _validate_role(record: dict[str, Any], errors: list[str]) -> None:
    role = record.get("best_role")
    if role not in ALLOWED_ROLES:
        errors.append(f"best_role must be one of {sorted(ALLOWED_ROLES)}")


def _validate_repo_coverage(record: dict[str, Any], errors: list[str]) -> None:
    coverage = record.get("repo_coverage")
    if coverage not in ALLOWED_REPO_COVERAGE:
        errors.append(f"repo_coverage must be one of {sorted(ALLOWED_REPO_COVERAGE)}")


def _validate_evidence(record: dict[str, Any], errors: list[str]) -> None:
    refs = record.get("evidence_refs", [])
    if refs is None:
        return
    if not isinstance(refs, list):
        errors.append("evidence_refs must be a list when present")
        return
    for ref in refs:
        if not isinstance(ref, str):
            errors.append("evidence_refs entries must be strings")
            continue
        if ref in FORBIDDEN_EVIDENCE_EXACT:
            errors.append(f"{ref} is not valid evidence for external intake")
        if ref.startswith(FORBIDDEN_EVIDENCE_PREFIXES):
            errors.append(f"{ref} is not valid evidence; memory/ is scratch context")
        lowered = ref.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_EVIDENCE_FRAGMENTS) or lowered.endswith(
            FORBIDDEN_EVIDENCE_SUFFIXES
        ):
            errors.append(f"{ref} is not valid evidence; screenshots/backtest images are provenance only")


def _validate_prereg(record: dict[str, Any], errors: list[str]) -> None:
    role = record.get("best_role")
    if role not in PREREG_ALLOWED_ROLES:
        errors.append(f"PREREG_CANDIDATE best_role must be one of {sorted(PREREG_ALLOWED_ROLES)}")

    budget = _require_mapping(record, "trial_budget", errors)
    mode = budget.get("mode")
    if mode not in {"clean", "proxy"}:
        errors.append("trial_budget.mode must be clean or proxy for PREREG_CANDIDATE")
    max_trials = budget.get("max_trials")
    if not isinstance(max_trials, int) or max_trials <= 0:
        errors.append("trial_budget.max_trials must be a positive integer")
    elif mode == "proxy" and max_trials > MAX_PROXY_TRIALS:
        errors.append(f"trial_budget.max_trials exceeds proxy cap of {MAX_PROXY_TRIALS}")
    elif mode == "clean" and max_trials > MAX_CLEAN_TRIALS:
        errors.append(f"trial_budget.max_trials exceeds clean cap of {MAX_CLEAN_TRIALS}")

    kill_criteria = record.get("kill_criteria")
    if not isinstance(kill_criteria, list) or not kill_criteria:
        errors.append("kill_criteria must list numeric refutation criteria")
    elif not any(isinstance(item, str) and _has_numeric_threshold(item) for item in kill_criteria):
        errors.append("numeric threshold kill_criteria are required")

    oos_policy = record.get("oos_policy")
    if _is_blank(oos_policy):
        errors.append("oos_policy is required for PREREG_CANDIDATE")
    elif not isinstance(oos_policy, str):
        errors.append("oos_policy must be a string")
    elif _has_oos_leakage(oos_policy):
        errors.append("oos_policy contains iterative OOS leakage wording")


def _has_numeric_threshold(text: str) -> bool:
    return NUMERIC_RE.search(text) is not None and THRESHOLD_RE.search(text) is not None


def _has_oos_leakage(text: str) -> bool:
    lowered = text.lower()
    if "no iterative oos" in lowered or "no iterative holdout" in lowered:
        return False
    return OOS_LEAKAGE_RE.search(text) is not None


def _validate_optimizer_claims(record: dict[str, Any], errors: list[str]) -> None:
    claims = record.get("source_claims", {})
    if not isinstance(claims, dict):
        errors.append("source_claims must be a mapping when present")
        return
    optimizer_variants = claims.get("optimizer_variants")
    if optimizer_variants is None:
        return
    budget = record.get("trial_budget", {})
    disclosed = isinstance(budget, dict) and isinstance(budget.get("source_trial_count_disclosed"), int)
    if not disclosed:
        errors.append("source_trial_count_disclosed is required when source claims optimizer variants")


def _validate_optimization_space(record: dict[str, Any], errors: list[str]) -> None:
    space = record.get("optimization_space")
    if not isinstance(space, dict):
        return
    params = space.get("parameters", [])
    if not isinstance(params, list):
        errors.append("optimization_space.parameters must be a list")
        return
    if len(params) <= 1:
        return
    constraints = space.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        errors.append("optimization_space.constraints is required for multi-parameter ideas")
    if record.get("stability_surface_required") is not True:
        errors.append("stability_surface_required must be true for multi-parameter ideas")


def _validate_pine(source: dict[str, Any], record: dict[str, Any], errors: list[str]) -> None:
    source_type = str(source.get("source_type", "")).lower()
    if source_type not in PINE_SOURCE_TYPES:
        return
    flags = record.get("pine_risk_flags")
    if not isinstance(flags, list):
        errors.append("pine_risk_flags must list Pine/TradingView risks for Pine imports")
        return
    missing = sorted(PINE_RISK_FLAGS - set(flags))
    if missing:
        errors.append(f"pine_risk_flags missing required risks: {missing}")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors for one intake record."""
    errors: list[str] = []
    source = _require_mapping(record, "source", errors)
    _require_source(source, errors)

    for key in (
        "mechanism_family",
        "baseline_to_beat",
        "next_action",
    ):
        _require_nonblank(record, key, errors)
    for key in LIST_FIELDS:
        _require_nonempty_list(record, key, errors)

    decision = _validate_decision(record, errors)
    _validate_role(record, errors)
    _validate_repo_coverage(record, errors)
    _validate_evidence(record, errors)
    _validate_optimizer_claims(record, errors)
    _validate_optimization_space(record, errors)
    _validate_pine(source, record, errors)

    if decision == "PREREG_CANDIDATE":
        _validate_prereg(record, errors)

    return errors


def validate_file(path: Path) -> list[str]:
    record, errors = _load_yaml(path)
    if record is None:
        return errors
    return [f"{path}: {error}" for error in validate_record(record)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="YAML intake files to validate")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for path in args.paths:
        errors.extend(validate_file(path))

    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"PASS external strategy intake validation ({len(args.paths)} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
