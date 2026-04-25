"""Regression guard: files that have been fixed to use canonical filter_signal
delegation must NOT reintroduce inline filter SQL or hardcoded commission
constants.

Origin: 2026-04-19 campaign (see `docs/audit/results/2026-04-19-research-filter-delegation-audit.md`
and its Addendum 2). The OVNRNG_100 ratio-vs-absolute incident of 2026-04-19
proved that inline filter SQL in research scripts silently drifts from
canonical `trading_app.config.ALL_FILTERS`. Once a file is delegated via
`research.filter_utils.filter_signal`, this guard prevents the inline
pattern from creeping back in.

This is a STATIC test — no DuckDB fixture required, runs in CI without
gold.db. Intended to catch regressions at PR-review time, not at runtime.

If a new legitimate non-filter use of `d.<col> >= N.N` SQL or a numeric
constant like `2.74` is required (e.g., a custom research-only feature
candidate that is NOT a canonical filter re-implementation), add the
file to `EXEMPT_FILES` below with a justifying comment.

Per `.claude/rules/institutional-rigor.md` rule 5 (no dead code) and rule 4
(delegate to canonical sources).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# Files that have been migrated to canonical filter_signal delegation.
# Add files here as each delegation PR merges — this is the "fixed shelf."
DELEGATED_FILES = [
    "research/garch_validated_role_exhaustion.py",
    # PR #13 (garch_comex_settle_institutional_battery.py) is added to this
    # list once it merges to main — tracked in the 2026-04-19 audit addendum.
]


# File-level exemptions for the hardcoded-commission guard. Empty for now.
# Add with justifying comment if a future research script legitimately
# needs to reference the pre-Rithmic MNQ commission ($2.74) for a
# historical / archival purpose — e.g., a dated pre-registered analysis
# frozen to the pre-Rithmic regime.
EXEMPT_FILES_COMMISSION: set[str] = set()


# Banned patterns. Each entry is (regex, human-readable description).
# NB: regexes use raw strings; they must match what grep would match.
BANNED_INLINE_FILTER_PATTERNS = [
    (
        r'"d\.atr_20_pct\s*>=',
        "Inline ATR_P* SQL fragment. Use filter_signal(df, 'ATR_P{N}', orb_label).",
    ),
    (
        r'"d\.overnight_range\s*>=',
        "Inline OVNRNG_* SQL fragment. Use filter_signal(df, 'OVNRNG_{N}', orb_label).",
    ),
    (
        r'"d\.orb_\w+_size\s*>=',
        "Inline ORB_G* SQL fragment. Use filter_signal(df, 'ORB_G{N}', orb_label).",
    ),
    (
        r"mes\.atr_20_pct\s*>=",
        "Inline cross-asset MES ATR SQL. Use filter_signal with 'X_MES_ATR{N}' "
        "and CrossAssetATRFilter enrichment path.",
    ),
]


BANNED_COMMISSION_PATTERNS = [
    (
        r"\b2\.74\s*/",
        "Hardcoded pre-Rithmic MNQ commission ($2.74). Source from "
        "pipeline.cost_model.COST_SPECS['MNQ'].commission_rt "
        "(currently $1.42 under Rithmic).",
    ),
    (
        r'FRICTION\s*=\s*\{[^}]*"MNQ"\s*:\s*2\.74',
        "Hardcoded pre-Rithmic FRICTION dict. Source from COST_SPECS.",
    ),
]


@pytest.mark.parametrize("relpath", DELEGATED_FILES)
def test_delegated_file_imports_filter_signal(relpath: str) -> None:
    """A file declared delegated must import the canonical wrapper."""
    path = REPO_ROOT / relpath
    assert path.exists(), f"Delegated file missing: {path}"
    text = path.read_text(encoding="utf-8")
    assert "from research.filter_utils import filter_signal" in text or "import research.filter_utils" in text, (
        f"{relpath} is listed as delegated but does not import "
        f"research.filter_utils.filter_signal. "
        f"Per research-truth-protocol.md § Canonical filter delegation, "
        f"every research script applying a canonical filter must route "
        f"through filter_signal."
    )


@pytest.mark.parametrize("relpath", DELEGATED_FILES)
def test_delegated_file_no_inline_filter_sql(relpath: str) -> None:
    """A delegated file must not contain inline canonical filter SQL."""
    path = REPO_ROOT / relpath
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern, desc in BANNED_INLINE_FILTER_PATTERNS:
        for match in re.finditer(pattern, text):
            line_no = text[: match.start()].count("\n") + 1
            violations.append(f"  {relpath}:{line_no} — {desc}\n    match: {match.group(0)!r}")
    assert not violations, (
        "Delegated file contains inline canonical filter SQL "
        "(research-truth-protocol.md § Canonical filter delegation):\n" + "\n".join(violations)
    )


@pytest.mark.parametrize("relpath", DELEGATED_FILES)
def test_delegated_file_no_hardcoded_commission(relpath: str) -> None:
    """A delegated file must not hardcode the pre-Rithmic MNQ commission
    (or any other cost constant that drifted in Q1 2026). Source from
    pipeline.cost_model.COST_SPECS instead.
    """
    path = REPO_ROOT / relpath
    if relpath in EXEMPT_FILES_COMMISSION:
        pytest.skip(f"{relpath} is on the commission exemption list")
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern, desc in BANNED_COMMISSION_PATTERNS:
        for match in re.finditer(pattern, text):
            line_no = text[: match.start()].count("\n") + 1
            violations.append(f"  {relpath}:{line_no} — {desc}\n    match: {match.group(0)!r}")
    assert not violations, (
        "Delegated file hardcodes a commission/friction constant that "
        "should be sourced from pipeline.cost_model.COST_SPECS:\n" + "\n".join(violations)
    )
