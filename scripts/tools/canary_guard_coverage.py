#!/usr/bin/env python3
"""Canary guard-coverage meta static-scanner (deliverable #3).

Answers the structural half of "can canompx3 reject fake edge?": WHICH research
scan scripts would INCORRECTLY PASS a fake edge because they never invoke a
canonical guard?

Tier-1 (``scripts/tests/canary_suite.py``) proves each guard FUNCTION fires.
But a guard that works and is never CALLED is still a hole — a scan that reads
``orb_outcomes`` / ``daily_features``, applies a filter (or uses the E2 entry
model) inline, and never delegates to ``filter_signal`` / ``session_guard`` /
``enforce_holdout_date`` can launder fake edge straight through. This scanner is
the static bridge between "the guard works" and "the guard is used".

Detection (AST-based for precision; grep is too noisy on comments/strings):

    A research module is FLAGGED when ALL of:
      1. it READS a canonical layer — an SQL string mentions ``orb_outcomes``
         or ``daily_features`` (these are the discovery-truth layers), AND
      2. it APPLIES A FILTER or USES E2 — references ``filter_type`` /
         ``ALL_FILTERS`` / ``.matches_df`` / ``matches_row``, OR an
         ``entry_model == 'E2'`` literal, AND
      3. it NEVER references a canonical guard — none of
         ``filter_signal`` / ``session_guard`` / ``is_feature_safe`` /
         ``enforce_holdout_date`` / ``is_e2_lookahead_filter`` /
         ``HOLDOUT_SACRED_FROM`` / ``t0_correlation`` /
         ``_valid_session_features`` appears, AND
      4. it carries no opt-out marker ``# canary-guard-coverage: cleared``.

Such a script COULD pass a fake edge because the guard is never invoked.

Mirrors ``pipeline.check_drift.check_e2_lookahead_research_contamination``: the
drift wrapper ``check_research_scans_call_guards`` produces one violation string
per flagged file, honours the ``# canary-guard-coverage: cleared`` marker, and
reads ``PROJECT_ROOT`` (monkeypatchable in tests via the ``fake_research_root``
fixture).

Scope dirs: ``research/`` and ``scripts/research/`` (the scan layer). The
``archive/`` subtree is skipped (frozen scans, not retro-edited).

Usage
-----
    python scripts/tools/canary_guard_coverage.py            # print the list
    python scripts/tools/canary_guard_coverage.py --json     # machine-readable
    from scripts.tools.canary_guard_coverage import scan_guard_coverage
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Canonical discovery-truth layers (RESEARCH_RULES). A scan that reads these and
# filters/uses-E2 must route through a guard.
_CANONICAL_LAYERS = ("orb_outcomes", "daily_features")

# Markers that the script applies a filter or uses the E2 entry model.
_FILTER_TOKENS = ("filter_type", "ALL_FILTERS", ".matches_df", "matches_row")
# The E2 entry model is matched as a STRING CONSTANT whose value is exactly
# "E2" (the AST strips the surrounding quotes, so the constant value is the bare
# two-character string), AND only when the source establishes an entry-model
# context ("entry_model" present) — mirrors check_e2_lookahead's
# "entry_model in content" gate. This avoids flagging an "E2" that appears as,
# say, a column suffix unrelated to the entry model.
_E2_VALUE = "E2"

# Any of these references means the script delegates to a canonical guard.
_GUARD_TOKENS = (
    "filter_signal",
    "session_guard",
    "is_feature_safe",
    "enforce_holdout_date",
    "is_e2_lookahead_filter",
    "_e2_look_ahead_reason",
    "HOLDOUT_SACRED_FROM",
    "t0_correlation",
    "_valid_session_features",
    "_overnight_lookhead_clean",
)

# Opt-out marker (case-insensitive). Mirrors e2-lookahead-policy: cleared.
_CLEARED_MARKER = "# canary-guard-coverage: cleared"

_SCAN_DIRS = ("research", "scripts/research")


def _string_literals(tree: ast.AST) -> list[str]:
    """Return every string-constant value in the module (SQL lives in these).

    AST-based so we only inspect REAL string literals — a ``filter_type``
    mentioned in a code comment is not a string constant and is excluded from
    the canonical-layer test (comments are stripped by the parser). Identifier
    tokens (``filter_type`` as a kwarg / attribute) are matched separately
    against source text in :func:`_flag_file`.
    """
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _reads_canonical_layer(string_literals: list[str]) -> bool:
    """True iff any SQL string literal names a canonical discovery layer."""
    return any(layer in lit for lit in string_literals for layer in _CANONICAL_LAYERS)


def _applies_filter_or_e2(source: str, string_literals: list[str]) -> bool:
    """True iff the script applies a filter or uses an E2 entry model.

    Filter tokens are identifiers/attributes — matched against source text.
    The E2 entry model is matched as a STRING CONSTANT whose value is exactly
    ``"E2"`` AND only when the source establishes an entry-model context
    (``entry_model`` present). The AST strips quotes, so the constant value is
    the bare ``E2`` — checking for the quoted ``'E2'`` would never match. The
    entry-model gate mirrors ``check_e2_lookahead``'s ``"entry_model" in
    content`` and avoids flagging an unrelated ``E2`` (e.g. a column suffix).
    """
    if any(tok in source for tok in _FILTER_TOKENS):
        return True
    if "entry_model" in source and any(s == _E2_VALUE for s in string_literals):
        return True
    return False


def _references_guard(source: str) -> bool:
    """True iff the script references any canonical guard token."""
    return any(tok in source for tok in _GUARD_TOKENS)


def _has_cleared_marker(source: str) -> bool:
    """True iff the script carries the opt-out marker (case-insensitive)."""
    low = source.lower()
    return _CLEARED_MARKER.lower() in low


def _flag_file(py_file: Path) -> bool:
    """Return True iff ``py_file`` should be flagged as guard-bypassing.

    Returns False on any read/parse failure (fail-open — a file we cannot read
    is never flagged, mirroring the E2 check).
    """
    try:
        source = py_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if _has_cleared_marker(source):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    literals = _string_literals(tree)
    if not _reads_canonical_layer(literals):
        return False
    if not _applies_filter_or_e2(source, literals):
        return False
    if _references_guard(source):
        return False
    return True


def scan_guard_coverage(project_root: Path | None = None) -> list[str]:
    """Return repo-relative paths of research scans that bypass every guard.

    Parameters
    ----------
    project_root
        Repo root to scan under. Defaults to module ``PROJECT_ROOT``. The drift
        wrapper passes the (monkeypatchable) ``check_drift.PROJECT_ROOT`` so
        tests can point it at a ``fake_research_root``.

    Returns
    -------
    list[str]
        Sorted repo-relative POSIX paths of flagged files. Empty == every
        canonical-reading, filtering/E2 scan delegates to a guard.
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    flagged: list[str] = []
    for scan_dir in _SCAN_DIRS:
        base = root / scan_dir
        if not base.is_dir():
            continue
        for py_file in sorted(base.rglob("*.py")):
            try:
                rel_to_base = py_file.relative_to(base)
            except ValueError:
                continue
            if rel_to_base.parts and rel_to_base.parts[0] == "archive":
                continue
            if _flag_file(py_file):
                flagged.append(py_file.relative_to(root).as_posix())
    return sorted(flagged)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    flagged = scan_guard_coverage()
    if args.json:
        print(json.dumps(flagged, indent=2))
    else:
        if not flagged:
            print("No guard-bypassing research scans found — every canonical-")
            print("reading, filtering/E2 scan delegates to a canonical guard.")
        else:
            print(
                f"{len(flagged)} research scan(s) read a canonical layer + "
                f"filter/use-E2 but reference NO canonical guard:\n"
            )
            for f in flagged:
                print(f"  {f}")
            print(
                "\nEach COULD pass a fake edge (the guard is never invoked). "
                "Route through filter_signal / session_guard / "
                "enforce_holdout_date, or add '# canary-guard-coverage: cleared' "
                "after manual verification."
            )
    return 0 if not flagged else 1


if __name__ == "__main__":
    raise SystemExit(main())
