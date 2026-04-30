"""AST-based column-reference scanner for Python source files.

Pure leaf module — no DB, no I/O beyond reading the supplied path, no callers
in `pipeline/` or `trading_app/` runtime. Used by `strategy_lineage` (PR-4b)
to build `REFERENCES_COLUMN` edges, and standalone as a contamination-detection
helper (e.g., catching look-ahead by scanning E2-cohort scripts for
`break_dir`/`break_bar*`/`rel_vol_<UPPER>`).

Algorithm — hybrid regex + AST, three passes:

  1. Regex pre-filter compiled from the column allowlist. If the file contains
     no allowlisted token at all, skip AST parsing entirely (cheap fast-path).
  2. `ast.walk` over the parsed tree:
       - `Subscript` with string-Constant slice → `df["col"]` hit.
       - `Attribute` whose `.attr` is in the allowlist → `df.col` hit.
       - `Constant` of `str` type → run a per-string regex to catch SQL
         fragments and assigned column-name literals.
     All three are labeled `evidence="ast_literal"`.
  3. Regex fallback: if `ast.parse` raises `SyntaxError` (malformed source) but
     the pre-filter matched, emit `evidence="regex_fallback"` rows at the line
     numbers where the regex matched. Per institutional rigor, weaker evidence
     is labeled and surfaced — never silently merged with strong evidence.

Reference: docs/plans/2026-04-30-crg-maximization-v2.md § PR-4a.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from pathlib import Path
from typing import Literal, NamedTuple

logger = logging.getLogger(__name__)

Evidence = Literal["ast_literal", "regex_fallback"]


class ColumnRef(NamedTuple):
    file: Path
    lineno: int
    column: str
    evidence: Evidence


def _compile_allowlist_regex(column_allowlist: set[str]) -> re.Pattern[str] | None:
    if not column_allowlist:
        return None
    # Sort longest-first so e.g. `atr_20_smooth` (if allowlisted) wins over
    # `atr_20`. \b treats underscore as a word char, so partial matches like
    # `atr_20` inside `atr_20_smooth` are already excluded — sorting is
    # belt-and-braces for future allowlists with prefix overlaps.
    alternation = "|".join(re.escape(c) for c in sorted(column_allowlist, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b")


def _scan_string_literal(
    file: Path,
    lineno: int,
    text: str,
    pattern: re.Pattern[str],
) -> list[ColumnRef]:
    """Find every allowlisted token inside a single string literal.

    SQL fragments often pack multiple columns ("SELECT atr_20, rsi_14 FROM"),
    so we yield one ColumnRef per match. lineno is the line of the literal in
    the source file (multi-line strings keep the opening line per `ast`).
    """
    return [
        ColumnRef(file=file, lineno=lineno, column=m.group(0), evidence="ast_literal") for m in pattern.finditer(text)
    ]


def _scan_ast(
    file: Path,
    tree: ast.AST,
    column_allowlist: set[str],
    pattern: re.Pattern[str],
) -> list[ColumnRef]:
    refs: list[ColumnRef] = []
    seen: set[tuple[int, str, Evidence]] = set()

    def _emit(lineno: int, column: str, evidence: Evidence) -> None:
        key = (lineno, column, evidence)
        if key in seen:
            return
        seen.add(key)
        refs.append(ColumnRef(file=file, lineno=lineno, column=column, evidence=evidence))

    for node in ast.walk(tree):
        # Pattern 1: `df["col"]` → Subscript with str-Constant slice.
        # Python 3.9+ stores the constant directly on `node.slice` (no Index wrapper).
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                col = sl.value
                if col in column_allowlist:
                    _emit(node.lineno, col, "ast_literal")
            continue

        # Pattern 2: `df.col` → Attribute whose .attr matches allowlist.
        if isinstance(node, ast.Attribute):
            if node.attr in column_allowlist:
                _emit(node.lineno, node.attr, "ast_literal")
            continue

        # Pattern 3: any string Constant (assigned, SQL fragment, f-string part).
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for ref in _scan_string_literal(file, node.lineno, node.value, pattern):
                _emit(ref.lineno, ref.column, ref.evidence)

    return refs


def _regex_fallback(
    file: Path,
    source: str,
    pattern: re.Pattern[str],
) -> list[ColumnRef]:
    """Emit ColumnRefs from line-by-line regex when AST parse failed."""
    refs: list[ColumnRef] = []
    seen: set[tuple[int, str]] = set()
    for lineno, line in enumerate(source.splitlines(), start=1):
        for m in pattern.finditer(line):
            col = m.group(0)
            key = (lineno, col)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ColumnRef(file=file, lineno=lineno, column=col, evidence="regex_fallback"))
    return refs


def scan_python_for_column_refs(
    path: Path,
    column_allowlist: set[str],
) -> list[ColumnRef]:
    """Scan a single Python file for references to canonical column names.

    Args:
        path: Source file. Read as UTF-8; non-decodable / missing files yield [].
        column_allowlist: Canonical column names to search for. Empty set → [].

    Returns:
        List of ColumnRef hits, deduped per (lineno, column, evidence).
        Order: AST walk order (top-to-bottom, child-before-sibling).

    Never raises on bad input — malformed Python returns [] (with regex
    fallback if the pre-filter matched), missing/binary files return [].
    """
    pattern = _compile_allowlist_regex(column_allowlist)
    if pattern is None:
        return []

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("strategy_lineage_ast: cannot read %s: %s", path, exc)
        return []

    if not pattern.search(source):
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        logger.warning(
            "strategy_lineage_ast: SyntaxError in %s line %s: %s; falling back to regex",
            path,
            exc.lineno,
            exc.msg,
        )
        return _regex_fallback(path, source, pattern)

    return _scan_ast(path, tree, column_allowlist, pattern)


# --- CLI ---------------------------------------------------------------------

# Smoke-test default for `--scan`. Real callers (PR-4b) supply the allowlist
# from `DESCRIBE daily_features`. Kept minimal to avoid embedding a stale
# schema snapshot in this leaf module.
_CLI_DEFAULT_ALLOWLIST: frozenset[str] = frozenset({"atr_20", "rsi_14", "break_dir", "break_bar_high", "break_bar_low"})


def _iter_python_files(scan_root: Path) -> list[Path]:
    if scan_root.is_file():
        return [scan_root] if scan_root.suffix == ".py" else []
    return sorted(scan_root.rglob("*.py"))


def _load_allowlist(allowlist_file: Path | None) -> set[str]:
    if allowlist_file is None:
        return set(_CLI_DEFAULT_ALLOWLIST)
    raw = allowlist_file.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in raw if line.strip() and not line.strip().startswith("#")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trading_app.strategy_lineage_ast",
        description="Scan Python files for canonical column references (AST + regex fallback).",
    )
    parser.add_argument("--scan", type=Path, required=True, help="File or directory to scan.")
    parser.add_argument(
        "--allowlist-file",
        type=Path,
        default=None,
        help="Optional newline-separated column names. Defaults to a small built-in set.",
    )
    parser.add_argument(
        "--show-fallback-only",
        action="store_true",
        help="Only print regex_fallback hits (useful for triage of malformed sources).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    allowlist = _load_allowlist(args.allowlist_file)
    if not allowlist:
        print("strategy_lineage_ast: empty allowlist; nothing to scan.", file=sys.stderr)
        return 0

    total = 0
    for py in _iter_python_files(args.scan):
        refs = scan_python_for_column_refs(py, allowlist)
        if args.show_fallback_only:
            refs = [r for r in refs if r.evidence == "regex_fallback"]
        for r in refs:
            print(f"{r.file}:{r.lineno}\t{r.column}\t{r.evidence}")
            total += 1

    print(f"# strategy_lineage_ast: {total} column references in {args.scan}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
