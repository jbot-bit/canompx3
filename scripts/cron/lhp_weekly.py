#!/usr/bin/env python3
"""LHP weekly cron runner — foundation (Stage 2a, Nugget 2a).

Scans the committed hypothesis corpus and maintains a dedup index. This
stage wires the runner infrastructure; the live trigger (fitness transition)
+ GHA schedule land in Stage 2b. No LLM call happens in --dry-run.

Exit codes
----------
0  dedup index written, no live trigger requested (or --dry-run)
1  IO error scanning hypothesis dir or writing index

Canonical paths
---------------
- Hypotheses corpus: docs/audit/hypotheses/*.yaml
- Dedup index: docs/runtime/lhp_dedup_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESES_DIR = _REPO_ROOT / "docs" / "audit" / "hypotheses"
DEDUP_INDEX_PATH = _REPO_ROOT / "docs" / "runtime" / "lhp_dedup_index.json"


def _scan_hypothesis_files(corpus_dir: Path) -> list[Path]:
    if not corpus_dir.is_dir():
        return []
    return sorted(p for p in corpus_dir.glob("*.yaml") if p.is_file())


def _build_dedup_index(files: list[Path]) -> dict[str, dict[str, str]]:
    """Build a {relative_path: {"size": str}} index keyed by repo-relative path.

    Initial key form per plan § "initial key = file path; full content-hash
    deferred". Sizes are stored as strings so the JSON round-trips cleanly
    across platforms without integer-overflow concerns.
    """
    index: dict[str, dict[str, str]] = {}
    for p in files:
        rel = p.relative_to(_REPO_ROOT).as_posix()
        try:
            size = p.stat().st_size
        except OSError as exc:
            print(f"WARN: cannot stat {p}: {exc}", file=sys.stderr)
            continue
        index[rel] = {"size": str(size)}
    return index


def _write_index(index: dict[str, dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lhp_weekly",
        description="LHP weekly cron — scan hypotheses, maintain dedup index, optionally trigger LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip any LLM invocation. Only refresh the dedup index. Stage 2a always runs in dry-run effect.",
    )
    args = parser.parse_args(argv)

    try:
        files = _scan_hypothesis_files(HYPOTHESES_DIR)
    except OSError as exc:
        print(f"FATAL: cannot scan {HYPOTHESES_DIR}: {exc}", file=sys.stderr)
        return 1

    index = _build_dedup_index(files)
    try:
        _write_index(index, DEDUP_INDEX_PATH)
    except OSError as exc:
        print(f"FATAL: cannot write {DEDUP_INDEX_PATH}: {exc}", file=sys.stderr)
        return 1

    print(f"Dedup index: {len(index)} hypotheses indexed")
    print(f"  source: {HYPOTHESES_DIR.relative_to(_REPO_ROOT).as_posix()}")
    print(f"  output: {DEDUP_INDEX_PATH.relative_to(_REPO_ROOT).as_posix()}")
    if args.dry_run:
        print("Mode: --dry-run (no LLM call). Stage 2b will add live triggers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
