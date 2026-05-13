#!/usr/bin/env python3
"""LHP weekly cron runner.

Scans the committed hypothesis corpus, maintains a dedup index, and
optionally checks portfolio fitness for a transition trigger.

Trigger logic (--check-fitness-transition)
------------------------------------------
- Calls canonical ``trading_app.strategy_fitness.compute_portfolio_fitness``
  for every instrument in ``pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS``.
- Aggregates ``decay + watch + stale`` across instruments.
- Compares against the cached snapshot at ``docs/runtime/lhp_fitness_snapshot.json``.
- Trigger fires when the aggregate count rose by ≥1 vs prior snapshot.
- First-run guard: when no prior snapshot exists, writes the snapshot but
  reports trigger ``first_run`` (never ``fitness_transition``). Prevents
  the cron from firing every Sunday until the snapshot stabilises.

Exit codes
----------
0  dedup index written; fitness check (if requested) completed
1  IO error scanning corpus or writing index/snapshot
2  Fitness computation failed (database unreachable, schema drift, etc.)

Canonical paths
---------------
- Hypotheses corpus: docs/audit/hypotheses/*.yaml
- Dedup index:       docs/runtime/lhp_dedup_index.json
- Fitness snapshot:  docs/runtime/lhp_fitness_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HYPOTHESES_DIR = _REPO_ROOT / "docs" / "audit" / "hypotheses"
DEDUP_INDEX_PATH = _REPO_ROOT / "docs" / "runtime" / "lhp_dedup_index.json"
FITNESS_SNAPSHOT_PATH = _REPO_ROOT / "docs" / "runtime" / "lhp_fitness_snapshot.json"


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


def _compute_fitness_snapshot(as_of: date) -> dict[str, Any]:
    """Compute current decay+watch+stale aggregate across ACTIVE_ORB_INSTRUMENTS.

    Delegates to canonical ``compute_portfolio_fitness`` (read-only on
    gold.db). Returns a snapshot dict ready to be persisted.
    """
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from trading_app.strategy_fitness import compute_portfolio_fitness

    by_instrument: dict[str, dict[str, int]] = {}
    decay_watch_count = 0
    for instr in ACTIVE_ORB_INSTRUMENTS:
        report = compute_portfolio_fitness(instrument=instr, as_of_date=as_of)
        summary = {k: int(v) for k, v in report.summary.items()}
        by_instrument[instr] = summary
        decay_watch_count += summary.get("decay", 0) + summary.get("watch", 0) + summary.get("stale", 0)

    return {
        "as_of": as_of.isoformat(),
        "decay_watch_count": decay_watch_count,
        "by_instrument": by_instrument,
    }


def _load_prior_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARN: cannot read prior snapshot {path}: {exc}", file=sys.stderr)
        return None


def _determine_trigger(prior: dict[str, Any] | None, current: dict[str, Any]) -> str:
    """Classify the trigger reason from prior+current snapshots.

    Returns one of: ``first_run`` | ``fitness_transition`` | ``no_change``.
    First-run guard prevents firing on bootstrap when no prior exists.
    """
    if prior is None or "decay_watch_count" not in prior:
        return "first_run"
    prior_count = int(prior["decay_watch_count"])
    if current["decay_watch_count"] - prior_count >= 1:
        return "fitness_transition"
    return "no_change"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lhp_weekly",
        description="LHP weekly cron — scan hypotheses, maintain dedup index, optionally trigger LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip any LLM invocation. Refresh the dedup index and (if requested) the fitness snapshot, then exit.",
    )
    parser.add_argument(
        "--check-fitness-transition",
        action="store_true",
        help="Compute portfolio fitness for ACTIVE_ORB_INSTRUMENTS and detect decay+watch+stale transitions vs the prior snapshot.",
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

    trigger = "no_change"
    if args.check_fitness_transition:
        try:
            current = _compute_fitness_snapshot(as_of=date.today())
        except Exception as exc:  # noqa: BLE001 — surface canonical-fitness failure with detail
            print(f"FATAL: fitness computation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        prior = _load_prior_snapshot(FITNESS_SNAPSHOT_PATH)
        trigger = _determine_trigger(prior, current)
        try:
            FITNESS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            FITNESS_SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"FATAL: cannot write {FITNESS_SNAPSHOT_PATH}: {exc}", file=sys.stderr)
            return 1
        print(f"Snapshot written: decay_watch_count={current['decay_watch_count']}")
        print(f"  output: {FITNESS_SNAPSHOT_PATH.relative_to(_REPO_ROOT).as_posix()}")
        print(f"Trigger: {trigger}")

    if args.dry_run:
        if args.check_fitness_transition and trigger == "fitness_transition":
            print("Mode: --dry-run — trigger would fire but LLM call suppressed.")
        else:
            print("Mode: --dry-run (no LLM call).")
        return 0

    if trigger == "fitness_transition":
        print("LIVE trigger fired (fitness_transition). LLM proposer wiring deferred to follow-up plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
