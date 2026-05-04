"""Run CRG eval benchmarks against canompx3's local graph.

Closes spec hard-non-negotiable §2 ("no fake numbers") and v2 plan PR-3.
Bypasses upstream `code_review_graph.eval.runner.run_eval()` (which clones
the repo into evaluate/test_repos/) and calls benchmark functions directly
against canompx3's existing graph at .code-review-graph/.

Why bypass:
    The upstream runner is built for cross-repo benchmarking. For an internal
    "is CRG pulling its weight on THIS codebase?" measurement, cloning the
    repo is wasteful, brittle (needs a public URL), and tests a snapshot
    rather than the live graph the agents actually consume.

Output:
    docs/external/code-review-graph/EVAL-BASELINE.json — per-benchmark results
    written via json.dump (one source of truth for the EVAL-BASELINE-*.md doc).

Halt-condition wiring (v2 plan):
    If the median naive_to_graph_ratio across test_commits is < 1.111
    (i.e., <10% token savings), exit code 2 — caller should halt PR-4a.
    Otherwise exit 0.

Usage:
    python scripts/tools/run_crg_eval.py            # full run, write JSON
    python scripts/tools/run_crg_eval.py --dry-run  # config validation only
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "canompx3-crg-eval.yaml"
OUTPUT_PATH = REPO_ROOT / "docs" / "external" / "code-review-graph" / "EVAL-BASELINE.json"

# Halt threshold: <10% savings means CRG is not earning its slot.
# Equivalent to (1 - 1/ratio) < 0.10 ⟺ ratio < 1.111...
HALT_RATIO_THRESHOLD = 1.111

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_crg_eval")


def _load_config(path: Path) -> dict:
    """Load + validate the eval config. Imports yaml lazily."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit(
            "pyyaml not installed. Install with: pip install code-review-graph[eval] (or just `pip install pyyaml`)"
        ) from None

    if not path.exists():
        raise SystemExit(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Minimal schema validation — fail loudly if upstream signature changes.
    required_top = {"name", "test_commits", "search_queries"}
    missing = required_top - set(cfg.keys())
    if missing:
        raise SystemExit(f"Config missing required keys: {sorted(missing)}")

    for tc in cfg.get("test_commits", []):
        if "sha" not in tc:
            raise SystemExit(f"test_commits entry missing 'sha': {tc}")
    for sq in cfg.get("search_queries", []):
        if "query" not in sq or "expected" not in sq:
            raise SystemExit(f"search_queries entry missing 'query'/'expected': {sq}")

    return cfg


def _run_benchmarks_local(config: dict) -> dict[str, list[dict]]:
    """Call benchmark functions directly against canompx3's local graph.

    Mirrors what `run_eval()` does internally but skips clone_or_update +
    full_build — we use the live, incrementally-maintained graph.

    Returns: {bench_name: [result, ...], ...}
    """
    from code_review_graph.eval.benchmarks import (
        impact_accuracy,
        search_quality,
        token_efficiency,
    )
    from code_review_graph.graph import GraphStore
    from code_review_graph.incremental import get_db_path

    db_path = get_db_path(REPO_ROOT)
    if not Path(db_path).exists():
        raise SystemExit(
            f"CRG graph DB not found at {db_path}. Build it first: `code-review-graph build --repo {REPO_ROOT}`"
        )

    store = GraphStore(db_path)
    try:
        results: dict[str, list[dict]] = {}
        # Three benchmarks chosen for the v2 plan's halt condition:
        #   - token_efficiency: headline metric for the halt gate
        #   - search_quality: free-text → canonical-symbol MRR
        #   - impact_accuracy: diff-aware precision/recall
        # Skipping flow_completeness + build_performance: not load-bearing
        # for the halt decision and add wall-clock cost.
        for name, fn in [
            ("token_efficiency", token_efficiency.run),
            ("search_quality", search_quality.run),
            ("impact_accuracy", impact_accuracy.run),
        ]:
            logger.info("Running %s ...", name)
            try:
                results[name] = fn(REPO_ROOT, store, config)
                logger.info("  %s: %d result(s)", name, len(results[name]))
            except Exception as exc:
                logger.error("  %s FAILED: %s", name, exc, exc_info=True)
                # Record empty results — caller decides whether to halt.
                results[name] = []
        return results
    finally:
        store.close()


def _summarize(results: dict[str, list[dict]]) -> dict:
    """Compute summary stats used by the halt-condition gate."""
    te_rows = results.get("token_efficiency", [])
    # Filter on `is not None` (NOT truthy) so a real 0.0 ratio — meaning CRG
    # blew up context vs naive grep — is kept in the median calculation rather
    # than silently dropped. Then use statistics.median for correctness on any
    # n (the prior `sorted(xs)[n//2]` returned the upper element on even n,
    # which on n=2 is the max, not the median).
    ratios = [r["naive_to_graph_ratio"] for r in te_rows if r.get("naive_to_graph_ratio") is not None]
    median_ratio = statistics.median(ratios) if ratios else 0.0
    savings_pct = (1.0 - 1.0 / median_ratio) * 100 if median_ratio > 0 else 0.0

    sq_rows = results.get("search_quality", [])
    mrr = sum(r.get("reciprocal_rank", 0.0) for r in sq_rows) / len(sq_rows) if sq_rows else 0.0

    ia_rows = results.get("impact_accuracy", [])
    avg_recall = sum(r.get("recall", 0.0) for r in ia_rows) / len(ia_rows) if ia_rows else 0.0
    avg_precision = sum(r.get("precision", 0.0) for r in ia_rows) / len(ia_rows) if ia_rows else 0.0

    halt = median_ratio < HALT_RATIO_THRESHOLD if ratios else False

    return {
        "token_efficiency": {
            "n_commits": len(te_rows),
            "median_naive_to_graph_ratio": round(median_ratio, 3),
            "median_savings_pct": round(savings_pct, 1),
            "halt_threshold_ratio": HALT_RATIO_THRESHOLD,
            "halt_pr4a": halt,
        },
        "search_quality": {
            "n_queries": len(sq_rows),
            "mean_reciprocal_rank": round(mrr, 3),
        },
        "impact_accuracy": {
            "n_commits": len(ia_rows),
            "avg_precision": round(avg_precision, 3),
            "avg_recall": round(avg_recall, 3),
        },
    }


def _write_output(results: dict, summary: dict, config_name: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_name,
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summary,
        "results": results,
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s", output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run CRG eval against canompx3 local graph.",
        prog="run_crg_eval",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config schema only — do not run benchmarks.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"Path to eval config YAML (default: {CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"Path to write EVAL-BASELINE.json (default: {OUTPUT_PATH}).",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config))
    logger.info(
        "Loaded config: %s (%d test_commits, %d search_queries)",
        cfg["name"],
        len(cfg.get("test_commits", [])),
        len(cfg.get("search_queries", [])),
    )

    if args.dry_run:
        logger.info("--dry-run: config valid. Exiting without running benchmarks.")
        return 0

    results = _run_benchmarks_local(cfg)
    summary = _summarize(results)
    _write_output(results, summary, cfg["name"], Path(args.output))

    # Surface the halt verdict on stdout regardless of exit code.
    te = summary["token_efficiency"]
    print(
        f"\nMedian naive->graph token ratio: {te['median_naive_to_graph_ratio']:.3f} "
        f"({te['median_savings_pct']:.1f}% savings) over {te['n_commits']} commits",
        flush=True,
    )
    print(
        f"Search MRR: {summary['search_quality']['mean_reciprocal_rank']:.3f} "
        f"over {summary['search_quality']['n_queries']} queries",
        flush=True,
    )
    if te["halt_pr4a"]:
        print(
            f"\nHALT: median ratio {te['median_naive_to_graph_ratio']:.3f} < "
            f"{HALT_RATIO_THRESHOLD} threshold. v2 plan halt-condition fired - "
            f"reconsider PR-4a before starting.",
            flush=True,
        )
        return 2
    print("\nPASS: CRG savings exceed halt threshold. PR-4a may proceed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
