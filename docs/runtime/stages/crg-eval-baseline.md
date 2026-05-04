---
task: crg-eval-baseline
mode: IMPLEMENTATION
phase: 1/1
spec: docs/plans/2026-04-30-crg-maximization-v2.md PR-3
created: 2026-04-30
scope_lock:
  - configs/canompx3-crg-eval.yaml
  - scripts/tools/run_crg_eval.py
  - tests/test_tools/test_run_crg_eval.py
  - docs/external/code-review-graph/EVAL-BASELINE-2026-04-30.md
  - docs/external/code-review-graph/EVAL-BASELINE.json
blast_radius: Reads .code-review-graph/graph.db (canonical CRG graph) and upstream code_review_graph.eval.benchmarks modules; writes docs/external/code-review-graph/EVAL-BASELINE.json (regenerable) plus telemetry to .code-review-graph/usage-log.jsonl (gitignored). No writes to pipeline/, trading_app/, gold.db, schema, or canonical sources. Halt: exit 2 if median naive->graph ratio <1.111.
acceptance:
  - schema_validation: configs/canompx3-crg-eval.yaml parses; required keys present
  - dry_run: `python scripts/tools/run_crg_eval.py --dry-run` exits 0
  - real_run: `python scripts/tools/run_crg_eval.py` exits 0 OR exits 2 (halt verdict)
  - tests: `python -m pytest tests/test_tools/test_run_crg_eval.py -v` 100% pass
  - drift: `pipeline/check_drift.py --fast` passes
  - no_fake_numbers: EVAL-BASELINE-2026-04-30.md updated with measured values
    (zero "NOT YET RUN" strings remaining)
  - halt_evaluated: stdout includes either "PASS" or "HALT" verdict
halt_condition:
  threshold: median_naive_to_graph_ratio < 1.111 (i.e., <10% token savings)
  action_on_halt: stop PR-4a; reassess CRG-vs-grep premise
---

# Stage: CRG eval baseline

**Status:** in progress as of 2026-04-30.

## Goal

Close spec hard-non-negotiable §2 ("no fake numbers") by replacing the current
"NOT YET RUN" placeholder in `EVAL-BASELINE-2026-04-30.md` with actual measured
token-savings numbers from the upstream `code_review_graph.eval.benchmarks`
suite.

## Approach

Bypass upstream `code_review_graph.eval.runner.run_eval()` (which clones the
target repo into `evaluate/test_repos/` and rebuilds the graph from scratch).
Call the benchmark functions directly against canompx3's existing
`.code-review-graph/graph.db`.

Why bypass:
- The runner is built for cross-repo benchmarking. Internal "is CRG pulling its
  weight on this codebase?" measurement does not need a clone.
- Cloning needs a public URL or `file://` path support that varies by Git
  version; tests a snapshot rather than the live incrementally-maintained graph.
- Avoids monkey-patching `CONFIGS_DIR` (the v2 plan considered that route, then
  rejected it as fragile to upstream changes).

Three benchmarks selected out of upstream's five:
- `token_efficiency` — headline metric for the halt gate
- `search_quality` — free-text → canonical-symbol MRR
- `impact_accuracy` — diff-aware precision/recall

`flow_completeness` and `build_performance` are skipped: not load-bearing for
the halt decision and add wall-clock cost.

## Files

- `configs/canompx3-crg-eval.yaml` — eval config (matches upstream schema; 3
  test commits with diverse change shapes: doctrine / canonical-param-removal /
  net-new utility; 10 ground-truth search queries).
- `scripts/tools/run_crg_eval.py` — wrapper. Loads config, calls benchmarks
  directly against local graph, writes JSON output, prints PASS/HALT verdict.
  Exit code 2 on halt-condition trigger.
- `tests/test_tools/test_run_crg_eval.py` — schema validation, dry-run path,
  output structure.
- `docs/external/code-review-graph/EVAL-BASELINE-2026-04-30.md` — UPDATE with
  measured numbers from the run; zero "NOT YET RUN" strings remaining.
- `docs/external/code-review-graph/EVAL-BASELINE.json` — regenerable, written
  by the wrapper.

## Hardening / future-proofing

- All target SHAs in `test_commits[]` are verified live via `git show --stat`
  before stage commit; documented in config comments.
- All entry-point + search-query qualified-names verified to exist via direct
  `Grep`/`Read` cross-check before stage commit. `deployable_validated_relation`
  located at canonical definition site (`pipeline/db_contracts.py:75`), not the
  re-export site.
- Wrapper imports CRG modules lazily and falls open on missing benchmarks (logs
  error, records empty result, halt logic continues to run on what landed).
- Schema validator in wrapper enforces required keys present before any
  benchmark invocation — fails loud on upstream signature drift.
- Halt threshold lives at one named constant (`HALT_RATIO_THRESHOLD`); future
  rebalance is one-line.
- `.code-review-graph/graph.db` existence pre-checked; clear error if missing.
