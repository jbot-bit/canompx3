# CRG Eval Baseline — Status: NOT YET RUN (config required)

**Date investigated:** 2026-04-30
**Spec reference:** `docs/plans/2026-04-29-crg-integration-spec.md` § Phase 1.5 M2
**Outcome:** Cannot run upstream `code-review-graph eval` without authoring a benchmark
config YAML for this repo. Recording the honest state per the "no silences" rule.

## What was attempted

```bash
cd C:/Users/joshd/canompx3
uvx code-review-graph eval --all --repo canompx3 --output-dir /tmp/crg-eval-2026-04-30
```

**Result:** `FileNotFoundError: code_review_graph/eval/configs/canompx3.yaml`

## Why it failed

`code-review-graph eval` (v2.3.2) loads benchmark configs from
`<package_root>/eval/configs/<name>.yaml`. Configs ship for reference repos
(`fastapi.yaml`, `flask.yaml`, `nextjs.yaml`, `express.yaml`, `httpx.yaml`,
`gin.yaml`) but not for arbitrary user repos. The `--repo` flag accepts the
config name, not a path.

The schema (verified by reading `fastapi.yaml`) requires:

- `name`, `url`, `commit`, `language`, `size_category`
- `test_commits[]` — known commits with expected `changed_files` count
- `entry_points[]` — qualified names of architectural entry points
- `search_queries[]` — `{query, expected}` pairs of ground-truth canonical hits

For canompx3 this means we'd need to curate:
- 2-3 representative commits with verified diff scope (token-efficiency benchmark)
- A short list of entry points (e.g., `pipeline/check_drift.py::main`,
  `trading_app/eligibility/builder.py::parse_strategy_id`)
- 5-10 search queries with ground-truth canonical matches

This is its own small calibration project. Not a one-line eval run.

## What we DO know (without the benchmark config)

Quick honest measurements possible without a benchmark suite:

### Build performance (verified 2026-04-30 from prior calibration)
- Initial build: ~16s (per `code_review_graph_calibration.md`)
- Incremental update via post-edit hook: <2s typical
- Graph DB on disk: ~5MB

### Search quality (single-query smoke test, 2026-04-30)
- Query: `"E2 break bar lookahead contamination"`
- search_mode: `keyword` (NOT hybrid; embeddings stored but not consumed —
  see `EMBEDDINGS.md`)
- Hits: 0
- Expected canonical: `check_e2_lookahead_research_contamination`
- **Verdict: known regression. See EMBEDDINGS.md for working theories.**

### Token efficiency (vendor claim, NOT measured here)
- Vendor publishes 6.8× / 49× token reduction. Per spec § non-negotiable rule
  2 ("no fake numbers"), this is THEIR number until we benchmark. Not cited
  as ours.

### Graph completeness
- 13,988 nodes / 152,215 edges / 1,056 files (canonical, post-Stage-A merge)
- 12,693 nodes embedded (stored — but unused at query time, see above)
- 4 node kinds populated: File, Class, Function, Test
- 6 edge kinds populated: CALLS, CONTAINS, IMPORTS_FROM, INHERITS, REFERENCES,
  TESTED_BY

## What this baseline DOES establish

Even without the upstream eval suite, this file pins three facts:

1. **Search quality has regressed** vs. the 2026-04-29 baseline in
   `EMBEDDINGS.md` (3/5 canonical → 0/1 canonical for the smoke query).
2. **Embeddings index exists but is unused** — semantic search runs in
   keyword mode at the MCP boundary.
3. **Build/incremental performance** is fast enough that it's not on the
   critical path; the bottleneck is search quality and tool ergonomics.

## Follow-ups (NOT in scope this PR)

1. **Author `eval/configs/canompx3.yaml`** following the shipped reference
   schema. Curate 5-10 ground-truth queries from the real research log
   (e.g., the E2 LA contamination registry, the GARCH silent-NULL incident).
2. **Diagnose the keyword-only fallback** (per EMBEDDINGS.md Working Theories):
   verify if `code-review-graph[embeddings]` extras are present in the uvx
   install path; if not, either install via uvx args or switch MCP back to
   the venv binary.
3. **Re-run eval** with both fixes in place. Commit JSON results.
4. **Re-run eval quarterly** per spec § M2.

## Refresh policy

Update this file (or replace with the JSON output) once the upstream
benchmark config exists and runs cleanly. Until then this markdown stub
IS the baseline, recording the honest state.
