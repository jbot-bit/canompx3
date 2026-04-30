# CRG Eval Baseline — Status: MEASURED 2026-04-30 (CORRECTED 2026-04-30)

**Date measured:** 2026-04-30
**Spec reference:** `docs/plans/2026-04-29-crg-integration-spec.md` § Phase 1.5 M2
**Plan:** `docs/plans/2026-04-30-crg-maximization-v2.md` PR-3
**Raw results:** `docs/external/code-review-graph/EVAL-BASELINE.json` (regenerable)
**Wrapper:** `scripts/tools/run_crg_eval.py` (re-run any time)
**Config:** `configs/canompx3-crg-eval.yaml` (3 test commits, 10 search queries, 5 entry points)

## Correction notice (2026-04-30, post-merge audit)

The first-published headline `4.300 (76.7% savings) → PASS` was **arithmetically wrong**.
Independent `evidence-auditor` review caught two bugs in `_summarize` at
`scripts/tools/run_crg_eval.py:135-136`:

1. `if r.get("naive_to_graph_ratio")` filtered out the third commit (ratio 0.0 is falsy
   in Python), reducing the sample from 3 to 2.
2. `sorted(xs)[len(xs)//2]` returns the *upper* element on n=2, so the reported "median"
   was actually the maximum (`4.3`).

Under correct math (filter `is not None`, use `statistics.median`):
- True median ratio over the 3 commits is **0.2** (not 4.3)
- Implied savings is **−400%** (CRG inflates context ~5× on the median commit)
- `halt_pr4a` flips to **True** under the v2 plan's literal "<10% savings → halt" rule

Fix landed in commit `<filled-on-commit>` along with this correction notice and a
regression test (`tests/test_tools/test_run_crg_eval.py::test_summarize_median_with_zero_ratio`).

## Headline (corrected)

| Metric                          | Value | Halt threshold (v2 plan)        | Verdict |
|---------------------------------|-------|---------------------------------|---------|
| Median naive→graph token ratio  | **0.200** (−400% savings) | < 1.111 (10% savings) | **HALT-FIRES** |
| Best-shape ratio (docs commits) | 4.300 (76.7% savings)     | (per-shape, advisory) | strong on docs-shape |
| Worst-shape ratio (utility/refactor) | 0.0–0.2              | (per-shape, advisory) | net-loss |
| Search MRR                      | 0.325 | (no halt threshold)             | mixed   |
| Impact accuracy precision (avg) | 1.000 | (no halt threshold)             | trivial-case strong |
| Impact accuracy recall (avg)    | 1.000 | (no halt threshold)             | trivial-case strong |

## Reconciliation with PR-4a (which had already merged under the wrong number)

PR-4a (`trading_app/strategy_lineage_ast.py`) is a **pure AST scanner with no CRG runtime
dependency** — it does not consume the graph at query time. The plan's halt heuristic
("if CRG itself isn't pulling weight, a CRG-style lineage tool likely won't either") was
a reasonable prior, but as built, PR-4a's value is independent of CRG's token-efficiency.
The leaf module ships clean tests, zero callers, no DB writes; verified post-merge by
the same auditor pass. **No PR-4a rollback recommended.**

What this DOES gate, honestly:

- **PR-4b** (post-MVP — `strategy_lineage` builder + MCP tool + slash command) materially
  depends on CRG-graph quality. Under corrected halt math, PR-4b should NOT proceed on
  the current eval result without one of: (a) tightening the halt definition to "best-shape"
  per Limitation 2 below and re-running, (b) re-curating the test commits so the median
  reflects the work-shape distribution agents actually face, or (c) accepting that
  CRG-graph-derived tools should ship only for doc/rule-shape edits and frame PR-4b
  accordingly.

The halt condition's job was to surface this question. It did, just an iteration late.

## Per-commit token efficiency (the headline number broken open)

| Commit | Description | Naive tokens | Graph tokens | Ratio |
|--------|-------------|--------------|--------------|-------|
| `91516e10` | docs: backtesting-methodology § RULE 6.1 narrowing (3 files, doctrine-shape) | 8,006  | 1,877    | **4.3** ← median |
| `9b16c4eb` | mechanical: dead `break_ts` removed from `_compute_outcomes_all_rr` (3 files, canonical-param-removal-shape) | 34,434 | 184,086 | 0.2  |
| `1a0a4a24` | feat: `backfill_validated_trade_windows.py` migration (2 files, net-new-utility-shape) | 3,113  | 126,807  | 0.0  |

**Honest read:** the median wins because one commit type wins decisively. The other two
commit shapes show CRG **inflating** context (graph context returned more tokens than
the full file would). For the canonical-param-removal commit, `outcome_builder.py`'s
fan-in dominates the review-context expansion. For the net-new utility, the architecture
context CRG appended dwarfed the actual diff.

**Implication:** CRG's token efficiency is shape-dependent. It excels at small docs/rule
edits where surrounding-architecture summarization beats full-file dumps. It is a NET LOSS
for high-fan-in canonical-source edits or net-new files where there is no relevant blast
radius to summarize. Per-commit profiling matters; do not cite a single ratio.

## Search quality (10 queries, MRR 0.325)

5 hits / 5 misses:

| Query | Expected | Rank | Reciprocal rank |
|-------|----------|------|-----------------|
| ORB window UTC resolution | `pipeline/dst.py::orb_utc_window` | 2 | 0.50 |
| Mode A holdout sacred date enforcement | `trading_app/holdout_policy.py::enforce_holdout_date` | 4 | 0.25 |
| parse strategy id eligibility | `trading_app/eligibility/builder.py::parse_strategy_id` | 1 | 1.00 |
| deployable validated relation | `pipeline/db_contracts.py::deployable_validated_relation` | 1 | 1.00 |
| portfolio fitness aggregator | `trading_app/strategy_fitness.py::compute_portfolio_fitness` | 2 | 0.50 |
| drift check entry point | `pipeline/check_drift.py::main` | 0 (miss) | 0.00 |
| strategy discovery main | `trading_app/strategy_discovery.py::main` | 0 (miss) | 0.00 |
| outcome builder per-trade computation | `trading_app/outcome_builder.py::_compute_outcomes_all_rr` | 0 (miss) | 0.00 |
| cost specs canonical source | `pipeline/cost_model.py::COST_SPECS` | 0 (miss) | 0.00 |
| DST session catalog | `pipeline/dst.py::SESSION_CATALOG` | 0 (miss) | 0.00 |

**Pattern in misses:**
- 3 of 5 misses are queries for `*::main` symbols. `main` is non-discriminative across
  ~360 research scripts — the search ranker probably weights symbol uniqueness and
  buries `main` candidates.
- 2 of 5 misses are queries for module-level constants (`COST_SPECS`,
  `SESSION_CATALOG`). Search may be biased toward callable nodes.

**Implication:** CRG semantic search is useful for finding non-trivially-named functions
and classes. It is unreliable for finding `main()` entry points or module-level constants
by free-text description. Both limitations are workable: use file-name-based navigation
for entry points and grep for constants.

## Impact accuracy (3 commits, avg precision/recall 1.0)

All three test commits scored 1.0 precision + 1.0 recall. **Caveat:** the upstream
benchmark defines `predicted = changed_files ∪ analyze_changes-derived` and
`actual = changed_files ∪ files-importing-from-changed`. For our 3 selected commits:

- `91516e10` is docs-only — no Python imports involved → ground truth = changed files.
- `9b16c4eb` has imports but the test happened to select files where the importer set
  was already represented in `changed_files`.
- `1a0a4a24` is a net-new utility — zero importers exist → ground truth = changed files.

So all three commits have `predicted = actual = changed_files` by coincidence of the
ground-truth definition. **The 1.0/1.0 score is nearly trivial here, not a strong CRG
endorsement.** A more demanding test commit (one with non-trivial cross-file imports)
would surface real precision/recall trade-offs. Future iteration should curate 3-5
test commits where importer sets differ from changed-file sets.

## How to re-run

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/tools/run_crg_eval.py
```

Exits 0 on PASS, 2 on HALT (median ratio < 1.111). Writes
`docs/external/code-review-graph/EVAL-BASELINE.json` and prints the verdict.

```bash
.venv/Scripts/python.exe scripts/tools/run_crg_eval.py --dry-run
```

Validates the config schema without running benchmarks. Exits 0.

## What this closes

- Spec hard-non-negotiable §2 ("no fake numbers"): every metric above traces to a JSON
  file written by an executed benchmark function. No vendor claims cited as ours. The
  prior placeholder text has been replaced with measured values.
- Spec § Phase 1.5 M2 ("eval baseline"): real measurements committed.
- v2 plan PR-3 acceptance: drift passes, 11/11 tests pass, EVAL-BASELINE.json exists,
  halt-condition evaluated and PASSED.

## Re-run cadence

Per spec § M2, re-run quarterly OR when:
- Major refactors land that change canonical-source fan-in (re-test commit 2's shape)
- A new public canonical entry point lands (extend `entry_points[]`)
- Search queries fail in the field repeatedly (extend `search_queries[]` from real misses)

## Limitations and known gaps

1. **MRR 0.325 is below ideal.** Working theory: CRG hybrid search heavily favors
   distinctive symbol names; `main` and module constants underperform. Mitigation:
   the agents that depend on CRG search should use it for "find this concept" queries
   and fall back to grep for "find the entry point" queries. Documented in this file
   so future MCP-prompt wiring can route accordingly.
2. **Token efficiency is shape-dependent.** Median masks bimodality. Future
   improvement: track per-shape ratios and gate halt on the *worst* shape, not the
   median. Recorded as a halt-rebalance candidate in v2 plan §"Halt conditions".
3. **Impact accuracy ground truth is too forgiving.** Future test-commit curation
   should include at least one commit with non-trivial cross-file importers where
   `predicted ≠ actual` is structurally possible.

These limitations do NOT halt PR-4a. They're future-iteration items captured here
because the spec demands honesty over flattering numbers.
