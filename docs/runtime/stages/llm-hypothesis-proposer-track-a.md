---
task: "Add LLM hypothesis proposer (Track A of i-stg-we-are-ticklish-cookie v2 plan)"
mode: IMPLEMENTATION
scope_lock:
  - scripts/research/llm_hypothesis_proposer.py
  - scripts/research/lhp/__init__.py
  - scripts/research/lhp/literature_index.py
  - scripts/research/lhp/adjacency.py
  - scripts/research/lhp/llm_client.py
  - scripts/research/lhp/yaml_emitter.py
  - scripts/research/lhp/static_checks.py
  - docs/prompts/hypothesis-proposer-system.md
  - docs/prompts/hypothesis-proposer-fewshot.md
  - tests/test_llm_hypothesis_proposer.py
  - tests/fixtures/lhp/good_yaml_1.yaml
  - tests/fixtures/lhp/good_yaml_2.yaml
  - tests/fixtures/lhp/bad_banned_feature.yaml
  - tests/fixtures/lhp/bad_wrong_holdout.yaml
  - tests/fixtures/lhp/bad_fabricated_citation.yaml
  - tests/fixtures/lhp/bad_minbtl_exceeded.yaml
  - .claude/skills/propose-hypothesis/SKILL.md
  - docs/runtime/stages/llm-hypothesis-proposer-track-a.md
---

## Blast Radius

- All scope files are NEW — zero callers, zero existing importers, zero existing tests touched.
- **Reads** (read-only, never modified):
  - `docs/institutional/literature/*.md` — 22 literature extract files for corpus indexing.
  - `docs/audit/hypotheses/*.yaml` — sampled as fewshot exemplars in the prompt; never written.
  - `gold.db` via `duckdb.connect(read_only=True)` on `validated_setups` only (adjacency context).
  - `.env` (or `os.environ`) for `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`.
  - `trading_app.hypothesis_loader` — canonical schema validator (delegated, not re-encoded).
  - `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` — canonical Mode A constant.
  - `trading_app.config.ALL_FILTERS` — canonical filter registry.
  - `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` — canonical instrument list.
  - `pipeline.dst.SESSION_CATALOG` — canonical session list.
  - `pipeline.paths.GOLD_DB_PATH` — canonical DB path.
- **Writes**:
  - `docs/audit/hypotheses/drafts/YYYY-MM-DD-llm-<slug>.yaml` — NEW files only, inside the
    `drafts/` subdirectory. `hypothesis_loader.find_hypothesis_file_by_sha()` uses
    `directory.glob("*.yaml")` non-recursively (line 139, verified), so files under
    `drafts/` are invisible to the loader. Human moves the file up one level to publish.
    (Earlier `.draft.yaml` suffix design was scrapped after a test caught that
    `Path.glob("*.yaml")` matches `a.draft.yaml`.)
  - On fatal-check failure: `docs/audit/hypotheses/drafts/YYYY-MM-DD-llm-<slug>.rejected.txt`
    (non-YAML extension, also invisible to loader; for audit only).
- **Does NOT modify**: pipeline/, trading_app/, research/, any drift check, any existing
  hypothesis YAML, gold.db schema, lane_allocation.json, live execution, no canonical
  source-of-truth file.
- **Network**: one OpenRouter or Anthropic API call per `--dry-run`-less invocation. Gated by
  `--cost-ceiling` (default $0.50). `--dry-run` makes zero network calls.
- **Failure isolation**: any exception in this module cannot propagate to existing pipelines
  because nothing existing imports from `scripts/research/lhp/`. The module is leaf.
- **Dependencies added**: none new at the package level. Uses existing `duckdb`, `yaml`,
  `httpx`/`requests` (already present), no new wheels required.

## Acceptance criteria

- [ ] All new tests pass (show output).
- [ ] `python pipeline/check_drift.py` passes from the worktree.
- [ ] `python scripts/research/llm_hypothesis_proposer.py --dry-run --slug fixture-1` produces
      a valid `.draft.yaml` from fixture without calling the LLM (exit 0).
- [ ] One real LLM invocation produces a YAML that survives all static checks (deferred to
      live-run step, optional for stage-close).
- [ ] No file in `pipeline/`, `trading_app/`, `research/` modified.
- [ ] Canonical delegation: `static_checks` calls `enforce_holdout_date`,
      `enforce_minbtl_bound`, and `load_hypothesis_metadata` — does not re-encode them.

## Notes on canonical delegation

Per `.claude/rules/institutional-rigor.md` §4 ("delegate to canonical sources, never
re-encode"), the static-check module wraps existing canonical authority:

| Check | Canonical source called |
|---|---|
| Schema parse | `trading_app.hypothesis_loader.load_hypothesis_metadata` |
| Holdout date | `trading_app.holdout_policy.enforce_holdout_date` |
| MinBTL budget | `trading_app.hypothesis_loader.enforce_minbtl_bound` |
| Filter registered | already enforced by `load_hypothesis_metadata` via `ALL_FILTERS` lookup |
| Instrument active | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| Session valid | `pipeline.dst.SESSION_CATALOG` |
| DB path | `pipeline.paths.GOLD_DB_PATH` |

Only `check_banned_features` (E2 look-ahead) and `check_citations_exist` (literature corpus
membership) add new logic not already present in canonical sources — and even
`check_banned_features` delegates to `trading_app.config.E2_EXCLUDED_FILTER_PREFIXES` /
`E2_EXCLUDED_FILTER_SUBSTRINGS` if those exist (verified at implementation time).
