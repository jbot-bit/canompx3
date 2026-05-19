# Stage — LHP citation grounding (Improvement 2 of 3)

task: Wire literature-grounding INTO the propose-hypothesis prompt instead of
catching missing/misaligned citations after the fact. Adds `--ground-via-mcp`
flag that performs a targeted top-K search over the existing literature corpus
(equivalent of `mcp__research-catalog__search_research_catalog`) and surfaces
the most relevant extracts in the LLM context block BEFORE drafting. Drops the
LLM-fabricated-citation failure class to ~zero by giving the model the real
text up front.

mode: IMPLEMENTATION

## Scope Lock

- scripts/research/llm_hypothesis_proposer.py
- scripts/research/lhp/literature_index.py
- scripts/research/lhp/static_checks.py
- tests/test_research/test_llm_proposer_grounding.py

## Blast Radius

- `scripts/research/lhp/literature_index.py` — adds `search_corpus(corpus, query, top_k)`
  function. Pure addition; existing exports unchanged.
- `scripts/research/llm_hypothesis_proposer.py` — adds `--ground-via-mcp` flag,
  derives query keywords from --user-instruction + --candidate-strategy-id
  context, injects targeted extracts into adjacency_context. Default OFF so
  existing operators see no change.
- `scripts/research/lhp/static_checks.py` — `check_citation_content` is already
  fatal. We extend `run_all` to also stamp the parsed dict with a
  `grounding_provenance` field if the YAML carries one, and add
  `check_grounding_provenance_block` validating the field when present
  (non-fatal — operators may pass an MD-only ground if they cite the
  retrieved extracts manually).
- Reads: docs/institutional/literature/*.md (read-only).
- Writes: docs/audit/hypotheses/<slug>.draft.yaml via existing emitter.
- No capital-path mutation. No edit under `trading_app/`, `pipeline/`,
  `docs/runtime/lane_allocation.json`, or `docs/runtime/chordia_audit_log.yaml`.

## Verification

1. `python pipeline/check_drift.py` passes.
2. `python -m pytest tests/test_research/test_llm_proposer_grounding.py
   tests/test_research/test_lhp/ -v` shows all-green.
3. `python scripts/research/llm_hypothesis_proposer.py --slug smoke-grounding
   --ground-via-mcp --dry-run` round-trip: confirms targeted-extracts block
   appears in stderr-emitted screen context.
