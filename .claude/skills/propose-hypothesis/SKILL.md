---
description: Propose a literature-grounded pre-reg hypothesis YAML via LLM. The script refuses to fabricate citations; output is a .draft.yaml requiring human review before promotion.
---

# /propose-hypothesis

## Invoke

```
python scripts/research/llm_hypothesis_proposer.py --slug <short-slug> [options]
```

## Behavior

1. Loads `docs/institutional/literature/*.md` corpus.
2. Queries `gold.db` for currently-active `validated_setups` (read-only).
3. Sends a bounded context to the LLM with strict refuse-or-ground instructions.
4. Validates the LLM output against `trading_app.hypothesis_loader` schema plus static checks.
5. Writes `docs/audit/hypotheses/drafts/YYYY-MM-DD-llm-<slug>.yaml` on success. The `drafts/` subdirectory is invisible to the canonical hypothesis-discovery glob (`directory.glob("*.yaml")` is non-recursive), so no draft can be picked up by the pipeline until it is promoted.
6. On fatal failure: writes `<...>.rejected.txt` (same subdirectory) and exits non-zero. No commit-ready file is produced.

## Human action required after script returns

- Read the draft plus the REVIEW CHECKLIST header.
- Verify each `theory_citation` actually supports the claimed `economic_basis`.
- Move the file up one level to publish:
  ```bash
  mv docs/audit/hypotheses/drafts/2026-05-11-llm-<slug>.yaml \
     docs/audit/hypotheses/2026-05-11-llm-<slug>.yaml
  ```
- Commit, then run:
  ```bash
  scripts/infra/prereg-loop.sh --hypothesis-file <published-path> --execute
  ```

## Dry-run smoke test (no LLM cost)

```
python scripts/research/llm_hypothesis_proposer.py --slug smoke --dry-run \
  --fixture tests/fixtures/lhp/good_yaml_1.yaml
```

## When NOT to use

- Confirmatory audits of already-validated strategies (use `prereg-loop.sh` directly).
- When the desired mechanism has no corpus extract — write the literature extract first under `docs/institutional/literature/`.
