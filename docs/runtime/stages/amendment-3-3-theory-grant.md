---
task: Doctrine Amendment 3.3 — explicit metadata.theory_grant, eliminate field-presence trap in hypothesis_loader
mode: IMPLEMENTATION
scope_lock:
  - trading_app/hypothesis_loader.py
  - scripts/research/lhp/static_checks.py
  - docs/institutional/pre_registered_criteria.md
  - docs/institutional/hypothesis_registry_template.md
  - docs/prompts/prereg-writer-prompt.md
  - tests/test_research/test_hypothesis_loader.py
  - tests/test_research/test_static_checks.py
  - scripts/tools/migrate_preregs_amendment_3_3.py
  - docs/audit/hypotheses/*.yaml
---

## Blast Radius

- `trading_app/hypothesis_loader.py:262-295` — replace field-presence inference with fail-closed explicit `theory_grant` read + cross-rule guards. Consumers: Chordia revalidation runner, audit script, future K=1 verify scripts (no consumer-side changes).
- `scripts/research/lhp/static_checks.py:324-369` — `check_citations_exist` short-circuits on `theory_grant=False`. Consumer: `propose-hypothesis` skill at draft-promotion time.
- Doctrine: append Amendment 3.3 to `pre_registered_criteria.md` (supersession-banner pattern, does not rewrite 3.0).
- Template + prompt: add `theory_grant` schema row; update FORBIDDEN list.
- Migration: one-shot script applies to all 203 preregs under `docs/audit/hypotheses/*.yaml`. Promote `drafts/2026-05-13-...rejected.txt` to `docs/audit/hypotheses/2026-05-13-...yaml` with `theory_grant: false`.
- Drift check `check_chordia_result_threshold_matches_prereg`: UNCHANGED.
- Reads: `docs/institutional/literature/` (corpus for citation matching during migration), `docs/audit/hypotheses/*.yaml` (in-place edit).
- Writes: same 8 file paths + 203 prereg yamls.
- No live capital impact; no broker/risk surface; no deployed-lane state change.

Hard stops (per user direction): HALT case in dry-run, test fail, drift fail, out-of-scope diff. No commit until final diff summary. No K=1 execution. No Stage 2 scan.
