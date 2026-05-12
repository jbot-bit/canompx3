---
paths:
  - "docs/audit/hypotheses/**"
---

# Hypothesis Pre-Reg Discipline — auto-loaded when editing pre-regs

**Authority:** `pre_registered_criteria.md` Criterion 1 (pre-reg file mandatory) and Criterion 2 (MinBTL K-budget bound).

This rule auto-injects when you edit anything under `docs/audit/hypotheses/`. Its job: stop you from writing a pre-reg that fails Bailey 2013 MinBTL or omits the required schema fields.

## Hard sequence (do these BEFORE saving the file)

1. **Decide N first.** What's the `total_expected_trials` (family) / `primary_selection_trials` (large scan) / `n_trials` (Pathway B K=1)? Pick the canonical key for your testing mode:
   - `testing_mode: family` → `total_expected_trials` (top-level)
   - `testing_mode: individual` (Pathway B) → `n_trials: 1`
   - Large research-only scan with no `experimental_strategies` write → `primary_selection_trials` under `trial_budget:`

2. **Run the K-budget gate IMMEDIATELY** — before any further content:
   ```bash
   .venv/Scripts/python.exe scripts/tools/estimate_k_budget.py --hypothesis <path-to-yaml>
   ```
   Or via MCP: call `mcp__research-catalog__estimate_k_budget` with `hypothesis_id=<file-stem>`.

   If it returns `FAIL`, **stop**. Either reduce N to the reported `n_max_at_horizon`, or amend Criterion 2 with explicit noise-floor disclosure. Do not commit a failing pre-reg.

3. **Run `/nogo <topic>`** to check the graveyard. If the topic appears with NO-GO / KILL verdict, re-litigation requires the reopen criteria from the NO-GO Registry row. Reference it explicitly.

4. **Write the pre-reg** per `docs/prompts/prereg-writer-prompt.md` schema. Don't reinvent — the prompt's § FORBIDDEN and § failure-mode table catch the Phase D D-0 framing error class.

## What the drift check enforces

`pipeline/check_drift.py::check_hypothesis_minbtl_compliance` (Check 57 in the registry) BLOCKS commits when a pre-reg dated `>= 2026-05-12`:
- declares an instrument without `total_expected_trials` / `primary_selection_trials` / `n_trials` (Criterion 1 violation), OR
- exceeds the operational cap (N>300 clean / N>2000 proxy) (Criterion 2), OR
- requires more clean-data years than the instrument has (Bailey horizon).

Pre-`2026-05-12` files emit warnings but don't block — they're grandfathered.

**Unknown instruments (6A/6B/6J, GC proxy) are always advisory** — extending coverage requires amending `pre_registered_criteria.md` Criterion 2 + `scripts/tools/minbtl_retro_report.CLEAN_YEARS_BY_INSTRUMENT`.

## Forgetting risk

You will forget. That is fine because:
- The drift check fires on **every commit** via `.githooks/pre-commit`.
- The Claude Code hook fires the drift check post-edit on production-code paths.
- Sentinel-date logic means only NEW pre-regs trigger the block.

But the moment to act is **before** the file is written, not after a commit is blocked. Calling `estimate_k_budget` inline takes ~50ms; debugging a blocked commit after writing 200 lines of yaml takes 20 minutes.

## Related

- `.claude/rules/research-truth-protocol.md` § Phase 0 Literature Grounding — the broader pre-reg ceremony.
- `docs/prompts/prereg-writer-prompt.md` — schema and forbidden patterns.
- `pre_registered_criteria.md` Criterion 1 + Criterion 2 — the locked rules.
- `scripts/tools/estimate_k_budget.py` — the gate itself.
- `pipeline/check_drift.py::check_hypothesis_minbtl_compliance` — the drift check.
