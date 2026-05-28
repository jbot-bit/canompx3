---
task: |
  Lane B Stage 4b — run the NYSE_PREOPEN MNQ E2 NFP-spillover v1 verdict
  against canonical gold.db and emit docs/audit/results/<slug>.md + <slug>.csv.

  User explicitly approved this stage 2026-05-28: "your approval to start
  Stage 4b (run the verdict against canonical gold.db and emit the result
  MD/CSV). start it".

  Scope: execute Stage 4a's canonical math (compute_full_verdict) against
  the live gold.db, format the K=27 result table as Markdown + CSV. The
  prereg's strict-Chordia promotion gate, BH-FDR composition, DST-balance
  check, and OOS power floor are ALL already implemented and unit-tested
  in research/mnq_nyse_preopen_e2_nfp_spillover_v1.py and were shipped in
  Stage 4a (commit dfd10116). Stage 4b is the I/O wrapper that runs them
  against real data and writes the artifact.

  Hard contract -- NO capital-path writes:
    - NO write to gold.db
    - NO write to experimental_strategies
    - NO write to validated_setups
    - NO write to allocator state / lane_allocation.json / live config
    - NO mutation of research/mnq_nyse_preopen_e2_nfp_spillover_v1.py
      (Stage 4a's scope_lock is closed)
    - Verdict MD + CSV are research artifacts only. Promotion to live
      trading remains separately gated.

  Architecture: separate emit script that *imports* the Stage 4a runner.

mode: IMPLEMENTATION
updated: 2026-05-28T00:00Z
agent: claude (opus 4.7)
supersedes: none

scope_lock:
  - scripts/research/emit_nyse_preopen_verdict.py
  - tests/test_scripts/test_emit_nyse_preopen_verdict.py
  - docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.md
  - docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.csv
  - HANDOFF.md

## Blast Radius
- scripts/research/emit_nyse_preopen_verdict.py — NEW. Thin I/O wrapper. Imports `compute_full_verdict` from `research.mnq_nyse_preopen_e2_nfp_spillover_v1`, opens gold.db read-only, writes MD + CSV. No statistical code; all math is canonical-source delegated to Stage 4a's runner (institutional-rigor § 10).
- tests/test_scripts/test_emit_nyse_preopen_verdict.py — NEW. Exercises the emitter against an in-memory DuckDB seed (same fixture style as the Stage 4a tests). Verifies: (a) MD layout contract — header + per-cell table + summary + BH-FDR section + DST-balance section; (b) CSV column contract; (c) refuses to overwrite without `--force` if outputs already exist; (d) read-only DB open (write attempt raises); (e) prereg SHA gate fires via the runner's `load_promoted_prereg`.
- docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.md — NEW verdict artifact (produced by running the emitter against canonical gold.db). Pooled-finding annotation NOT required: K=27 cell breakdown is the primary output, not a pooled headline.
- docs/audit/results/2026-05-25-mnq-nyse-preopen-e2-nfp-spillover-v1.csv — NEW row-level CSV (one row per cell, with t_IS, N_IS, ExpR_IS, BH q, OOS power tier, verdict label).
- HANDOFF.md — updated to point Last Session at Stage 4b commit.
- Reads: gold.db (READ-ONLY); promoted prereg YAML (read; SHA-locked at 40f032aa...). Writes: NEW MD + CSV files on disk, plus HANDOFF.md tweak. No DB writes anywhere.

## Acceptance (all required before deleting this stage file)
- New emit script tests pass — show output.
- Stage 4a tests STILL pass (regression check, since we import from the runner) — show output.
- Emitter runs against canonical gold.db read-only — show output.
- Produced MD declares the verdict per-cell (K=27 rows) with: cell_id, t_IS, N_IS_on, ExpR_IS, n_oos_on, expr_oos, n_est_is, n_edt_is, dir_match_oos, oos_power_tier, BH q, dst_balance_verdict, verdict_label, verdict_reason.
- Produced CSV columns match the MD-side report.
- `python pipeline/check_drift.py` PASSES (167 checks) — show output.
- Self-review against institutional-rigor § 1 + § 2 against the new emitter.
- Verify MD declares whether any cells PASS_CHORDIA_STRICT (the headline). Note: this answers the prereg's research question — does NYSE_PREOPEN MNQ E2 NFP-spillover have signal at the strict-Chordia hurdle?

## NOT done by this stage (deferred / separately gated)
- Pinecone upsert of the verdict.
- MEMORY.md verdict entry (deferred until user reviews the headline; if PASS, capital-class memory upsert; if FAIL, project memory only).
- Any write to experimental_strategies regardless of verdict (the prereg's single-use SHA gate is armed but Stage 4b explicitly does NOT pull that trigger).
- Any allocator / live-config / lane_allocation change.
- Promotion to live trading.
