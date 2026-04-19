---
mode: IMPLEMENTATION
slug: a2b-2-shape-e-eq9-population-fix
stage: 1/1
started: 2026-04-19
task: "A2b-2 Shape E follow-up — fix Bailey-LdP Eq 9 population mismatch + close test/integration gaps for A+"
---

# Stage 1 — Eq 9 population fix + multi-pair test + integration smoke

## Context
Self-review of A2b-2 Shape E (commits `fc05db8f` + `fc77eb9e`) graded the
implementation A-. One MEDIUM and two LOW findings:

- MEDIUM: Eq 9 (`N̂ = ρ̂ + (1-ρ̂)·M`) is computed with ρ̂ from ~6 deployable
  lanes but M = ~21 from `edge_families` — population mismatch. Per
  Bailey-LdP 2014 (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md:92`),
  ρ̂ must be the average correlation among the M trials being deflated.
- LOW-1: Test T1 + T4 don't exercise multi-pair ρ̂ averaging (T1 uses
  empty pairs, T4 uses single pair).
- LOW-2: `scripts/tools/rebalance_lanes.py` not smoke-tested end-to-end
  on canonical DB after changes.

## Approach (institutional fix, not band-aid)

The fix recognizes that DSR has two distinct selection layers:

1. **Validation layer (existing `dsr_at_n_eff_raw`)**: M = `n_eff_raw`
   from `edge_families` = ~21. Models "this lane passed BHY out of all
   validated families." No ρ̂ correction (edge_families are designed to
   be roughly independent). Mirrors validator's framing exactly.

2. **Allocation layer (`dsr_at_n_hat_eq9` after fix)**: M = number of
   deployable candidates from which ρ̂ is measured. Models "this lane
   was selected from the deployable allocation candidates." Eq 9 applied
   with INTERNALLY CONSISTENT population: ρ̂ from deployable, M from
   deployable.

Both are valid framings of different selection layers. Reporting both
gives the operator a complete picture of the noise floor at each layer.

## Scope Lock
- trading_app/lane_allocator.py
- tests/test_trading_app/test_lane_allocator.py
- scripts/tools/rebalance_lanes.py
- .claude/skills/regime-check/SKILL.md

Notes on each file:
- lane_allocator.py — Eq 9 math + new m_deployable field + lit_ref update
- test_lane_allocator.py — T4 update + T5 multi-pair + T6 fallback
- rebalance_lanes.py — added mid-stage: stdout print consumes new m_deployable
- regime-check/SKILL.md — Step 1b note mentions the field; doc update only

## Acceptance criteria
1. `enrich_scores_with_dsr_diagnostics` computes Eq 9 with M = count of
   distinct deployable strategy_ids represented in `pairs` (or 0 if no
   pairs available).
2. JSON `dsr_diagnostics.lit_ref` documents the allocator-layer framing
   honestly (no false "Eq 9 with raw N_eff" claim).
3. Existing T4 test updated to reflect new math (M=2 in fixture, not 21).
4. New T5 test: multi-pair averaging — 3 pairs with mixed ρ values.
5. `pytest tests/test_trading_app/test_lane_allocator.py` → 47/47 pass.
6. `python pipeline/check_drift.py` → no new violations.
7. Smoke test: rebalance_lanes.py runs on canonical DB and produces a
   `dsr_diagnostics` block in the JSON with the new framing.
8. Self-review pass.

## Blast Radius
- New fields are still informational-only; no selection logic touched.
- `dsr_at_n_eff_raw` unchanged (still mirrors validator).
- Only `dsr_at_n_hat_eq9` semantics change: was "Eq 9 with raw M=21",
  now "Eq 9 with deployable M".
- JSON consumers: only the regime-check skill reads `dsr_diagnostics`;
  it presents both fields verbatim — no consumer-side fix needed.
- LaneScore fields, save_allocation signature: UNCHANGED.
