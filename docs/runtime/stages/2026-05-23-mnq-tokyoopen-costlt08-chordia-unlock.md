---
task: |
  Rewrite the 2026-04-22 MNQ TOKYO_OPEN COST_LT08 take prereg into the
  current 2026-05-21 strict-unlock-v1 schema, downgrading theory_grant from
  true to false (Crabel extract not in docs/institutional/literature/),
  run chordia_strict_unlock_v1.py end-to-end against canonical
  orb_outcomes x daily_features under Mode A IS, and propose the
  chordia_audit_log.yaml entry as a DRY-RUN YAML block for user approval.
  No allocator/profile mutation. No chordia_audit_log.yaml write.
mode: IMPLEMENTATION
scope_lock:
  - docs/audit/hypotheses/2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.yaml
  - docs/audit/results/2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.md
blast_radius: |
  Research-only. Two new files in docs/audit/. No edits to pipeline/,
  trading_app/, research/, scripts/, or runtime config. chordia_audit_log.yaml
  is explicitly OUT OF SCOPE (audit-log append is the next stage, user-gated).
  lane_allocation.json is unaffected; no rebalance. Reads gold.db read-only
  via the strict-unlock runner. Writes only the two scope_lock files.
updated: 2026-05-23
---

## Decision context

- Tier-C unlock candidate ranked highest non-overlapping in current allocator
  funnel (memory: `project_chordia_audit_unblock_real_edge_location_2026_05_19.md`).
- Target validated_setups row (canonical query 2026-05-23):
  - strategy_id: MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08
  - sample_size=427, win_rate=0.510, expectancy_r=+0.2037, sharpe_ann=+1.471
  - last_trade_day=2025-12-29 (entirely strict-IS under Mode A)
  - all_years_positive=False (Criterion 9 caveat — surface per-year breakdown)
  - years_tested=6

## Theory-grant audit (per institutional-rigor.md § 7)

- Apr-22 prereg claimed `theory_grant: true` citing Crabel ORB.
- `docs/institutional/literature/` inventory (29 files) contains NO crabel_*.md
  extract. Crabel appears only as a referenced name inside
  fitschen_2013_path_of_least_resistance.md and the still-pending
  PENDING_ACQUISITION_market_profile.md.
- Decision: theory_grant downgrades to false. Strict Chordia threshold
  (t ≥ 3.79) applies. No theory-grant rescue attempt this stage.

## Out of scope

- Adjacent thresholds (COST_LT10, COST_LT12, COST_LT15) — each needs its own prereg
- Adjacent apertures (O15, O30) or RR targets (1.0, 2.0)
- Theory-grant write-up for Crabel (would require new literature extract first)
- chordia_audit_log.yaml mutation (user-gated next stage)
- Allocator rebalance, validated_setups write, or live deployment changes

## Acceptance

1. Prereg passes `estimate_k_budget.py` PASS.
2. `chordia_strict_unlock_v1.py` runs to completion; result MD + CSV emitted.
3. Cohort match: |N_canonical − 427| reported with explanation.
4. Verdict evidence reported with all of: t_IS, ExpR_IS, Sharpe_IS, BH-FDR
   (K_family=1), N_unique_trading_days, OOS sign + dir_match, OOS power tier
   via `research.oos_power`, per-year breakdown (Criterion 9 check).
5. Proposed `chordia_audit_log.yaml` entry shown as dry-run YAML for user
   approval. STOP before any write.
