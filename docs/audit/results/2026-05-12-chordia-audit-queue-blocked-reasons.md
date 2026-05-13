---
pooled_finding: false
date: 2026-05-12
status: READ_ONLY_GAP_IMPACT_MAP
author: claude (main, v2 worktree)
amendments_applied: [E1, E2, E3, E4, E5, E6]
scope: read-only gap-impact map derived from companion CSV; no allocator/DB/live state mutation
companion_csv: docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv
methodology: docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md
---

# Chordia-Audit Queue v2 — Blocked Reasons Gap-Impact Map

## Metadata

- **Date:** 2026-05-12
- **Scope:** Read-only per-gap unblocking impact analysis over `2026-05-12-chordia-audit-queue-candidates.csv`. Counts each blocker code; reports per-instrument breakdown; identifies which single gap closure unblocks the most rows.
- **Live impact:** None. This document does not change `lane_allocation.json`, `chordia_audit_log.yaml`, `validated_setups`, or any live state.
- **Companion CSV:** `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv`
- **Methodology:** `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`

## Scope

**Question:** For each of the 10 canonical blocker codes declared in the methodology MD, how many of the 844 candidate rows carry it, how many carry it as their **sole** blocker, and what is the per-instrument breakdown? Which single gap closure would unblock the most rows — and does that closure actually move any rows to TOP/READY, or only to the AUDIT_GAP_ONLY triage bucket?

**In scope:** count + crosstab over the `blockers` column of `2026-05-12-chordia-audit-queue-candidates.csv`; explicit MES/MGC narrative explaining why per-strategy Chordia audits cannot unblock those instruments.

**Out of scope:** new audits, allocator/DB/pre-reg mutation, decisions to author any specific pre-reg, deployment gate evaluation, MGC LONDON_METALS regime check (HANDOFF #3).

## Verdict

`READ_ONLY_GAP_IMPACT_MAP`. This is a derived view over the CSV's `blockers` column, not a new audit. The verdict token is deliberately outside the deployment taxonomy.

**Locked headline (verbatim):** `0 TOP / 0 READY / 8 AUDIT_GAP_ONLY / 243 BLOCKED_ON_GAP / 593 DEFERRED_FILTER_EXCLUDED`.

Per Amendment E6 (`memory/feedback_triage_bucket_not_readiness.md`): **gap closure does not equal deployment**. Even if every gap closed, the resulting AUDIT_GAP_ONLY count rises, but AUDIT_GAP_ONLY itself is triage — the next-audit-candidate selector, not readiness.

## Gap-impact table

The `blockers` column in the CSV is pipe-delimited (`|`). Counts below split on `|` per row; "sole blocker" means a row's `blockers` field contains exactly one code.

Counts derived live from `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` (844 rows total).

| Blocker code | Total rows carrying | Sole blocker | MNQ | MES | MGC |
|--------------|--------------------:|-------------:|----:|----:|----:|
| `NO_CHORDIA_AUDIT_LOG_ENTRY` | **832** | **511** | 771 | 48 | 13 |
| `c8_not_passed` | 256 | 0 | 221 | 22 | 13 |
| `MODE_A_IS_EMPTY` | 77 | 2 | 68 | 0 | 9 |
| `sample_size_below_deploy_threshold` | 60 | 0 | 23 | 24 | 13 |
| `replay_mismatch` | 19 | 0 | 19 | 0 | 0 |
| `family_purged` | 18 | 0 | 18 | 0 | 0 |
| `slippage_missing` | 12 | 0 | 12 | 0 | 0 |
| `family_singleton` | 6 | 0 | 6 | 0 | 0 |
| `e2_deployment_unsafe_filter` | 0 | 0 | 0 | 0 | 0 |
| `INSTRUMENT_REGIME_COLD_OR_WARM` | 0 | 0 | 0 | 0 | 0 |

Notes:
- `e2_deployment_unsafe_filter` and `INSTRUMENT_REGIME_COLD_OR_WARM` are declared blocker codes in the methodology MD § "Blocker code taxonomy" but no row in the current CSV carries them. The first absence reflects `trading_app/config.py:3540-3568` `E2_EXCLUDED_FILTER_PREFIXES` already gating banned filters upstream of `validated_setups` row creation (per `.claude/rules/backtesting-methodology.md` § 6.3). The second absence reflects the current regime — all three active instruments (MNQ, MES, MGC) are HOT as of 2026-05-12 per live state. The codes remain in the taxonomy for future use; their zero counts are not a script bug.
- 10 rows in the CSV have an empty `blockers` field. 8 of those are tier `AUDIT_GAP_ONLY` (the 8 rows with `NO_CHORDIA_AUDIT_LOG_ENTRY` as their **only** blocker — the methodology MD's "Tier rules" promote these out of the per-blocker view because they are themselves the triage bucket). The remaining 2 are `DEFERRED_FILTER_EXCLUDED` rows whose filter family alone routes them out of the audit queue without any specific blocker code attaching.

### Per-blocker semantics (cross-reference)

Code definitions are canonical to the methodology MD § "Blocker code taxonomy". Brief reference:

- `NO_CHORDIA_AUDIT_LOG_ENTRY`: strategy is absent from `docs/runtime/chordia_audit_log.yaml`. The HANDOFF #2 gap (12 entries in the log vs 844 active strategies).
- `c8_not_passed`: `validated_setups.c8_oos_status ∈ {NEGATIVE_OOS_EXPR, FAILED_RATIO}` per methodology Crit 1(e).
- `MODE_A_IS_EMPTY`: Mode-A IS recompute returned `n_is < 50` (methodology Crit 1(f)). Indicates the strategy's IS bars do not survive the canonical Mode-A SQL join (typically because the strategy is `validated_setups`-active but has limited representation in `orb_outcomes` ∩ `daily_features`).
- `sample_size_below_deploy_threshold`: `validated_setups.sample_size < 100` per methodology Crit 1(b) and `pre_registered_criteria.md` C7.
- `replay_mismatch`: from `2026-05-11-mnq-all-active-deployability.json` hard_issues — replay drift between `validated_setups` stored values and a deployability re-run.
- `family_purged` / `family_singleton`: family-status fields from the deployability JSON (MNQ-only — UNKNOWN for MES/MGC).
- `slippage_missing`: deployability JSON flag — strategy missing slippage estimate.

## Which single gap unblocks the most rows?

**Answer:** `NO_CHORDIA_AUDIT_LOG_ENTRY` carries the most (832 of 844 rows total; 511 rows carry it as their **sole** blocker).

**But:** of those 511 sole-blocker rows, only **8** also have `chordia_t ≥ 3.79` AND non-EXCLUDE filter family AND non-empty Mode-A — i.e., they are the AUDIT_GAP_ONLY tier (§ methodology "Tier rules"). The other 503 sole-blocker rows are in `DEFERRED_FILTER_EXCLUDED` (filter family in {COST_LT, OVNRNG, ORB_VOL, ORB_G, NO_FILTER}). Closing the audit-log gap alone:
- Moves **0** rows from `BLOCKED_ON_GAP` → `AUDIT_GAP_ONLY` (every BLOCKED_ON_GAP row carries at least one additional blocker beyond NO_CHORDIA_AUDIT_LOG_ENTRY).
- Moves **0** rows from any tier → `READY` or `TOP` (the OOS power floor at 70 trading days dominates — see methodology MD § "Caveats").
- The 8 AUDIT_GAP_ONLY rows are already in their final tier; "closing" their audit-log gap means writing a per-strategy Chordia pre-reg + strict replay + audit-log entry, which is the next-thread work, not this work block.

**Honest framing:** the audit-log gap is the **widest** blocker by count, but closing it does **not** generate new READY/TOP candidates. The OOS power floor — 70 trading days vs the canonical Cohen's-d ≈ 0.1–0.2 effect sizes — is the binding constraint on readiness across all 844 rows. Per `backtesting-methodology.md` RULE 3.3 tier-table, that constraint is structural to the OOS window, not closeable by audit-log work.

The four blockers that would each meaningfully shift the BLOCKED_ON_GAP → AUDIT_GAP_ONLY count if closed (all of them require gap closure on rows that already carry the audit-log gap too):

1. `c8_not_passed` (256 carriers). Closure requires re-evaluating c8 status (e.g., letting strategies accumulate OOS sample, or running c8 with a different OOS window when more days are available). Largest BLOCKED_ON_GAP contributor after the audit-log gap.
2. `sample_size_below_deploy_threshold` (60 carriers). Closure requires accumulating more trading days post-2026-04-23 — passive, not actionable in this work block.
3. `MODE_A_IS_EMPTY` (77 carriers). Closure requires Mode-A IS recompute returning `n_is ≥ 50` — typically blocked by `orb_outcomes ∩ daily_features` join sparsity. Likely indicates `validated_setups` carries strategies whose `orb_outcomes` rows are missing or sparse.
4. `replay_mismatch` (19 carriers, MNQ-only). Closure requires investigating per-strategy replay drift — out of scope for this work block.

## MES/MGC narrative

This subsection is explicitly required by the plan (§ Files to be written, MD 2 § 5).

**MES (48 active validated rows, 0 AUDIT_GAP_ONLY)** — the 5 BLOCKED_ON_GAP rows are all `INTRA_ASSET_PERCENTILE` (ATR_P30/P50/P70) on `CME_PRECLOSE`. All 5 carry `sample_size_below_deploy_threshold` (Crit 7 floor `N ≥ 100`); 4 of 5 also carry `c8_not_passed`. The 43 `DEFERRED_FILTER_EXCLUDED` rows are split across `COST_LT`, `ORB_G`, `ORB_VOL`, `OVNRNG`, `NO_FILTER` — all in the methodology Crit 3 EXCLUDE list.

**Implication for MES:** per-strategy Chordia audits **cannot** unblock MES. The binding constraints are:
- **Sample size**: 5 of 5 BLOCKED_ON_GAP rows need `N ≥ 100`. Currently the top BLOCKED_ON_GAP MES row has `N = 56` (`MES_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15`, t=5.134). These rows will accumulate sample passively as MES trades more days through their preferred sessions; the highest-t BLOCKED_ON_GAP rows could plausibly cross the floor within several more months of trading.
- **C8 status**: 4 of 5 BLOCKED_ON_GAP rows carry `c8_not_passed`. Closure requires either accumulating OOS sample (passive) or reclassifying the c8 verdict (a `validated_setups` upstream change, out of scope).
- **`AUDIT_GAP_ONLY` migration path**: when an MES BLOCKED_ON_GAP row drops both `c8_not_passed` and `sample_size_below_deploy_threshold`, it becomes a candidate for AUDIT_GAP_ONLY if its `chordia_t ≥ 3.79` survives the larger sample. The two MES rows that currently meet the t-threshold (`ATR_P70_O15` t=5.134, `ATR_P30_O30` t=5.021) are plausible AUDIT_GAP_ONLY candidates in a future rerun once gap closure happens upstream.

**Mirror of HANDOFF item #2 (2026-05-12):** "the canonical Chordia audit log (`docs/runtime/chordia_audit_log.yaml`) carries 12 entries — all MNQ. MGC (13 active validated) and MES (48 active validated) have **zero** per-instrument Chordia audits." That gap is real, but closing it alone for MES means writing 5 pre-regs (one per BLOCKED_ON_GAP row) **after** those rows clear their sample/c8 gaps. Doing a Chordia audit on an MES row with `N = 56` would produce a pre-reg that fails C7 at deployment — wasted authorship cycles.

**MGC (13 active validated rows, 0 AUDIT_GAP_ONLY, 0 BLOCKED_ON_GAP, 13 DEFERRED_FILTER_EXCLUDED)** — every MGC row is `DEFERRED_FILTER_EXCLUDED` because every MGC active validated strategy uses a filter family in the EXCLUDE list (ORB_VOL = 12 rows, ORB_G = 1 row). All 13 also carry `c8_not_passed`. 9 of 13 carry `MODE_A_IS_EMPTY` (MGC's Mode-A IS recompute returns `n_is < 50` — `orb_outcomes ∩ daily_features` is sparse on `LONDON_METALS` because MGC trades fewer days in that session relative to MNQ on US sessions). 12 of 13 carry `sample_size_below_deploy_threshold`.

**Implication for MGC:** the situation is worse than MES. MGC has **zero** rows in a PREFER or OTHER filter family at the current snapshot. Per-strategy Chordia audits do not apply — there is nothing to audit that would clear Crit 3 (filter family must be non-EXCLUDE for the AUDIT_GAP_ONLY tier to be reachable).

The route to MGC participation is **upstream** of this queue:
- **HANDOFF item #3 (2026-05-12)** flags an MGC `LONDON_METALS` regime check — out of scope for this work block, but the most plausible next step. If the regime check reaches a verdict that supports MGC trading on `LONDON_METALS`, a fresh `strategy_discovery` run for MGC in non-EXCLUDE filter families could populate `validated_setups` with audit-eligible rows.
- Alternatively, a `strategy_discovery` run on `MGC × CME_REOPEN` (already represented at 1 row in the EXCLUDED bucket) with non-EXCLUDE families could surface candidates — but this is also out of scope here.

**Net summary for MES + MGC:** per-strategy Chordia audits are the unlock path for MNQ, not for MES/MGC. The plan's framing — "for MNQ this queue selects the next pre-reg; for MES/MGC, the queue **confirms** that per-strategy audits won't help and points the next session at upstream gap closure" — is what the data supports.

## Caveats

1. **Gap closure ≠ deployment.** Even if every blocker closed, the resulting AUDIT_GAP_ONLY count would rise, but AUDIT_GAP_ONLY is still triage. The OOS power floor (70 trading days vs Cohen's d ≈ 0.15) prevents any current row from reaching READY/TOP under canonical `research/oos_power.py:POWER_TIERS`. This is structural to the OOS window — no audit-log work can fix it.
2. **AUDIT_GAP_ONLY is not readiness.** Per E6 + `memory/feedback_triage_bucket_not_readiness.md`. Forwarding any AUDIT_GAP_ONLY row to `lane_allocation.json` from this artifact is a doctrine violation.
3. **Co-occurrence is the norm, not the exception.** 832 of 844 rows carry `NO_CHORDIA_AUDIT_LOG_ENTRY`; many of those also carry 2–4 additional codes. The "sole blocker" column intentionally undercounts a code's impact because almost every row with another blocker **also** carries the audit-log gap. For triage decisions, the "sole blocker" column over-weights how easy closure is; for understanding the structural debt, the "total carriers" column is the truer view.
4. **Source-of-truth chain.** Counts are derived from CSV column `blockers`. The CSV is derived from `research/chordia_queue_recompute.py`. The script's blocker assignment logic is canonical (E2 amendment); this MD does not re-derive any blocker logic — it only counts.
5. **Deployability JSON coverage is MNQ-only.** Per methodology MD § "Inputs": MNQ enrichment from `2026-05-11-mnq-all-active-deployability.json` is the source of `family_status`, `replay_mismatch`, `family_purged`, `family_singleton`, `slippage_missing`. MES (48) and MGC (13) candidates have `family_status=UNKNOWN`; their blocker rows show only codes derivable from `validated_setups` columns plus the canonical Mode-A SQL. This is documented honestly in the methodology MD § "Caveats" too.

## Reproduction

From the worktree root `C:/Users/joshd/canompx3/.worktrees/chordia-audit-queue-v2-2026-05-12`:

```bash
# Reproduce the entire gap-impact table from CSV
python -c "
import csv
from collections import Counter, defaultdict

with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    rows = list(csv.DictReader(f))

total = Counter()
sole = Counter()
by_inst = defaultdict(Counter)

for r in rows:
    codes = [c.strip() for c in (r['blockers'] or '').split('|') if c.strip()]
    for c in codes:
        total[c] += 1
        by_inst[c][r['instrument']] += 1
    if len(codes) == 1:
        sole[codes[0]] += 1

for code in sorted(total, key=lambda c: -total[c]):
    bi = by_inst[code]
    print(f'{code:45s} total={total[code]:>4} sole={sole[code]:>4} '
          f'MNQ={bi.get(\"MNQ\",0):>4} MES={bi.get(\"MES\",0):>3} MGC={bi.get(\"MGC\",0):>3}')"
```

Expected output (pipe-delimited blocker splitting):

```
NO_CHORDIA_AUDIT_LOG_ENTRY                    total= 832 sole= 511 MNQ= 771 MES= 48 MGC= 13
c8_not_passed                                 total= 256 sole=   0 MNQ= 221 MES= 22 MGC= 13
MODE_A_IS_EMPTY                               total=  77 sole=   2 MNQ=  68 MES=  0 MGC=  9
sample_size_below_deploy_threshold            total=  60 sole=   0 MNQ=  23 MES= 24 MGC= 13
replay_mismatch                               total=  19 sole=   0 MNQ=  19 MES=  0 MGC=  0
family_purged                                 total=  18 sole=   0 MNQ=  18 MES=  0 MGC=  0
slippage_missing                              total=  12 sole=   0 MNQ=  12 MES=  0 MGC=  0
family_singleton                              total=   6 sole=   0 MNQ=   6 MES=  0 MGC=  0
```

Companion artifacts:
- Methodology: `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md`
- Candidates render: `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md`
- Top-3 next-audit candidates: `docs/audit/results/2026-05-12-chordia-audit-queue-top3-prereg-recommendation.md`
- Plan: `docs/plans/2026-05-12-chordia-audit-queue-v2-plan.md`
