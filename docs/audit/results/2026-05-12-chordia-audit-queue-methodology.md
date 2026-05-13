---
pooled_finding: false
date: 2026-05-12
status: METHODOLOGY_LOCKED_PRE_RERUN
author: claude (main, v2 worktree)
amendments_applied: [E1, E2, E3, E4, E5, E6]
scope: read-only research artifact; no allocator/DB/live state mutation
---

# Chordia-Audit Queue v2 — Methodology

## Scope / Question

**Question:** Of the 844 active validated strategies (MES=48, MGC=13, MNQ=783), which are the most defensible candidates for the next per-strategy Chordia pre-reg audit?

**Scope:** Read-only triage layer over `validated_setups WHERE status='active'`. Per-row canonical gate evaluation. No allocator, DB, live, or pre-reg state mutation.

**Out of scope:** Per-strategy Chordia pre-reg authoring; allocator activation; deployment decisions; MES/MGC regime/profile changes.

## Verdict / Decision

**Headline (locked):** `0 TOP / 0 READY`. No candidate currently passes both Chordia-strict (t ≥ 3.79) AND has enough OOS data to power-refute under canonical `oos_power.py` tiers.

**Next-audit triage (AUDIT_GAP_ONLY bucket, 8 candidates):** post-result triage of strategies whose only blocker is `NO_CHORDIA_AUDIT_LOG_ENTRY` AND Chordia-t ≥ 3.79. All 8 are MNQ E2/E1.

**Decision for next session:** select top-3 from AUDIT_GAP_ONLY for per-strategy Chordia pre-regs (see top-3 recommendation MD). MES + MGC unlock requires more than missing audit-log entries — they're blocked on `c8_oos_status`, `family_status`, or `sample_size` (HANDOFF item #2 follow-up).

## Reproduction / Outputs

**Outputs (this stage):**
- `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` (844 rows × 36 cols)
- `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md` (this file)
- `research/chordia_queue_recompute.py` (entry point)

**Reproduce:**
```bash
cd .worktrees/chordia-audit-queue-v2-2026-05-12
python research/chordia_queue_recompute.py
```

Expected runtime: ~3-5 min on a warm DB. Script prints self-consistency deltas + tier histogram + top-15 preview.

## Caveats / Limitations

- **OOS window is only 70 trading days** (2026-01-01 → 2026-05-10). At typical d=0.1 and n=70, one-sample power is ~0.13 — well below the 0.50 DIRECTIONAL_ONLY threshold. The `0 TOP / 0 READY` headline is the Bailey/Harvey/LdP reality of this OOS span, not a script bug.
- **`scratch_drop_rate` is an approximation.** True realized-eod scratch handling requires bar-level session-end MTM via `pipeline.cost_model.pnl_points_to_r` against `bars_1m` — deferred to per-strategy pre-reg author per RULE 11.
- **MES/MGC family_status = UNKNOWN.** The 2026-05-11 deployability JSON covers only MNQ. MES (48) and MGC (13) candidates have `family_status=UNKNOWN`; their canonical-gate evaluation is limited to fields present in `validated_setups`.
- **Mode-B grandfathered rows.** 12 MNQ rows have `last_trade_day ∈ [2026-01-01, 2026-04-08]` — their `expectancy_r_stored` reflects pre-Phase-0 Mode-B IS. The script flags this via the `stored_minus_mode_a` column; `stored` values for these rows are never a sort key.
- **`AUDIT_GAP_ONLY` is NOT a readiness category.** It is a post-result triage label only. No candidate in this bucket is approved for live capital or replacement.

## Why this artifact exists

HANDOFF item #2 (2026-05-12): the canonical Chordia audit log
(`docs/runtime/chordia_audit_log.yaml`) carries 12 entries — all MNQ. MGC
(13 active validated) and MES (48 active validated) have **zero**
per-instrument Chordia audits. Profile activation cannot deploy them; a
per-strategy Chordia pre-reg + replay is the unlock path.

This methodology produces a ranked CSV
(`2026-05-12-chordia-audit-queue-candidates.csv`) so the next session can
pick the **next** Chordia pre-reg audit candidate without re-litigating
which strategy to audit first.

**This artifact is NOT:**
- a deployment recommendation
- a replacement candidate list
- a readiness gate
- an allocator state change

**This artifact IS:**
- a triage layer over `validated_setups` (status=active)
- a record of canonical gate evaluation per strategy
- the input for selecting per-strategy Chordia pre-reg targets

## Headline result framing (locked BEFORE rerun)

Per Amendment E6 (user-directed 2026-05-12):

> **`0 TOP / 0 READY` is the headline.** Nothing currently passes both
> Chordia-strict (t ≥ 3.79) AND has enough OOS data to power-refute under
> canonical `research.oos_power.POWER_TIERS`. This is the
> Bailey/Harvey/LdP reality of a 70-trading-day OOS window
> (2026-01-01 → 2026-05-10).

`AUDIT_GAP_ONLY` is a post-result triage bucket — strategies whose **only**
blocker is `NO_CHORDIA_AUDIT_LOG_ENTRY` AND whose Chordia-t already clears
strict floor (≥ 3.79) on stored Sharpe × √N. It answers the question
*"of the not-yet-audited, which is most defensible to audit first?"* — it
does **not** redefine readiness.

`AUDIT_GAP_ONLY` strategies are **NOT** approved for live capital. They are
**NOT** replacement candidates. They are **NOT** ready-ish lanes. They are
**candidates for the next Chordia pre-reg audit**, nothing more.

## Amendments E1-E6 (folded into the script before any artifact landed)

| # | Issue | Fix |
|---|---|---|
| E1 | `StrategyFilter` has no `.family` attribute | Subclass-name dispatch via locked `_FAMILY_MAP` (26 entries; halts on unmapped subclass) |
| E2 | G1-G14 codes undefined in repo | 7 canonical `hard_issues` from deployability JSON + 3 new gap codes |
| E3 | `WHERE pnl_r IS NOT NULL` contradicts `realized-eod` | Drop for ExpR; surface `scratch_drop_count` + `scratch_drop_rate` per row |
| E4 | PR numbers unverified | Stamped at execution time via `gh pr list` |
| E5 | Self-consistency 0.02 threshold too tight | Raised to 0.05 (observed strict-unlock deltas ~0.04R) |
| E6 | AUDIT_GAP_ONLY semantic role | **TRIAGE-only**: post-result bucket for picking next audit; not readiness |

## Inputs (canonical sources only)

| Input | Source | Used for |
|---|---|---|
| Candidate inventory (844 rows) | `validated_setups WHERE status='active'` | Primary truth for instrument/orb_label/orb_minutes/rr_target/entry_model/confirm_bars/filter_type/sample_size/sharpe_ratio/expectancy_r/last_trade_day/c8_oos_status |
| MNQ enrichment (~250 rows) | `docs/audit/results/2026-05-11-mnq-all-active-deployability.json` | `family_status`, `hard_issues`, `verdict` for MNQ rows present in the JSON's 4 buckets |
| Allocator state | `docs/runtime/lane_allocation.json` | Per-strategy `allocator_status` annotation |
| Audit log verdicts | `docs/runtime/chordia_audit_log.yaml` | Per-strategy `chordia_log_verdict` + audit age |
| Mode-A IS / OOS metrics | `gold.db orb_outcomes JOIN daily_features` (triple-join on trading_day + symbol + orb_minutes per `daily-features-joins.md`) | Recomputed `n_is_mode_a`, `mode_a_expr`, `mode_a_std`, `n_oos`, `oos_expr` |

## Canonical delegations (no re-encoding)

| Helper | Source | Purpose |
|---|---|---|
| `oos_ttest_power` / `one_sample_power` / `power_verdict` | `research.oos_power` | Criterion 2 — OOS power tier (CAN_REFUTE / DIRECTIONAL_ONLY / STATISTICALLY_USELESS) |
| `filter_signal` | `research.filter_utils` | Fire-mask via `ALL_FILTERS[key].matches_df(df, orb_label)` |
| `compute_chordia_t` / `CHORDIA_T_WITHOUT_THEORY` | `trading_app.chordia` | Strict floor (t ≥ 3.79) |
| `HOLDOUT_SACRED_FROM` | `trading_app.holdout_policy` | Mode-A IS/OOS boundary |
| `ALL_FILTERS` | `trading_app.config` | Canonical filter registry |
| `GOLD_DB_PATH` | `pipeline.paths` | DB connection |

## Gates evaluated per row

**Criterion 1 — Chordia readiness** (`trading_app/chordia.py`):
- (a) `validated_setups.status = 'active'` (upstream filter)
- (b) `sample_size ≥ 100` (Criterion 7 deployable floor — `pre_registered_criteria.md`)
- (c) `sharpe_ratio` non-null (Chordia-t identity input)
- (d) `family_status` not in {`PURGED`, `SINGLETON`} (MNQ only; UNKNOWN for MES/MGC absent from JSON)
- (e) `c8_oos_status` not in {`NEGATIVE_OOS_EXPR`, `FAILED_RATIO`}
- (f) Mode-A `n_is ≥ 50` (the SQL recompute output)

**Criterion 2 — OOS power** (`research.oos_power.one_sample_power`):
- d = |mode_a_expr| / mode_a_std (Cohen's d on strategy's own returns)
- power = one_sample_power(d=d, n=n_oos, alpha=0.05)
- tier = power_verdict(power) ∈ {CAN_REFUTE, DIRECTIONAL_ONLY, STATISTICALLY_USELESS}
- **Thresholds are canonical from `oos_power.py:POWER_TIERS`. No relaxation.**

**Criterion 3 — Filter classification** (`_FAMILY_MAP` via subclass dispatch):
- EXCLUDE families: COST_LT, OVNRNG, ORB_VOL, ORB_G, NO_FILTER
- PREFER families: CROSS_ASSET_PERCENTILE, INTRA_ASSET_PERCENTILE, DIRECTION_CONDITIONAL
- OTHER: any non-EXCLUDE, non-PREFER family (PD_*, VWAP_MID_ALIGNED, VOL_RV*, etc.)

## Blocker code taxonomy (10 codes, canonical-anchored — E2 amendment)

From deployability JSON `hard_issue_counts` (7):
- `c8_not_passed`
- `e2_deployment_unsafe_filter`
- `family_purged`
- `family_singleton`
- `replay_mismatch`
- `sample_size_below_deploy_threshold`
- `slippage_missing`

New for HANDOFF #2-4 gaps (3):
- `NO_CHORDIA_AUDIT_LOG_ENTRY` — strategy absent from `chordia_audit_log.yaml`
- `INSTRUMENT_REGIME_COLD_OR_WARM` — instrument regime ≠ HOT (HANDOFF #4)
- `MODE_A_IS_EMPTY` — Mode-A recompute n_is < 50

## Tier rules (post-gate)

| Tier | Definition | Meaning |
|---|---|---|
| `TOP` | Passes Crit 1 + Crit 2=CAN_REFUTE + Crit 3 in PREFER | None currently — **0 TOP is the canonical headline** |
| `READY` | Passes Crit 1 + Crit 2 in {CAN_REFUTE, DIRECTIONAL_ONLY} | None currently — **0 READY is the canonical headline** |
| `AUDIT_GAP_ONLY` (E6) | Only blocker is `NO_CHORDIA_AUDIT_LOG_ENTRY` AND Chordia-t ≥ 3.79 | **Triage bucket only — candidates for the next Chordia pre-reg audit; not deployable; not replacement; not readiness** |
| `BLOCKED_ON_GAP` | Any other blocker present | Cannot be audited as-is; gap closure required first |
| `DEFERRED_FILTER_EXCLUDED` | Filter family in {COST_LT, OVNRNG, ORB_VOL, ORB_G, NO_FILTER} | Deprioritized per Crit 3 EXCLUDE list |

## Sort rule (post-tier)

1. Tier rank: TOP > READY > AUDIT_GAP_ONLY > BLOCKED_ON_GAP > DEFERRED_FILTER_EXCLUDED
2. Power tier within row: CAN_REFUTE > DIRECTIONAL_ONLY > STATISTICALLY_USELESS
3. Filter preference: PREFER > OTHER
4. `mode_a_expr` descending
5. `years_tested` descending
6. `sample_size_stored` descending

**`expectancy_r_stored` is never a sort key.** It appears as a CSV metadata column with `stored_minus_mode_a` companion so reviewers can see Mode-B contamination magnitude (Mode-B-grandfathered rows had `last_trade_day ∈ [2026-01-01, 2026-04-08]` per `research-truth-protocol.md`).

## Scratch policy (E3 amendment — realized-eod approximation)

ExpR computation uses `WHERE pnl_r IS NOT NULL`. NULL-`pnl_r` row count is
surfaced as `scratch_drop_count` + `scratch_drop_rate` per row so reviewers
can see the realized-eod approximation. **True realized-eod requires
bar-level session-end MTM recompute via `pipeline.cost_model.pnl_points_to_r`
against `bars_1m`** — deferred to per-strategy pre-reg authoring per RULE 11.

## Self-consistency gate (E5 amendment — 0.05 threshold)

Script halts if any of the 3 canonical Chordia-PASSED strategies shows
`|stored - mode_a| ≥ 0.05R`. These strategies were measured under canonical
Mode-A audits in 2026-05-02 (chordia_audit_log entries); a class-wide drift
above 0.05R signals a `validated_setups` or recompute-pipeline class bug,
not per-row noise.

**Stage-1 result (2026-05-12):** all 3 deltas ≤ 0.008R. Self-consistency PASSES.

## Anti-patterns explicitly refused

Per v2 plan §"Anti-patterns this plan refuses" + E6 amendment:
- ❌ No invented thresholds (the 0.02 replay-drift gate v1 used is gone; the
  5% scratch-drop top-3 block ChatGPT proposed is also refused — `scratch_drop_rate`
  is a metadata column, not a gate).
- ❌ No `n_oos`-bucket OOS power proxy (canonical `one_sample_power` only).
- ❌ No regex filter classification (subclass-name dispatch only).
- ❌ No `validated_setups.expectancy_r` as sort key for Mode-B rows.
- ❌ No top-3 citation without `mcp__research-catalog__get_literature_excerpt` verbatim inline.
- ❌ No OOS power relaxation to manufacture TOP/READY candidates (E6).
- ❌ No redefinition of "readiness" to absorb AUDIT_GAP_ONLY (E6).
- ❌ No allocator / chordia_audit_log / validated_setups / live state mutation from this script.

## Verification (post-run)

Per v2 plan §"Verification":

```bash
# 1. Self-consistency (3 canonical Chordia strategies, |delta| < 0.05R) — script prints these.
# 2. CSV row count == 844 (== validated_setups WHERE status='active').
# 3. No mutation outside scope:
git diff origin/main -- docs/runtime/lane_allocation.json     # empty
git diff origin/main -- docs/runtime/chordia_audit_log.yaml   # empty
git diff origin/main -- docs/audit/hypotheses/                # empty
git diff --stat origin/main -- pipeline/ trading_app/         # empty
git diff --stat origin/main -- research/                      # ONE new file
# 4. Canonical delegations are one-liner imports (no re-encoding).
# 5. pipeline/check_drift.py passes.
```

## What this methodology does NOT do

- Does not audit any strategy (per-strategy pre-reg is the unlock path; this artifact ranks **which to audit next**).
- Does not write to `chordia_audit_log.yaml`.
- Does not change `lane_allocation.json`.
- Does not change `validated_setups`.
- Does not author any pre-reg yaml.
- Does not commission live capital.

The next session's work is: pick an AUDIT_GAP_ONLY candidate from the top-3
recommendation MD → author a per-strategy Chordia pre-reg → run the bounded
strict-replay → write an audit row to `chordia_audit_log.yaml`.
