---
plan_id: chordia-audit-queue-v2-2026-05-12
supersedes: prior in-conversation plan (v1) — never committed; archived in
  transcript at C:\Users\joshd\.claude\projects\C--Users-joshd-canompx3\
status: APPROVED_2026_05_12_WITH_AMENDMENTS_E1_E5
target_worktree: .worktrees/chordia-audit-queue-v2-2026-05-12
target_branch: feat/chordia-audit-queue-v2-2026-05-12
target_base: origin/main
risk_tier: HIGH
mutation_policy: docs-only + one read-only research script
date: 2026-05-12
---

# Chordia-Audit Queue v2 — Lit-Grounded Plan

## Amendments before execution (2026-05-12 main thread)

Five gaps surfaced when verifying v2 against canonical sources. All folded into
the executable script before any artifact lands.

### E1 (LOAD-BEARING) — `StrategyFilter` does NOT expose `.family`

v2 §Methodology #6 and Crit 3 claim `ALL_FILTERS` entries expose `key`, `family`,
`predicate_class`. Verified false against `trading_app/config.py:413` —
`StrategyFilter` carries only `filter_type`, `description`, `requires_micro_data`.
Filter family identity lives in the **subclass name** (e.g.,
`OvernightRangeFilter` → `OVNRNG`).

**Fix.** `research/chordia_queue_recompute.py` derives family via
`type(filter_instance).__name__` against a locked `_FAMILY_MAP` (25 subclass
keys); script asserts every concrete subclass in `ALL_FILTERS` resolves or halts.

### E2 — Blocker taxonomy: drop undefined G1-G14, use canonical hard_issues codes

v2 referenced G1-G14 blocker codes that exist nowhere in the repo (v1
in-conversation working memory). User decision 2026-05-12: use the 7 canonical
`hard_issues` codes from `2026-05-11-mnq-all-active-deployability.json`
(`c8_not_passed`, `e2_deployment_unsafe_filter`, `family_purged`,
`family_singleton`, `replay_mismatch`, `sample_size_below_deploy_threshold`,
`slippage_missing`) + 3 new codes for HANDOFF #2-4 gaps
(`NO_CHORDIA_AUDIT_LOG_ENTRY`, `INSTRUMENT_REGIME_COLD_OR_WARM`,
`MODE_A_IS_EMPTY`). 10 codes total, all anchored to canonical artifacts.

### E3 — Scratch policy: measure drop rate, surface as CSV column, no hard block

v2 §Methodology #4 SQL has `WHERE pnl_r IS NOT NULL` but Criterion 13 default is
`realized-eod` (count scratches at session-end MTM, do not drop). True
realized-eod requires bar-level recompute from `bars_1m` (separate research
stage).

**Fix.** Keep `WHERE pnl_r IS NOT NULL` for ExpR computation; separately count
NULL-`pnl_r` rows as `scratch_drop_count` and emit `scratch_drop_rate` per
candidate in the CSV. Halt the script only if the 3 canonical Chordia-PASSED
strategies show > 5pp deviation in `scratch_drop_rate` vs each other (signal of
class-wide outcome_builder change). Reviewer-visible number, not a top-3 gate;
per-strategy pre-reg author decides whether bar-level recompute is required
before authoring.

### E4 — PR# stamping at execution time

v2 cites PR #271/#272 unverified. Script run will `gh pr list --search ...` and
stamp actual PR numbers (or `NOT_YET_OPENED`) in the recommendation MD.

### E5 — Self-consistency threshold 0.02 → 0.05

v2 verification step 7 used `delta < 0.02`. Strict-unlock CSVs already show
deltas up to 0.037 between stored and canonical strict-unlock measures (e.g.,
OVNRNG_100 RR1.5). Use `delta < 0.05` as the halt threshold; tighter would
false-positive on noise.

### Read-only execution scope (user constraint 2026-05-12)

- Read-only: no allocator mutation, no DB writes, no live state mutation, no
  bootstrap-runtime-control.
- No pre-reg YAML writes (top-3 produces a *recommendation*, not a pre-reg).
- Stop after queue CSV + top-3 recommendation MD. Per-strategy pre-reg authoring
  is the next session's work.

---



**Why v2:** the v1 plan, when executed honestly against canonical inputs,
required ad-hoc proxies for three load-bearing gates. Caught at the build
step before any artifact landed. User constraint: "no fucking adhoc.
resource and lit grounded." This plan removes the proxies by adding one
read-only bulk-recompute script that the queue MDs consume.

---

## v1 failure inventory (what was about to ship and why it was wrong)

| # | v1 ad-hoc move | Rule violated | v2 fix |
|---|---|---|---|
| 1 | `REPLAY_DRIFT_TOLERANCE_R = 0.02` invented as gate threshold | `institutional-rigor.md` §4 (re-encode logic with no canonical source) — no literature anchor for "magnitude of replay drift that invalidates Chordia-t" | DROP gate. Use the 786-batch's canonical `verdict` field instead — `BLOCKED_REPLAY_MISMATCH` is the existing canonical verdict for material drift. Surface every candidate's `expectancy_match` + `expr_drift` in the CSV as data, not gate. |
| 2 | Substituted `n_oos`-bucket tiers for `oos_ttest_power()` | `backtesting-methodology.md` RULE 3.3, `feedback_oos_power_floor.md`, `integrity-guardian.md` §2 (canonical-source delegation) | CALL `research.oos_power.oos_ttest_power(is_delta, is_pooled_std, n_oos_a, n_oos_b)` per candidate from the bulk script. `is_delta` = strategy's IS ExpR. `is_pooled_std` = std(pnl_r) over IS trades. `n_oos_*` = OOS trade counts. All four come from `orb_outcomes` directly. |
| 3 | Sorted on `metrics.expectancy_r` (Mode-B-grandfathered for `last_trade_day` in `[2026-01-01, 2026-04-08]`) | `research-truth-protocol.md` §"Mode B grandfathered validated_setups baselines" | RECOMPUTE Mode A ExpR per candidate from `orb_outcomes` with `trading_day < HOLDOUT_SACRED_FROM` (2026-01-01). Surface both stored and Mode-A-recomputed; sort on Mode-A. Flag every row with delta > 0.02R between them. |
| 4 | Heuristic `last_trade_day >= 2026-01-01 AND n_oos==0` as Mode A contamination | Not how Mode A contamination is detected | DROP heuristic. The canonical contamination check is the IS-window endpoint. After recompute (Fix 3) the issue dissolves: we always have a strict Mode A ExpR; the stored one is reported as metadata, not as a gate. |
| 5 | Hand-wrote filter-class → literature mapping without `mcp__research-catalog__get_literature_excerpt` | `institutional-rigor.md` §7 "Ground in local resources before training memory" | For each of the 3 PREFER filter classes the queue's top-3 will cite, the recommendation MD will include a verbatim excerpt fetched via `mcp__research-catalog__get_literature_excerpt` and the source file path. No citation lands without the excerpt fetch. |
| 6 | Filter classification by regex on `strategy_id` suffix | `integrity-guardian.md` §2 (Canonical Sources — never hardcode) | Source filter classification from `trading_app.config.ALL_FILTERS` registry via Python import. The `StrategyFilter` instance carries the canonical `key` + family attribution; the script reads it instead of regex-parsing `strategy_id`. |

---

## v2 scope (exact deliverables, all docs except 1 read-only research script)

| Path | Content | Mutation class |
|---|---|---|
| `research/chordia_queue_recompute.py` | NEW — one-shot read-only script. Loads candidate inventory (786-batch + strategy-lab MES/MGC payloads), joins `validated_setups.sharpe_ratio` via `gold-db.strategy_lookup`, recomputes per-candidate Mode A `is_delta`, `is_pooled_std`, `n_oos_*` from `orb_outcomes`, calls `research.oos_power.oos_ttest_power()` for each, classifies filter via `trading_app.config.ALL_FILTERS`, writes the CSV. No DB writes. No production mutation. NEW research script lives in `research/` per RULE 11. Carries `# scratch-policy: live-only-rows-from-orb-outcomes-with-pnl_r-not-null` per `feedback_scratch_pnl_null_class_bug.md`. | research code (new) |
| `docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md` | Methodology — gate definitions citing canonical sources line-by-line, gap inventory (G1..G14), per-candidate column schema for the CSV, sort rule. Frontmatter: `pooled_finding: false` per `pooled-finding-rule.md`. | docs |
| `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv` | Output of the recompute script. ~837 rows with: queue_rank, queue_tier (TOP / READY / BLOCKED_ON_GAP / DEFERRED_FILTER_EXCLUDED / DEFERRED_NOT_READY), Chordia-t (canonical from `compute_chordia_t(sharpe_ratio, sample_size)`), oos_ttest_power output (power, cohen_d, tier), Mode A IS ExpR, stored vs Mode-A delta, family_status, filter_class (from `ALL_FILTERS`), c8_oos_status, allocator_status, gap-code blockers. | docs |
| `docs/audit/results/2026-05-12-chordia-audit-queue-candidates.md` | Human-readable: top-15 detailed, full table appendix grouped by instrument. | docs |
| `docs/audit/results/2026-05-12-chordia-audit-queue-blocked-reasons.md` | Per-gap impact (G1-G14): count of candidates each gap blocks, the single gap that unblocks the most top-of-queue rows. | docs |
| `docs/audit/results/2026-05-12-chordia-audit-queue-top3-prereg-recommendation.md` | Top 3 with verbatim `mcp__research-catalog__get_literature_excerpt` content inline for each cited filter class. Numeric kill criteria. MinBTL check. | docs |
| `HANDOFF.md` | Append entry. | docs |

---

## Methodology — every gate has a canonical anchor

### Inputs (canonical only — no validated_setups reads for ranking)

1. **786-row MNQ batch** — `docs/audit/results/2026-05-11-mnq-all-active-deployability.json`. Used for: strategy_id enumeration, `metrics.family_status` (Bonferroni FDR survivor classification), `c8_oos.c8_oos_status`, `c8_oos.n_oos`, `verdict`, `replay.*` metadata. NOT used for ranking baseline.
2. **MES + MGC candidate IDs** — `mcp__strategy-lab__list_promotable_candidates(instrument='MES')` + `(instrument='MGC')`. Returns 39 MES + 12 MGC FIT candidates. Used for strategy_id enumeration only.
3. **Sharpe ratio (canonical Chordia input)** — `gold-db.strategy_lookup` MCP per candidate, returning `validated_setups.sharpe_ratio`. Note: `validated_setups` is read for the Sharpe number ONLY; this is the canonical Chordia input per `trading_app/chordia.py` docstring lines 8-17. Per `research-truth-protocol.md` Mode B warning, the Sharpe **value** is grandfathered if `last_trade_day in [2026-01-01, 2026-04-08]` — the script flags this on the row and downgrades any rank-1 placement of such a row to `BASELINE_PENDING_MODE_A_RECOMPUTE`.
4. **Mode A ExpR + std + n_oos** — single bulk SQL pass against `orb_outcomes` joined to `daily_features` for filter application:
   ```sql
   SELECT strategy_id,
          COUNT(*) FILTER (WHERE trading_day < '2026-01-01' AND pnl_r IS NOT NULL) AS n_is,
          AVG(pnl_r) FILTER (WHERE trading_day < '2026-01-01' AND pnl_r IS NOT NULL) AS mode_a_expr,
          STDDEV_SAMP(pnl_r) FILTER (WHERE trading_day < '2026-01-01' AND pnl_r IS NOT NULL) AS mode_a_std,
          COUNT(*) FILTER (WHERE trading_day >= '2026-01-01' AND pnl_r IS NOT NULL) AS n_oos,
          AVG(pnl_r) FILTER (WHERE trading_day >= '2026-01-01' AND pnl_r IS NOT NULL) AS oos_expr
   FROM <canonical orb_outcomes JOIN daily_features applying filter_type>
   GROUP BY strategy_id;
   ```
   `HOLDOUT_SACRED_FROM` imported from `trading_app.holdout_policy`, not inlined. Filter application delegates to `research.filter_utils.filter_signal(df, filter_key, orb_label)` per `research-truth-protocol.md` §"Canonical filter delegation". One SQL pass per filter family (or one pass with CASE WHEN dispatching on filter_type) — TBD by what `filter_utils` supports. NULL `pnl_r` rows excluded via the `WHERE pnl_r IS NOT NULL` clause per `feedback_scratch_pnl_null_class_bug.md`.
5. **Allocator status + Chordia log** — `docs/runtime/lane_allocation.json` + `docs/runtime/chordia_audit_log.yaml`. As in v1.
6. **Filter class** — Python import: `from trading_app.config import ALL_FILTERS`; the dataclass exposes `key`, `family`, `predicate_class` per filter. The script reads these instead of regex-parsing strategy_id.

### Gates (every gate cites its canonical source)

**Criterion 1 — Chordia readiness.** Source: `trading_app/chordia.py`.
- (a) `validated_setups` row exists with `status='active'`.
- (b) `metrics.sample_size >= 100` (Criterion 7 of `pre_registered_criteria.md`).
- (c) `validated_setups.sharpe_ratio` non-null (Chordia-t identity from chordia.py:71-105).
- (d) `metrics.family_status` in {`ROBUST`, `WHITELISTED`, `SINGLETON`} (NOT `PURGED`).
- (e) `c8_oos.c8_oos_status` in {`PASSED`, `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`, `NO_OOS_DATA`}. EXCLUDE `NEGATIVE_OOS_EXPR` and `FAILED_RATIO`.
- (f) Mode-A-recomputed `n_is >= 50` (the SQL pass output). If recompute yields <50, candidate has no Mode-A IS sample and ranks as `BLOCKED_ON: MODE_A_IS_EMPTY` — *this is the canonical replacement for v1's e1/e2 heuristics.*
- (g) For MES/MGC: also blocked on G1+G2+G7+G8 as labeled.

**Criterion 2 — OOS power.** Source: `research.oos_power.oos_ttest_power()`.
- Direct call, never re-implement. Inputs: `is_delta = mode_a_expr` (single-sample interpretation — Cohen's d on the strategy's edge, using one-sample formulation `_n_for_power` if zero contrast). Per `oos_power.py:235-263`, `one_sample_power(d, n)` is the right helper for "is OOS large enough to detect the IS effect on this strategy's own returns" — the bull-short-avoidance precedent.
- Tier from `power_verdict()` — canonical mapping `POWER_TIERS`.

**Criterion 3 — Filter classification.** Source: `trading_app.config.ALL_FILTERS`.
- EXCLUDE: filter.family in {`COST_LT`, `OVNRNG`, `ORB_VOL`, `ORB_G`} ∪ {`NO_FILTER`}.
- PREFER: filter.family in {`CROSS_ASSET_PERCENTILE`, `INTRA_ASSET_PERCENTILE`, `DIRECTION_CONDITIONAL`}.
- OTHER: anything not in EXCLUDE or PREFER (PD_*, PDR_*, VOL_RV*, etc.) — surface with prior-verdict annotation if `mcp__research-catalog__search_research_catalog` returns a KILL/PARK/NO-GO for the filter key.
- The actual family attribution comes from each `StrategyFilter` instance's metadata, not from regex on suffix.

**Criterion 4 — Correlation / additive EV.** Deferred to per-candidate audit MD per v1 plan (acknowledged out-of-band of queue scope).

**Criterion 5 — Live deployability.** Same as v1: MNQ open, MES/MGC blocked on G1+G2+G7+G8.

### Sort rule (post-gate)

1. Tier: `TOP` (Crit 2 = `CAN_REFUTE`) > `READY` (Crit 2 in `DIRECTIONAL_ONLY`) > `BLOCKED_ON_GAP` > `DEFERRED_*`.
2. Power: `CAN_REFUTE` > `DIRECTIONAL_ONLY` > `STATISTICALLY_USELESS`.
3. Filter pref: `PREFER` > `OTHER`.
4. Mode-A-recomputed ExpR descending.
5. `years_tested` descending.
6. `metrics.sample_size` descending.

**Stored ExpR is NEVER a sort key.** It appears only as a metadata column with a `stored_minus_mode_a_delta` companion column so reviewers can see Mode B contamination magnitude.

### Anti-pigeonhole assertions (verified at MD-commit time)

Unchanged from v1, but now verifiable because the inputs are canonical:
- Top-3 must contain ≥1 non-MNQ row OR explicitly document zero MES/MGC pass Steps 1-3.
- Top-3 must not contain any EXCLUDE-filter row.
- X_MGC_ATR70 enters top-3 only if its Mode-A recompute passes Crit 1(f) AND its `power_verdict` is at least `DIRECTIONAL_ONLY`. Both numbers are surfaced verbatim.
- If zero `TOP` candidates emerge, headline is `NO_ACTIONABLE_CAN_REFUTE_CANDIDATE` and top-3 is filled from `READY` per the canonical sort; the recommendation MD's "next action" identifies which gap (G1, G2, G7, or "OOS extension on the highest-ranked candidate") unblocks the most rows.

---

## Read-only research script — `research/chordia_queue_recompute.py`

### Design contract

```python
"""Chordia-audit queue recompute — Mode A canonical, lit-grounded.

Read-only. No DB writes. No `validated_setups` mutation. Carries
`# scratch-policy: rows with pnl_r IS NOT NULL only` per
feedback_scratch_pnl_null_class_bug.md.

Inputs:
  - docs/audit/results/2026-05-11-mnq-all-active-deployability.json
  - docs/runtime/lane_allocation.json
  - docs/runtime/chordia_audit_log.yaml
  - strategy-lab MCP list_promotable_candidates (MES + MGC) — payload
    captured as YAML alongside this script for reproducibility
  - gold-db query_trading_db / strategy_lookup (validated_setups Sharpe join)
  - orb_outcomes JOIN daily_features (Mode A + OOS bulk recompute)

Outputs:
  - docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv

Canonical delegations (no re-encoding):
  - oos_ttest_power            : research.oos_power
  - compute_chordia_t          : trading_app.chordia
  - filter_signal              : research.filter_utils
  - HOLDOUT_SACRED_FROM        : trading_app.holdout_policy
  - GOLD_DB_PATH               : pipeline.paths
  - ALL_FILTERS                : trading_app.config
  - COST_SPECS                 : pipeline.cost_model (unused here; cited for completeness)
"""
```

### Phases inside the script

1. **Load candidate inventory** (~5 sec): read JSON + YAML + MCP payloads → ~837 strategy_ids with filter metadata.
2. **Bulk Sharpe join** (~10 sec): single query against `validated_setups` returning `sharpe_ratio, sample_size, last_trade_day` for the 837 ids. Out-of-set rows flagged `NO_VALIDATED_SETUPS_ROW`.
3. **Bulk Mode A recompute** (~60-180 sec): one SQL pass per unique filter_type via `filter_signal`. Returns `(strategy_id, n_is_mode_a, mode_a_expr, mode_a_std, n_oos, oos_expr)`. NULL `pnl_r` excluded.
4. **Per-row gate application** (~5 sec): apply Crit 1 (a-g), Crit 2 via `one_sample_power`, Crit 3 via `ALL_FILTERS`. Build CSV rows.
5. **Sort + write CSV**.

### Validation built into the script

- Pressure test: inject a known-bad row (synthetic strategy_id with `last_trade_day < 2020`) and confirm Crit 1(f) fires.
- Self-consistency: `mode_a_expr` for the 3 currently-Chordia-PASSED strategies must match the `validated_setups.expectancy_r` of those rows within ±0.005 (they were already Mode A pre-2026-04-08 audit).
- Print row count by `queue_tier` at exit. If 0 TOP and 0 READY, halt with explanatory message — do not write an empty CSV.

### Test discipline

Per `feedback_test_clock_injection.md`: any date references in script tests use `monkeypatch` for `HOLDOUT_SACRED_FROM`, not literal strings. The script itself reads the constant via import.

---

## Verification (reviewer cross-checks after the script runs)

```bash
# Worktree state
git -C C:/Users/joshd/canompx3/.worktrees/chordia-audit-queue-v2-2026-05-12 \
    log --oneline origin/main..HEAD

# 1. No allocator mutation
git diff origin/main -- docs/runtime/lane_allocation.json  # expect empty

# 2. No chordia log mutation
git diff origin/main -- docs/runtime/chordia_audit_log.yaml  # expect empty

# 3. No pre-reg yaml created
git diff origin/main -- docs/audit/hypotheses/  # expect empty

# 4. Production code touched ONLY by `research/chordia_queue_recompute.py`
git diff --stat origin/main -- pipeline/ trading_app/  # expect empty
git diff --stat origin/main -- research/  # expect ONE new file

# 5. Canonical delegations verified — these grep hits should all be ONE-LINER imports
grep -n "from research.oos_power import" research/chordia_queue_recompute.py
grep -n "from trading_app.chordia import" research/chordia_queue_recompute.py
grep -n "from research.filter_utils import" research/chordia_queue_recompute.py
grep -n "from trading_app.holdout_policy import HOLDOUT_SACRED_FROM" research/chordia_queue_recompute.py
grep -n "from trading_app.config import ALL_FILTERS" research/chordia_queue_recompute.py
grep -n "from pipeline.paths import" research/chordia_queue_recompute.py

# 6. No re-implementation of the canonical helpers
! grep -E "def _compute.*power|def _ttest|def _chordia_t|def _classify_filter" \
    research/chordia_queue_recompute.py

# 7. Self-consistency sanity check (run after CSV exists)
python -c "
import csv
canonical_passed = {
    'MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100',
    'MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15',
    'MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12',
}
with open('docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv') as f:
    rows = {r['strategy_id']: r for r in csv.DictReader(f)}
for sid in canonical_passed:
    r = rows.get(sid)
    assert r is not None, f'canonical PASS_CHORDIA strategy missing from CSV: {sid}'
    stored = float(r['expectancy_r_stored'] or 0)
    mode_a = float(r['mode_a_expr'] or 0)
    delta = abs(stored - mode_a)
    print(f'{sid}: stored={stored:.4f} mode_a={mode_a:.4f} delta={delta:.4f}')
    # These 3 strategies were Chordia-audited Mode A; delta should be tiny
    assert delta < 0.02, f'{sid}: Mode A drift >0.02R on a Chordia-audited canonical strategy'
"

# 8. Drift check
python pipeline/check_drift.py

# 9. Pooled-finding rule
grep '^pooled_finding: false' docs/audit/results/2026-05-12-chordia-audit-queue-methodology.md
```

---

## Anti-patterns this plan refuses

- No invented magnitude thresholds (the v1 0.02 replay-drift gate is gone).
- No n_oos-bucket OOS power proxy.
- No regex-based filter classification.
- No ranking sort that touches stored `validated_setups.expectancy_r` for Mode B grandfathered rows.
- No top-3 citation without `mcp__research-catalog__get_literature_excerpt` verbatim inline.
- No "we'll fix it in the next PR" placeholders for load-bearing gates.

---

## Risk tier + reasoning load

HIGH per user instruction. Escalate at:
- The bulk-SQL pass (one wrong WHERE clause poisons every downstream rank).
- The top-3 recommendation MD (the artifact a downstream pre-reg PR quotes verbatim).

Treadmill check: if Mode A recompute reveals >100 candidates with `stored_minus_mode_a_delta > 0.05R`, that's a class-bug in `validated_setups` itself, not a queue issue — surface and stop, propose a separate `validated_setups` revalidation PR before the queue lands.

---

## Pending PRs to be aware of (unchanged from v1)

- PR #271 `feat/allocator-pause-l1-2026-05-12` — allocator pause, docs.
- PR #272 `feat/sr-alarm-diagnosis-2026-05-12` — SR-alarm diagnostic MDs + the two source MDs this plan cites.

Both must land before the v2 candidate queue's "L1 slot reasoning" is current.

---

## Worktree + branch handover

The new worktree gets:
1. This plan file (`docs/plans/2026-05-12-chordia-audit-queue-v2-plan.md`) — same path, committed to v2 branch.
2. No code from the v1 worktree (the v1 worktree never committed any ad-hoc artifact; current `feat/chordia-audit-queue-2026-05-12` is `clean == origin/main`).
3. Branch off `origin/main`, not the v1 branch.

**v1 worktree disposition:** kept but left clean. Either fold into v2 by retargeting, or tear down after v2 lands. User decision.

---

## What "done" looks like

1. CSV exists with every row's gate evaluation traceable to a canonical source.
2. Top-3 recommendation MD contains verbatim literature excerpts inline.
3. `git diff origin/main` touches only `research/chordia_queue_recompute.py`, the 5 audit MDs, the CSV, the plan, and `HANDOFF.md`.
4. Self-consistency check passes (3 canonical Chordia-audited strategies within ±0.02R between stored and Mode A recompute).
5. `pipeline/check_drift.py` passes.
6. User reads top-3 recommendation MD and decides path A/B/C per v1 §"Gap-closure handoff".

This plan does not authorize any of those; it authorizes the work that produces them. User says GO → script runs → outputs reviewed → final commit lands.
