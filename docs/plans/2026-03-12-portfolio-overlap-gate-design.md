# Design: Portfolio Day-Overlap Gate for Promotion Candidates

**Date:** 2026-03-12
**Status:** Design — awaiting implementation green-light
**Triggered by:** Code review finding on commit cfdb516 (3 of 4 promoted specs had 95–99.9% trade-day overlap with existing portfolio specs)
**Blast radius:** 2 production files + 1 test file. No schema changes. No pipeline changes.

---

## Problem

`generate_promotion_candidates.py` surfaces strategies not yet in `LIVE_PORTFOLIO` using a
`covered` check on `(orb_label, entry_model, filter_type)` tuples. This correctly excludes
exact duplicates, but misses nested-subset relationships.

**Root cause — ORB_G filter nesting:**
```
G8 ⊆ G6 ⊆ G5 ⊆ G4   (G4 = ORB >= 4pts, fires most days)
```
At NYSE_OPEN (MNQ), nearly every day has an ORB > 4pts. So when ORB_G4 is already in the
portfolio, adding ORB_G6 or ORB_G8 fires on a 95–99.9% overlapping set of days. Different
`filter_type` value → passes the `covered` check → surfaces as a candidate → gets promoted
→ double-sizes the same trades.

**Concrete damage from cfdb516:**
- NYSE_OPEN ORB_G6 vs existing ORB_G4: **99.9% overlap** (1287/1288 days shared)
- NYSE_OPEN ORB_G5 vs existing ORB_G4: **95.2% overlap** (1226/1288 days shared)
- US_DATA_1000 ORB_G8 vs existing ORB_G5: **99.3% overlap** (1291/1300 days shared)
- Required immediate revert in commit 4d01288.

The SINGAPORE_OPEN VOL_RV12_N20 promotion was **correct** — different filter dimension,
genuinely independent day selection. The coverage check works when filter types are
orthogonal; it fails when they are nested subsets.

---

## Analysis: Right Approach

Three candidate fix layers were considered:

| Layer | Gate | Pros | Cons |
|-------|------|------|------|
| A. Promotion report | Before the PM makes the decision | Human sees overlap %, can override with reasoning | Only fires when using the report — doesn't catch manual live_config edits |
| B. live_config.py startup assertion | Catches everything at import | Defense-in-depth | Requires DB access at import — bad pattern for a config module |
| C. Drift check (full overlap) | Every commit | Automatic | orb_outcomes join is too slow for a pre-commit gate |

**Decision: A + D (report + lightweight drift check)**

- **Report** (`generate_promotion_candidates.py`): Full overlap computation using actual
  `orb_outcomes` trade days. Runs once at decision time. The correct primary gate.
- **Drift check** (`pipeline/check_drift.py`): Pure-Python G-filter nesting heuristic —
  no DB query. Detects the most common form of the problem (ORB_G overlaps) on every commit.
  Safety net for manual promotions that bypass the report.

Why NOT just the drift check alone: full overlap computation requires joining `orb_outcomes`
across all portfolio-spec pairs, which is too expensive for pre-commit. The report already
opens a DB connection, so the cost is zero marginal. The drift check gets the cheap fast
heuristic; the report gets the accurate computation.

Why NOT a `live_config.py` startup assertion: importing live_config in production code
(live trader, engine) must not spin up a DB connection. That pattern risks deadlocks under
concurrent writes and couples config loading to data availability.

---

## Overlap Metric Definition

**Directional overlap** — from the candidate's perspective:

```
overlap_pct = |candidate_days ∩ existing_days| / |candidate_days|
```

"What fraction of the candidate's firing days are already covered by an existing spec?"

If this is 99%, the candidate adds 1 day per 100 to the portfolio. Near-zero marginal value.

Also compute `marginal_days = |candidate_days \ ∪ all_existing_days_same_session_instrument|`
— absolute count of new days this candidate uniquely contributes.

**Thresholds:**

| overlap_pct | Verdict | Display |
|-------------|---------|---------|
| ≥ 80% | WARN — near-duplicate | Red badge / WARN prefix in terminal |
| 50–79% | NOTE — substantial overlap | Yellow badge / NOTE prefix |
| < 50% | OK — independent | No badge |

The 80% threshold is not arbitrary: at > 80% overlap, a candidate provides < 20% new
information at the portfolio level. With typical session volatility, this is below noise.

---

## Blast Radius

### Files Modified (production)

**1. `scripts/tools/generate_promotion_candidates.py`**
- New function: `load_portfolio_day_sets(con, candidates)` — bulk loads trade days for all
  LIVE_PORTFOLIO specs that share an `(instrument, orb_label)` with any candidate. One SQL
  query per `(instrument, orb_label)` pair (not per candidate). Returns:
  `dict[(instrument, orb_label, entry_model, filter_type), frozenset[str]]` (trading_day strings)
- Modify: `enrich_candidate(candidate, portfolio_day_sets)` — new optional arg. Adds
  `overlap_pct`, `overlap_with` (filter_type of worst overlap), `marginal_days` to candidate dict.
- Modify: `format_terminal` — add OVERLAP column to table. Prefix lines with `[WARN]`/`[NOTE]`.
- Modify: `generate_html` — add overlap badge in candidate card header.
- Modify: `main` — call `load_portfolio_day_sets` once, pass through `enrich_candidate`.

**2. `pipeline/check_drift.py`**
- New check (appended at end): `check_live_portfolio_g_filter_nesting`
- Pure Python — no DB query. Imports LIVE_PORTFOLIO from `trading_app.live_config`.
- For each `(instrument, orb_label, entry_model)` group in LIVE_PORTFOLIO, checks if the
  group contains multiple ORB_G filters (e.g., ORB_G4 + ORB_G6 for same session).
- G-filter ordering: parse integer N from `ORB_GN` — lower N = superset. Flag any group
  where a stricter G filter is present alongside a looser one (guaranteed ≥ 90% overlap
  by construction, no DB needed).
- Registered in `CHECKS` list. Runs on every commit.

### Files Modified (tests)

**3. `tests/test_generate_promotion_candidates.py`**
- Add tests for `load_portfolio_day_sets` and the overlap computation logic.
- Use mock `orb_outcomes` data with known overlaps to verify thresholds trigger correctly.

### Files NOT Changed

- `trading_app/live_config.py` — read-only reference
- `pipeline/init_db.py` / schema — no schema changes
- Any pipeline module — no data flow changes
- `gold.db` — read-only queries only

---

## Implementation Steps (ordered)

### Step 1 — `load_portfolio_day_sets`
New standalone function. Queries `orb_outcomes` (via `daily_features` join for filter
eligibility — same pattern used in portfolio PnL computation) for all LIVE_PORTFOLIO specs
that match any `(instrument, orb_label)` in the candidate list. Returns the day-set dict.

Query shape:
```sql
SELECT df.trading_day
FROM orb_outcomes oo
JOIN daily_features df ON oo.instrument = df.instrument AND oo.trading_day = df.trading_day
WHERE oo.instrument = ?
  AND oo.orb_label = ?
  AND df.<filter_column> = TRUE   -- from filter_type → column mapping
  AND oo.entry_model = ?
  AND oo.orb_minutes = ?
```

Filter-type to column mapping already exists in the codebase (used by outcome_builder).
Use the same mapping; do not reinvent.

### Step 2 — Enrich with overlap fields
Modify `enrich_candidate` signature: `enrich_candidate(candidate, portfolio_day_sets=None)`.
If `portfolio_day_sets` is None (e.g., in existing tests), skip overlap computation.
Compute directional overlap against each same-instrument+session portfolio spec, keep max.
Add: `overlap_pct`, `overlap_with`, `marginal_days` to candidate dict.

### Step 3 — Terminal output
Add column header `OVERLAP` (8 chars). For each candidate:
- ≥ 80%: prefix `[WARN]` in red (if terminal supports color) or plain if not
- 50–79%: prefix `[NOTE]`
- < 50%: no prefix

### Step 4 — HTML output
In the candidate card header, add a badge alongside the dollar gate badge:
- ≥ 80%: red badge `OVERLAP 99% vs ORB_G4 — WARN`
- 50–79%: yellow badge `OVERLAP 72% vs ORB_G6 — NOTE`
- < 50%: no badge (clean)

### Step 5 — Drift check
New check function `check_live_portfolio_g_filter_nesting`:
```python
def check_live_portfolio_g_filter_nesting(live_portfolio) -> list[str]:
    # Group by (instrument, orb_label, entry_model)
    # For each group, extract integer N from filter_type matching ORB_G[0-9]+
    # If group has both a lower N and a higher N → violation
    # Return violation strings, empty list = pass
```
This is O(n²) at most over ~30 specs. Trivial. No imports beyond what check_drift.py
already has.

### Step 6 — Tests
- `test_overlap_below_threshold`: candidate with 30% overlap → no badge
- `test_overlap_warn_threshold`: candidate with 99% overlap → WARN
- `test_load_portfolio_day_sets_returns_frozensets`: shape check on return value
- `test_g_filter_nesting_check_passes_clean_portfolio`: existing LIVE_PORTFOLIO passes
- `test_g_filter_nesting_check_fails_nested`: synthetic portfolio with G4+G6 → fail

---

## Open Questions — RESOLVED

**Q1: Filter-type → column mapping.**
`ALL_FILTERS` in `trading_app/config.py:646` is the canonical registry:
`dict[str, StrategyFilter]`. Each filter has `matches_row(row: dict, orb_label: str)`
operating on raw `daily_features` dict rows. No separate column-name map exists.

- `OrbSizeFilter` reads `row[f"orb_{orb_label}_size"]` — present in daily_features directly.
- `VolumeFilter` reads `row[f"rel_vol_{orb_label}"]` — NOT a daily_features column. It is
  pre-computed on-the-fly by `strategy_discovery._compute_relative_volumes`. Cannot be
  queried directly from the DB without replicating that computation.

**Implementation consequence:** Use `ALL_FILTERS[filter_type].matches_row()` on fetched
`daily_features` rows — same pattern as `strategy_discovery._build_filter_day_sets`.
For cross-class comparisons (e.g., ORB vs VOL portfolio spec vs candidate), skip the
overlap computation and output `overlap_pct=N/A (different filter class)`. Different
filter dimensions select on orthogonal criteria — high overlap is structurally unlikely.

**Q2: orb_minutes per LIVE_PORTFOLIO spec.**
`LiveStrategySpec` has no `orb_minutes` field — confirmed. `orb_minutes` is resolved at
runtime via the `family_rr_locks` join (live_config.py lines 243–256).

**Implementation consequence:** Use the **candidate's** `orb_minutes` when querying
`orb_outcomes` for both the candidate's day set and the portfolio spec's day set. This
answers "of the days this candidate fires (at its aperture), what fraction are days the
existing spec is also eligible?" — the directionally correct question, and consistent
with how `sample_size` is computed in `validated_setups`.

**Q3: Drift check import side effects.**
`check_drift.py` already imports `from trading_app.live_config import LIVE_MIN_EXPECTANCY_R,
LIVE_PORTFOLIO` lazily inside a check function at line 1900. The new G-filter nesting
check can reuse the same lazy import pattern inside its function body — zero additional
overhead. Confirmed: `live_config` import takes 0.464s with no DB connections opened.
The 0.464s is already paid by the existing check; the new check adds no marginal cost.

---

## What This Does NOT Fix

- Conceptual overlaps between non-G-filter types (e.g., two VOL filters at different
  thresholds). These are less likely to be nested subsets, but could be. The overlap
  computation (Step 1-2) handles them generically. Only the drift check is G-filter specific.
- Overlaps across different sessions or instruments — not a concern. The gate only applies
  within `(instrument, orb_label)` pairs.
- Promotions made by directly editing live_config.py without running the report. The drift
  check (Step 5) is the backstop for this case.

---

## Guardian Prompts

Not required. No schema changes, no entry model changes, no pipeline data flow changes.
This is a read-only reporting enhancement + a pure-Python drift check.

Run `python pipeline/check_drift.py` after implementing to verify the new check passes
on the current clean portfolio.
