---
task: Resolve 3 deferred compound filters from external trading AI framework — pending predicate disambiguation + schema work
mode: DESIGN
slug: filter-library-deferred-3-disambiguation-2026-05-12
created: 2026-05-12
updated: 2026-05-12
scope_lock:
  - docs/audit/hypotheses/<future external AI clarification>
  - pipeline/build_daily_features.py (potential — for OVN_DRIFT only)
  - trading_app/config.py (potential — for the 3 deferred filter classes)
acceptance:
  - "External trading AI clarifies predicate-level definitions for CLEAN_AIR_R125_ATR035 and OPEN_VS_PDMID_LONG/SHORT (current-price-at-decision-time semantics)."
  - "Decision on OVN_DRIFT_LT_06ATR: extend build_daily_features.py with overnight_open/overnight_close, or compute drift via available columns, or drop the filter."
  - "Each clarified filter gets its own implementation stage (separate from this disambiguation stage)."
---

## Blast Radius

- **No code change in this stage.** Decision-only.
- If clarifications resolve to schema extensions: blast radius expands to `pipeline/build_daily_features.py` (add columns) + `daily_features` rebuild + drift checks.
- If clarifications resolve to pre-ORB-only predicates: blast radius is filter-class-only (similar to the 2026-05-12 band-extension stage).
- If a filter resolves to "uses post-entry data on E2" → that filter is rejected on look-ahead grounds (per RULE 1.2 of backtesting-methodology.md) and dropped from the framework.

## Why this is a stage, not silent omission

The external trading AI's framework specified 5 compound filters. The 2026-05-12 band-extension stage shipped the 2 that were predicate-clean and data-available. The remaining 3 have unresolved ambiguities documented here so the next session does not silently lose them or backfill assumptions.

## The 3 deferred filters

### Filter 1: `CLEAN_AIR_R125_ATR035`

**Spec (external AI):** distance from "current price" to nearest opposing reference level (prev_day_high, prev_day_low, overnight_high, overnight_low) >= 1.25 × ATR(20), normalized to 0.035 of ATR clearance.

**Ambiguity:** "current price" is not defined at decision time.

| If "current price" means | Status | Implementation |
|---|---|---|
| Session open bar | ⚠️ Need to define `session_open_price` in daily_features (which bar?) — does not exist. | New column + filter class |
| ORB midpoint at ORB end | ✅ Computable from `orb_{label}_high/low` at ORB-end before entry | Filter class only |
| Break price (entry price) | ❌ **E2 look-ahead** — entry bar timing is post-decision for stop-market entry; reject | N/A |
| Pre-ORB last close | ⚠️ Need to define which bar — does not exist as a feature | New column + filter class |

**Action needed from external AI:** "When does 'current price' resolve, and from which canonical timestamp?"

### Filter 2: `OPEN_VS_PDMID_LONG` / `OPEN_VS_PDMID_SHORT`

**Spec (external AI):** session open is above (LONG) or below (SHORT) prior-day midpoint = (prev_day_high + prev_day_low) / 2.

**Ambiguity:** "session open" is not defined.

| If "session open" means | Status | Implementation |
|---|---|---|
| Exchange session open bar (e.g. MGC London opens 17:00 Brisbane) | ⚠️ This is a 1m bar timestamp — `bars_1m` has it but it's not surfaced in `daily_features` | New column + filter class |
| First bar of ORB | ⚠️ Same — surfaceable from `bars_1m` | New column + filter class |
| `gap_open_points` reference (open vs prior close) | ✅ Column exists | Filter class only |

`prev_day_mid` is derivable inline (`(prev_day_high + prev_day_low) / 2`) — both halves of prev_day_mid columns exist.

**Action needed from external AI:** "Define session open canonically. Is it the first 1m bar of the exchange session, the first bar of the ORB window, or something else?"

### Filter 3: `OVN_DRIFT_LT_06ATR`

**Spec (external AI):** absolute value of overnight drift (close - open over the overnight window) < 0.6 × ATR(20).

**Hard gap:** `daily_features` has `overnight_high`, `overnight_low`, `overnight_range` — but NOT `overnight_open` or `overnight_close`.

**Options:**

| Option | Cost | Blast radius |
|---|---|---|
| A — Extend `build_daily_features.py` to compute `overnight_open` + `overnight_close` from `bars_1m`, full rebuild of daily_features | Highest — schema change + rebuild + drift checks | Pipeline + schema |
| B — Proxy: use `overnight_range` as a magnitude proxy (rejects sign info — directional asymmetry lost) | Moderate — works only if the mechanism is volatility-magnitude, not directional drift | Filter class only |
| C — Drop the filter from the framework — accept that this predicate is not feasible without the schema work | Zero | None |

**Recommended option:** A, but only if either (i) the external AI's mechanism requires sign info, OR (ii) other future research justifies the schema cost. Otherwise drop.

**Action needed from external AI:** "Does the mechanism require directional drift sign, or is magnitude sufficient? If magnitude — what's the difference vs already-existing `overnight_range_pct`?"

## Procedure

1. Send the 3 disambiguation questions back to the external trading AI.
2. Based on answers, classify each filter as:
   - **Code-only filter** → file separate implementation stage modeled on `filter-library-band-extension-2026-05-12.md`
   - **Schema-extending filter** → file a `pipeline/build_daily_features.py` extension stage (full blast radius — rebuild + drift)
   - **Rejected** → document why, drop from framework
3. For each filter that survives disambiguation, write its prereg targeting the canonical lane (MGC LONDON_METALS for CLEAN_AIR + OPEN_VS_PDMID; instrument/session TBD for OVN_DRIFT).

## Cross-references

- Sibling stage: `docs/runtime/stages/filter-library-band-extension-2026-05-12.md` (the 2 filters shipped 2026-05-12)
- Framework origin: external trading AI message (3-lane MGC/MNQ/MNQ framework), captured in prior session resume note
- Authority: `.claude/rules/backtesting-methodology.md` RULE 1 (feature temporal alignment — bans use of post-entry data on E2)
- Authority: `.claude/rules/institutional-rigor.md` § 6 (no silent failures — explicit disambiguation, not assumed predicate)
