# Canonical Data Redownload — Phase 2-5 Plan

**Date:** 2026-04-07
**Status:** AWAITING APPROVAL — Phase 0 (literature) done, Phase 1 (data audit) done, Phase 2 blocked on user decision
**Authority:** Extends `docs/institutional/` Phase 0 literature grounding

---

## Why this file exists

A prior session (Apr 7 02:26-14:37, log file `55cece77-3a8d-4e70-af54-d8986d13b273.jsonl`) performed a full data audit and found that our gold.db bars_1m table contains parent-futures data masquerading as micro-futures data for all three active ORB instruments. The session ended with a Phase 2-5 design proposal awaiting user approval, then was sidetracked by other work. This file captures that state so the next session can resume without re-doing the audit.

**The audit itself is trustworthy — era-split queries against gold.db produced numeric evidence, not narrative claims.**

---

## Phase 1 findings (already done)

### The data mixup

| Symbol | What's in bars_1m 2010-2024 | When real micro data starts | Missing years of real micro |
|---|---|---|---|
| MNQ | NQ parent (Nasdaq full-size, $20/pt) | 2024-02-05 | 4.75 yr (launch 2019-05-06 → our first real bar 2024-02-05) |
| MES | ES parent (S&P full-size, $50/pt) | 2024-02-12 | 4.75 yr (launch 2019-05-06 → our first real bar 2024-02-12) |
| MGC | GC parent (gold full-size 100oz, $100/pt) | NEVER | 100% of data is GC parent. MGC micro launched 2023-09-11. We have 0 months of real MGC data. |

Canonical cutoff: **`MNQ_CUT = DATE '2024-02-05'`** — documented in MEMORY.md.

### Per-lane era split for the 4 deployed MNQ lanes

Evidence that the deployed lanes' edge is NOT a data artifact — all 4 are positive in all 3 eras:

| Lane | NQ era (14 yr) | MNQ era (22 mo) | 2026 OOS (3.2 mo) | Verdict |
|---|---|---|---|---|
| SINGAPORE_OPEN COST_LT12 RR2.0 | N=453 ExpR +0.126 | N=253 ExpR +0.114 | N=60 ExpR +0.402 | Stable, hot 2026 |
| COMEX_SETTLE OVNRNG_100 RR1.5 | N=327 ExpR +0.106 | N=210 ExpR +0.343 | N=54 ExpR +0.145 | MNQ era 3x NQ era, fading 2026 |
| EUROPE_FLOW COST_LT10 RR3.0 | N=590 ExpR +0.098 | N=335 ExpR +0.176 | N=53 ExpR +0.349 | Stable growing |
| TOKYO_OPEN COST_LT10 RR2.0 | N=510 ExpR +0.088 | N=279 ExpR +0.139 | N=60 ExpR +0.287 | Stable growing |

Portfolio totals (4 MNQ lanes, excluding MGC):
- NQ era (168 months parent proxy): N=1880, total +194.22 R, ExpR +0.103, annualized +13.9 R/yr per copy
- MNQ era (22 months actual micro): N=1077, total +198.71 R, ExpR +0.185, annualized +108.4 R/yr per copy
- 2026 OOS (3.2 months): N=227, total +67.71 R, ExpR +0.298, annualized +254 R/yr per copy
- Pure MNQ era (IS+OOS, 25 months): N=1304, total +266.41 R, ExpR +0.204, annualized +128 R/yr per copy (**256 R/yr at 2 copies**)

### Why the deployed lanes are valid despite the mixup

All 4 deployed MNQ lanes use **price-based filters only**: ORB_G, COST_LT, OVNRNG. Per:
- Carver (*Systematic Trading*): proxy data valid for price-level signals where proxy-target correlation > 0.99 — NQ and MNQ track tick-for-tick identical prices.
- Lopez de Prado (*ML for Asset Managers*, Chapter 1): warns about composite representation creating phantom regime shifts — but era-split confirms no such shift for these lanes.
- Harvey & Liu (2015): proxy must be disclosed — our era-split provides that disclosure.

**Price signals transfer. Volume/liquidity/OI signals do not.** Volume-based filters (ORB_VOL_2K, ORB_VOL_8K) would be contaminated if they appeared in any deployed lane, but none do.

### Where the mixup DOES bite

1. **MGC lane** (MGC CME_REOPEN ORB_G6 RR2.5) — all 88 trades are from GC price data. MGC tbbo pilot showed real slippage 6.75 ticks vs 1 modeled. The signal may be price-valid but the cost model is wrong. **Treat as unvalidated until real MGC data is tested.**
2. **Volume-based rediscovery candidates** — any strategy with filter_type in (ORB_VOL_*, VOL_RV_*, liquidity-dependent) is contaminated by parent volume profile. Do not deploy.
3. **All rediscoveries post-2024-02-05** are clean. All rediscoveries pre-2024-02-05 need era-split disclosure.

---

## Phase 0 — Literature grounding (DONE 2026-04-07)

Completed in a separate session thread today. Six literature extracts plus framework, criteria, and template now live in `docs/institutional/`. See:
- `docs/institutional/README.md`
- `docs/institutional/HANDOFF.md`
- `docs/institutional/pre_registered_criteria.md` — 12 locked criteria
- `docs/institutional/literature/` — 7 verbatim extracts (Bailey 2013, Bailey-LdP 2014, LdP-Bailey 2018, Harvey-Liu 2015, Chordia 2018, Pepelyshev-Polunchenko 2015, LdP 2020)

These extracts establish the institutional standards any redownload-based rediscovery must meet: pre-registered hypothesis budget (≤ 300 trials on clean micro, ≤ 2000 trials on proxy), DSR > 0.95, Chordia t ≥ 3.00 (with theory) or 3.79 (without), WFE ≥ 0.50, 2026 OOS positive, era stability, Shiryaev-Roberts live monitoring.

---

## Phase 2 — Databento redownload (BLOCKED on user approval)

**Objective:** Pull real micro-futures 1m bars from each contract's launch date so we have a clean data set for rediscovery.

### Download targets

| Asset | Download from | Real usable from | Usable years for edge discovery |
|---|---|---|---|
| MNQ | 2019-05-06 (launch) | 2021-01 (liquidity mature) | ~5 years |
| MES | 2019-05-06 (launch) | 2021-01 (liquidity mature) | ~5 years |
| MGC | 2023-09-11 (launch) | 2024-06 (thin trading cleared) | ~22 months |

Carver Table 5 says Sharpe 0.5 detection needs ~20 years of data. Even the post-redownload MNQ/MES gives us ~5 years. That does not pass Carver's bar for new discoveries, but it does let us VALIDATE existing deployed lanes against clean data. MGC with 22 months is too thin for any new discovery — provisional sizing only, per `docs/institutional/finite_data_framework.md`.

### Pre-download requirements (MANDATORY)

1. **Cost verification** per `feedback_verify_costs.md` — run `pipeline.databento_client.get_cost()` for each of the 3 download windows BEFORE pulling anything. The prior session estimated ~$50 per 5-year 1m window but this was a heuristic, not a measured number.
2. **Data availability verification** — confirm Databento Standard has the requested date ranges, check for gaps, holidays, contract roll alignment.
3. **CME lead-month roll schedule** — get the authoritative roll schedule to match our existing continuous-contract stitching.
4. **User approval** of cost + scope before any download command runs.

### Download steps (if approved)

1. Pull 1m bars for each micro symbol from launch date to today, preserving `source_symbol` (MNQH5, MNQM5, etc.)
2. Validate: row count vs expected, timestamp monotonicity, price sanity, contract-roll continuity
3. Idempotent DB merge:
   - `DELETE FROM bars_1m WHERE symbol='MNQ' AND ts_utc BETWEEN <launch> AND <MNQ_CUT>`
   - `INSERT` new real MNQ data
   - Same for MES
   - For MGC: `DELETE FROM bars_1m WHERE symbol='MGC'` then `INSERT` real MGC data
4. Validate post-merge: bars_1m symbol count, source_symbol distribution, no duplicate keys

**Blast radius:** bars_1m table (read-write), nothing else in Phase 2. This is data acquisition only — no schema or pipeline changes.

---

## Phase 3 — Schema + pipeline era discipline (BLOCKED on Phase 2)

**⚠️ SCOPE CONFLICT WARNING:** Phase 3 touches `pipeline/build_daily_features.py` and `pipeline/check_drift.py`. Both files are currently in `e2-canonical-window-fix` worktree scope_lock. **Phase 3 cannot start until e2-fix merges.**

### 3a. Add data_era classification

- Derive from `bars_1m.source_symbol` — if source starts with 'MGC' → MICRO, 'GC' → PARENT. Same for MNQ/NQ, MES/ES.
- Add a view or computed column on `orb_outcomes` that joins through `bars_1m` and exposes era.

### 3b. Filter self-description extension

Add `requires_micro_data: bool` attribute to each `StrategyFilter` class in `trading_app/config.py`:
- ORB_G, COST_LT, OVNRNG → `False` (price-based, era-invariant)
- ORB_VOL, VOL_RV → `True` (volume-based, requires real micro)
- ATR filters → verify implementation; most are price-based
- OI-based filters (not currently deployed) → `True`

This hooks into drift check #85 (filter self-description coverage) which already exists.

### 3c. Rebuild orb_outcomes for new date ranges

For dates covered by the redownload:
- `DELETE FROM orb_outcomes WHERE trading_day >= <launch>` for each new instrument
- Rerun `pipeline/outcome_builder.py` (also in e2-fix scope_lock — **BLOCKED**)
- Rerun `pipeline/build_daily_features.py` (also in e2-fix scope_lock — **BLOCKED**)
- Validate row counts, no gaps

### 3d. Drift check — era discipline enforcement

Add a new drift check to `pipeline/check_drift.py` (**BLOCKED by e2-fix**) that:
- For every `validated_setups` row with a volume-based filter, verify `first_trade_day >= MICRO_START[instrument]`
- For every `validated_setups` row with a price-based filter, no gate (era-invariant)
- Fail-closed if any volume strategy references PARENT era trades

Test by injecting a violation (insert a fake validated_setup pointing to pre-micro data) and verify the check catches it.

### 3e. Stage-gate the Phase 3 sub-steps

Phase 3 exceeds the 3-file hook limit of stage-gate-protocol and must be split into sub-stages:
- Stage 3a: data_era classification (2 files)
- Stage 3b: filter self-description extension (1 file + tests)
- Stage 3c: orb_outcomes/daily_features rebuild (scripts only)
- Stage 3d: drift check addition (1 file + tests)

Each sub-stage needs its own `docs/runtime/stages/<slug>.md` file.

---

## Phase 4 — Clean rediscovery with pre-registered holdout (BLOCKED on Phase 3)

**Objective:** Run discovery on real micro data with 2026 held out, under the Phase 0 criteria.

### 4a. Write pre-registered hypothesis file FIRST

Per `docs/institutional/hypothesis_registry_template.md`, write `docs/audit/hypotheses/2026-<date>-mnq-post-redownload.yaml` BEFORE any backtest. Must include:
- Numbered hypotheses with economic theory citations
- Total trial count (≤ 300 for clean MNQ data)
- Kill criteria per hypothesis
- Holdout commitment (2026-01-01 as the sacred boundary)

### 4b. MNQ discovery

```bash
python -m trading_app.strategy_discovery \
  --instrument MNQ \
  --start 2019-05-06 \
  --holdout-date 2026-01-01 \
  --orb-minutes 5 \
  --hypothesis-file docs/audit/hypotheses/2026-<date>-mnq-post-redownload.yaml
```

**Note:** `strategy_discovery.py` does not currently accept a `--hypothesis-file` argument. Adding it is part of the "bigger Phase 4+" work from `docs/institutional/HANDOFF.md`. Will need a Phase 3.5 stage to wire this in.

### 4c. Same for MES

Same command with `--instrument MES` and `--start 2019-05-06`.

### 4d. MGC discovery — LOW POWER, flag accordingly

```bash
python -m trading_app.strategy_discovery \
  --instrument MGC \
  --start 2023-09-11 \
  --holdout-date 2026-01-01 \
  --orb-minutes 5 \
  --hypothesis-file ...
```

With ~22 months of usable data, MGC discovery will FAIL MinBTL bounds for any non-trivial hypothesis count. Expected outcome: very few or zero discoveries. Document honestly and do not deploy MGC lanes based on this run. May be usable only for provisional sizing of lanes that pass every other criterion except sample size.

### 4e. Apply the 12 criteria from `pre_registered_criteria.md`

Every discovered strategy must pass all 12 before promoting to `validated_setups`. No exceptions.

### 4f. Compare results to current deployed lanes

For each of the 5 currently-deployed lanes, check: does this lane survive the clean rediscovery? Is its post-redownload DSR > 0.95? Does it pass Chordia t ≥ 3.00?

Expected outcome (honest estimate from the prior audit): MNQ COMEX_SETTLE OVNRNG_100 likely survives (only lane currently passing Chordia at pre-redownload data). The other 3 MNQ lanes are marginal and may get demoted to provisional. MGC will fail or be insufficient sample.

---

## Phase 5 — Deploy decision (BLOCKED on Phase 4)

**Objective:** Commit to a clean, criteria-passing portfolio.

1. Lock in the cleaned top-N MNQ strategies — the prior session estimated 8-10 lanes feasible. Actual number depends on Phase 4 output.
2. Drop MGC from deployment until real MGC data accumulates to N ≥ 100 trades on actual micro.
3. Update `prop_profiles.ACCOUNT_PROFILES` with the new lane roster.
4. Start Shiryaev-Roberts drift monitors for each deployed strategy (Criterion 12).
5. Preserve the pre-redownload deployed set as a baseline for forward comparison — do NOT simply overwrite the active profiles.

---

## Artifacts currently on disk from the prior session

- `gold.db` — canonical (live writes)
- `gold_snap.db` (6.3 GB, untracked) — read-only snapshot for audit work, used by prior session
- `gold.db.pre-e2-fix.bak` (6.3 GB, untracked) — backup before e2-fix started
- `docs/runtime/baselines/2026-04-07-orb-window-snapshot.json` — pre-e2-fix baseline state
- `docs/runtime/baselines/2026-04-07-pre-e2-canonical-fix.json` — same
- Memory: `MEMORY.md` has `MNQ_CUT = DATE '2024-02-05'` in the "Key facts" section
- Memory: `databento_backfill_todo.md` is Apr 1 and stale (pre-dates this mixup audit)

---

## User approval gates (in order)

1. **Approve Phase 2 scope** — authorise the redownload windows for MNQ, MES, MGC
2. **Approve Phase 2 cost** — after `get_cost()` returns, confirm the dollar cost before any download
3. **Approve Phase 3 timing** — wait for e2-fix merge OR coordinate Phase 3 work into the e2-fix worktree
4. **Approve Phase 4 hypothesis file** — review the pre-registered hypothesis YAML before discovery runs
5. **Approve Phase 5 deployment changes** — review the new lane roster before touching prop_profiles

No gate can be skipped. Phase 2-5 are ALL institutional-rigor territory.

---

## Related docs

- `docs/institutional/HANDOFF.md` — Phase 0 literature status
- `docs/institutional/pre_registered_criteria.md` — the 12 locked criteria (locked 2026-04-07)
- `docs/institutional/finite_data_framework.md` — methodology for short-sample discovery
- `docs/plans/2026-04-02-16yr-pipeline-rebuild.md` — prior rebuild plan, now superseded by this file for the data layer
- `.claude/rules/research-truth-protocol.md` — Phase 0 grounding rules (updated 2026-04-07)
- `.claude/rules/institutional-rigor.md` — canonical-source and self-review rules (updated 2026-04-07)
- Memory: `era_research_caveat.md`, `era_contamination_trap.md` — prior notes on parent-vs-micro issue

## Session reference

Prior audit session log: `~/.claude/projects/C--Users-joshd-canompx3/55cece77-3a8d-4e70-af54-d8986d13b273.jsonl` (Apr 7, 02:26-14:37). Do not rely on it — this plan file is the canonical summary.
