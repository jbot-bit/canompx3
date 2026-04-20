# Live Book Truth-Status Re-Audit v2

**Date:** 2026-04-20  
**Branch:** `codex/live-book-reaudit`  
**Scope:** Re-audit the diagnosis that the current live prop book is primarily constrained by under-deployment rather than discovery / truth-state. This pass explicitly audits the prior audit for overreach.

**Same-day correction:** fresh verification showed `trading_app/prop_portfolio.py` is healthy, not broken. Phase 1 truth-surface repair also updated `TRADING_RULES.md`, `trading_app/prop_profiles.py`, and `RESEARCH_RULES.md` to remove stale live-book wording. The scale-readiness verdict in this audit does not change.

## Evidence Rules

Trusted, in order:
1. Current code / current files
2. Direct command output
3. Canonical tables: `orb_outcomes`, `daily_features`, `bars_1m`, `paper_trades`
4. Verbatim local resource text
5. Docs for intent / policy only when direct proof is unavailable

Not trusted as proof:
- extract-file metadata
- criticality labels
- project-written "application to our project" sections in literature extracts
- memory files
- prior summaries

## Pre-Flight

Directly observed:
- `orb_outcomes` MNQ max trading day = `2026-04-16`
- `daily_features` MNQ max trading day = `2026-04-17`
- `bars_1m` MNQ max ts = `2026-04-17 09:59:00+10:00`
- canonical schemas for `orb_outcomes` and `daily_features` present

No pre-flight halt triggered.

## Pass 1 — Truth Check

### Claim table

| # | Claim | Direct proof | Truth-check | Audit label |
|---|---|---|---|---|
| 1 | Active live book is `topstep_50k_mnq_auto` with 6 allocator-driven MNQ lanes | `prop_profiles.py` + `lane_allocation.json` | `VALID` | `VALID` |
| 2 | Book is under-deployed | `copies=2`, comment says scale after proving loop | `CONDITIONAL` | `ALIVE` |
| 3 | Scale is the highest-confidence lever | No proving-loop definition; doctrine says no scaling until re-audited under Mode A | `WRONG` | `MISCLASSIFIED` |
| 4 | Missing current-lane realized attribution is a major live-truth gap | `paper_trades` has `0` rows for current 6 strategy_ids | `VALID` | `VALID` |
| 5 | At audit time, truth-surface failures were stale doctrine / profile notes, not a live `prop_portfolio.py` runtime break | `py_compile` passes; CLI loads; tests pass; `TRADING_RULES.md` + `prop_profiles.py` notes were stale vs code/JSON | `VALID` | `VALID` |
| 6 | Routine-day MNQ slippage is not proven to be the primary bottleneck | slippage evidence partial, coverage incomplete | `CONDITIONAL` | `ALIVE` |
| 7 | Correlation control is not ready for larger scale | `corr_lookup` empty; doc explicitly blocks scaling without it | `VALID` | `VALID` |
| 8 | Discovery is not the main issue | canon says current shelf is research-provisional under Mode A | `WRONG` | `MISCLASSIFIED` |

### Proof excerpts

#### Claim 1 — current live book

Direct code:
- `topstep_50k_mnq_auto` has `copies=2`, `stop_multiplier=0.75`, `allowed_instruments={"MNQ"}`, dynamic lanes from `lane_allocation.json`.
- Current allocation JSON contains 6 MNQ lanes:
  - `EUROPE_FLOW`
  - `SINGAPORE_OPEN`
  - `COMEX_SETTLE`
  - `NYSE_OPEN`
  - `TOKYO_OPEN`
  - `US_DATA_1000`

#### Claim 3 — scale is the highest-confidence lever

Direct counterproof:
- `prop_profiles.py` says: `copies=2  # Start with 1-2 Express, scale to 5 after proving loop`
- No explicit proving-loop or `scale_ready` gate found in current code.
- Institutional doctrine says current deployed lanes are research-provisional and:
  - `No scaling until re-audited under Mode A.`

This kills "highest-confidence" as a valid description. Arithmetic upside exists. Decision-grade readiness does not.

#### Claim 4 — missing live attribution

Direct query result on current 6 strategy IDs:
- `paper_trade_rows = 0`
- `pnl_rows = 0`

This is not a subtle gap. The current live book has no realized lane-level execution journal for the active 6.

#### Claim 5 — control-plane truth-surface drift

Fresh runtime verification:

```text
python3 -m py_compile trading_app/prop_portfolio.py            -> PASS
/mnt/c/.../.venv-wsl/bin/python -m trading_app.prop_portfolio --help -> PASS
/mnt/c/.../.venv-wsl/bin/python -m pytest tests/test_trading_app/test_prop_portfolio.py -q -> 46 passed
```

At audit time, direct doctrine mismatch was:
- `TRADING_RULES.md` described a stale 5-lane MNQ+MGC active book.
- Current code / JSON showed a 6-lane MNQ-only live book.
- `prop_profiles.py` active-profile notes claimed a "7-lane" allocator book while the current allocation JSON had 6 lanes.

Status after Phase 1 repair:
- `TRADING_RULES.md` now matches the current allocator-managed 6-lane MNQ live book.
- `prop_profiles.py` no longer hardcodes a stale live lane count in the active profile notes.
- `RESEARCH_RULES.md` no longer uses a stale hardcoded deployed-lane count in current-language prose.

#### Claim 6 — slippage not primary bottleneck

Direct evidence says only this:
- MNQ TBBO pilot found routine-day median `0` ticks, p95 `0.35`, max `+2`.
- Still open:
  - missing deployed-session coverage: `EUROPE_FLOW`, `COMEX_SETTLE`, `US_DATA_1000`
  - event-tail MNQ not measured
  - MES pilot not run

That is partial closure, not full closure.

#### Claim 8 — discovery is not the main issue

Direct policy truth:
- `RESEARCH_RULES.md` says 2026 holdout is sacred.
- Existing `validated_setups` are research-provisional after the brief Mode B contamination.
- `pre_registered_criteria.md` says:
  - deployed lanes remain research-provisional
  - no scaling until re-audited under Mode A
  - path to proof includes clean rediscovery under `--holdout-date 2026-01-01`

This kills the blanket anti-discovery framing. Broad brute-force discovery is not the move. Clean rediscovery of active families still is.

## Pass 2 — De-Tunnel

Alternative framings that the diagnosis under-tested:

| Framing | Was it actually tested? | Status |
|---|---|---|
| Standalone: "is the current live book evidence-clean enough for capital decisions?" | No | Missed |
| Filter: "use this as a block on scaling until truth-state is fixed" | No | Missed |
| Conditioner: "allocator stats are provisional priors only" | No | Missed |
| Allocator: "is the current selected book itself wrong?" | No | Missed |
| Confluence: "need rediscovery + attribution + slippage closure + correlation control together" | No | Missed |

Conclusion:
- The diagnosis was tunnel-visioned toward throughput.
- It under-tested truth-state, governance, and book-selection questions.

## Pass 3 — Reframe

### What was actually tested?
- live profile identity
- current lane set
- current lane absence of realized attribution
- current operator breakage
- partial slippage evidence
- current research-provisional status under doctrine

### What was not actually tested?
- whether the current book is the right book
- whether current live mechanism matches modeled expectancy
- whether any proving-loop / scale-ready criterion exists and is met
- whether current families survive clean Mode A rediscovery

### Better questions

1. What is operationally true, what is modeled, and what is cleanly proven?
2. What must be true before `copies 2→5` becomes admissible rather than merely attractive?

Highest-EV question:
- What evidence is missing before the current live book is **scale-ready under canon**?

## Pass 4 — Brutal Filter

### What looks good but isn't?
- `annual_r_sum = 199.2` and `2 copies = 398.4R` look impressive. They are modeled allocator arithmetic, not realized current-book proof.
- `all 6 are FIT` sounds stronger than it is. `FIT` is a rolling regime-monitoring label, not clean validation.
- `deployable` status sounds stronger than it is. Canon says research-provisional + operationally deployable is not institutional proof.

### What is fragile / overfit / noise-sensitive?
- Any conclusion that leans on current allocator winners as if they are cleanly selected from an uncontaminated research process.
- Any conclusion that ignores `n_trials_at_discovery` in the mid-35k range and the Mode A contamination history.
- Any scale thesis that assumes routine-day slippage closure is enough.

### What would a prop desk reject?
- zero current-lane realized attribution
- stale operator truth surfaces
- stale doctrine surface
- empty correlation guard before scale
- undefined proving-loop
- research-provisional evidence sold as high-confidence scale proof

## Pass 5 — Edge Extractor

No new alpha edge was proven by this audit.

What survives:
- an **operational edge**: prevent a false-positive scale decision by forcing truth-state repair first

Where it lives:
- truth-surface repair
- live attribution restoration
- scale-ready governance
- clean Mode A rediscovery of active families

How it should be used:
- `conditioner`
- `governance gate`
- **not** standalone alpha

## Final Classification

### Survives
- current live-book identity
- major live-attribution gap
- scaling blocked by missing correlation controls

### Conditional only
- under-deployment as arithmetic opportunity
- routine-day slippage likely not primary bottleneck

### Killed / downgraded
- "scale is the highest-confidence lever"
- "discovery is not the main issue"
- "best next move is scale + attribution + parity, not discovery"

## What changed from the first audit

The first audit was directionally right but still gave the scale thesis too much oxygen.

This re-audit tightens the line:
- under-deployment is real but not decision-grade
- attribution is a major gap but not the only governing gap
- clean Mode A rediscovery was ruled out too aggressively in the first pass
- fresh verification also corrected one stale claim inside the re-audit itself: `prop_portfolio.py` is currently healthy

## Bottom line

The diagnosis contains real operational facts, but the governing conclusion is wrong.

The live book is not primarily blocked by "lack of one more setup." It is blocked by **truth-state debt**:
- research-provisional shelf provenance
- absent current-lane attribution
- incomplete cost closure
- missing scale-ready governance

Until those are fixed, scale is not a high-confidence action.

## Current Status After Phase 1 Repair

- `TRADING_RULES.md` live-book section repaired to current allocator truth
- `trading_app/prop_profiles.py` active-profile notes made lane-count-agnostic
- `RESEARCH_RULES.md` current-language deployed-lane wording generalized
- `prop_portfolio.py` verified healthy: compile pass, CLI load pass, targeted tests pass

Current blockers that still survive:
- research-provisional provenance under restored Mode A
- zero current-lane realized attribution in `paper_trades`
- unresolved scale-ready governance
- incomplete cost / risk closure before scale

## Current Status After Phase 2 Instrumentation Repair

Phase 2 is now split honestly into:
- implemented instrumentation surfaces
- still-missing live evidence

### Implemented and verified

- `trading_app/paper_trade_store.py`
  - single write surface for `paper_trades`
  - real execution rows replace modeled rows for the same `(strategy_id, trading_day)`
  - backfill rows cannot overwrite `live` / `shadow` rows
  - helper now bootstraps the full `paper_trades` schema it needs directly instead of assuming a pre-migrated DB
- `trading_app/paper_trade_logger.py`
  - backfill now deletes only modeled rows and routes inserts through the shared store helper
- `trading_app/log_trade.py`
  - manual live logging now routes through the same write helper as automated attribution
- `trading_app/live/trade_journal.py`
  - added durable `live_signal_events` table
  - added `record_signal_event(...)` API for skip / reject / submitted / filled states
- `trading_app/live/session_orchestrator.py`
  - profile-backed exits bridge completed live/shadow trades into `paper_trades`
  - durable signal-event writes now fire on:
    - orphan / paused entry blocks
    - ORB cap skips
    - max-risk skips
    - DD halt blocks
    - regime-paused blocks
    - duplicate entry rejects
    - entry submit failures
    - entry submitted / filled
    - fill-poller filled / cancelled / rejected outcomes
    - explicit `REJECT` events
  - bridge is fail-open and profile-aware: non-profile/test portfolios do not write synthetic `paper_trades` rows
- `scripts/tools/live_attribution_report.py`
  - read-only current-book report for the current allocated strategy IDs
  - merges:
    - allocator snapshot (`lane_allocation.json`)
    - modeled comparator (`validated_setups`)
    - completed rows (`paper_trades`)
    - event rows (`live_signal_events`)
  - labels modeled priors as comparator-only, not proof
- `docs/audit/hypotheses/2026-04-20-first-live-mechanism-audit.yaml`
  - locked first mechanism-audit pre-reg for the new report surface
- `trading_app/pre_session_check.py`
  - canonical pre-session operator surface now warns when the current session lane(s) still have no live attribution evidence

### Verification evidence

Commands run in this worktree:

```text
python3 -m py_compile trading_app/paper_trade_store.py trading_app/paper_trade_logger.py trading_app/log_trade.py trading_app/live/trade_journal.py trading_app/live/session_orchestrator.py tests/test_trading_app/test_paper_trade_store.py tests/test_trading_app/test_trade_journal.py tests/test_trading_app/test_session_orchestrator.py
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m pytest tests/test_trading_app/test_paper_trade_store.py tests/test_trading_app/test_trade_journal.py tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_paper_trade_logger.py -q
-> 170 passed
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m pytest tests/test_trading_app/test_paper_trade_store.py tests/test_trading_app/test_trade_journal.py tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_paper_trade_logger.py tests/test_trading_app/test_prop_profiles.py tests/test_trading_app/test_prop_portfolio.py -q
-> 277 passed
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m ruff check trading_app/paper_trade_store.py trading_app/paper_trade_logger.py trading_app/log_trade.py trading_app/live/trade_journal.py trading_app/live/session_orchestrator.py tests/test_trading_app/test_paper_trade_store.py tests/test_trading_app/test_trade_journal.py tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_paper_trade_logger.py tests/test_trading_app/test_prop_profiles.py tests/test_trading_app/test_prop_portfolio.py
-> All checks passed
python3 -m py_compile scripts/tools/live_attribution_report.py tests/test_tools/test_live_attribution_report.py
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m pytest tests/test_tools/test_live_attribution_report.py -q
-> 2 passed
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m ruff check scripts/tools/live_attribution_report.py tests/test_tools/test_live_attribution_report.py
-> All checks passed
python3 -m py_compile trading_app/pre_session_check.py tests/test_trading_app/test_pre_session_check.py
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m pytest tests/test_trading_app/test_pre_session_check.py -q
-> 33 passed
/mnt/c/Users/joshd/canompx3/.venv-wsl/bin/python -m ruff check trading_app/pre_session_check.py tests/test_trading_app/test_pre_session_check.py
-> All checks passed
```

### What is still NOT proven

- Phase 2 gate is **not** satisfied yet.
- Current 6 strategy IDs still do not have realized rows in `paper_trades`.
- What changed is that the path for those rows is now implemented and guarded.
- First real runtime evidence is still required before any mechanism audit or Phase 3 scale-ready discussion.
- Canonical operator surface now makes that absence visible before session start:
  - `pre_session_check --session NYSE_OPEN --profile topstep_50k_mnq_auto`
  - emits `WARN: no live attribution evidence yet for current lane(s): NYSE_OPEN`
