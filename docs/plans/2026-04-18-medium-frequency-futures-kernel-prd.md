# Medium-Frequency Futures Kernel PRD

**Date:** 2026-04-18
**Status:** ACTIVE DESIGN + PHASE 4 RESEARCH HARNESS IMPLEMENTED
**Purpose:** define the correct architecture for a research-only,
literature-grounded medium-frequency futures engine that reuses the repo's
control plane without contaminating the new alpha plane with ORB-specific
runtime assumptions.

## 1. Executive decision

Build a **separate research-only medium-frequency futures kernel** for
**trend + futures carry**.

Do **not** try to turn the current ORB runtime into this engine.

The correct split is:

- **Reuse the control plane**
  - `trading_app/prop_profiles.py`
  - `trading_app/risk_manager.py`
  - `trading_app/sr_monitor.py`
  - broker adapters / contract metadata / deployment governance
- **Build the alpha plane fresh**
  - continuous-contract research surfaces
  - daily settlement feature store
  - EWMAC + futures-carry forecast engine
  - target-position engine
  - research-only execution-intent model

This subsystem is **not greenfield**, but it is also **not a small patch**.

## 2. Why this boundary is correct

### 2.1 What already exists

The repo already contains:

- strong live-control and prop-firm governance surfaces
- a Criterion 12 Shiryaev-Roberts monitor
- allocator / sizing doctrine influenced by Carver
- ORB-specific execution, portfolio, and lane-allocation machinery

### 2.2 What does not exist

The repo does **not** currently have:

- a futures basis / roll-yield carry pipeline
- a daily-signal forecast store for medium-frequency research
- a continuous-contract research layer separated from tradable-chain pricing
- a production-grade forecast combiner module
- a medium-frequency position engine independent of ORB session logic

### 2.3 Hard architecture rule

The new kernel must **not** depend on:

- ORB formation objects
- ORB outcome schemas
- ORB lane allocation as its core decision model
- intraday breakout-specific execution state machines

It may later publish artifacts that a shared deployment/control plane can
consume.

## 3. Literature grounding

### 3.1 Local repo-grounded literature

- `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`
  - volatility targeting
  - cash-vol target framing
  - forecast scaling
  - half-Kelly / bounded risk discipline
- `docs/institutional/literature/chan_2008_ch7_regime_switching.md`
  - current-state classification is valid
  - transition-probability storytelling is not useful for deployment
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`
  - live change-detection monitoring

### 3.2 External evidence already verified in this session

- AQR trend-following / time-series momentum evidence across broad asset sets:
  - <https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-Trends-Everywhere_JOIM.pdf?sc_lang=en>
- CME liquidity and contract practicality:
  - <https://www.cmegroup.com/media-room/press-releases/2026/4/02/cme_group_reachesall-timerecordmonthlyandquarterlyaveragedailyvo.html>
  - <https://www.cmegroup.com/markets/microsuite/fx.html>

### 3.3 Honest grounding limit

There is **not yet** a local dedicated literature extract for:

- futures carry / roll-yield theory
- continuous-contract construction methodology

So the new kernel may implement those concepts as **research objects**, but
must not overclaim them as fully literature-grounded canon until the local
extracts exist.

## 4. Scope

### 4.1 V1 in scope

- research-only kernel
- daily/settlement-clock alpha
- trend sleeve
- futures carry sleeve
- combined forecast
- target position engine
- cost / turnover / inertia-aware research output
- anchored non-overlapping walk-forward summaries on the supported surface
- unit-tested pure functions

### 4.2 V1 explicitly out of scope

- live broker order submission
- intraday smart execution logic
- passive-join execution alpha
- ORB strategy integration
- real-time portfolio automation
- profile-aware live contract routing

## 5. Research universe

### 5.1 Research universe

- Equity index: `ES`, `NQ`
- Rates: `ZN`, `ZB`
- Metals: `GC`
- FX: `6E`, `6J`

### 5.2 Live incubation target (future)

- `MES`, `MNQ`, `MGC`, `M6E`

### 5.3 Deferred

- `CL` is Phase 2 only after the kernel is stable and the first asset groups
  are behaving honestly.

### 5.4 Current canonical coverage reality

Phase 1 must respect the current repo data surface, not the aspirational
universe:

- `bars_1m` / `daily_features` currently cover: `GC`, `MES`, `MNQ`, `MGC`,
  `M6E` (plus unrelated ORB research symbols)
- `exchange_statistics` currently covers only: `MES`, `MNQ`, `MGC`

Implications:

- trend snapshots can be assembled now for the covered symbols
- front-contract settlement / OI / volume can be attached only where
  `exchange_statistics` exists
- true futures carry remains **not fully available** because there is no
  current next-contract price surface in canonical storage

Phase 1 therefore exposes missing carry coverage explicitly instead of
manufacturing a synthetic next-contract input.

## 6. Alpha design

### 6.1 Signal clock

All alpha signals are **daily / settlement-clock**.

Minute data may later be used for:

- order placement research
- spread/slippage estimation
- health checks

Minute data must **not** become an alpha input in this subsystem unless
explicitly promoted by a separate design.

### 6.2 Sleeve A — Trend

Use EWMAC variants:

- `16 / 64`
- `32 / 128`
- `64 / 256`

Rules:

- compute on a research price series that is stable across rolls
- normalize forecast by recent realized volatility
- combine sleeve members by fixed weights
- cap the sleeve output

### 6.3 Sleeve B — Carry

Use futures carry / roll yield from the **front vs next tradable contracts**
observed on the same research day.

Rules:

- long positive carry
- short negative carry
- annualize by the difference in source-backed expiry dates
- smooth over fixed windows only
- do **not** reuse ORB “prior-session carry” semantics here

### 6.4 Sleeve C — not in v1

Short-horizon reversal remains out of scope.

## 7. No-lookahead rules

1. **Signal inputs must be fully known at the decision timestamp.**
2. **Back-adjusted series may be used for signal continuity, but not for fill/PnL truth.**
3. **Tradable-chain prices must remain separate from research-chain prices.**
4. **Roll decisions must be rule-based and time-consistent.**
5. **Carry must use same-day observed curve points, not future roll outcomes.**
6. **Any smoothing window must be backward-looking only.**
7. **Execution-intent generation must use only current target vs previous target.**

## 8. Data model

The kernel needs two parallel price surfaces:

### 8.1 Research series

Purpose:

- compute stable trend signals across contract rolls

Requirements:

- back-adjusted or otherwise research-stable
- no use for fills or broker reconciliation

### 8.2 Tradable chain

Purpose:

- compute carry
- compute actual notional risk
- drive future execution intents

Requirements:

- explicit front contract
- explicit next contract
- explicit source-backed expiries
- observed settlement / close values

### 8.3 Phase 1 implementation note

Phase 1 implements:

- `DailyMarketSnapshot`
- `KernelInputSlice`
- canonical loader from `daily_features`
- optional same-day front-contract attachment from `exchange_statistics`
- backward-looking realized-vol adapter into the pure kernel functions

Phase 1 does **not** yet implement:

- next-contract loading
- full back-adjusted chain construction beyond the repo's existing logical
  symbol series

### 8.4 Phase 2 implementation note

Phase 2 now implements:

- canonical raw-statistics contract observations
- deterministic same-day front/next pairing
- honest carry input slices with explicit unavailable reason codes
- contract-gap-in-months metadata from canonical contract symbols

Phase 2 still does **not** implement:

- synthetic expiry dates
- annualized carry when expiry dates are absent
- any live routing or broker integration

Current honest outcome:

- `MNQ` / `MGC` can produce front price, next price, spread, ratio, and
  contract-gap metadata from canonical raw statistics
- annualized carry still fails closed on `missing_expiry_date`
- `M6E` still fails closed on `missing_raw_statistics`

### 8.5 Phase 3 implementation note

Phase 3 now implements:

- source-backed expiry dates for currently covered stats families:
  - `MES`
  - `MNQ`
  - `MGC`
- `next_contract` semantics corrected to mean the nearest later listed contract
  by expiry ordering, not the most-liquid farther deferred contract
- annualized carry where those expiries are available and front / next
  settlements are both observed

Phase 3 still does **not** implement:

- synthetic expiry dates
- holiday contingency logic beyond the exact official rule shape already encoded
- rates / FX annualized carry where raw statistics are still absent

Current honest outcome:

- `MGC` annualized carry is now available where front / next observations exist
- `MES` / `MNQ` expiry dates are now source-backed and usable for annualization
  when paired observations exist
- `M6E`, `6J`, `ZN`, and `ZB` remain blocked by missing raw stats, not by
  missing carry math

### 8.6 Phase 4 implementation note

Phase 4 now implements:

- a research-only daily simulation harness on the supported surface only:
  - `MES`
  - `MNQ`
  - `MGC`
- fixed-rule forecast generation from the existing trend/carry kernel
- turnover and fixed-cost accounting
- anchored non-overlapping walk-forward summaries

Phase 4 still does **not** implement:

- tradable-chain PnL reconciliation across rolls
- live execution routing
- rates / FX support without raw statistics
- parameter fitting or optimizer search inside the harness

Current honest outcome:

- the harness uses `research_close_to_close` as its return basis
- contract sizing still uses same-day contract notional from the tradable snapshot
- this is a research monitor / evaluator, not a deployable execution simulator

## 9. Core objects

The kernel should publish these objects:

- `TrendForecast`
- `CarryForecast`
- `CombinedForecast`
- `TargetPosition`
- `ExecutionIntent`

The kernel should **not** publish live orders in v1.

## 10. Reuse map

### 10.1 Reuse directly

- `trading_app/risk_manager.py`
  - control concepts and later adapter hooks
- `trading_app/sr_monitor.py`
  - live monitoring pattern
- `trading_app/prop_profiles.py`
  - future deployment boundary
- broker contract resolvers
  - future live mapping

### 10.2 Reuse conceptually only

- `trading_app/lane_allocator.py`
  - useful as a governance pattern, not as the kernel core
- `research/research_portfolio_engine.py`
  - sizing / correlation ideas, not canonical runtime
- `docs/institutional/mechanism_priors.md`
  - role vocabulary (`R3`, `R7`, `R8`)

### 10.3 Do not reuse as the core

- `trading_app/execution_engine.py`
- ORB outcome / lane / session objects
- ORB-specific allocators and strategy shelves

## 11. Implementation phases

### Phase 0 — implemented in this change

- create a separate `mf_futures/` research-only package
- add clean schemas
- implement pure no-lookahead signal and position functions
- add unit tests

### Phase 1 — next

- daily market snapshot ingestion / assembly
- research-chain vs tradable-chain split
- contract roll / expiry normalization surfaces

### Phase 2

- canonical next-contract / carry surface
- source-backed expiry surface for covered stats families

- walk-forward research harness
- cost and roll accounting
- publication of forecast and target artifacts

### Phase 3

- adapter from kernel outputs to shared control plane
- shadow-only publication for later deployment

## 12. Acceptance criteria for the scaffold

The scaffold is acceptable if:

- it is separated from ORB runtime objects
- it has no hidden intraday alpha assumptions
- all pure functions are unit-tested
- carry and trend semantics are explicit and not overloaded
- target-position logic is deterministic and bounded

## 13. Immediate code decision

Implement a **research-only kernel scaffold now**:

- `mf_futures/config.py`
- `mf_futures/models.py`
- `mf_futures/kernel.py`
- tests

No live execution code in this stage.
