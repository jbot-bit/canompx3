---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Microstructure Gate 0 — Design

**Date:** 2026-04-23  
**Status:** design only  
**Authority:** `RESEARCH_RULES.md`, `TRADING_RULES.md`, `docs/STRATEGY_BLUEPRINT.md`, `docs/plans/2026-04-22-current-meta-overlay-program.md`, `pipeline/cost_model.py`, `research/research_mgc_e2_microstructure_pilot.py`, `research/research_mnq_e2_slippage_pilot.py`, `research/databento_microstructure.py`

---

## Executive decision

Track D is a **valid future new-data program**, but not because the repo has proved that all 1-minute OHLCV is exhausted.

What is actually proved:

- specific 1-minute participation overlays failed honestly
- specific exact-parent structure shadow buckets failed honestly
- current runtime still cannot express true conditional sizing or same-session half-size coexistence at `max_contracts=1`

What remains open:

- the repo still does not directly observe order-book state before a break
- the breakout-authenticity question ("real breakout or sweep / absorption?") is still not answered by canonical `bars_1m`

So the honest Track D framing is:

> add a new event-time market-data tier to test whether pre-break order-book state predicts breakout follow-through net of instrument-specific friction.

This is a **design and falsification** program, not an HFT build and not a replacement for the still-open current-stack `PR48` translation branch.

---

## What repo truth already says

### [MEASURED] Current-stack overlays did not solve the breakout-authenticity question

- `docs/audit/results/2026-04-23-mes-e1-rel-vol-family-v1.md`
  - exact `MES E1 rel_vol` family is a durable `KILL`
- `docs/audit/results/2026-04-23-mnq-parent-structure-shadow-buckets-v1.md`
  - exact-parent MNQ structure shadow buckets are a durable `KILL`
- `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`
  - surviving O5 prior-day geometry rows are blocked by runtime expression, not by outright signal death

### [MEASURED] Repo already has microstructure substrate, but it is slippage-oriented

- `research/research_mgc_e2_microstructure_pilot.py`
  - deterministic microstructure pilot pack
- `research/databento_microstructure.py`
  - Databento pull + repricing utility
- `research/research_mnq_e2_slippage_pilot.py`
  - MNQ TBBO gap-fill pilot
- `pipeline/cost_model.py`
  - now records MGC and MNQ TBBO pilot findings and remaining debt

### [MEASURED] Existing pilots answer a different question

The current TBBO pilots answer:

- is modeled stop-market slippage too optimistic?

They do **not** answer:

- did pre-break order-book state predict whether the breakout would actually follow through?

That predictive question requires a new feature layer.

### [GROUNDED] Current external data options

Databento currently documents:

- `tbbo`
  - trade-space BBO snapshots immediately before each trade
- `mbp-1`
  - every top-of-book update, including trades and BBO depth changes
- `mbo`
  - full order-by-order depth with order IDs, adds, cancels, modifies, trades, and fills

Official docs:

- https://databento.com/docs/schemas-and-data-formats
- https://databento.com/docs/schemas-and-data-formats/mbp-1
- https://databento.com/docs/schemas-and-data-formats/mbo
- https://databento.com/docs/faqs
- https://databento.com/docs/knowledge-base/datasets/glbx-mdp3

### [GROUNDED] Why these features are worth testing

- Cont, Kukanov, Stoikov (2014): short-horizon price changes are more robustly linked to order-flow imbalance than to raw traded volume, with impact stronger when depth is thinner.
- Gould and Bonart (2016): queue imbalance has predictive power for the direction of the next mid-price move.

Primary sources:

- https://academic.oup.com/jfec/article-abstract/12/1/47/816163
- https://arxiv.org/abs/1512.03492

This is enough to justify a falsification program. It is **not** enough to claim deployable edge before testing.

---

## Correct object

This program must be split into three objects:

1. **Data-tier object**
   - event-time order-book storage around ORB break windows

2. **Feature object**
   - pre-break microstructure state over a frozen lookback window, initially 60 seconds before first ORB touch

3. **Gate 0 decision object**
   - can one small, theory-grounded microstructure family improve breakout follow-through enough to matter after friction?

This is **not** a live execution project yet.

---

## Data architecture

### Design principle

Start with the **cheapest sufficient truth layer**, not the richest one.

That means:

1. **Phase D0 / Gate 0**
   - use `MBP-1` or `TBBO` first
   - cheaper, simpler, enough to test top-of-book imbalance and trade-space pressure

2. **Phase D1 escalation**
   - use `MBO` only if Gate 0 shows genuine predictive signal that top-of-book data cannot fully explain

This avoids paying L3 complexity before L1/L1.5 proves there is something there.

### Proposed DuckDB tables

#### 1. `microstructure_windows`

Purpose:
- immutable manifest of every paid pull / replay window

Key columns:
- `window_id`
- `hypothesis_slug`
- `symbol`
- `databento_symbol`
- `orb_label`
- `orb_minutes`
- `entry_model`
- `rr_target`
- `direction`
- `trading_day`
- `window_start_utc`
- `window_end_utc`
- `lookback_seconds`
- `schema_used`
- `source_dbn_path`
- `pull_cost_usd`
- `git_sha`
- `created_at_utc`

#### 2. `micro_tbbo_events`

Purpose:
- raw trade-space top-of-book around candidate breaks

Key columns:
- `window_id`
- `ts_event`
- `ts_recv`
- `instrument_id`
- `symbol`
- `price`
- `size`
- `side`
- `bid_px_00`
- `ask_px_00`
- `bid_sz_00`
- `ask_sz_00`
- `sequence`
- `flags`

Use:
- aggressor-side imbalance at trades
- trade-conditioned spread state
- pre-touch trade pressure

#### 3. `micro_mbp1_events`

Purpose:
- every top-of-book update in event space

Key columns:
- `window_id`
- `ts_event`
- `ts_recv`
- `instrument_id`
- `symbol`
- `action`
- `side`
- `price`
- `size`
- `bid_px_00`
- `ask_px_00`
- `bid_sz_00`
- `ask_sz_00`
- `bid_ct_00`
- `ask_ct_00`
- `sequence`
- `flags`

Use:
- OFI
- top-of-book queue imbalance
- microprice drift
- spread shock and depth thinning

#### 4. `micro_mbo_events`

Purpose:
- full order-by-order escalation layer

Key columns:
- `window_id`
- `ts_event`
- `ts_recv`
- `instrument_id`
- `symbol`
- `order_id`
- `action`
- `side`
- `price`
- `size`
- `sequence`
- `flags`

Use:
- actual queue depletion at the break-side price
- cancel/add asymmetry
- depth replenishment / resiliency
- passive-vs-aggressive decomposition

#### 5. `micro_gate0_features`

Purpose:
- frozen, reproducible feature extracts per trade candidate

Key columns:
- `feature_row_id`
- `window_id`
- `trading_day`
- `symbol`
- `orb_label`
- `orb_minutes`
- `direction`
- `touch_ts_utc`
- `lookback_seconds`
- `ofi_60s`
- `ofi_z_60s`
- `trade_imbalance_60s`
- `queue_imbalance_last_1s`
- `queue_imbalance_mean_10s`
- `microprice_drift_10s`
- `spread_mean_ticks_10s`
- `spread_max_ticks_10s`
- `same_side_depth_change_10s`
- `opp_side_depth_change_10s`
- `break_side_queue_depletion_10s`
- `schema_level`
- `feature_version`

#### 6. `micro_gate0_labels`

Purpose:
- exact frozen labels for the falsification test

Key columns:
- `feature_row_id`
- `pnl_r`
- `risk_dollars`
- `followthrough_hit_rr`
- `mae_r`
- `mfe_r`
- `label_version`

Source:
- canonical `orb_outcomes`

---

## Feature definitions

All first-pass features must be computable **before or at first touch**, never after the trade outcome is known.

### 1. Order Flow Imbalance (`OFI_60s`)

Recommended first-pass definition on `MBP-1`:

For each event `n`, define

`e_n = 1[b_n >= b_{n-1}] * q^b_n - 1[b_n <= b_{n-1}] * q^b_{n-1} - 1[a_n <= a_{n-1}] * q^a_n + 1[a_n >= a_{n-1}] * q^a_{n-1}`

where:

- `b_n`, `a_n` are best bid / ask prices at event `n`
- `q^b_n`, `q^a_n` are displayed best bid / ask sizes

Then:

- `OFI_60s = Σ e_n` over the 60 seconds before first ORB touch
- `OFI_z_60s = z-score of OFI_60s` using IS-only calibration
- `OFI_signed_to_break_dir = OFI_60s * break_sign`

Interpretation:

- positive signed OFI means book pressure aligned with the breakout direction

### 2. Trade-space imbalance (`TBI_60s`)

On `TBBO`:

- classify trade side from Databento `side`
- compute
  - `buy_volume_60s`
  - `sell_volume_60s`
  - `TBI_60s = (buy_volume_60s - sell_volume_60s) / (buy_volume_60s + sell_volume_60s)`
- sign to breakout direction the same way as OFI

This is cheaper than MBO and directly tied to actual prints.

### 3. Queue imbalance (`QI`)

On `MBP-1`:

- `QI_t = (bid_sz_00 - ask_sz_00) / (bid_sz_00 + ask_sz_00)`

Derive:

- `QI_last_1s`
- `QI_mean_10s`
- `QI_min_10s`
- `QI_signed_to_break_dir`

Interpretation:

- positive signed QI means displayed top-of-book depth is heavier on the break side

### 4. Break-side queue depletion

On `MBO` escalation only:

- choose the queue at the touch-side boundary price
- over the pre-touch 10-second window compute:
  - `depletion = fills + cancels - adds`
- normalize by starting displayed queue:
  - `depletion_ratio = depletion / max(start_queue_size, 1)`

Interpretation:

- high positive depletion on the opposing side before touch suggests the wall is being consumed

### 5. Spread / resiliency stress

From `MBP-1` or `TBBO`:

- `spread_mean_ticks_10s`
- `spread_max_ticks_10s`
- `mid_reversion_1s_after_trade` for later phases

Gate 0 keeps these as secondary controls, not the thesis.

---

## Gate 0 test

### Exact question

Can a frozen pre-touch top-of-book imbalance state improve breakout follow-through on one exact active lane enough to matter after modeled friction?

### Exact first lane

Use:

- `MNQ`
- `COMEX_SETTLE`
- `O5`
- `E2`
- `RR1.5`

Why this lane:

- active live relevance
- existing repo notes already flag a targeted `MNQ COMEX_SETTLE` TBBO pull as Phase D-relevant
- existing MNQ TBBO slippage coverage already includes `COMEX_SETTLE`, so the data path is partially de-risked

Do **not** start on `MGC` just because `$5.74` sounds familiar. That friction number is instrument-specific and belongs to MGC, not MNQ. If a later Gate 0 is run on MGC, use MGC’s own cost model then.

### Exact first family

Keep the family tiny:

1. `signed_ofi_60s_high`
2. `signed_tbi_60s_high`
3. `signed_qi_last_1s_high`
4. `signed_ofi_60s_high AND signed_qi_last_1s_high`

No more than `K=4` in the first falsification pass.

### Cheapest honest falsification

1. Pull one deterministic IS-only sample of exact candidate windows for the lane.
2. Build frozen thresholds from IS only:
   - e.g. top quartile signed OFI
   - top quartile signed TBI
   - top quartile signed QI
3. Compare:
   - parent vs on-signal subset
   - exact parent policy EV vs filtered policy EV
4. Apply unchanged to sacred `2026-01-01+` OOS.

### Success condition

Gate 0 passes only if:

- at least one feature clears the chosen prereg family rule
- selected-trade quality improves
- **and** policy value improves versus the exact parent
- **and** the sign does not flip in OOS once the OOS subset is non-trivial

### Failure condition

If all four fail, the honest conclusion is:

- no evidence yet that top-of-book pre-break imbalance helps this exact breakout lane
- do **not** escalate to MBO immediately
- reassess whether the problem is signal, lane choice, or execution horizon before spending on L3

---

## What this is not

- not proof that OHLCV is globally exhausted
- not a license to reopen killed confluence families
- not a sizing framework
- not an HFT execution system
- not a live deployment plan

It is a **new-data falsification stage**.

---

## Recommended next order of operations

1. Close the current exact-parent shadow-bucket family cleanly.
2. Continue the open current-stack `PR48` translation branch separately.
3. If Track D is prioritized, write one locked Gate 0 prereg for the exact `MNQ COMEX_SETTLE O5 E2 RR1.5` lane and one exact top-of-book family.
4. Only escalate to `MBO` after the cheapest `MBP-1` / `TBBO` Gate 0 either:
   - passes and leaves explanatory gaps, or
   - clearly shows queue-state information is required.

---

## Sources

### Repo-local

- `docs/plans/2026-04-22-current-meta-overlay-program.md`
- `docs/STRATEGY_BLUEPRINT.md`
- `docs/audit/results/2026-04-23-mnq-parent-structure-shadow-buckets-v1.md`
- `docs/audit/results/2026-04-23-pr48-conditional-edge-recovery-audit.md`
- `docs/audit/results/2026-04-20-mnq-e2-slippage-pilot-v2-gap-fill.md`
- `research/research_mgc_e2_microstructure_pilot.py`
- `research/research_mnq_e2_slippage_pilot.py`
- `research/databento_microstructure.py`
- `pipeline/cost_model.py`

### External primary / official

- Databento schemas:
  - https://databento.com/docs/schemas-and-data-formats
  - https://databento.com/docs/schemas-and-data-formats/mbp-1
  - https://databento.com/docs/schemas-and-data-formats/mbo
  - https://databento.com/docs/faqs
  - https://databento.com/docs/knowledge-base/datasets/glbx-mdp3
- Cont, Kukanov, Stoikov:
  - https://academic.oup.com/jfec/article-abstract/12/1/47/816163
- Gould and Bonart:
  - https://arxiv.org/abs/1512.03492
