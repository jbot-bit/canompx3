# Canonical Values — Instruments, Costs, Sessions, Holdout, Profiles

**Purpose:** Single-page reference for the numeric constants that govern this project. These are extracted from the canonical source files (`pipeline/asset_configs.py`, `pipeline/cost_model.py`, `pipeline/dst.py`, `trading_app/holdout_policy.py`, `trading_app/prop_profiles.py`).

**Snapshot date:** 2026-04-18. These values change rarely; if a number here contradicts something the user tells you or a file they paste, **trust the user / the paste** and warn that CANONICAL_VALUES.md may be stale.

---

## 1. Instruments

### Active ORB universe (traded live + in discovery)
| Symbol | Name | Contract size | Point value | Tick size | Launch | parent_symbol |
|--------|------|---------------|-------------|-----------|--------|---------------|
| **MGC** | Micro Gold | 10 oz | $10/pt | 0.10 | 2022-06-13 | GC |
| **MNQ** | Micro E-mini Nasdaq-100 | — | $2/pt | 0.25 | 2019-05-06 | NQ |
| **MES** | Micro E-mini S&P 500 | — | $5/pt | 0.25 | 2019-05-06 | ES |

### Dead for ORB (tested — confirmed NO breakout edge)
`MCL`, `SIL`, `M6E`, `MBT`, `M2K` — **do not propose ORB strategies for these**. Exported as `DEAD_ORB_INSTRUMENTS` in `pipeline/asset_configs.py`.

### Research-only (not in ORB universe, no live trading)
`2YY`, `ZT` — for event-window macro work. `GC`, `NQ`, `ES` exist as parent-contract data for pre-launch history of the micros only — discovery runs on the micros, not the parents.

### Deployable expectation flag
**MGC has `deployable_expected=False`** as of 2026-04 — real-micro horizon is ~3.8yr, below T7 era-discipline threshold (needs ~5yr). MGC pipeline still runs; pulse just stops alerting on empty deployable state. Expected to flip to deployable ~2027-06.

---

## 2. Cost specs (per-instrument friction, round-trip $)

Source: `pipeline/cost_model.py::COST_SPECS`. Commissions sourced from TopStep canonical article (Rithmic baseline) — see `@canonical-source docs/research-input/topstep/topstep_xfa_commissions.md`.

| Instrument | Point value | Commission RT | Spread (×2) | Slippage | **Total friction** | Friction in points |
|------------|-------------|---------------|-------------|----------|--------------------|--------------------|
| MGC | $10/pt | $1.74 | $2.00 | $2.00 | **$5.74** | 0.574 pts |
| MNQ | $2/pt | $1.42 | $0.50 | $1.00 | **$2.92** | 1.46 pts |
| MES | $5/pt | $1.42 | $1.25 | $1.25 | **$3.92** | 0.784 pts |

**Stress-test:** `stress_test_costs(spec, multiplier=1.5)` — default +50% costs per CANONICAL_LOGIC.txt §9.

**Session-aware slippage** (live execution only — backtests use flat): `SESSION_SLIPPAGE_MULT` in `cost_model.py`. Range roughly 0.8× (US_DATA_830) to 1.3× (thin Asian sessions).

**Known limitation (adversarial review 2026-03-18):** backtest uses FLAT slippage. Real slippage likely correlates with ORB size and liquidity. Until paper TCA data arrives, cost model is structurally optimistic. Kill criterion: if live slippage > 2× modeled, all backtest results are overstated.

---

## 3. Session catalog

Source: `pipeline/dst.py::SESSION_CATALOG`. All sessions are dynamic (DST-aware resolver per trading day). Local timezone is **Australia/Brisbane (UTC+10, no DST)**. Trading day = 09:00 Brisbane → next 09:00 Brisbane.

| Label | Event | Break group | Winter Brisbane | Summer Brisbane |
|-------|-------|-------------|-----------------|-----------------|
| **CME_REOPEN** | CME Globex electronic reopen (5 PM CT) | cme | 09:00 | 08:00 |
| **TOKYO_OPEN** | Tokyo Stock Exchange (9 AM JST) | asia | 10:00 | 10:00 |
| **BRISBANE_1025** | Fixed 10:25 Brisbane (no event) | asia | 10:25 | 10:25 |
| **SINGAPORE_OPEN** | SGX/HKEX (9 AM SGT) | asia | 11:00 | 11:00 |
| **LONDON_METALS** | London metals AM (8 AM London) | london | 18:00 (GMT) | 17:00 (BST) |
| **EUROPE_FLOW** | Adjacent to London metals, opposite DST side | london | 17:00 | 18:00 |
| **US_DATA_830** | US economic data release (8:30 AM ET) | us | 23:30 | 22:30 |
| **NYSE_OPEN** | NYSE cash open (9:30 AM ET) | us | 00:30 (next cal day) | 23:30 |
| **US_DATA_1000** | Post-equity flow (10 AM ET) | us | 01:00 (next cal day) | 00:00 (next cal day) |
| **COMEX_SETTLE** | COMEX gold settlement (1:30 PM ET) | us | 04:30 | 03:30 |
| **CME_PRECLOSE** | CME equity futures pre-close (2:45 PM CT) | us | 06:45 | 05:45 |
| **NYSE_CLOSE** | NYSE closing bell (4 PM ET) | us | 07:00 | 06:00 |

**Per-instrument enabled sessions:** see `ASSET_CONFIGS[instrument]["enabled_sessions"]`. MNQ has all 12. MES has 11 (no BRISBANE_1025). MGC has 9 (no CME_PRECLOSE, no NYSE_CLOSE, no BRISBANE_1025).

### DOW alignment (Brisbane day-of-week vs exchange day-of-week)

**All sessions aligned EXCEPT NYSE_OPEN** — Brisbane 00:30 crosses midnight from the previous UTC day. Brisbane-Friday 00:30 = US Thursday 9:30 AM equity open. `validate_dow_filter_alignment()` in `pipeline/dst.py` fails-closed on DOW filters applied to NYSE_OPEN.

### ORB apertures

Canonical set: **5, 15, 30 minutes**. Canonical window function: `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)`. **Single source of truth** for ORB window end UTC across backtest (`outcome_builder`), live engine (`execution_engine`), and feature builder (`build_daily_features`). Never re-implement; never fall back to `break_ts`.

---

## 4. Holdout policy (Mode A — Amendment 2.7)

Source: `trading_app/holdout_policy.py`. Authority: `docs/institutional/pre_registered_criteria.md` Amendment 2.7 (2026-04-08).

| Constant | Value | Meaning |
|----------|-------|---------|
| `HOLDOUT_SACRED_FROM` | **2026-01-01** | Sacred OOS boundary. Discovery MUST NOT use `trading_day >= this`. |
| `HOLDOUT_GRANDFATHER_CUTOFF` | 2026-04-08 00:00 UTC | Amendment 2.7 commit. Experimental rows created at/before this are grandfathered (known Mode-B contamination from Apr 3 deviation). |
| `PHASE_4_1_SHIP_DATE` | 2026-04-09 00:00 UTC | Drift check #94 SHA-integrity cutoff. |
| `HOLDOUT_OVERRIDE_TOKEN` | `"3656"` | Speed-bump only (not crypto). Invocations destroy OOS validity; output is research-provisional. |

**Rule:** any discovery run with `holdout_date > 2026-01-01` raises `ValueError` unless `override_token="3656"` is passed. CLI enforcement via `--holdout-date` + `--unlock-holdout TOKEN`.

Enforcement helper: `enforce_holdout_date(holdout_date, override_token=None) -> date`.

**IS/OOS windows:**
- IS: `trading_day < 2026-01-01`
- OOS: `2026-01-01 ≤ trading_day < current_date`
- `dir_match` required: `sign(delta_IS) == sign(delta_OOS)`. OOS direction flip = FAIL regardless of significance.

---

## 5. Prop firm account profiles (summary)

Source: `trading_app/prop_profiles.py::ACCOUNT_PROFILES`. Firm rules verbatim in `prop-firm-official-rules.md` (bundled file).

Active profiles as of 2026-04-18 (snapshot — ask user for current state):

| profile_id | firm | size | copies | max_slots | instruments | active |
|------------|------|------|--------|-----------|-------------|--------|
| `topstep_50k_mnq_auto` | TopStep | $50K | 2 | 7 | MNQ | ✅ |
| `topstep_50k_mes_auto` | TopStep | $50K | — | — | MES | ask |
| `topstep_50k` | TopStep | $50K | 5 | 4 | MGC | ❌ (conditional shadow) |
| `tradeify_50k` | Tradeify | $50K | 5 | 6 | MNQ | ❌ (pending Tradovate API) |
| `topstep_50k_type_a`, `topstep_100k_type_a` | TopStep | — | — | — | — | — |
| `tradeify_50k_type_b`, `tradeify_100k_type_b` | Tradeify | — | — | — | — | — |
| `bulenox_50k` | Bulenox | $50K | — | — | — | — |
| `self_funded_tradovate` | self-funded | — | — | — | — | — |

**Scaling plan (2026-04-15 final landscape, see memory `topstep_scaling_corrected_apr15.md`):**
- Phase 1: 1-2 TopStep XFA ($5-10K/yr)
- Phase 2: +3 Bulenox ($15-25K/yr)
- Phase 3: +5 MFFU Rapid ($30-50K/yr)
- Phase 4: +self-funded 1 NQ mini @ $50K ($65-105K/yr total)

**Dead for our plan:** Apex (bots banned), Tradeify (exclusivity clash), FundedNext (red flags), Elite/ETF (TradersPost gate, deferred vs MFFU direct).

**Key constraints:**
- TopStep XFA ↔ LFA are mutually exclusive; LFA promotion destroys XFAs (<1% LFA survival)
- Bulenox: 3 → 11 account unlock ladder
- MFFU: 5 sim + 5 DTF
- Rithmic integration unlocks Bulenox + AMP/EdgeClear self-funded
- 1 NQ mini vs 10 MNQ = ~77% commission cut at self-funded scale

Canonical firm comparison: `memory/prop_firm_complete_comparison_apr1.md` (not in this bundle — ask user if needed).

---

## 6. Where each value lives (for refresh)

| Value | Canonical source file |
|-------|-----------------------|
| Active / dead instruments | `pipeline/asset_configs.py` (`ACTIVE_ORB_INSTRUMENTS`, `DEAD_ORB_INSTRUMENTS`) |
| Cost specs | `pipeline/cost_model.py` (`COST_SPECS`) |
| Session catalog | `pipeline/dst.py` (`SESSION_CATALOG`) |
| ORB window timing | `pipeline/dst.py::orb_utc_window()` |
| Holdout constants | `trading_app/holdout_policy.py` |
| Prop profiles | `trading_app/prop_profiles.py` (`ACCOUNT_PROFILES`) |
| DB path | `pipeline/paths.py::GOLD_DB_PATH` |

**Rule of thumb:** if ChatGPT is asked "what's the cost for MGC?" — cite the table above. If the user says "but I want to simulate 2× slippage," tell them to set `stress_test_costs(spec, multiplier=2.0)`. **Never inline numbers into code suggestions** — always reference the canonical function.
