# Prop Firm Portfolio Construction — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build a profile-based portfolio selector that filters the validated strategy universe down to realistic, tradeable portfolios per prop firm account — with cognitive load caps, DD budget allocation, and clear excluded-with-reasons reporting.

**Architecture:** Option A — profile layer on top of existing `live_config`. Three config layers (firm specs → account tiers → user profiles). New files only — zero modification to existing production code.

**Tech Stack:** Python dataclasses, DuckDB (read-only), existing `build_live_portfolio()` + `COST_SPECS` patterns.

---

## Context

### Problem
- 30 `LiveStrategySpec` entries expand to 60-90+ active strategy slots across 4 instruments
- Real prop firm account: 5-10 slots max (DD budget + cognitive load)
- No existing mechanism to filter validated universe per firm constraints
- User trades multiple accounts simultaneously (needs multi-profile support)

### Verified Prop Firm Rules (March 2026)

| Firm | Account | Max DD | DD Type | Max Mini | Max Micro | Split | Auto | Metals |
|------|---------|--------|---------|----------|-----------|-------|------|--------|
| TopStep | $50K | $2,000 | EOD trailing | 5 | 50 | 50/50→90/10 @$5K | YES | YES |
| TopStep | $150K | $4,500 | EOD trailing | 15 | 150 | same | YES | YES |
| MFFU Core | $50K | $1,500 | EOD trailing | TBD | TBD | 80/20 | YES | YES |
| MFFU Rapid | $50K | $2,000 | Intraday trailing | TBD | TBD | 90/10 | YES | YES |
| MFFU Pro | $100K | $3,000 | EOD trailing | TBD | TBD | 80/20 | YES | YES |
| Tradeify Sel | $50K | $2,000 | EOD→static lock | 4 | 40 | 90/10 | YES | YES |
| Tradeify Gr | $150K | $6,000 | EOD trailing | 12 | 120 | 100%→90/10 @$15K | YES | YES |
| Apex | any | — | — | — | — | — | Semi | **NO metals** |
| Self-funded | $50K | user-defined | N/A | unlimited | unlimited | 100% | YES | YES |

Sources:
- TopStep: https://www.proptradingvibes.com/prop-firms/topstep
- Apex metals halt: https://support.apextraderfunding.com/hc/en-us/articles/46641038974107
- MFFU: https://www.proptradingvibes.com/prop-firms/myfundedfutures
- Tradeify: https://www.proptradingvibes.com/prop-firms/tradeify

### Key Constraints Modeled
1. **DD budget** — median DD ~$935/contract (0.75x stop), ~$1,350 (1.0x). Intraday trailing 0.7x adjustment.
2. **Contract caps** — per instrument and total (MGC capped at 2c on $50K accounts)
3. **Instrument bans** — Apex metals suspended (pattern supports future restrictions)
4. **Profit split** — adjusts effective ExpR for ranking (50/50 split halves the effective edge)
5. **Consistency rules** — no single session > X% of portfolio ExpR (TopStep 40%, Tradeify 35%)
6. **Cognitive cap** — hard max slots per profile regardless of DD room (default 8)
7. **Multiple accounts** — `copies` field for identical account duplication (x3 TopStep)
8. **Stop multiplier** — 0.75x for prop (DD=death), 1.0x for self-funded (DD=temporary)

### Data Flow
```
PROP_FIRM_SPECS (static)  +  ACCOUNT_TIERS (static)  +  ACCOUNT_PROFILES (editable)
                                    │
                    LIVE_PORTFOLIO (existing, unchanged)
                                    │
                    build_live_portfolio() per instrument (existing)
                                    │
                    Pool all eligible strategies cross-instrument
                                    │
                    select_for_profile() [NEW]
                    ├── Filter: banned instruments
                    ├── Deduplicate: one per session×instrument
                    ├── Adjust: ExpR × profit_split_factor
                    ├── Rank: Sharpe/DD ratio
                    ├── Greedy fill: DD budget, contracts, slots, consistency
                    └── Output: TradingBook + ExcludedReport
                                    │
                    CLI: python -m trading_app.prop_portfolio
                    ├── --profile topstep_50k  (single profile)
                    ├── --all                  (all active profiles)
                    └── --summary              (cross-account aggregate)
```

### Files Touched
- **Create:** `trading_app/prop_profiles.py` (config + data structures)
- **Create:** `trading_app/prop_portfolio.py` (selection logic + CLI)
- **Create:** `tests/test_trading_app/test_prop_profiles.py`
- **Create:** `tests/test_trading_app/test_prop_portfolio.py`
- **Modify:** NONE (zero blast radius on existing code)

---

## Implementation Tasks

### Task 0: Data Structures & Firm Config (`prop_profiles.py`)

Create `trading_app/prop_profiles.py` with all dataclasses and verified firm data.

**Dataclasses:**
- `PropFirmSpec` — static firm rules (DD type, split, consistency, bans, auto policy)
- `PropFirmAccount` — account tier (size, DD amount, contract limits)
- `AccountProfile` — user's actual account (firm+tier ref, copies, stop mult, slot cap)
- `TradingBookEntry` — selected strategy with sizing and ranking info
- `ExcludedEntry` — excluded strategy with reason
- `TradingBook` — collection of entries + excluded + summary stats

**Config dicts:**
- `PROP_FIRM_SPECS` — keyed by firm name
- `ACCOUNT_TIERS` — keyed by `(firm, account_size)` tuple
- `ACCOUNT_PROFILES` — keyed by profile_id, user-editable

### Task 1: Tests for Data Structures

Test dataclass validation, firm spec lookups, profile resolution.

### Task 2: Profit Split Calculator

Function `compute_profit_split_factor(firm_spec, cumulative_profit=0)` that returns
the effective percentage the trader keeps. Used to adjust ExpR for ranking.

### Task 3: Selection Algorithm (`prop_portfolio.py`)

Core function `select_for_profile(profile, all_strategies)`:
1. Filter banned instruments
2. Deduplicate session×instrument (keep best)
3. Compute effective_expr (ExpR × split factor)
4. Rank by Sharpe/DD ratio
5. Greedy fill with constraints (DD budget, contracts, slots, consistency)
6. Return TradingBook

### Task 4: Tests for Selection Algorithm

Test each constraint independently: DD exhaustion, contract cap, slot cap, consistency rule,
instrument ban, profit split ranking adjustment.

### Task 5: Cross-Profile Aggregation

Function `build_all_books(profiles)` that builds books for all active profiles and
computes cross-account summary (total exposure, aggregate DD budget used).

### Task 6: CLI (`__main__` block)

`python -m trading_app.prop_portfolio` with `--profile`, `--all`, `--summary` flags.
Follows existing `live_config.py` CLI pattern.

### Task 7: Integration Test

End-to-end test: build live portfolios from test DB → run selection → verify output format.

### Task 8: Design Doc Commit

Commit design doc + all implementation.
