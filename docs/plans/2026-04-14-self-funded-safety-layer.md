# Self-Funded Safety Layer Design

**Date:** 2026-04-14
**Author:** Claude Code session
**Status:** DESIGNED — awaiting approval

## Problem

Self-funded accounts (NinjaTrader/IBKR) have no external DD enforcement.
The bot needs to BE the prop firm — enforce its own DD limits, scale
position size as equity changes, and prevent catastrophic loss.

Without this, a $5K account on a 33R losing streak becomes $520.

## Decision

Extend existing two-layer safety system (RiskManager + AccountHWMTracker)
rather than building new modules. 4-5 files touched, no new modules.

## Changes

### 1. Self-funded profile (prop_profiles.py)

New PropFirmSpec for "self_funded":
- dd_type = "eod_trailing" (HWM advances at session end only)
- close_time_et = "16:10" (CME hard close)
- profit_split = 100%
- auto_trading = "full"

Account tiers at $5K, $10K, $25K, $50K:
- max_dd = 20% of account
- daily_loss_limit = 5% of account
- max_contracts_micro = account / $2,500
- max_contracts_mini = account / $25,000

### 2. Equity-scaling (risk_manager.py)

In can_enter(), use existing suggested_contract_factor:
- Equity > 90% HWM: factor 1.0 (full size)
- Equity 80-90% HWM: factor 0.5 (half size)
- Equity < 80% HWM: factor 0.0 (no entries — hard pause)

### 3. Cooling period (session_safety_state.py)

New field: cooldown_until (ISO datetime or None).
On kill switch: set cooldown_until = now + 24h.
Pre-session check blocks if cooldown active.
Auto-clears after 24h.

### 4. HWM tracker wiring (session_orchestrator.py)

Extend prop-firm-only HWM creation to self-funded profiles.
Same dd_limit, dd_type, equity polling. No freeze_at (no MLL).

### 5. Pre-session cooldown gate (pre_session_check.py)

Read session_safety_state. If cooldown_until > now, BLOCK.

## Files Touched

1. trading_app/prop_profiles.py — new firm spec + tiers + profiles
2. trading_app/risk_manager.py — equity-scaling in can_enter()
3. trading_app/live/session_safety_state.py — cooldown_until field
4. trading_app/live/session_orchestrator.py — HWM tracker for self-funded
5. trading_app/pre_session_check.py — cooldown gate

## Blast Radius

- No pipeline changes
- No schema changes
- No existing profile behavior changes (all gated on firm="self_funded")
- Existing prop firm flow untouched

## Risks

1. Equity polling fails — already handled (3 strikes = halt)
2. Bot crashes mid-trade — firm close time force-flatten handles this
3. User overrides cooling — must edit state file manually (deliberate)
4. Account equity stale on restart — first broker poll corrects

## Test Plan

- Self-funded profile loads with correct DD limits
- can_enter() blocks at 80% equity, halves at 90%
- Cooling period blocks session start
- Cooling period auto-clears after 24h
- HWM tracker creates for self-funded
- Pre-session check gates on cooling
