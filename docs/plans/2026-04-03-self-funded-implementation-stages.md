# Self-Funded Tradovate — Implementation Stages

**Spec:** `docs/plans/2026-04-03-self-funded-tradovate-design.md`
**Total effort:** ~6 hours code + 30 min manual setup

---

## Stage 1: Profile Config (30 min)

**Files:** `trading_app/prop_profiles.py`

Update `self_funded_tradovate` profile:
- [ ] `account_size`: 50000 → 30000
- [ ] `payout_policy_id`: None → `"self_funded"`
- [ ] Lane 3 (COMEX_SETTLE): cap None → 150pt
- [ ] Lane 1 (MGC CME_REOPEN): verify cap = 30pt
- [ ] Lane 4 (EUROPE_FLOW): verify cap = 120pt
- [ ] Add 6 new lanes from Type-A book (lanes 6-11 from spec)
- [ ] Set `max_slots`: 5 → 11

**Acceptance:**
- `python -c "from trading_app.prop_profiles import ACCOUNT_PROFILES; p=ACCOUNT_PROFILES['self_funded_tradovate']; print(len(p.daily_lanes), p.account_size, p.payout_policy_id)"` → `11 30000 self_funded`
- Drift clean

---

## Stage 2: Daily/Weekly Loss Limits (2 hours)

**Files:** `trading_app/account_hwm_tracker.py`, `tests/test_trading_app/test_account_hwm_tracker.py`

Extend HWM tracker with self-imposed risk rules:
- [ ] Add `daily_loss_limit` parameter (default None = prop behavior, set for self-funded)
- [ ] Add `weekly_loss_limit` parameter
- [ ] `check_halt()` returns halt reason: DD_TRAILING | DAILY_LOSS | WEEKLY_LOSS | NONE
- [ ] Daily limit resets at session boundary (09:00 Brisbane)
- [ ] Weekly limit resets Monday 09:00 Brisbane
- [ ] Does NOT ratchet — this is the key difference from prop trailing DD

**Self-funded limits from spec:**
- daily_loss_limit: -600 (2% of 30K)
- weekly_loss_limit: -1500 (5% of 30K)
- drawdown_halt: -3000 from peak (10%)

**Acceptance:**
- Test: daily limit hit → check_halt returns DAILY_LOSS
- Test: daily limit resets next day → check_halt returns NONE
- Test: weekly limit hit → check_halt returns WEEKLY_LOSS
- Test: weekly limit resets Monday → check_halt returns NONE
- Test: prop profile (no daily/weekly) → unchanged behavior

---

## Stage 3: Per-Trade Max Risk Guard (1 hour)

**Files:** `trading_app/live/session_orchestrator.py`, test file

Add a `max_risk_per_trade` check before order submission:
- [ ] If `risk_dollars > profile.max_risk_per_trade` → reject trade, log warning
- [ ] `max_risk_per_trade` on profile (default None = no limit)
- [ ] Set to 300 on self_funded_tradovate profile

**Acceptance:**
- Test: trade with risk_dollars=250 → accepted
- Test: trade with risk_dollars=350 → rejected with log
- Test: profile without max_risk → all trades accepted

---

## Stage 4: Fix Unverified Filter Columns (1.5 hours)

**Files:** `scripts/tools/_tmp_honest_stress_test.py` (or permanent version), investigation

Two lanes have filter column issues:
- [x] MNQ_NYSE_OPEN ATR70_VOL: column `atr_20_pct` is CORRECT (not `atr_pct_rank`). Filter works. But strategy NOT in validated_setups — never pipeline-validated.
- [x] MNQ_US_DATA_1000 X_MES_ATR70: cross-instrument injection works at all levels (discovery, live, paper). But strategy fdr_significant=False in experimental — too weak for BH FDR.

**Result:** Both lanes marked UNDEPLOYABLE. Validated alternatives identified:
- NYSE_OPEN → `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50` (N=1441, Sharpe=1.15)
- US_DATA_1000 → `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10` (N=1941, Sharpe=0.73)

**Acceptance:**
- ~~Both lanes have verified filter-applied P&L in spec~~
- [x] Both lanes are marked UNDEPLOYABLE with explanation

---

## Stage 5: Tradovate Personal Auth (30 min, manual)

**No code changes. Manual setup.**

- [ ] Create personal Tradovate account (separate from prop)
- [ ] Get API credentials
- [ ] Test auth: `python -m trading_app.live.tradovate.auth --test`
- [ ] Fund account ($30K via ACH or wire)

**Acceptance:**
- Auth test succeeds with personal account credentials
- Account visible in Tradovate dashboard

---

## Stage 6: Integration Test (1 hour)

**Files:** None new — verification only

- [ ] Set `self_funded_tradovate.active = True`
- [ ] Run `python -m trading_app.pre_session_check --profile self_funded_tradovate`
- [ ] Run paper trade simulation for 1 session (CME_REOPEN)
- [ ] Verify: order builds correctly, risk check passes, ORB cap enforced, daily limit tracked
- [ ] Run `python pipeline/check_drift.py`
- [ ] Run `python -m pytest tests/test_trading_app/ -x -q`

**Acceptance:**
- Pre-session check passes
- Paper trade produces expected output
- All tests green
- Drift clean

---

## Stage Order

```
Stage 1 (profile config)
  → Stage 2 (loss limits) — needs profile to test against
  → Stage 3 (risk guard) — needs profile
  → Stage 4 (filter fix) — independent, can parallel with 2-3
  → Stage 5 (auth) — manual, independent
  → Stage 6 (integration) — needs all above
```

Stages 1-3 are sequential. Stage 4 and 5 can run in parallel with 2-3.

---

## After Implementation

1. **Phase 1 deployment:** Activate profile, start with 5 core lanes at 1ct
2. **Monitor:** Daily P&L vs backtest expectations. If slippage > 3 ticks avg → review.
3. **Phase 2 gate:** If net positive >$2K after 60 trades → add extra lanes
4. **Phase 3 gate:** If account >$35K → scale to 2ct on best lanes
