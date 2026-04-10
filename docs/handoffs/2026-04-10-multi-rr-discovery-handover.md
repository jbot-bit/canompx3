# Handover — MNQ Multi-RR Discovery + Bot Readiness Audit

**Date:** 2026-04-10
**Branch:** `research/gc-proxy-validity` (pushed)
**Session focus:** Bot readiness check, multi-RR discovery, XFA payout analysis

---

## What happened

### 1. Bot readiness audit — NOT READY for live XFA

**F-1 Scaling Plan enforcement is DORMANT.** The code exists (`risk_manager.py:192-243`, `topstep_scaling_plan.py`) but `session_orchestrator.py:229-233` never wires it up:
- `topstep_xfa_account_size` not passed to `RiskLimits` → check never fires
- `set_topstep_xfa_eod_balance()` never called → fail-closed (safe but zero trades)
- Day 1 of 50K XFA: bot would run 5 lots vs 2-lot cap → **instant rule review**

**F-2 (hedging guard) IS wired** — `execution_engine.py:903-904` passes instrument+direction to `can_enter()`.

**Fix needed:** ~20 lines in `session_orchestrator.py`. HARD GATE before any live XFA.

### 2. XFA payout analysis — Standard vs Consistency

User decided: will do a new Trading Combine first (can't afford XFA activation fee now). Using existing 3 blown practice accounts for bot testing.

**Accounts discovered via ProjectX API:**
- `20092334` — 100KTC (ending in 502, blown, user says ignore)
- `20859313` — 50KTC
- `21390438` — 50KTC

**Payout recommendation:** XFA Consistency (higher cap, faster cycle). But user correctly pushed back — goal is maximize XFA time as proving ground, not churn payouts. LFA call-up is at TopStep's discretion. ToU says "no call up" but help articles describe the process. Contradiction documented.

### 3. MNQ Multi-RR discovery — 5/5 PASSED

**Problem found:** Profile `topstep_50k_mnq_auto` had 5 stale G8/RR2.0 lanes. Only 3 RR1.0 G5 strategies existed in `validated_setups`. User correctly asked "why are we stuck to RR1.0?"

**Answer:** RR2.0 hypothesis file (`mnq-final.yaml`) was written but NEVER RUN. Only `mnq-rr10-individual.yaml` (all RR1.0) was executed.

**Raw data confirmed higher RR edge:**
- EUROPE_FLOW: RR1.5/2.0 both ExpR=0.040 (2x the RR1.0 baseline)
- TOKYO_OPEN: positive all the way to RR2.5
- NYSE_OPEN: RR1.5 ExpR=0.075 (strong)

**Created + committed:** `2026-04-10-mnq-multi-rr-individual.yaml` (5 hypotheses, Pathway B)

**Discovery + validation results — ALL 5 PASSED:**

| Strategy | RR | WR | ExpR | WFE | OOS ExpR | WF Windows |
|---|---|---|---|---|---|---|
| MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5 | 1.5 | 45.8% | 0.107 | 1.91 | +0.123 | 10/11 (91%) |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 1.5 | 47.7% | 0.074 | 2.86 | +0.100 | 11/11 (100%) |
| MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5 | 2.0 | 40.5% | 0.096 | 1.84 | +0.110 | 10/11 (91%) |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5 | 1.5 | 49.2% | 0.095 | 0.57 | +0.078 | 7/11 (64%) |
| MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5 | 2.0 | 40.7% | 0.087 | 0.63 | +0.079 | 8/11 (73%) |

### 4. DATA ISSUE — Old RR1.0 strategies wiped

The validator run **deleted the 3 existing RR1.0 strategies** (COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN at RR1.0 G5). The `validated_setups` table now has ONLY the 5 new multi-RR strategies. MES and MGC validated_setups were already at 0 before this session.

**Root cause:** Validator appears to do DELETE+INSERT per instrument, not per strategy_id. Needs investigation.

**Fix:** Re-run discovery + validation with `2026-04-09-mnq-rr10-individual.yaml` to restore the 3 RR1.0 strategies.

### 5. GC proxy discovery (other terminal)

The other terminal completed GC proxy discovery with 3 validated strategies on 16yr gold data. See `gc_proxy_discovery_results.md` memory file. Commits `4cde0c33` through `145840cf`.

---

## Immediate TODO (next session)

1. **Re-run RR1.0 discovery** — `python -m trading_app.strategy_discovery --instrument MNQ --hypothesis-file docs/audit/hypotheses/2026-04-09-mnq-rr10-individual.yaml` then validate. Restores COMEX_SETTLE, EUROPE_FLOW, NYSE_OPEN at RR1.0.

2. **Investigate validator DELETE behavior** — why did it wipe old strategies? Should be INSERT OR REPLACE on strategy_id, not DELETE all for instrument.

3. **Update `topstep_50k_mnq_auto` profile** — replace 5 stale G8 lanes with validated strategies. User decides which RR targets per session.

4. **Wire F-1 scaling plan** — `session_orchestrator.py` needs `topstep_xfa_account_size` and `set_topstep_xfa_eod_balance()`. HARD GATE before any XFA.

5. **Launch bot on practice accounts** — signal-only mode on 50KTC accounts (20859313 or 21390438) once profile is updated.

---

## User decisions needed

- **Which RR per session for the bot profile?** Full menu after RR1.0 restore: COMEX_SETTLE (RR1.0 only), EUROPE_FLOW (RR1.0/1.5/2.0), NYSE_OPEN (RR1.0/1.5), TOKYO_OPEN (RR1.5/2.0). Can deploy multiple RR targets per session if desired.
- **XFA Standard or Consistency?** When ready to activate. User leans toward maximizing XFA time.
- **Merge branch to main?** 17 commits ahead. Contains GC proxy + multi-RR discovery + orchestrator fixes.
