# Brisbane-Morning ORB Session Coverage & Edge Audit (Read-Only)

**Date:** 2026-06-03
**Branch/worktree:** session/joshd-morning-orb-audit @ C:\Users\joshd\canompx3-morning-orb-audit (off origin/main f505b69a)
**Truth layers used:** orb_outcomes, daily_features, bars_1m (read-only, main gold.db). Canonical pipeline.dst.SESSION_CATALOG + orb_utc_window. Derived layers (validated_setups/live_config) consulted ONLY for live/deployment status, never for truth-finding.
**Holdout:** HOLDOUT_SACRED_FROM = 2026-01-01. Discovery scan strictly trading_day < 2026-01-01. 2026 was peeked report-only, fed no selection.

## Scope

**Question:** Is the Brisbane-morning band (08:00-12:00 local) under-tested for ORB edge, or already over-mined? Read-only coverage + edge audit of the 4 morning sessions across MNQ/MES/MGC — maps catalog↔orb_outcomes alignment, DST correctness, and pre-2026 edge potential. Decision: warrant a NEW morning session, or PARK. No new session built, no orb_outcomes rebuilt, no config changed.

## VERDICT: PARK (morning is over-mined, not under-tested). No new morning session warranted.

## 1. Session map
13 sessions in SESSION_CATALOG, all type=dynamic (DST-aware resolvers). Zero legacy numeric aliases (0900/1000/1100) survive; zero type==alias entries. The Feb-2026 event rename is clean.
Brisbane-morning band (08:00-12:00) is anchored by exactly 4 sessions:
- CME_REOPEN  08:00 (US summer) / 09:00 (US winter)  [SHIFTS w/ US DST]
- TOKYO_OPEN  10:00 fixed (JST no DST)
- BRISBANE_1025 10:25 fixed (no market anchor)
- SINGAPORE_OPEN 11:00 fixed (SGT no DST)

## 2. Coverage matrix (truth layer)
catalog<->orb_outcomes perfectly aligned: 0 catalog sessions with no rows; 0 orphan labels in DB.
| candidate event | true Bris time | catalog session | exists | MNQ/MES/MGC enabled | orb_outcomes | tested | gap |
|---|---|---|---|---|---|---|---|
| CME futures reopen | 08:00/09:00 | CME_REOPEN | YES | all 3 (5/15/30) | deep (2010+ GC, 2019+ MNQ/MES, 2022+ MGC) | YES | none (negative edge) |
| ASX cash open | 09:00 summer / 10:00 winter | NONE | NO | - | - | NO | **DST-shifting event, no session. Conflated with Tokyo at 10:00 in winter only** |
| Tokyo open | 10:00 | TOKYO_OPEN | YES | all 3 | deep | YES (LONG-ONLY) | none |
| Brisbane 10:25 fixed | 10:25 | BRISBANE_1025 | YES | MNQ + M2K only (NOT MES/MGC) | MNQ+M2K | YES | MES/MGC never built for 10:25 |
| Singapore SGX open | 11:00 | SINGAPORE_OPEN | YES | all 3 (MGC OFF: 74% double-break) | deep | YES | none |
| HK pre-open auction | 11:00 | (== SINGAPORE_OPEN slot) | partial | - | shares 11:00 | partial | label says 'SGX/HKEX open' but HK *continuous* is 11:30 |
| HK cash open (continuous) | 11:30 | NONE | NO | - | - | NO | **no 11:30 session at all** |

## 3. DST audit: PASS
- Stored CME_REOPEN entries track the DST shift: winter first-entry >=09:00 Bris, summer >=08:00 Bris (canonical resolver was used historically; no fixed-UTC bug).
- TOKYO_OPEN does NOT shift across seasons (fixed-Brisbane behaves correctly).
- No duplicate event windows on winter or summer probe dates.
- entry_ts stored tz-aware (Brisbane). orb_utc_window fail-closes outside trading-day range.
- DST_AFFECTED_SESSIONS = {} by design (all dynamic) -> dst_split is correctly N/A.

## 4. Edge/potential (PRE-2026 ONLY, confirm_bars=1)
Family: 4 morning sessions x {MNQ,MES,MGC} x {5,15,30} x {E1,E2} x {1.0,1.5,2.0} = 180 slices (N>=30). BH-FDR alpha=0.05. K=180 < 300 MinBTL bound (Bailey 2013).
- Only 24/180 slices have ExpR>0. Only 5 survive BH-FDR with ExpR>0 (all MNQ E2 Tokyo/Singapore). The ~123 other BH "survivors" are significantly NEGATIVE.
- MGC and MES are uniformly negative in the morning (t down to -12). CME_REOPEN negative for all 3 instruments.
- Top-5 pre-2026 positive survivors (pooled long+short, unfiltered):
  1. SINGAPORE_OPEN MNQ 30m E2 rr2.0  N=1704 WR=0.397 ExpR=+0.1011 Sharpe_a=1.19 p=2.1e-3  7/7 positive years
  2. SINGAPORE_OPEN MNQ 30m E2 rr1.5  N=1707 ExpR=+0.0736 p=8.5e-3  6/7 yrs
  3. TOKYO_OPEN     MNQ 5m  E2 rr1.5  N=1722 ExpR=+0.0700 p=8.4e-3  5/7 yrs (front-loaded 2019-22)
  4. TOKYO_OPEN     MNQ 15m E2 rr1.0  N=1721 ExpR=+0.0583 p=7.6e-3  6/7 yrs
  5. TOKYO_OPEN     MNQ 5m  E2 rr1.0  N=1722 ExpR=+0.0462 p=2.7e-2  5/7 yrs
- These are WEAKER, unfiltered versions of edges the pipeline ALREADY exploits with filters:
  - MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ATR_P70_O30_S075 = validated ExpR +0.190 (N=586), rolling +0.277, FIT
  - MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ATR_VEL_GE105 = validated ExpR +0.205 (N=339), rolling +0.376, FIT
- 668 promotable FIT MNQ candidates exist; only 3 are allocated. Bottleneck = ALLOCATION, not discovery.

## 5. Bias checks
- DST drift: PASS (canonical resolver proven in stored data).
- Lookahead: scan reads only outcome columns (pnl_r/outcome), no daily_features predictors joined.
- Pooled long/short hiding failure: REAL LIMITATION. orb_outcomes has NO direction column. TOKYO_OPEN is registered LONG-ONLY (shorts negative) -> pooled ExpR UNDERSTATES the long-only edge. Cannot split direction from orb_outcomes alone.
- Duplicate windows: none.
- 2026 peeking: none in selection (peek report-only; pre-2026 = sacred boundary).
- Multiplicity: K=180 < 300 MinBTL. Within bound.
- Volume-spike-as-edge: not triggered (no vol filter in this baseline scan).
- Cost: pnl_r is fully cost-inclusive (to_r_multiple). +50% cost stress NOT run exactly (needs outcome rebuild) - flagged, not faked.

## 6. Genuine gaps (NOT recommending build)
- (G1) ASX cash open (09:00 Bris summer / 10:00 winter, DST-shifting via Australia/Sydney) has NO session. Mechanistically the closest untested *distinct* morning event. To test later: add SESSION_CATALOG entry with a Sydney-tz resolver -> rebuild orb_outcomes for it (full backfill) -> prereg. Est K budget ~36 (1 sess x 3 instr x 3 apert x 2 EM x 2 RR).
- (G2) HK cash continuous open 11:30 Bris has NO session (SINGAPORE_OPEN 11:00 != HK continuous). Same build/prereg cost as G1.
- (G3) BRISBANE_1025 not built for MES/MGC (MNQ+M2K only). Cheapest gap to close.
- All three require a canonical orb_outcomes rebuild for the new (session x instrument) cells; resolver edits in pipeline/dst.py SESSION_CATALOG only (never re-encode timing).

## Reproduction

- Truth layers: `orb_outcomes`, `daily_features`, `bars_1m` (read-only, main `gold.db`); session timing via `pipeline.dst.SESSION_CATALOG` + `orb_utc_window`.
- Edge scan: 4 morning sessions × {MNQ,MES,MGC} × {5,15,30} × {E1,E2} × {1.0,1.5,2.0} = 180 slices (N≥30), BH-FDR α=0.05, K=180 < 300 MinBTL bound (Bailey 2013). Discovery strictly `trading_day < 2026-01-01`; 2026 peeked report-only, fed no selection.
- Validated-lane comparison (§4) consulted `validated_setups`/live status for deployment context ONLY, never for truth-finding.

## Limitations

- **READ-ONLY audit — verdict is PARK; nothing built or changed.** The 3 genuine gaps (G1 ASX cash open, G2 HK 11:30, G3 BRISBANE_1025 MES/MGC) are documented as future-test candidates, NOT recommendations to build now.
- **Direction-split limitation (REAL):** `orb_outcomes` has no direction column, so pooled long+short ExpR UNDERSTATES long-only edges (TOKYO_OPEN is registered LONG-ONLY). The pre-2026 positive survivors are pooled and therefore conservative lower bounds, not the deployable long-only numbers.
- **Cost stress not fully run:** `pnl_r` is cost-inclusive (`to_r_multiple`), but a +50% cost-stress rerun was NOT executed (needs an outcome rebuild) — flagged, not faked.
- Edge numbers are unfiltered baseline slices; the pipeline already exploits stronger filtered versions (§4). This audit does not re-validate those deployed lanes.
- Single-band scope (08:00-12:00 Brisbane); not a pooled cross-lane universality claim (pooled-finding front-matter not owed).
