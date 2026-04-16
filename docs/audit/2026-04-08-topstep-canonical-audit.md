# TopStep Canonical Audit — 2026-04-08

**Date:** 2026-04-08
**Branch:** `topstep-canonical-info`
**Auditor:** Claude Opus 4.6 (interactive session)
**Canonical corpus:** [`docs/research-input/topstep/`](../research-input/topstep/) — 20 sources + 1 image, scraped 2026-04-08
**Authority hierarchy:** see [`docs/research-input/topstep/README.md`](../research-input/topstep/README.md)

## Purpose

Audit all TopStep compliance code (`trading_app/prop_profiles.py`, `prop_firm_policies.py`, `consistency_tracker.py`, `risk_manager.py`, `account_hwm_tracker.py`, `live/session_orchestrator.py`, `live/copy_order_router.py`, `pipeline/cost_model.py`) against primary-source canonical text. Identify violations, miscalculations, missing enforcement, and unverified claims. Persist findings as a tracker so each can be marked FIXED individually.

## Re-verification log

This is a re-verification of an initial audit done earlier the same session. The initial audit had 9 findings; 4 of those were marked UNVERIFIED (i.e., reasoned from training memory or incomplete sources). This audit verifies all of them against primary-source canonical text. Outcome:

- **Confirmed** (no change): F-1 (Scaling Plan), F-3 (LFA DLL), F-4 (commissions, escalated)
- **Refuted** (initial finding was wrong): F-9 (DLL exemption claim is canonically TRUE)
- **Revised** (initial finding was incomplete or partly wrong): F-2 (hedging — found dedicated article with 3-strike enforcement), F-7 (Trade Copier — bot uses CopyOrderRouter not native Trade Copier)
- **New** (only visible from canonical sources): F-2 dedicated single-user-hedging policy; F-3 LFA DLL = MLL; F-5 HWM freeze formula bug for XFA; F-8 Risk Adjustments article; F-13 Account stacking pattern; F-14 30-day inactivity; F-15 Back2Funded reactivations

## Severity matrix (sorted by today's blast radius)

| ID | Severity | Status | Title | Today's blast radius |
|---|---|---|---|---|
| F-1 | ⚪ RESOLVED | FALSE ALARM (2026-04-11) | Scaling Plan Day 1 violation | Was NOT a real violation — `total_open_lots` had a per-trade ceiling aggregation bug that reported 5 MNQ micros = 5 lots instead of canonical 1 lot. 5-lane bot is at 25% of Day-1 cap. See `docs/audit/2026-04-11-criterion-11-f1-false-alarm.md`. `account_survival` now reports 0.0% scaling breach, gate PASS. |
| F-2 | 🔴 BLOCKER | OPEN | Single-user / cross-account hedging not detected | Active MNQ lanes can produce opposing positions; CopyOrderRouter mirrors them across copies |
| F-2b | 🟡 HIGH | OPEN | CopyOrderRouter shadow failures create asymmetric account state | `copy_order_router.py:57-71` logs but does not propagate shadow failures → divergent positions across copies |
| F-4 | 🟡 HIGH | OPEN | MNQ/MES commissions 8-13% LOW vs canonical Tradovate/Rithmic rates | Backtest understates costs for ALL active MNQ lanes |
| F-5 | 🟡 MED | OPEN | XFA HWM freeze formula incorrect (uses TC starting balance for XFA) | Bot halts earlier than necessary (safe direction); missed trades, not violations |
| F-6 | 🟡 MED | OPEN | 5-XFA aggregate cap unenforced across active profiles | Future activation of 2nd profile → violation |
| F-7 | 🟡 MED | REVISED | Trade Copier "Lead lowest size" rule does NOT apply (project uses CopyOrderRouter, not native Trade Copier) | None — replaced by F-2b |
| F-8 | 🟡 MED | OPEN | Vol-event temporary risk adjustments unmodeled | Vol events are rare but catastrophic |
| F-3 | 🔴 deferred | DEFERRED | LFA DLL not modeled (= MLL with $10K low-balance override) | ZERO today (no LFA), critical at LFA promotion |
| F-9 | 🟢 INFO | REFUTED | DLL exemption on TopstepX claim verified TRUE for TC/XFA | None — code is correct; needs `@canonical-source` annotation only |
| F-10 | 🟢 INFO | CONFIRMED | All payout-policy fields verified (90/10, 5d/$150, 50%/$5K, 3d/40%/$6K) | None |
| F-11 | 🟢 INFO | INFERRED | Standard `additional_days_after_payout=5` plausible but not in canonical | None — annotate as `@inferred` |
| F-12 | 🟢 INFO | OPEN | VPN/VPS prohibition (ToU Section 28) | Hosting decision — confirm bot runs on home internet, not VPS |
| F-13 | 🟢 INFO | OPEN | Account stacking pattern explicitly prohibited | Behavioral note for post-blow recovery decisions |
| F-14 | 🟢 INFO | OPEN | 30-day inactivity closes XFA | Operational reminder if bot is paused |
| F-15 | 🟢 INFO | OPEN | Back2Funded gives 2 reactivations per XFA | Cost modeling only |

**Outstanding question for the user:** When did you sign up for TopStep? If before 2026-01-12, you qualify for the legacy "100% of first $10K, then 90/10" payout policy (per `topstep_payout_policy.txt:101-103`). Code currently uses flat 90/10. If you joined on or after 2026-01-12 → flat 90/10 is correct. If before → code is wrong and you're being shorted by 10% on your first $10K of payouts.

---

# Detailed findings

## 🔴 F-1 — BLOCKER: Scaling Plan Day 1 violation

**Status:** OPEN
**Severity:** 🔴 BLOCKER

**Canonical source:** [`docs/research-input/topstep/images/xfa_scaling_chart.png`](../research-input/topstep/images/xfa_scaling_chart.png) (extracted from `https://help.topstep.com/en/articles/8284223-what-is-the-scaling-plan`, image URL `https://topstep-949ca9db770d.intercom-attachments-7.com/i/o/813170406/95c64a1874bfd81176cc2d57/19155321446035`)

**Canonical XFA Scaling Plan ladder (mini-equivalent lots):**

| Account Balance | $50K XFA | $100K XFA | $150K XFA |
|---|---|---|---|
| Below $1,500 | **2 lots** | **3 lots** | **3 lots** |
| $1,500 – $2,000 | 3 lots | 4 lots | 4 lots |
| $2,000 – $3,000 | 5 lots (above $2K) | 5 lots | 5 lots |
| $3,000 – $4,500 | — | 10 lots (above $3K) | 10 lots |
| Above $4,500 | — | — | 15 lots |

**Cited rule, intra-day no-scaling-up** (`topstep_scaling_plan_article.md:68`):
> "Your maximum number of contracts allowed to trade under the scaling plan does not increase throughout the trading day."

**Cited rule, 10-second grace** (`topstep_scaling_plan_article.md:104`):
> "Errors in the Scaling Plan corrected in less than 10 seconds will be ignored. If traders leave on too many contracts for 10 seconds or more, even if only by a few seconds, their account may be reviewed."

**Code reality:** `trading_app/prop_profiles.py:251-253` ACCOUNT_TIERS encodes only the *top-tier* max (5/10/15 mini for 50K/100K/150K). Grep `trading_app/` for `scaling_plan|allowed_contracts_today|scale_level|contract_limit_today` returns **zero matches**.

**Active deployment violation:**

- `topstep_50k_mnq_auto` is `active=True`, `copies=2`, 5 daily lanes, 1ct each (`prop_profiles.py:316-418`).
- 2 copies × 5 lanes × 1ct = up to 5 mini-equivalents per copy when all lanes overlap.
- Day 1 of any new XFA 50K: cap = **2 lots**.
- 5 lanes simultaneously = **2.5× over Day 1 cap → instant rule review**.
- The 10-second grace is explicitly for **fat-finger errors corrected manually**, not sustained automated exposure.

**Required fix:** Per-account, per-day Scaling Plan tier resolver that:
1. Reads end-of-day (5pm CT) account balance from broker.
2. Maps balance → max lot count using the canonical ladder.
3. Pre-trade gate: refuses any entry that would push **net mini-equivalent exposure** above the day's tier.
4. Counts long+short opposing positions toward gross exposure correctly.
5. MGC follows the 10:1 ratio per `topstep_scaling_plan_article.md:78-94` (TopstepX). Project bot uses ProjectX API which feeds TopstepX → 10:1 applies.

**Followup canonical source needed:** `https://intercom.help/topstep-llc/en/articles/8284209` — Net position calculation across products (referenced from `topstep_scaling_plan_article.md:71`). Not yet fetched. Critical for correct enforcement of simultaneous-long+short exposure.

---

## 🔴 F-2 — BLOCKER: Single-user (cross-account) hedging guard missing

**Status:** OPEN
**Severity:** 🔴 BLOCKER

**Canonical source:** [`docs/research-input/topstep/topstep_cross_account_hedging.md`](../research-input/topstep/topstep_cross_account_hedging.md) (article 13747047, "Updated yesterday" per scraped header)

**Cited definition** (`topstep_cross_account_hedging.md:55-71`):
> "Cross-account hedging occurs when you hold opposite positions across multiple accounts at the same time. This means you're simultaneously long and short the same instrument (or highly correlated/fungible instruments) across different Combines, Express Funded Accounts (XFAs), or Live Funded Accounts (LFAs)."

**Crucial clarification** (`topstep_cross_account_hedging.md:317`):
> "Yes! You can trade the same instrument across multiple accounts. What's prohibited is holding opposite positions simultaneously."

**Cited enforcement progression** (`topstep_cross_account_hedging.md:143-247`):
1. **1st detection:** Real-time modal warning + brief un-hedge window. If not corrected → forced liquidation, account FLAGGED.
2. **2nd detection same day:** Account blocked rest of day. Temporary Hedging Violation across all hedged accounts.
3. **Next-day acknowledgment:** Forced "I agree" modal before any trading.
4. **Any subsequent offense:** **Permanent Hedging Violation. Accounts CLOSED.** No exceptions.

**Cited trader responsibility for automation** (`topstep_cross_account_hedging.md:273-288`):
> "You remain fully responsible for all activity across your accounts, including positions created through... Automated trading systems. Any third-party tools."

**Code reality:** `trading_app/risk_manager.py` has `max_concurrent_positions`, `max_per_orb_positions`, `max_per_session_positions` — none check for opposing direction on same instrument. No grep hits for `same_instrument_opposite|opposing.*direction|hedge` in `trading_app/`.

**Bot risk profile (active deployment `topstep_50k_mnq_auto`):**
- 5 daily lanes, 4 of which are MNQ (`SINGAPORE_OPEN`, `COMEX_SETTLE`, `EUROPE_FLOW`, `TOKYO_OPEN`).
- E2 entry direction is determined by ORB break direction (above ORB high = LONG, below = SHORT).
- Different sessions can independently break in opposite directions on the same day.
- Hold periods for E2 trades are typically 30-120 min. Sessions are spread across 24h but **adjacent sessions can overlap**.
- **If MNQ COMEX_SETTLE enters LONG and MNQ EUROPE_FLOW enters SHORT 30 minutes later while the COMEX_SETTLE position is still open → intra-account opposing position → matches the canonical hedge definition.**
- Because `copies=2`, both XFAs hold the same long+short → intra-account hedge in BOTH copies simultaneously.

**Required fix:** Pre-trade gate that refuses entry into a position opposite an existing open position on the same instrument within the same account, OR closes the existing position before entering the new one. Plus a runtime monitor that fails-closed if the bot ever observes itself simultaneously net-long and net-short the same instrument in any active TopStep XFA copy.

**Followup investigation:** Run a 1-day historical replay of the active 5 lanes against the past 90 days of data to count how many days WOULD have produced opposing-direction overlaps. If "≥1," we have a smoking gun.

---

## 🟡 F-2b — HIGH: CopyOrderRouter shadow failure asymmetry

**Status:** OPEN
**Severity:** 🟡 HIGH (NEW finding from re-verification)

**Source:** [`trading_app/live/copy_order_router.py:41-71`](../../trading_app/live/copy_order_router.py)

The bot uses `CopyOrderRouter` (project-side fan-out via independent ProjectX API connections per account) NOT TopstepX's native Trade Copier UI. Per the docstring:
> "ONE auth token → ONE DataFeed → ONE SessionOrchestrator → CopyOrderRouter. CopyOrderRouter wraps N ProjectXOrderRouters (one per account_id). submit() → primary first, then shadows. Returns primary result."

**The bug** (`copy_order_router.py:57-71`):
```python
for shadow in self.shadows:
    try:
        shadow_result = shadow.submit(spec)
        log.info("Shadow copy account %s: %s", ...)
    except Exception:
        log.warning("Shadow copy FAILED account %s — primary unaffected", ...)
```

Shadow failures are LOGGED but NOT propagated. Primary fills, shadow doesn't. **Account state diverges between copies.**

**Concrete failure mode:**
1. Lane A enters MNQ LONG. Submit to primary → fills. Submit to shadow → broker rejects (e.g., transient rate limit). Logged as warning. Primary has long, shadow has nothing.
2. Lane A's exit signal fires. Submit to primary → flattens primary. Submit to shadow → tries to flatten a position that doesn't exist → broker rejects or, worse, opens an opposite position depending on broker semantics.
3. Now primary is flat, shadow holds an unintended position.
4. If a different lane on the same instrument later enters in the OPPOSITE direction, the shadow now has an opposing position vs the primary → **cross-account hedge per F-2 canonical rules**.

**Required fix:** Either (a) make shadow failures fatal (rollback primary), (b) reconcile positions per-account periodically and force-flatten any divergence, or (c) gate entries on a "all copies in sync" precondition. Recommended: (b) periodic reconciliation in `session_orchestrator.py` with halt-on-divergence.

---

## 🔴 F-3 — DEFERRED (pre-LFA): Live Funded DLL not modeled

**Status:** DEFERRED (no LFA today, blocker at LFA promotion)
**Severity:** 🔴 BLOCKER at LFA-promotion time, ZERO blast radius today

**Canonical source:** [`docs/research-input/topstep/topstep_live_funded_parameters.md:122-130`](../research-input/topstep/topstep_live_funded_parameters.md)

**Cited rule:**
> "Live Funded Accounts begin with a Daily Loss Limit based on account size:
> - $2,000 for $50K accounts
> - $3,000 for $100K accounts
> - $4,500 for $150K accounts.
>
> Regardless of account size, if the tradable balance reaches $10,000 or below, the Daily Loss Limit will be set to $2,000."

**Cross-checked against:** [`topstep_dynamic_live_risk_expansion.md:50-58`](../research-input/topstep/topstep_dynamic_live_risk_expansion.md) — same starting values, with progressive expansion as profits grow (10 active trading days per tier).

**Crucial observation:** LFA DLL == LFA MLL at the starting tier. A single bad trading day equal to MLL = simultaneous DLL hit (soft breach) AND MLL hit (hard rule break) → **account permanently closed**, not just paused.

**Code reality:**
- `prop_profiles.py:251-253` ACCOUNT_TIERS sets `daily_loss_limit=None` (constructor default) for all TopStep tiers.
- `account_hwm_tracker.py:89` SUPPORTS `daily_loss_limit` constructor param but `session_orchestrator.py:410-416` never passes it.
- No `prop_firm_policies.py` field captures LFA DLL.
- No code anywhere knows about Dynamic Live Risk Expansion.

**Today's blast radius:** ZERO. Per `topstep_payout_policy.txt:280`: *"Most Traders are called to Live after five Express Funded Account payouts (around 25 winning days)"*. The user has no LFA.

**Required fix (deferred):** Add LFA-specific tracking. Differentiate `topstep_express_*` vs `topstep_live_*` profiles. Wire `daily_loss_limit` parameter into `AccountHWMTracker` for LFA profiles. Model the DLRE tier ladder and the $10K-tradable-balance override.

---

## 🟡 F-4 — HIGH: Commission rates 8-13% LOW for MNQ/MES vs canonical

**Status:** OPEN
**Severity:** 🟡 HIGH (escalated from initial audit — affects backtest accuracy across the entire active portfolio)

**Canonical source:** [`docs/research-input/topstep/topstep_xfa_commissions.md`](../research-input/topstep/topstep_xfa_commissions.md) (article 8284213, "Updated today" per header, *"Fees updated as of May 12th, 2025"*)

| Instrument | TopStep Tradovate RT | TopStep Rithmic RT | `cost_model.py` line | Code Status |
|---|---|---|---|---|
| Micro Gold (MGC) | $1.64 | $1.72 | $1.74 (line 74) | Conservative ✓ |
| Micro NASDAQ 100 (MNQ) | **$1.34** | **$1.42** | $1.24 (line 93) | **8-15% LOW** ⚠️ |
| Micro S&P 500 (MES) | **$1.34** | **$1.42** | $1.24 (line 102) | **8-15% LOW** ⚠️ |
| Micro Crude Oil (MCL) | $1.64 | $1.72 | $1.24 (line 111) | LOW (instrument dead) |
| Micro Silver (SIL) | $2.64 | $2.72 | $1.24 (line 120) | **2x LOW** (instrument dead) |
| Micro EUR/USD (M6E) | $1.12 | $1.20 | $1.24 (line 129) | OK (slightly conservative, dead) |
| Micro E-mini Russell 2000 (M2K) | $1.34 | $1.42 | $1.24 (line 138) | LOW (instrument dead) |
| Micro E-mini Bitcoin (MBT) | $5.64 | $5.72 | $2.50 (line 147) | **2.3x LOW** (instrument dead) |

**Active impact:** MNQ is the workhorse instrument for `topstep_50k_mnq_auto` (4 of 5 lanes are MNQ). Backtest understates MNQ commission by $0.10–$0.18 per RT.

**Source-of-truth issue:** Code uses **one value per instrument**, but TopStep's canonical doc shows **per-platform** rates. Tradovate, Rithmic, and Plus500 all differ. The bot's deployment platform determines the correct value:
- TopStep XFA via ProjectX/TopstepX → ?? (the article doesn't specifically list "TopstepX/ProjectX" — it lists the underlying clearing platforms)
- Tradeify via Tradovate → Tradovate rates
- Bulenox / Elite Trader Funding via Rithmic → Rithmic rates

**Required fix:** Either (a) add per-firm commission overrides to `cost_model.py`, (b) bump MNQ/MES baselines to the **Rithmic** value as the most conservative, or (c) verify TopstepX rate via direct experimentation. Recommend (b) in the short term and (a) in the medium term.

**Followup:** The article does not list TopstepX as a separate row. Either ProjectX uses one of the existing clearing rates, or the user's TopstepX commission is governed by a different article (the one I tried at `14363528` — 404'd today, may have been retired). Confirm with TopStep support if a fresh XFA gets a billed-fees email after first trade.

---

## 🟡 F-5 — MEDIUM: AccountHWMTracker `freeze_at_balance` formula wrong for XFA

**Status:** OPEN
**Severity:** 🟡 MEDIUM (safe direction — bot halts earlier than necessary, missed opportunity not violation)

**Canonical source:** [`docs/research-input/topstep/topstep_mll_article.md:60-66`](../research-input/topstep/topstep_mll_article.md)

**Cited rule:**
> "Express Funded Accounts work the same way, but start at a $0 balance. For a $50,000 Express Funded Account, your Maximum Loss Limit starts at -$2,000 and trails upward as your balance grows. Once your balance reaches $2,000, the Maximum Loss Limit stays at $0."

**Code reality** ([`trading_app/live/session_orchestrator.py:407-409`](../../trading_app/live/session_orchestrator.py)):
```python
if firm_spec.dd_type == "eod_trailing":
    freeze = prof.account_size + tier.max_dd + 100
```

For TopStep XFA 50K: `freeze = 50,000 + 2,000 + 100 = $52,100`.

**Bug:** This is the formula for a **Trading Combine** where starting equity = $50,000. For an **Express Funded Account**, starting equity = **$0** (per canonical). XFA peak balance won't reach $52,100 unless you've made $52K of profit. So `_hwm_frozen` never trips → trailing DD continues advancing forever instead of locking when XFA balance hits $2,000.

**Behavioral effect:**
- After $5K profit on XFA: tracker peak = $5K, trailing dd_limit = $2K → tracker halts at equity = $3K.
- Canonical rule: MLL is locked at $0 forever after first $2K profit reached. Account doesn't close until equity = $0.
- Tracker is **MORE conservative** than canonical (halts at $3K when canonical allows trading down to $0). Lost trading opportunity, not a violation.

**Required fix:** Add `is_express_funded: bool` field to `AccountProfile`, default True for active TopStep profiles (the bot only runs in XFA, never TC). Apply correct formula:
```python
freeze = (tier.max_dd + 100) if prof.is_express_funded else (prof.account_size + tier.max_dd + 100)
```

---

## 🟡 F-6 — MEDIUM: 5-XFA aggregate cap unenforced

**Status:** OPEN
**Severity:** 🟡 MEDIUM (no current violation; future activation risk)

**Canonical source:** [`docs/research-input/topstep/topstep_xfa_parameters.txt:35,222`](../research-input/topstep/topstep_xfa_parameters.txt)

**Cited rule:**
> "You can have up to 5 active Express Funded Accounts at the same time."

**Code reality:** ACCOUNT_PROFILES has 4 TopStep profiles with `copies` ranging from 2 to 5. No code asserts `sum(copies for p in ACCOUNT_PROFILES.values() if p.active and p.firm == 'topstep') ≤ 5`.

**Today:** Only `topstep_50k_mnq_auto` is active (copies=2). Within cap. ✓
**Risk:** Activating a second TopStep profile without disabling the first → cap violation.

**Required fix:** Startup assertion in `pre_session_check.py` (or wherever the bot launches) that asserts the active TopStep copies sum ≤ 5. Trivial implementation (~10 lines).

---

## 🟡 F-7 — REVISED: TopstepX Trade Copier rules NOT applicable (replaced by F-2b)

**Status:** REVISED (initial finding was wrong — replaced by F-2b)

**Initial concern:** TopstepX native Trade Copier requires "Lead account must have the lowest Maximum Position Size" per `topstep_topstepx_general.md:253`.

**Re-verification:** The bot uses `trading_app/live/copy_order_router.py` (per-account ProjectX dispatch), NOT the TopstepX UI Trade Copier. The "Lead lowest size" rule is a constraint of the UI feature only — bot bypasses it entirely. New finding F-2b replaces this concern with the actually-applicable risk (asymmetric account state from shadow failures).

---

## 🟡 F-8 — MEDIUM: Vol-event temporary risk adjustments unmodeled

**Status:** OPEN
**Severity:** 🟡 MEDIUM (rare but catastrophic)

**Canonical source:** [`docs/research-input/topstep/topstep_risk_adjustments.md:47-95`](../research-input/topstep/topstep_risk_adjustments.md)

During high-volatility events, TopStep imposes additional caps on Express Funded Accounts:

**Per lot tier (general instruments):**
- 2-3 lots → 1 micro
- 5 lots → 2 micros
- 10 lots → 5 micros
- 15 lots → 7 micros

**MGC special during vol events:**
- 50K XFA → 5 micros (current Scaling Plan limit)
- 100K XFA → 10 micros
- 150K XFA → 15 micros

**Mini-contract trading may be temporarily halted** on affected products.

**Code reality:** No vol-event detection or override logic. Bot doesn't poll TopstepX for active risk adjustments.

**Real-world impact:** Vol events are rare but happen (Aug 2024 yen carry unwind, March 2020, news shocks). A vol event hitting during the trading day with the bot at MAX contracts → instant violation. Bot would learn about this only via account-review email after the fact.

**Required fix:** Either (a) daily check of TopStep dashboard for active risk adjustments and respect them, or (b) accept the residual risk and add to runbook as "manual halt during major vol events." Option (b) is acceptable if the user agrees.

---

## 🟢 F-9 — REFUTED: DLL exemption claim verified TRUE

**Status:** REFUTED initial finding (code claim is correct)

**Canonical source:** [`docs/research-input/topstep/topstep_dll_article.md:39-78`](../research-input/topstep/topstep_dll_article.md)

**Cited rule:**
> "The Daily Loss Limit should be viewed as a safety net. It's a risk feature that can be turned on and off in your Trading Combine or Express Funded Account, but will automatically be applied to all Live Funded Accounts."
>
> "The Daily Loss Limit used to be automatically applied to all accounts. However, by removing the Daily Loss Limit on TopstepX™, we're giving traders more freedom and flexibility to trade their way."

**Verdict:** Code claim at `prop_profiles.py:248` ("DLL removed on TopstepX since Aug 25, 2024") is canonically correct. The exemption is REAL for TC and XFA on TopstepX. ToU Section 27 is outdated relative to the help-center, but the help-center is what the Risk team enforces.

**Action:** Add `@canonical-source` annotation to `prop_profiles.py:248` pointing to the canonical article URL + scrape date. The code claim is correct but unverifiable from the comment alone, which violates the Volatile Data Rule.

---

## 🟢 F-10 — CONFIRMED: All payout-policy fields verified

**Status:** CONFIRMED — no fix needed beyond annotation

| Field | Canonical Source | Code Location | Verdict |
|---|---|---|---|
| 90/10 flat profit split (Jan 12 2026 cutoff) | `topstep_payout_policy.txt:93-103` | `prop_profiles.py:165` | ✓ |
| Standard: 5 winning days @ $150+, 50%/$5K cap | `topstep_payout_policy.txt:26-32` | `prop_firm_policies.py:42-49` | ✓ |
| Consistency: 3 days, 40% rule, $6K cap | `topstep_xfa_parameters.txt:120,165` | `prop_firm_policies.py:60-67` | ✓ |
| Min payout $125 | `topstep_payout_policy.txt:63` | `prop_firm_policies.py:48,66` | ✓ |
| LFA: 30 winning days unlocks daily payouts | `topstep_payout_policy.txt:17` | `prop_firm_policies.py:85-86` | ✓ |
| Consistency formula = best_day / total_profit | `topstep_xfa_parameters.txt:148` | `consistency_tracker.py:137,246` | ✓ |
| Close time 3:10 PM CT (16:10 ET) | `topstep_terms_of_use.txt:368-372` | `prop_profiles.py:168` | ✓ |
| Force-flatten 5 min before close | n/a (project-side conservative buffer) | `session_orchestrator.py:1030-1040` | ✓ more conservative than firm 10s |

---

## 🟢 F-11 — INFERRED: Standard `additional_days_after_payout=5` not in canonical

**Status:** INFERRED (no explicit documentation; inference is reasonable)

**Code:** `prop_firm_policies.py:49`: `additional_days_after_payout=5` for `topstep_express_standard`.

**Canonical source:** I grepped `topstep_payout_policy.txt` for `after.*payout|additional.*day|next payout|new.*payout|cycle`. The canonical Standard payout doc does NOT explicitly state *"after payout, you need 5 NEW winning days before next payout"*. It only says the policy is "5 winning days of $150+." For Consistency, the doc DOES explicitly describe the after-payout reset (`topstep_payout_policy.txt:178-211`). Standard's after-payout behavior is implicit.

**Action:** Mark the value as `@inferred`, not `@canonical-source`. Note that post-payout reset for Standard is reasonable interpretation, not canonical fact.

---

## 🟢 F-12 — INFO: VPN/VPS prohibition (ToU Section 28)

**Status:** OPEN — needs user confirmation

**Canonical source:** [`topstep_terms_of_use.txt:383`](../research-input/topstep/topstep_terms_of_use.txt)

**Cited rule:**
> "Using any VPN or VPS on Accounts is strictly prohibited and may result in account termination and forfeiture of profits."

**Required action:** Confirm where the bot runs.
- ✓ Local Windows desktop on home internet → OK
- ✗ AWS / Linode / Hetzner / DigitalOcean / any cloud VM → VPS by definition → would void any account it touches

---

## 🟢 F-13 — INFO: Account stacking explicitly prohibited

**Status:** Behavioral note, no code change

**Canonical source:** [`topstep_prohibited_conduct_helpcenter.md:98`](../research-input/topstep/topstep_prohibited_conduct_helpcenter.md)

**Cited rule:**
> "Account stacking. This is a trading practice where a trader repeatedly trades in an aggressive way, then hits the Maximum Loss Limit in one account, and switches to another account to repeat the process."

**Bot risk:** If `topstep_50k_mnq_auto` ever blows MLL on one of its 2 copies, and the user immediately spins up a fresh XFA to replace the dead one, the pattern matches. TopStep's risk team would notice. Add to operator runbook: do NOT replace blown XFAs immediately.

---

## 🟢 F-14 — INFO: 30-day inactivity closes XFA

**Status:** Operational reminder

**Canonical source:** [`topstep_funded_rule_violation_faq.md:54-58`](../research-input/topstep/topstep_funded_rule_violation_faq.md)

**Cited rule:**
> "You can take a break from trading your Express Funded Account for less than 30 days at a time. If there is no trading activity (no trades entered) on your Express Funded Account for more than 30 days, it may be subject to closure due to inactivity."

**Bot risk:** If the bot is paused (manual or programmatic halt) for 30+ days, the XFA expires. Worth a calendar reminder or a heartbeat check.

---

## 🟢 F-15 — INFO: Back2Funded gives up to 2 reactivations per XFA

**Status:** Cost-modeling note

**Canonical source:** [`topstep_funded_rule_violation_faq.md:33-35`](../research-input/topstep/topstep_funded_rule_violation_faq.md)

**Cited rule:**
> "Starting September 3rd, 2025, if you lose your Express Funded Account due to a rule break, Back2Funded gives you the option to reactivate it up to 2 times. Each reactivation keeps the same account size and payout policy, so you can continue trading toward a payout without starting over in the Trading Combine."

**Bot benefit:** A bot-caused account blow is recoverable up to 2x without paying a new Combine fee. Worth knowing for the worst-case cost calc when modeling deployment economics.

---

# Open questions for the user

1. **Joining date:** When did you sign up for TopStep? If before 2026-01-12 → legacy 100%/$10K policy applies, code is wrong. If on/after → flat 90/10 is correct.
2. **Active XFA status:** Is `topstep_50k_mnq_auto` running on a live XFA right now, or only a Trading Combine, or only in dry-run mode? This determines whether F-1 (Scaling Plan) is an active violation or future-state risk.
3. **Hosting:** Is the bot on home internet (✓) or any cloud VPS (✗ per F-12)?
4. **Fix scope:** Which findings do you want fixed in the next pass?
   - **Scope A (minimum-safe):** F-1 + F-2 + F-2b (the 3 BLOCKERs/HIGHs that affect active deployment)
   - **Scope B (recommended):** A + F-4 + F-5 + F-6 (small surgical fixes that close all active-deployment risks)
   - **Scope C (full):** B + F-3 stub (LFA-aware profile fields, no behavior yet) + F-8 acceptance decision

# Followup canonical fetches needed

- `https://intercom.help/topstep-llc/en/articles/8284209` — Net position calculation across products (needed for F-1 enforcement of simultaneous long+short)
- `https://help.topstep.com/en/articles/8284217-express-funded-account-activation` — XFA activation rules (might have lead/follower platform binding info)
- `https://help.topstep.com/en/articles/12060405-back2funded-rules-guidelines-and-how-it-works` — Back2Funded mechanics
- The TopstepX commissions article that 404'd today (`14363528`) — try alternate URL or contact support to confirm TopstepX-specific RT rate

# Status tracker

Updated 2026-04-08 after stages 1-8 of `docs/plans/2026-04-08-topstep-canonical-fixes.md`:

```
F-1  Scaling Plan Day 1 violation              [FIXED]     stage 7  140fcf3
F-2  Single-user hedging guard                 [FIXED]     stage 6  4a75e89
F-2b CopyOrderRouter shadow failure asymmetry  [FIXED]     stage 5  ffd12b1
F-3  LFA DLL not modeled                       [DEFERRED]  (no LFA today; stub field added in stage 4)
F-4  MNQ/MES commissions LOW                   [FIXED]     stage 2  4a9163e
F-5  XFA HWM freeze formula bug                [FIXED]     stage 4  9226e7c
F-6  5-XFA aggregate cap unenforced            [FIXED]     stage 3  9f8ce4e
F-7  Trade Copier rules                        [REVISED]   replaced by F-2b
F-8  Vol-event risk adjustments                [DEFERRED]  operator runbook entry, not automation
F-9  DLL exemption claim                       [ANNOTATED] stage 1  923c5ce  (code was correct)
F-10 Payout-policy fields verified             [ANNOTATED] stage 1  923c5ce
F-11 Standard additional_days_after_payout     [ANNOTATED] stage 1  923c5ce  (marked @inferred)
F-12 VPN/VPS prohibition                       [USER CONFIRMED] home hosting OK
F-13 Account stacking prohibited               [DOCUMENTED] behavioral note
F-14 30-day inactivity                         [DOCUMENTED] operational reminder
F-15 Back2Funded reactivations                 [DOCUMENTED] cost modeling note

Drift check 92 (added stage 8, commit 21b2b64) validates every @canonical-source
annotation in production code. Total drift count now 85 + 7 advisory.
```

## Final follow-ups (out of scope for stages 1-8)

1. **Wire Scaling Plan into orchestrator** — `RiskLimits.topstep_xfa_account_size` is set to None today, so the F-1 check is dormant. To activate, the orchestrator must:
   - Read `prof.account_size` for active TopStep profiles and pass via `RiskLimits(topstep_xfa_account_size=...)`
   - Call `risk_manager.set_topstep_xfa_eod_balance(broker_equity)` at session start (from HWM tracker `last_equity`)

   This was deferred to keep stage 7's diff surgical. The check is fully implemented and tested; only the wiring is pending. **Must be done before going live with a real XFA.**

2. **Net position calculation article** — `https://intercom.help/topstep-llc/en/articles/8284209` is referenced from the canonical Scaling Plan article but not yet fetched. Stage 7's `total_open_lots` uses GROSS exposure (conservative). If the canonical rule turns out to use NET (long minus short), the function should be updated and re-tested.

3. **Quarterly re-scrape** — next: 2026-07-08. Re-run Firecrawl on all 17 help-center articles in `docs/research-input/topstep/`. Drift check 92 will catch any annotation that points to a renamed/removed file.
