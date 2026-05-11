# family_singleton doctrine — analysis and disposition options

**Author:** Claude Code (read-only analysis; no DB or production-code writes)
**Date:** 2026-05-11 (initial); 2026-05-11 post-pick self-audit corrections applied
**Stage doc:** `docs/runtime/stages/stage2-family-singleton-doctrine.md`
**Companion to:** Stage 1 PR #258 (`stage1/generalize-tbbo-slippage-inference`).
**Decision-owner:** User. This document does NOT recommend a disposition.

**THIRD-PASS CORRECTION (2026-05-11):** the entire "Criterion 5 not enforced"
framing in this document was WRONG. `pre_registered_criteria.md` Amendment
2.1 (2026-04-07) explicitly downgraded C5 from binding to **CROSS-CHECK
only** until N_eff is formally solved in-repo. The current state (zero
strategies pass DSR ≥ 0.95) is **the doctrine working as designed**, not
drift. Direct quote from Amendment 2.1: *"A policy that makes DSR binding
while the repo computes DSR against an unknown N_eff is a policy that
rejects every strategy the repo has ever produced — including ones that
may be valid."* Additional facts I missed:
- `trading_app/dsr.py:35` carries explicit "Until N_eff is properly
  estimated, DSR is INFORMATIONAL, not a hard gate."
- `trading_app/strategy_validator.py:2316` uses `N_eff = 527` (count
  distinct family_hash), NOT the brute-force discovery K I assumed. So
  the "K=8856 → DSR=0" mechanism I proposed was misdiagnosis.
- Per-trade Sharpe across the active shelf (0.09-0.18) is below
  `sr0 = 0.4831` (the noise floor at N_eff=527, V[SR]_E2=0.0248). DSR≈0
  is a per-trade-Sharpe-vs-noise-bar problem, not an N_eff bug.

Stage 3 reframe consequence: option (b) "re-derive discovery_k" is the
WRONG fix because discovery_k is not what the formula consumes. The
honest Stage 3 question is much narrower than I framed: should
`family_singleton` deployability gate require DSR-as-cross-check
satisfied OR something else as the floor?

**Post-pick self-audit summary (2026-05-11 second pass):**
After the user picked Disposition C, a self-audit per `.claude/rules/institutional-rigor.md`
§ 1-2 (review the work, review the fix) caught five issues in the original
draft, all corrected in-place with diff visible:
1. § 1 PURGED definition was too narrow — PURGED is the `classify_family()`
   fall-through default and is a heterogeneous bucket. Corrected.
2. § 2.5 Carver Ch 4 citation was unverifiable — no local extract exists.
   Treated as "project convention with unverified citation."
3. Disposition A's Carver citation softened to match § 2.5.
4. Disposition D's "Carver Ch 11 diversification multiplier" citation was a
   conflation of two distinct frameworks. Removed; D now framed as
   project-invented heuristic motivated by H-L intuition.
5. § 3.5 "DSR drift" framing was misleading — DSR≈0 is the formula
   computing correctly given brute-force discovery K. The honest issue is
   a doctrine-integrity question (C5 locked but not satisfied by ANY current
   deployment, including the live 3-lane MNQ portfolio). Reframed.
6. § 3.3 + Disposition C unlock estimates corrected — all 5 candidate MES
   rows ALSO carry `c8_missing` and `slippage_missing` hard-blockers; the
   "blocked only by family_singleton" framing was wrong on first draft.
   Realistic post-Stage-1 unlock is **zero**; even with C5 + C8 resolved,
   ≤ 5 pending separate workstreams.

The disposition pick itself (Disposition C with locked-criteria floor)
remains correct, but the dependency chain is deeper than initially stated:
Stage 3 must address DSR/C5 policy AND C8 backfill BEFORE Stage 4's
conditional-downgrade code is implementable in a non-no-op way.

---

## Scope

This document is a doctrine analysis for the `family_singleton` deployability
gate. It is read-only; it produces a recommendation set, not a code change.
The Verdict section lists the dispositions for the user to pick from — no
single recommendation is made here.

## 1. The doctrine question

`trading_app/deployability.py:539` raises a HARD blocker (`family_singleton`,
verdict `BLOCKED_FAMILY_FRAGILE`) whenever the strategy's `edge_families.robustness_status`
is `SINGLETON`. The same verdict bucket is used for `robustness_status='PURGED'`
(line 537) — i.e. PURGED and SINGLETON are treated as equivalent deployment
blockers despite being **asymmetric evidence states**:

- **PURGED** = the **fall-through default** of `classify_family()` per
  `trading_app/edge_families.py:67`. ANY family that fails the ROBUST,
  WHITELISTED, AND SINGLETON tests gets PURGED. This is a heterogeneous
  bucket including (i) 2-member families that fail WHITELISTED's CV/N
  thresholds, (ii) singletons that fail the SINGLETON thresholds
  (member_count==1 but min_trades<100 OR avg_shann<0.8), AND (iii)
  families that "got tested with peers and were rejected." It is NOT
  homogeneously "rejected family."
- **SINGLETON** = `member_count == 1 AND min_trades ≥ 100 AND avg_shann ≥ 0.8`
  per `trading_app/edge_families.py:59-66`. The family **has not been rejected**;
  it simply has only one variation tested AND that one variation cleared the
  per-strategy individual-quality thresholds. So no peer-cross-check is
  available BUT the lone member is individually well-evidenced.

The doctrine question:

> Is the absence of peer-cross-check evidence (SINGLETON) sufficient grounds for
> a hard deployment block when the strategy passes every individual-strategy
> criterion (C1 pre-reg, C3 BH-FDR, C4 t-stat, C5 DSR, C6 WFE, C7 N, C8 OOS,
> C9 era stability, C11 account-risk Monte Carlo, C12 SR monitor)?

A "yes" answer treats peer-evidence as **non-substitutable** by individual
evidence. A "no" answer treats peer-evidence as **redundant with** sufficiently
strong individual evidence (the Harvey-Liu position). A third path treats it as
**additive** — discounted-Sharpe / reduced position size / longer probation —
rather than binary.

---

## 2. Literature anchors

### 2.1 Harvey & Liu 2015 — individual-strategy Sharpe haircut

`docs/institutional/literature/harvey_liu_2015_backtesting.md`:

> "We argue that it is a serious mistake to use the usual 50% haircut. Our
> results show that the multiple testing haircut is nonlinear. The highest
> Sharpe ratios are only moderately penalized, while the marginal Sharpe ratios
> are heavily penalized."

H-L treat individual-strategy multiple-testing penalty as a **continuous Sharpe
discount**, not a binary block. Their `HSR` (haircut Sharpe ratio) is the
admissible decision variable; a strategy with `HSR > 0` after the haircut
remains deployable, just at lower expected utility.

H-L do not address peer-cross-check; their framework is entirely individual.
So H-L is silent on whether family-of-1 is a separate problem.

### 2.2 Bailey-López de Prado 2014 — Deflated Sharpe Ratio

`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`:

DSR is the canonical individual-strategy overfitting penalty in this repo's
Criterion 5 (`DSR > 0.95`). It bakes in N_trials, kurtosis, skew, and sample
size. Two strategies with the same raw Sharpe and different N_trials get
different DSRs.

DSR is computed per-strategy from per-strategy statistics. It does **not**
require peer evidence.

### 2.3 López de Prado-Bailey 2018 — False Strategy Theorem

`docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md`:

The False Strategy Theorem proves that with sufficient backtest count, an
arbitrary positive-Sharpe finding is expected by chance. This is the
foundational argument for DSR / MinBTL / BH-FDR — **all three are individual-
strategy gates**. The False Strategy Theorem does not introduce a separate
peer-cross-check requirement.

### 2.4 Bailey et al 2014 — PBO §4.2 (SINGLETON_MIN_SHANN source)

Cited in `trading_app/edge_families.py:19-22`. ShANN = Shannon entropy of
per-period returns. The 0.8 threshold is the PBO paper's empirical cut for
"diversified return paths." It is applied per-strategy in the SINGLETON
classifier — not a peer-cross-check threshold.

### 2.5 Carver 2015 Ch 4 — member consistency (WHITELIST_MAX_CV source)

**Citation status: UNVERIFIED locally.** `trading_app/edge_families.py:22`
comments cite "Carver Systematic Trading Ch.4 (member consistency)" as the
source of the `WHITELIST_MAX_CV = 0.5` threshold. No local extract of Carver
Ch 4 exists in `docs/institutional/literature/`; only Ch 11 (portfolios) and
Ch 12 (speed-and-size) are extracted. Neither extract contains the
member-consistency argument. `resources/Systematic_Trading_Carver.pdf` is
listed in `.claude/rules/quant-audit-protocol.md` but does NOT currently
exist in `resources/`.

So Carver Ch 4 is referenced in code but cannot be verified against a local
source from this analysis. The institutional-rigor rule (§ 7
"Extract-before-dismiss") means we should not load-bear on this citation
without retrieving the PDF and extracting the relevant passage. For this
Stage 2 analysis, I treat the family-member-consistency framework as a
**project convention with a stated-but-unverified literature citation**.

The qualitative point stands regardless of citation status: family-member
consistency (whatever its literature provenance) is undefined when
member_count == 1. Disposition decisions about SINGLETONs cannot lean on
Carver Ch 4 because the framework requires peer evidence by construction.

### 2.6 What the literature does NOT say

No anchor file in `docs/institutional/literature/` contains a rule of the form
"a strategy whose family has member_count == 1 must not be deployed even if
its individual evidence is strong." The hard-block doctrine encoded at
`deployability.py:539` is a **project convention**, not a literature-grounded
gate.

`pre_registered_criteria.md` does not list family_singleton among the 12
locked criteria. The closest is criterion 9 (era stability) and criterion 5
(DSR > 0.95), both individual-strategy.

---

## 3. Empirical landscape

### 3.1 Universe distribution

`SELECT vs.instrument, ef.robustness_status, COUNT(*) ... WHERE status='active'`:

| Instrument | PURGED | SINGLETON | WHITELISTED | ROBUST |
|---|---|---|---|---|
| MES | 26 | **22** | 0 | 0 |
| MGC | 8 | 0 | 0 | 5 |
| MNQ | 273 | **254** | 224 | 35 |

SINGLETON-status active rows total: 276 (22 MES + 254 MNQ + 0 MGC).

### 3.2 Individual-evidence quality among SINGLETONs

`SELECT COUNT(*) FROM validated_setups vs JOIN edge_families ef
 WHERE status='active' AND robustness_status='SINGLETON'
 AND fdr_significant AND wfe>=0.50 AND sample_size>=100
 AND expectancy_r>0 AND oos_exp_r>0 AND SIGN(expectancy_r)=SIGN(oos_exp_r)`:

- MNQ SINGLETON rows passing **all** of FDR + WFE ≥ 0.50 + N ≥ 100 + positive
  OOS + IS/OOS sign-match: **254 / 254** (100%).
- MES SINGLETON rows passing same: **22 / 22** (100%).

Interpretation: every active SINGLETON has already cleared `strategy_validator`
which enforces those gates as a precondition for `status='active'`. So
`family_singleton` is NOT keeping out individually-poor strategies; it is
keeping out strategies that lack peer-cross-check.

### 3.3 The 5 candidate MES rows blocked ONLY by family_singleton (+ slippage_missing pre-Stage-1)

After Stage 1 lands, two rows lose `slippage_missing` and are blocked solely by
`family_singleton`; three more retain `slippage_missing` because US_DATA_1000 is
not in the MES pilot v1 scope.

| strategy_id | N | ExpR | OOS | WFE | DSR | q | yrs |
|---|---|---|---|---|---|---|---|
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10 | 266 | 0.126 | 0.110 | 0.61 | 0.00 | 0.0401 | 6.0 |
| MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075 | 269 | 0.112 | 0.091 | 0.55 | 0.00 | 0.0372 | 6.0 |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15 | 905 | 0.102 | 0.116 | 1.14 | 0.00 | 0.0233 | 6.0 |
| MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15 | 1109 | 0.087 | 0.094 | 1.26 | 0.00 | 0.0321 | 6.0 |
| MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15 | 1033 | 0.090 | 0.102 | 1.28 | 0.00 | 0.0328 | 6.0 |

Three of the five have N ≈ 900–1100 and WFE > 1.0 (OOS Sharpe ≥ IS Sharpe);
all five have positive OOS, IS/OOS sign-match, BH-FDR q < 0.05, and 6 years
of test data. Slippage on the COMEX_SETTLE rows is covered by the 2026-04-24
MES pilot v1 PASS doc; the US_DATA_1000 rows still need `slippage_missing`
resolved separately (not in scope for Stage 2).

**ADDITIONAL HARD BLOCKERS BEYOND family_singleton (verified 2026-05-11
self-audit):** all 5 rows have `c8_oos_status IS NULL` and `slippage_validation_status
IS NULL`. The deployability gate hard-blocks on both. The "blocked only by
family_singleton" framing in § 3.3's header was misleading on first draft;
honestly, these rows are blocked by **family_singleton + c8_missing**
unconditionally, and ALSO `slippage_missing` until Stage 1's pilot v1
registry catches them. The "Stage 1 unblocks 2" claim is correct for the
slippage axis only; both rows retain `c8_missing` until C8 evidence is
backfilled. So the immediate post-Stage-1 MES verdict-flip is ALSO not
sufficient to deploy any MES row — only the verdict bucket changes.
The 5 candidates remain BLOCKED on multiple grounds; family_singleton is
one of several. Disposition C alone unlocks zero rows even after DSR is
resolved, until C8 evidence is also backfilled.

`validation_pathway = family` on all 5 rows (Pathway A — BH-FDR exploratory
search, not Pathway B theory-driven K=1). This matters for Harvey-Liu
applicability: H-L's continuous-haircut framing is most defensible for
theory-driven candidates; for family-pathway candidates it requires the
BH-FDR family K to be the multiple-testing penalty (rather than upstream
brute-force K). Connects to § 3.5 honest framing.

### 3.4 The currently-deployed MNQ baseline

`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` member_count=4, robustness=WHITELISTED.
`MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` member_count=3, robustness=WHITELISTED.
`MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` member_count=3, robustness=WHITELISTED.

The 3 live MNQ lanes are all WHITELISTED. The portfolio has never deployed a
SINGLETON lane; Stage 2 is the first time the doctrine question is genuinely
under the microscope.

### 3.5 DSR drift — flagged not fixed

```
SELECT instrument, COUNT(*) AS active, AVG(dsr_score), MAX(dsr_score)
FROM validated_setups WHERE status='active' GROUP BY 1
```

| Instrument | active rows | avg DSR | max DSR |
|---|---|---|---|
| MES | 48 | 0.319 | 0.941 |
| MGC | 13 | (small sample) | < 0.95 |
| MNQ | 786 | 0.005 | 0.712 |

`SELECT COUNT(*) WHERE status='active' AND dsr_score >= 0.95`: **0**.

The locked Criterion 5 (`DSR > 0.95`) is **not being enforced** on any deployed
lane. The 3 currently-live MNQ lanes have DSR effectively zero:

- MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100: DSR = 7.6e-12
- MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12: DSR = 0.0
- MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15: DSR = 1.1e-16

### THIRD-PASS CORRECTION — § 3.5 framing was wrong

The framing below ("DSR drift is partially a misnomer") was a step in the
right direction but still wrong. The correct framing per the doctrine
itself (`pre_registered_criteria.md` Amendment 2.1, 2026-04-07): DSR has
been **OFFICIALLY DOWNGRADED** from binding gate to cross-check, with
explicit reasoning that making DSR binding under unresolved N_eff would
"reject every strategy the repo has ever produced." So:

- C5 is NOT "non-operational" — it is officially a cross-check.
- The DB state (zero rows pass DSR≥0.95) is exactly what Amendment 2.1
  anticipated and accepted.
- No "doctrine drift" exists between code and pre-reg on this point.
- The 3 deployed MNQ lanes are not in violation of locked doctrine.

What this means for Stage 3:
- Path (a)/(b)/(c) below was framed against a non-existent problem.
- The actual Stage 3 question is **smaller**: Disposition C's floor
  ("pass all locked criteria") needs an honest read of what those locked
  criteria actually require. C5-as-cross-check means singletons clearing
  Disposition C's floor must have DSR **computed and reported**, but
  NOT pass DSR ≥ 0.95.
- Real binding criteria for Disposition C's floor (from §§ "Required
  threshold" lines that remain binding): C1, C2, C3, C4, C6, C7, C8,
  C9, C10, C11, C12. C5 stays cross-check.

The text below is retained for the audit trail but should be read with
this correction in mind.

### Original (incorrect) framing follows
### Honest framing — "DSR drift" is partially a misnomer

A first read frames DSR≈0 across the shelf as "drift" or "bug." On re-read
with `validated_setups.discovery_k` examined:

- The 5 candidate MES rows carry `discovery_k = 8856` (COMEX_SETTLE) and
  `9568` (US_DATA_1000).
- The 3 currently-deployed MNQ lanes were promoted under VALIDATOR_NATIVE
  provenance with comparable brute-force K from the pre-Phase-0 era.
- Bailey-LdP 2014 Eq. 2 deflation grows with `(1-γ)·Z⁻¹[1 - 1/N] + γ·Z⁻¹[1 - 1/(Ne)]`.
  At N=8856, the deflation factor is large enough that any finite Sharpe
  produces DSR≈0. **The formula is computing correctly given the input N.**

So the "Criterion 5 non-operational" state is not a numerical bug — it is
the formula correctly reporting that **strategies discovered via large
brute-force K cannot clear the 0.95 confidence threshold** at our data
horizon. This is exactly what Bailey 2013 MinBTL said would happen
(`docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` § "Applying
MinBTL to us"): at 35,000 trials on 2.2 years of clean data, the maximum
"trustable" Sharpe is 3.24; at 5,000-9,000 trials it's still well above
what these candidates achieve.

The "fix" for DSR is therefore NOT to recompute the formula with a different
N — it is to either:
(a) **Accept** that pre-Phase-0 grandfathered strategies cannot pass C5 by
    construction, and create an explicit "grandfathered-from-pre-reg-K"
    exception in the deployability gate. This is essentially Mode B from
    `research-truth-protocol.md`.
(b) **Re-derive** the discovery_k field to use pre-reg-family K (≤ 300)
    rather than upstream brute-force K, on the grounds that the relevant
    multiple-testing penalty is the pre-reg's, not the upstream scan's.
    Requires a doctrine decision separate from Stage 2.
(c) **Genuinely retire** the strategies that cannot clear C5 once
    recomputed — meaning the 3 deployed MNQ lanes themselves come back
    into question. This is the most institutionally honest path.

Stage 2 does NOT pick between (a)/(b)/(c). The relevance to the
family_singleton doctrine is: **no disposition can lean on "Criterion 5
already gates this" without first picking among (a)/(b)/(c).** Disposition
C's floor "passes all locked criteria" is therefore contingent on a separate
pre-decision.

Severity classification: this is a **doctrine integrity** issue (C5 is locked
in `pre_registered_criteria.md` and not enforced), not a code drift. Severity
HIGH. Not immediately capital-impacting because verdict-flips here are
reporting-only per `rebalance_lanes.py:109`. But it casts substantial
shadow on the current 3-lane MNQ deployment — those lanes do not clear C5
either, and there is no exception in the doctrine that authorises their
deployment despite that fact.

---

## Verdict

No single verdict — five dispositions (A–E) below, decision owed by the user.
Each disposition lists its empirical unlock count, literature support, pros,
cons, and Stage 3 follow-up. Section 8 restates the decision ask.

## 4. Disposition options

Each option lists the rule change, the MES rows it would unblock (post-Stage-1),
the MNQ rows it would unblock (post-Stage-1 + assuming individual gates pass),
the supporting literature, and the capital risk.

### Disposition A — STATUS QUO (family_singleton stays HARD)

- Rule: leave `deployability.py:83/103/539` unchanged.
- MES unblock: 0.
- MNQ unblock: 0.
- Literature support: family-member-consistency framework — its literature
  citation to Carver Ch 4 is unverified locally (§ 2.5). Treat as project
  convention rather than literature-anchored gate.
- Pros: Maximally conservative. No expansion of deployable shelf. No new
  capital surface area. Aligns with the project default of treating
  family-status as a load-bearing gate.
- Cons: The current code conflates PURGED (rejected family) with SINGLETON
  (no peer evidence), which is asymmetric evidence-handling. The 276
  SINGLETON-active rows are permanently dark even if they individually pass
  every other gate.
- Stage 3 follow-up if picked: optionally rename the verdict bucket to
  distinguish `BLOCKED_FAMILY_PURGED` from `BLOCKED_FAMILY_SINGLETON` for
  clarity, no semantic change. Otherwise, Stage 2 closes with no code edit.

### Disposition B — DOWNGRADE to WARNING, deploy on individual evidence

- Rule: change `deployability.py:539` from hard `family_singleton` to warning
  `family_singleton_warning`. Remove from `HARD_BLOCKER_TO_VERDICT` and
  `RETIRE_OR_PURGE_ISSUES`.
- MES unblock: 2 (the COMEX_SETTLE rows whose `slippage_missing` is now
  covered by Stage 1's registry pilot). 3 more US_DATA_1000 rows unblock if
  US_DATA_1000 is added to the MES slippage pilot v2 (separate stage).
- MNQ unblock: up to 254 SINGLETON-active rows. Subject to lane-correlation
  rho > 0.70 and 80% subset gates at `lane_correlation.py`; most will be
  filtered there. Realistic incremental deploy count is likely small (≤ 10).
- Literature support: Harvey-Liu's framework — individual-strategy multiple-
  testing penalty (BH-FDR + DSR + MinBTL) is treated as sufficient when
  satisfied; peer-cross-check is informative but not gating.
- Pros: Aligns the code with the project's actual literature. Treats SINGLETON
  as the asymmetric evidence state it is (no peer info), not as evidence of
  fragility. Unlocks the strongest-individual-evidence MES candidates.
- Cons: Expands the deployable universe without enforcing DSR ≥ 0.95
  (Criterion 5 drift, § 3.5). If DSR were operational, this disposition would
  be safer; absent that, individual-evidence enforcement is weaker than the
  pre_registered_criteria.md text claims.
- Required pre-conditions for safety: (i) flag DSR drift as a parallel debt
  with a HIGH severity, (ii) treat capital deployment of any individual
  SINGLETON as a per-strategy `/capital-review` decision rather than automatic
  allocator picks, (iii) the 2 MES COMEX_SETTLE rows that unblock immediately
  must still pass cross-instrument correlation against the deployed
  `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` (same session, different
  instrument — `lane_correlation.py` rho gate).
- Stage 3 follow-up if picked: implementation stage to make the code change
  with full test coverage on the verdict bucket transition + chronology
  fixture-pin per Stage 1 pattern.

### Disposition C — CONDITIONAL DOWNGRADE with individual-evidence floor

- Rule: `family_singleton` remains hard UNLESS the strategy passes the locked
  pre-registered criteria (`pre_registered_criteria.md` C1-C12) AND the
  SINGLETON classifier's own thresholds (`member_count == 1 AND min_trades >= 100
  AND avg_shann >= 0.8` per `edge_families.py:59-66`). No additional thresholds
  invented in this analysis; only doctrine values already locked.
- **AMEND 2026-05-11 post-pick:** an earlier draft of this disposition listed
  `wfe >= 0.70 AND sample_size >= 300 AND years_tested >= 6` as additional
  floor values. Those were NOT literature-grounded — they were "stricter than
  baseline" Claude-invented thresholds without a citation. Post-user audit
  the floor is corrected to "pass the locked Criteria 1-12 fully." The
  stage doc (`docs/runtime/stages/stage2-family-singleton-doctrine.md`)
  records the per-value audit against `docs/institutional/literature/`.
- MES unblock (current shelf state): **0**. All 5 candidate rows have
  `c8_oos_status IS NULL` (verified 2026-05-11) — so they hit `c8_missing`
  hard-blocker at `deployability.py:565-566` regardless of DSR status.
  Even if C5 path is decided (per § 3.5), C8 stays a separate hard block.
- MES unblock (if C5 path AND C8 backfill BOTH resolved): **at most 5**,
  pending the specific recomputed values. Two MES rows still carry
  `slippage_missing` for US_DATA_1000 (not in pilot v1), so realistically
  ≤ 2 immediately and ≤ 5 after MES slippage pilot v2.
- MNQ unblock (current shelf state): **0** (DSR identically zero across
  254 active MNQ singletons by the same K-arithmetic argument). Per-row
  C8 status not enumerated here.
- MNQ unblock (if C5 path AND C8 backfill resolved): unknown without a
  per-row re-audit. Realistically still small after lane-correlation gates.
- Literature support: Harvey-Liu Exhibit 4 hurdles (page 22) — strong individual
  Sharpe survives BHY haircut. Bailey-LdP DSR > 0.95 is the project's locked
  threshold for "highest Sharpe" tier.
- Pros: Most defensible position. Requires both individual + family evidence,
  but allows individual evidence to substitute for family if it crosses a
  high bar. Forces resolution of the DSR drift before any expansion.
- Cons: Effectively blocked-by-DSR-drift until DSR computation is fixed (a
  separate stage). The 2 MES COMEX_SETTLE candidates would NOT unblock even
  with this disposition until DSR is fixed.
- Required pre-conditions: DSR drift must be diagnosed and the formula
  recomputed (or grandfathered with explicit cohort treatment per
  `research-truth-protocol.md` Mode B note). This is a larger workstream
  than Stage 2 alone.
- Stage 3 follow-up if picked: (i) DSR re-computation stage, (ii) then the
  conditional-downgrade code change with floor tests.

### Disposition D — DOWNGRADE with sizing penalty (Harvey-Liu Sharpe haircut)

- Rule: `family_singleton` becomes a warning; deployed singletons get a
  per-strategy `position_size_multiplier < 1.0` via `trading_app/prop_profiles.py`
  or analogous. Carver Ch 11 has the relevant framework for sub-system sizing
  with confidence-discounted weights.
- MES unblock: same as B (2 immediately, 3 after pilot v2).
- MNQ unblock: same as B.
- Literature support: Harvey-Liu nonlinear haircut applied as a sizing
  penalty rather than a binary gate.
- **Carver Ch 11 citation removed (2026-05-11 audit):** a prior draft cited
  "Carver Ch 11 forecast combination with diversification multiplier" as
  support. On re-read of `docs/institutional/literature/carver_2015_ch11_portfolios.md`,
  Ch 11's diversification multiplier is a **portfolio-scaling-up-for-low-correlation**
  framework, NOT an "evidence-discounted sizing for low-confidence strategies"
  framework. The two ideas were conflated. Honest framing: no read literature
  anchor prescribes evidence-discounted position sizing for singletons; D's
  proposal is a project-invented heuristic motivated qualitatively by H-L's
  continuous-haircut intuition, not directly literature-grounded.
- Pros: Preserves Disposition B's unlock count while encoding the residual
  uncertainty as smaller position size — a continuous penalty matching the
  H-L framework.
- Cons: Sizing-multiplier infrastructure does not exist today as a per-lane
  metadata field on `validated_setups` or `prop_profiles`. Building it is a
  meaningful infra stage. Stage 2 alone cannot enact this.
- Required pre-conditions: per-lane sizing-multiplier field + allocator-honour
  the multiplier + monitoring on whether the discounted-size singletons
  perform in line with the discount (live forward test for the haircut).
- Stage 3 follow-up if picked: (i) sizing-multiplier infra stage, (ii) then
  the downgrade + multiplier code change.

### Disposition E — PARK Stage 2; pursue Disposition E from the MES survey

- The MES feasibility survey (Section 8, Option E) recommended parking the
  MES profile expansion entirely and pursuing operational scaling (Bulenox,
  MFFU) on existing MNQ deployed lanes. Stage 2 closes with "no doctrine
  change; family_singleton remains hard; operational scaling carries near-
  term EV without expanding the validated shelf."
- Rule: no code change. Doctrine stays as-is.
- MES unblock: 0.
- MNQ unblock: 0.
- Literature support: agnostic.
- Pros: Zero new code, zero new capital surface, zero new audit debt. The
  254 MNQ SINGLETON rows stay dark; the 22 MES SINGLETON rows stay dark.
- Cons: Forecloses 1 of 4 universal MES blockers permanently; the 5
  strong-individual-evidence MES candidates stay parked even though their
  individual evidence is arguably stronger than several deployed MNQ lanes.
- Stage 3 follow-up if picked: close Stage 2 worktree; HANDOFF notes the
  doctrine question was raised and parked; revisit only if MNQ universe
  decays meaningfully.

---

## Limitations

(See § 5 for full enumeration.)

## 5. What this analysis does NOT do

- Does NOT recommend a disposition. User picks.
- Does NOT propose a code change. Any code change is a separate IMPLEMENTATION
  stage with its own scope_lock, blast radius, and adversarial-audit gate.
- Does NOT touch `lane_allocation.json`, broker, schema, validated_setups, or
  any DB state.
- Does NOT decide whether DSR drift is in scope. It is flagged here for the
  user's awareness because every disposition's safety profile depends on it,
  but Stage 2 stops at "DSR drift exists and is HIGH severity."
- Does NOT re-derive the SINGLETON classifier thresholds (`MIN_TRADES=100`,
  `MIN_SHANN=0.8`). Those are downstream of Bailey-LdP 2014 PBO §4.2 and are
  outside Stage 2's doctrine question.
- Does NOT consider lane-correlation effects for the 2 MES COMEX_SETTLE
  candidates against `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`. That is a
  pre-deployment check, not a doctrine question.

---

## 6. Bailey-LdP MinBTL: this audit ran 0 trials. No brute force.

The analysis is read-only against `gold.db`, with all queries scoped to
`validated_setups`-joined-`edge_families`. No new strategies tested.

---

## Reproducibility

```python
import duckdb
con = duckdb.connect('C:/Users/joshd/canompx3/gold.db', read_only=True)

# Section 3.1 — universe distribution
con.execute("""
    SELECT vs.instrument, ef.robustness_status, COUNT(*)
    FROM validated_setups vs LEFT JOIN edge_families ef ON vs.family_hash=ef.family_hash
    WHERE vs.status='active' GROUP BY 1,2 ORDER BY 1,2
""").fetchall()

# Section 3.2 — SINGLETON individual-evidence pass-through
con.execute("""
    SELECT COUNT(*) FROM validated_setups vs
    JOIN edge_families ef ON vs.family_hash=ef.family_hash
    WHERE vs.status='active' AND vs.instrument='MNQ' AND ef.robustness_status='SINGLETON'
      AND vs.fdr_significant AND vs.wfe>=0.50 AND vs.sample_size>=100
      AND vs.expectancy_r>0 AND vs.oos_exp_r>0
      AND SIGN(vs.expectancy_r)=SIGN(vs.oos_exp_r)
""").fetchall()

# Section 3.3 — the 5 MES candidate rows
con.execute("""
    SELECT vs.strategy_id, vs.sample_size, vs.expectancy_r, vs.oos_exp_r,
           vs.wfe, vs.dsr_score, vs.fdr_adjusted_p, vs.years_tested,
           ef.member_count, ef.robustness_status
    FROM validated_setups vs LEFT JOIN edge_families ef ON vs.family_hash=ef.family_hash
    WHERE vs.strategy_id IN (
        'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10',
        'MES_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_S075',
        'MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15',
        'MES_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15',
        'MES_US_DATA_1000_E2_RR1.5_CB1_ORB_G8_O15'
    )
    ORDER BY vs.strategy_id
""").fetchall()

# Section 3.5 — DSR drift check
con.execute("""
    SELECT instrument, COUNT(*), COUNT(dsr_score), AVG(dsr_score), MAX(dsr_score)
    FROM validated_setups WHERE status='active' GROUP BY 1 ORDER BY 1
""").fetchall()
```

---

## 8. Decision asked of user

Pick one of A–E (or PARK / RECONSIDER). The picked disposition locks the
Stage 3 scope. If A or E, Stage 2 closes with no code change.
