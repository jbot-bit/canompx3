# Deploy-Live Next Move — Decision Memo

**Date:** 2026-04-21
**Worktree:** `deploy/live-trading-buildout-v1` @ `634278e7`
**Outcome class:** ESCALATE
**Reason:** (1) material cross-terminal state change invalidates the framing of my most recent commit (A); (2) 6e build remains gated on user decisions from the 2026-04-21 Phase 6e design doc § 4 + § 9.

---

## 1. What happened since A+C landed

### 1.1 Material cross-terminal commit I did not know about when I wrote A

`origin/research/mnq-pr51-dsr-audit` @ `9cc88b78` landed 2026-04-21 00:57 (before my A amendment at 12:28). Its result doc `docs/audit/results/2026-04-21-pr48-participation-shape-oos-replication-v1.md` runs OOS β₁-replication of PR #48 on the 2026 window. Verdicts:

| Instrument | N_OOS | β₁_OOS | t_OOS | p (one-tailed) | Verdict |
|---|---:|---:|---:|---:|---|
| MNQ | 771 | +0.14433 | +0.964 | 0.168 | **OOS_WEAK_BUT_RIGHT_SIGN** |
| MES | 702 | +0.36543 | +2.492 | 0.007 | **OOS_CONFIRMED** |
| MGC | 601 | +0.42276 | +2.519 | 0.006 | **OOS_CONFIRMED** |

Pre-commit gate: sign(β₁_OOS) == sign(β₁_IS) AND t_OOS ≥ +2.0. MES and MGC pass; MNQ sign-matches but fails the t-gate.

### 1.2 What that terminal explicitly says about my test framework

> "DSR is a SR-based test, not applicable to OLS slope parameters. The sign+power gate (K=3 Pathway B) is the OLS-appropriate Phase 0 complement."

> "Pooled mean vs slope interpretation: ... the sizer rule is about DIFFERENTIAL performance across participation buckets, not pooled expectancy. β₁_OOS = +0.365 says exactly that."

My F7 (commit `fd44d215`) applied Bailey-LdP DSR to the **per-trade Sharpe of the unfiltered-baseline return stream** and got:
- MNQ SR = +0.03 → DSR PASS
- MES SR = −0.11 → DSR FAIL
- MGC SR = −0.14 → DSR FAIL

Both results are correct for their respective questions. They are NOT contradictory. They answer different questions:
- F7 asks: "If I trade the unfiltered baseline E2 CB1 RR1.5 5m on IS data, what is my per-trade Sharpe?"
- The OOS replication asks: "Does the rank-regression slope replicate on OOS data?"

### 1.3 Where my A amendment went wrong

A (commit `d88ecde7`) narrowed the shadow pre-reg scope to MNQ-only using F7 as the driver. This conflates:
- **Empirical**: does the PR #48 shape signal replicate? (Answer: G7 PASS in-sample, OOS replication CONFIRMS MES + MGC, MNQ weak.)
- **Deployment posture for one specific form**: is the unfiltered baseline capital-viable? (Answer: MNQ yes, MES + MGC no per F7 per-trade SR.)

PR #48's actual deployment-form claim is a rank-based SIZER overlay (R3 in `mechanism_priors.md` §4), NOT unfiltered baseline. The OOS replication doc confirms this framing: "ALIVE as a sizer-overlay deploy candidate pending IS-trained-rule backtest" for MES + MGC.

Under a sizer-overlay framing, MES + MGC are the OOS-confirmed instruments. Under an unfiltered-baseline framing, MNQ is the only positive-SR instrument. My A implicitly committed the shadow form to unfiltered baseline by using F7 as the scope driver. That is the category slip.

Per the `deploy-live-participation-shape-shadow-v1.yaml` pathway_rationale (which I preserved in A), the shadow is about observing the **shape signal's forward-OOS persistence**, not about verifying the per-trade SR of any particular deployment form. That contradicts what A actually did.

### 1.4 What else landed in the last ~3 hours (scan, not deep-read)

- `origin/research/mes-mgc-filter-overlay-v2` @ 2026-04-21 00:26: "MES + MGC filter-overlay family: 0 survivors at K=171, confirms MES/MGC dead for single-filter ORB 5m". Confirms prior memory; no change.
- `origin/research/correction-aperture-audit-rerun`: L2/L6 aperture correction; unrelated to PR #48 scope.
- `origin/research/ovnrng-*`: routing / sweetspot work; not PR #48.
- `origin/research/l1-europe-flow-filter-diagnostic`: not PR #48.

Only the mnq-pr51-dsr-audit is material.

---

## 2. Candidate next-moves — scored

| Candidate | E1 EV to first live | E2 Critical path | E3 Session fit | E4 Deps met | E5 Reversibility | E6 Integrity risk | Notes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| **M1. Revise A amendment** (scope the shadow pre-reg honestly in light of the OOS replication) | 3 | 3 | 3 | 3 | 3 | 0 | Doc-only. Corrects a category slip I just committed. |
| **M2. Surface 6e § 4 + § 9** to user as decision questions | 3 | 3 | 3 | 3 | 3 | 0 | Unblocks 6e build for a future session. |
| M3. EXECUTE 6e build | 3 | 3 | 0 (gated) | 0 (gated) | 1 | 3 | Rigor violation. Disqualified. |
| M4. EXECUTE B (MES+MGC R1 FILTER pre-reg) | 2 | 2 | 0 (wrong terminal) | 1 | 2 | 2 | Belongs on research terminal. HANDOFF-appropriate, not EXECUTE. |
| M5. Hygiene D (MEMORY.md F-1 cleanup) | 1 | 0 | 2 | 2 | 3 | 1 | Low EV; outside worktree scope for 4 of 5 files. |
| M6. Mode-A readiness scan for MNQ under new scope | 2 | 2 | 1 | 2 | 3 | 1 | Possible but depends on A resolution first. Not standalone. |
| M7. HANDOFF B to research terminal via paste-ready prompt | 2 | 2 | 3 | 3 | 3 | 0 | Legitimate; but without user guidance on A, the research prompt for B is premature. |

**Top two tied at max score:** M1 (revise A) and M2 (surface 6e §4+§9). Both are doc-only, agent-safe, on the critical path, dependency-satisfied, reversible, integrity-positive.

Both can legitimately live in the same ESCALATE memo — this document. One session, one outcome: ESCALATE with combined decision questions.

---

## 3. Recommended single outcome

**ESCALATE — via this memo.** It covers:
- M1: propose a specific corrected A amendment (user picks approval or alternative).
- M2: surface the 6e § 4 + § 9 decisions (user answers unblock build).

No code. No further pre-reg scope commits until user decides. No 6e build.

---

## 4. Decision questions — M1: shadow pre-reg scope correction

The A amendment `d88ecde7` currently locks `deployment_target.instruments = ["MNQ"]` and frames the shadow as an unfiltered-baseline observation. Given the OOS replication, three honest corrections are possible:

**Option M1-a: Restore all 3 instruments, classify each per evidence.** (RECOMMENDED.)
- MNQ: UNVERIFIED on OOS t-gate, shape right-sign. Shadow as watch-only (observe fresh β₁ without any deployment form).
- MES: OOS-CONFIRMED. Shadow as candidate for a future IS-trained sizer rule backtest (not action yet).
- MGC: OOS-CONFIRMED. Same as MES.
- Deployment form in YAML: explicitly set to `OBSERVATION_ONLY` — signal logging + rank-regression tracking, no sizing action, no capital. This is closer to the original pre-reg intent.
- F7 DSR result on unfiltered baseline stays in G3 cert as a **separate factual finding** about the unfiltered-baseline form; not a gate on the observation-only shadow.

**Option M1-b: Keep MNQ-only unfiltered-baseline shadow; drop MES+MGC entirely from this pre-reg.**
- Matches my A amendment exactly. Defensible if we accept that "shadow" here means "prepare for unfiltered-baseline capital deployment on MNQ". 
- Downside: the OOS-confirmed MES+MGC shape is then unrepresented by any pre-reg on the deploy-live branch, and the pre-reg's own pathway_rationale about observing the shape signal is inconsistent with the narrower scope.

**Option M1-c: MNQ watch-only + MES+MGC observation-only in same pre-reg.**
- Hybrid: MNQ gets the watch-only role (weak OOS), MES+MGC get observation-only (OOS-confirmed shape pending sizer rule).
- Close to M1-a in practice; may be clearer about the instrument-specific status.

**Question to user:** which of M1-a, M1-b, M1-c is the intended direction?

My recommendation: **M1-a.** Reasons:
1. It matches the pre-reg's own pathway_rationale (observe the shape).
2. It honors the OOS replication evidence.
3. It treats empirical and deployment-posture as separate (directive invariant).
4. It does not lose information — the F7 unfiltered-baseline DSR result stays visible in G3 cert as a form-specific finding.

---

## 5. Decision questions — M2: Phase 6e § 4 + § 9 unblock

Phase 6e design doc (`docs/plans/2026-04-21-phase-6e-monitoring-design.md`, commit `634278e7`) contains locked threshold proposals in § 4 and build-gate decisions in § 9. These need user confirmation BEFORE any code is written. Verbatim:

### § 4 thresholds — confirm or amend each

| # | Threshold | Proposed value | Confirm? |
|---|---|---|---|
| 1 Drawdown | Daily PnL floor | −3R WARNING | ? |
| 2 Circuit Break | Daily PnL floor | −5R CRITICAL (auto-halt hook to session_orchestrator) | ? |
| 3 WR Drift | Window / delta | 50 trades / 10pp below backtest WARNING | ? |
| 4 ExpR Drift window | Rolling window | 50 trades CRITICAL | ? |
| 4 ExpR Drift ratio | Threshold | 0.50 of backtest ExpR | ? |
| 4 SR ARL₀ | Target | 1000 trades | ? |
| 5 ORB Size Regime | Window / ratio | 30d / 2.0× median INFO | ? |
| 6 Missing Data | Ratio | 0.80 of expected bar count WARNING | ? |
| 7 Strategy Stale | Inactivity | 30 calendar days INFO | ? |

All values are verbatim from `docs/plans/2026-02-08-phase6-live-trading-design.md` § 6e lines 420-428, except SR ARL₀ = 1000 (cross-referenced from the shadow pre-reg G9 kill criteria).

### § 9 build decisions — answer before code

1. **Threshold lock confirmation.** Do § 4 values freeze as-is, or are any amended?
2. **Dashboard panel priority.** Risk-utilization panel depends on prop-firm sizing scheme. Resolve the 2026-02-08 spec Q5 (prop firm vs personal account) first, or accept a simplified panel for v1?
3. **Test-coverage target confirmation.** Is the ~88-test estimate acceptable, or should coverage be narrowed/widened?
4. **Build sequence confirmation.** Proposed: detectors-first (parallel commits) → `monitor_runner` → dashboard → orchestrator hook. Each sub-step gets its own commit. Confirm or amend?

---

## 6. Follow-up prompt (once decisions land)

If M1-a + § 4 locked as-is + § 9 answered:

```
EXECUTE PROMPT — DEPLOY-LIVE NEXT SESSION

Worktree: C:/Users/joshd/canompx3-deploy-live
Branch: deploy/live-trading-buildout-v1
Parent of current HEAD: 634278e7 (assume no further updates unless you fetch and find them)

Task sequence (one commit per step, push after each):

Step 1 — Revise the shadow pre-reg amendment per M1-a resolution from
         docs/decisions/2026-04-21-deploy-live-next-move.md §4.
         - docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml:
           * status flips LOCKED_SHADOW_ONLY_MNQ_ONLY → LOCKED_OBSERVATION_ONLY_3_INSTRUMENTS
             (or equivalent; name decided by consistency with G* certs)
           * scope_after_amendment_2026_04_21 block rewritten: active_instruments =
             ["MNQ","MES","MGC"]; per-instrument classification block citing the
             OOS replication (origin/research/mnq-pr51-dsr-audit @ 9cc88b78).
           * deployment_target.instruments = ["MNQ","MES","MGC"];
             deployment_cell field replaced with deployment_form = "OBSERVATION_ONLY"
             with explicit note that F7 unfiltered-baseline DSR result remains
             a separate factual finding recorded in G3 cert, NOT a shadow
             gate.
           * heterogeneity_veto rewritten for 3-instrument observation-only scope
             — per-instrument sign-flip veto after N_fires ≥ 50 per instrument.
           * real_capital_flip_block rewritten to reflect the "no deployment form
             yet" state — no real-capital path exists without either:
               (a) an IS-trained sizer rule + its own pre-reg + backtest, or
               (b) a per-instrument-K=1 MNQ DSR pre-reg on the unfiltered baseline.
         - docs/decisions/2026-04-21-pr48-g3.md: update the Cascade section to
           reference the corrected scope; re-emphasize that the G3 DSR FAIL
           is on unfiltered baseline only, not on the observation-only shadow.
         - Commit prefix: fix(shadow-scope): revise to observation-only 3-instrument

Step 2 — Begin 6e build per docs/plans/2026-04-21-phase-6e-monitoring-design.md §5.
         Sub-step 2.a: monitor_thresholds.py — frozen dataclass with §4-locked values.
         Commit prefix: feat(6e): locked threshold dataclass. Tests first.
         Each subsequent sub-step (2.b detectors, 2.c monitor_runner, 2.d dashboard,
         2.e orchestrator hook) as separate commits with their own design+verify
         cycle per CLAUDE.md 2-Pass Method.

Invariants: as in prior deploy-live session — no --live, no creds, no writes to
gold.db, no canonical config touches, pre-commit never bypassed, CRLF churn
restored never committed, no training-memory fallbacks, pass/fail gates shown
with evidence.
```

If M1-b: a similar prompt but narrower (keep MNQ-only scope, skip Step 1, go straight to Step 2).

If M1-c: hybrid; Step 1 YAML text differs but Step 2 is identical.

---

## 7. What this memo is NOT

- NOT a pre-reg. Does not change any pre-reg YAML.
- NOT a code change. No code written.
- NOT a cherry-pick or revert of A. A stays on the branch; this memo proposes the CORRECTED amendment that would supersede A in a future commit.
- NOT an attempt to resolve both the scope question and the 6e build in one session. Picks ESCALATE, stops.

---

## 8. Abort triggers hit

- `mnq-pr51-dsr-audit` OOS replication landing after A amendment → PAUSE, recompute, escalate. Triggered at PF2.

## 9. Invariants held

- Live/creds/lookahead: all NO.
- Canonical-only truth: this memo cites `orb_outcomes`-derived OOS β₁ statistics from the other terminal's result doc (not from `validated_setups` / `edge_families` / `live_config`).
- DSR-alone decisions (Amendment 2.1): not made here; G3 FAIL stays as cross-check only, not as a gate on shadow scope.
- 2-Pass Method + Design Proposal Gate: this memo IS the design proposal for the M1 correction + 6e gate.
- CRLF: restored at PF1, status clean.
- No narration of internal process in user-facing output (this memo is user-directed — narration is expected for decision surface).

---

## 10. Provenance

- F7 commit `fd44d215` on `deploy/live-trading-buildout-v1`.
- A commit `d88ecde7` on same.
- `origin/research/mnq-pr51-dsr-audit` @ `9cc88b78` (2026-04-21 00:57:52 +1000).
- OOS replication doc: `docs/audit/results/2026-04-21-pr48-participation-shape-oos-replication-v1.md` on that branch.
- Phase 6e design doc: `docs/plans/2026-04-21-phase-6e-monitoring-design.md` @ `634278e7` on this branch.
- Pre-reg criteria pin: `docs/institutional/pre_registered_criteria.md` @ `126ed6b8` (unchanged).
- No live-DB query this session (design-only outcome).
