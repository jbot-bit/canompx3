# Consolidation Summary — Claude terminal session 2026-04-21

**Terminal:** Claude in `canompx3-6lane-baseline` worktree, branch `research/ovnrng-router-rolling-cv`
**Sibling:** Codex in `canompx3` main worktree, branch `research/pr48-sizer-rule-oos-backtest`
**Codex-done handshake:** `.claude/handshake/codex-done.txt` (MGC sizer REJECT) + follow-on filter-form run (commit `96b9e358`)

---

## Session arc

Starting state: cold reset with 2 Claude + 1 Codex terminals open. Uncommitted work believed lost.

Actual loss: **zero.** The CRLF churn on `pipeline/check_drift.py` + `tests/test_pipeline/test_check_drift_db.py` across both worktrees was Windows line-ending noise, not real edits. All significant work was already committed or stashed.

After recovery, two terminals resumed in strict territorial partition:
- **Codex (MES/MGC/MNQ rel_vol lineage, main worktree, pr48 branch)** — completed MGC sizer institutional clearance REJECT + locked the MES/MGC filter-form pre-reg + executed that pre-reg's runner + updated HANDOFF. See `docs/audit/results/2026-04-21-rel-vol-filter-form-v1.md` (commit `96b9e358`).
- **Claude (non-rel_vol lineage + meta-framework + remediations, 6lane-baseline worktree, ovnrng branch)** — this session's deliverables below.

---

## The 6 Claude-terminal deliverables (committed + pushed)

| # | Artifact | Commit | Status |
|---|---|---|---|
| 1 | `research/audit_ovnrng_router_rolling_cv.py` + `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md` | `4dfd3000` | KILL verdict on ovnrng allocator router (rolling 4-fold CV 1/4 wins, mean ΔSR −0.525). Retracts PR #62. |
| 2 | `docs/audit/results/2026-04-21-post-hoc-rejection-sweep.md` | `39315b52` | Sweep of today's non-rel_vol branches for post-hoc-rejection pathology. Found 2 instances: PR #51+#50 DSR kill (HIGH damage), RULE 14 retroactive audit (LOW damage). |
| 3 | `.claude/rules/backtesting-methodology.md` RULE 3.4 + RULE 3.5 + failure-log entries | `631bda30` | Codified: (3.4) multi-fold WF for routers, (3.5) post-hoc criterion creep = post-hoc REJECTION. |
| 4 | `docs/handoff/2026-04-21-claude-terminal-follow-on.md` | `2a17ecc5` | HANDOFF.md delta (standalone file — user folds in at consolidation to avoid race with Codex's HANDOFF writes). |
| 5 | `docs/audit/hypotheses/draft-2026-04-21-rel-vol-role-selection-meta-v1.yaml` + `docs/audit/remediations/2026-04-21-pr51-pr50-dsr-remediation.md` | `a777e3e4` | Role-selection meta-framework (12-cell matrix, DRAFT_PENDING_REVIEW) + PR #51/#50 DSR remediation restoring 5+2 CANDIDATE_READYs with DSR-PENDING label. |
| 6 | `docs/audit/remediations/2026-04-21-rule14-retroactive-heterogeneity-remediation.md` | `30b73a7d` | RULE 14 retroactive audit reclassified as LOW-damage RETROACTIVE_FRAMING_REFINEMENT (no live signals affected). |

Plus uncommitted drafts (per prompt: not to commit until user review):
- `docs/audit/hypotheses/draft-2026-04-21-mnq-participation-exhaustion-v1.yaml` (DRAFT_PENDING_REVIEW)

All of items 1-6 are on `origin/research/ovnrng-router-rolling-cv`. Head: `30b73a7d`.

---

## Critical late-breaking finding (from Codex, affects Claude deliverables)

Codex's filter-form execution (commit `96b9e358`) surfaced a **structural E2 timing veto**:

> `trading_app.config.VolumeFilter` marks `rel_vol` as `E2`-excluded — it uses break-bar volume and resolves at `BREAK_DETECTED`, which is unknown at E2 order placement.

**Both Q5-only filter forms passed all 4 locked gates** (MGC + MES, bootstrap CI positive on ΔSR, T0 tautology clean against 5 canonical filters × 7 sessions = 35 cells):

| Instrument | Fire rate | Filter SR | Uniform SR | ΔSR 95% CI | Gate 1-4 |
|---|---:|---:|---:|---|---|
| MGC F1_Q5_only | 16.8% | +0.0749 | −0.0866 | [+0.0954, +0.2338] | PASS × 4 |
| MES F1_Q5_only | 18.8% | +0.0576 | −0.0963 | [+0.0984, +0.2137] | PASS × 4 |

BUT — **not deployable as an E2 pre-entry filter**. The math is real, the execution framing fails.

### Implications for Claude deliverables

1. **Role-selection meta-pre-reg (item 5) needs amendment before lock.** The FILTER role on rel_vol under E2 has a structural timing veto. Need to add either:
   - A new role type **POST_BREAK_CONDITIONER** (evaluates at entry time, after break bar closes — distinct from pre-break FILTER)
   - Or an execution-model axis (E1 / E2 / E2-CB2 etc.) where some models have filter timing that works
   - Or redefine the rel_vol feature to use pre-break-bar data (changes the feature, not the role)

2. **MNQ biphasic draft (uncommitted) inherits same veto.** H1 skip-Q5 / H2 Q4-only / H3 curvature — all FILTER forms evaluated at BREAK_DETECTED under E2 = all vetoed. Draft needs amendment listing the reframing options before the user reviews it.

3. **MGC sizer REJECT (Codex's decision) is now the ONLY institutionally-clean research finding** for rel_vol in the near term. Pre-reg pass stands as legitimate research; deployment blocked by institutional stack (Sharpe/WFE/N_eff/era stability), independent of the timing veto.

---

## Full role-selection picture (post-Codex finding, post-all-remediations)

| Instrument | FILTER (pre-break, E2) | SIZER (at entry) | ALLOCATOR | POST_BREAK_CONDITIONER | NOTHING |
|---|---|---|---|---|---|
| **MGC** | E2 VETO (math CANDIDATE_READY_IS) | Pre-reg PASS (+0.032R t=2.00) · Institutional REJECT (SR/WFE/N_eff gates) | Untested | **Candidate role — design pre-reg** | Unlikely |
| **MES** | E2 VETO (math CANDIDATE_READY_IS) | Dead (Sharpe negative both uniform and sized) | Untested | **Candidate role — design pre-reg** | Likely |
| **MNQ** | E2 VETO + biphasic shape undiscovered on IS-only | Dead (Spearman p=0.12) | Untested | **Candidate role — biphasic may fit here** | Likely default |

**Clean read:** rel_vol is a real rank signal (per 2026-04-20 participation-shape cross-instrument β₁ IS t=+9.59/+11.80/+7.54, 3/3 MONOTONIC_CONFIRMED) but its deployment form under E2 is now the narrow open question. Sizer is weak, filter is timing-vetoed. POST_BREAK_CONDITIONER or entry-model-switch are the live candidates.

---

## Decisions needed from user

Ordered by EV / urgency:

### 1. Restore PR #51 + PR #50 CANDIDATE_READYs (HIGH urgency — simple accept)

**Action:** read `docs/audit/remediations/2026-04-21-pr51-pr50-dsr-remediation.md`. If accepted, rename status of 5+2 MNQ CANDIDATE_READY cells from "MISCLASSIFIED" back to "CANDIDATE_READY, DSR-PENDING" in whatever memory/index tracks them.

**Rationale:** DSR was applied as hard kill despite Amendment 2.1 of `pre_registered_criteria.md` being explicit that DSR is cross-check until N_eff is formally resolved via ONC. Pre-reg gates (H1/C6/C8/C9) were honestly passed. Canonical data verified. Not deployable yet — CANDIDATE_READY is research-grade, not deploy-grade.

### 2. Role-selection meta-pre-reg review + E2 veto amendment (HIGH EV — defines next month of work)

**Action:** read `docs/audit/hypotheses/draft-2026-04-21-rel-vol-role-selection-meta-v1.yaml`. Five user-review checkpoints inside the file. Before locking, add the E2 timing veto handling either as a new POST_BREAK_CONDITIONER role or an execution-model axis.

**If locked with POST_BREAK_CONDITIONER added:** next pre-regs are (a) MGC POST_BREAK_CONDITIONER Q5-only, (b) MES same, (c) MNQ biphasic post-break form. All require fresh-OOS accrual post-2026-04-22.

### 3. MNQ biphasic draft review (LOW urgency until #2 resolved)

**Action:** read `docs/audit/hypotheses/draft-2026-04-21-mnq-participation-exhaustion-v1.yaml`. Currently proposes FILTER forms. Depends on #2 outcome — if POST_BREAK_CONDITIONER role is accepted, MNQ biphasic reframes to that role. If rejected, MNQ biphasic closes along with MNQ rel_vol entirely.

**Additional:** MNQ biphasic IS-only verification query still pending. The Q5-crash pattern was seen in PR #59 re-audit OOS (contaminated). Before lock, IS-only quintile query on MNQ is required to avoid locking on a peek-artifact. This query is a 5-minute task; I did not run it because Codex was concurrent on the same feature family.

### 4. RULE 14 remediation accept (LOW urgency — rubber-stamp)

**Action:** accept `docs/audit/remediations/2026-04-21-rule14-retroactive-heterogeneity-remediation.md`. No live signals affected; framing refinements already in memory. Cautionary principle codified in RULE 3.5.

### 5. ONC N_eff workstream (MEDIUM-HIGH EV — institutional unblock)

**Action:** schedule a separate session to implement ONC (Optimal Number of Clusters) clustering per LdP 2020 Ch 4. Resolves Amendment 2.1's gating condition, makes DSR a binding hard gate, unblocks PR #51/#50 CANDIDATE_READY → deploy promotion path, and enables C5 application across the institutional stack.

**Scope (sketch — needs its own pre-reg):**
- Implement ONC on current validated_setups feature-fire correlation matrix
- Compare ONC-derived N_eff to Bailey A.3 pairwise-correlation N_eff (~99)
- Take more conservative as canonical institutional N_eff
- Amend `pre_registered_criteria.md` to remove Amendment 2.1's gating condition
- Recompute DSR for all pending CANDIDATE_READYs at resolved N_eff

**Why MEDIUM-HIGH not HIGH:** doesn't directly enable a new trade. Enables clean promotion path for research already in the queue. But also prevents every future DSR audit from reopening this same debate.

### 6. Consolidate HANDOFF.md (LOW urgency — housekeeping)

**Action:** at next consolidation pass, fold `docs/handoff/2026-04-21-claude-terminal-follow-on.md` into HANDOFF.md below Codex's RELVOL_RESET_RECOVERY fence. Not time-critical.

---

## What this session deliberately did NOT do

- Did NOT execute any canonical-data runner for rel_vol (Codex's territory during active run).
- Did NOT touch HANDOFF.md during Codex's active HANDOFF write window.
- Did NOT write on Codex's branch (research/pr48-sizer-rule-oos-backtest).
- Did NOT promote any CANDIDATE_READY to deploy-eligible.
- Did NOT lock any draft pre-reg without user review.
- Did NOT propose re-opening closed NO-GOs (ML V1/V2/V3, IBS, NR7, cross-asset lead-lag, etc.).
- Did NOT run MNQ biphasic IS-only verification query (wise to wait until Codex's rel_vol work settles).

---

## Git state (final)

Claude terminal branch `research/ovnrng-router-rolling-cv`:

```
30b73a7d remediate: RULE 14 retroactive audit → RETROACTIVE_FRAMING_REFINEMENT
a777e3e4 design + remediate: role-selection meta-framework + PR #51/#50 DSR restoration
2a17ecc5 handoff(follow-on): ovnrng closure + post-hoc-rejection sweep delta
631bda30 methodology: RULE 3.4 + RULE 3.5 (post-hoc rejection codified)
39315b52 sweep(post-hoc-rejection): PR #51 DSR kill was post-hoc
4dfd3000 KILL: ovnrng allocator router (rolling 4-fold CV retracts PR #62)
```

All pushed to `origin/research/ovnrng-router-rolling-cv`. Upstream restored. No uncommitted work except:
- CRLF churn on `pipeline/check_drift.py` + `tests/test_pipeline/test_check_drift_db.py` (autocrlf artifact, safe to discard)
- Uncommitted draft `docs/audit/hypotheses/draft-2026-04-21-mnq-participation-exhaustion-v1.yaml` (per prompt: user-review-gated)

Codex terminal branch `research/pr48-sizer-rule-oos-backtest` — head `9991695a`. All pushed.

No cross-branch writes. No HANDOFF.md races. No production code touched. 2026 OOS (Mode A sacred) UNTOUCHED.

---

## One-line verdict per 6-step process

- **Verdict:** the rel_vol rank signal on MGC/MES is **CONDITIONAL** — math survives every locked gate, but deployment form under E2 is structurally vetoed. Requires role/execution-model redesign, not more statistical validation.
- **Edge location:** MGC rank signal primarily; MES secondary (tail-only); MNQ pending IS-only biphasic verification.
- **Biggest mistake avoided this session:** Three instances of post-hoc-criterion-creep on one repo in two days (PR #59 sizer re-audit, PR #51 DSR kill, RULE 14 retroactive audit). Codified RULE 3.5 to catch the pattern going forward.
- **Best next action:** user reviews + approves/rejects the 6 numbered decisions above. Highest EV is #2 (role-selection meta-pre-reg lock with POST_BREAK_CONDITIONER addendum) because it unblocks the ONLY live path for rel_vol deployment.
