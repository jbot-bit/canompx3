# Phase 5.1b Teardown Handoff — 2026-04-21

**Worktree:** `deploy/live-trading-buildout-v1` @ final HEAD `fd44d215` (after F7).
**Mission anchor:** minimum honest path to first live strategy under institutional-grade proof.

---

## A. Fix inventory

| # | Finding | Fix commit | Final state |
|---|---|---|---|
| F1 | CRITICAL — RITHMIC_APPKEY fabricated in xfa-root-cause.md | `bd6747a9` | RESOLVED. Replaced with canonical RITHMIC_USER / RITHMIC_PASSWORD / RITHMIC_GATEWAY (hard-required per auth.py L61-L65); three optional-with-defaults listed. |
| F2 | HIGH — teardown .venv junction safety | `83ca986f` | RESOLVED. Mandatory `cmd //c rmdir` step inserted before `git worktree remove` in keep-worktree + full-delete scenarios; shell-specific (Git Bash + PowerShell); junction-gone verification added; "git worktree remove automatic cleanup" claim struck. |
| F3 | HIGH — G7 WEAK directive-letter ambiguity | `16aa1120` | RESOLVED. Destruction-shuffle + RNG-null ran per-instrument (200 each, 6.1s total). All 3 instruments reject null at empirical p = 0.0000. G7 upgraded to PASS. Pre-reg YAML real_capital_flip_block shortened from 3 to 2 blockers (G7 remediation retired). |
| F4 | MEDIUM — G5 SD arithmetic (3.87 → 1.12) | `a2664725` | RESOLVED. Full formula SE(β)=SD_res/(SD_x·√N) shown; SD_x=0.289 for uniform percentile ranks; SD_res=1.12 consistent with pnl_r bounds. |
| F5+F6 | MEDIUM — MFFU Fair Play permissive + Bulenox memory-only | `06d56667` | RESOLVED. MFFU automation now stacked conditional (no sim-fill exploit + no HFT + CME-compliant) in prop-firm doc. B2 routing table: Bulenox Master/Funded allowed-venues UNVERIFIED + default venue changed from "Rithmic (memory-referenced)" to "UNVERIFIED — NOT SELECTED". MFFU routing flagged CONDITIONAL on Fair Play stack. |
| F7 | MEDIUM — G3 numerical DSR deferral | `fd44d215` | RESOLVED (verdict FAIL_2_OF_3 on evidence). DSR computed with Bailey-LdP 2014 Eq. 2 + Eq. 9; Terminal 2 bracketed-ρ̂ methodology (ρ̂ ∈ {0.3, 0.5, 0.7}, threshold at ρ̂=0.7). MNQ PASS (DSR=1.0000); MES FAIL (SR=−0.112); MGC FAIL (SR=−0.135). Unfiltered baseline cost-dead on MES+MGC — matches PR #59 `ec8198f3` and `recent_findings.md` 2026-04-20 entry. Real-capital flip remains blocked. |
| F8 | NEW — fabrication gap-scan | n/a (no commits) | 2 candidates found, both fixed by F1 (`RITHMIC_APPKEY` at xfa-root-cause.md:80 and :94). 0 residual. |
| F9 | OPT — MEMORY.md F-1 correction | SKIPPED | MEMORY.md lives in `C:/Users/joshd/.claude/projects/.../memory/` (outside worktree scope); other-terminal hold status unverifiable; chatgpt_bundle/ is user-curated. Safer to leave as user-owned decision (Step 9 below). |

---

## B. Re-review verdict

All 8 prior /code-review findings resolved (C1–C6 original; C8 G3 by F7). 0 residual from fabrication gap-scan. 0 NEW findings from remediation surface. Pre-commit 6/6 passed on every fix commit; no `--no-verify`; no CRLF noise committed (`pipeline/check_drift.py` + `tests/test_pipeline/test_check_drift_db.py` restored at §0 and never re-touched).

**Final grade: B+.** Downgraded from A- because F7 evidence re-classified G3 from CONDITIONAL_PASS → FAIL_2_OF_3 — a genuine re-interpretation of the deployment-form question, not a regression from the review. The rank-shape effect (G7) is real; the unfiltered-baseline deployment form is cost-dead on MES+MGC. That's the truth surface, not a defect.

Remaining findings (by design, not bugs):
- Bulenox venue still UNVERIFIED (directive § guardrails prohibits memory-fallback; resolved to surface).
- G3 FAIL on MES+MGC under unfiltered baseline (evidence-based; pre-reg YAML captures remediation path = separate R1 FILTER pre-reg).
- G6 RESEARCH_PROVISIONAL caveat unchanged (structural, not fixable this run).

---

## C. Critical path to first live strategy (mission anchor)

| Step | Gate / Action | Status after this session | Blocker | Owner |
|---|---|---|---|---|
| 1 | All G1–G10 shadow-ready | **G1–G2 PASS, G3 FAIL_2_OF_3, G4–G10 PASS**. Shadow-AUTHORIZED for signal-only (family-wise G3 FAIL does NOT revoke shadow; only real-capital flip). | G3 FAIL on MES+MGC unfiltered baseline; MNQ unfiltered baseline PASSES DSR but family-wise multiplicity gate applies. | Agent — resolved. |
| 2 | PR #48 shadow authorization | **VALID for signal-only**. Shadow wiring itself NOT built this run (Stage 4 XFA halted, Stage 5.1c fill capture halted). Authorization exists on paper; implementation gated by credentials. | Credentials + Stage 4 XFA wiring + Stage 5.1c fill capture. | User (credentials); Agent v2 (wiring). |
| 3 | Rithmic / broker creds on prod | UNCHANGED (still absent from this worktree env; prod env unverifiable). | RITHMIC_USER / RITHMIC_PASSWORD / RITHMIC_GATEWAY not provisioned. | User. |
| 4 | Profile flip choice | UNCHANGED. Primary candidate `topstep_50k_mnq_auto` (memory-referenced). MNQ-only because G3 FAIL on MES+MGC under unfiltered baseline. Routing venue = Rithmic or Tradovate, NOT ProjectX on LFA. | Credentials + venue decision. | User. |
| 5 | MFFU written confirmation | UNCHANGED. MFFU Fair Play "coordinating identical/opposite strategies across separate accounts" scope ambiguity (fair-play vs copy-trading tension). | Support ticket. | User. |
| 6 | Bulenox written confirmation | UNCHANGED. Help pages silent on auto/copier/news/prohibited; Terms-of-Use PDF fetch failed (binary). Routing default = UNVERIFIED. | Support ticket. | User. |
| 7 | Phase 6e monitoring build | UNCHANGED. Hard prereq for real-capital flip. | Engineering build per `docs/plans/2026-02-08-phase6-live-trading-design.md` § 6e. | User. |
| 8 | Real-capital flip | **BLOCKED.** Remaining gates: Steps 3, 4, 5 (if multi-firm), 6 (if Bulenox), 7. Additionally for MES/MGC: new R1 FILTER pre-reg required (see Step 9). | Multiple. | User. |
| 9 | New R1 FILTER pre-reg for MES+MGC (post-G3) | **NEWLY SURFACED by F7.** The rank-shape effect (G7 PASS) is real but deployment-form on MES+MGC must be R1 FILTER (Q5 or Q4+Q5 threshold), not unfiltered baseline. PR #59 `ec8198f3` already flagged semi-contaminated OOS — requires ≥ 50 fresh OOS trades before CANDIDATE_READY. | Research branch work (not deploy-live scope). | User / research terminals. |
| 10 | MEMORY.md F-1 correction (stale references) | **NOT ACTIONED this run (F9 skipped).** Four cross-referenced docs (chatgpt_bundle/04_DECISION_LOG.md, chatgpt_bundle/06_RD_GRAVEYARD.md, docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md, docs/audit/2026-04-15-topstep-scaling-reality-audit.md) still carry "F-1 XFA dormant" claims. F-1 code is LIVE-READY since 2026-04-14 per commits `5fae4d0c` / `ebc2b30f` / `306d16a0` / `13700958` / `527ac13b`. | Ops hygiene; not on critical path. | User. |

**The honest minimum path:** MNQ-only, unfiltered baseline, shadow-signal-only → Step 3 creds → Step 4 profile flip `topstep_50k_mnq_auto` → Step 7 Phase 6e → MNQ-only per-instrument-K=1 re-audit DSR under Mode A forward-OOS accumulation. MES+MGC are NOT on this path; they require Step 9's new filter-form pre-reg first.

---

## D. Cross-terminal state (at `origin`)

| Worktree | origin branch @ SHA | Recent work summary | Impact on PR #48 shadow posture |
|---|---|---|---|
| `canompx3` (main checkout) | `research/pr48-sizer-rule-oos-backtest` @ `74a876ce` | "audit(live): re-evaluate phase b framing" — Phase B institutional re-eval at `docs/audit/2026-04-21-phase-b-institutional-reeval.md`. Re-frames 6 live MNQ lanes: `PROVENANCE-BLOCKED` under Mode A, DSR `PENDING-ONC`, Chordia 3.79 strict still binding. Reframes edge vs posture. | No direct impact on PR #48 participation-shape. PR #48 is a shape-hypothesis on `orb_outcomes` truth layer, not on `active_validated_setups`. But the "empirical edge separated from deployment posture" reframe SUPPORTS our G3 FAIL cascade treatment (shadow stays authorized even when deployment-form SR is negative, because the edge reality is separately attested). |
| `canompx3-6lane-baseline` | `research/ovnrng-router-rolling-cv` @ `7f4f85ba` | "handover: Claude terminal session end 2026-04-21" — D1-D7 decisions rollup at `docs/audit/remediations/2026-04-21-decisions-resolved-rollup.md`. D1 restores PR #51 + PR #50 CANDIDATE_READYs DSR-PENDING; D5b unblocks Terminal 2 with bracketed-ρ̂ DSR method used in our F7. | F7 used D5b's bracketed-ρ̂ methodology directly (cited in g3.md). Ops-layer methodology alignment is now cross-terminal. |
| `canompx3-deploy-live` (this) | `deploy/live-trading-buildout-v1` @ `fd44d215` | 6 original buildout commits (Stage 0-6) + 6 fix commits (F1-F7). Branch pushed to origin during F1. Teardown patched for junction safety. | All fixes landed on branch; ready for teardown or merge per user decision. |

---

## E. Teardown status

- **NOT executed.** Patched at F2 (`83ca986f`) for Windows `.venv` junction safety: mandatory `cmd //c rmdir` of the junction BEFORE `git worktree remove` in any scenario; junction-gone verification step; shell-specific syntax for Git Bash and PowerShell; removed the unsafe "automatic cleanup" claim.
- Teardown doc at `docs/decisions/2026-04-21-deploy-live-teardown.md` covers three scenarios (merge / keep-branch-drop-worktree / full-delete) with updated step numbers to reflect the junction-removal insertion.
- Remote branch is now at `origin/deploy/live-trading-buildout-v1` (created during F1 push); teardown full-delete scenario's step-5 remote deletion is no longer hypothetical.

Awaits user command.

---

## F. What's NOT on the critical path but matters (accrued tech debt)

1. **Lesson from F7:** rank-regression |t| and deployment-form Sharpe are different animals. DSR targets deployment SR, not effect-size t. A gate certificate that punts DSR on structural grounds ("effect size is large, therefore DSR passes") is unsound when the SR is small or negative. Future gate certs should compute DSR inline, not reason about it.
2. **Participation-shape role refinement:** the shape signal survives G7 but deployment form is instrument-specific. MNQ = R3 POSITION-SIZE viable (unfiltered baseline SR+); MES+MGC = R3 fails, must be R1 FILTER (Q5 threshold). This aligns with PR #59 re-audit — the ovnrng-router terminal's D2 decision (`106bdec0`) LOCKs a 12-cell role-selection meta framework (ACTIVATION DORMANT), which is the natural home for this refinement.
3. **Bulenox UNVERIFIED status:** until ops obtains written Bulenox support confirmation of (a) automation permitted, (b) copier-compatibility with Rithmic, (c) news trading policy, and (d) prohibited conduct list, any Bulenox routing work is blocked. WebFetch cannot extract from the binary Terms-of-Use PDF; future work could use a local PDF text extractor (e.g., pypdf, pdfminer) to bypass this.
4. **MEMORY.md F-1 correction:** 5 files (MEMORY.md + 4 cross-references) still claim "F-1 hard gate dormant". F-1 has been LIVE-READY since 2026-04-14. Stale memory references reduce context quality for every future agent session.
5. **Teardown doc full-delete scenario** cherry-pick advice: if user chooses full-delete, the prop-firm-official-rules.md rewrite (substantive first-party provenance work) should NOT be lost — cherry-pick to a feature branch off main first.
6. **G3 FAIL cascade implication for workstream C pre-reg YAML:** the `deployment_target.mode` is currently "SHADOW_SIGNAL_ONLY" which correctly contains the FAIL (no capital at risk). But the shadow observation window (2 weeks) will produce forward-OOS on a signal-only basis for a deployment form (unfiltered baseline) known to be cost-dead on MES+MGC. Ops should consider whether the 2-week window is better spent on the Q5 FILTER form instead — a user-owned scope call.

---

## Provenance

- Integrity anchor: `git fetch origin` ran clean; no `[behind]` state. Parent of this branch `origin/main` @ `f567cfe6`.
- Running invariants held: `live=NO`, `creds=NO`, `lookahead=NO` (all computations on IS window < 2026-01-01), `time-box=2/2 respected` (F3 6.1s, F7 0.4s, both far under 15 min / 4 GB).
- Literature citations: all from `docs/institutional/literature/*.md` local extracts. Bailey-LdP 2014 Eq. 2 + Eq. 9; Chan Ch 7 p.155-157; Fitschen p.34/p.41; Chordia bands; edge-finding-playbook §3/§4. `pre_registered_criteria.md` @ `126ed6b8` pin retained.
- No touches to `gold.db` writes, `pipeline/`, `trading_app/`, other worktrees, research branches, or canonical config modules.
