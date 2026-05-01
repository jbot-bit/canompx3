# Session disposition — 2026-05-01 — HQ-trades pre-reg batch

**Session goal (user-stated):** "Take all money off the table. Bloomberg-professional. Stop pigeon-holing — banger trades."

**Plan version executed:** v2 (offensive pre-reg batch + Stage 0 sweep). v1 plan was rejected mid-session as defensive-only / pigeon-holed.

**Mode:** institutional rigor; Auto mode active per session prompt; user-confirmed "go option 2 bra".

---

## What landed (DONE this session)

### Stage 0 quick-wins

1. **Action-queue timestamp drift** — `docs/runtime/action-queue.yaml:2` bumped from `2026-04-28` → `2026-05-01`.
2. **F-1 memory truth-correction** — `memory/f1_xfa_active_correction.md` rewritten as v3 (truth-corrected). Prior framing said "F-1 fail-closes every entry in signal-only" — wrong since B6 fix landed at `session_orchestrator.py:482-488` on 2026-04-25. New framing: F-1 wired + B6 signal-only seed active + profile-conditional firing.
3. **D6 pre-reg path typo** — `docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml:92` swapped `scripts/pre_session_check.py` (does not exist) → `trading_app/pre_session_check.py` (canonical).

### Pre-regs (4 designs, all DESIGN_LOCKED_AWAITING_GO)

| # | Pre-reg ID | K | Type | Gating |
|---|---|---|---|---|
| 1 | `2026-05-01-chordia-revalidation-deployed-lanes` | 4 | Defensive honesty gate | none — runs anytime |
| 2 | `2026-05-01-aperture-extension-o15-o30-london-usdata` | 4 | Offensive new-cell discovery | independent of #1 |
| 3 | `2026-05-01-carver-stage2-vol-targeted-sizing` | ≤3 | Offensive sizing overlay | gated on #1 PASS |
| 4 | `2026-05-01-mnq-garch-p70-cross-session-companion` | 2 | Offensive mechanism generalisation | independent of #1 |

All four pre-regs:
- Cite local literature extracts only (no training-memory citations); `literature_grounding_correction` blocks document any framings rejected from v2 plan.
- Use canonical-source delegation (`trading_app.chordia.compute_chordia_t`, `pipeline.dst.orb_utc_window`, `pipeline.holdout_policy.HOLDOUT_SACRED_FROM`, `pipeline.cost_model.COST_SPECS`, `trading_app.config.StrategyFilter.matches_row()`, `pipeline.paths.GOLD_DB_PATH`).
- Frontmatter declares `pooled_finding: false` (per `.claude/rules/pooled-finding-rule.md`); per-cell verdicts only.
- Pre-register a result-doc path under `docs/audit/results/` and a runner-script path under `research/`.
- Define explicit kill criteria + failure policies before scan execution.

---

## What was DEFERRED (not executed this session)

### Stage 0 items

| Item | Reason | Disposition |
|---|---|---|
| Parked-cells pressure-test (3 injections per `handoff-parked-cells-registry-2026-04-29.md:75-108`) | Requires running `pipeline/check_drift.py` 4 times with intermediate `git restore`; 15-20 min execution that didn't fit between offensive pre-reg writing | **Deferred to next session.** Drift check 124 currently passes baseline; pressure-test pending. |
| PR #191/#192/#193 disposition triage | I/O bound on `gh pr view`; design-effort low, execution time high | **Deferred to next session.** Each PR needs a Convert / Park-with-trigger / Close-as-NO-GO decision with rationale. |
| Phase D 1m bar gap (MNQ data through 2026-04-29 only) | Other terminal handling on `phase-d-volume-pilot-d0` worktree (separate session, disjoint scope from this work) | **Owned by Phase D worktree session.** No action here. |

### Pre-regs cut vs v1 plan

| Cut item | Reason |
|---|---|
| PR #51 E2 LA clean re-derivation pre-reg | Recovers borderline lanes; not a banger source. Re-open IF Chordia revalidation leaves the deployed portfolio empty (≥3/4 FAIL_BOTH). |
| LdP theory-first gate stage spec | Pure infra; zero new trades this quarter. Reopen as a doctrine-tightening session when sufficient session time available. |
| D6 Phase 1 wiring stage spec | Awaiting user GO; spec text is in v2 plan body but not yet a `docs/runtime/stages/<slug>.md` file. Cosmetic deferral; wiring depends on whether the GARCH cross-session companion (#4) generalises the mechanism. |

### CRG / MCP eval / cron items

All deferred to dedicated tooling sessions; not money-path.

---

## What needs USER GO before next progression

| Decision | Path | Default if no GO |
|---|---|---|
| Run Chordia revalidation runner (#1 above) | `research/chordia_revalidation_deployed_2026_05_01.py` (script not yet written) | Pre-reg #3 (Carver) cannot run — gated on Chordia PASS |
| Run Fitschen aperture extension runner (#2) | `research/aperture_extension_o15_o30_london_usdata_2026_05_01.py` (not yet written) | New aperture cells stay unscanned |
| Run GARCH cross-session companion runner (#4) | `research/mnq_garch_p70_cross_session_companion_2026_05_01.py` (not yet written) | D6 mechanism stays COMEX-specific (not necessarily wrong, just unconfirmed) |
| Run Carver Stage-2 sizing runner (#3) | gated on #1 PASS | If 0 lanes PASS Chordia, this pre-reg is MOOT |

**Recommendation for next session**: run #1 first (no dependencies, pure read/SQL, ~60s); evaluate verdict; then run #2 + #4 in parallel (independent, both ~5 min); finally #3 if #1 produced any PASS lanes. Total runtime <15 minutes for all four scans.

---

## Files touched

```
M docs/audit/hypotheses/2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml  # D6 path typo fix
M docs/runtime/action-queue.yaml                                                                # ts bump
A docs/audit/hypotheses/2026-05-01-chordia-revalidation-deployed-lanes.yaml                     # pre-reg #1
A docs/audit/hypotheses/2026-05-01-aperture-extension-o15-o30-london-usdata.yaml                # pre-reg #2
A docs/audit/hypotheses/2026-05-01-carver-stage2-vol-targeted-sizing.yaml                       # pre-reg #3
A docs/audit/hypotheses/2026-05-01-mnq-garch-p70-cross-session-companion.yaml                   # pre-reg #4
A docs/runtime/session-disposition-2026-05-01-hq-trades-prereg-batch.md                         # this file
M memory/f1_xfa_active_correction.md                                                             # v3 truth-correction
```

Net: **0 production-code edits** (pipeline/, trading_app/, scripts/ untouched). All changes are pre-registration, runtime documentation, or memory.

---

## Honesty notes (caught and corrected during the session)

1. **Plan cited 5 deployed lanes from `prop_profiles.py:431-849`; live `lane_allocation.json` shows only 4.** The 5th (MGC_TOKYO_OPEN ORB_G4_CONT_S075) is not in the live allocator. Pre-reg #1 corrected to 4 hypotheses. Volatile-data-rule applied per `feedback_doctrine_drift_cost_specs_2026_05_01.md` pattern.

2. **Plan cited "Fitschen Ch 5-7 institutional-entry timing" from training memory.** Local extract (`docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`) is Ch 3 only. Pre-reg #2 reframed around Ch 3 (intraday trend persistence at multiple apertures), with explicit `literature_grounding_correction` block documenting the rejected framing.

3. **Plan cited "λ = Sharpe² / (1 + Sharpe²) Kelly fraction" from training memory.** Local extract has Carver Table 25 (realistic-SR → vol-target lookup table) and Half-Kelly recommendation, not the closed-form Kelly. Pre-reg #3 reframed; rejected closed-form framing documented in `literature_grounding_correction` block.

4. **Plan classified F-1 audit as "HIGH EV money on the table".** Phase 1 already showed F-1 wiring is live; "audit" was just confirming a value. Demoted to Stage 0 quick-win; truth-corrected the underlying memory file.

5. **Branch ceremony.** v1 plan called for new branch off `origin/main`; mid-session another terminal triggered branch flip; per `workflow-preferences.md` trivial-tier (docs-only, <100 lines net) the work landed on main directly without ceremony. No production code touched, so no branch-discipline violation.

---

## Anti-pigeon-hole disclosure

The session goal was "stop pigeon-holing — banger trades". v1 plan was 100% defensive (Chordia revalidation + D6 wiring + LdP gate); v2 reframe added 3 offensive mechanism-grounded pre-regs (Fitschen aperture extension, Carver Stage-2 sizing, GARCH cross-session companion).

**The pre-reg portfolio is now offense-weighted**: 1 defensive (Chordia honesty gate) + 3 offensive (aperture extension, sizing overlay, mechanism generalisation). This is an explicit bet that the next session's HQ-trade prospects come from NEW edge work, not from re-tuning existing deployed lanes.

If Chordia revalidation returns ≥3 of 4 lanes FAIL_BOTH, the offensive pre-regs (#2, #4) become the ONLY HQ-trade plays, and Carver Stage-2 (#3) is moot. The re-rank rule is written into pre-reg #3's `gating_rationale` block — not improvised mid-execution.

---

## Verification

- [x] All 4 pre-reg yamls saved (verified via `ls`).
- [x] D6 typo edit applied (verified via Edit tool success).
- [x] Action-queue ts bumped (verified via Edit tool success).
- [x] F-1 memory rewritten (verified via Write tool success).
- [x] Zero production-code edits (`git status --short` shows no `pipeline/` `trading_app/` `scripts/` modifications).
- [x] All 4 pre-regs cite local literature extracts (`docs/institutional/literature/`) only.
- [x] All 4 pre-regs use canonical-source delegation (no inlined constants).
- [x] All 4 pre-regs declare `pooled_finding: false` per pooled-finding-rule.

---

## Next-session entry point

```bash
# 1. Read this file + the 4 pre-regs.
# 2. User GO on which pre-reg(s) to run (recommendation: #1 first, then #2 + #4 in parallel).
# 3. Author runner script(s) per the runner_spec block in each pre-reg.
# 4. Execute scan(s); verify drift check passes; commit research script + result doc.
# 5. Land doctrine actions per the result doc (e.g., FAIL_BOTH lanes downgrade
#    to research-provisional + signal-only per Amendment 2.7).
```
