# Edge Extraction Phased Plan — 2026-04-28

**Status:** DRAFT — pending user approval at decision points
**Owner:** claude (autonomous loop, with human go/no-go at Phase D and E)
**Authority:** `docs/institutional/pre_registered_criteria.md` (12 criteria), `.claude/rules/backtesting-methodology.md` (16 rules), `.claude/rules/research-truth-protocol.md`, `.claude/rules/pooled-finding-rule.md` (RULE 14)
**Origin:** 2026-04-28 audit cycle uncovered E2 break-bar look-ahead contamination across 18 of 73 research scripts; PR #48 "participation universality" TAINTED. Comprehensive scan rebuilt on clean features yields 5 mechanism-grounded research-survivors.

## Goal

Extract every valid edge currently sitting in canonical `gold.db` for the active instrument set (MNQ/MES/MGC) and active session set (12 sessions), with full Phase 0 institutional discipline, no look-ahead, no scratch-NULL drops, instrument-family-aware multiple-testing, OOS-power-floor honesty, and Pathway B confirmation before any capital deployment.

## Constraints (acknowledged, not negotiable)

1. **Small OOS:** ~3 months of post-2026-01-01 data. Phase 0 C8 power floor: cells with N_OOS<50 → UNVERIFIED (not failed). Dir-match OOS is ordinal evidence only at this sample.
2. **MGC ≠ equity-index regime:** Gold has different macro drivers (USD, real yields, geopolitics) vs MNQ/MES (equity sentiment + macro). Pooled `MNQ+MES+MGC` claims violate RULE 14. Treat MGC as its own family.
3. **Bailey 2013 MinBTL:** ~6.9 years clean MNQ data caps clean-K at ~300 per family. Comprehensive scan K=13,415 is discovery layer; Pathway B K=1 is the legitimate confirmatory step per Amendment 3.0.
4. **Holdout sacred:** `trading_app.holdout_policy.HOLDOUT_SACRED_FROM = 2026-01-01`. No tuning against OOS. No re-running with different thresholds to "rescue" an OOS verdict.
5. **High-risk tier:** every Phase D and E task requires explicit user go/no-go before execution.

---

## PHASE A — Contamination sweep (mechanical, 100% actionable autonomously)

**Goal:** Stop the bleeding. Lock the audit work. Make sure no further research is built on contaminated foundations.

| Task | Status | Owner | Acceptance |
|---|---|---|---|
| A1. Append hygiene tail to 4 audit MDs (verdict / repro / caveat sections) | DONE 2026-04-28 | claude | `scripts/tools/check_claim_hygiene.py` returns OK on all 4 files |
| A2. Modify `comprehensive_deployed_lane_scan.py::emit()` to include hygiene sections natively | DONE 2026-04-28 | claude | Future scan runs land verdict / repro / caveat tails automatically |
| A3. Gate E2 break-bar features in `comprehensive_deployed_lane_scan.py::build_features()` | DONE 2026-04-28 | claude | `rel_vol_*`, `bb_volume_ratio_*`, `break_delay_*`, `break_bar_continues_*` removed; cell count drops 17,075→13,415 |
| A4. Write `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` | DONE 2026-04-28 | claude | 18 contaminated scripts enumerated + status |
| A5. Append RULE 16 entry to `.claude/rules/backtesting-methodology-failure-log.md` | DONE 2026-04-28 | claude | Entry includes magnitude, root cause, lesson, literature cite |
| A6. Update `MEMORY.md` flagging PR #48 TAINTED + adding registry pointer | DONE 2026-04-28 | claude | "Prior (2026-04-20)" entry replaced with TAINTED note |
| A7. Commit Phase A as one atomic commit | PENDING | claude | Pre-commit gauntlet passes; HEAD includes scan fix + hygiene + registry |

**Phase A acceptance:** all 7 tasks complete, single atomic commit, pre-commit clean, no new contaminated paths added.

---

## PHASE B — Per-candidate verification (autonomous, no capital touched)

**Goal:** Apply the full Phase 0 deploy-gate evidence stack to the 4 mechanism-grounded clean candidates from the rebuilt scan. **Strict enforcement of N_OOS < 50 → UNVERIFIED rule (Phase 0 C8 power floor).**

The 4 candidates (drop dow_thu — no mechanism per Aronson Ch6):

| ID | Cell | Mechanism | Strict |t| | OOS/IS | Bootstrap p |
|---|---|---|---|---|---|
| B-MES-EUR | MES EUROPE_FLOW O15 RR1.0 long + ovn_range_pct_GT80 | Chan Ch7 + Fitschen Ch3 | 3.47 | 1.36 | 0.0005 |
| B-MES-LON | MES LONDON_METALS O30 RR2.0 long + ovn_range_pct_GT80 | Chan Ch7 + Fitschen Ch3 | 3.10 | 2.41 | 0.0005 |
| B-MNQ-NYC | MNQ NYSE_CLOSE O5 RR1.0 long + ovn_range_pct_GT80 | Chan Ch7 + Fitschen Ch3 | 3.19 | 2.14 | 0.0010 |
| B-MNQ-COX | MNQ COMEX_SETTLE O5 RR1.0 long + garch_vol_pct_GT70 | Carver Ch9-10 vol-targeting | 3.18 | 0.99 | 0.0035 |

| Task | Acceptance |
|---|---|
| B1. Re-run scan post-A5 + A6 (sanity confirm 15 actionable + 0 OOS-flips holds) | Survivor count & flip-rate stable within ±2 cells |
| B2. C5 DSR (Bailey-LdP Eq.9 effective-N corrected) per cell | Each cell labelled DSR_PASS / DSR_FAIL with M, ρ̂, N̂ shown |
| B3. C6 WFE = Sharpe_OOS / Sharpe_IS per cell, CONDITIONAL on N_OOS≥50 | Each cell labelled WFE_PASS / WFE_FAIL / WFE_UNVERIFIED |
| B4. C8 dir-match with explicit power label per Phase 0 Amendment 3.2 (`feedback_oos_power_floor.md`) | dir_match printed alongside Cohen's d and OOS power |
| B5. C9 era-stability already done (per-year breakdown N≥20 → no era ExpR < -0.05) | DONE in audit pass; document in deliverable |
| B6. Lane-correlation matrix (Carver Ch11) — 4 candidates × 6 deployed lanes | Matrix doc; lanes with corr ≥ 0.5 to existing flagged for portfolio review |
| B7. Result doc per cell at `docs/audit/results/2026-04-28-bN-{cell-id}-phase0-evidence.md` | All Phase 0 columns + verdict CANDIDATE_READY / RESEARCH_SURVIVOR / KILL |

**Phase B acceptance:** 4 result docs, each with full Phase 0 evidence; pre-commit-clean; verdict per cell on the CANDIDATE_READY / RESEARCH_SURVIVOR / KILL ladder.

**Decision gate at end of Phase B:** present aggregate verdict to user. Phase D pre-regs only proceed for cells reaching CANDIDATE_READY.

---

## PHASE C — Instrument-family discipline (autonomous, doctrine work)

**Goal:** Codify per-instrument discipline so future scans don't re-pool MGC with equity-index data. Honest treatment of small-OOS.

| Task | Acceptance |
|---|---|
| C1. Define `INSTRUMENT_FAMILIES` constant: `{"EQUITY_INDEX": ["MNQ","MES"], "METAL": ["MGC"]}` | Lives in `pipeline/asset_configs.py` as a canonical export |
| C2. Update `pipeline/check_drift.py` with `check_pooled_instrument_family_violation` that flags any new audit MD with pooled MNQ+MES+MGC claims missing per-family flip-rate | Drift check #N+1 added; existing MDs grandfathered with deadline |
| C3. Add OOS power-floor section to `docs/institutional/pre_registered_criteria.md` Criterion 8 (already in Amendment 3.2 — verify currency) | Criterion 8 amendment-status verified; pointer in plan |
| C4. Document MGC regime caveat in `docs/institutional/mechanism_priors.md` | Section "Gold-specific microstructure" with Chan Ch5 currencies/futures grounding |

**Phase C acceptance:** 4 doctrine artifacts; drift check passing; pre-commit clean.

---

## PHASE D — Pathway B confirmation pre-regs (REQUIRES USER GO/NO-GO)

**Goal:** Promote validated research-survivors to deploy-ready candidates via Pathway B K=1 confirmation per Amendment 3.0.

**GATE:** present Phase B aggregate verdict to user before any Phase D task fires.

**Order of execution:** strongest evidence first (per Phase B B-MES-EUR is highest EV).

| Task | Pre-reg path | Acceptance |
|---|---|---|
| D1. B-MES-EUR Pathway B K=1 pre-reg + run | `docs/audit/hypotheses/2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml` | Numeric kill criteria; theory citations; K=1 lock; result doc |
| D2. B-MES-LON Pathway B K=1 pre-reg + run | `docs/audit/hypotheses/2026-04-28-mes-london-metals-ovn-range-pathway-b-v1.yaml` | Same |
| D3. B-MNQ-NYC Pathway B K=1 pre-reg + run | `docs/audit/hypotheses/2026-04-28-mnq-nyse-close-ovn-range-pathway-b-v1.yaml` | Same |
| D4. B-MNQ-COX Pathway B K=1 pre-reg + run | `docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml` | Same |
| D5. (HELD until C1-C4) — MGC LONDON_METALS short ovn_range — separate METAL-family pre-reg | TBD | Per-instrument family discipline applied |

**Phase D acceptance:** each surviving Pathway B run produces CANDIDATE_READY / KILL verdict. CANDIDATE_READYs eligible for Phase E.

---

## PHASE E — Capital integration (REQUIRES USER GO/NO-GO + capital-review skill)

**Goal:** Integrate Phase D survivors into deployed allocator with full risk discipline.

**GATE:** explicit user authorisation per task. Capital-review skill mandatory.

| Task | Acceptance |
|---|---|
| E1. Carver Ch11 lane-correlation matrix updated with Phase D survivors | Correlation < 0.5 to existing 6 lanes for promotion candidates; > 0.5 = portfolio overlap concern |
| E2. C11 90-day account-death MC simulation per prop ruleset (TopStep 50k, others if applicable) | ≥70% survival per Phase 0 Criterion 11 |
| E3. Slot allocation revision in `trading_app/prop_profiles.py` if needed | New `lane_allocation` entry; tested via paper trade for ≥5 sessions before live |
| E4. Shiryaev-Roberts monitor (C12) registration in `trading_app/sprt_monitor.py` | Drift detection active per cell on day-1 of live |
| E5. Live deployment | Capital-review skill verdict GO; user explicit confirmation |

---

## PHASE F — Adjacent edge hunts (parallel, low-priority but high optionality)

**Goal:** While Phase B/C/D run on the main candidates, hunt for orthogonal edge in parallel.

| Task | Status | Notes |
|---|---|---|
| F1. Vol-conditional RR per trade pre-reg (Carver Ch9-10) | DEFERRED until Phase D returns | Would test if RR_target should be `f(vol_regime)` rather than fixed |
| F2. Direction-split per session on existing 6 deployed lanes (rebuilt data) | DEFERRED | Could surface latent 6→3-lane portfolio improvements |
| F3. Pre-velocity × vol-regime interaction | EXISTING pre-reg (2026-04-26) — not yet run | Run after Phase D |
| F4. Re-run `participation_optimum_*.py` clean (drop rel_vol; use ovn_range / garch / atr) | DEFERRED until Phase D returns | Re-derives PR #48 claims on clean predictors |
| F5. MGC-family-specific scan with metal-microstructure features (Chan Ch5) | DEFERRED until Phase C lands | Per-family discipline first |

---

## Sequencing

```
Phase A (mechanical, autonomous) — ~1h
   |
Phase B (per-candidate verification) — ~2-4h autonomous
   |
   |--- DECISION GATE — present to user
   |
Phase C (doctrine, autonomous) — ~1h, can run parallel with B
   |
   |--- DECISION GATE — Phase D pre-reg approval per cell
   |
Phase D (Pathway B pre-regs + runs) — ~2h per cell, sequential
   |
   |--- DECISION GATE — Phase E capital authorization
   |
Phase E (capital integration) — multi-session, paper-trade gated
   |
   |--- (parallel throughout) — Phase F
```

## Risk register

| Risk | Mitigation |
|---|---|
| Phase B exposes more contaminated paths in candidates | Re-classify to KILL before promotion; document in registry |
| Phase D Pathway B kills 1-3 of the 4 candidates | Acceptable — Pathway B is supposed to kill underpowered IS findings; pre-reg numeric kill criteria do not allow rescue |
| Phase E lane-correlation reveals high overlap | Don't promote; revise allocator slots |
| OOS too small to validate any cell | UNVERIFIED is the legitimate verdict; cells park until Q3-2026 |
| MGC regime instability invalidates MES/MNQ universality at scale | Per-family discipline (Phase C) prevents this from being a downstream surprise |
| Capital-review skill flags GO unwarranted | Hard stop; do not deploy |

## Honesty checks

- Every Phase D pre-reg MUST cite this plan as upstream provenance (`upstream_discovery_provenance: role: PROVENANCE_ONLY`)
- No Phase D K-framing inherits Phase B's K
- No "rescue" runs after Phase D verdict — kill is kill
- Phase B verdicts that rely on N_OOS<50 cells are explicitly UNVERIFIED, not VERIFIED

## Verdict

Plan: APPROVED at Phase A (mechanical), pending user go for Phase B execution and explicit gate at C/D/E.

## Reproduction

This plan is the reproduction. Each phase task points to its acceptance criteria. Result documents under `docs/audit/results/` per Phase B/D; doctrine docs under `docs/institutional/` per Phase C.

## Caveats / limitations

- Plan does NOT auto-execute Phase D or E
- Phase F items are explicitly deferred; revisit only after main path lands
- "Edge extraction" here means "validate what's already in the data"; novel mechanism research (Phase 0 hypothesis-generation) is out of scope
- Small-OOS reality means some Phase D verdicts will be UNVERIFIED, not VERIFIED — this is a feature, not a bug
- MGC handling assumes equity-index ≠ metal regime; if cross-asset literature establishes otherwise (which it doesn't currently in `docs/institutional/literature/`), revisit C1 family definition
