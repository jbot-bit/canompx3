# Correction-cycle handoff — 2026-04-19

**Scope:** 9 corrections + 1 provenance fix + 1 multi-angle synthesis, 16 commits landed on `main` (not yet pushed). Builds on the overnight session (commits `341132ba..a281481e`, already pushed) and the subsequent self-audit that identified 9 remediation items.

---

## HEADLINE — what changed tonight

### Reframes

1. **Committee RETIRE verdict on 4 CRITICAL MNQ lanes WITHDRAWN.** Regime-drift control (Correction 3) shows the 4 lanes track portfolio-wide −0.41 Sharpe drop within 0.30; two are actually BETTER-than-peers. The KILL framing was over-attributed to lane decay without environment control.
2. **Honest retirement candidates identified instead:** 4 MNQ lanes with excess-drop > 0.60 vs portfolio — CROSS_SGP RR1.5/2.0, COST_LT12 RR1.5, US_DATA_1000 ORB_G5 O15. The last has NEGATIVE late Sharpe. Committee should vote THESE, not the originally-flagged 4.

### New negative evidence (closes audit gaps)

3. **MGC short Pathway A K=4** → FAMILY KILL (commit `ac1fe028`). Symmetric to the long sibling kill. MGC has no Pathway A edge either direction under 3.5-year native data.
4. **MES comprehensive K=40** (5 sessions × 4 filters × 2 directions × RR=1.5) → FAMILY KILL (commit `76475a34`). No broad-generalising MES feature signal beyond the 1 already-validated `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` lane.

### Integrity confirmations

5. **Rel_vol_HIGH_Q3 13-survivor finding holds under IS-only quantile** (Correction 6). 0/13 drift; max |Δt|=0.44; the quantile-time look-ahead concern did NOT materialise into survivor changes. Parent scan (`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md`) stands.
6. **0/23 retired setups revive under Mode A** (Correction 2). The retirement decisions were correct.
7. **RULE 13 pressure-test passes** on the 4 new scan scripts (Correction 1). Guards fire on synthetic BAD inputs.

### New parked research

8. **MGC long signal-only shadow pre-reg** (Correction 4, commit `8c7c228e`). Parks 4 underpowered-but-positive-ExpR MGC long cells for re-evaluation at N>=200.
9. **MES US_DATA_1000 ORB_G5 short shadow pre-reg** (Correction 5, commit `ebf46243`). Parks H4/H5 from Phase 7 (t=2.48/2.55, 7/7 positive years, dir_match OOS).

### Multi-angle synthesis (opens the KILL lens)

10. **MES `ATR_P70` is a MES-specific inverse signal** (synthesis doc `2a466e83`). All 10 ATR_P70 cells across 5 MES sessions × 2 directions show NEGATIVE ExpR. Candidate for a follow-up Pathway A pre-reg testing `ATR_LT70` (bottom-70% ATR) as a skip-filter.

---

## Commit summary (16 commits, not yet pushed)

| SHA | Correction | Summary |
|---|---|---|
| `9937ebf6` | C3 | regime-drift control — 4 CRITICAL reframed as REGIME |
| `c75d7a3b` | C9 | committee pack addendum — RETIRE withdrawn |
| `b9bacd0a` | C9 | overnight handover addendum — RETIRE banner |
| `8c7c228e` | C4 | MGC long shadow pre-reg |
| `ebf46243` | C5 | MES US_DATA_1000 short shadow pre-reg |
| `4ac0b688` | C7 | MGC short pre-reg DRAFT |
| `9e32c002` | C7 | MGC short pre-reg LOCK |
| `ac1fe028` | C7 | MGC short scan — FAMILY KILL |
| `560ff83a` | C6 | IS-only quantile sensitivity — 0/13 drift |
| `d8ad96f2` | C8 | MES comprehensive K=40 pre-reg DRAFT |
| `1ff59c4a` | C8 | MES comprehensive K=40 pre-reg LOCK |
| `76475a34` | C8 | MES comprehensive K=40 scan — FAMILY KILL |
| `f04f9745` | fix | provenance headers on 2 scan result docs |
| `2a466e83` | synthesis | multi-angle re-read of correction cycle |

Plus `b91b92d9` (C1) and `5bfcc53d` (C2) earlier this cycle, already part of local main.

## Code-review of this cycle

Grade: B+. Stats clean, FAMILY KILL verdicts correct, DRAFT→LOCK cadence followed, no look-ahead, canonical filter delegation throughout. Two copy-paste bugs in scan-result-doc HEADER metadata (wrong pre-reg + commit SHA cited) caught and fixed in `f04f9745`. Data bodies were unaffected.

Suggested code hardening (not yet done):
- Add `research/result_doc_header.py::build_header(prereg_path)` helper so scan scripts read title/SHA/script path from the pre-reg YAML instead of hardcoded literals. Prevents a third occurrence of the copy-paste bug.
- `regime_drift_control_critical_lanes.py` infers direction from `execution_spec` via string contains — brittle if serialization changes. Assert non-null and match an enum.

## What a committee should see first

1. **`docs/audit/results/2026-04-19-correction-cycle-multi-angle-synthesis.md`** — the 5 under-actioned opportunities, ranked. Start here.
2. **`docs/audit/results/2026-04-19-regime-drift-control-critical-lanes.md`** — the 13-lane MNQ excess-decay list. Honest retirement candidates.
3. **`docs/audit/results/2026-04-19-mnq-mode-a-committee-review-pack.md`** § ADDENDUM — the reframed verdict on the original 4 CRITICAL lanes.
4. **`docs/audit/results/2026-04-19-mes-comprehensive-mode-a-feature-v1-scan.md`** — K=40 scan; note the ATR_P70 inverse-signal pattern called out in the synthesis doc.
5. **`docs/audit/results/2026-04-19-rel-vol-is-only-quantile-sensitivity.md`** — confirms the 13-survivor narrative from 2026-04-15 is integrity-clean.

## What's queued (decisions/actions beyond this cycle)

| Item | Recommendation |
|---|---|
| Committee vote on 4 excess-decay MNQ lanes (CROSS_SGP RR1.5/2.0, COST_LT12 RR1.5, US_DATA_1000 ORB_G5 O15) | THIS WEEK — data shows genuine decay |
| Pre-reg MES `ATR_LT70` inverse-filter Pathway A K=20 | NEXT SESSION — novel angle from K=40 data |
| 2nd MES coverage pre-reg: NYSE_OPEN + US_DATA_830 | Optional — before declaring MES structurally flat |
| Build Phase-15 shadow-recorder infrastructure | Design-doc exists; build blocks capital deployment on C4/C5 shadows |
| Push tonight's 16 commits to origin | On user sign-off |
| Apply code-review code-hardening suggestions | Low priority, housekeeping |

## What was NOT touched (clean)

- `pipeline/*` untouched.
- `trading_app/*` untouched.
- `scripts/*` untouched.
- `validated_setups` / `experimental_strategies` — zero writes, read-only audits throughout.
- gold.db opened READ_ONLY by every new script.

## Data integrity

- 16 new commits on local main; all clean; `git log origin/main..HEAD` shows 16.
- `pipeline/check_drift.py` — pre-existing env-only failure unchanged.
- Every scan that claimed a verdict ran to completion and emitted a result doc (verified by running each).

## Closing honesty check

The correction cycle DID initially frame in a KILL-centric lens — that's what Pathway A pre-reg tests do. The "NO TUNNEL VISION" pushback from the user caught the framing bias. The multi-angle synthesis doc (`2a466e83`) is the explicit answer: 5 opportunities that Pathway A t>=3 framing filters out but the data supports. The ATR_P70 MES inverse signal (10/10 negative cells across 5 sessions × 2 directions) is the biggest NEW finding buried in the correction cycle's "40 kills" headline.

---

**Session ended:** 2026-04-19 post-correction cycle
**HEAD:** `2a466e83`
**Branch:** `main`
**Origin:** 16 commits ahead (ready to push)
