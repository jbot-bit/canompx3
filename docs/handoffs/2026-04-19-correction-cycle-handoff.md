# Correction-cycle handoff — 2026-04-19

**Scope:** 9 corrections + 1 provenance fix + 1 multi-angle synthesis + "FULL PROPER PROCESS" addendum (6 more commits — hardening trio, MNQ retirement queue doc, MES low-ATR pre-reg + scan, review action items). Original 15 correction-cycle commits pushed at `64b281d7`; 6 addendum commits pushed at `00906e9e`. See "FULL PROPER PROCESS ADDENDUM" section below.

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

---

# FULL PROPER PROCESS ADDENDUM — 2026-04-19 extended

User asked for "FULL PROPER PROCESS" after correction cycle closure. Executed 6 additional commits addressing code-review action items + the MES ATR inverse-filter hypothesis surfaced by the multi-angle synthesis.

## Addendum commits (6 commits, `bf1595e3..00906e9e`)

| SHA | Commit | Purpose |
|---|---|---|
| `bf1595e3` | chore: hardening trio | Direction resolver + result_doc_header helper + synthesis relabel |
| `898a0b75` | docs: MNQ retirement queue | Formal committee-action doc with 4 Tier 1 DECAY + 9 Tier 2 REVIEW |
| `b9f38e33` | pre-reg(DRAFT): MES low-ATR K=20 | Hypothesis from synthesis Angle 1 → formal test |
| `e89c0d4f` | pre-reg(LOCK): MES low-ATR | commit_sha stamped |
| `0a86bc8d` | research: MES low-ATR scan — FAMILY KILL + hypothesis REFUTED | All 10 cells negative; Angle 1 falsified |
| `00906e9e` | chore: review action items | K-mismatch forward-check shipped + docstring alignment |

## Addendum headline — the scientific value of the low-ATR scan

The multi-angle synthesis § Angle 1 proposed: "MES ATR_P70 10/10-negative pattern from K=40 → bottom-70% ATR (~ATR_P70) should be profitable." Running this as a proper Pathway A family (K=10, all cells with N 500-600) produced:

**Every cell strongly NEGATIVE** (t from −2.04 to −7.33). All 10 cells' unfiltered baseline (`ExpR_b`) was ALREADY negative (−0.084 to −0.232). The ~ATR_P70 subset barely moves expectancy from that baseline.

**Correct interpretation:** MES unfiltered E2 breakouts are structurally unprofitable across the ENTIRE ATR distribution. The top-30% negative pattern in K=40 was NOT a low-ATR-edge diagnostic — it reflected MES's instrument-wide negative baseline. Every subset of unfiltered MES E2 is negative on average.

**Methodological value:** this is the correction cycle's most scientifically rigorous moment. A post-hoc pattern from the K=40 data was converted into a formal Pathway A hypothesis and immediately REFUTED by proper pre-registration. The synthesis doc's "NEW FINDING" framing (already relabelled to "NEW HYPOTHESIS — pre-reg required") was correctly flagged as data-snooping-adjacent.

## Honest bug found during the addendum: K=20 → K=10 arithmetic error

My pre-reg DRAFT declared `k_family: 20` with arithmetic "5 × 2 × 1 × 1 = 20" (wrong; correct = 10). Caught at scan-run-time. Fixed via:
1. Erratum block in result doc explaining the mismatch.
2. LOCKED pre-reg preserved unchanged per RULE 11 audit trail.
3. New `observed_cell_count` parameter in `result_doc_header.build_header()` that emits a K MISMATCH WARNING banner on every future run where pre-reg's declared K differs from the scan's observed K. This would have caught the bug at scan-start.

## New infra shipped

- **`research/result_doc_header.py`** — shared helper that reads title/SHA/script from pre-reg YAML. Enforces: LOCKED status required, commit_sha required, raises on missing fields. Optional `observed_cell_count` cross-check against `k_family`. Prevents the scan-clone copy-paste-bug class (caught 2 instances earlier in the correction cycle).

- **`research/regime_drift_control_critical_lanes.py::_resolve_direction`** — robust direction resolver with priority chain (execution_spec JSON → strategy_id _LONG_/_SHORT_ segment → long default with stderr warning). Current MNQ book is all long, so zero material impact; but future short lanes get explicit detection.

## New committee-action doc

**`docs/audit/results/2026-04-19-mnq-retirement-queue-committee-action.md`** — replaces the SUPERSEDED Category D RETIRE framing from the earlier committee pack with environment-controlled excess-decay ranking:

**Tier 1 (vote THIS WEEK, excess > 0.60 vs portfolio drop of −0.41):**
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` (excess −1.00)
- `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` (excess −0.87)
- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` (excess −0.85)
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` (excess −0.76, **late Sharpe NEGATIVE — highest urgency**)

**Tier 2 (vote within 2 weeks, excess 0.10 – 0.60):** 9 lanes listed with full rationale.

Pattern flagged for follow-up research: 8 of 13 decay candidates on EUROPE_FLOW session; 4 of 5 better-than-peers on COMEX_SETTLE / SINGAPORE_OPEN. Potential regime shift favouring pre-US-liquidity sessions.

## Addendum code-review grade

**A−** (one round). Action items addressed in commit `00906e9e`. Remaining non-blocking nit: scan scripts still hard-code scope tables (SESSIONS, DIRECTIONS) rather than reading from pre-reg YAML — acceptable since pre-reg YAMLs are scope-locked anyway.

## Push target
**Final HEAD:** `00906e9e` — pushing to `origin/main` now.

