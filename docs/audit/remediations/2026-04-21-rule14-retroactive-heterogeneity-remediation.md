# Remediation — RULE 14 retroactive heterogeneity audit classification

**Date:** 2026-04-21
**Branch:** `research/ovnrng-router-rolling-cv` (Claude terminal, 6lane-baseline worktree)
**Source finding:** `docs/audit/results/2026-04-21-post-hoc-rejection-sweep.md` (commit `39315b52`)
**Affected artifact:** `docs/audit/results/2026-04-20-heterogeneity-audit-phase3-results.md` (commit `aa3399b3`, branch `research/retroactive-heterogeneity-audit`)

---

## Original sweep classification

My sweep (commit `39315b52`) labelled the RULE 14 retroactive audit as **BORDERLINE_POST_HOC** — applying a new rule (≥25% cell-flip = heterogeneity artefact) to prior findings after-the-fact is structurally the same pathology as post-hoc rejection, even when the rule is mechanistically sound.

After detailed read of the audit artifact, this remediation REFINES that classification. RULE 14 is **not currently damaging** live or CANDIDATE_READY findings — it refined framings on claims that were already dead or NO-GO. The cautionary principle stands for future applications.

---

## Per-claim assessment of the RULE 14 audit's 4 classifications

### A1 — pit_range signal: BLOCKED (not a RULE 14 issue)

**RULE 14 audit verdict:** BLOCKED — `daily_features.pit_range_atr` is 0% populated; feature cannot be retroactively re-evaluated.

**Remediation status:** HONEST_FINDING. This is an infrastructure gap, not a heterogeneity kill. The memory claim "F5 pit range/ATR — FEATURE validated, FILTER deployed (PIT_MIN ≥ 0.10)" was stale because the backfill step was never run. No deployed strategy uses PIT_MIN. No remediation needed beyond the memory update already recommended in the audit doc.

**Action:** NONE. Memory entry already corrected in `memory/exchange_range_signal.md` per memory log `2026-04-20 (final)`. Infrastructure gap to be addressed separately (run `pipeline/ingest_statistics.py --all` + backfill).

### A2 — H2 Path C garch_vol_pct≥70 "universal": HETEROGENEOUS

**RULE 14 audit verdict:** HETEROGENEOUS — 31.5% cell flip; NYSE_OPEN is inverse; Asia-heavy positive. "Universal" framing REJECTED.

**Key fact:** the underlying finding `H2 Path C garch_vol_pct≥70 + NO CAPITAL` was already NO-GO before the RULE 14 audit. The audit did NOT kill a deployed signal or a CANDIDATE_READY cell. It corrected the framing of an already-shelved finding from "universal" to "session-heterogeneous."

**Remediation status:** REFINEMENT_VALID. The framing correction is legitimate and should stand. No restoration required — nothing was deployed, nothing was CANDIDATE_READY.

**Going-forward rule (from the audit):** any future garch-overlay proposal must be per-session pre-reg, not universal. This is sound institutional practice.

**Action:** NONE. Framing correction accepted.

### A3 — comprehensive scan 13 K_global survivors "universal volume confirmation": PARTIALLY HETEROGENEOUS

**RULE 14 audit verdict:** PARTIALLY HETEROGENEOUS — 13 K_global survivors all use the SAME feature (`rel_vol_HIGH_Q3`), and 5 of 12 sessions (NYSE_OPEN, NYSE_CLOSE, US_DATA_830, US_DATA_1000, CME_REOPEN) have zero survivors. "Universal" framing incorrect; it's "rel_vol_HIGH_Q3 robust at 7/12 sessions, absent at US cash sessions".

**Key fact:** the comprehensive-scan finding was a LIST of K_global BH-FDR survivor cells. It was never a single-decision deployable unit. Individual survivor cells (like MES + MNQ COMEX_SETTLE short) remain RESEARCH_SURVIVOR status per their own prior pre-regs. The RULE 14 audit corrected the INTERPRETATION of the list from "universal" to "session-subset" — it did not kill the individual cells.

**Remediation status:** REFINEMENT_VALID. The framing correction is legitimate and should stand. No individual cell's status changed.

**Going-forward rule (from the audit):** any future overlay proposal stating "volume confirmation universal" must cite the per-session distribution, not the K_global count alone. Session-targeting in future rel_vol pre-regs should focus on the 7 sessions where rel_vol_HIGH_Q3 is robust.

**Action:** NONE. Framing correction accepted. Note that this interacts with the role-selection meta-pre-reg (`draft-2026-04-21-rel-vol-role-selection-meta-v1.yaml`): the ALLOCATOR role may want to underweight NYSE cash sessions for rel_vol-keyed allocation.

### B3 — break_quality "universal null": NO-GO HOLDS

**RULE 14 audit verdict:** NO-GO HOLDS; memory framing updated.

**Key fact:** pooled N=18-27K + pooled p~0.5 means the underlying effect is genuinely zero, not masked by heterogeneity. The B3 NO-GO conclusion was correct both before and after the RULE 14 audit. The audit's memory-framing correction is minor housekeeping.

**Remediation status:** HONEST. No action.

---

## Overall conclusion

The RULE 14 retroactive audit, on inspection of its actual content, **did not damage any live signal, CANDIDATE_READY cell, or deploy-eligible finding**. It refined framings on claims that were already dead, NO-GO, or interpretation-level. The original sweep classification "BORDERLINE_POST_HOC" is slightly harsh — a more accurate label is **RETROACTIVE_FRAMING_REFINEMENT** (valid, low-damage, does not trigger the post-hoc-rejection remediation path).

**The cautionary principle still stands for future RULE-14-type audits:** if a retroactive rule-application kills a LIVE or CANDIDATE_READY finding, that IS post-hoc rejection proper (per `backtesting-methodology.md` RULE 3.5) and requires the full remediation pathway. The current audit just didn't hit any such findings.

---

## Going-forward rules (from audit, accepted)

1. Any future garch-overlay proposal must be per-session pre-reg, not universal.
2. Any future "universal volume confirmation" framing must cite per-session distribution, not K_global count alone.
3. Memory entries referencing "universal" pooled findings must be cross-checked for per-cell distribution before propagating to future work.
4. Pre-regs must include a per-lane heterogeneity gate (cell-flip rate ≤ 25%) alongside pooled statistics — codified in `backtesting-methodology.md` RULE 14 (existing).

These are institutional hygiene improvements, not remediation actions.

---

## Relationship to PR #51 + PR #50 DSR remediation

**Important contrast:** the PR #51/#50 DSR kill WAS damaging — it moved 5+2 CANDIDATE_READY cells to "MISCLASSIFIED" status via post-hoc criterion application. That remediation (companion file `2026-04-21-pr51-pr50-dsr-remediation.md`) actively restores CANDIDATE_READY status.

The RULE 14 audit didn't cause equivalent damage. Both are POST_HOC_REJECTION pathology, but impact severity is very different:

| Audit | Pathology pattern | Damage severity | Remediation action |
|---|---|---|---|
| PR #59 sizer re-audit | Post-hoc gates (Sharpe/Spearman/bootstrap/rival form) | MODERATE (relabelled MGC pre-reg pass as MISCLASSIFIED) | Codex already corrected (`3df2acb1` — research-pass + deploy-reject) |
| PR #51 DSR kill | Post-hoc gate (C5 DSR as hard kill despite Amendment 2.1) | HIGH (killed 5+2 CANDIDATE_READYs) | This session: restore with DSR-PENDING label |
| RULE 14 retroactive audit | Retroactive rule application to memory claims | LOW (no live/CANDIDATE_READY signals affected) | This doc: accept framing refinements, no restoration needed |

---

## What this remediation does NOT do

- Does NOT challenge the RULE 14 rule itself — it is codified correctly in `backtesting-methodology.md` going forward.
- Does NOT restore any claim the RULE 14 audit touched — all were already dead/NO-GO/framing-level.
- Does NOT alter memory entries — those were corrected in the audit doc's recommendations.
- Does NOT modify `validated_setups` / `edge_families` / `lane_allocation` / `live_config`.
- Does NOT touch Codex's active rel_vol filter-form runner work (different branch, different worktree, different feature family).

---

## Provenance

- RULE 14 audit artifact: `docs/audit/results/2026-04-20-heterogeneity-audit-phase3-results.md` (commit `aa3399b3`)
- RULE 14 codification: `.claude/rules/backtesting-methodology.md` RULE 14 (pre-existing)
- RULE 3.5 codification (post-hoc rejection): `.claude/rules/backtesting-methodology.md` (commit `631bda30` this session)
- Failure-log entry: `.claude/rules/backtesting-methodology-failure-log.md` 2026-04-21 (commit `631bda30`)
- Parallel-session safety: Codex is executing MES/MGC filter-form runner on `research/pr48-sizer-rule-oos-backtest` branch in main worktree; this remediation is methodology/memory housekeeping, entirely non-overlapping.
- 2026 OOS (Mode A sacred) UNTOUCHED.
