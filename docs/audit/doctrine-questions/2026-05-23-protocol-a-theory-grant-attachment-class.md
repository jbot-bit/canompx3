# Doctrine Question — Protocol A theory_grant attachment class

**Date raised:** 2026-05-23
**Raised by:** Claude Code (Opus 4.7), explanatory mode
**Trigger:** `2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1` audit landed PASS_PROTOCOL_A under a mechanism-class-transfer construction. Local-literature audit found Chan Ch 7 extract grounds the cash-session-open stop-cascade mechanism but does NOT name the COST_LT08 cost-screen filter as alpha. Three sibling lanes already use the identical construction.
**Status:** **PROVISIONAL_OPTION_B + RE_AUDIT_OPEN** (operator decision 2026-05-23).

## Operator decision (2026-05-23)

**Adopted: provisional Option B for current-live continuity only. NOT permanent doctrine.**

Specifically:

1. The one currently live lane (`MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`, status=DEPLOY in `topstep_50k_mnq_auto`) stays deployed. No panic reversal.
2. The TOKYO_OPEN audit-log entry at `chordia_audit_log.yaml:709-722` stays as PASS_PROTOCOL_A-with-caveat. Result-MD body resolves to match the header under the same caveat.
3. The two audit-log-only PASS_PROTOCOL_A entries (NYSE_OPEN RR1.5 COST_LT12, TOKYO_OPEN RR1.5 COST_LT08) remain in the audit log but are NOT exemplars for new grants.
4. **NO NEW PASS_PROTOCOL_A grants using the mechanism-class-transfer pattern until Amendment 3.4 closes an independent audit.** Until then, new cells must either (a) clear strict no-theory Chordia t≥3.79, or (b) cite a literature extract that names the specific filter / cell class as an alpha mechanism.
5. The recommendation Option B was treated as a *proposal*, not as a permanent doctrine adoption. The "existing sibling precedents" do not strengthen the pattern — they may be a *repeated error* the project hasn't yet caught. Amendment 3.4 audit must independently establish whether the mechanism-class-transfer construction is doctrinally sound.

The provisional adoption is recorded in `pre_registered_criteria.md` as **Amendment 3.4 (PROVISIONAL)** with the same gating language.

---

## The question

When a Protocol A audit (`pre_registered_criteria.md` Amendment 3.0 / 3.3, Harvey-Liu t ≥ 3.00 with-theory hurdle) is granted, does the theory_grant attach to:

**Option A — Exact tested cell / filter**
The theory grant must cite a literature extract that grounds *the specific (session × entry × RR × filter) cell* as an alpha mechanism. If the extract grounds the parent session-entry mechanism but the cell's filter is project-engineered (e.g., a cost-screen, volume threshold, ATR percentile cutoff) with no extract naming that filter as alpha, the theory_grant is unsupported. Verdict reverts to no-theory; strict Chordia t ≥ 3.79 applies.

**Option B — Parent session-entry mechanism with cost-screen / filter riding on top**
The theory grant attaches to the parent session-entry mechanism (the literature-grounded edge primitive — e.g., Chan Ch 7's cash-session-open stop-cascade). A deployable filter sitting on top of a session-grounded edge does not require its own theory grant; it is treated as an operational deployment screen on a literature-grounded edge. Verdict stays Protocol A. The cell carries an explicit caveat that the grant covers the session-entry mechanism, not the filter class.

---

## Why this matters now

The two readings produce categorically different verdicts on the SAME row in `chordia_audit_log.yaml`:

- **Option A → FAIL_STRICT_CHORDIA** (revert prereg, revert result-MD header, delete entry at lines 709-722, append FAIL entry).
- **Option B → PASS_PROTOCOL_A stands** (existing line 709-722 entry is correct; result-MD body needs revision to match the header, not the other way around).

This is a **capital-decision-class** question because the answer also retroactively re-litigates already-deployed lanes (see § Affected sibling lanes below). One of those siblings is currently live-deployed in the auto-trading profile.

---

## What the local literature actually says

**Chan 2013 Ch 7** (`docs/institutional/literature/chan_2013_ch7_intraday_momentum.md:1-80`):

- p.155: grounds the **stop-cascade breakout mechanism** generically ("the triggering of stops … breakout strategies").
- p.156-157: FSTX gap-momentum strategy on Eurex equity-index futures — **cash-session-open momentum** with Sharpe 1.4 over 8 years.
- p.167: support / resistance level breach + stop-cluster mechanism.
- **Zero mentions** of `cost`, `cost ratio`, `cost screen`, `friction`, `narrow ORB`, `small range`, `COST_LT*`.

Chan Ch 7 is unambiguously a **parent session-entry mechanism** extract. It does not contain a named alpha mechanism for a cost-screen filter at any granularity. Whether that grounds the COST_LT08 (or COST_LT12, or ORB_G5) cell depends entirely on which attachment doctrine applies.

**Crabel ORB** — no extract present in `docs/institutional/literature/` (confirmed by `ls`). The Apr-22 prereg cited Crabel as the COST_LT08 theory-grant basis; that claim does not survive `institutional-rigor.md` § 7 inventory. The 2026-05-23 prereg supersedes it by switching the grant basis to Chan Ch 7. Whether that switch is doctrinally valid is the substance of this question.

**institutional-rigor.md § 7 wording** (cited verbatim):

> "Design decisions resting on literature claims must cite the specific extract file in `docs/institutional/literature/`. If the extract file does not yet exist for a paper you need to cite, write it first (extract the relevant passage from the PDF), then reference it."

The rule requires that the extract *exist* and that the claim be cited *from it*. It does NOT explicitly answer the attachment-class question. Both Option A and Option B can be read as compatible with § 7 — Option A reads "the relevant passage" strictly to mean the passage grounding the exact tested cell/filter; Option B reads it as the passage grounding the parent mechanism that the cell rides on.

**pre_registered_criteria.md Amendment 3.0** introduced the with-theory / no-theory threshold split (3.00 vs 3.79). Its language describes "theory grant" without further specifying the attachment class. This is the ambiguity that the present doctrine question seeks to close.

---

## Affected sibling lanes (using identical construction)

Audit log enumeration (`docs/runtime/chordia_audit_log.yaml`, `verdict: PASS_PROTOCOL_A` filter; 3 total entries):

| Strategy ID | Audit date | t_stat | Threshold | Live-deployed? | Theory-grant basis (per audit-log note) |
|---|---|---:|---:|---|---|
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 2026-05-01 | 3.412 | 3.00 | **YES** — `topstep_50k_mnq_auto` profile, status=DEPLOY | Chan Ch 7 (cash-session-open mechanism) |
| `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | 2026-05-07 | 3.600 | 3.00 | No | Chan Ch 7 (consistency grant inheriting from the RR1.0 sibling above) |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08` | 2026-05-23 | 3.566 | 3.00 | No | Chan Ch 7 (consistency grant; this audit) |

Additionally, the prereg for the new TOKYO_OPEN cell cites `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` as a precedent for "consistency grant" doctrine. That lane appears in the audit log with verdict `PASS_CHORDIA` (cleared the strict t ≥ 3.79 without needing a theory grant), so it is NOT affected by this doctrine question — strict-Chordia survival is independent of how theory-grant attachment is interpreted.

**Capital-at-risk count:**
- 1 currently live lane (NYSE_OPEN RR1.0 COST_LT12) would need re-audit under Option A.
- 2 audit-log entries (NYSE_OPEN RR1.5 COST_LT12; TOKYO_OPEN RR1.5 COST_LT08) would need re-verdict to FAIL_STRICT_CHORDIA under Option A.

Under Option B, all three stand as-is.

---

## What is paused

Per operator instruction (2026-05-23 conversation):
- No audit-log edits — entry at lines 709-722 of `chordia_audit_log.yaml` is left as written.
- No result-MD rewrite — `2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.md` retains its existing header / body contradiction (header PASS_PROTOCOL_A, body still FAIL_STRICT_CHORDIA from the earlier closeout pass). Will be revised to match whichever doctrine wins.
- No prereg edit — `2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock-v1.yaml` retains `theory_grant: true` as written.
- No stage closure — `docs/runtime/stages/2026-05-23-mnq-tokyoopen-costlt08-chordia-unlock.md` retains `mode: IMPLEMENTATION`.
- No allocator action on the new TOKYO_OPEN cell.
- No commits.

---

## Recommended doctrine rule (proposed for review — NOT adopted)

**Recommendation: Option B with explicit guardrails.** The mechanism-class-transfer pattern is the existing operational doctrine across three audit-log entries (one already live-deployed). Reverting to Option A would invalidate already-deployed capital decisions and would also force a stricter interpretation of `institutional-rigor.md` § 7 than the rule's wording requires.

But Option B is only safe with the following guardrails, which should be encoded in `pre_registered_criteria.md` as an Amendment (proposed Amendment 3.4) before being treated as canonical:

1. **Parent mechanism must be explicitly literature-extract-grounded.** Not a paraphrase, not "in the spirit of," not training-memory citation. The extract must be a file in `docs/institutional/literature/` and the cited passage must name the mechanism class.

2. **The filter must be operationally non-adversarial to the parent mechanism.** A cost-screen, volume threshold, or ATR percentile is *operationally compatible* with a session-entry mechanism — it gates *when* the parent mechanism is taken, not *whether* the parent mechanism exists. A filter that selects *against* the parent mechanism's premise (e.g., a counter-trend filter on a momentum-grounded entry) does NOT inherit the grant.

3. **The filter must be E2-deployment-safe.** Any look-ahead or post-entry-data filter is excluded (already enforced via `E2_EXCLUDED_FILTER_PREFIXES`).

4. **The cell-level t ≥ 3.00 hurdle still applies.** Theory grant does not waive the t-statistic floor; it only lowers the strict-Chordia 3.79 ceiling to the Harvey-Liu 3.00 with-theory hurdle. A cell failing 3.00 fails Protocol A regardless of attachment class.

5. **Explicit caveat in audit-log note.** The note must state "Grant covers parent session-entry mechanism, not filter class" — exactly the wording the current three audit-log entries already use. Without that caveat, the entry is ambiguous and triggers re-audit.

6. **No sizing-up.** PASS_PROTOCOL_A under Option B doctrine yields 1-contract deployment eligibility only. Sizing-up requires PASS_CHORDIA (strict t ≥ 3.79), which is independent of attachment class.

7. **Filter-class theory grant requires its own extract.** If we want to *claim* that COST_LT08 is itself a literature-grounded alpha mechanism (not merely a cost-screen riding on a parent mechanism), a separate extract must be written grounding cost-ratio-as-alpha. Harris 2002 Ch 14 § 14.2 (adverse-selection-dominates-spread) is the closest candidate but per the 2026-05-23 closeout analysis, Harris treats adverse-selection cost as a *deduction* from edge, not a *predictor* of edge — so even Harris would not lift COST_LT* to filter-class alpha grounding.

If guardrails 1-7 are accepted, the three existing PASS_PROTOCOL_A entries (including the new TOKYO_OPEN cell) are doctrine-consistent and stand. If any guardrail fails, the affected entry needs re-audit.

---

## Decision needed from operator

1. **Adopt Option A, Option B, or some refinement?**
2. **If Option B: accept guardrails 1-7 as Amendment 3.4, or revise?**
3. **If Option A: re-audit MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 immediately (live-deployed via this doctrine), revert the two PASS_PROTOCOL_A entries that depend on the doctrine, and pause any new mechanism-class-transfer grants until the doctrine is re-formalized?**

Once decided, the result MD body for the TOKYO_OPEN cell and any sibling adjustments follow mechanically. No code changes are required to enforce either option — the question is purely a doctrine-interpretation call on the existing `pre_registered_criteria.md` Amendment 3.0 wording.
