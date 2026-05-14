---
task: Enumerate the actual BINDING criteria that Disposition C's family_singleton floor must enforce, given that pre_registered_criteria.md Amendment 2.1 already downgraded C5 (DSR) from binding to cross-check. Identify which of the 5 MES candidate rows + broader singleton universe would clear the corrected floor. Flag the intra-doc inconsistency in pre_registered_criteria.md acceptance-matrix vs enforcement-summary. No code change; design + analysis only.
mode: DESIGN
scope_lock:
  - docs/runtime/stages/stage3-c5-cross-check-floor.md
  - docs/audit/results/2026-05-11-stage3-c5-cross-check-floor-analysis.md
blast_radius:
  Reads gold.db read-only. Reads pre_registered_criteria.md and
  trading_app/dsr.py + strategy_validator.py for the C5-binding semantics.
  Writes only the two files in scope_lock (stage doc + analysis doc).
  No production code changes. No DB writes. No allocator changes.
  Stage 3 produces (a) the enumerated BINDING-criteria list for the
  family_singleton floor, (b) per-row evidence on which of the 5 MES
  candidates clear it today, (c) a doctrine-fix recommendation for the
  intra-doc inconsistency in pre_registered_criteria.md. Stage 4 (code
  change to deployability.py) consumes Stage 3's BINDING list.
---

## Background — what changed since Stage 2

Stage 2 (analysis doc + disposition pick) framed the family_singleton
floor as gated on Disposition C's locked criteria including a
DSR ≥ 0.95 hard gate (Criterion 5). The third-pass self-audit on Stage 2
(`2026-05-11-family-singleton-doctrine-analysis.md` §§ "THIRD-PASS
CORRECTION") established that:

- `pre_registered_criteria.md` Amendment 2.1 (committed 2026-04-07)
  officially downgraded C5 from binding to CROSS-CHECK ONLY pending
  N_eff resolution.
- `trading_app/dsr.py:35` documents the same status in code.
- The current state (zero rows pass DSR ≥ 0.95) is the doctrine working
  as designed, not drift.

So Stage 3's scope is much narrower than first framed: enumerate the
actual BINDING criteria the Disposition C floor needs to honour, and
audit which singletons clear them today.

## Stage 3 scope

In scope:

1. Enumerate which of C1-C12 are BINDING per
   `pre_registered_criteria.md` § "Enforcement summary table" (line 480-494).
2. Audit per-row evidence on the 5 MES candidates (already-named in
   Stage 1 / Stage 2 docs) against those BINDING criteria. Specifically:
   does each row have the criterion-status column populated, and what
   value does it carry?
3. Audit the broader MNQ + MES singleton universe (254 + 22 = 276 rows)
   for how many would clear the BINDING criteria if the family_singleton
   block were removed.
4. Flag the intra-doc inconsistency in `pre_registered_criteria.md`
   between the Acceptance Matrix (line 296 says "C5 DSR > 0.95 — YES")
   and the Enforcement summary (line 486 says "C5 DSR computed +
   reported — CROSS-CHECK ONLY"). Recommend a doctrine fix (separate
   workstream from Stage 4 code).
5. Document the corrected Disposition C floor explicitly: a list of
   BINDING criterion names + thresholds + where each is stored on
   `validated_setups` for the Stage 4 implementer to consume.

Out of scope:

- Stage 4 code change to `trading_app/deployability.py`. Stage 4 is a
  separate IMPLEMENTATION stage that consumes Stage 3's output.
- DSR computation re-engineering. The doctrine is C5-as-cross-check;
  Stage 3 honours that doctrine without modifying it.
- Stage 3.5 C8 backfill for the 5 MES candidates. Parallel workstream
  with its own stage file when triggered.
- The N_eff / ONC resolution that would let C5 become binding again.
  Separate research workstream; pre-registered as "task for next
  session" in `pre_registered_criteria.md` § "Amendment 2.1 Required
  follow-up."

## Done criteria

1. `docs/audit/results/2026-05-11-stage3-c5-cross-check-floor-analysis.md`
   exists with: (a) the BINDING criteria list with thresholds and
   `validated_setups` column references, (b) per-row evidence on the 5
   MES candidates, (c) universe-level pass-count on the 276 singletons,
   (d) the intra-doc inconsistency flag with a proposed doctrine fix,
   (e) NO code change, (f) the Stage 4 implementer's input is a
   structured criteria list that can be read mechanically. **[DONE]**
2. Stage doc parser checks pass (this file). **[DONE]**
3. User reviews and either (i) accepts the Stage 3 floor enumeration as
   Stage 4's input, or (ii) requests amendments before Stage 4 opens.
   **[DONE — Disposition C floor accepted as-is; see § Decisions Locked]**
4. No production files modified. **[DONE]**

## Decisions Locked — 2026-05-11

User picked on all three Stage 3 decisions:

**Decision 1 — Floor spec: ACCEPT AS-IS.** § 4 of the analysis doc is
Stage 4's binding input. Floor delegates to canonical t-stat helper for
C4; tests on each criterion; adversarial-audit gate before Stage 4 merge.

**Decision 2 — Doctrine fix: APPROVE AS SEPARATE WORKSTREAM.** Amend
`pre_registered_criteria.md:290-303` so the Acceptance matrix matches the
post-amendment Enforcement summary (line 480-494). MEDIUM severity,
audit-trail clarity, not capital-impacting. Not blocking Stage 4.
Separate stage file (`docs/runtime/stages/stage-doctrine-fix-pre-reg-matrix.md`)
will be opened in its own worktree.

**Decision 3 — C8 timing: HOLD STAGE 4; BACKFILL C8 SHELF-WIDE FIRST.**
Run the C8 OOS validator on all 844 rows where `c8_oos_status IS NULL`
before changing deployability code. Establishes the actual empirical
landscape before doctrine codification. This is **Stage 3.5** and is now
the next concrete action.

### Stage 3.5 — Shelf-wide C8 backfill (new stage to open)

Scope:
- Run the C8 OOS validator (whichever script writes `c8_oos_status` /
  `oos_exp_r` to `validated_setups`) on the 844 rows currently NULL.
- This is a DB-write operation: status flips from NULL to one of
  {PASSED, FAILED_RATIO, NEGATIVE_OOS_EXPR, INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH}.
- IS/OOS data is the canonical `orb_outcomes` joined to `daily_features`;
  the validator is pre-existing infrastructure.

Decisions Stage 3.5 must itself answer before running:
- Which script is canonical for writing `c8_oos_status`? Audit
  `trading_app/strategy_validator.py` and any C8-specific helpers.
- Does the C8 backfill require a pre-reg? Probably not — it is a
  confirmatory measurement against the pre-registered criterion, not new
  discovery (per `research-truth-protocol.md` § "Confirmatory audits do
  NOT require new pre-reg").
- What's the runtime? 844 rows × per-row OOS computation = likely hours.
  Worth a progress meter + checkpointing.
- Holdout policy: Mode A (sacred 2026-01-01) per
  `trading_app/holdout_policy.py`. The C8 evaluation must use that.

New stage file to write: `docs/runtime/stages/stage3.5-shelf-wide-c8-backfill.md`
(in its own worktree).

### What follows Stage 3.5

- Stage 4 (code change to deployability.py for family_singleton
  conditional downgrade) opens AFTER Stage 3.5 produces shelf-wide C8
  evidence.
- Stage 4's empirical unlock count is then a real number, not "0 today,
  unknown tomorrow."

## Sibling worktree status

- `stage1/generalize-tbbo-slippage-inference` — PR #258 open, CI green,
  awaiting merge.
- `stage2/family-singleton-doctrine` — Disposition C locked + 3 self-audit
  passes committed; awaiting docs-only PR (post-Stage-3, possibly bundled).
- `stage3/c5-doctrine-resolution` — THIS BRANCH. Stage 3 design + decisions
  locked.
- Next: open `stage3.5/shelf-wide-c8-backfill` worktree.
