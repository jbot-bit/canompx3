# Finite-Data ORB Remediation Plan — Clean Worktree

**Date:** 2026-04-07
**Branch:** `wt-codex-finite-data-reaudit2`
**Status:** PROPOSED
**Purpose:** Convert the clean-worktree re-audit into an executable plan with explicit stage gates, failure conditions, and anti-bias controls.

---

## Goal

Bring the finite-data ORB research program to a state where:
- the evidence regime is unambiguous
- no-lookahead assumptions are enforced rather than narrated
- discovery scope is narrow enough to be statistically defensible
- DSR is used honestly relative to the repo's actual implementation state
- operational overlays are not smuggled into discovery claims

This plan is deliberately narrow. It is not a general research reboot.

---

## Governing rules

These rules apply to every stage below:

1. **Canonical layers only for truth-finding.**
   Use `bars_1m`, `daily_features`, and `orb_outcomes` for evidence. Derived layers and docs are advisory unless confirmed against canonical data or code.

2. **No new policy by implication.**
   A rule is only "real" when it is:
   - documented
   - implemented or explicitly marked informational
   - consistent with current code

3. **No holdout ambiguity.**
   The project must be in exactly one evidence regime at a time.

4. **No post-hoc threshold rescue.**
   If a threshold changes, the change must be justified by literature or a better estimator, not by disappointment with current lane outcomes.

5. **No overlay laundering.**
   Discovery evidence and execution overlays must be evaluated and reported separately.

6. **Branch hygiene.**
   This branch stays review-only. No runtime code changes unless a later explicit implementation branch is opened.

---

## Stage 0 — Freeze the evidence regime

### Objective

Choose whether the project is currently operating in:
- **Mode A: holdout-clean**
- **Mode B: post-holdout-monitoring**

### Deliverable

A single decision note that states:
- which mode is active
- what 2026 data may and may not be used for
- which docs are superseded
- what `pipeline/check_drift.py` should enforce

### Pass criteria

- One mode selected
- Contradictory docs identified
- Reviewers can answer "Is 2026 still clean OOS?" with a single unambiguous sentence

### Fail conditions

- "2026 is sacred" and "2026 is now included in discovery" both remain live truths
- Holdout cleanliness is left as a narrative interpretation rather than a policy choice

---

## Stage 1 — Repair governance gaps before new research

### Objective

Close the current governance holes without changing the research result set.

### Tasks

1. Activate holdout contamination enforcement:
   - make `check_holdout_contamination()` enforce the chosen Stage 0 policy
   - fail closed if the policy cannot be evaluated

2. Resolve the DSR policy mismatch:
   - either document DSR as informational until `N_eff` is solved
   - or implement and verify the missing `N_eff` path before claiming DSR is binding

3. Resolve missing literature grounding:
   - add the missing LdP 2020 extract to the clean branch
   - or remove/soften any rule that currently depends on that missing local artifact

### Deliverable

A governance patch set or governance doc amendment set, separately reviewable from research results.

### Pass criteria

- Holdout guard is active and reviewable
- DSR policy in docs matches the code reality
- Every rule cited in the finite-data framework has a local source extract or is explicitly demoted from binding status

### Fail conditions

- DSR remains a hard rule only in prose
- Holdout checker still passes without checking anything
- A core literature citation is still missing locally while being used as authority

---

## Stage 2 — Define the narrow rediscovery universe

### Objective

Constrain the next discovery family to a small, mechanically defensible E2 ORB universe.

### Allowed scope

- Entry model: `E2` only
- ORB aperture: fixed
- Filters: price-based only
- Instruments: only those justified by the chosen evidence regime and source disclosure
- Sessions: explicitly listed in advance
- RR targets: explicitly listed in advance

### Excluded scope

- `VOL_RV*`
- `ATR70_VOL`
- `_FAST`
- `_CONT`
- any filter that depends on break-bar completion for E2
- any execution overlay folded into the discovery ID

### Deliverable

A pre-registered hypothesis file with:
- exact family definition
- exact trial count
- theory citation per hypothesis
- kill criteria

### Pass criteria

- K is known before any run
- every tested family member is mechanically available at decision time
- every family member maps to one declared hypothesis

### Fail conditions

- vague "we’ll test a few sessions and see"
- overlays sneaking into discovery claims
- new dimensions added after seeing outputs

---

## Stage 3 — Run discovery under honest K

### Objective

Run a narrow discovery pass that is actually auditable.

### Required controls

- use `--holdout-date` if Stage 0 selected holdout-clean mode
- preserve the exact pre-registered universe
- record total raw K and the declared family K
- keep output lineage explicit

### Deliverable

A discovery result bundle tied to the pre-registration:
- query/script path
- date span
- instrument/session/filter list
- K used for BH FDR

### Pass criteria

- run matches pre-registration exactly
- no family growth during execution
- no contaminated E2 filter families appear in the output

### Fail conditions

- discovery scope drifts from the registration
- undocumented exclusions/inclusions appear after the run
- result interpretation starts before K is locked

---

## Stage 4 — Validate with honest status labels

### Objective

Separate what is proven, what is suggestive, and what is merely operational.

### Validation stack

Required:
- BH FDR on the pre-registered family
- WFE
- sample size / time-span disclosure
- era stability
- OOS reporting under the chosen evidence regime

Informational until solved:
- DSR, unless Stage 1 converted it into a truly implemented and verified gate

### Deliverable

A lane-by-lane table with four status classes:
- `REJECT`
- `RESEARCH-PROVISIONAL`
- `CANDIDATE`
- `OPERATIONAL-PROVISIONAL`

### Pass criteria

- no lane is called "validated" unless the repo can actually defend that label
- DSR status is disclosed honestly
- OOS status matches the chosen Stage 0 evidence regime

### Fail conditions

- "validated" used loosely
- DSR omitted when it is inconvenient
- OOS described as clean when the regime says it is not

---

## Stage 5 — Audit overlays as overlays

### Objective

Prevent operational improvements from being narrated as discovery proof.

### Overlay classes to isolate

- calendar skip / half-size
- ATR velocity skip
- E2 timeout
- market-state gating
- any live-only position sizing or risk-layer filters

### Deliverable

A separate overlay audit that answers:
- what the raw discovered strategy does
- what the live system adds on top
- whether the overlay was itself researched and in what evidence regime

### Pass criteria

- overlays are documented independently from the discovered filter
- reviewers can separate "edge claim" from "live execution hygiene"

### Fail conditions

- overlays described as part of the original statistical evidence
- live performance attributed to discovery filters alone when overlays changed the decision rule

---

## Stage 6 — Deployment language and monitoring

### Objective

Make the deployment label match the true evidence level.

### Default deployment labels after this plan

- `RESEARCH-PROVISIONAL`: survives narrow rediscovery but lacks enough forward evidence
- `OPERATIONAL-PROVISIONAL`: allowed to trade at minimal scale with monitoring
- `PRODUCTION`: reserved for strategies that pass future forward-evidence requirements beyond this plan

### Deliverable

A language policy for docs and dashboards so current lanes are not overstated.

### Pass criteria

- current lanes are no longer framed as stronger than the repo can defend
- monitoring requirements are explicit

### Fail conditions

- "institutionally validated" used before the governance and DSR gaps are closed

---

## Recommended order of execution

1. Stage 0 — freeze evidence regime
2. Stage 1 — repair governance
3. Stage 2 — write narrow pre-registration
4. Stage 3 — run clean discovery
5. Stage 4 — validate with honest status labels
6. Stage 5 — audit overlays separately
7. Stage 6 — update deployment language

Do not skip directly to Stage 3. That would just recreate the current ambiguity in a neater wrapper.

---

## Minimal success condition

This plan succeeds if, after completion:
- one evidence regime governs the repo
- the next ORB discovery run is narrow and pre-registered
- DSR is described honestly relative to implementation state
- overlays stop contaminating discovery claims
- Claude can review the work without needing to guess what standard is supposed to apply

Anything weaker than that is another narrative cleanup pass, not a methodological fix.
