---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Multi-Phase Audit Roadmap — Post-Adversarial-Audit

**Date:** 2026-04-18
**Authority:** supersedes the single-action framing in `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md` § Next actions
**Scope:** all remaining work surfaced by the adversarial portfolio audit + A2b Stage-1 scope doc, sequenced with dependencies + parallel tracks + kill criteria
**Method:** 6 design iterations before commit

**Principle:** each phase gets its OWN full plan-iterate-design cycle. This roadmap is the gate, not the execution. It orders the phases and defines dependencies; it does NOT prescribe implementation details for any single phase.

**Non-goals of this roadmap:**
- Does not consume OOS
- Does not modify canonical code
- Does not pre-commit to any specific patch or implementation detail
- Does not block parallel decisions (copies scaling, literature expansion) on the main-line sequence

---

## Inventory of work items

From the adversarial audit, A2a/A2b scope docs, and open-minded brainstorm:

### Main-line audit track (serial — each depends on prior)

| ID | Description | Lit | Blast | Est. cost | EV δ |
|---|---|:---:|:---:|:---:|:---:|
| A2a | Rho audit: 15 ALLOCATOR_NOT_SELECTED lanes vs live 6 (Stage-2 design already locked) | n/a | LOW | 1-2 hrs | 0-30 R/yr |
| A2b-1 | Filtered per-lane regime gate (fix verified bug in `_compute_session_regime`) | ✓ Pepelyshev-Polunchenko + Chan Ch 7 | MED | 2-3 days | 5-15 R/yr + systemic |
| A2b-2 | DSR ranking (replace `_effective_annual_r` with Bailey-LdP 2014 Deflated Sharpe or Harvey-Liu 2015 haircut) | ✓ Bailey-LdP 2014 + Harvey-Liu 2015 | LOW | 3-5 days | 10-30 R/yr |
| A2b-3 | Half-Kelly sizing (replace flat-1-contract with Carver vol-target-scaled) | ✓ Carver 2015 Ch 9 Table 25 | MED | 5-7 days | 10-15% dollar-R uplift |

### Implementation-quality track (independent)

| ID | Description | Blast | Est. cost | EV δ |
|---|---|:---:|:---:|:---:|
| Q1 | Order-routing / fill-quality audit on live 6 lanes | LOW (read-only) | 1 day | 0-5% fill improvement |

### Capital-scaling track (user-decision, parallel)

| ID | Description | Blast | Gate | EV δ |
|---|---|:---:|---|:---:|
| C1 | Copies 2→5 operationalization | MED-HIGH | User-defined proving-loop criterion | 2.5× aggregate (~600 R/yr) |

### Background enablement track (parallel)

| ID | Description | Est. cost | Unlocks |
|---|---|:---:|---|
| L1 | Literature expansion: Markowitz 1952 | 2-4 hrs | Correlation threshold axis |
| L2 | Literature expansion: Ledoit-Wolf 2004 | 4-6 hrs | Shrinkage covariance axis |
| L3 | Literature expansion: LdP 2020 Ch 4-7 | 6-10 hrs | HRP + CPCV axes |
| L4 | Literature expansion: Meucci 2010 | 2-4 hrs | Effective-bets slot count |
| L5 | Literature expansion: Artzner et al 1999 | 2-4 hrs | Coherent DD measure axis |
| L6 | Literature expansion: Leland 1999 | 2-4 hrs | Rebalance-frequency / turnover axis |

### Discovery track (deferred to Phase 6+)

| ID | Description | Lit | Blast | Est. cost | EV δ |
|---|---|:---:|:---:|:---:|:---:|
| D1 | MGC fresh discovery under strict Mode A holdout | ✓ covered by existing rules | MED | 1-2 sessions | 0-20 R/yr (P=30-50%) |
| D2 | Tick-level order-flow infrastructure (Databento MBO pilot) | Partial (needs Harris/O'Hara) | LARGE | 1-2 weeks | HIGH if real |
| D3 | Linked: HRP allocator (requires L3 literature) | depends on L3 | HIGH | 1-2 weeks | MED-HIGH |

---

## Dependency graph

```
                     ┌─────────────────────────────────────────────┐
                     │                                             │
                     │           PARALLEL TRACKS                   │
                     │                                             │
                     │  L1-L6 (literature expansion, any order)    │
                     │  C1 (copies 2→5, user-decision)             │
                     │                                             │
                     └─────────────────────────────────────────────┘

  MAIN LINE (serial):

  ┌─────┐    ┌───────┐    ┌───────┐    ┌───────┐    ┌─────┐
  │ A2a │───▶│A2b-1  │───▶│A2b-2  │───▶│A2b-3  │───▶│ Q1  │
  │     │    │regime │    │ DSR   │    │Half-K │    │fill │
  │     │    │ gate  │    │rank   │    │ size  │    │audit│
  └─────┘    └───────┘    └───────┘    └───────┘    └─────┘
    │           │            │            │            │
    │           │            │            │            │
    ▼           ▼            ▼            ▼            ▼
 (unlocks?) (bug fix)    (theory        (dollar-R   (fill
            + systemic    rerank)         uplift)    quality)

  DEFERRED (Phase 6+, contingent):
  D1 (MGC discovery)     — any time after main-line
  D3 (HRP allocator)     — requires L3 extracted first
  D2 (tick-level infra)  — long-term infrastructure project
```

**Rationale for ordering:**

- **A2a first:** fastest, read-only, informs A2b-1 scope. If A2a unlocks a lane via rho-gate audit, that lane may interact with A2b-1's regime-gate fix.
- **A2b-1 second:** fixes a VERIFIED bug (unfiltered regime gate). Per `.claude/rules/institutional-rigor.md` Rule 4 (bug-fix-first discipline), verified defects land before theory-grounded optimizations.
- **A2b-2 third:** DSR ranking uses regime-classifications from A2b-1. If regime gate is wrong, DSR ranks the wrong inputs. Strict dependency.
- **A2b-3 fourth:** Half-Kelly sizing requires realistic SR estimates per lane, which A2b-2's DSR machinery provides. Strict dependency.
- **Q1 fifth:** implementation-quality. Lower EV than the 4 main-line items. Sequence it after main-line stabilizes so the routing assumptions aren't changing under the audit's feet.
- **Parallel tracks (C1, L1-L6):** no technical dependency on main line. Run independently.
- **Discovery track (D1-D3):** lower certainty, deferred until post-main-line. Allocator-optimization finishes first so discovery benefits from upgraded ranking/sizing.

---

## Per-phase required deliverables

Every phase must produce:

1. **Stage-1 scope doc** — plan-iterate-design locked, committed to `docs/audit/hypotheses/` BEFORE implementation
2. **Stage-1 verified extract citations** — every literature claim verified against `docs/institutional/literature/` content (not title-match)
3. **Pre-committed kill criteria** — numeric, unambiguous, no "revisit" wording
4. **Pre-committed rollback plan** — `git revert` path + live-lane regression test spec
5. **Mode A OOS consumption audit** — explicit declaration that no 2026-sacred data is consumed for selection/tuning
6. **Stage-2 implementation** — only after Stage-1 locked
7. **Stage-3 verification** — py_compile + drift check + tests + self-review
8. **Stage-4 commit + push** — single focused commit with full message
9. **Stage-5 post-audit** — HANDOFF.md update + memory file entry + TaskList cleanup

---

## Cross-phase invariants (must hold at every commit)

- Drift check 0 attributable violations (pre-existing unrelated violations documented but not introduced)
- Shiryaev-Roberts live monitor pipeline unaffected (C12 reviews continue working)
- Live 6 lanes continue trading with no unintended status change
- All test files pass (existing test suite regression-tested)
- Mode A holdout boundary unchanged
- Canonical sources remain single-source-of-truth (no re-encoding)
- HANDOFF.md reflects current state accurately
- `docs/institutional/literature/` extracts are authoritative for any literature claim

---

## Kill criteria per phase (template)

**Pre-commit kill criteria must be:**
- **Numeric** (not "we'll see")
- **Pre-committed** (written BEFORE running)
- **Non-tunable** (cannot be weakened mid-audit to rescue a bad result)

**Generic template per phase:**

| Gate | Metric | Threshold | Action if tripped |
|---|---|---|---|
| K1 | Tests fail | any pre-existing test fails post-patch | HALT, do not commit |
| K2 | Drift check regresses | violation count increases | HALT, audit the addition |
| K3 | Live-lane status changes unintended | unexpected DEPLOY→PAUSE or vice-versa | HALT, re-audit gate logic |
| K4 | Regression benchmark | IS performance on live 6 degrades post-change | HALT, revert, investigate |
| K5 | OOS contamination detected | any 2026-sacred data used for selection | HALT, scrub, pre-reg amendment |
| K6 | Cross-phase invariant broken | any of the 8 invariants above | HALT, fix before continuing |

**Phase-specific kill criteria** are defined in each phase's Stage-1 scope doc.

---

## Rollback protocol (template per phase)

Each phase's commit must include an explicit rollback plan:

1. Identify the EXACT commit SHA that landed the change
2. Document the `git revert <sha>` command
3. List files that need restoration (code, config, docs)
4. Live-lane verification after revert (must match pre-commit state)
5. Post-revert HANDOFF.md entry documenting the revert + reason

Each phase's Stage-1 design must include the rollback plan BEFORE implementation begins.

---

## Parallel tracks — how to run them

### Capital scaling (C1): copies 2→5

- **Owner:** user
- **Gate:** user-defined "proving-loop" criterion committed in writing before scaling
- **Proposed criteria (to finalize):** N live trades ≥ 200, zero DD-rule breaches, aggregate trailing 6mo Sharpe > 0.5, no major lane paused in prior month
- **Rollback:** copies can drop back to 2 at any time
- **Cross-interaction with A2b-3:** if A2b-3 ships first, copies scale compounds with Half-Kelly sizing (both multiplicative). If copies scale first, A2b-3 replaces flat sizing later. Either order works.

### Literature expansion (L1-L6): 2-10 hrs per paper

- **Owner:** any session with slack time
- **Gate:** none
- **Priority order** (by unlock value):
  1. L3 (LdP 2020 Ch 4-7) — unlocks HRP + CPCV (biggest axis unlock)
  2. L2 (Ledoit-Wolf 2004) — unlocks shrinkage covariance
  3. L1 (Markowitz 1952) — unlocks correlation-threshold theory
  4. L4 (Meucci 2010) — unlocks effective-bets
  5. L5 (Artzner 1999) — unlocks coherent DD
  6. L6 (Leland 1999) — unlocks turnover theory
- **Interaction with main-line:** L3 unlocks D3 (HRP allocator) as a post-main-line discovery track. Others unlock future A2b audits beyond A2b-1/2/3.

---

## Decision tree — when each phase starts

```
START (now) → A2a ready-to-execute (no new design work — locked in conversation)
              │
              ▼
         [A2a ships]
              │
              ▼
         A2b-1 Stage-1 design
              │
              ▼  (user reviews, approves design)
         A2b-1 Stage-2 implementation
              │
              ▼
         [A2b-1 ships] + Mode A verification
              │
              ▼
         A2b-2 Stage-1 design
              │
              ▼
         A2b-2 ships + A2b-3 Stage-1 design
              │
              ▼
         A2b-3 ships
              │
              ▼
         Q1 (fill-quality audit)
              │
              ▼
         Pause for stocktake — review accumulated changes
              │
              ▼
         Discovery track opens (D1 / D2 / D3)
```

Each node in the tree has its own plan-iterate-design cycle. User approves transition at each node.

---

## What this roadmap is NOT

- Not a commitment to implement all phases within any timeframe
- Not a pre-registration (no hypothesis, no scan, no trial budget)
- Not an assertion that the ordering is the only valid one — user may re-order at any transition point based on new evidence
- Not a substitute for each phase's own Stage-1 scope doc

## Acceptable alternative orderings (for user awareness)

If the user prefers a different sequence, these orderings are also defensible:

**Alt-1 — "User-decision first":** C1 (copies scaling criterion) + A2a first, then main-line. Rationale: C1 is the biggest aggregate EV and non-technical.

**Alt-2 — "Theory first":** A2b-2 (DSR) before A2b-1 (regime gate). Rationale: DSR upgrade is bigger EV than bug fix. Caveat: uses wrong-regime inputs.

**Alt-3 — "Paper-grade first":** all L1-L6 literature expansion completed BEFORE any sub-audit. Rationale: lock in maximum future-axis optionality. Caveat: slow to ship any EV.

**Alt-4 — "Discovery parallel":** D1 (MGC discovery) runs parallel to A2b-1/2/3. Rationale: independent workstream, may surface new lanes. Caveat: execution capacity split.

The ordering in this roadmap is **bug-fix-first + EV-graduated** (Iter 6 locked). User may pick a different ordering at any decision point — this doc provides the dependency graph to evaluate tradeoffs honestly.

---

## Commit log entry (what this roadmap gates)

- Commit `f501ca7c` — A2b Stage-1 scope doc
- Commit `79ab4c47` — VWAP A+ hardening (canonical filter delegation)
- Commit `fc909741` — VWAP comprehensive scan DOCTRINE-CLOSED
- Commit `495810f5` — VWAP comprehensive pre-reg

Future commits are Phase-gated — each phase commits its Stage-1 scope doc, Stage-2 implementation, and Stage-3 verification independently.

---

## Kill criteria for THIS roadmap

- If any user-decision tree node is taken in a direction the roadmap does not anticipate, the roadmap is **superseded**, not rewritten retroactively. Pin the supersession date.
- If literature expansion (L1-L6) produces a finding that changes axis priorities, the Top-3 in the A2b scope doc is re-audited.
- If a phase's Stage-1 design identifies a dependency I missed, the dependency graph is updated with an amendment block.

---

## Ownership / responsibility (for future reader)

- **User:** final decision at each phase transition; copies scaling (C1); provenance of proving-loop criterion
- **Claude (or next agent):** per-phase plan-iterate-design cycle; implementation; verification; commit discipline
- **Canonical sources:** never modified without explicit design-gate + user approval
- **Mode A boundary:** sacred; no phase touches 2026-sacred data for selection
