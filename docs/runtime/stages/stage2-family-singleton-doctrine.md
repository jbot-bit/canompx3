---
task: Decide doctrine for family_singleton — hard blocker, conditional warning with individual-evidence floor, or context-dependent disposition. No code changes until user decision is locked.
mode: DESIGN
scope_lock:
  - docs/runtime/stages/stage2-family-singleton-doctrine.md
  - docs/audit/results/2026-05-11-family-singleton-doctrine-analysis.md
blast_radius:
  Reads gold.db read-only; reads docs/institutional/literature/; reads
  trading_app/deployability.py, trading_app/edge_families.py, trading_app/strategy_fitness.py
  for current semantics. Writes only the two files in scope_lock (a stage doc + an
  analysis doc). No production code changes. No DB writes. No lane_allocation.json
  changes. No broker / live-state impact. Stage 2 produces a doctrine recommendation;
  any code change derived from the user's decision is a separate stage (stage2-impl-*).
---

## Purpose

Stage 1 (PR #258) generalised the routine-TBBO slippage inference gate so MES rows
that share a session with a shipped pilot can clear `slippage_missing`. Empirically
that change flips 2 MES rows from `BLOCKED_FAMILY_FRAGILE [family_singleton,
slippage_missing]` to `BLOCKED_FAMILY_FRAGILE [family_singleton]` — i.e. their **only
remaining hard block is `family_singleton`**. The other 3 closest-to-deployable MES
rows are blocked by `[family_singleton, slippage_missing]` because their session
(US_DATA_1000) is not in the MES pilot v1 scope.

The question Stage 2 must answer is **doctrine, not code**:

> Given a strategy that individually clears every other deployability gate
> (replay-clean, current K-FDR pass, C8 OOS pass, slippage-PASS, account-risk-pass,
> WFE ≥ 0.50, N ≥ 100, dir-match, era-stable), is the absence of peer-cross-check
> evidence (the SINGLETON classification) sufficient grounds for a hard deployment
> block, or is it sufficient grounds for a warning + sizing penalty, or is it
> redundant with the individual-evidence gates that already passed?

Stage 2 produces a literature-grounded recommendation; the user picks the
disposition; the code change to enact it (if any) is a separate stage.

## Stage 1 sufficiency check

PR #258 was performed on **medium effort**; this Stage 2 doc verifies Stage 1 is
nonetheless sufficient before building on it.

- CI: GREEN. `gh pr view 258 --json mergeStateStatus` = CLEAN.
- Adversarial audit: an independent-context `evidence-auditor` pass surfaced 2
  findings the implementer missed; both closed in commit `c97f9e58`. Self-review
  of that fix surfaced a 3rd architectural issue (set-difference loop over-coverage
  blind spot); closed in `3b648c1d` by inverting to a symmetric registry-vs-evidence
  loop that asserts coverage in both directions.
- Drift: `python pipeline/check_drift.py` passes 124/124 on each commit, including
  the new `check_routine_tbbo_slippage_registry_coverage`.
- Tests: 42 tests in scope; 200 fast tests on pre-commit gauntlet; 4551 broader
  tests passing (1 pre-existing WSL-doctor failure unrelated).
- Mutation matrix on `_slippage_is_controlled_event_tail_pending`: 11/11 OK.
- Chronology matrix on drift check: 4/4 valid mutations OK.
- Disconfirming checks: capital-safety grep (zero hits on `deployable`/verdict
  string mutating capital state outside deployability_state.py), import-failure
  fail-closure probe (registers as 1 violation, not silent pass).

Verdict: Stage 1 is sufficient. The audit-trail discipline + independent-auditor
pass + symmetric-loop refactor exceeded what "medium effort" usually produces.
The remaining risk is doctrine, not implementation, which is exactly what
Stage 2 is meant to resolve.

## What is and is not in scope for Stage 2

In scope:
- Reading the literature anchors for individual-strategy multiple-testing penalty
  (Harvey-Liu 2015 Sharpe haircut, Bailey-LdP 2014 DSR, López de Prado-Bailey 2018
  False Strategy Theorem, Bailey et al 2013 MinBTL) and the family-cross-check
  framing (Carver 2015 Ch 4 member consistency, Bailey et al 2014 PBO §4.2).
- Empirically counting which MES rows would be unlocked under each disposition
  and what their individual evidence quality looks like (DSR, WFE, N, OOS, era
  stability).
- Producing a written disposition recommendation in
  `docs/audit/results/2026-05-11-family-singleton-doctrine-analysis.md`.
- Listing the dispositions the user can pick from (A, B, C, ...).
- Identifying the residual doctrine drift uncovered en route (e.g. DSR=0 universal
  on currently-deployed MNQ lanes — Criterion 5 is not being enforced in practice).

Out of scope:
- Code changes to `trading_app/deployability.py`, `trading_app/edge_families.py`,
  `pre_registered_criteria.md`, or any other production file.
- Allocator changes; `lane_allocation.json` is untouched.
- Per-strategy chordia-unlock-style audits on the 5 candidate MES rows
  (separate Stage 3 if Stage 2 unlocks them).
- DSR re-computation infrastructure (separate workstream; flagged as drift but
  not Stage 2's fix).
- Replay backfill for the 35 MES rows blocked by `replay_mismatch` (separate stage).

## Done criteria

1. `docs/audit/results/2026-05-11-family-singleton-doctrine-analysis.md` exists
   with: (a) the literature anchor section, (b) per-row empirical evidence on the 5
   currently-blocked-only-by-family_singleton MES rows, (c) the doctrine question
   stated as a falsifiable choice, (d) 4–5 explicit dispositions with their pros,
   cons, and capital-impact, (e) the DSR=0 drift finding flagged with severity
   and a recommended follow-up, (f) NO recommendation between dispositions — the
   user picks. **[DONE]**
2. Stage doc parser checks pass (this file). **[DONE]**
3. User locks a disposition; that locks Stage 3 scope (or closes Stage 2 with
   "no code change, keep status quo"). **[DONE — see § Disposition Locked]**
4. No production files modified; `git status` in the worktree shows only the two
   files in scope_lock. **[DONE]**

## Disposition Locked — 2026-05-11

**User picked: Disposition C — Conditional downgrade with individual-evidence floor.**

### Floor — literature audit (post-pick honest correction)

The original Disposition C in the analysis doc listed five floor values
(DSR≥0.95, WFE≥0.70, N≥300, years≥6, avg_shann≥0.8). User asked whether
those values are correct per `resources/` literature. Honest audit against
`docs/institutional/literature/`:

| Floor value | Status vs literature |
|---|---|
| `dsr_score >= 0.95` | **GROUNDED** — `bailey_lopez_de_prado_2014_deflated_sharpe.md` p.60 numerical example explicitly cites 0.95 as the 95%-confidence rejection threshold. Locked as Criterion 5 in `pre_registered_criteria.md`. |
| `wfe >= 0.70` | **NOT GROUNDED** — WFE is a project convention not derived from any of the read literature extracts. `pre_registered_criteria.md` C6 locks 0.50 as binding. The 0.70 was a "stricter than baseline" speculation by Claude, not a literature-anchored threshold. |
| `sample_size >= 300` | **NOT GROUNDED** — conflates Harvey-Liu Exhibit 4 "300 tests" (number of strategies in the multiple-testing pool) with N per strategy. They are different quantities. C7's N≥100 floor is the locked project value; raising it to 300 has no literature basis read so far. |
| `years_tested >= 6` | **NOT GROUNDED** — project convention. Bailey 2013 MinBTL talks about backtest length as a function of N_trials, not as a fixed-year minimum. |
| `avg_shann >= 0.8` | **GROUNDED** — PBO §4.2 (Bailey et al 2014) per `edge_families.py:19-22`. Already required for SINGLETON classification. |

Two of the five proposed floor values were not literature-anchored. Stage 2
corrects this: the actual disposition-C floor enforced by Stage 4 will be
**ALL 12 locked criteria in `pre_registered_criteria.md`** rather than
extra-strict ad-hoc thresholds. The rule becomes:

### Rule (corrected, to be enacted by Stage 4)

`family_singleton` remains a hard blocker UNLESS the strategy passes ALL
twelve criteria in `pre_registered_criteria.md` (C1 pre-reg, C2 MinBTL,
C3 BH-FDR, C4 t-statistic, C5 DSR ≥ 0.95, C6 WFE ≥ 0.50, C7 N ≥ 100,
C8 OOS ExpR ≥ 0.40 × IS, C9 era stability, C10 data-era compatibility,
C11 account-death MC ≥ 70% survival, C12 SR monitor active) AND the
SINGLETON classifier's own thresholds (member_count == 1 AND
min_trades ≥ 100 AND avg_shann ≥ 0.8). No floor values invented by
Claude; only thresholds already locked in the doctrine pre-reg file.

Empirical block: with current DSR drift (all 847 active rows have DSR < 0.95),
this disposition unlocks **0 rows** until DSR is recomputed. Therefore the
work sequence is:

### Required ordering (hard dependency chain) — THIRD-PASS CORRECTED

THIRD-PASS CORRECTION 2026-05-11: § "Stage 3 — DSR / Criterion 5 doctrine
resolution" below was framed against a problem that does not exist. C5 is
already officially CROSS-CHECK per `pre_registered_criteria.md` Amendment
2.1 (2026-04-07). The 3 deployed MNQ lanes are not in violation. The path
(a)/(b)/(c) decision was unnecessary. Stage 3 collapses to: "what does the
'passes locked criteria' floor for singletons actually require, given C5
is cross-check?" — a smaller scoped question.

The section below is RETAINED for audit-trail visibility but is superseded
by the THIRD-PASS analysis-doc correction. Stage 3 worktree
(`stage3/c5-doctrine-resolution`) carries the corrected scope.

### Original (incorrect) Stage 3 framing follows

**Stage 3 — DSR / Criterion 5 doctrine resolution** (blocking precondition):

Honest reframe per the post-pick self-audit (see analysis doc § 3.5):
DSR ≈ 0 across the shelf is NOT drift in the bug sense — the formula is
correctly reporting that strategies discovered via large brute-force K
(`validated_setups.discovery_k` ranges 8856–9568 on MES candidates;
comparable on deployed MNQ lanes) cannot clear DSR ≥ 0.95 at our data
horizon. Bailey 2013 MinBTL anticipated exactly this.

Stage 3 picks one of three paths:
- **(a) Grandfather pre-Phase-0 K:** add a deployability gate exception
  that recognises strategies promoted under VALIDATOR_NATIVE provenance
  with pre-Phase-0 K used a brute-force search predating the locked
  criteria. Treated as Mode B per `research-truth-protocol.md`.
  Effect: live 3-lane MNQ portfolio remains deployable; new candidates
  must use pre-reg-family K.
- **(b) Re-derive `discovery_k` field to use pre-reg-family K** (≤ 300
  per Bailey MinBTL). Recompute DSR shelf-wide. Some candidates would
  then clear C5. Doctrine question: is this honest, or post-hoc relaxation
  per `research-truth-protocol.md`?
- **(c) Genuine retirement** of any strategy that cannot clear C5 once
  recomputed under its real discovery K. Includes the 3 currently-deployed
  MNQ lanes. Most institutionally honest but largest capital impact.

This is NOT a Claude decision. User picks the path. New stage file:
`docs/runtime/stages/stage3-c5-doctrine-resolution.md`.
Truth-layer + capital-impact change; requires independent-context
`evidence-auditor` pass per `.claude/rules/adversarial-audit-gate.md`.

**Stage 3.5 — C8 OOS evidence backfill** (parallel blocking precondition):
All 5 candidate MES rows have `c8_oos_status IS NULL`. The C8 gate at
`deployability.py:565-566` hard-blocks them regardless of Stage 3's path.
Backfilling C8 means running the C8 OOS validator on each candidate, which
produces a `c8_oos_status` value of either `PASSED`, `FAILED_RATIO`,
`NEGATIVE_OOS_EXPR`, or `INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH`. Outcomes
unknown a priori — some candidates may fail C8 substantively.
New stage file: `docs/runtime/stages/stage3.5-mes-c8-backfill.md`.

**Stage 4 — family_singleton conditional-downgrade code change**:
- Replace the unconditional hard blocker at `deployability.py:539` with a
  conditional that delegates to a canonical "passes all locked criteria"
  check derived from `pre_registered_criteria.md` C1-C12. No new threshold
  constants — the check reads from the same canonical sources every other
  deployability gate reads from (the row's `dsr_score`, `wfe`,
  `sample_size`, `oos_exp_r` etc.).
- Add tests covering: (a) singleton clearing every locked criterion ->
  warning, (b) singleton failing ANY locked criterion -> hard, (c) chronology
  of DSR-based pass/fail flips as DSR is recomputed in Stage 3,
  (d) verdict-bucket transition (BLOCKED_FAMILY_FRAGILE retained for PURGED;
  new bucket or distinct verdict for floor-passing singletons).
- Add drift check ensuring the singleton-pass branch cannot silently degrade:
  any threshold the check inlines must carry a `@research-source` annotation
  pointing at `pre_registered_criteria.md`, with a drift check verifying the
  inlined value matches the canonical pre-reg file.
- New stage file: `docs/runtime/stages/stage4-family-singleton-conditional.md`.

**Stage 5 (optional, depends on Stage 3+4 outcomes)** — per-strategy
`/capital-review` on each newly-unlocked singleton before any allocator
expansion. The 5 MES candidate rows + any post-DSR-fix MNQ candidates
each get a Bailey-LdP / Harvey-Liu individual-evidence audit before
`lane_allocation.json` mutation. Cross-instrument correlation against
deployed MNQ lanes is part of this stage.

### What does NOT need to happen

- Stage 4 cannot be implemented before Stage 3. The user's disposition is
  contingent on DSR being a real gate. Implementing Stage 4 first would
  ship a no-op (no row clears the floor) and would have to be reworked
  once DSR lands.
- Stage 1 (PR #258) does not need re-validation — it is independent of the
  doctrine choice.

### Stage 2 close-out

Stage 2 has produced its decision artefact. The stage doc can be archived
to `docs/runtime/stages/archive/` once Stage 3 is open. The
`stage2/family-singleton-doctrine` branch carries the analysis doc + this
stage doc + the HANDOFF update; it can be merged as a docs-only PR or
folded into the Stage 3 branch. Recommended: merge Stage 2 as its own
docs-only PR so the doctrine pick is part of the durable record on `main`,
then branch Stage 3 from the post-merge tip.

## Why this is DESIGN mode

The decision affects deployability gate behaviour, which is on the truth-layer
adjacent path (`integrity-guardian.md` Rule 1 — authority hierarchy). A doctrine
shift requires user consent before code changes. Until the user picks a
disposition, the only artefact is the analysis doc. After the user picks, a
separate IMPLEMENTATION stage will carry the code change with its own
scope_lock, blast radius, and adversarial-audit gate per
`.claude/rules/adversarial-audit-gate.md`.
