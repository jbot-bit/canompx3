---
paths:
  - "trading_app/live/**"
  - "trading_app/risk/**"
  - "pipeline/**"
---

# Adversarial-Audit Gate — After Every CRIT/HIGH Fix

**Non-negotiable.** After any judgment-classified commit that touches a truth-layer or live-trading path, dispatch an adversarial audit before the next phase commits.

This rule formalizes `institutional-rigor.md` § 2 ("After any fix, review the fix"). Single-agent implementation and self-review are necessary but not sufficient. The implementer's mental model is the same one that produced the bug; an independent context is required to surface what the implementer's tests did not probe.

## Proof case

Iteration 174 (2026-04-25) landed the F4 bracket-submit fix with four mutation-proof tests, all passing. The commit cited `institutional-rigor.md` § 6 and `integrity-guardian.md` § 3. Drift was clean. Ralph's own verification said the fix was institutionally sound.

An independent-context evidence-auditor pass (iteration 178 precedent) surfaced a CRITICAL race: the kill-switch guard lived at the top of the bar handler but not inside the per-event dispatcher. The F4 fix itself could create a new naked broker position on event N plus one in the same bar while flattening event N's position. The fix mechanism created the very failure mode it was supposed to prevent.

The implementer's tests never sent a bar with two entry events because the implementer reasoned about the single-event case. The auditor, reading the fix adversarially and tracing the event loop, caught the gap in minutes.

Without an audit gate, the race would have sat in production until it triggered in live trading.

## Gate trigger

Dispatch the audit when a commit meets ALL of the following:

- Classification is `[judgment]` (not `[mechanical]` and not ledger-only).
- The commit touches at least one file in `trading_app/live/`, `trading_app/risk/`, or `pipeline/` (truth-layer).
- The commit's severity tag is CRITICAL or HIGH, OR the commit closes a finding in `docs/ralph-loop/deferred-findings.md`.

Mechanical commits, ledger updates, test-only refactors, and doc-only commits are exempt.

## Actor

Dispatch the `evidence-auditor` subagent (independent context, separate conversation, treats summaries as claims requiring proof or disproof). Never use the same agent that produced the fix; the point is independent reasoning.

## Artifact required

The audit returns a structured report with these fields, all present:

- Verdict: PASS, CONDITIONAL, or FAIL.
- Per-commit verdict if multiple commits under review.
- Critical issues (each with file-and-line citation, claim, evidence, impact).
- Silent gaps (untested behavior, unguarded paths).
- Unsupported assumptions (commit-message or code claims without grounding).
- Tests missing.
- Do-not-touch (anything audit-verified correct that must not be refactored).
- Highest-priority fix (single recommendation, no implementation).

If the verdict is CONDITIONAL or FAIL, the plan-supervisor routes the critical findings into the next iteration. The next phase does NOT dispatch until the audit findings are either closed or explicitly deferred with written justification.

## Scope

The audit reviews:

- The commit itself.
- Any commit it depends on that has not yet been audited.
- Cross-fix interactions if multiple truth-layer commits landed without audit between them.

The audit does NOT review unrelated prior commits; the gate is forward-looking, not retrospective.

## What this rule forbids

- Dispatching the next phase fix iteration before the audit of the previous fix returns.
- Using ralph's own self-review as the audit artifact. Self-review is part of the fix iteration; audit is separate.
- Skipping the audit because the fix "looks obvious" or the tests "pass clearly". The C1 finding is the canonical counterexample.
- Quiet deferral of audit findings. Deferred findings go to `docs/ralph-loop/deferred-findings.md` with written justification; they do not fall through.

## Exemption path

If time pressure requires landing a fix without audit (rare), the commit MUST:

- Include `AUDIT-SKIPPED:` in the commit message body with the reason.
- Create an entry in `docs/ralph-loop/deferred-findings.md` scheduling the audit for a named later iteration.
- Never occur for a CRITICAL severity commit.

## Enforcement

The rule is process-level and enforced by the plan-supervisor loop. The plan file in `docs/plans/` tracks audit verdicts per iteration. A drift check that inspects commit metadata for the audit-skipped marker may be added later; not required for initial gate adoption.
