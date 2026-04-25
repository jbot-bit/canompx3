---
name: capital-review
description: >
  Capital-at-risk review orchestration. Use when a review touches live trading,
  deployment readiness, research validity, security boundaries, CI/supply chain,
  or when the user asks for a thorough anti-bias review before real capital is
  affected.
effort: high
---

# Capital Review

This is the Claude-facing counterpart to Codex's `canompx3-capital-review`.
It does not replace `/code-review`, `/audit`, `/verify`, or the agents; it
routes them so broad review requests do not collapse into a narrow diff scan.

## External Anchors

Use these as review anchors, not claims of formal compliance:

- NIST SSDF for secure development and recurrence prevention
- OWASP Code Review Guide / ASVS for manual security review and control coverage
- CISA Secure by Design for ownership, secure defaults, and transparency
- Fed SR 11-7 for model validation, limitations, governance, and monitoring
- SEC Rule 15c3-5 / FINRA algorithmic trading guidance for pre-trade controls,
  validation, supervision, and post-trade reporting
- OpenSSF Scorecard for supply-chain and repository security posture checks

## Route Stack

Classify the target before reviewing. Apply every material route:

| Target | Required route |
| --- | --- |
| Normal code diff | `/code-review` plus `/verify` |
| Runtime/live/broker/session/profile/webhook path | `/audit phase 7` plus live-control checks |
| Deployment, promotion, scaling, account routing, or go/no-go decision | deploy-readiness audit plus `evidence-auditor` |
| Research result, strategy claim, validation claim, or result doc | `/research` plus `evidence-auditor` |
| External input, auth, credentials, subprocess, filesystem path, network, broker API, or order route | security best-practice review |
| New or changed trust boundary | threat model before code-review conclusions |
| CI, workflow, dependency, secret, or release surface | supply-chain/security review |

## Review Rules

- Treat prior summaries, handoffs, docs, and agent claims as claims, not proof.
- Lead with disconfirming checks: how could this be false or unsafe?
- Label every conclusion `MEASURED`, `INFERRED`, or `UNSUPPORTED`.
- High and critical findings require `PREMISE -> TRACE -> EVIDENCE -> VERDICT`.
- Cite files and lines. Do not report a behavior bug without tracing callers.
- Review capital impact explicitly: no impact, research-only, shadow-only,
  deploy-readiness, live-order risk, account-risk, or kill-switch risk.
- Separate validation, deployment, and execution. `validated_setups` is not the
  live book, and `paper_trades` is execution evidence only.
- A clean review still reports checked scope, unchecked scope, and residual risk.

## Output

```text
CAPITAL REVIEW
Scope:
Route stack:
Decision: BLOCK | FIX_REQUIRED | VERIFY_MORE | ACCEPT_WITH_RISK | CLEAR

Findings:
- [Severity] [MEASURED|INFERRED|UNSUPPORTED] [capital impact]
  File:
  Premise:
  Trace:
  Evidence:
  Verdict:
  Fix:

Disconfirming checks:
- ...

Verification:
- Commands run:
- Commands not run:

Residual risk:
- ...
```

`CLEAR` is allowed only with targeted verification or an explicit non-test
rationale. If live capital can be affected and kill/flatten/risk-limit behavior
was not exercised or traced, the highest possible decision is `VERIFY_MORE`.
