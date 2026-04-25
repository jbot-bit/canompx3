---
name: canompx3-capital-review
description: Capital-at-risk review orchestration for canompx3. Use when the user asks for a thorough code review, anti-bias review, real-capital safety review, deploy/readiness scrutiny, or broad review of changes where trading capital, research validity, security, or runtime controls may be affected.
---

# Canompx3 Capital Review

Use this skill when a normal review may be too narrow for canompx3's risk
surface. This is an orchestration layer: it wraps the official OpenAI security
skills and repo-local audit skills without replacing the Claude authority layer.

## External Anchors

Use these as review anchors, not as claims of formal compliance:

- NIST SSDF: secure development and recurrence prevention
- OWASP Code Review Guide / ASVS: manual review and security-control coverage
- CISA Secure by Design: secure defaults, ownership, and transparency
- Fed SR 11-7: model validation, limitations, governance, and monitoring
- SEC Rule 15c3-5 / FINRA algorithmic trading guidance: pre-trade controls,
  validation, supervision, and post-trade reporting
- OpenSSF Scorecard: supply-chain and repository security posture checks

If a threshold, regulatory claim, or methodology requirement is not grounded in
repo-local doctrine or a primary source, label it `UNSUPPORTED`.

## Route Stack

Start by classifying the review target. Apply every route that materially fits,
but do not over-escalate low-risk docs-only changes.

| Target | Required route |
| --- | --- |
| Normal code diff | `.claude/skills/code-review/SKILL.md` plus `canompx3-verify` |
| Runtime/live/broker/session/profile/webhook path | `canompx3-live-audit` plus security review |
| Deployment, promotion, scaling, account routing, or go/no-go decision | `canompx3-deploy-readiness` plus evidence-auditor |
| Research result, strategy claim, validation claim, or result doc | `canompx3-research` plus evidence-auditor |
| External input, auth, credentials, subprocess, filesystem path, network, broker API, or order route | `security-best-practices` |
| New or changed trust boundary | `security-threat-model` before code-review conclusions |
| CI, GitHub workflow, dependency, secret, or release surface | security review plus OpenSSF-style supply-chain checklist |

## Review Rules

- Treat prior summaries, handoffs, docs, and agent claims as claims, not proof.
- Lead with disconfirming checks: how could this be false or unsafe?
- Every conclusion must be labeled `MEASURED`, `INFERRED`, or `UNSUPPORTED`.
- Every high or critical finding must include:
  `PREMISE -> TRACE -> EVIDENCE -> VERDICT`.
- Findings need concrete file and line references. Do not report a bug without a
  traced execution path when the behavior depends on callers.
- Review capital impact explicitly: no impact, research-only, shadow-only,
  deploy-readiness, live-order risk, account-risk, or kill-switch risk.
- Separate validation, deployment, and execution. `validated_setups` is not the
  live book, and `paper_trades` is execution evidence only.
- A clean review must still say what was checked, what was not checked, and what
  residual risk remains.

## Output Format

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

`CLEAR` is allowed only when targeted tests or explicit non-test rationale are
reported. If live capital can be affected and kill/flatten/risk-limit behavior
was not exercised or traced, the highest possible decision is `VERIFY_MORE`.
