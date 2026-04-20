---
stage: mnq-pr51-dsr-audit
mode: IMPLEMENTATION
task: "Compute Deflated Sharpe Ratio (Bailey-López de Prado 2014 Eq. 2) on PR #51's 5 CANDIDATE_READY cells. Phase 0 C5 gate (DSR >= 0.95) was never verified — H1/C6/C8/C9 passed but C5 is missing. Confirmatory audit, no new discovery, no pre-reg required per research-truth-protocol § 10."
updated: "2026-04-21"
scope_lock:
  - "research/mnq_pr51_dsr_audit_v1.py"
  - "docs/audit/results/2026-04-21-mnq-pr51-dsr-audit-v1.md"
  - "HANDOFF.md"
acceptance:
  - "Recomputes PR #51's 105-cell family pnl_r distributions exactly."
  - "Computes trade-level SR per cell + family V[SR]."
  - "Computes per-cell skewness, kurtosis, DSR per Eq. 2."
  - "DSR rejection threshold SR_0 reported."
  - "Per-cell PASS/FAIL at DSR >= 0.95."
  - "Sanity-verifies one cell against canonical Bailey worked example (pp 9-10)."
  - "Drift check passes."
---
