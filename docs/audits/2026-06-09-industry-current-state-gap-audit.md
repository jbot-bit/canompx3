# 2026-06-09 Industry Current-State Gap Audit

**Class:** result / current-state audit snapshot, not live truth.
**Current integration commit:** `9677f7b10c0c0051775d9c75693433a859b6825d`.
**Prior audit branch found:** `origin/codex/audit-project-for-improvements-and-standards` at `5ca7cd3c`.
**Scope:** merge the useful baseline security automation and implement the first concrete clean-checkout evidence fix.

## Evidence Boundary

- `git fetch origin` completed before this branch was created.
- Work was isolated on `codex/clean-checkout-db-evidence` from `origin/main`.
- The primary checkout was dirty and behind `origin/main`; it was not rebased or edited.
- This audit is not a live-readiness claim. DB-backed readiness still requires canonical `gold.db` evidence.

## Official-Source Checks

- GitHub Dependabot official supported-ecosystems documentation lists `github-actions` and `uv` as supported `package-ecosystem` values.
- GitHub CodeQL code-scanning documentation supports push, pull request, and scheduled analysis.
- GitHub SARIF upload documentation requires `security-events: write` for SARIF upload workflows, with `actions: read` and `contents: read` for private repositories.
- OpenSSF Scorecard action documentation allows Scorecard plus `actions/checkout`, `actions/upload-artifact`, and `github/codeql-action/upload-sarif` in the Scorecard job.

## Implemented In This Patch

| Area | Change | Evidence boundary |
|---|---|---|
| Dependency hygiene | Added `.github/dependabot.yml` for GitHub Actions and `uv`. | Opens review PRs only; does not change runtime dependencies. |
| SAST | Added `.github/workflows/codeql.yml` for Python CodeQL analysis. | CodeQL uploads findings to code scanning; first run still needs GitHub execution evidence. |
| Supply-chain posture | Added `.github/workflows/scorecard.yml` with SARIF upload. | Scorecard result is advisory evidence; first run still needs GitHub execution evidence. |
| Clean-checkout DB evidence | `scripts/tools/audit_integrity.py` now exits with explicit `NEED_DB` when canonical `gold.db` is unavailable before DB checks run. | Fail-closed: missing DB remains nonzero and cannot be read as live/readiness pass. |
| Audit orchestration | `scripts/audits/phase_1_automated.py` preserves the `NEED_DB` signal instead of reporting ambiguous zero integrity failures. | Clean checkout can distinguish “repo broken” from “DB-backed evidence unavailable.” |

## Remaining High-ROI Gaps

1. Add a deterministic sanitized fixture DB or approved exported snapshot lane for CI smoke tests.
2. Emit a signed live launch manifest from the live preflight path before any live arming claim.
3. Add a unified metrics surface for feed lag, order latency, errors, saturation, and kill state.
4. Harden webhook authentication with signed headers for non-TradingView callers.
5. Generate a current model/strategy inventory from canonical DB/read-model sources.

## Bottom Line

This patch does not make the system deployable and does not validate any strategy edge. It makes the previous audit/security-baseline work mergeable on top of current `origin/main` and closes the first misleading clean-checkout evidence gap: missing canonical DB is now an explicit `NEED_DB` condition, not an ambiguous audit failure count.
