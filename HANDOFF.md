# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-06-06
- **Summary:** Bloomberg-style live-capital code-review continuation after `git fetch --all --prune --tags`; current `work` HEAD is `669e036` (`Webhook live-mode risk ack/profile + project pending contracts in F-1 scaling (with tests)`) and this checkout still has no remotes/upstream tracking info, so there was nothing to merge/rebase locally. Re-read the latest live-webhook/F-1 diff, production `can_enter()` callers, profile/ack references, and webhook execution route. Verdict: the pending-contract F-1 fix and live-webhook ack/profile gates remain relevant; no additional proven high/critical code defect was found in the reviewed diff. Small alignment fixes made this session: documented `WEBHOOK_PROFILE_ID` in the webhook launcher env list and corrected the stale `RiskManager.can_enter()` usage docstring. Deployment/live readiness remains BLOCKED in this WSL checkout because `gold.db` is absent and the repo pre-commit hook is not active.

## Next Steps — Active
1. Stage 4 (Tier B, conditional): if a bigger tier wins on NET return per the now-landed firm/tier economics docs (`docs/audit/results/2026-06-05-c11-firm-tier-economics.md` + `...scaling-axes-and-wiring-coherence.md`), build that profile on an isolated branch off main, prove account_survival clears, adversarial-audit. STOP at designed+proven. No arming. CAVEAT (memory): $150k earns SAME as $100k at hardcoded 1 micro/lane — bigger account buys DD headroom, not earnings; verify before recommending a buy.
2. C11 cap_x0.75 fork remains the live-clearance path for topstep_50k_mnq_auto (max90dDD 1535 <= 1600); gated behind OPEN bracket-parity audit 9b3fc530. Edits prop_profiles.py (peer-owned) — coordinate, don't collide.
3. Tier-B doc-only: DSR drift-cache stale-PASS audit (stage `docs/runtime/stages/dsr-cache-stale-pass-audit.md`) — PLAUSIBLE not PROVEN; verify by EXECUTION, do not patch/merge without approval.

## Blockers / Warnings
- 9b3fc530 bracket-parity adversarial-audit gate is OPEN — do not close. No profile arming, no --live/--demo flip without explicit operator GO.
- main and origin/main were previously reported in sync at `5e4712b3`; current `work` branch fetched cleanly and is at `669e036` before this baton-refresh commit. This checkout has no remote-tracking branch info in `git branch -vv`; do NOT reconcile/merge autonomously (Tier B default-branch decision).
- Local WSL audit environment lacks `gold.db`; live preflight and opportunity/data freshness checks therefore fail with DuckDB read-only open errors. Do not claim live readiness from this environment.
- The main checkout `C:/Users/joshd/canompx3` holds UNCOMMITTED peer C11 work (staged throttle-doc deletions + untracked `docs/plans/2026-06-05-c11-stage1-stage2-design.md`); its HEAD was moved to `5e4712b3` by a ref-only FF so its tree is intact but reads ahead/behind. Do NOT touch — let its owning session reconcile.

## Durable References
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/handoffs/archived/2026-06-05-root-handoff-archive.md`
