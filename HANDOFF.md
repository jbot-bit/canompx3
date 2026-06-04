# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-06-05
- **Summary:** C11 deploy-unblock autonomous pass: Stage 2/3 self-funded de-leak + bool-guard parity confirmed already on origin/main (9429c540, zero divergence — no loss risk). Restored missing handoff-compaction plan from git history (a1ac60c8) and documented the post-commit accretion root cause. Compacted this baton. Firm/tier economics research (Stage 3) + conditional profile build (Stage 4) follow.

## Next Steps — Active
1. Stage 3 (Tier A, read-only): firm/tier net-return economics — does 100k/150k beat 50k at 1 micro/lane? Reconcile prop_profiles.py:16 vintage vs 2026-06-05 memory by fresh execution. Web-verify bot/payout policy (Firecrawl NOT connected — using WebFetch/WebSearch).
2. Stage 4 (Tier B, conditional): if a bigger tier wins on NET return, build that profile on an isolated branch off main, prove account_survival gate clears, adversarial-audit. STOP at designed+proven. No arming.
3. C11 cap_x0.75 fork still the live-clearance path for topstep_50k_mnq_auto (max90dDD 1535 <= 1600); gated behind OPEN bracket-parity audit 9b3fc530.

## Blockers / Warnings
- 9b3fc530 bracket-parity adversarial-audit gate is OPEN — do not close. No profile arming, no --live/--demo flip without explicit operator GO.
- main and origin/main are IN SYNC at 9429c540; do NOT reconcile/merge autonomously (Tier B default-branch decision).

## Durable References
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/handoffs/archived/2026-06-05-root-handoff-archive.md`
