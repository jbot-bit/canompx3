# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-06-05
- **Summary:** Rebased 2 stranded local commits (HANDOFF compaction + C11 firm/tier economics research) onto moved origin/main and integrated to main at `5e4712b3`. Originals had diverged (2 ahead / 5 behind) after origin advanced through PR #359; backed them up to `origin/session/joshd-wt-06Thu04-20261811` FIRST, then cherry-picked onto fresh origin/main on branch `session/joshd-handoff-c11-research-rebase`. Resolved the lone HANDOFF.md conflict by re-archiving origin's CURRENT 194-line baton (preserving 2 newer Codex batons + corrected SHA ee23c3fe) while keeping the compact baton. FF'd main via CAS `git update-ref` (NOT checkout) because the main checkout is a DIRTY PEER tree. So Stage 3 firm/tier economics is now LANDED, not pending.

## Next Steps — Active
1. Stage 4 (Tier B, conditional): if a bigger tier wins on NET return per the now-landed firm/tier economics docs (`docs/audit/results/2026-06-05-c11-firm-tier-economics.md` + `...scaling-axes-and-wiring-coherence.md`), build that profile on an isolated branch off main, prove account_survival clears, adversarial-audit. STOP at designed+proven. No arming. CAVEAT (memory): $150k earns SAME as $100k at hardcoded 1 micro/lane — bigger account buys DD headroom, not earnings; verify before recommending a buy.
2. C11 cap_x0.75 fork remains the live-clearance path for topstep_50k_mnq_auto (max90dDD 1535 <= 1600); gated behind OPEN bracket-parity audit 9b3fc530. Edits prop_profiles.py (peer-owned) — coordinate, don't collide.
3. Tier-B doc-only: DSR drift-cache stale-PASS audit (stage `docs/runtime/stages/dsr-cache-stale-pass-audit.md`) — PLAUSIBLE not PROVEN; verify by EXECUTION, do not patch/merge without approval.

## Blockers / Warnings
- 9b3fc530 bracket-parity adversarial-audit gate is OPEN — do not close. No profile arming, no --live/--demo flip without explicit operator GO.
- main and origin/main are IN SYNC at `5e4712b3` (this session's integration); do NOT reconcile/merge autonomously (Tier B default-branch decision).
- The main checkout `C:/Users/joshd/canompx3` holds UNCOMMITTED peer C11 work (staged throttle-doc deletions + untracked `docs/plans/2026-06-05-c11-stage1-stage2-design.md`); its HEAD was moved to `5e4712b3` by a ref-only FF so its tree is intact but reads ahead/behind. Do NOT touch — let its owning session reconcile.

## Durable References
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/handoffs/archived/2026-06-05-root-handoff-archive.md`
