# Powered-OOS reform — LANDED (reconciled against live git 2026-06-04)

**Status: LANDED on `main` / `origin/main`.** This plan formerly instructed a
fresh-worktree cherry-pick of `bcfc581c` to land the powered-OOS helper. That work
is **already done** — superseded by a Codex adversarial review (`needs-attention`,
2026-06-04) that caught this doc as stale-against-live-git. Do **NOT** cherry-pick;
do **NOT** create a `land-powered-oos` worktree. There is nothing to land.

---

## Evidence (live git, 2026-06-04)

- **Command:** `git pull --ff-only origin main` → `Updating 4badd61c..de1f9089` (fast-forward).
- **Landing commit:** `de1f9089` — "Powered-OOS reform Stage 1: powered_oos_split helper (no-wait OOS)".
  Confirmed ancestor of HEAD (`git merge-base --is-ancestor de1f9089 HEAD` → true).
- **Files verified present on `main`** (`git show --name-only de1f9089`):
  - `research/oos_holdout.py` (340 L)
  - `tests/test_research/test_oos_holdout.py` (170 L)
  - `docs/runtime/stages/2026-05-31-powered-oos-no-wait-reform.md`
- **Content identity** (`git rev-parse main:<f>` vs the rescue branch): the two code
  files are **byte-identical** to the old branch's version
  (`research/oos_holdout.py` blob `4e0ad69c…`, test blob `3cbf8265…`). The land
  carries the exact intended content — not a divergent re-land.
- **Local/main sync:** `main...origin/main` → `0/0` (was `behind 1` before the FF).

## What happened to the old branch

- `session/joshd-powered-oos-reform` no longer exists. The remaining remote ref is
  `origin/rescue/2026-06-03/session-joshd-powered-oos-reform` at `bcfc581c`.
- `bcfc581c` is **NOT** contained in `main` (the content landed via the re-commit
  `de1f9089`, not by merging `bcfc581c`). The rescue branch is therefore a
  **superseded duplicate** — its file content is already on main. Safe to delete
  when convenient; no action required to preserve the work.

## Dormancy note (honest, not a blocker)

`powered_oos_split` is defined at `research/oos_holdout.py:131` and is currently
exercised **only by its test** — `git grep powered_oos_split` shows the def + the
test, no production caller. This is the intended **land-as-library** posture for a
Stage-1 helper (test-covered, awaiting a downstream caller). If a future strand
needs no-wait OOS splitting, wire it there. Not dead code; not yet wired.

## Next strand — UNVERIFIED, do not act on this doc's old claim

The previous version of this file named `session/joshd-stale-work-radar`
(`774f856b`) as the "second stranded strand." **That branch and SHA do NOT exist
in live git** (no local ref, no remote ref, SHA unknown to the repo). Do not treat
it as real work. The genuine next strand must be re-derived from live git in a
fresh session (`git branch -a --verbose`, cherry-status by FILES not branch names,
per `feedback_stranded_work_audit_local_only_is_the_risk_not_merge_state`), not
inherited from this doc.

**Status of "what's next": BLOCKED_NEEDS_VERIFICATION** — no verified unlanded
strand identified this session. Powered-OOS is closed.
