# Worktree Launcher — `START_WORKTREE.bat`

The piece hooks cannot provide. A `SessionStart` hook fires *inside* an
already-running Claude session, so nothing in-process can create a worktree and
then re-launch Claude *into* it. `START_WORKTREE.bat` is the thin out-of-process
launcher that does exactly that — create/reuse an isolated git worktree and
start Claude there, refusing to launch into a hot (dirty or peer-held) one.

It complements `scripts/tools/new_session.sh` (which creates a worktree but does
not enter or launch Claude in it).

---

## Usage

Run it **from the main checkout** (`C:\Users\joshd\canompx3`):

```bat
START_WORKTREE.bat hwm-fix              :: create/reuse <repo>-hwm-fix, launch Claude there
START_WORKTREE.bat                      :: descriptor defaults to a timestamp
START_WORKTREE.bat --dry-run hwm-fix    :: classify + print the decision; launch NOTHING
set CANOMPX3_LAUNCHER_DRYRUN=1 & START_WORKTREE.bat hwm-fix   :: env-var dry-run
```

> **Run it from the main checkout, not a worktree.** The worktree path is
> computed as `<repo_parent>/<repo_name>-<descriptor>` relative to wherever you
> launch (mirrors `new_session.sh`). From the main checkout `canompx3` +
> descriptor `hwm-fix` → `canompx3-hwm-fix`. From a worktree you'd get a
> double-suffixed path. `git worktree add` is also rooted at the main checkout.

---

## What it does

1. Shells to the read-only classifier
   `scripts/tools/worktree_launch_preflight.py --descriptor <d>` and reads back
   `DECISION` / `WTPATH` / `BRANCH`.
2. Acts on the decision:

   | Decision | Meaning | Action |
   |---|---|---|
   | `NEW` | worktree path does not exist | `git fetch` + `git worktree add -b session/<user>-<d> <path> origin/main`, then launch |
   | `REUSE_CLEAN` | exists, clean, no live peer | launch into it |
   | `REFUSE_HOT` | exists AND (dirty OR a live peer holds the lease) | **refuse, exit 3** |

3. Launches via `start "<title>" /d "<WTPATH>" claude.exe` (mirrors `claude.bat`).

Exit codes: `0` ok/dry-run · `3` refused-hot · `4` `git worktree add` failed ·
`5` preflight produced no decision.

### Guards are NOT bypassed

Launching Claude in the worktree triggers `session-start.py` and the
`worktree_guard.py` PreToolUse hook normally. The launcher's `REFUSE_HOT` is a
*pre-launch* convenience (don't even start a doomed session); the in-session
lease guard remains the authoritative mutex.

---

## The classifier (`worktree_launch_preflight.py`)

Read-only; `main()` always returns 0 (classification is never an error).

- **dirty** = `git -C <wt> status --porcelain --untracked-files=no` non-empty
  (untracked ignored, matching `new_session.sh`'s collision concern). Fail-open
  to *not* dirty on git error.
- **lease-hot** delegates entirely to canonical
  `python scripts/tools/worktree_guard.py --status --json --cwd <wt>` and reads
  `lease_present AND peer_live` — **no inline lease logic** (institutional-rigor
  §4). Any subprocess/parse failure → *not* hot (fail-open); the independent
  dirty-check still guards.

So a worktree is `REFUSE_HOT` if **either** signal fires, and the two are
independent — a broken lease subprocess can never silently drop the dirty guard.

---

## CRLF requirement (`.gitattributes`)

`START_WORKTREE.bat` is pinned `text eol=crlf` in `.gitattributes`. `cmd.exe`
mis-parses LF-only batch files (it drops the first byte of each line —
`setlocal` → `etlocal`). The pin guarantees a fresh checkout / CI runner with
`core.autocrlf` cannot renormalize it to LF and corrupt the launcher. Mirrors
the existing CRLF blob of `claude.bat`.

---

## Related

- `scripts/tools/new_session.sh` — worktree creation (no launch); the path/branch
  convention this launcher mirrors.
- `.claude/hooks/worktree_guard.py` + `scripts/tools/worktree_guard.py` — the
  canonical lease module the classifier delegates to.
- `.claude/rules/parallel-session-isolation.md` — why two Claudes in one
  worktree must be prevented.
