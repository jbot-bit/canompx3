# Per-Worktree venv Isolation — peers can't strip each other's env

**Load-policy:** auto-injected when editing `scripts/tools/new_session.sh`,
`START_WORKTREE.bat`, or `.githooks/pre-commit`. Read on demand when reasoning
about venv resolution across worktrees.

**Authority:** 2026-06-03 hardening (Stage 1) after a recurring failure: the
single `canompx3/.venv` was shared by ~13 git worktrees + every peer Codex/Claude
session + the live bot. A peer's bare `uv sync` reconciles the shared env to
**base-only** and strips the `dev` group (pytest/ruff/httpx), silently breaking
the toolchain for every other worktree. The same sharing made ~24 processes
contend the venv's DLLs, so a clean `uv sync` hit `Access is denied` on locked
`.pyd`/`.dll` files. Both classes trace to one root: **one venv, many writers.**

---

## The rule

**Each git worktree owns its own `.venv`.** A peer's `uv sync` in another
worktree can never touch this tree's environment.

### Why this works (grounded in official uv docs)

uv resolves `UV_PROJECT_ENVIRONMENT` (default `.venv`) **relative to the
workspace root**, and *"if the environment is not found, uv will create it."*
A git worktree IS its own workspace root (own checkout, own `pyproject.toml`).
So running `uv sync` **from inside the worktree** with `UV_PROJECT_ENVIRONMENT`
**unset** creates an isolated `.venv` in that worktree — no flag gymnastics.

### The footgun the docs name explicitly — DO NOT do this

> *"If an absolute path is used across multiple projects, the environment will
> be overwritten by each project."* — uv docs, `concepts/projects/config.md`

That overwrite IS the shared-venv bug. **Never point
`UV_PROJECT_ENVIRONMENT` at an absolute path** for normal worktree work. The
bootstrap scripts explicitly `unset` it before syncing so no inherited absolute
path leaks in.

### WSL exception

The WSL/Linux side uses a *relative* `UV_PROJECT_ENVIRONMENT=.venv-wsl` (Windows
`.pyd` and Linux `.so` wheels are incompatible, so the two OSes need separate
venvs). Relative is fine — it still resolves per-worktree. This is the one place
an explicit override is correct.

---

## How it's wired

- **`scripts/tools/new_session.sh`** (bash) and **`START_WORKTREE.bat`**
  (Windows) run `uv sync --locked --group dev` from inside the new worktree
  (UV_PROJECT_ENVIRONMENT unset) right after `git worktree add`. On failure they
  WARN and fall back to the shared venv — never silently corrupt.
- **`.githooks/pre-commit`** probes the worktree-local `.venv` **first**
  (isolated case), then the canonical sibling venv (legacy shared-venv worktrees
  that predate this rule), then PATH `python`. Additive — no behavior change for
  existing worktrees.

## Migration posture (opt-in, not big-bang)

Existing worktrees keep sharing the canonical venv until they opt in by
re-running the bootstrap. The main checkout's `.venv` is untouched (zero risk to
the live bot's environment). New worktrees get isolation automatically. Cost:
each isolated venv is a few hundred MB of disk — paid only by worktrees that
opt in.

---

## Related

- `.claude/rules/parallel-session-isolation.md` — the worktree pattern itself
  (one Claude per tree); this rule isolates that pattern's *environment*.
- `.githooks/pre-commit` § "Quick venv health" — the dev-dep probe.
- uv docs: `concepts/projects/config.md` § "Project environment path".
