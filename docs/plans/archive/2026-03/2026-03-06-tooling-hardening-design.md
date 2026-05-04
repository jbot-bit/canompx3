---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Tooling Hardening Design

**Date:** 2026-03-06
**Status:** Approved
**Scope:** Type checking, formatting, linting, dependency management, security scanning, coverage enforcement

## Problem

Production trading system (178 .py files, 1496 tests) has zero static type analysis, no formatter, minimal ruff rules, mixed dependency pinning, no lock file, no security scanning, and no coverage threshold enforcement.

## Design Decisions

### Type Checking: pyright basic (NOT mypy strict)

**Why pyright over mypy:**
- 3-5x faster, especially on Windows
- Better pandas/numpy/duckdb inference without external stubs
- Powers VS Code/Pylance — same engine for CI and editor
- No plugin system needed (no Django/SQLAlchemy in this project)

**Why basic mode, not strict:**
- Only 42% of functions have return annotations
- pandas-stubs are notoriously incomplete (false positives on `df["col"]`, `.groupby().agg()`)
- duckdb has no usable type stubs — every `conn.execute()` returns `Any`
- Start at `basic`, ratchet to `standard` when annotation coverage >80%

**Config:** `pyrightconfig.json` at repo root with `typeCheckingMode: "basic"`.

### Formatting: ruff format

- Zero-config, 30x faster than black, 99.9% compatible output
- Run once across all 178 files as a single commit (big diff, do it first)
- Eliminates whitespace/quote/trailing-comma noise from all future diffs

### Linting: Expand ruff rules + extend scope

Current: `F`, `E`, `W` on `pipeline/` and `trading_app/` only.

New: Add `I` (isort), `B` (bugbear), `UP` (pyupgrade), `SIM` (simplify).
Extend to: `scripts/` directory (production tooling).

- `I` — import sorting, auto-fixable
- `B` — mutable default args, assert-in-except, real bug patterns
- `UP` — modernize to 3.13 syntax (`Optional[X]` -> `X | None`, etc.)
- `SIM` — simplify boolean expressions, unnecessary else, etc.

### Dependencies: Commit to uv

- Move all deps from `requirements.txt` to `pyproject.toml [project.dependencies]`
- Use `uv lock` to generate `uv.lock` (already exists as skeleton)
- CI uses `uv sync --frozen` (fails if lock file is stale)
- Keep `requirements.txt` as generated fallback until CI is confirmed working, then delete
- All deps use `>=` lower bounds in pyproject.toml; exact versions live in `uv.lock`

### Security: pip-audit in CI

- `pip-audit --strict` as CI step (5 seconds, queries OSV database)
- Non-blocking initially (advisory), then blocking gate after clearing initial findings
- Not in pre-commit (needs network, too slow)

### Coverage: Measure then enforce

1. Measure current baseline: `pytest --cov=pipeline --cov=trading_app`
2. Set `--cov-fail-under` at (measured - 3%) in CI
3. Never set aspirational thresholds — only enforce what already exists

### Pre-commit: Keep custom hook as-is

- Staged-file-aware test selection is genuinely valuable
- Do NOT add pyright or heavy tools — keep <30s target
- Do NOT migrate to pre-commit framework (overhead for single-developer project)
- Add `ruff format --check` to the hook (instant, <1s)

### Editor Integration

- `pyrightconfig.json` — matches CI config
- `.vscode/settings.json` — enable ruff extension, use pyright for type checking

### Version Pinning

- Add `.python-version` file: `3.13`

## Scope Boundaries

**In scope:** pipeline/, trading_app/, ui/, scripts/
**Out of scope:** research/ (experimental scripts, not production)
**Deferred:** mypy strict mode (only after annotation coverage >80%)
**Skipped:** pre-commit framework migration, poetry

## Implementation Order

| Step | Action | Effort |
|------|--------|--------|
| 0 | Measure coverage baseline + add `.python-version` | 5 min |
| 1 | `ruff format` — run once, commit | 30 min |
| 2 | Expand ruff rules (I/B/UP/SIM), fix violations, commit | 30 min |
| 3 | Consolidate deps into pyproject.toml + uv lock | 1-2 hours |
| 4 | Add pyright basic + pyrightconfig.json | 1 hour |
| 5 | Add pip-audit + coverage threshold to CI | 15 min |
| 6 | Add .vscode/settings.json + editor config | 15 min |
| 7 | Add ruff format check to pre-commit hook | 10 min |
| 8 | Add drift checks for tool config consistency | 30 min |
| 9 | Verify: full CI green, pre-commit <30s, all tools integrated | 15 min |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| ruff format creates massive diff | One dedicated commit before any other changes |
| UP rewrites 50-100 files | Same — dedicated commit, review then merge |
| pyright finds errors we can't fix (duckdb stubs) | `basic` mode is lenient; add `# type: ignore` for known bad stubs |
| uv migration breaks CI | Keep requirements.txt as fallback until confirmed |
| pip-audit finds CVEs in current deps | Start as advisory (non-blocking), fix, then enforce |
| Coverage threshold too high | Measure first, set floor 3% below actual |

## Rollback

Revert these files: `pyproject.toml`, `ruff.toml`, `pyrightconfig.json`, `.github/workflows/ci.yml`, `.githooks/pre-commit`, `.vscode/settings.json`, `.python-version`. All config — no production code changes except auto-fixed lint/format.
