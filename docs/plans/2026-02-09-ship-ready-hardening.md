# Ship-Ready Hardening Plan
**Date:** 2026-02-09

## Scope

1. Re-enable Claude hooks
2. Add ruff linting to pre-commit + fix all lint issues
3. Root directory cleanup (move archives, delete junk)
4. Add pytest-cov for coverage visibility
5. DB backup script
6. README rewrite

## Implementation Order

### Step 1: Ruff setup
- `pip install ruff` + add to requirements.txt
- Create `ruff.toml` (line-length=120, F+E rules only)
- Run `ruff check`, fix all issues in one pass
- Add `ruff check` to `.githooks/pre-commit` as step [0]

### Step 2: Re-enable Claude hooks
- Set `disableAllHooks: false` in `.claude/settings.local.json`

### Step 3: Root cleanup
- Create `docs/archive/`
- Move: ARCHIVE_*.txt, fix*.txt, FIX*.txt, ideas.txt, notes.txt, nested.txt, volume.txt, volumepass.txt, entrymodels*.txt, backtest10.txt, audit.txt, list.txt, retrievebars.txt, walkforward.txt, TCA.txt
- Delete: _ul, chmod (if they exist as junk artifacts)

### Step 4: Coverage tracking
- Add `pytest-cov` to requirements.txt
- Update CI to run `--cov=pipeline --cov=trading_app --cov-report=term-missing`

### Step 5: Backup script
- Create `scripts/backup_db.py` (~50 LOC)
- Copy gold.db to backups/gold_YYYYMMDD.db, keep last 5

### Step 6: README rewrite
- Project overview, quick start, file structure, key commands
