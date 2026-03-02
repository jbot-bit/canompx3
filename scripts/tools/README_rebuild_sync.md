# Rebuild with Pinecone Sync

## Quick Start

### Sync only (after doc changes, memory updates):
```bash
python scripts/tools/sync_pinecone.py
```

### Full rebuild + sync (after outcome_builder changes):
```bash
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

### Dry run (see what would sync):
```bash
python scripts/tools/sync_pinecone.py --dry-run
```

### Force re-upload everything:
```bash
python scripts/tools/sync_pinecone.py --force
```

## What Gets Synced

| Tier | Files | When Re-uploaded |
|------|-------|-----------------|
| Static | Authority docs, findings | On content change |
| Living | config.py, dst.py, cost_model.py, live_config.py | Every sync |
| Memory | 19 ClaudeMem memory files | Every sync |
| Research Output | ~50 narrative .md/.txt files | On content change |
| Generated | 4 snapshot docs from gold-db | Regenerated every sync |

## When to Sync

- After editing TRADING_RULES.md or RESEARCH_RULES.md
- After a rebuild (use the wrapper script)
- After a productive Claude Code session (memory files updated)
- After adding new research output files

## Rebuild Chain Details

The wrapper runs 6 steps in order:

1. **Outcome Builder** -- recompute trade outcomes (`--force` flag)
2. **Strategy Discovery** -- grid search for profitable combos
3. **Strategy Validator** -- multi-phase validation + walk-forward (MNQ uses WF; others skip it)
4. **E3 Retirement** -- validator promotes E3 to active; this script retires them
5. **Edge Families** -- cluster validated strategies by trade-day hash
6. **Pinecone Sync** -- upload changed docs + regenerated snapshots

For all-instrument rebuilds without sync, use `full_rebuild.sh` instead.
