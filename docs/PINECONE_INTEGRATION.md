# Pinecone Knowledge Assistant Integration

**Status**: ✅ Complete and verified (Mar 2, 2026)

The Pinecone Assistant (`orb-research`) is a managed RAG service providing project-wide knowledge search with citations. 51 files across 5 content tiers (static, living, memory, research_output, generated) are synced automatically.

## What It Solves

1. **Live and Learning**: Generated snapshots from gold-db refresh every sync (portfolio state, fitness report, live config, research index)
2. **No Drift**: Change detection (SHA256) ensures only modified files upload
3. **High-Quality Answers**: Built from authority docs, memory, and research findings — no raw data dumps
4. **Three-System Separation**: Pinecone (project knowledge) / gold-db (live data) / NotebookLM (academic methodology)

## Quick Start

```bash
# Query the assistant
/pinecone:assistant-chat assistant orb-research message "What research have we done on MGC?"

# Sync after rebuilds
python scripts/tools/sync_pinecone.py

# Full rebuild + sync
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

## Architecture

### Content Tiers

| Tier | Files | What | Updates |
|------|-------|------|---------|
| **static** | 14 | Authority docs (TRADING_RULES, RESEARCH_RULES, guardians, audits, prompts) | On change |
| **living** | 4 | Config as markdown (config.py, dst.py, cost_model.py, live_config.py) | Every sync |
| **memory** | ~20 | Claude memory files (research findings, instrument analysis, audit notes) | Every sync |
| **research_output** | 9 | Bundled research (82 files → 9 topic bundles by prefix) | Every sync |
| **generated** | 4 | Snapshots from gold-db (portfolio state, fitness, live config, research index) | Every sync |

### Snapshot Generators

Each generated snapshot is a synthesis of live data:

- **portfolio_state**: Active strategies by instrument/session/entry_model/aperture (from `validated_setups` and `edge_families`)
- **fitness_report**: Recent fitness assessment by strategy (from `get_strategy_fitness()`)
- **live_config**: Current trading portfolio specs (from `trading_app.live_config.LIVE_PORTFOLIO`)
- **research_index**: Catalog of research output files by topic

Scripts: `scripts/tools/pinecone_snapshots.py` with CLI `--snapshot <name>`, `--list`, or all 4.

### Sync Automation

**File**: `scripts/tools/sync_pinecone.py`

**Key features**:
- SHA256 change detection — only uploads modified files
- Research bundling — 82 files → 9 topics via `RESEARCH_BUNDLE_GROUPS` prefix mapping
- `.py` → `.md` conversion — wraps config files in markdown code blocks
- UTF-8 sanitization — handles encoding errors with `errors='replace'`
- Conflict resolution — deletes old file before uploading new version

**Usage**:
```bash
python scripts/tools/sync_pinecone.py              # Delta sync (fast)
python scripts/tools/sync_pinecone.py --dry-run    # Show what would upload
python scripts/tools/sync_pinecone.py --force      # Upload everything
```

### Rebuild Wrapper

**File**: `scripts/tools/run_rebuild_with_sync.sh MGC`

6-step orchestration:
1. `outcome_builder` (pre-compute trade outcomes)
2. `strategy_discovery` (grid search)
3. `strategy_validator` (validation + walk-forward for MNQ, skip for others)
4. `retire_e3_strategies` (soft-retire E3 after validator promotion)
5. `build_edge_families` (cluster by trade-day hash)
6. `sync_pinecone.py` (upload generated snapshots + changed files)

## Routing Rules

See `.claude/skills/pinecone-assistant/SKILL.md` for complete decision framework.

**Summary**:
- "What research have we done?" → **Pinecone**
- "How many strategies right now?" → **gold-db MCP**
- "How does BH FDR work?" → **NotebookLM**

## Implementation Details

### Manifest

**File**: `scripts/tools/pinecone_manifest.json`

Defines all 5 tiers with:
- Static file list (14 authority docs)
- Living file list (4 configs)
- Memory base_path + glob pattern
- Research glob patterns (5 patterns covering `research/output/`)
- Generated snapshots + output dir

### State & Tracking

**File**: `scripts/tools/.pinecone_sync_state.json` (git-ignored)

Stores:
- `last_sync`: ISO 8601 timestamp of last successful sync
- `assistant_name`: "orb-research"
- `hashes`: {rel_key: sha256} for all 51 files
- `file_count`: Total files synced

**File**: `scripts/tools/.pinecone_assistant_id` (git-ignored)

Contains: `orb-research` (assistant name for all uploads)

### Testing

**Files**:
- `tests/tools/test_pinecone_manifest.py` — 3 tests (manifest structure, file existence)
- `tests/tools/test_pinecone_snapshots.py` — 4 tests (all snapshot generators)
- `tests/tools/test_sync_pinecone.py` — 11 tests (bundling, .py→.md, UTF-8, change detection, etc.)

**Run**: `pytest tests/tools/test_pinecone_*.py -v`

**Status**: All 18 passing ✅

## Verification

**Live query test** (Mar 2, 2026):
```
Q: "What sessions work for MGC and why is E0 dead?"

A: [Returned detailed answer with citations from BREAD_AND_BUTTER_REFERENCE.md
    and e0_entry_model.md memory file, covering CME_REOPEN, TOKYO_OPEN, and E0
    retirement reasons]
```

✅ Pinecone Assistant successfully: indexed 51 files, computed vector embeddings, returned RAG response with citations, and resolved cross-file references.

## When to Sync

After:
- Any rebuild chain (`outcome_builder` → `strategy_validator` → `build_edge_families`)
- Updating authority docs (TRADING_RULES.md, RESEARCH_RULES.md, etc.)
- Writing new research findings to `research/output/`
- Updating memory files (research audit notes, findings)
- Modifying config.py, dst.py, cost_model.py, live_config.py

Sync is idempotent — safe to run anytime. Only changed files upload.

## Files Created/Modified

**New**:
- `scripts/tools/pinecone_manifest.json` — Tier definitions + file inventory
- `scripts/tools/pinecone_snapshots.py` — 4 snapshot generators + CLI
- `scripts/tools/sync_pinecone.py` — Main sync orchestrator (bundling, .py→.md, UTF-8, change detection)
- `scripts/tools/run_rebuild_with_sync.sh` — Rebuild wrapper
- `scripts/tools/README_rebuild_sync.md` — Rebuild+sync documentation
- `.claude/skills/pinecone-assistant/SKILL.md` — Three-system routing rules
- `tests/tools/test_pinecone_manifest.py` — Manifest tests
- `tests/tools/test_pinecone_snapshots.py` — Snapshot tests
- `tests/tools/test_sync_pinecone.py` — Sync tests
- `docs/PINECONE_INTEGRATION.md` — This file

**Modified**:
- `.gitignore` — Added patterns for generated snapshots, bundles, living wrappers
- `.claude/rules/notebooklm.md` — Narrowed scope to academic methodology only

**Ignored** (git-ignored):
- `.pinecone_sync_state.json` — Sync state tracking
- `.pinecone_assistant_id` — Assistant identifier
- `scripts/tools/_snapshot_*.md` — Generated snapshots
- `scripts/tools/_bundle_*.md` — Research bundles
- `scripts/tools/_living_*.md` — Config wrappers
- `scripts/tools/_clean_*.md` — UTF-8 sanitized files

## Git Commits

1. `e980592` — .gitignore for Pinecone state files
2. `b20b7f7` — Pinecone manifest + tests
3. `480fd77` — Snapshot generators
4. `af0b654` — Main sync orchestrator (bundling, .py→.md, UTF-8, change detection)
5. `b5472a4` — Rebuild wrapper + documentation
6. `5bae668` — Routing rules + NotebookLM scope update

## Known Limitations

- **100-file quota** on Pinecone Assistant free tier — solved by bundling 82 research files into 9 topics
- **`.py` files rejected** by Pinecone — solved by converting to `.md` wrappers with code blocks
- **UTF-8 encoding issues** on 4 files — solved by `errors='replace'` sanitization
- **Rate limiting** on file deletes — mitigated with `time.sleep()` between delete batches
- **Generated snapshots go stale** — user must re-run sync after rebuilds

## Next Steps

1. ✅ Pinecone sync fully automated in rebuild chain
2. ✅ All tests passing (1,923 total, 18 Pinecone-specific)
3. ✅ Routing rules documented and in place
4. ⏳ Optional: Add Pinecone drift check (verify assistant file count matches manifest)
5. ⏳ Optional: Wire sync into GitHub Actions CI
6. ⏳ Optional: Add alerts when sync fails

## Troubleshooting

**Q**: "Assistant not found" error
**A**: Set `PINECONE_API_KEY` env var and restart Claude Code: `export PINECONE_API_KEY="..."`

**Q**: Sync uploads but response is stale
**A**: Re-run after rebuilds: `bash scripts/tools/run_rebuild_with_sync.sh MGC`

**Q**: Query returns "no relevant documents"
**A**: Sync may be incomplete. Run `python scripts/tools/sync_pinecone.py --force` to re-upload all 51 files.

**Q**: File UTF-8 error during sync
**A**: Sync automatically sanitizes and creates `_clean_<filename>`. Safe to ignore.

## References

- `.claude/skills/pinecone-assistant/SKILL.md` — Decision framework for three-system routing
- `.claude/rules/notebooklm.md` — NotebookLM scope (academic methodology only)
- `scripts/tools/sync_pinecone.py` — Implementation details
- `trading_app/mcp_server.py` — gold-db MCP for live data queries
