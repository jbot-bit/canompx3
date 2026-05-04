---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Parallel Rebuild: Multi-Aperture Optimization

**Date:** 2026-03-19
**Status:** IMPLEMENTED

## Problem

The rebuild pipeline (`scripts/infra/parallel_rebuild.py`) took a single `--orb-minutes` value. To rebuild all apertures (O5/O15/O30), the user had to invoke it 3 times. Each invocation copied the full DB per instrument, ran all steps, and merged — meaning 3x DB copies, 3x merges, 3x validation passes, and 3x FDR corrections when only 1 of each was needed.

## Solution

Added `--orb-minutes-list` (default: `[5, 15, 30]`) to the rebuild orchestrator. Aperture-dependent steps (features, outcome, discovery) loop per aperture within each instrument. Validation runs ONCE after all apertures complete — it reads all `orb_minutes` from `experimental_strategies` and computes global FDR with correct K.

## Savings

| Resource | Before (3 runs) | After (1 run) | Savings |
|----------|-----------------|---------------|---------|
| DB copies | 9 (3 per run × 3 instruments) | 3 | 67% |
| Merge operations | 9 | 3 | 67% |
| Validation runs | 9 | 3 | 67% |
| FDR corrections | 9 (partial K each) | 3 (full K) | 67% + correctness |
| CLI invocations | 3 | 1 | 67% |

## Blast Radius

**1 file changed:** `scripts/infra/parallel_rebuild.py`

All downstream tools (outcome_builder, strategy_discovery, strategy_validator, build_edge_families) already accept `--orb-minutes` as a CLI arg. No downstream changes needed.

## Key Design Decisions

1. **Apertures sequential within instrument** — DuckDB single-writer constraint prevents parallel aperture writes to the same DB file.
2. **Validation runs ONCE** — reads all experimental_strategies regardless of orb_minutes, correct global FDR K.
3. **`APERTURE_STEPS` set** — explicitly marks which steps are aperture-dependent (features, outcome, discovery) vs global (validation).

## Usage

```bash
# Full rebuild — all instruments, all apertures (default)
python scripts/infra/parallel_rebuild.py --all

# Single aperture (legacy behavior)
python scripts/infra/parallel_rebuild.py --instruments MGC --orb-minutes-list 5

# Specific instruments, all apertures
python scripts/infra/parallel_rebuild.py --instruments MGC MNQ MES
```
