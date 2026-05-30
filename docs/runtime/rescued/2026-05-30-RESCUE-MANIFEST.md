# Rescue Manifest — 2026-05-30 (cross-terminal stash recovery)

**Why this exists:** during a post-`/clear` reconciliation on `main`, a
`git stash push -u` swept THREE unrelated workstreams into one stash. An early
(wrong) read dismissed 5 of the files as "CRLF noise." A whitespace-ignoring
re-diff (`git diff -w HEAD stash@{0}`) proved they are **real, substantial,
NOT-mine, incomplete work**. Operator instruction: "ensure other bugs and shit
are saved and not stashed/hidden or forgotten." This manifest + the sibling
files in this dir are the durable save. Nothing here is mine to commit.

## Saved artifacts in this directory

| File | What it is |
|------|-----------|
| `2026-05-30-lane-allocator-feature-cache-WIP.patch` | Real code diff (7266 B, 64 added lines) vs `b10df244` |
| `2026-05-30-lane_allocation.STASHED.json` | Regenerated lane allocation (rebalance_date 2026-05-30) |
| `2026-05-30-topstep_50k_mnq_auto.STASHED.json` | Regenerated MNQ auto lane file |

The stash `stash@{0}` ("skip-crg+live-preflight+handoff WIP pre-rebase") is
**left intact** as a second safety net — do NOT `git stash drop` it until the
work below is committed by its owner.

## The NOT-MINE work that was nearly dropped (capital-path — needs its owner)

### 1. `trading_app/lane_allocator.py` — feature_cache optimization
Adds `feature_cache: dict[...] | None` param to `_per_month_expr()` so
`daily_features` + cross-asset ATR enrichment loads ONCE per
`(instrument, orb_minutes, start, end)` instead of re-querying per strategy.
Threaded through `compute_lane_scores`. **`trading_app/` = capital path.**
Status: incomplete — no companion test changes, no stage doc. NOT verified.

### 2. `scripts/tools/rebalance_lanes.py` — `--strict-live-clean` flag
New CLI flag + imports `apply_c8_gate` / `apply_chordia_gate` /
`apply_live_tradeability_gate`. Gates allocation on current SR CONTINUE
evidence for strict live-readiness. Status: incomplete, NOT verified.

### 3. `docs/runtime/lane_allocation*.json` — regenerated allocation
Real rebalance output (2026-05-23 → 2026-05-30): updated trailing_expr /
trailing_n / status_reason / lane assignments. Capital-relevant. This is the
OUTPUT of running #1 + #2, so it must land WITH them (or be regenerated), not
independently.

## MINE (already handled, listed so the picture is complete)

- `b10df244` perf(drift): `--skip-crg-advisory` — committed, unpushed.
- `78a7cdfb` fix(ralph): fast drift gate — committed, unpushed.
- `live_readiness_report.py` + test — live-preflight import fix; staged, commit
  BLOCKED by Check 191 false positive (see below).
- `HANDOFF.md` — baton edit (stale SHA, will fix to post-rebase HEAD).

## KNOWN BUG blocking commits — NOT mine to fix

**Check 191 `Drift cache meta cold-recheck`** is a non-deterministic FALSE
POSITIVE: PASSES standalone (`check_drift.py --skip-crg-advisory`), FAILS as the
sole violation inside the full pre-commit suite. Root cause = cumulative
cache/sys.modules state from preceding checks pollutes the tail cold-rerun
(documented in memory + being fixed by a sibling terminal that owns the
`check_drift.py` meta-recheck lines). My staged change does NOT touch
`check_drift.py` or `_drift_cache.py`, so the block is unrelated to my work.

## Recovery commands

```bash
# Restore the not-mine code work onto a clean tree:
git apply docs/runtime/rescued/2026-05-30-lane-allocator-feature-cache-WIP.patch

# Or recover everything from the still-intact stash:
git stash show stash@{0} --stat
git checkout stash@{0} -- <path>     # selective
```
