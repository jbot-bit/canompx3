# 2026-05-28 Codex audit — institutional-grade remote lanes/blockers + DB access path

## Scope and standard
- Request interpreted as: institutional-grade triage from current workspace, explicitly avoiding overlap with active Claude live-readiness stream.
- Audit class: **non-mutating**, evidence-first, fail-closed on unknowns.
- Boundary in this pass: no broker/live session start, no DB writes, no lane-allocation mutation.

## Evidence base used
- `docs/runtime/action-queue.yaml` (open/blocked lane-expansion workstream state)
- `HANDOFF.md` (cross-tool in-flight work to avoid duplicate effort)
- `pipeline/paths.py` (canonical `gold.db` path contract)
- local filesystem check for canonical DB presence at `/workspace/canompx3/gold.db`

## Institutional findings (ordered by capital/reliability impact)

### F1 — Current highest-EV lane-opener remains NQ-mini Stage 2 wiring
- Queue still shows one explicit `status: open` lane-expansion item: `nq_mini_stage2_wiring_2026_05_15`.
- This is a **production wiring** gap (not a research gap): symbol substitution + qty divisor + fail-closed non-integer qty behavior on live order path.
- This is the cleanest non-duplicative next lane-opener while Claude terminals are handling live-readiness Stage 2–5.

### F2 — Track D Gate 0 remains blocked, but unblocks via design-contract work (not heavy data)
- `track_d_mnq_comex_settle_gate0_runner_design` is blocked by MBP-1/TBBO availability/cost confirmation + implementation approval.
- This can progress with lightweight deliverables only: schema/runner contract, acceptance gates, and cost-bound plan before any dataset pull.

### F3 — Canonical DB is currently unavailable in this workspace, so only non-DB audit is truth-safe right now
- Canonical path resolves to `/workspace/canompx3/gold.db`.
- File is absent in this environment (path contract exists; artifact missing).
- Result: a true data-backed institutional capital audit cannot be completed **from this container right now** without first attaching the canonical DB.

## Answer: “Do you have access to DB? If not, how to get it?”

### Current state
- **No** — not in this session right now (`/workspace/canompx3/gold.db` missing).

### Zero-duplication, low-load way to attach DB
1. Keep single canonical DB source (do not copy). Place DB at canonical path or symlink to it.
2. In WSL clone/session, point to canonical file via env var:
   - `export DUCKDB_PATH=/absolute/path/to/canonical/gold.db`
3. Verify from code contract:
   - `python3 - <<'PY'\nfrom pipeline.paths import GOLD_DB_PATH\nprint(GOLD_DB_PATH)\nPY`
4. Run read-only probes first (no refresh/rebuild):
   - `python3 scripts/tools/live_readiness_report.py --profile topstep_50k_mnq_auto --strict-zero-warn`
   - `python3 scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto --dry-run --output /tmp/rebalance_preview.json`
5. Only after read-only audit passes provenance checks, decide whether any mutating workflow is justified.

## What is missing to open more lanes (project-aligned)
1. Live-path substitution wiring for NQ-mini Stage 2.
2. Profile-level activation (`execution_symbol_map`, `execution_qty_divisor`) behind explicit flag.
3. Integration test proving integer qty enforcement + fail-closed rejection on non-integer divisor results.
4. Track D Gate 0 runner/table contract acceptance so blocked research path can move without large data operations.

## Non-duplicative next-step recommendation
- **Immediate next:** execute `nq_mini_stage2_wiring_2026_05_15` in isolated branch/worktree.
- **Deconflict rule:** do not touch live-readiness Stage 2–5 files currently owned by Claude stream; only take NQ-mini Stage 2 scope.
