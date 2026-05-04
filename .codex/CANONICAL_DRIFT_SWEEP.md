# Canonical Drift Sweep

Use this when you want one aggressive pass for repo contradictions, stale docs, and compatibility traps without blindly "fixing" the first weird thing you see.

## Ground Rules

- Do not auto-fix canonical code from a single mismatch.
- First decide whether the mismatch is:
  - `REAL_FINDING`: behaviorally dangerous and not already explained
  - `STALE_DOC`: documentation or prompt drift
  - `DOCUMENTED_TRAP`: surprising on purpose; abstraction layer explains it
  - `COMPAT_SURFACE`: deprecated or compatibility-only path that still exists intentionally
- If code, docs, and current comments disagree, find the authoritative layer first.
- Prefer guardrails, drift checks, and explicit inventory before behavior edits.

## Current Repo Hits

### 1. Stale audit prompt: DB authority is wrong

- `docs/prompts/SYSTEM_AUDIT.md:152`
- It says the canonical DB is `C:/db/gold.db`.
- Current code says the opposite:
  - `pipeline/paths.py:20`
  - `pipeline/paths.py:33`
  - `pipeline/paths.py:45`
- Current truth: project-root `gold.db` is canonical; `C:/db/gold.db` is deprecated scratch and is blocked when passed through `DUCKDB_PATH`.
- Classification: `STALE_DOC`

### 2. Stale audit prompt: live authority points to the wrong module

- `docs/prompts/SYSTEM_AUDIT.md:175`
- It tells the audit to compare live portfolio tiers against `live_config.py`.
- Current code says:
  - `trading_app/live_config.py:137`
  - `trading_app/live_config.py:143`
- Current truth: `trading_app.prop_profiles.py` is the sole authority for live strategy config; `live_config.py` is a deprecated compatibility surface.
- Classification: `STALE_DOC`

### 3. Real raw-flag misuse: direct `orb_active` read

- `scripts/tools/refresh_data.py:231`
- It branches on `cfg.get("orb_active", True)` directly.
- Canonical warning already exists:
  - `docs/STRATEGY_BLUEPRINT.md:122`
  - `pipeline/asset_configs.py:301`
  - `pipeline/asset_configs.py:309`
- Current truth: `orb_active` is not safe to consume directly because `DEAD_ORB_INSTRUMENTS` can override it. Use `ACTIVE_ORB_INSTRUMENTS` or a canonical helper, not the raw flag.
- Classification: `REAL_FINDING`

### 4. M2K is a documented trap, not a standalone contradiction

- `docs/STRATEGY_BLUEPRINT.md:122`
- `pipeline/asset_configs.py:186`
- `pipeline/asset_configs.py:304`
- `pipeline/asset_configs.py:313`
- `M2K` still has `orb_active=True`, but is excluded by `DEAD_ORB_INSTRUMENTS`.
- That is surprising, but it is explicitly documented and encoded in the canonical derived list.
- The dangerous bug is not the bool by itself. The dangerous bug is code that consumes the bool directly.
- Classification: `DOCUMENTED_TRAP`

### 5. Deprecated runtime fallback still exists in active code

- `scripts/run_live_session.py:71`
- `trading_app/live/session_orchestrator.py:104`
- These still lazy-import deprecated `build_live_portfolio()`.
- `trading_app/live_config.py:578`
- `trading_app/live_config.py:591`
- Current truth: this path is deprecated and known-bad as a source of live strategy resolution, but it is still present as compatibility/fail-closed fallback.
- Classification: `COMPAT_SURFACE`

### 6. Compatibility-heavy `live_config` consumers

These are not necessarily wrong today, but they are the blast-radius list any audit must review before claiming `live_config.py` can be removed or ignored:

- `pipeline/check_drift.py:1904`
- `pipeline/check_drift.py:2458`
- `scripts/tools/generate_trade_sheet.py:32`
- `scripts/tools/generate_promotion_candidates.py:34`
- `scripts/tools/pinecone_snapshots.py:190`
- `scripts/tools/sensitivity_analysis.py:37`
- `scripts/infra/daily_paper_run.py:30`
- `scripts/audits/phase_7_live_trading.py:20`
- `ui_v2/server.py:267`
- `ui_v2/server.py:535`
- Classification: `COMPAT_SURFACE`

### 7. Legacy/scratch DB path references still exist across scripts

Examples:

- `pipeline/run_full_pipeline.py:10`
- `scripts/tools/explore.py:7`
- `scripts/tools/orb_size_deep_dive.py:13`
- `scripts/tools/backfill_dollar_columns.py:11`
- `scripts/infra/scratch_run.py:10`
- `scripts/infra/scratch_ingest.py:41`

Do not classify every `C:/db/gold.db` mention as a bug. Some are intentional scratch workflows or examples. Audit whether the reference is:

- a blocked deprecated default in active production code
- a scratch-only helper
- an old doc/example that now misstates canonical DB authority

Classification: mixed `COMPAT_SURFACE` and `STALE_DOC`

## Single-Pass Sweep Prompt

```text
Run a full canonical drift sweep on this repo.

Rules:
1. Do not auto-fix code from a single mismatch.
2. For every mismatch, first classify it as REAL_FINDING, STALE_DOC, DOCUMENTED_TRAP, or COMPAT_SURFACE.
3. Prefer canonical derived interfaces over raw config flags.
4. Treat deprecated compatibility layers as blast-radius surfaces, not immediate deletion candidates.
5. Report exact file:line evidence for every claim.

Mandatory checks:
- Find stale docs/prompts that still claim C:/db/gold.db is canonical.
- Find stale docs/prompts that still treat trading_app/live_config.py as live authority instead of trading_app/prop_profiles.py.
- Find any production code that reads orb_active directly instead of using ACTIVE_ORB_INSTRUMENTS or another canonical helper.
- Find active runtime callers of deprecated build_live_portfolio().
- Inventory all non-test imports of trading_app.live_config and classify each as required compatibility, stale dependency, or true bug.
- Inventory hardcoded scratch DB defaults and separate intentional scratch tooling from stale canonical claims.

Output format:
- Findings first, ordered by severity.
- For each item: classification, file:line, why it is risky, and the likely fix surface.
- Then a separate section for documented traps and compatibility-only surfaces so they do not get "fixed" blindly.
```

## Grep Battery

Run these together when doing the sweep:

```bash
rg -n "orb_active" pipeline trading_app scripts tests docs --glob '*.py' --glob '*.md' --glob '!scripts/tmp*'
rg -n "from trading_app\\.live_config|import trading_app\\.live_config" . --glob '*.py' --glob '!scripts/tmp*' --glob '!**/.venv*'
rg -n "build_live_portfolio\\(|LIVE_PORTFOLIO|LIVE_MIN_EXPECTANCY_R|PAPER_TRADE_CANDIDATES|_load_best_regime_variant" trading_app scripts pipeline ui ui_v2 tests --glob '*.py' --glob '!scripts/tmp*'
rg -n -F "C:/db/gold.db" CLAUDE.md docs pipeline trading_app scripts tests --glob '!docs/ralph-loop/**'
rg -n "canonical DB is|Live portfolio tiers 1/2/3" docs/prompts/SYSTEM_AUDIT.md
```

## Recommended Next Checks

- Add a drift guard for direct raw `orb_active` reads outside `pipeline/asset_configs.py`.
- Update `docs/prompts/SYSTEM_AUDIT.md` so it no longer audits against stale DB and live-authority assumptions.
- Keep `live_config.py` blast-radius inventory separate from "remove it now" work.
