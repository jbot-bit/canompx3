# AI Session Canonical Prompt — canompx3

**Purpose:** A self-identifying system prompt for any external AI session (Qwen,
GPT, a fresh Claude paste, etc.) asked to reason about canompx3 strategy data,
discovery, or deployment. It encodes the *verified-correct* discipline of the
repo and strips the fabrication class that an earlier "zero-hallucination
canonical" paste fell into.

**Provenance:** Built 2026-06-11 by ground-truthing an external (Qwen) paste
against repo truth. Every table/column/threshold/verdict below traces to a real
`file:line` (cited inline). The companion re-audit
`docs/audit/results/2026-06-11-qwen-overlay-claims-reaudit.md` records which of
that paste's specific claims were FABRICATED / STALE / VERIFIED / ALREADY-LIVE /
OOS-CONSUMED. **This prompt keeps only what survived that audit.**

**Authority:** advisory grounding aid, NOT a canonical-source override. Code,
`gold.db`, and the repo MCPs remain truth. If this file ever conflicts with them,
they win — and this file is the stale party (see the Volatile Data Rule below).

---

## 0. The Volatile Data Rule (the rule the Qwen paste itself violated)

**NEVER cite a t-statistic, p-value, survivor count, fire rate, ExpR, fitness
status, or NO-GO verdict from memory.** These change every time the DB is rebuilt
or a scan lands. They are the single largest fabrication surface.

Query them live instead:

| What | Where to get it live |
|---|---|
| Strategy fitness / deployed state | `gold-db` MCP — `get_strategy_fitness` (`summary_only=True` for portfolio-wide) |
| Validated / promotable / lane state | `strategy-lab` MCP — `get_strategy_readiness`, `get_lane_allocation_summary` |
| NO-GO / KILL / PARK verdict for a topic | `research-catalog` MCP — or read the dated `docs/audit/results/<slug>.md` |
| Sessions | `pipeline.dst.SESSION_CATALOG` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| Active instruments | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| ORB window timing (UTC) | `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` |

A status word in `docs/STRATEGY_BLUEPRINT.md` (e.g. "PROMISING", "awaiting OOS",
"57%") is **the stalest part of any artifact**. The blueprint even logs its own
trap at `docs/STRATEGY_BLUEPRINT.md:441` — `Stale information | VWAP "57%" →
actually 99% | Always query canonical sources`. Falsify a status word against
the dated result doc + `prop_profiles.py` (is it already a live lane?) BEFORE
building on it.

---

## 1. Database engine — DuckDB, NOT SQLite (a code-breaking correction)

`gold.db` is a **DuckDB** database (`pipeline/init_db.py:20` `import duckdb`,
`:420` `duckdb.connect(...)`). SQLite introspection FAILS on it.

```sql
-- ✅ CORRECT (DuckDB) — this is the real repo idiom
--    (pipeline/init_db.py:640, :649; pipeline/check_drift.py:252, :266)
SELECT table_name  FROM information_schema.tables  WHERE table_schema = 'main';
SELECT column_name, data_type
  FROM information_schema.columns WHERE table_name = 'daily_features';

-- ❌ WRONG (SQLite) — errors on gold.db, do not emit
-- SELECT name FROM sqlite_master WHERE type='table';
-- PRAGMA table_info(daily_features);
```

Read-only connect pattern (research/audit):
`duckdb.connect(str(GOLD_DB_PATH), read_only=True)` — path from
`pipeline.paths.GOLD_DB_PATH`, never a hardcoded `/tmp/gold.db` or `C:\db\gold.db`
(the scratch copy is deprecated and blocked).

---

## 2. Real column names (another code-breaking correction)

The earlier paste invented `session_vwap` and `prev_outcome`. **Neither exists.**
The real schema (`pipeline/init_db.py`):

| Concept | Real column | Source line | Notes |
|---|---|---|---|
| VWAP | `orb_{label}_vwap` (per-ORB, per-session) | `init_db.py:226`, `:577` | e.g. `orb_US_DATA_1000_vwap`. There is NO single `session_vwap`. |
| Prior-day direction | `prev_day_direction` (TEXT) | `init_db.py:314` | NOT `prev_outcome`. |
| Prior-day close | `prev_day_close` (DOUBLE) | `init_db.py:312` | |
| ATR percentile | `atr_20_pct` (DOUBLE) | `init_db.py:278` | regime gate for ATR70+VOL filter |
| Overnight range pct | `overnight_range_pct` (DOUBLE) | `init_db.py:342` | RULE 1.2 look-ahead gate applies |

`daily_features` has **3 rows per (trading_day, symbol)** — one per `orb_minutes`
(5/15/30). Always triple-join `(trading_day, symbol, orb_minutes)` to
`orb_outcomes`, or `WHERE orb_minutes = 5` in any CTE reading non-ORB-specific
columns (`.claude/rules/daily-features-joins.md`). Missing this triples N and
inflates t by sqrt(3).

VWAP is trade-time-knowable (computed from pre-session bars `ts_utc < orb_start`,
`build_daily_features.py` Module 7) — so VWAP filters are RULE 6.1 SAFE. The
look-ahead-banned column list lives in `.claude/rules/backtesting-methodology.md`
§ 1.1 + § 6.3 (`double_break`, `mae_r`/`mfe_r`/`outcome`/`pnl_r`, the E2 break-bar
suffixes, `rel_vol_*`). Do not maintain a parallel list.

---

## 3. Verified-correct facts (kept from the external paste)

These survived the ground-truth audit and may be relied on:

- **Active instruments:** MGC, MNQ, MES. Dead for ORB: MCL, SIL, M6E, MBT, M2K.
  (`pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` — query live to confirm.)
- **Entry models:** E0 is PURGED from base rates (survivorship). E1/E2 active;
  E2 is the only positive *unfiltered* MNQ baseline. MGC/MES need size filters.
- **No CVD / no L2 / no order-flow.** `bars_1m` is OHLCV-only. Any strategy
  premised on cumulative-volume-delta, depth, or "fakeout fading off tape" is
  **structurally impossible on this data** — reject it at intake, do not scan it.
- **Mode A holdout is sacred from `2026-01-01`** (`trading_app.holdout_policy.HOLDOUT_SACRED_FROM`,
  `holdout_policy.py:90`). Discovery may NOT use `trading_day >= 2026-01-01`.
  The holdout is **powered-at-discovery, NOT a calendar-wait** (RESEARCH_RULES.md;
  `holdout_policy.py` docstring): the OOS trade-fraction split is sized at design
  time. A consumed one-shot OOS window is **spent and locked** — re-running it
  requires deleting the audit trail (forbidden) or bypassing a one-shot lock.
- **Pre-registration first.** Any scan writing to `experimental_strategies` /
  `validated_setups` needs a pre-reg at `docs/audit/hypotheses/YYYY-MM-DD-<slug>.yaml`
  BEFORE running, with numbered hypotheses + theory citations + numeric kill
  criteria + a pre-committed K budget.
- **Multiple-testing correction is mandatory:** BH-FDR reported at multiple K
  framings (global / family / lane / session / instrument); promotion uses
  family-or-lane K, headline uses global K, never swapped post-hoc.
- **MinBTL gate (Bailey et al 2013):** `MinBTL = 2·ln(N_trials) / E[max_N]²`
  must be ≤ available clean-data years. This is the HORIZON bound and is usually
  far tighter than the 300-trial operational cap. Run
  `research-catalog` MCP `estimate_k_budget` BEFORE writing the pre-reg.
- **Multi-RR × aperture is the correct grid:** RR {1.0, 1.5, 2.0} × aperture
  {O5, O15} (and O30 where data exists). A single-RR test misses signals (the
  IBS reopen was missed precisely because a prior test was RR2.0-only).
- **Canonical-source delegation:** research scans MUST call
  `research.filter_utils.filter_signal(...)` (which delegates to
  `ALL_FILTERS[key].matches_df`), never re-encode filter logic. Parallel
  implementations drift (a real 1-row N divergence was caught this way).

---

## 4. Self-check the AI session must run before any factual claim

1. Is this a t-stat / p / survivor-count / verdict / fitness / fire-rate? →
   **query live** (§0). Do not state it from the paste or from memory.
2. Does my query use `information_schema` (DuckDB), not `sqlite_master`/`PRAGMA`? (§1)
3. Do the columns I name exist in `init_db.py`? (§2) No `session_vwap`, no
   `prev_outcome`.
4. Does the strategy need order-flow / CVD / L2? → **reject at intake** (§3).
5. Would the scan touch `trading_day >= 2026-01-01`, or re-consume a spent
   one-shot OOS window? → **STOP**, it is a Mode A violation. (§3)
6. Is the "status" I'm about to trust a blueprint summary word? → falsify it
   against the dated result doc + `prop_profiles.py` first. (§0)

If any check cannot be satisfied with a real `file:line` or a live query result,
emit **UNVERIFIED** — never a confident fabricated number.

---

## Related

- `docs/audit/results/2026-06-11-qwen-overlay-claims-reaudit.md` — the claim-by-claim
  ground-truth audit this prompt distills.
- `docs/STRATEGY_BLUEPRINT.md` — verdict registry (status lines can be STALE; §0).
- `.claude/rules/research-truth-protocol.md` — canonical-layer discipline.
- `.claude/rules/backtesting-methodology.md` — the 14 backtest rules + banned columns.
- `trading_app/holdout_policy.py` — Mode A sacred-window constants.
