# Research Modes and Lineage — Canonical Spec

**Authority:** This spec wins on questions of (a) what counts as research truth, (b) which tables a query may use in which mode, (c) how research findings are provenanced, and (d) what mechanically enforces the rules. It supersedes the contradictory sections of `RESEARCH_RULES.md` § Discovery Layer Discipline and `.claude/rules/research-truth-protocol.md` § Validated Universe Rule. Those documents will be rewritten to point at this spec in Phase A.

**Status:** RESOLVED. The § 9 WFE policy decision is made (see § 9). Phase A is complete pending the doc cross-link updates listed in § 10. Phase E implementation is **unblocked** as of 2026-04-07 `cd9b5e9` (the `canonical-filter-self-description` stage that previously held `pipeline/check_drift.py` in its scope_lock has closed) but is **not yet scheduled** — the implementation contract in § 9.4 is ready to execute when the user authorizes.

**Audience:** Future Claude sessions, the user, any human reviewer auditing how research findings on this project were produced.

**Grounding:** This spec is grounded in the project's local literature in `resources/`:

- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` — backtests are not theory; theory must come first
- `resources/false-strategy-lopez.pdf` — full tested space disclosure is mandatory; survivor-only reporting is fraud
- `resources/deflated-sharpe.pdf` — multiple testing requires correction over the entire search, not just the survivors
- `resources/backtesting_dukepeople_liu.pdf` — very few trials are needed to produce an impressive but false in-sample result
- `resources/Algorithmic_Trading_Chan.pdf` — implementation details and classic biases are why live diverges from backtest
- `resources/Building_Reliable_Trading_Systems.pdf` — finite data means curve-fitting is the default failure mode
- `resources/benjamini-and-Hochberg-1995-fdr.pdf` — FDR correction discipline

---

## 1. Why this spec exists

Two truth-protocols were added 11 days apart:

- **2026-03-24** — `RESEARCH_RULES.md` § Discovery Layer Discipline declares `validated_setups`, `edge_families`, `live_config` **BANNED** for truth-finding.
- **2026-04-04** — `.claude/rules/research-truth-protocol.md` § Validated Universe Rule declares research queries **MUST** be scoped to `validated_setups`.

Both rules were correct reactions to a real failure. The first was added because derived layers were being treated as ground truth. The second was added because researchers were running unbounded queries against `orb_outcomes` (3M+ rows) and finding fake signals in noise. Each rule prevents the previous disaster. **Together they contradict each other**, and a new researcher (or future Claude session) cannot follow both.

The audit at `docs/audits/edge_family_audit_2026-03-30.md:87, 89, 445-450` separately documented:

- `validated_setups.p_value` is NULL for all 488 promoted strategies
- `n_trials_at_discovery` is NULL for all 488 promoted strategies
- A `RESEARCH_RULES.md:59` standard ("WFE > 50% = strategy likely real") that is not enforced in `walkforward.py` or `strategy_validator.py` — a doc/code gap that qualifies as a silent failure under `.claude/rules/institutional-rigor.md`. (The audit's original empirical example of "3 MNQ strategies with WFE 0.33-0.43" was valid as of 2026-03-30 but those rows no longer exist in `validated_setups` as of 2026-04-07 — see § 9 empirical state.)

The audit at `docs/audits/2026-03-21-rebuild-truth-audit.md:207` separately documented that discovery+validation has run **outside** the standard rebuild pipeline with no `rebuild_manifest` entry and no `validation_run_log` entry. The audit caught it after the fact. Nothing blocked it at write time.

The combination of incoherent rules + un-provenanced promoted layer + bypassable validation path + ambiguous gate is a single defect: **the project distinguishes "audited" from "enforced" inconsistently**. This spec fixes the meta-policy. Schema and code patches that follow from it are listed in § 10 (migration plan) and tracked separately.

---

## 2. The Four Research Modes

Every research-related activity in this project fits exactly one of four modes. Mode determines scope, allowed inputs, and write destination. **A run that needs two modes is two runs.**

### 2.1 DISCOVERY — "Does this hypothesis create new edge anywhere in the universe?"

- **Inputs allowed:** `bars_1m` (BASE_TRUTH); `bars_5m`, `daily_features`, `orb_outcomes`, `exchange_statistics`, `experimental_strategies` (COMPUTED_FACTS).
- **Inputs forbidden:** `validated_setups`, `edge_families`, `validated_setups_archive`, `family_rr_locks`, `live_config`, `prop_profiles.ACCOUNT_PROFILES`, all docs, all comments, all memory files. These bias the search toward what already survived; using them in DISCOVERY is survivor-selection by policy.
- **Pre-registration required:** every DISCOVERY run MUST insert a row into `research_hypotheses` BEFORE the run executes, then a row into `research_runs` linking that hypothesis to a snapshot. See § 5.
- **Per-test logging required:** every test in the broad scan writes a row to `research_run_tests`. Survivors and failures are recorded with equal rigor. See § 5C.
- **Write destinations:** `experimental_strategies` (the run's results) and `research_run_tests` (per-test record). DISCOVERY runs may NEVER write to `validated_setups` directly.
- **Failure record:** if a DISCOVERY run finds nothing, the `research_runs` row is updated to `result='DEAD'`. Failures are not silent.

### 2.2 CONFIRMATION — "Does a discovered candidate survive walk-forward, OOS holdout, and FDR?"

- **Inputs allowed:** same as DISCOVERY, plus the specific candidate spec being confirmed.
- **Inputs forbidden:** same as DISCOVERY (no peeking at the deployment portfolio when confirming an edge).
- **Holdout discipline:** the holdout_boundary recorded in the parent DISCOVERY hypothesis is **immutable**. CONFIRMATION uses dates strictly outside that boundary. The 2026 holdout rule from `RESEARCH_RULES.md:26` applies and is non-negotiable.
- **K accounting:** CONFIRMATION runs do NOT reset K. The K from the originating DISCOVERY hypothesis carries forward. If a candidate is re-tested under a new variation, that's a new DISCOVERY hypothesis with a new K plan, not a free CONFIRMATION pass.
- **Write destinations:** a `research_runs` row with `mode='CONFIRMATION'` and `parent_hypothesis_id` set; per-test rows in `research_run_tests`; on success, a row in `validated_setups` (with provenance fields populated — see § 8) and a row in `validation_run_log` linked to the originating `run_id`.

### 2.3 DEPLOYMENT_ANALYTICS — "Within already-validated strategies, what does the live portfolio look like?"

- **Inputs allowed:** everything, including `validated_setups`, `edge_families`, `family_rr_locks`, `prop_profiles.ACCOUNT_PROFILES`, `paper_trades`, plus all COMPUTED_FACTS and BASE_TRUTH for context joins. Filter introspection should use the canonical `filter.describe()` method on filter instances (defined in `trading_app/config.py`) — do not re-implement filter inspection.
- **Inputs forbidden:** none from the project. The only restriction is on what this mode may CONCLUDE (see below).
- **What it cannot do:** generate new edge claims, promote new strategies, modify `validated_setups`. Its outputs are reports, dashboards, and monitoring — never table writes that affect what gets traded.
- **Write destination:** none. DEPLOYMENT_ANALYTICS is read-only against the database. Outputs go to stdout, files in `reports/`, or dashboards.
- **Examples of legitimate questions in this mode:** "what's the live equity curve of the topstep_50k_mnq_auto profile?", "which deployed lanes had a losing month?", "how does feature X correlate with the survival of already-validated strategies?", "is the live ExpR tracking the backtest ExpR within tolerance?".
- **Hard rule:** any finding in DEPLOYMENT_ANALYTICS that looks like a new edge **must be re-tested in DISCOVERY mode against a fresh snapshot** before it can become a claim. Mode boundaries do not have escape hatches.

### 2.4 OPERATIONS — "Run the live trading system."

- **Inputs allowed:** everything live execution requires — `live_config`, `prop_profiles`, `paper_trades`, broker adapters, signal generation paths.
- **What it cannot do:** make any research truth claim. OPERATIONS has zero authority over what's an edge. It executes what's already been validated and routes through what's already been deployed.
- **Write destinations:** `paper_trades`, `prospective_signals`, broker journals — operational records only.
- **Why this is its own mode:** to prevent the silent escalation pattern where "monitoring shows X works" becomes "X is an edge" without any of the discovery-confirmation discipline. OPERATIONS may notice things; it may not conclude things. Anything it notices must be promoted into DISCOVERY for proper testing.
- **Out of scope for this spec:** the live execution rules themselves live in `TRADING_RULES.md`, `trading_app/live/`, and `trading_app/prop_profiles.py`. This spec only declares that OPERATIONS is a distinct mode with no research authority.

### 2.5 The contradiction, dissolved

| Old rule | New scope |
|---|---|
| RESEARCH_RULES § Discovery Layer Discipline (ban validated_setups) | Applies to **DISCOVERY** and **CONFIRMATION** only. Stays in force in those modes. |
| research-truth-protocol § Validated Universe Rule (require validated_setups join) | Applies to **DEPLOYMENT_ANALYTICS** only. Reframed: not a research truth rule, a portfolio-analytics scope rule. |

A researcher who is testing whether (e.g.) "prev_day_high distance" predicts future ORB outcomes is in DISCOVERY mode and may NOT join validated_setups. A researcher who is asking "of the 124 deployed strategies, which had losing weeks last month?" is in DEPLOYMENT_ANALYTICS mode and SHOULD join validated_setups. Same database, same tables, different mode, different rules.

---

## 3. The Trust Hierarchy

Three tiers. Every research-relevant table is classified into exactly one tier. The tier determines whether a query result can be cited as evidence and whether a finding based on that table is reproducible.

### 3.1 BASE_TRUTH — only one table is sacred

| Table | Defined in | Why it's BASE_TRUTH |
|---|---|---|
| `bars_1m` | `pipeline/init_db.py:34` | Raw 1-minute OHLCV from Databento DBN files. The only layer that is not the output of project-internal code. Bit-for-bit reproducible from the source DBN files. |

**Rules for BASE_TRUTH:**
- A claim grounded in `bars_1m` is reproducible from raw market data without trusting any project code beyond the DBN parser.
- BASE_TRUTH may not be modified, only extended (new days appended).
- If `bars_1m` is wrong, the source DBN file is wrong. There is no other failure mode.

### 3.2 COMPUTED_FACTS — reproducible from BASE_TRUTH

| Table | Defined in | Built from |
|---|---|---|
| `bars_5m` | `pipeline/init_db.py:48` | Aggregation of `bars_1m` |
| `daily_features` | `pipeline/init_db.py:218` | `bars_1m` + `pipeline/cost_model.py` (G6/G8/COST_LT/etc. computed from cost specs) |
| `orb_outcomes` | `trading_app/db_manager.py:61` | `bars_1m` + `bars_5m` + entry-model logic in `pipeline/outcome_builder*.py` |
| `exchange_statistics` | `pipeline/init_db.py:344` | Pre-session statistics computed from `bars_1m` |
| `experimental_strategies` | `trading_app/db_manager.py:106` | Cartesian grid scan over `orb_outcomes` filtered by `daily_features` filter conditions |
| `strategy_trade_days` | `trading_app/db_manager.py:255` | Filter conditions applied to `daily_features` to enumerate trade days per strategy |

**Rules for COMPUTED_FACTS:**
- Defeasible by recomputation. If you suspect a row is wrong, rebuild it from BASE_TRUTH at the same git SHA and compare.
- Trustworthy IFF: the builder code is correct AND the table was rebuilt from current BASE_TRUTH AND the build is recorded in `rebuild_manifest`.
- A finding grounded in COMPUTED_FACTS is reproducible IFF the snapshot_id at finding time can be reproduced. Hence the snapshot mechanism in § 4.
- If the builder code changes, ALL downstream COMPUTED_FACTS are stale until rebuilt. This is enforced by `rebuild_manifest.git_sha`.

### 3.3 METADATA — non-evidentiary by default

| Table | Defined in | What it records |
|---|---|---|
| `validated_setups` | `trading_app/db_manager.py:172` | Promotion decisions for which strategies are deployable |
| `validated_setups_archive` | `trading_app/db_manager.py:239` | Historical snapshots of retired strategies |
| `edge_families` | `trading_app/db_manager.py:272` | Strategy clustering by trade-day overlap |
| `family_rr_locks` | `pipeline/init_db.py:121` | Per-family RR target lock decisions |
| `paper_trades` | `trading_app/db_manager.py:303` | Live or paper execution journal |
| `prospective_signals` | `pipeline/init_db.py:101` | Live signal generation log |
| `rebuild_manifest` | `pipeline/init_db.py:141` | Pipeline rebuild operational record |
| `pipeline_audit_log` | `pipeline/init_db.py:154` | Per-write audit log with git_sha and rebuild_id |
| `validation_run_log` | `trading_app/db_manager.py:587` | Per-validation-run rejection counts |
| All `docs/`, `memory/`, comments, docstrings | — | Human-readable narrative |

**Rules for METADATA:**
- METADATA is not evidence. Citing METADATA in a research claim is a methodology error.
- METADATA is useful for **routing** ("which strategies are currently deployed?"), **operational tracking** ("did the rebuild finish?"), and **audit trails** ("what was promoted on what date?").
- If METADATA conflicts with a fresh recomputation against BASE_TRUTH or COMPUTED_FACTS, **METADATA loses**. Mark the METADATA STALE and update it.
- Any code that mutates METADATA must record the git SHA, rebuild_id, and (for promoted strategies) the dataset_snapshot_id. Without lineage, METADATA cannot be audited and is therefore liable to silent drift.

### 3.4 What this hierarchy is NOT

- It is not a permissions system. Any code can read any table. The hierarchy governs **what kind of claim** a result supports, not **who** can read it.
- It is not a deprecation ladder. METADATA is necessary and useful — it's just not evidence.
- It does not replace the existing fail-closed pipeline rules. Fail-closed governs WHEN a write happens. Trust hierarchy governs WHAT a read justifies.

---

## 4. Dataset Snapshot Identity

The single mutable `gold.db` is a known weakness (`pipeline/paths.py:20-21`, see § 1). This spec proposes a two-tier snapshot system: lightweight fingerprint snapshots for everyday research, and frozen physical snapshots for any state that survives to a `validated_setups` promotion.

### 4.1 The snapshot table

```sql
CREATE TABLE IF NOT EXISTS dataset_snapshots (
    snapshot_id              TEXT        PRIMARY KEY,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    snapshot_type            TEXT        NOT NULL CHECK (snapshot_type IN ('FINGERPRINT','FROZEN')),
    git_sha                  TEXT        NOT NULL,
    rebuild_id               TEXT,                    -- nullable: not all snapshots correspond to a rebuild
    instruments              TEXT[]      NOT NULL,    -- which instruments are in scope
    bars_1m_max_ts           JSON        NOT NULL,    -- {"MNQ": "2026-04-06T...", ...}
    bars_1m_row_count        JSON        NOT NULL,    -- {"MNQ": 5234567, ...}
    daily_features_max_day   JSON        NOT NULL,    -- {"MNQ": "2026-04-06", ...}
    orb_outcomes_max_day     JSON        NOT NULL,
    fingerprint_sha256       TEXT        NOT NULL,    -- hash of the above tuples; identity check
    physical_artifact_path   TEXT,                    -- non-null only for FROZEN snapshots
    physical_artifact_sha256 TEXT,                    -- non-null only for FROZEN snapshots
    notes                    TEXT
);
```

### 4.2 Two snapshot tiers

**FINGERPRINT snapshot** — the routine case.
- Computed by `pipeline.snapshot_freeze --type fingerprint`.
- Records the identity of the database state but does NOT copy the file.
- Reproducibility relies on rebuilding gold.db until the fingerprint matches at a known git_sha.
- Suitable for DISCOVERY and exploratory CONFIRMATION runs that do not lead to promotion.
- Cheap: ~1KB per row, instant to create.

**FROZEN snapshot** — for promotion-eligible state.
- Required whenever a CONFIRMATION run is expected to promote a strategy to `validated_setups`.
- Created by `pipeline.snapshot_freeze --type frozen` which copies `gold.db` to `snapshots/<snapshot_id>.duckdb` and records the file's SHA256.
- The artifact is immutable. If the artifact is deleted, the snapshot row is marked `status='LOST'` and any strategy promoted from it is flagged for re-validation.
- Trust comes from re-running the fingerprint computation against the artifact at audit time and confirming the SHA256 matches.
- Expected volume: ~5-10 frozen snapshots per year × ~3-5GB each = 15-50GB. Manageable on local disk.

### 4.3 The snapshot helper

A new module `pipeline/snapshot.py` exposes:

```python
def freeze_current_state(
    snapshot_type: Literal["FINGERPRINT","FROZEN"],
    instruments: list[str] | None = None,
    notes: str = "",
) -> str:
    """Compute a fingerprint of the current gold.db state and write a row
    to dataset_snapshots. If snapshot_type='FROZEN', also copy gold.db to
    snapshots/<snapshot_id>.duckdb and record the file SHA256.
    Returns the snapshot_id.
    """

def verify_snapshot(snapshot_id: str) -> bool:
    """For FROZEN snapshots: re-fingerprint the physical artifact and confirm
    SHA256 matches. For FINGERPRINT snapshots: re-fingerprint current gold.db
    and confirm match (only valid if no rebuilds have occurred since)."""
```

CLI wrapper: `python -m pipeline.snapshot_freeze --type fingerprint|frozen --instruments MNQ MGC MES --notes "..."`. Prints the snapshot_id.

### 4.4 Snapshot doctrine — manifest is locator, fingerprint is trust

**The snapshot row in `dataset_snapshots` is METADATA**, not evidence. By the trust hierarchy in § 3, METADATA cannot be cited as proof of anything by itself.

- The **manifest** (the row) tells you WHERE to look — the artifact path, the git_sha, the expected fingerprint.
- **Trust** comes from successfully running `verify_snapshot(snapshot_id)` and getting `True`. That re-runs the fingerprint computation and confirms the artifact (or current DB, for FINGERPRINT type) matches the recorded SHA256.
- A snapshot row that has never been verified is a CLAIM about a state, not a proof of it.
- An auditor who finds a `validated_setups` row with `snapshot_id=X` MUST run `verify_snapshot(X)` before trusting that the strategy was validated against the recorded state. If verification fails, the strategy is flagged for re-validation.

This is the philosophical correction: even our own provenance metadata is non-evidentiary. The only way to trust a snapshot is to re-fingerprint it.

### 4.5 What snapshots do NOT do

- They do not version the schema. Schema versioning is handled by `init_db.py` migrations and `check_drift.py`.
- They do not freeze code. Code is versioned by `git_sha`. The snapshot records the git_sha at creation time so reproducibility includes the code.
- They do not eliminate the mutable gold.db. They make state at a moment **addressable and verifiable**, not immutable.

---

## 5. Research Tracking — Three Tables

The hypothesis ledger is split into three tables. The split is grounded in `false-strategy-lopez.pdf` and `deflated-sharpe.pdf`: K accounting requires per-test row-level records, not just summary counts. A single hypothesis can spawn multiple runs across different snapshots; each run produces multiple per-test results.

### 5A. `research_hypotheses` — the THEORY

```sql
CREATE TABLE IF NOT EXISTS research_hypotheses (
    hypothesis_id        TEXT        PRIMARY KEY,    -- e.g. "2026-04-07-mnq-prevhigh-001"
    registered_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title                TEXT        NOT NULL,
    theory               TEXT        NOT NULL,       -- what is being claimed and why
    mechanism            TEXT        NOT NULL,       -- the structural market reason this should work
    mode                 TEXT        NOT NULL CHECK (mode IN ('DISCOVERY','CONFIRMATION','DEPLOYMENT_ANALYTICS')),
    target_universe      JSON        NOT NULL,       -- {"instruments": [...], "sessions": [...], ...}
    allowed_params       JSON        NOT NULL,       -- {"orb_minutes": [5,15,30], "rr_target": [...]}
    forbidden_params     JSON,                       -- pre-declared off-limits variations
    holdout_policy       TEXT        NOT NULL,       -- "OOS_2026" or "WALK_FORWARD_24M_12M" etc.
    parent_hypothesis_id TEXT,                       -- for CONFIRMATION linking back to DISCOVERY
    author               TEXT        NOT NULL DEFAULT 'josh',
    status               TEXT        NOT NULL DEFAULT 'OPEN',  -- OPEN | CLOSED | SUPERSEDED
    notes                TEXT,
    FOREIGN KEY (parent_hypothesis_id) REFERENCES research_hypotheses(hypothesis_id)
);
```

**Rules:**
- A hypothesis is registered BEFORE any run executes against it. No retroactive hypothesis creation.
- `theory` and `mechanism` are mandatory text fields. "Numbers show it works" is not a mechanism (per `RESEARCH_RULES.md` § Mechanism Test). Mechanism-less hypotheses fail closed at registration.
- `forbidden_params` is the explicit defense against parameter sweep creep. Pre-declare what variations are off-limits, e.g. `{"rr_target": "no values outside [0.75, 1.0, 1.5, 2.0]"}` to forbid a researcher from optimizing RR mid-run.
- `holdout_policy` is recorded as text and is **immutable**. Once registered, the holdout cannot be changed without superseding the hypothesis.

### 5B. `research_runs` — the EXECUTION

```sql
CREATE TABLE IF NOT EXISTS research_runs (
    run_id                  TEXT        PRIMARY KEY,
    hypothesis_id           TEXT        NOT NULL,
    snapshot_id             TEXT        NOT NULL,
    git_sha                 TEXT        NOT NULL,
    mode                    TEXT        NOT NULL CHECK (mode IN ('DISCOVERY','CONFIRMATION','DEPLOYMENT_ANALYTICS')),
    script_path             TEXT        NOT NULL,    -- e.g. "research/scan_prev_day_range.py"
    started_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at            TIMESTAMPTZ,
    status                  TEXT        NOT NULL DEFAULT 'RUNNING',  -- RUNNING | COMPLETE | ABORTED | DEAD
    tested_count            INTEGER,                  -- total number of tests in the run
    reported_count          INTEGER,                  -- subset that survived to the report
    k_for_fdr               INTEGER,                  -- K used for BH FDR — must equal tested_count
    search_space_summary    TEXT,
    result                  TEXT,                     -- DEAD | POSITIVE_DISCOVERY | CONFIRMATION_FAIL | CONFIRMATION_PASS
    p_value_observed        DOUBLE,
    notes                   TEXT,
    FOREIGN KEY (hypothesis_id) REFERENCES research_hypotheses(hypothesis_id),
    FOREIGN KEY (snapshot_id) REFERENCES dataset_snapshots(snapshot_id)
);
```

**Rules:**
- A run executes AGAINST a snapshot at a git_sha. Both are non-null.
- `tested_count` MUST equal the actual number of test rows in `research_run_tests` for this run_id. Drift check enforces this.
- `k_for_fdr` MUST equal `tested_count`. Reporting K < tested_count is the silent fraud the false-strategy paper warns about; this constraint catches it.
- `reported_count` is the number of tests that the researcher chose to highlight in the output. The gap between `tested_count` and `reported_count` is the visible measure of selection bias — present, just disclosed.
- A run ends in exactly one of four results: DEAD (nothing survived), POSITIVE_DISCOVERY (DISCOVERY mode found candidates), CONFIRMATION_FAIL (CONFIRMATION mode rejected the candidate), CONFIRMATION_PASS (CONFIRMATION mode confirmed and the candidate is eligible for promotion).

### 5C. `research_run_tests` — the per-test record

```sql
CREATE TABLE IF NOT EXISTS research_run_tests (
    run_id            TEXT        NOT NULL,
    test_id           INTEGER     NOT NULL,           -- sequential within run
    family            TEXT,                            -- e.g. "PREVHIGH_DIST_LT"
    instrument        TEXT,
    session           TEXT,
    orb_minutes       INTEGER,
    entry_model       TEXT,
    confirm_bars      INTEGER,
    rr_target         DOUBLE,
    filter_type       TEXT,
    filter_params     TEXT,
    n                 INTEGER,                         -- sample size for this test
    raw_p             DOUBLE,                          -- uncorrected p-value
    adj_p             DOUBLE,                          -- BH-adjusted p-value
    effect_size       DOUBLE,                          -- e.g. ExpR or Sharpe
    oos_metric        DOUBLE,                          -- OOS ExpR or similar
    is_survivor       BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, test_id),
    FOREIGN KEY (run_id) REFERENCES research_runs(run_id)
);
```

**Rules:**
- EVERY test in a broad scan writes a row. Survivors and failures are equally recorded.
- `is_survivor=TRUE` marks rows that passed the run's selection criteria. The COUNT of survivors should equal `research_runs.reported_count`.
- The COUNT of all rows for a run_id should equal `research_runs.tested_count`. Drift check enforces both equalities.
- This table is the mechanical defense against selection bias. K accounting can be re-derived from this table at any time without trusting the researcher's claimed K.

### 5D. `claims` — derived view

`claims` is intentionally NOT a fourth physical table. It is a VIEW over the surviving rows:

```sql
CREATE VIEW claims AS
SELECT
    rrt.run_id,
    rrt.test_id,
    rh.hypothesis_id,
    rh.mode,
    rr.snapshot_id,
    rr.git_sha,
    rrt.family,
    rrt.instrument,
    rrt.session,
    rrt.entry_model,
    rrt.confirm_bars,
    rrt.rr_target,
    rrt.filter_type,
    rrt.n,
    rrt.raw_p,
    rrt.adj_p,
    rrt.effect_size,
    rrt.oos_metric,
    rr.tested_count AS k_at_origin,
    (rr.result = 'CONFIRMATION_PASS' AND rh.mode = 'CONFIRMATION') AS promotion_eligible
FROM research_run_tests rrt
JOIN research_runs rr ON rrt.run_id = rr.run_id
JOIN research_hypotheses rh ON rr.hypothesis_id = rh.hypothesis_id
WHERE rrt.is_survivor = TRUE;
```

A "claim" is therefore a survivor of a run, with full lineage to hypothesis + snapshot + git_sha + the K used for FDR. Promotion eligibility is computed on the fly. Adding a fourth physical table would just duplicate this for no benefit.

### 5E. Workflow

1. **Before a research run:** insert a `research_hypotheses` row (or reuse an existing one for CONFIRMATION).
2. **At run start:** freeze a snapshot (FINGERPRINT for routine, FROZEN if promotion is on the table). Insert a `research_runs` row with status='RUNNING', linked to the hypothesis and the snapshot.
3. **During the run:** for each test executed, write a `research_run_tests` row immediately. Do not buffer survivors-only.
4. **At run end:** update the `research_runs` row with `status='COMPLETE'`, `tested_count`, `reported_count`, `k_for_fdr`, `result`, `p_value_observed`. Drift check verifies tested_count matches the per-test row count.
5. **For CONFIRMATION_PASS:** the surviving rows in `research_run_tests` become eligible for promotion to `validated_setups` (see § 8).

### 5F. What this fixes

- **Researcher degrees of freedom:** every test costs a row. K accounting is mechanical.
- **Survivor-only fraud:** the gap between `tested_count` and `reported_count` is auditable. A reviewer can compute `1 - reported_count/tested_count` to see how much of the search space was hidden.
- **Cross-session continuity:** a future Claude session can query `research_hypotheses` + `research_runs` to see what's been tried, with what K, against what snapshot, with what result.
- **Honest FDR:** `k_for_fdr` MUST equal `tested_count`. If the researcher tries to report K=30 against 480 tests, the constraint fails.

### 5G. Migration of existing memory files

The 30+ memory files documenting tested ideas (`break_speed_signal_retest.md`, `oi_research_killed.md`, `depth_at_break_killed.md`, `presession_feature_scan.md`, etc.) are converted to ledger rows by a one-shot migration script. Each becomes:
- A `research_hypotheses` row with `mode='DISCOVERY'` and `theory`/`mechanism` extracted from the memory file
- One or more `research_runs` rows with `snapshot_id='LEGACY_PRE_2026_04_07'` (a placeholder marker since we cannot reconstruct the original state)
- Per-test rows in `research_run_tests` IF the memory file recorded them with enough detail; otherwise the run is marked `tested_count=NULL` with a note explaining the gap

Memory files survive as one-line summaries pointing at the hypothesis_id. The ledger is the canonical record going forward.

---

## 6. Reporting Contract

**Every research result that the project considers valid MUST disclose all of the following.** Results that omit any field are non-evidentiary by default and cannot be promoted, cited in commits, or stored in `validated_setups`.

| Field | Required for | Example |
|---|---|---|
| `mode` | All modes | DISCOVERY |
| `hypothesis_id` | DISCOVERY, CONFIRMATION | 2026-04-07-mnq-prevhigh-001 |
| `run_id` | DISCOVERY, CONFIRMATION | 2026-04-07-mnq-prevhigh-001-run01 |
| `snapshot_id` | DISCOVERY, CONFIRMATION | 2026-04-07-MNQ-abc1234 (FINGERPRINT or FROZEN) |
| `git_sha` | All modes | abc1234 |
| `script_path` | All modes | research/scan_prev_day_range.py |
| `target_universe` | DISCOVERY, CONFIRMATION | {"instruments": ["MNQ"], "sessions": ["NYSE_OPEN"]} |
| `tested_count` | DISCOVERY, CONFIRMATION | 480 |
| `reported_count` | DISCOVERY, CONFIRMATION | 4 |
| `k_for_fdr` | DISCOVERY, CONFIRMATION | 480 (must equal tested_count) |
| `holdout_policy` | DISCOVERY, CONFIRMATION | OOS_2026 (immutable from hypothesis) |
| `pre_registered` | All modes | TRUE if the hypothesis row predates the run; FALSE for retroactive |
| `is_post_selection_analysis` | All modes | TRUE if examining already-validated rows; FALSE if generating new claims |

**Honest summary block** at the end of every research script's output:

```
SURVIVED SCRUTINY: <list with N, raw_p, adj_p>
DID NOT SURVIVE: <count and reason>
TESTED: <total>
REPORTED: <subset>
K_FOR_FDR: <must equal tested>
SNAPSHOT: <snapshot_id>
HYPOTHESIS: <hypothesis_id>
GIT_SHA: <sha>
PRE_REGISTERED: <yes/no>
POST_SELECTION: <yes/no>
HOLDOUT: <policy>
```

This format extends the existing one in `RESEARCH_RULES.md` § Honest Summary Format. The new fields are mandatory.

---

## 7. Validation Standards

Beyond what `RESEARCH_RULES.md` already requires (BH FDR, sample size rules, mechanism test), this spec adds four mechanical requirements grounded in the local literature.

### 7.1 Negative control on broad scans

Every broad scan in DISCOVERY mode MUST include a **negative control** — a parallel run on a label-permuted or temporally-shuffled version of the data, scored on the same metric. The negative control must produce statistically distinguishable results from the real run; if it does not, the real run's findings are dismissed as artifact.

Grounding: `false-strategy-lopez.pdf` and `deflated-sharpe.pdf` — without a null distribution, statistical significance is meaningless.

### 7.2 Permutation / null sanity check

For any new feature or filter being tested, run a **permutation null** at least once:
- Shuffle the feature value across days, preserving the marginal distribution
- Re-run the scan
- The shuffled feature should produce no significant survivors at the same K

If the shuffled run produces survivors, the feature is contaminated (look-ahead, autocorrelation, or join error) and the original run is invalid.

### 7.3 Predeclared simple baseline

Complex models, multi-feature combinations, and ML-style scoring runs MUST predeclare a **simple baseline** (e.g., "the existing G6/G8 filter") in the hypothesis row. The complex model MUST beat the simple baseline on the same metric on the same OOS data. If it does not, the complex model is dismissed regardless of in-sample performance.

Grounding: `Lopez_de_Prado_ML_for_Asset_Managers.pdf` — theory first, simple before complex.

### 7.4 Purge / embargo for temporal features

Any feature whose computation window overlaps the prediction window MUST use **purge + embargo** to prevent leakage:
- **Purge:** training/test splits drop overlapping observations from training
- **Embargo:** a buffer period after the test window is excluded from the training set

This is currently used in ML splits but should be generalized as a utility (`pipeline.purge_embargo`) and required for any temporal feature study, not just ML.

Grounding: López de Prado, Chapter 7 of `Lopez_de_Prado_ML_for_Asset_Managers.pdf`.

### 7.5 What this section does NOT do

- It does not replace `RESEARCH_RULES.md` § Statistical Rigor. Sample size rules, p-value thresholds, mechanism test, sensitivity analysis all remain in force.
- It does not specify how to compute the negative control or permutation null — those are tooling work in Phase G of the migration plan.
- It does not gate every research run on having all four checks. The checks scale with the seriousness of the claim: a quick exploratory query needs none; a CONFIRMATION run that may promote a strategy needs all four.

---

## 8. Promoted-Layer Self-Defense (`validated_setups` patches)

`validated_setups` (db_manager.py:172) currently cannot explain itself. The audit at edge_family_audit_2026-03-30.md:87, 89 documented that `p_value` and `n_trials_at_discovery` are NULL for all 488 rows. The schema also has no link back to a snapshot, a rebuild, a hypothesis, or a git SHA. This spec patches the schema.

### 8.1 New columns on validated_setups

```sql
ALTER TABLE validated_setups ADD COLUMN raw_p_value           DOUBLE;
ALTER TABLE validated_setups ADD COLUMN n_trials_at_discovery INTEGER;
ALTER TABLE validated_setups ADD COLUMN snapshot_id           TEXT;
ALTER TABLE validated_setups ADD COLUMN git_sha               TEXT;
ALTER TABLE validated_setups ADD COLUMN run_id                TEXT;
ALTER TABLE validated_setups ADD COLUMN provenance            TEXT DEFAULT 'NATIVE';
-- provenance: NATIVE | RECONSTRUCTED | LEGACY_PRE_2026_04_07
```

### 8.2 Write-time invariants (enforced in `strategy_validator.py` promotion path)

For new promotions (`provenance='NATIVE'`):
- `raw_p_value` is copied from the originating `research_run_tests` row at promotion time. NOT NULL.
- `n_trials_at_discovery` is copied from `research_runs.tested_count`. NOT NULL.
- `snapshot_id` references a `dataset_snapshots` row of `snapshot_type='FROZEN'`. NOT NULL. Promotion fails closed if the snapshot does not exist or is FINGERPRINT-type.
- `git_sha` matches the current git HEAD. NOT NULL. Promotion from a dirty working tree is allowed but logged as a warning.
- `run_id` references a `research_runs` row with `mode='CONFIRMATION'` and `result='CONFIRMATION_PASS'`. NOT NULL. Promotion fails closed if the run does not exist or is in a different state.

For backfilled rows (`provenance='RECONSTRUCTED'` or `'LEGACY_PRE_2026_04_07'`):
- All five fields above MAY be NULL.
- `provenance` is NOT NULL and explicitly marks the row as un-traceable to the new standard.
- `check_drift.py` reports the count of legacy rows on every run, so the backlog is visible until cleared.

### 8.3 What this fixes

- The promoted layer can now explain itself without joining other tables. p_value, n_trials, K, snapshot, git_sha, run are all on the row.
- A future audit can verify a strategy by calling `verify_snapshot(snapshot_id)` and re-running the originating run. If the result differs, the lineage shows where to look.
- The bypass that the 2026-03-21 audit caught (validated_setups written without a rebuild_manifest entry) becomes a hard-fail at write time: no FROZEN snapshot → no promotion.

---

## 9. WFE Policy — RESOLVED

**Status: DECIDED 2026-04-07. Option A (Modified) — hard gate with explicit literature-grounding caveat. Zero-cost (no retirements needed). Implementation is unblocked (the concurrent `canonical-filter-self-description` stage closed at commit `cd9b5e9` and drift check #85 was added at `32356b8`, releasing `pipeline/check_drift.py` from that stage's scope_lock) but is not yet scheduled — awaiting user authorization to execute the § 9.4 implementation contract.**

### 9.0 Background

The audit at `edge_family_audit_2026-03-30.md:445-450` flagged an audit ambiguity:

> "Either add WFE ≥ 0.5 as a hard gate or document that WFE is informational-only. Current ambiguity is confusing for audit."

`RESEARCH_RULES.md:59` says:
> "Walk-forward efficiency (WFE) > 50% = strategy likely real. < 50% = likely overfit."

But `trading_app/walkforward.py:91-96` does not enforce WFE. The actual pass rule is four conditions (ALL required, fail-closed):

1. `n_valid >= min_valid_windows`
2. `pct_positive >= min_pct_positive` (default 60%)
3. `agg_oos_exp_r > 0` (trade-weighted across windows)
4. `total_oos_trades >= min_trades_per_window * min_valid_windows`

`wfe` is computed (lines 292-313, trade-weighted OOS/IS ExpR — the mean-of-ratios pathology was already refined out) and stored on `validated_setups.wfe`, but it is not in the pass rule. The doc/code gap qualifies as a silent failure under `.claude/rules/institutional-rigor.md`.

### 9.1 Empirical state — verified 2026-04-07

Direct query of `validated_setups` (gold.db, canonical source):

| Metric | Value |
|---|---|
| Active strategies | 124 |
| Strategies with `wfe < 0.5` (any status) | **0** |
| `wfe` min across active | **0.5141** |
| `wfe` p25 | 0.6754 |
| `wfe` median | 0.8584 |
| `wfe` p75 | 1.1911 |
| `wfe` max | 3.8157 |
| Active MNQ `US_DATA_1000` strategies | 1 (`MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10`, wfe=1.4992) |

**The original audit example (3 MNQ strategies with WFE 0.334, 0.433, 0.434) no longer exists in `validated_setups`.** Those rows were cleaned up through normal validation cycles between 2026-03-30 and 2026-04-07. The doc/code gap is the remaining defect; the empirical cost of closing it is now zero.

### 9.2 Literature grounding — verified against local sources

The institutional-rigor rule requires grounding in local literature before citing a threshold. The verification below was done against every local PDF in `resources/` on 2026-04-07. Every quote cited here was extracted from the local file — none from training memory.

#### Primary source (Pardo — NOT extractable locally)

| Source | Status | Finding |
|---|---|---|
| `resources/rober prado - optimization of trading strategies.pdf` (Pardo, 2008, Wiley, ISBN 978-0-470-12801-5) | **PARTIAL — 30 pages only (front matter)** | Only pages i-xvi (title / copyright / TOC / foreword / preface / Dunn foreword) are present. Chapter 11 "Walk-Forward Analysis" starts at printed page 237 and is **not extractable from our local PDF**. TOC confirms the chapter exists with sections: "Robustness and Walk-Forward Efficiency" (p238), "The Cure for Overfitting" (p239), "A More Reliable Measure of Risk and Return" (p241), "The Theory of Relevant Data" (p243), "Is the Strategy Robust?" (p256), "What Rate of Profit Should We Expect?" (p260), "Walk-Forward Analysis and the Portfolio" (p261). Chapter 12 "The Evaluation of Performance" (p263) has a section on "Model Efficiency" (p273). **None of these pages are extractable.** See § 9.5 for the acquisition list. |

#### Local grounding chain (verified quotes)

Although the Pardo primary is missing, two local academic-practitioner sources cite Pardo and describe the methodology directly:

| Source | Page | Verified content |
|---|---|---|
| **Chan, Ernest P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. ISBN 978-1-118-46014-6.** | **p25** | *"[W]e must perform a walk-forward test as a final, true out-of-sample test. This walk-forward test can be conducted in the form of paper trading, but, even better, the model should be traded with real money (albeit with minimal leverage) so as to test those aspects of the strategy that eluded even paper trading. **Most traders would be happy to find that live trading generates a Sharpe ratio better than half of its backtest value.**"* — extracted verbatim from `resources/Algorithmic_Trading_Chan.pdf` p25. This is the **practitioner 0.5 halving heuristic** applied to Sharpe ratio. It is not identical to Pardo's WFE (which is OOS/IS ExpR ratio), but it is the same conceptual 0.5 conservative lower-bound for OOS/IS performance preservation. Chan is a Wiley-published quant with a PhD and a commercially active quant fund. |
| **Chan, Ernest P. (2013). *Algorithmic Trading*. Wiley.** | **p55** | *"Out-of-sample testing, cross-validation, and high Sharpe ratios are all good practices for reducing data-snooping bias, but none is more definitive than walk-forward testing."* — Chan's summary bullet ranking walk-forward as the most load-bearing data-snooping defense. Supports using walk-forward as the canonical gate mechanism. |
| **Aronson, David R. (2007). *Evidence-Based Technical Analysis: Applying the Scientific Method and Statistical Inference to Trading Signals*. Wiley. ISBN 978-0-470-00874-4.** | **pp 336-339** | Dedicated section "Walk-Forward Testing" citing Pardo as the primary source: *"walk-forward testing. It is described in Pardo,[35] De La Maza,[36] Katz and McCormick,[37] and Kaufman.[38]"* — extracted verbatim from `resources/Evidence_Based_Technical_Analysis_Aronson.pdf` p338. Aronson's bibliography at p515 footnote 10 gives the full Pardo citation: *"R. Pardo, Design, Testing and Optimization of Trading Systems (New York: John Wiley & Sons, 1992)"* — this is Pardo's 1st edition (1992), later revised as the 2008 2nd edition we have front matter for. |
| **Aronson (2007). *Evidence-Based Technical Analysis*. Wiley.** | **p339** | *"[T]he decision about how to apportion the data between the in-sample and out-of-sample subsets is arbitrary. **There is no theory that suggests what fraction of the data should be assigned to training and testing.** Results can be very sensitive to these choices. **Often it's a seat-of-the-pants call.**"* — extracted verbatim. **This is the critical institutional-grounding statement:** a rigorous academic-practitioner source explicitly stating that walk-forward partition thresholds have no theoretical basis. Aronson is a Wiley-published ex-trader whose book is widely cited in quant TA academic literature and Chartered Market Technician (CMT) programs. |

#### Negative findings (local PDFs checked, no WFE threshold prescription)

- `resources/Robert Carver - Systematic Trading.pdf` (326pp) — 3 walk-forward mentions at pp 73, 296, 325 (verbatim checked). Carver prescribes walk-forward as a required back-testing practice but gives no numeric threshold: *"make sure it is expanding out of sample or equivalently walk forward. Also it should allow you to fit across multiple instruments and include conservative cost estimates."* (p296)
- `resources/Building_Reliable_Trading_Systems.pdf` (Fitschen, 290pp) — no "walk-forward efficiency" or "WFE" hits in body.
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` (45pp, bibliography excerpt only — not main text) — no chapter content. The "Walk Forward" hit is a bibliography entry for Żbikowski (2015), not a definition. See § 9.5 for what to acquire.
- `resources/false-strategy-lopez.pdf`, `resources/deflated-sharpe.pdf`, `resources/Pseudo-mathematics-and-financial-charlatanism.pdf`, `resources/backtesting_dukepeople_liu.pdf`, `resources/man_overfitting_2015.pdf`, `resources/Quantitative_Trading_Chan_2008.pdf`, `resources/Two_Million_Trading_Strategies_FDR.pdf`, `resources/real_time_strategy_monitoring_cusum.pdf` — no "walk-forward efficiency" or "WFE" hits. The modern statistical literature (Bailey/Lopez de Prado, Harvey/Liu) uses DSR (Deflated Sharpe Ratio), PBO (Probability of Backtest Overfitting), and combinatorial purged CV as primary gates, not a WFE ratio.

#### Web search (supplementary — read-only, no primary-source text acquired)

Web search on 2026-04-07 for Pardo's exact WFE definition and threshold returned only (a) Wikipedia's `Walk_forward_optimization` article, which cites **Pardo 2008 2nd ed pp 237-262** and **Pardo 1992 1st ed pp 108-119** as the primary source but quotes neither, and (b) practitioner blog posts (QuantInsti, AlgoTrading101, Unger Academy, Surmount, FXNX) which repeat the 0.5 convention without traceable primary citations. Google Books preview of the Pardo 2008 edition does not include Chapter 11 body pages. **No web source provided verbatim extracts of Pardo's definition or threshold.**

#### Conclusion — grounding is PARTIAL but defensible

The "WFE ≥ 0.5" threshold is:
1. **Not directly verifiable against Pardo's primary source** (our local PDF is front-matter only; web previews don't include the relevant pages).
2. **Locally grounded via Aronson's explicit statement** that the partition decision is "arbitrary" and has "no theory" — a rigorous academic-practitioner source explicitly demoting it from theoretical principle to practitioner convention.
3. **Locally grounded via Chan's practitioner statement** that live Sharpe of "half" the backtest value is "what most traders would be happy with" — the same 0.5 conservative lower-bound conceptually, in a Wiley-published source.
4. **Empirically irrelevant to the current portfolio** — all 124 active strategies are comfortably above it (min = 0.5141).

The honest framing: **the 0.5 threshold is a conservative practitioner lower-bound, not a theoretically justified number.** Aronson is the authoritative local citation for that framing. Enforcing it as a hard gate is defensible as a future-facing sanity check; treating it as theologically correct would misrepresent the literature. The spec's decision (§ 9.3) and the implementation contract (§ 9.4) reflect this posture by requiring the gate AND the explicit grounding caveat in both code comments and `RESEARCH_RULES.md:59`.

### 9.3 Decision — Modified Option A

**Enforce `wfe >= 0.5` as a hard gate in `walkforward.py` and `strategy_validator.py`, documented with an explicit literature-grounding caveat.**

Rationale:

1. **Zero cost.** No current strategy fails the gate (min wfe = 0.5141, all 124 ≥ 0.5). No retirements. No re-validation. No lost expected value. Option A's original "cost of consistency" argument is moot.
2. **Closes a real silent failure.** `RESEARCH_RULES.md:59` states a standard the code does not enforce — the definition of a silent failure under `.claude/rules/institutional-rigor.md`. A gate that matches the doc eliminates the gap.
3. **Future-proof.** Any future strategy with `wfe < 0.5` gets blocked at promotion time. Researchers cannot inadvertently introduce the same class of doc/code divergence.
4. **Additive, not replacing.** The 4-condition load-bearing gate stays intact. WFE becomes a 5th condition. This respects the fact that the existing gate is the evidence-based rule and WFE is a conservative single-number sanity check.
5. **Honest about grounding.** The 0.5 threshold gets an explicit caveat in both the code and `RESEARCH_RULES.md:59` noting: (a) the Pardo primary source (2008 Ch 11 pp 237-262) is not locally extractable, (b) Aronson p339 explicitly states "there is no theory" for the partition threshold, (c) Chan p25 is the secondary local source for the 0.5 halving heuristic applied to Sharpe, and (d) the value is practitioner-conventional, not theoretically derived. Future researchers will see the caveat and can propose a different threshold if they acquire Pardo's chapter body or better peer-reviewed literature. This prevents cargo-culting the number.

### 9.4 Implementation contract (for Phase E)

When Phase E is executed, the implementation MUST:

1. Add a 5th condition `wfe is not None and wfe >= 0.5` to `trading_app/walkforward.py` pass rule (lines 91-96). The condition is fail-closed: if `wfe` cannot be computed (e.g., all windows have `is_exp_r` too small), the strategy fails the gate.
2. The `WalkForwardResult.rejection_reason` must use a specific string — suggested: `"WFE below threshold: {wfe:.4f} < 0.5"` — so audit logs distinguish this failure class.
3. Add a comment block in `walkforward.py` above the new condition with the literature-grounding caveat from § 9.2, citing this spec section as authority.
4. Verify `trading_app/strategy_validator.py` consumes `WalkForwardResult.passed` as the canonical gate signal — if it re-implements any of the four existing conditions, the re-encoding must be deleted (canonical delegation principle from `integrity-guardian.md`).
5. Update `RESEARCH_RULES.md:59` to: (a) mark the gate enforced in code with a file:line reference, (b) include the literature-grounding caveat, (c) link to this spec section.
6. Add a drift check in `pipeline/check_drift.py` asserting every `validated_setups` row with `status='active'` has `wfe >= 0.5`. Paired negative test in `tests/test_pipeline/test_check_drift.py` that injects a row with `wfe=0.3` and confirms detection.
7. Add positive + boundary + negative tests in `tests/test_trading_app/test_walkforward.py`:
   - Strategy that passes all 4 existing conditions but has `wfe=0.45` → `passed=False`, `rejection_reason` starts with `"WFE below threshold"`
   - Strategy that passes all 4 existing conditions and has `wfe=0.50` exactly → `passed=True`
   - Strategy where `wfe is None` (pathological windows) → `passed=False`
8. Run the full `validated_setups` population through a dry-run verification: `SELECT COUNT(*) FROM validated_setups WHERE status='active' AND (wfe IS NULL OR wfe < 0.5)` must return **0** before and after the gate change (proves zero empirical impact).

### 9.5 Literature acquisition backlog

The local grounding in § 9.2 is partial. These sources would strengthen the grounding if acquired, but **none of them are blocking** the decision in § 9.3 or the implementation contract in § 9.4. The acquisition list is ordered by marginal value:

1. **Pardo, R. E. (2008). *The Evaluation and Optimization of Trading Strategies* (2nd ed.). Wiley. ISBN 978-0-470-12801-5. Chapters 11–12 (pages 237–280).**
   - Why: this is the canonical primary source for Walk-Forward Analysis and Walk-Forward Efficiency. Would replace the Aronson-secondary and Chan-adjacent citations with the actual Pardo definition and (possibly) threshold justification.
   - What to acquire: either (a) the Wiley eBook (~$120 USD, full Ch 11-12 accessible), (b) a used print copy via AbeBooks / eBay / university library, or (c) university-library digital access via `Wiley Online Library` (`onlinelibrary.wiley.com/doi/book/10.1002/9781119196969`).
   - What to verify in the acquired text: (i) Pardo's exact formula for WFE — is it OOS_ExpR / IS_ExpR, OOS_profit / IS_profit, or something else? (ii) Does Pardo recommend a specific numeric threshold and if so, does he justify 0.5 with an argument? (iii) Does Pardo address the mean-of-ratios pathology that our `walkforward.py:297-299` already corrected for?
   - Expected finding: Pardo probably gives 0.5 as a practitioner heuristic without theoretical derivation, which would *confirm* Aronson's "no theory" characterization rather than overturn it. If Pardo actually derives the threshold, the local grounding table in § 9.2 should be updated with the Pardo quote and the Aronson "no theory" framing softened.

2. **Pardo, R. E. (1992). *Design, Testing, and Optimization of Trading Systems* (1st ed.). Wiley. Pages 108–119.**
   - Why: Aronson p515 footnote 10 cites this as the canonical walk-forward source. Wikipedia also cites it as pp 108-119. Older edition, possibly easier to find in used bookstores or academic archives. Lower priority than the 2008 edition because the 2008 edition is the one expanded and refined.
   - What to acquire: used book market, academic library interlibrary loan.

3. **Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN 978-1-119-48208-2. Chapters 11 ("The Dangers of Backtesting"), 12 ("Backtesting through Cross-Validation"), and 13 ("Backtesting on Synthetic Data").**
   - Why: the modern rigorous alternative to walk-forward. Chapter 12 introduces Combinatorial Purged Cross-Validation (CPCV), which Lopez de Prado argues is strictly superior to walk-forward because it generates multiple backtest paths from the same data while respecting temporal ordering and purging label leakage. This is important for the `validation_standards` section of this spec (§ 7) and for deciding whether to eventually replace walk-forward entirely in CONFIRMATION mode (§ 2.2).
   - What to acquire: Wiley eBook, Amazon Kindle, university library.
   - What to verify: (i) whether Lopez de Prado explicitly criticizes Pardo's WFE threshold, (ii) whether CPCV provides a replacement metric with stronger theoretical backing, (iii) the PBO formula and its threshold (commonly `PBO < 0.5` in the Bailey-Lopez de Prado paper series).

4. **Kirkpatrick, C. D., & Dahlquist, J. R. (2010). *Technical Analysis: The Complete Resource for Financial Market Technicians*. FT Press. Page 548.**
   - Why: Wikipedia cites this page as a walk-forward reference. Supplementary confirmation of the practitioner convention.
   - What to acquire: Pearson eBook, print.
   - Low priority — likely just a restatement of Pardo.

5. **Pardo & Company original corpus** (`pardo.space/bob-pardo`) — Pardo himself runs a consulting firm and may have published white papers or client documents that define WFE without the book's copyright restriction. Worth checking for publicly available materials.

**What this backlog does NOT include:**
- Bailey, Borwein, Lopez de Prado, Zhu (2014) "Pseudo-mathematics and financial charlatanism" — **already in local resources** (`resources/Pseudo-mathematics-and-financial-charlatanism.pdf`, 32pp). Does not discuss walk-forward specifically but is relevant to the broader statistical-grounding context. Uses in-sample/out-of-sample terminology once at p3.
- Bailey, Lopez de Prado (2014) "The Deflated Sharpe Ratio" — **already in local resources** (`resources/deflated-sharpe.pdf`, 22pp). Does not discuss WFE; uses DSR as the primary gate metric.
- Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns" — cited in Lopez de Prado MLAM bibliography, referenced in this project's FDR methodology. Not directly relevant to WFE threshold.

### 9.6 What this decision does NOT do

- Does not prescribe a WFE threshold for any mode other than `CONFIRMATION` (§ 2.2). DISCOVERY and DEPLOYMENT_ANALYTICS do not run `walkforward.py`.
- Does not retroactively retire any current strategies. The empirical state confirms no retirements are needed.
- Does not claim 0.5 is theoretically optimal. It is a conservative lower-bound justified by (a) practitioner-literature convention, (b) existing project doc precedent, and (c) current strategies all comfortably above it.
- Does not replace the 4-condition load-bearing gate. WFE is additive.
- Does not address the *mean-of-ratios WFE pathology* documented in `walkforward.py:297-299` — the trade-weighted formulation already fixed that. Any future researcher who proposes a new WFE formulation must preserve the trade-weighting.
- Does not extend to real-time monitoring. A live strategy that drifts below WFE 0.5 post-deployment is handled by CUSUM / regime tracking, not by this promotion-time gate.

---

## 10. Migration Plan

Phased. Each phase is independently shippable and independently revertable. Phases are gated on completion of the previous phase.

### Phase A — Doctrine (this spec)

- This file exists and is reviewed.
- WFE decision (§ 9) is made.
- `RESEARCH_RULES.md` § Discovery Layer Discipline is rewritten to use the BASE_TRUTH / COMPUTED_FACTS / METADATA hierarchy and to reference this spec as authority on mode boundaries.
- `.claude/rules/research-truth-protocol.md` is rewritten: the Validated Universe Rule moves out of the truth protocol and into a new "deployment_analytics_scope" rule, scoped to DEPLOYMENT_ANALYTICS only.
- `CLAUDE.md` adds a cross-link to this spec under "Document Authority".
- **Cost:** docs only. Zero schema changes. Zero code changes. Trivially revertable.

### Phase B — Snapshot infrastructure

- Add `dataset_snapshots` table to `pipeline/init_db.py`.
- Add `pipeline/snapshot.py` with `freeze_current_state()` and `verify_snapshot()` helpers.
- Add `python -m pipeline.snapshot_freeze` CLI wrapper supporting `--type fingerprint|frozen`.
- Add `snapshots/` directory to `.gitignore` (frozen artifacts must not be committed to git).
- Add `check_drift.py` rule: every research run logged in the trace system must reference a snapshot_id.
- Backfill is not required for Phase B. The snapshot mechanism starts empty and is populated going forward.
- **Cost:** one new table, one new module, one new CLI, one .gitignore entry.

### Phase C — Research tracking (three tables)

- Add `research_hypotheses`, `research_runs`, `research_run_tests` tables to `pipeline/init_db.py`.
- Add the `claims` VIEW alongside.
- Add `pipeline/hypothesis_ledger.py` with `register_hypothesis()`, `start_run()`, `record_test()`, `close_run()` helpers.
- One-shot migration script: scan memory files for tested-feature documentation and create ledger rows with `provenance='LEGACY_PRE_2026_04_07'` (best effort, no synthetic per-test rows).
- `check_drift.py` rule: open `research_runs` rows older than 30 days are reported (not failed) so stale runs are visible.
- `check_drift.py` rule: `research_runs.tested_count` MUST equal COUNT(`research_run_tests` rows for that run_id).
- `check_drift.py` rule: `research_runs.k_for_fdr` MUST equal `research_runs.tested_count`.
- **Cost:** three new tables, one VIEW, one new module, one one-shot backfill script, three drift rules.

### Phase D — Promoted-layer schema patch

- Add the six new columns to `validated_setups` (§ 8.1).
- Backfill `provenance='LEGACY_PRE_2026_04_07'` for all existing 488 rows.
- Update `strategy_validator.py` promotion path to enforce write-time invariants for `provenance='NATIVE'`.
- Update `check_drift.py` to fail if any `provenance='NATIVE'` row has NULL on any of the new fields.
- Update `check_drift.py` to fail if any `validated_setups.run_id` references a `research_runs` row with `mode != 'CONFIRMATION'` or `result != 'CONFIRMATION_PASS'`.
- **Cost:** schema migration, validator changes, drift check additions. Higher risk than B and C — touches the deployment-decision write path. Do this only after B and C are stable. **Coordinate with e2 terminal if it's still touching `trading_app/config.py`** — Phase D touches `strategy_validator.py` (different file), but coordinate timing.

### Phase E — WFE gate (Modified Option A, per § 9.3)

**Decision:** Enforce `wfe >= 0.5` as a hard gate in `walkforward.py` and `strategy_validator.py`, with explicit literature-grounding caveat (see § 9.2 and § 9.3).

**Blast radius (files touched):**

1. `trading_app/walkforward.py` — add 5th pass condition (§ 9.4 step 1), new rejection_reason format (§ 9.4 step 2), literature-grounding comment block (§ 9.4 step 3). ~15 LOC.
2. `trading_app/strategy_validator.py` — verify canonical delegation to `WalkForwardResult.passed`, delete any re-encoded gate logic if found (§ 9.4 step 4). Likely 0 LOC net if already canonical, possibly -20 LOC if a parallel re-implementation is discovered.
3. `pipeline/check_drift.py` — add drift check asserting `active` + `wfe >= 0.5` (§ 9.4 step 6). ~20 LOC.
4. `tests/test_trading_app/test_walkforward.py` — add 3 new tests for the 5th condition (§ 9.4 step 7). ~40 LOC.
5. `tests/test_pipeline/test_check_drift.py` — add paired negative test for the new drift check (§ 9.4 step 6). ~15 LOC.
6. `RESEARCH_RULES.md:59` — rewrite the WFE standard line with file:line reference to walkforward.py, grounding caveat, pointer to § 9 of this spec (§ 9.4 step 5). ~5 LOC.
7. `docs/audits/edge_family_audit_2026-03-30.md` — if still referenced as an open finding, add a closeout note citing this spec section as resolution. ~3 LOC.

**Files NOT touched (canonical discipline):**

- `validated_setups` table schema — no schema change. The existing `wfe` column (populated at promotion time) is the canonical source.
- `validated_setups` rows — no DELETE, no UPDATE. Zero retirements (empirical state in § 9.1 confirms zero-cost).
- `trading_app/prop_profiles.py`, `trading_app/prop_portfolio.py`, `docs/runtime/lane_allocation.json` — lane allocation unchanged.
- `pipeline/asset_configs.py`, `pipeline/cost_model.py`, `pipeline/dst.py` — no instrument/cost/session changes.
- `trading_app/config.py`, `trading_app/eligibility/*` — no filter changes.
- ML code paths — ML is dead (`ml_institutional_audit_p1.md`), not touched.

**Coordination with concurrent stages:**

- **Resolved:** The `canonical-filter-self-description` stage that previously held `pipeline/check_drift.py` in its `scope_lock` **closed at commit `cd9b5e9`** on 2026-04-07. Drift check #85 (filter self-description coverage) was added at `32356b8`. `pipeline/check_drift.py` is now released — Phase E step 3 (the new WFE drift check, which will become #86 or later) is no longer blocked.
- **Still active:** The `e2-canonical-window-fix` stage (in `C:\Users\joshd\canompx3-e2-fix`) also adds drift checks to `pipeline/check_drift.py`, but on a separate branch/worktree. Coordinate merge order: the WFE drift check can land in either order relative to e2-fix as long as both stages rebase on each other's check additions. If e2-fix lands first, the WFE check is appended after its checks; if WFE lands first, e2-fix rebases on top.
- **Numbering advisory:** When Phase E adds the new check, run `grep -c "^Check [0-9]" pipeline/check_drift.py` (or equivalent) first to find the current highest check number and assign the next integer. Do not hard-code the number from this spec — it may drift.

**Verification gates:**

- Pre-change dry-run: `SELECT COUNT(*) FROM validated_setups WHERE status='active' AND (wfe IS NULL OR wfe < 0.5)` must return **0**.
- Post-change: same query must still return **0**, AND `python pipeline/check_drift.py` must pass with one additional check recorded relative to the count immediately before the Phase E commit (run `python pipeline/check_drift.py 2>&1 | tail -5` pre-change and post-change to compare; the pre-existing Check 57 parallel-process data drift, if still open, is unrelated and can be excluded from the delta calculation).
- Pytest: full `tests/test_trading_app/test_walkforward.py` green (existing + 3 new tests).
- Rejection_reason audit: grep `validated_setups.retirement_reason` for the new `"WFE_GATE_..."` pattern post-change — should return 0 rows (no historical retirements retroactively labeled).

**Cost:** ~95 LOC touched, ~60 LOC added. One gate condition addition, one doc rewrite, one drift check, three tests, one caveat block. Moderate-low risk. Zero empirical impact on current portfolio.

**Rollback:** Single `git revert` of the Phase E commit. No schema changes, no DB writes to revert.

### Phase F — Convert audits to write-time contracts (deferred)

- Lift the most load-bearing checks from `phase_5_database.py` and `audit_integrity.py` into write-time enforcement (DuckDB CHECK constraints, schema NOT NULL, or assertions in the relevant Python write paths).
- This is the most expensive phase and is intentionally deferred. The new structure from B–E will reveal which checks are actually load-bearing; that data drives Phase F.

### Phase G — Generalize purge/embargo tooling (Validation Standards § 7.4)

- Lift purge/embargo logic out of ML-specific code into `pipeline/purge_embargo.py` as a general utility.
- Add tests covering temporal feature studies, not just ML splits.
- Document in RESEARCH_RULES.md as a required tool for any feature whose computation window overlaps the prediction window.
- **Cost:** moderate. Mostly refactoring existing logic with a wider API.

### Phase H — Negative control / null permutation tooling (Validation Standards § 7.1, § 7.2)

- Add `research/null_control.py` with helpers to generate label-permuted and temporally-shuffled control runs.
- Add a research-script convention: every broad scan calls the helper to produce a parallel control run.
- Update the honest_summary block to include negative control results.
- **Cost:** moderate. New tooling, no schema changes.

---

## 11. Conflict Resolution Rules

When this spec conflicts with another document, this spec wins on:

- The definition of DISCOVERY / CONFIRMATION / DEPLOYMENT_ANALYTICS / OPERATIONS modes
- The classification of any table into BASE_TRUTH / COMPUTED_FACTS / METADATA
- The provenance fields required on `validated_setups` for NATIVE rows
- The structure of the `research_hypotheses`, `research_runs`, `research_run_tests`, `dataset_snapshots` tables
- The Reporting Contract (§ 6) — every field is mandatory

This spec defers to other documents on:

- All trading logic (entry models, filters, sessions, cost specs) → `TRADING_RULES.md`
- All statistical methodology beyond the K accounting and ledger discipline defined here → `RESEARCH_RULES.md`
- All session times, instrument configs, cost specs → canonical Python sources (`pipeline.dst.SESSION_CATALOG`, `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, `pipeline.cost_model.COST_SPECS`)
- All deployment / portfolio routing → `trading_app/prop_profiles.py`
- All filter introspection / `describe()` semantics → `trading_app/config.py` (filter classes own their own introspection — this spec only requires DEPLOYMENT_ANALYTICS to use the canonical method, not specify it)

This spec does NOT touch:

- The cost model (`pipeline/cost_model.py`)
- DST and session resolution (`pipeline/dst.py`)
- Live execution paths (`trading_app/live/`)
- Filter class definitions (`trading_app/config.py`) — owned by the filter introspection workstream
- The fail-closed pipeline rules in `CLAUDE.md`

---

## 12. Worked Examples

### 12.1 A DISCOVERY run

**Question:** Does the previous day's high-to-low range predict ORB outcomes in the MNQ NYSE_OPEN session?

```python
from pipeline.snapshot import freeze_current_state
from pipeline.hypothesis_ledger import register_hypothesis, start_run, record_test, close_run

# 1. Freeze a fingerprint snapshot
snapshot_id = freeze_current_state(
    snapshot_type="FINGERPRINT",
    instruments=["MNQ"],
    notes="prev_day_range hypothesis",
)
# → "2026-04-07-MNQ-abc1234"

# 2. Register the hypothesis (theory and mechanism are mandatory)
hypothesis_id = register_hypothesis(
    title="prev_day_range predicts MNQ NYSE_OPEN ORB outcomes",
    theory="Larger prior-day ranges may indicate elevated overnight vol that carries into NYSE open.",
    mechanism="Volatility regime persistence across sessions: a high-vol day primes inventory imbalance that NYSE participants must work through.",
    mode="DISCOVERY",
    target_universe={"instruments": ["MNQ"], "sessions": ["NYSE_OPEN"]},
    allowed_params={"orb_minutes": [5, 15, 30], "rr_target": [0.75, 1.0, 1.5, 2.0, 2.5]},
    forbidden_params={"rr_target": "no values outside the listed set; no continuous optimization"},
    holdout_policy="OOS_2026",
)
# → "2026-04-07-mnq-prevhigh-001"

# 3. Open a run against the snapshot
run_id = start_run(
    hypothesis_id=hypothesis_id,
    snapshot_id=snapshot_id,
    mode="DISCOVERY",
    script_path="research/scan_prev_day_range.py",
)

# 4. Execute the scan. For EVERY test (15 cells = 3 orb_minutes × 5 rr_target):
for cell in scan_cells:
    raw_p, adj_p, n, expr, oos = run_test(cell)
    record_test(
        run_id=run_id,
        family="PREVHIGH_DIST",
        instrument="MNQ", session="NYSE_OPEN",
        orb_minutes=cell.orb_minutes,
        rr_target=cell.rr_target,
        n=n, raw_p=raw_p, adj_p=adj_p, effect_size=expr, oos_metric=oos,
        is_survivor=(adj_p < 0.05 and expr > 0),
    )

# 5. Close the run with full disclosure
close_run(
    run_id=run_id,
    tested_count=15,
    reported_count=4,         # number of survivors
    k_for_fdr=15,             # MUST equal tested_count
    result="POSITIVE_DISCOVERY",  # or "DEAD" if nothing survived
    p_value_observed=0.012,
)
```

The script's stdout includes the honest_summary block from § 6. The 4 survivors are now `claims` (via the VIEW) but are NOT yet in `validated_setups` — that requires a separate CONFIRMATION run.

### 12.2 A CONFIRMATION run

**Question:** A DISCOVERY run found a candidate `break_speed_60s_lt_30 + MNQ + CME_PRECLOSE`. Does this survive walk-forward, OOS, and FDR with K from the originating discovery?

```python
# 1. Freeze a FROZEN snapshot — promotion is on the table, so we need an artifact
snapshot_id = freeze_current_state(
    snapshot_type="FROZEN",  # <-- physical artifact created
    instruments=["MNQ"],
    notes="break_speed CME_PRECLOSE confirmation",
)

# 2. Register the CONFIRMATION hypothesis, linked to the parent DISCOVERY
hypothesis_id = register_hypothesis(
    title="CONFIRM: break_speed_60s_lt_30 on MNQ CME_PRECLOSE",
    theory="...",
    mechanism="...",
    mode="CONFIRMATION",
    parent_hypothesis_id="2026-04-01-mnq-breakspd-007",  # the DISCOVERY hypothesis
    holdout_policy="OOS_2026",   # MUST match parent
    target_universe={"instruments": ["MNQ"], "sessions": ["CME_PRECLOSE"]},
    allowed_params={"break_speed_threshold": [30]},  # narrow, no exploration
    forbidden_params={"any other parameter sweep": "this is confirmation, not search"},
)

# 3. Open the run, execute walk-forward + OOS + FDR
run_id = start_run(...)
# ... per-test record ...
close_run(
    run_id=run_id,
    tested_count=12,    # 12 walk-forward windows
    reported_count=12,  # all reported, even failures
    k_for_fdr=12,
    result="CONFIRMATION_PASS",
    p_value_observed=0.003,
)

# 4. The strategy_validator now sees a CONFIRMATION_PASS run with a FROZEN snapshot.
#    It can promote to validated_setups with run_id, snapshot_id, git_sha,
#    raw_p_value, n_trials_at_discovery, and provenance='NATIVE' all populated.
```

### 12.3 A DEPLOYMENT_ANALYTICS run

**Question:** Of the 5 deployed lanes on `topstep_50k_mnq_auto`, which had a losing week last week?

```python
# No snapshot needed (this is operational, not research).
# May join validated_setups, prop_profiles, paper_trades, orb_outcomes freely.
# May call filter.describe(row) for filter introspection.
# Output is a report. NO writes to validated_setups, NO new ledger row required.
```

### 12.4 What happens if a researcher tries to skip the rules

| Attempted shortcut | Caught how |
|---|---|
| DISCOVERY query joins `validated_setups` | The hypothesis row is `mode='DISCOVERY'`; the trace system flags METADATA reads in DISCOVERY-mode runs. |
| CONFIRMATION run promotes to `validated_setups` without a FROZEN snapshot | Write-time invariant fails closed: snapshot_type must be FROZEN for promotion. |
| Research script forgets to register a hypothesis | The script writes to `experimental_strategies` without a `run_id`. Drift check reports orphans. |
| Researcher runs 30 variations of a dead idea over 5 sessions | Ledger has 30 hypothesis rows. K accounting can include all 30 across sessions. |
| Researcher reports K=30 against 480 actual tests | `research_runs.k_for_fdr` MUST equal `tested_count`. Constraint fails. |
| Researcher reports survivors only and hides failures | `research_run_tests` row count won't match `tested_count`. Drift check fails. |
| OPERATIONS notices a pattern and starts trading it | OPERATIONS has zero authority for research truth. Anything noticed must be promoted to a DISCOVERY hypothesis with proper K. |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **BASE_TRUTH** | The single layer of raw market data: `bars_1m`. |
| **COMPUTED_FACTS** | Tables that are deterministic transforms of BASE_TRUTH at a known git_sha. Defeasible by recomputation. |
| **METADATA** | Records of decisions, claims, or operational state. Not evidence. |
| **DISCOVERY** | Research mode for finding new edges. Forbidden from joining METADATA. |
| **CONFIRMATION** | Research mode for testing a discovered candidate against OOS / WF / FDR. Forbidden from joining METADATA. |
| **DEPLOYMENT_ANALYTICS** | Operational analytics on the deployed portfolio. Read-only. May join METADATA. May not generate new edge claims. |
| **OPERATIONS** | Live trading and execution. Zero research authority. |
| **FINGERPRINT snapshot** | Lightweight snapshot recording state identity by hash. No file copy. |
| **FROZEN snapshot** | Snapshot with a physical gold.db artifact and SHA256. Required for promotion. |
| **snapshot_id** | A human-readable identifier for the state of gold.db at a moment, with a fingerprint hash for verification. |
| **hypothesis_id** | A pre-registered row in `research_hypotheses` recording theory, mechanism, scope, K plan, and holdout policy. |
| **run_id** | A row in `research_runs` recording one execution of one hypothesis against one snapshot at one git_sha. |
| **claim** | A row in the `claims` VIEW — a survivor of a `research_run_tests` query joined to its hypothesis and snapshot. |
| **provenance** | A label on `validated_setups` rows: `NATIVE` (promoted under the new rules) vs `LEGACY_PRE_2026_04_07` vs `RECONSTRUCTED`. |
| **WFE** | Walk-forward efficiency: ratio of OOS ExpR to IS ExpR. Currently informational; gate decision pending in § 9. |
| **k_for_fdr** | The number of tests counted in BH FDR correction. MUST equal `tested_count` on the run. |
| **tested_count vs reported_count** | The visible measure of selection bias — every researcher reports both. Hidden survivors are forbidden. |

---

## 14. What This Spec Deliberately Does Not Address

- **Schema versioning beyond what `init_db.py` already does.** Not in scope.
- **Live execution data flow.** `paper_trades` is classified as METADATA for research purposes; live execution rules live elsewhere.
- **Backtest engine implementation.** Outcome builders and their correctness are governed by the existing pipeline rules.
- **Multi-user / concurrent-write coordination.** The project is single-user; the existing rule "never run two write processes against the same DuckDB file" suffices.
- **Hypothesis text quality.** A bad hypothesis is still a registered hypothesis. The ledger records what was tested; it does not judge whether it should have been tested.
- **Whether any specific feature in the memory backlog should be retested.** The ledger backfill makes the backlog visible; deciding what to revisit is a separate task.
- **Filter introspection (`filter.describe()`) implementation details.** Owned by the filter introspection workstream in `trading_app/config.py`. This spec only requires DEPLOYMENT_ANALYTICS to use the canonical method.

---

## 15. Open Questions for Review

1. ~~**WFE policy (§ 9).** Option A or Option B. Cannot merge this spec without deciding.~~ **RESOLVED 2026-04-07** — Modified Option A adopted. See § 9.3. Zero-cost (no current strategies fail the gate; min wfe = 0.5141 across 124 active). Phase E implementation is unblocked (canonical-filter stage closed at `cd9b5e9`) but not yet scheduled — awaiting user authorization to execute the § 9.4 implementation contract.
2. **Snapshot scope.** Per-instrument snapshots are allowed; should there be a "promotion-ready snapshot must cover all active instruments" rule? Argument for: consistency across portfolios. Argument against: unnecessary work for single-instrument research.
3. **Hypothesis ledger backfill aggressiveness.** Best-effort migration of memory files vs. manual review of every backfilled row. Current draft is best-effort.
4. **Frozen snapshot retention.** How long to keep physical artifacts? Suggest: keep all FROZEN snapshots that have at least one `validated_setups` row pointing at them; eligible for purge if ALL referencing strategies are retired.
5. **Phase F timing.** The audit-to-contract conversion is deferred. When does it become urgent? Suggest: after first month of running with the new structure, when actual failure modes are visible.
6. **Phase G urgency.** Purge/embargo tooling generalization — does any current research need it immediately, or can it wait?
7. **Negative control / null permutation tooling.** Phase H. Required for new broad scans, but how much retroactive coverage of existing dead/alive features is needed?
8. **Any tables I missed in the classification (§ 3).** The grep covered `pipeline/init_db.py` and `trading_app/db_manager.py`. If new tables exist elsewhere, they should be added before merge.

---

**End of spec.** § 9 WFE decision RESOLVED 2026-04-07 (Modified Option A, zero-cost, literature-grounding caveat). Phase A completion pending the doc cross-link updates listed in § 10. Phase E implementation is unblocked (canonical-filter stage closed at `cd9b5e9`) but not yet scheduled — awaiting user authorization to execute the § 9.4 implementation contract.
