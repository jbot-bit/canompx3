---
slug: evidence-pack-generator
classification: DESIGN
mode: DESIGN
created: 2026-04-28
updated: 2026-04-28
authority: derivative tooling design — composes existing canonical artefacts; not authoritative for any decision
governs:
  - scripts/tools/build_evidence_pack.py (PR1)
  - scripts/tools/evidence_pack/* (PR1)
  - tests/test_scripts/test_build_evidence_pack.py (PR1)
locked_constraints:
  - no edits to pipeline/
  - no edits to trading_app/
  - no schema changes
  - no new external dependencies (stdlib + duckdb + pyyaml only)
  - no Parquet extracts in PR1
  - no drift check in PR1
  - text-only outputs in PR1
  - generator is derivative; not authoritative for decision-ledger entries
---

# Evidence-Pack Generator — Design

## 0. Purpose (one paragraph)

A deterministic CLI tool that, given one pre-registered hypothesis slug or one
`validated_setups` identity, emits a single audit-grade evidence pack composed
from the artefacts that already exist in this repo: the pre-reg YAML, the
result MD, the validated_setups row, the canonical lit extracts, and the
canonical computation modules (`trading_app/dsr.py`, `trading_app/pbo.py`,
`trading_app/spa_test.py`, `trading_app/strategy_validator._evaluate_criterion_8_oos`).
The pack is text-only, machine-readable, and reproducible by SQL text + SHA-256
hash + rerun command. It does not summarise, narrate, or interpret. It reports.

## 1. Scope and non-scope

**In scope (PR1):**
- A new tool at `scripts/tools/build_evidence_pack.py` (CLI entry point).
- A small helper package at `scripts/tools/evidence_pack/` with three modules:
  `manifest.py`, `gates.py`, `renderers.py`.
- Two Markdown templates under `scripts/tools/evidence_pack/templates/`.
- Tests at `tests/test_scripts/test_build_evidence_pack.py`.
- Light additive edits: two paragraphs in `docs/specs/research_modes_and_lineage.md`.
  No `.gitignore` edit needed — `reports/` is already covered at L134 by the
  umbrella rule, so `reports/evidence_packs/` inherits coverage. Verified
  2026-04-28 during implementation.

**Out of scope (PR1):**
- Any edit to `pipeline/`.
- Any edit to `trading_app/`.
- Any schema change.
- Any new external dependency.
- Parquet extracts of canonical query results.
- Drift check #122 (template-placeholder integrity).
- HTML rendering (the renderer module ships with the Markdown path only;
  the `--html` CLI flag is reserved but inert in PR1).
- Marimo viewer.
- Any LLM call.
- Any narrative or summary text.

## 2. Authority routing

This tool sits above the codebase's authority chain; it consumes truth from
canonical sources and emits a derivative artefact. Specifically:

- `RESEARCH_RULES.md` § Discovery Layer Discipline — discovery-truth comes from
  canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). The pack
  inherits this rule: every "truth" field traces to a canonical layer or an
  authoritative module.
- `docs/institutional/pre_registered_criteria.md` — the twelve locked criteria
  define the pack's gate table one-to-one.
- `CLAUDE.md` § Architecture — pipeline → trading_app one-way. The tool sits
  in `scripts/tools/` and is allowed to import from both, but never the other
  way round.
- `CLAUDE.md` § Volatile Data Rule — sessions, costs, instruments, lanes,
  ORB-window timing all come from their canonical Python sources, never from
  memory or docs.
- `.claude/rules/backtesting-methodology.md` § RULE 4.3 — bare-threshold labels
  must carry per-cell p-values. The pack enforces this when reading result
  MDs that declare categorical labels.
- `memory/feedback_pooled_not_lane_specific.md` — pooled-finding lane misuse is
  banned. The pack hard-codes a `LANE_NOT_SUPPORTED_BY_POOLED` verdict path.

## 3. Failure-prevention design (lessons applied)

The pack's design is bottom-up from prior failures in this repo:

1. **NO-GO `claude-mem` (2026-04-27).** A sidecar store would create a second
   source of truth versus existing MDs/YAMLs. **Applied:** the pack is a
   compiler, never an authoritative writer. Its outputs go under
   `reports/evidence_packs/<slug>/` and are gitignored by default.
2. **`C:\db\gold.db` scratch-copy stale-data bug (Mar 2026).** Two DB paths gave
   inconsistent answers. **Applied:** the pack uses `pipeline.paths.GOLD_DB_PATH`
   exclusively, fingerprints the DB at run time (schema fingerprint already
   surfaced in result MDs — see `2026-04-27-sizing-substrate-diagnostic.md`),
   embeds the fingerprint in the manifest, and refuses to render on fingerprint
   drift without an explicit `--allow-fingerprint-drift` flag.
3. **Pooled vs lane-specific p (`feedback_pooled_not_lane_specific.md`).**
   **Applied:** if the result MD frontmatter declares `pooled_finding: true`
   and the slug requests a single lane outside the per-lane breakdown, the
   verdict is forced to `LANE_NOT_SUPPORTED_BY_POOLED`.
4. **E2 break-bar look-ahead contamination registry (2026-04-28).** Result MDs
   written against tainted commits are unsafe. **Applied:** the pack performs
   a sorted recursive glob for any registry file matching
   `docs/audit/**/*e2*lookahead*contamination*.md` (Python `pathlib.Path.rglob`
   does not guarantee deterministic order, so the matches are sorted before
   use). If at least one registry is found, the pack reads it and surfaces a
   top-of-card red band when the source result MD's commit is in any tainted
   list. If no registry is found, the contamination check is reported as
   `UNCOMPUTED` and the decision card carries an amber "registry missing"
   banner — never a silent pass. Verified 2026-04-28: the canonical registry
   exists on two unmerged research branches
   (`research/2026-04-28-phase-d-mes-europe-flow-pathway-b`,
   `research/mnq-unfiltered-high-rr-family`) at commit `96bba7a7` but is
   **not yet on `origin/main`**, so the tool must operate without assuming
   its presence.
5. **Bare-threshold labels without per-cell p (RULE 4.3).** **Applied:** the
   gate evaluator requires per-cell p-values when the result declares a
   categorical label like DRAG/BOOST/RECURRING_REGIME. Missing per-cell p
   downgrades the gate to `UNCOMPUTED`, never `PASS`.
6. **Performative self-review (`CLAUDE.md` 2-Pass Method).** **Applied:** every
   field in the manifest carries provenance (`value`, `source`, `as_of`,
   `git_sha`, `db_fingerprint`). Missing fields render as `UNCOMPUTED` or
   `UNSUPPORTED`, never silently dropped.

## 4. Approach considered and chosen

Three approaches were weighed:

**Approach A — Pure compiler (CHOSEN).** A single CLI tool plus a small helper
package. Reads existing YAMLs/MDs/DB rows, calls canonical computation
modules, writes a directory of artefacts. Zero new dependencies. Around 250-350
LOC plus templates. The renderer is plain Python with stdlib string
templates — less pretty than Quarto/Marimo, but the value is the manifest, not
the rendering.

**Approach B — Compiler + Soda contracts.** Approach A plus Soda for data
contracts. Adds a parallel data-quality framework that does not know about
temporal-validity gates. The repo's drift checks (#1-#121) already cover
schema-level contracts more rigorously than Soda would. **REJECTED.**

**Approach C — Marimo cockpit on top.** Approach A plus an interactive
notebook. Useful only for interactive auditors; the canonical `evidence-auditor`
agent runs headless. **DEFERRED to a possible follow-up PR.**

**Recommendation taken:** Approach A.

## 5. Pack contents (PR1, text-only)

Each pack lives at `reports/evidence_packs/<slug>/<run-iso8601>/` and contains
exactly four files:

1. `manifest.json` — full provenance shape (see § 6).
2. `decision_card.md` — one-page verdict, mirrors the existing
   `docs/audit/deploy_readiness/2026-04-15-sgp-momentum-deploy-readiness.md`
   format so auditors do not have to learn a new layout.
3. `gate_table.json` — twelve-criteria pass/fail/uncomputed/cross-check-only
   matrix with per-criterion source path and threshold.
4. `report.md` — full pack report, including SQL text + SHA-256 hash + rerun
   command for every query the manifest references.

Deferred to a follow-up PR: `query_outputs/*.parquet` extracts behind an
`--include-extracts` flag.

## 6. Manifest shape (informal — exact dataclass shape lives in code at PR1 time)

The manifest is a JSON document whose top-level keys are:

- `pack_version` — semver string for the manifest schema itself.
- `slug` — the candidate slug.
- `run_iso8601` — the timestamp the pack was generated.
- `git_sha` — the repo commit at generation time, from `pipeline.audit_log.get_git_sha`.
- `db_fingerprint` — schema fingerprint of the DB at generation time.
- `db_path` — the resolved `pipeline.paths.GOLD_DB_PATH`.
- `hypothesis` — block carrying the prereg path, the prereg's own
  `commit_sha` (from its frontmatter), and the K-budget, kill criteria, and
  mechanism citations from the prereg.
- `result` — block carrying the result MD path, the result's own git HEAD
  (parsed from the MD), the JSON twin path if present, the `pooled_finding`
  flag, the per-lane breakdown path, and the `flip_rate_pct` /
  `heterogeneity_ack` fields when the result is pooled.
- `validated_setups` — block (when applicable) carrying the
  `validation_run_id`, `promotion_git_sha`, `family_hash`, IS/OOS metrics, and
  the canonical traded-day window from
  `trading_app.validation_provenance.StrategyTradeWindow`.
- `tables_used` — list of canonical layers the pack queried (must be a subset
  of `bars_1m`, `daily_features`, `orb_outcomes`).
- `holdout_date` — `2026-01-01` from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`.
- `is_oos_split` — IS window, OOS window, and counts.
- `k_framings` — every framing the result reports: `K_global`, `K_family`,
  `K_lane`, `K_session`, `K_instrument`, `K_feature`. Missing ones render as
  `UNCOMPUTED`, never zero.
- `gates` — array of twelve `GateResult` objects, one per locked criterion.
- `queries` — array of `(label, sql_text, sql_sha256, rerun_command)` tuples.
- `contamination` — block carrying any contamination-registry hits against
  the result MD's commit. Shape: `{registry_paths: [sorted glob results],
  registry_status: PRESENT | MISSING, hits: [...], status: CLEAN | TAINTED |
  UNCOMPUTED}`. When `registry_status == MISSING` the `status` is forced to
  `UNCOMPUTED` and the decision card emits an amber banner — never `CLEAN`.
- `verdict` — one of `PASS`, `CONDITIONAL`, `KILL`, `INCOMPLETE_EVIDENCE`,
  `LANE_NOT_SUPPORTED_BY_POOLED`.
- `verdict_reasons` — array of plain-text reason strings; required when
  verdict is anything other than `PASS`.

Every value-bearing leaf field carries a provenance sibling: `<field>_source`
(file path or SQL hash), `<field>_as_of` (timestamp).

## 7. Twelve-criteria gate table

One pure helper per criterion, each returning a `GateResult` of `name`,
`status` (one of `PASS`, `FAIL`, `UNCOMPUTED`, `CROSS_CHECK_ONLY`), `value`,
`threshold`, `source`. The criteria mirror
`docs/institutional/pre_registered_criteria.md` exactly. DSR is wired as
`CROSS_CHECK_ONLY` per the criteria file's v2 amendment policy (N_eff
unresolved). Criterion 4 (Chordia t ≥ 3.79) is wired as a severity benchmark,
not a hard bar, per the same amendment.

## 8. Slug resolver — three input modes

The CLI accepts three mutually exclusive input modes:

- `--prereg <path-or-slug>` — a hypothesis YAML at
  `docs/audit/hypotheses/<date>-<slug>.yaml` (or just the slug; resolver
  globs by date prefix).
- `--validated-setup <identity>` — an identity tuple
  `(instrument, session, orb_minutes, entry_model, rr, filter, direction)`,
  resolved against the `validated_setups` table.
- `--experimental <row-id>` — an `experimental_strategies` row id.

The resolver canonicalises by SHA when present in the prereg frontmatter, not
by path. On path miss, falls back to filename glob and reports a `path_drift`
warning in the manifest (never silently substitutes).

## 9. Fail-closed gates and soft warnings

Before render, the pack runs five fail-closed checks plus one soft check. A
fail-closed failure leaves the manifest emitted but forces verdict
`INCOMPLETE_EVIDENCE` (or `LANE_NOT_SUPPORTED_BY_POOLED` for the pooled case)
and prints a top-of-card **red band** on `decision_card.md`. The soft check
emits an **amber banner** — verdict is unaffected, but the auditor is
warned the pack ran without that input.

Fail-closed (red band, verdict downgrade):

1. **Hypothesis missing.** Prereg path does not resolve.
2. **Holdout not declared.** Prereg lacks `holdout_date` field, or holdout
   field is not `2026-01-01` per `trading_app.holdout_policy`.
3. **Derived layers as truth.** Any field claims its source as
   `validated_setups`, `edge_families`, or `live_config` for a discovery-truth
   field. Allowed only as audit-trail breadcrumbs, never as truth.
4. **Manifest fingerprint drift.** Re-run produces a different DB fingerprint
   than a prior run for the same git_sha without `--allow-fingerprint-drift`.
5. **Pooled-finding lane misuse.** Result MD frontmatter has
   `pooled_finding: true` and the slug names a single lane outside the
   per-lane breakdown.

Soft warning (amber banner, no verdict change):

6. **Contamination registry missing.** Sorted recursive glob
   `docs/audit/**/*e2*lookahead*contamination*.md` returns no match. The
   `contamination.status` field is set to `UNCOMPUTED`, the manifest records
   `registry_status: MISSING`, and the card prints an amber banner naming the
   expected path and the two known unmerged-research branches that carry it.
   Reasoning: a registry-missing state is plausible and recoverable (the
   research branch may merge any day); it is not a defect of the candidate
   under review. Forcing `INCOMPLETE_EVIDENCE` here would block every pack
   until the unmerged research lands, which is the wrong dependency
   direction. When the registry is later promoted to `origin/main`, the
   amber banner disappears automatically with no code change.

## 10. CLI surface (PR1)

```
python scripts/tools/build_evidence_pack.py \
    --prereg <date-or-slug> | --validated-setup <identity> | --experimental <row-id> \
    [--out reports/evidence_packs/<slug>/] \
    [--allow-fingerprint-drift] \
    [--update-snapshots]
```

- `--out` defaults to `reports/evidence_packs/<slug>/<run-iso8601>/`.
- `--allow-fingerprint-drift` is the only escape hatch and prints a warning
  banner on the card.
- `--update-snapshots` is for the test fixture; not for production use.
- Reserved but inert in PR1: `--html`, `--include-extracts`.

## 11. Tests (PR1)

- **Unit:** every gate function tested against a synthetic manifest with each
  of `PASS`, `FAIL`, `UNCOMPUTED`, `CROSS_CHECK_ONLY` cases.
- **Slug resolver:** all three input modes tested with real fixtures.
- **Determinism:** two runs against the same git SHA + DB fingerprint produce
  byte-identical JSON manifest. Hash assertion in CI.
- **Fail-closed:** synthetic prereg without `holdout_date` produces
  `INCOMPLETE_EVIDENCE`.
- **Pooled-finding:** synthetic result with `pooled_finding: true,
  flip_rate_pct: 67.0` produces `LANE_NOT_SUPPORTED_BY_POOLED` for the flipped
  lanes.
- **DSR cross-check:** gate function returns `CROSS_CHECK_ONLY` for DSR.
- **Contamination band:** synthetic result MD whose commit is in the
  contamination registry produces a red band on the card.
- **Snapshot:** rendered Markdown for the
  `2026-04-27-sizing-substrate-prereg.yaml` fixture matches a committed golden
  file under `tests/test_scripts/fixtures/evidence_pack/`.

## 12. Failure modes and rollback

**Failure modes covered above (§ 3, § 9).**

**Rollback plan:** pure additive PR. `git revert` removes it. No schema
change, no canonical-layer change, no production touch. If the tool produces
incorrect packs in the wild, operators continue using the existing manual
workflow (open the prereg YAML + result MD + deploy-readiness MD by hand).
No live decision depends on this tool.

**Guardian prompts:** none required. Not `ENTRY_MODEL_GUARDIAN` (no entry-model
touch). Not `PIPELINE_DATA_GUARDIAN` (no pipeline data touch).

**Risk tier:** Low. Read-only. New file area. No canonical surface modified.

## 13. Deferred follow-up PRs (named, not committed)

- **PR2 (optional):** `--include-extracts` flag with DuckDB
  `COPY ... TO ... (FORMAT parquet)` per query, gated to `research/output/`
  parity.
- **PR3 (optional):** drift check #122 — assert Markdown templates contain
  every placeholder the manifest dataclass declares.
- **PR4 (optional):** `--html` rendering using stdlib `html` module. No new
  dependency.
- **PR5 (optional):** marimo notebook for interactive pack inspection.

Each is its own design conversation; none is bundled into PR1.

## 14. Approval state

This design is **APPROVED** by the user as of 2026-04-28 with the locked
constraints listed in the frontmatter. The next action is to write the Stage 1
file at `docs/runtime/stages/evidence-pack-generator-stage1.md` (companion to
this design doc) and stop. Implementation begins only when the user
explicitly green-lights it in a future session under the
`stage-gate-protocol.md` workflow.
