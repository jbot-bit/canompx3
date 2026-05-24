# Check 107 — Orphan `hypothesis_file_sha` Audit (no remediation)

- **Date:** 2026-05-17
- **Tool:** Claude Code (Opus 4.7)
- **Scope:** AUDIT-FIRST per HANDOFF.md "Next Session — Active" item. Enumerate the 11 orphaned `experimental_strategies.hypothesis_file_sha` values that Check 107 (`check_phase_4_sha_integrity`) flags every pre-commit run, classify each, and identify the root-cause commit class. **No DB mutation. No allowlist amendment. No hypothesis-file edits.** Remediation options are listed at the end; the decision is deferred to a follow-up stage.
- **Canonical authority:** `pipeline/check_drift.py:5998-6095` (`check_phase_4_sha_integrity`), `trading_app/holdout_policy.py::PHASE_4_1_SHIP_DATE = 2026-04-09 UTC`, `trading_app/hypothesis_loader.py::find_hypothesis_file_by_sha`.

## Verdict

**Root-cause class: re-stamped real discovery rows (not test-fixture leaks, not corruption).** All 11 orphan SHAs trace to legitimate post-`PHASE_4_1_SHIP_DATE` discovery runs whose source hypothesis YAML was later edited in-place (file content changed → SHA changed → previously stamped SHA in `experimental_strategies` became orphaned). The strategies themselves are real — 23 of the 115 affected rows survived to `validated_setups`.

The drift-check noise (11 violations on every commit) is therefore a **stale-pointer artifact**, not evidence of data corruption or audit-trail tampering. Check 107's docstring (`pipeline/check_drift.py:6005-6008`) anticipates this exact class as cause (b): "a rebase that dropped the hypothesis commit". The audit confirms cause (b), not (a) tampering or (c) test-fixture leak.

## Method

```python
import duckdb
from pipeline.paths import GOLD_DB_PATH
from trading_app.holdout_policy import PHASE_4_1_SHIP_DATE
from trading_app.hypothesis_loader import find_hypothesis_file_by_sha

con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
rows = con.execute('''
    SELECT DISTINCT hypothesis_file_sha
    FROM experimental_strategies
    WHERE hypothesis_file_sha IS NOT NULL
      AND created_at >= ?
''', [PHASE_4_1_SHIP_DATE]).fetchall()

orphans = [sha for (sha,) in rows if find_hypothesis_file_by_sha(sha) is None]
# orphans = 11 of 11 distinct post-ship stamped SHAs
```

**Total post-ship distinct stamped SHAs:** 11.
**Orphan count:** 11 / 11 = 100%.

The 100% rate is the decisive signal: individual file deletions would produce a mixed population (some resolve, some orphan). 11 of 11 unresolved means a **systemic class** is at work, not ad-hoc deletions. That class is identified below.

## Per-SHA enumeration

Sorted by earliest `created_at`. "VS hits" = number of distinct `strategy_id`s from that SHA that appear in `validated_setups` (i.e., reached the live truth layer).

| SHA (prefix) | Rows | First created | Instr | Entry | Filters | VS hits | Class |
|---|---|---|---|---|---|---|---|
| `ee87f887f271` | 8 | 2026-04-10 | GC | E2 | NO_FILTER, PDR_R080 | 1 | Re-stamped (April bump cluster) |
| `62e7631661f1` | 9 | 2026-04-10 | GC | E2 | COST_LT10, NO_FILTER, ORB_G5, ORB_G6 | 1 | Re-stamped (April bump cluster) |
| `e78ada335aae` | 20 | 2026-04-10 | GC | E2 | NO_FILTER, OVNRNG_50, PDR_R080 | 2 | Re-stamped (April bump cluster) |
| `7777e5f4f18d` | 3 | 2026-04-10 | GC | E2 | OVNRNG_25, OVNRNG_50 | 0 | Re-stamped (April bump cluster) |
| `29c87962c4bf` | 48 | 2026-04-11 | GC | E2 | ATR_P50, ATR_P70 | 8 | Re-stamped (April bump cluster) |
| `946e7ba29ef1` | 18 | 2026-04-11 | GC | E2 | OVNRNG_10 | 5 | Re-stamped (April bump cluster) |
| `5e37d9381c23` | 3 | 2026-04-11 | MNQ | E2 | GAP_R015 | 3 | Re-stamped (April bump cluster) |
| `d047546988dc` | 3 | 2026-04-11 | MNQ | E2 | ORB_G5_NOFRI | 3 | Re-stamped (April bump cluster) |
| `51c4d3e90d8f` | 1 | 2026-05-11 | MNQ | E2 | ATR_VEL_GE105 | 0 | Re-stamped (Amendment 3.3 migration) |
| `4a639daeeeb7` | 1 | 2026-05-11 | MNQ | E2 | ORB_VOL_16K | 0 | Re-stamped (Amendment 3.3 migration) |
| `f505cf7982c2` | 1 | 2026-05-11 | MNQ | E2 | ATR_P30 | 0 | Re-stamped (Amendment 3.3 migration) |

**Totals:** 115 affected rows across 11 distinct SHAs. 23 rows reached `validated_setups`. All rows are E2 (no other entry-model class). 8 SHAs are GC-instrument, 3 are MNQ. No malformed SHAs (no row failed the `isinstance(sha, str) and sha` guard at `pipeline/check_drift.py:6066`).

## Root-cause: two re-stamping events

### Class 1 — April timestamp-bump cluster (2026-04-10/11, 8 SHAs)

Two commits explicitly re-stamped GC-proxy hypothesis YAML to force fresh SHAs after upstream fixes:

- `dc3a53be` (2026-04-10 18:09 +1000) — "fix: bump hypothesis file timestamps (force new SHA after cost spec fix)"
- `d53b5bc4` (2026-04-10 18:30 +1000) — "fix: bump hypothesis timestamps (fresh SHA for re-run after DELETE fix)"

Each touched `2026-04-10-gc-proxy-broad-sweep.yaml`, `2026-04-10-gc-proxy-era-dependent.yaml`, `2026-04-10-gc-proxy-stable-sessions.yaml`. The April orphans were stamped during discovery runs **before** these bumps (timestamps 18:32 → 18:58 land just after `d53b5bc4` 18:30, suggesting an even earlier intermediate state on disk that's no longer reachable). The MNQ April orphans (`5e37d938`, `d047546988`) follow the same pattern — discovery stamped the pre-edit content; the file was then edited.

Note that not every GC orphan maps cleanly to one of the two named bump commits — some SHAs were captured between intermediate edits not recorded in the explicit "bump" commits (e.g., earlier same-day in-place edits). The classification stands at file-class level (these files have been edited since stamping), not at commit-pair level.

### Class 2 — Amendment 3.3 bulk migration (2026-05-17, 3 SHAs)

Commit `8ab4fe13` (2026-05-17, "Amendment 3.3: explicit metadata.theory_grant, fail-closed loader (#292)") migrated **203 hypothesis YAML files** under `docs/audit/hypotheses/` by adding the `metadata.theory_grant` field. Adding one line to a file changes its SHA. Every post-ship stamped SHA whose source file was migrated by Amendment 3.3 became orphaned in that commit.

The May 11 MNQ orphans (`51c4d3e9`, `4a639daee`, `f505cf798`) map to:
- `docs/audit/hypotheses/2026-05-11-llm-tokyo-open-atr-vel-ge105.yaml` — current SHA `78c0e376…`, orphan SHA `51c4d3e9…`
- (analogous files for the other two May 11 orphans — Amendment 3.3 modified them too)

These were the only May 11 stamped SHAs that survived the migration as orphans because all other post-ship-date discovery runs that stamped SHAs either touched files that Amendment 3.3 did not migrate (file already had `theory_grant: false`) or the rows aren't in scope of this check.

## Distribution of remediation outcomes by class

There is no test-fixture leak in the data. 0 of 11 orphans match Check 107 docstring cause (c) "test fixture leaked into gold.db". The taxonomy in the docstring is therefore incomplete — it should add cause (d): "hypothesis file was modified in place after discovery (legitimate edit, e.g., bulk doctrine migration)". The current docstring's framing ("tampering" / "deleted") is misleading for the dominant real-world class.

## Remediation options (no decision in this stage)

Three viable paths; user must pick before any follow-up stage executes. None are implemented here.

### Option A — Backfill `hypothesis_file_sha` to current on-disk SHA

For each of the 11 orphan SHAs, find the source hypothesis file via prior commit lookups (`git log -G "$old_sha"` on the YAML directory, plus filter+session matching from `strategy_id`), confirm the file was re-stamped not deleted, and `UPDATE experimental_strategies SET hypothesis_file_sha = <current SHA> WHERE hypothesis_file_sha = <orphan SHA>`. Drift-check noise drops to zero.

**Risk:** Loses the original SHA pointer, which was the audit trail to "what content did discovery actually see when it ran this row". For rows that reached `validated_setups`, this is the genuine concern — we are mutating evidence about a row's discovery provenance. Mitigation: store the orphan SHA in a new `hypothesis_file_sha_original` column and put the current SHA in the existing column, preserving both. Adds a schema change and a new drift check for parity.

### Option B — Add 11 orphan SHAs to a documented grandfather allowlist in Check 107

Extend `check_phase_4_sha_integrity` to accept an explicit allowlist (e.g., a YAML file at `docs/audit/check_107_grandfathered_orphans.yaml`) with each orphan SHA + the commit class that re-stamped it + the original source file path. Check 107 skips listed SHAs.

**Risk:** Grandfathering normalizes future occurrences. Every doctrine migration like Amendment 3.3 would generate new orphans; the allowlist would grow indefinitely. Mitigation: require allowlist entries to cite the commit hash that re-stamped the file, so it's evidence-grounded — and add a drift check that fails if the cited commit doesn't exist or doesn't touch the cited file.

### Option C — Refactor: re-stamp at SHA-edit time, not at discovery time

The root cause is that `hypothesis_file_sha` stores the **discovery-time content SHA**, but no mechanism updates downstream rows when the source file is legitimately edited. A migration-time hook could update `experimental_strategies.hypothesis_file_sha` whenever a hypothesis YAML's SHA changes via a tracked operation (e.g., `Amendment 3.x` migration). Drift-check semantics shift from "find me orphaned SHAs" to "find me SHAs the migration tracker doesn't know about".

**Risk:** Adds infrastructure (migration tracker, hook) for an arguably-minor drift-check noise problem. Mitigation: only pursue if Option B's allowlist grows past ~30 entries (signal that re-stamping is a regular operation, not a one-off).

## Recommendation (NOT a decision)

**Option B with the cited-commit constraint** is the lowest-risk path for the immediate problem (silence Check 107 noise) and preserves the original SHA as evidence. Option A's evidence-mutation risk is real for the 23 `validated_setups` rows. Option C is overkill for n=2 re-stamping events in 7 months.

A follow-up stage should:
1. Author `docs/audit/check_107_grandfathered_orphans.yaml` with 11 entries, each citing the re-stamping commit (`dc3a53be` / `d53b5bc4` / `8ab4fe13`) and the source YAML.
2. Extend `pipeline/check_drift.py::check_phase_4_sha_integrity` to skip allowlisted SHAs.
3. Add a sibling check verifying every allowlist entry's cited commit exists and touched the cited file.
4. Update Check 107 docstring to add cause (d).

Estimated effort: ~30-45min implementation + tests.

## 2026-05-24 Addendum — Local MNQ Orphan Set

## Scope

Codex re-ran Check 107 against the restored local `gold.db` in
`/home/joshd/canompx3` after the TopstepX/live-readiness setup repair. That DB
contained 21 additional orphaned `experimental_strategies.hypothesis_file_sha`
values. The same root-cause class applies: the DB rows preserve discovery-time
content SHAs, while the source YAML files were later edited in tracked commits.

No DB rows were mutated. The repair path remains the evidence-preserving
manifest approach: append entries to `docs/audit/check_107_sha_migrations.yaml`
so Check 107 can accept the orphan SHAs only when the sibling manifest-integrity
check proves:

- the current file exists,
- the current file SHA matches `current_sha`,
- the introducing commit contains file content hashing to `orphan_sha`,
- the migration commit exists and touched the file,
- this audit file exists.

## Outputs

The 2026-05-24 set maps to these source files:

| SHA prefix | Source file | Introducing commit |
|---|---|---|
| `c08e58cc6ddb` | `2026-04-09-mnq-rr10-individual.yaml` | `7aaf256c` |
| `430b8bb8ea7e` | `2026-04-11-mnq-o15-expansion.yaml` | `7aaf256c` |
| `fc5ab497b54c` | `2026-04-11-mnq-cost-gate.yaml` | `7aaf256c` |
| `71659a9f9a6c` | `2026-04-11-mnq-overnight.yaml` | `7aaf256c` |
| `75568f06196e` | `2026-04-11-mnq-cross-asset.yaml` | `7aaf256c` |
| `f32c9738985f` | `2026-04-11-atr-vel-expansion.yaml` | `7aaf256c` |
| `289360162082` | `2026-04-12-wave5-garch-nyse-open.yaml` | `7aaf256c` |
| `0cfbc2e784cf` | `2026-04-13-mnq-wider-aperture-session-structure.yaml` | `7aaf256c` |
| `b7cbb26c9e83` | `2026-04-13-mnq-wider-aperture-vol-regime-v2.yaml` | `7aaf256c` |
| `2ffe496799e1` | `2026-04-13-mnq-vwap-us-data-1000.yaml` | `7aaf256c` |
| `b91e21328ead` | `2026-04-13-cross-session-comex-cme-preclose.yaml` | `7aaf256c` |
| `135ca453989f` | `2026-04-13-cross-session-sgp-europe-flow.yaml` | `7aaf256c` |
| `b17a346adb37` | `2026-04-22-mnq-layered-candidate-board-v1.yaml` | `9a5e66e0` |
| `9a754e6fd2a7` | `2026-04-22-mnq-usdata1000-near-pivot-50-avoid-v1.yaml` | `9a5e66e0` |
| `7e3c4f5cdd89` | `2026-04-22-mnq-usdata1000-downside-displacement-take-v1.yaml` | `9a5e66e0` |
| `e93af94608b6` | `2026-04-22-mnq-usdata1000-clear-of-congestion-take-v1.yaml` | `9a5e66e0` |
| `87bb3d73f3cd` | `2026-04-22-mnq-usdata1000-positive-context-union-v1.yaml` | `9a5e66e0` |
| `d5d3df102fe1` | `2026-04-22-mnq-usdata1000-rr15-positive-context-union-v1.yaml` | `9a5e66e0` |
| `560459105e70` | `2026-04-22-mnq-comex-pd-clear-long-take-v1.yaml` | `9a5e66e0` |
| `fdf262d36f55` | `2026-04-22-mnq-tokyo-costlt08-take-v1.yaml` | `9a5e66e0` |
| `1fff43df8290` | `2026-04-24-mnq-usdata1000-f5-below-pdl-rr15-v1.yaml` | `6887632f` |

## Caveats

This addendum does not prove the historical trading conclusions again; it only
proves that each orphan SHA maps to a tracked source-file state and that the
manifest repair is evidence-grounded. It also does not close the live chart
ring-buffer smoke stage; that still needs fresh market bars after CME reopen.

## Verification — 2026-05-17 Audit-Only Pass

```
$ python pipeline/check_drift.py 2>&1 | grep -c "PHASE 4 SHA INTEGRITY: orphaned SHA"
11
```

Before this audit: 11. After this audit: 11. The MD adds context for future operators; it changes no behavior.

The 2026-05-24 addendum is different: it pairs the audit evidence with
manifest entries in `docs/audit/check_107_sha_migrations.yaml`, so current
verification must be read from the fresh Check 107 and sibling manifest checks.

## References

- `pipeline/check_drift.py:5998-6095` — Check 107 implementation
- `trading_app/holdout_policy.py::PHASE_4_1_SHIP_DATE`
- `trading_app/hypothesis_loader.py::find_hypothesis_file_by_sha`
- `HANDOFF.md` § "Next Session — Active" (2026-05-17)
- `feedback_chordia_loader_audit_log_independent_trust_surfaces.md` — Amendment 3.3 context
- Commits: `dc3a53be`, `d53b5bc4`, `8ab4fe13`
