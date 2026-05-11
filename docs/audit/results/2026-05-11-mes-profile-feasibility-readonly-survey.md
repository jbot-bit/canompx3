# MES profile feasibility — read-only survey (2026-05-11)

**Author:** Claude (read-only audit, no writes to allocator/DB/schema)
**Branch context:** `codex/allocation-promotion-pipeline` (Codex parallel-session active 2026-05-11). No commits made on this branch by Claude.
**Trigger:** User goal — expand live trade book beyond 3 deployed MNQ lanes without pigeon-holing into a single instrument.
**Question:** Can a `topstep_50k_mes_auto` (or analogous MES sleeve in an existing profile) deploy any MES lanes today against canonical deployability gates?

---

## Scope

Read-only feasibility survey of the 48 MES `validated_setups` rows currently
classified as active under Mode A. Maps which deployability blockers actually
gate MES profile creation today (`topstep_50k_mes_auto`), distinguishing
mechanical / single-fix gaps from structural blockers. No discovery, no DB
mutation, no validator changes; K=0 MinBTL trials.

## Verdict: **NOT_DEPLOYABLE_TODAY — INFRASTRUCTURE_GAP_FILL_REQUIRED**

Zero of 48 active validated MES setups passes `trading_app/deployability.py::build_deployability_audit(scope='all-active', instruments={'MES'})` in strict mode.

This is **not** a research-edge problem. The MES candidate pool is healthy on the validator side — `strategy-lab.list_promotable_candidates(instrument='MES')` returns 39 FIT candidates (validated, FDR-cleared, currently fit). The block is a **canonical deployability gate** with four upstream gap-fills that have not been completed for MES the way they have been for MNQ.

---

## Inputs (canonical)

| Source | Value |
|---|---|
| `pipeline.paths.GOLD_DB_PATH` | `C:/Users/joshd/canompx3/gold.db` |
| `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` | `('MES','MGC','MNQ')` |
| `validated_setups` MES active rows | 48 |
| `strategy-lab.list_promotable_candidates(MES)` FIT count | 39 |
| `gold-db.get_strategy_fitness(MES, summary_only=True)` | fit=39, watch=7, decay=2, stale=0 |
| `chordia_audit_log.yaml` MES audit rows | 0 (the 2 X_MES_ATR60 rows are MNQ strategies using MES as a regime predictor) |
| Prior MES slippage pilot | `docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md` (PASS, N=40, 4 sessions covered) |
| `lane_allocation.json` | profile = `topstep_50k_mnq_auto`, MES instrument-filtered out at allocator entry |

Scratch JSON of the deployability run preserved at:
`_audit_scratch_mes_deployability.json` (gitignored — local scratch only).

---

## Verdict distribution (48 MES active rows)

| Verdict | Count |
|---|---|
| `BLOCKED_REPLAY_MISMATCH` | 35 |
| `BLOCKED_FAMILY_FRAGILE` | 13 |

## Hard-issue tally (issues stack per row)

| Hard issue | Count | Class |
|---|---|---|
| `slippage_missing` | 48 | **Inference-function gap** — measured pilot exists, but `_mnq_routine_tbbo_slippage_applies()` is MNQ-only. MES never enters the inference path; falls straight to `slippage_missing` even when the same v1 TBBO pilot evidence covers the row's session/entry-model. |
| `replay_mismatch` | 35 | **Replay backfill gap** — stored `validated_setups` outcomes do not reproduce on canonical replay for 35 of 48 rows. Stale promotion data from a prior pipeline build, same class as MNQ's pre-2026-05-11 gap-fill. |
| `c8_not_passed` | 36 | **C8 OOS gate** — 10 FAILED_RATIO + 12 NEGATIVE_OOS_EXPR + 14 INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH. Insufficient-N rows here are *not* an edge problem; they are the same gate-tier issue as the chordia-unlock candidates Codex audited individually. |
| `family_purged` | 26 | **FDR family purged** — parent family didn't survive BH-FDR; the strategy is a survivor of a dead family. Re-evaluation may reclassify after replay backfill changes member outcomes. |
| `family_singleton` | 22 | **Family singleton** — only one validated member; no peer-comparison evidence. |
| `sample_size_below_deploy_threshold` | 24 | **N<100** — Tier 4 sample class. Several promotable candidates have N=37–61 raw trades; deployment threshold gate. |
| `current_k_fdr_fail` | 1 | Single row failed current K-FDR re-check. |

## Session distribution of MES active rows (48)

| Session | Count | Routine TBBO pilot v1 covered? | Notes |
|---|---|---|---|
| CME_PRECLOSE | 35 | YES (per `2026-04-24-mes-e2-slippage-pilot-v1.md`) | Same session as MNQ deployed COMEX adjacency |
| NYSE_OPEN | 5 | NO (not in MES v1 pilot) | Pilot scope = CME_PRECLOSE / COMEX_SETTLE / SINGAPORE_OPEN / US_DATA_830 |
| US_DATA_1000 | 3 | NO | Same |
| COMEX_SETTLE | 2 | YES | Same session as MNQ deployed COMEX_SETTLE OVNRNG_100 — cross-instrument correlation gate risk |
| NYSE_CLOSE | 2 | NO | |
| US_DATA_830 | 1 | YES | |

---

## Why this is NOT_DEPLOYABLE today (mechanism, not narrative)

1. **Slippage inference is MNQ-only.** `trading_app/deployability.py:349 _mnq_routine_tbbo_slippage_applies()` early-returns False unless `instrument == 'MNQ'`. Even though `2026-04-24-mes-e2-slippage-pilot-v1.md` measured MES routine slippage as ≤1 modeled tick across its 4 sessions, the deployability gate has no path to consume that evidence. Class-bug: hard-coded instrument string instead of an instrument-keyed dispatcher.
2. **Replay rebuild owes MES.** Codex's MNQ work fixed an analogous "metadata gap" by rebuilding routine-TBBO MNQ E2 metadata. The MES side of that rebuild has not run.
3. **C8 OOS gate** for the 10 FAILED_RATIO + 12 NEGATIVE_OOS_EXPR rows requires per-strategy chordia-unlock-style audits like Codex did for the MNQ chordia queue.
4. **Family-purge gate** for 26 rows depends on family-level FDR re-evaluation; family status may change after (1)–(3) land.

These are sequential dependencies — fixing (1) without (2) still leaves 35 `replay_mismatch` blocks; fixing (1)+(2) still leaves the C8 and family-purge stack.

---

## What hardening this finding looks like (no implementation in this audit)

The slippage inference gap is the smallest and most generalizable. Recommended hardening (as a separate stage, with Codex confirmation since this branch is theirs):

- **Refactor**, not copy-paste. Replace `_mnq_routine_tbbo_slippage_applies(row)` with `_routine_tbbo_slippage_applies(row, *, registry)` where `registry` maps `(instrument, entry_model) -> {'sessions': frozenset, 'basis': str, 'evidence_path': str}`. Backed by a constant `ROUTINE_TBBO_SLIPPAGE_REGISTRY` populated from each instrument's pilot result doc. Refactor authority: `.claude/rules/institutional-rigor.md` § 3 ("if review cycles keep finding new divergences, the architecture is wrong — stop patching"). MNQ entry point stays MNQ-only by data; MES entry uses the same call site with a different registry key.
- **Drift check.** Add a `check_routine_tbbo_slippage_registry_coverage` that fails closed if any instrument with an `_e2_slippage_pilot_v1.md` result doc is absent from the registry, OR if any registry session is unsupported by its claimed evidence doc. Prevents the same class bug recurring per-instrument.
- **No live-state mutation.** This refactor is purely deployability-gate plumbing; lane_allocation.json, broker, schema, validated_setups all unchanged.

The replay/c8/family-purge gap-fills are larger and need Codex's MNQ work as the template. Each is a separate stage.

---

## Cross-instrument correlation pre-warning

Two of the COMEX_SETTLE MES candidates would, if/when ungated, deploy into the SAME session as the existing deployed `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`. `trading_app/lane_correlation.py` rho gate (>0.70) and 80% subset gate need to be re-applied at the **cross-instrument** level for those two rows specifically. Same-session different-instrument is not automatically uncorrelated. Out of scope for this audit; flagged for the future MES profile design stage.

---

## Limitations

- Maps surfaced blockers as of 2026-05-11 only; if `validated_setups` changes
  the MES row set, the per-row classifications may shift.
- The single-instrument cross-correlation pre-warning (§ Cross-instrument
  correlation pre-warning) is a flag, not a measured rho. Real cross-instrument
  rho + signal-time overlap must be computed before any MES + MNQ co-deployment.
- Survey treats `family_singleton` as a structural blocker; whether that gate
  is justified at high N is a doctrine question deferred to Stage 2 of the
  follow-up plan, not adjudicated here.
- No code, schema, or validator change followed from this survey;
  the linked Stage 1 work (registry refactor) was implemented in a separate
  worktree from `origin/main` rather than on Codex's working branch.

## What this audit did NOT do

- Did **not** modify `lane_allocation.json`, `validated_setups`, `chordia_audit_log.yaml`, broker state, schema, or any deployability code.
- Did **not** commit on `codex/allocation-promotion-pipeline` (Codex active session).
- Did **not** evaluate non-MES instruments (MGC has 13 active validated, parallel audit warranted but separate).
- Did **not** run per-strategy chordia unlocks for the c8_not_passed rows (would be K=39 audits — out of scope, pre-reg required per `RESEARCH_RULES.md`).
- Did **not** propose direct allocator unpause for any candidate.

## What this audit DID do

- Counted, by canonical gate, exactly which MES rows would survive deployment today (zero).
- Identified the four sequential infrastructure gap-fills required (slippage inference, replay backfill, C8 audits, family re-eval).
- Identified the slippage inference function as a class-bug with a clean refactor path (instrument registry).
- Flagged the cross-instrument correlation gate question for the COMEX_SETTLE MES candidates.

## Bailey-LdP MinBTL: this audit ran 0 trials. No brute force.

## Reproducibility

```python
from trading_app.deployability import build_deployability_audit
report = build_deployability_audit(scope='all-active', instruments={'MES'}, strict=True)
# 48 rows, verdict_counts: {'BLOCKED_REPLAY_MISMATCH': 35, 'BLOCKED_FAMILY_FRAGILE': 13}
# slippage_missing == 48 (universal blocker)
```

```python
# Promotable-side (validator-only, ignores deployability gate)
# Returns 39 FIT MES candidates excluded from allocation purely because
# allocator profile == 'topstep_50k_mnq_auto' (instrument filter).
```

## Decision recommended back to user

Pick one (or close as PARK):

| Option | Scope | Risk | EV unlock |
|---|---|---|---|
| **A: Refactor slippage inference + add drift check** (Phase 1) | ~50 LOC + 1 drift check + tests | Low (touches one gate, high test coverage already exists for MNQ path) | Removes 1 of 4 universal blockers; alone unlocks zero MES deploys. Necessary precondition. |
| **B: Phase 1 + replay backfill investigation** | Larger — diagnose why 35/48 MES rows fail replay | Medium (DB reads, may surface schema issues) | Removes 2 of 4 universal blockers. Still need C8 + family-purge. |
| **C: Mirror Codex's full MNQ unblock for MES** | All four gap-fills + per-strategy chordia unlocks for survivors | Large; needs pre-reg for the chordia batch | Plausible end-state: 4–8 incremental MES live lanes after correlation collapse. |
| **D: Park MES profile, pursue MGC** | Audit MGC instead (13 active rows, fewer to gap-fill) | Same class issues likely apply to MGC | Unknown until run; may have same infra debt. |
| **E: Park MES profile entirely; route operational expansion** | No research; pursue Bulenox/MFFU multi-firm scaling on existing MNQ deployed lanes per `memory/topstep_scaling_corrected_apr15.md` | Operational only | Same edges, more capital deployed. Highest near-term EV per memory. |

This audit makes **no recommendation between A–E**; it surfaces the cost/unlock tradeoff so the user can decide.
