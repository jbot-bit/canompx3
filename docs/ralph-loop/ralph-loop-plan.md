## Iteration: 179
## Target: pipeline/build_daily_features.py:1736-1742
## Finding: enrich_date_range() called AFTER con.execute("COMMIT") but inside the try-except-ROLLBACK block; if enrich_date_range raises, the except clause calls ROLLBACK on a closed transaction which raises TransactionContext Error, masking the original exception.
## Classification: [mechanical]
## Blast Radius: 1 production file (pipeline/build_daily_features.py), companion test file (tests/test_pipeline/test_build_daily_features.py)
## Invariants:
##   1. daily_features rows MUST still be committed before enrich_date_range runs (enrichment reads the committed rows)
##   2. ROLLBACK must still protect the daily_features INSERT if it fails (exception before COMMIT line 1726)
##   3. enrich_date_range must still be called after successful INSERT+COMMIT
## Diff estimate: ~5 lines moved (no logic change)
## Doctrine cited: integrity-guardian.md § 6 (No silent failures / exception masking); institutional-rigor.md rule 6
## Invariants: [1] R3 stable-run reset logic must remain in-memory; [2] C1 guard must remain ENTRY-only; [3] EXIT/SCRATCH events must not be guarded by kill-switch
## Diff estimate: 0 lines production code (audit-only)
## Doctrine cited: adversarial-audit-gate.md (independent context audit after every judgment CRIT/HIGH commit); institutional-rigor.md § 2 (review the fix); integrity-guardian.md § 3 (fail-closed)

### Adversarial Audit Findings

#### R3 (iter 176) — Reconnect ceiling + stable-run reset

| ID | Severity | Finding | Verdict |
|----|----------|---------|---------|
| R3-A | LOW | "Exhausted N reconnects" message is off by 1 (N+1 feed attempts run). Inherited from original range(MAX+1) loop — not new regression. | ACCEPTABLE (inherited behavior, cosmetic only) |
| R3-B | LOW | Stable-run reset can theoretically allow infinite reconnects if feed alternates stable/crash. | ACCEPTABLE (by design — rate-limit ceiling vs monotonic ceiling is the R3 fix intent) |
| R3-C | LOW | `last_connected_at` field is persisted and loaded, but no code ever reads it to make a decision. Informational field only. | ACCEPTABLE (informational-only persistence; in-memory counter reset is the actual mechanism) |

VERDICT: R3 PASS (3 low findings, all ACCEPTABLE)

#### C1 (iter 177) — Kill-switch ENTRY guard

| ID | Severity | Finding | Verdict |
|----|----------|---------|---------|
| C1-A | PASS | Guard correctly scoped to ENTRY branch; EXIT/SCRATCH at line 2362 structurally cannot reach the guard | PASS |
| C1-B | PASS | T4 test patches _handle_event itself but structural constraint (guard is inside ENTRY branch, not at function top) is stronger than a test | PASS |
| C1-C | LOW | _notify within C1 guard can raise if Telegram dispatch throws — same pattern used at 20+ other sites in orchestrator; not new regression | ACCEPTABLE (already guarded by upstream Telegram error handling pattern) |

VERDICT: C1 PASS (1 low finding ACCEPTABLE)

### Overall Audit Verdict: PASS
No CRITICAL or HIGH findings discovered in adversarial audit of iters 176 and 177.
Stage 2 (tracker integrity package) is cleared to proceed.
