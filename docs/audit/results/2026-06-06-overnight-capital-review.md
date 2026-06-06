# Overnight Capital Code-Review — running log (2026-06-06)

Disk-durable state for the overnight `/loop` (15-min cadence). Survives `/clear`.
Each tick: pick next un-reviewed capital commit (oldest-first), review, classify
(Tier A → fix+commit; Tier B → PENDING-OPERATOR-APPROVAL.md), append verdict here.

## Scope

Adversarial code-review of the last 7 days of capital-path commits: `trading_app/live/`,
`trading_app/prop_profiles.py`, `pipeline/account_survival.py`, `pipeline/cost_model.py`,
`scripts/run_live_session.py`, webhook/broker paths. When exhausted → sweep
`trading_app/` + `pipeline/` fresh. This is a running review log, not a single-audit
result; it makes no statistical claims.

## Verdict

Per-commit verdicts are recorded inline in the Reviewed table below (CLEAN / FINDING).
Tier A findings are fixed+committed in place; Tier B findings route to
`PENDING-OPERATOR-APPROVAL.md` for explicit operator GO. As of this snapshot: 3 CLEAN,
1 FINDING (resolved as `99caaa4e`), remainder queued.

## Outputs

The single FINDING this cycle (`fb76e8cf` drift-gate over-broad `endswith`) was applied
to origin/main as `99caaa4e` (exact-match frozenset + rename parse + 4 anti-regression
tests, drift 178/0). Reproduction of each verdict: re-run the commit's companion tests
named in the Action column.

## Limitations

This log reviews only the declared 7-day capital-path scope, oldest-first; commits in
the "Not yet reviewed" queue are unaudited. A CLEAN verdict means no capital-impacting
defect was found in that commit's diff under review — not a proof of whole-system
safety. Lower-risk dashboard/CTA commits are deliberately deferred to last.

---

## Reviewed

| Commit | Subject | Verdict | Tier | Action |
|---|---|---|---|---|
| `f1413178` | instance-lock self-heal X-close orphan | ✅ CLEAN | — | none (live-PID refusal test-proven) |
| `fb76e8cf` | drift gate ignores always-dirty files | ⚠️ FINDING | B | → PENDING (endswith over-broad + rename parse) |
| `f1fd7a90` | fail-closed guard present-but-≤0 risk_points | ✅ CLEAN | — | none (both call sites fail closed) |
| `1e79c65d` | C11 cap×0.75 wiring | ✅ CLEAN | — | none (gate↔live parity structural, no seam) |

## Not yet reviewed (oldest-first queue)

7782a573 webhook MAX_ORDER_QTY on broker-mapped qty (#330)
1566a43e self-funded sizing doctrine
2810fead MFFU Builder+Flex specs/tiers/payouts; Rapid sim-cap leak
399fbce6 max_live_accounts drift guard (MFFU Layer C)
857a6388 mffu_builder_addon tier ($1500 MLL); flex payout TODO
09236046 broker-factory bypass in reconnect contract re-resolution
8a289cb7 projectx auth validate fallback
d4c1dd65 MFFU Flex per-payout cap size-specific
1c581c65 auto-expire orphaned handoff
8ec1f00f restore strict-zero-warn gate on dashboard live launch
6fe428a7 fail-closed C11/C12 capital gates when routing without --profile
baf99cfe harden live capital evidence gates
(+ dashboard CTA/lane/control-room commits — lower capital risk, review last)
