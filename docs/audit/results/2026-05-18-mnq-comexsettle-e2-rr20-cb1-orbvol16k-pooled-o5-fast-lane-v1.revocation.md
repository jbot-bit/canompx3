# Revocation note — MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K

**Original PROMOTE result:** `docs/audit/results/2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.md`
**Original FAST_LANE verdict (now REVOKED):** PROMOTE (pooled t=3.300)
**Revocation reason:** pooling-inflation artifact + per-direction fire-rate extreme.
**Spent K to discover:** 0 (re-inspection of the same prereg's emitted per-direction stats).
**Authored:** 2026-05-18.

## Evidence (taken from the original result MD — no new compute)

### Pooled IS row (original result MD line 25)

| N_universe | N_fired | Fire% | ExpR | Sharpe | t | p_two |
|---:|---:|---:|---:|---:|---:|---:|
| 1494 | 97 | 6.49% | 0.4676 | 0.3350 | **3.300** | 0.00097 |

### Per-direction IS breakdown (original result MD line 32)

| Direction | N | ExpR | t | Implied fire-rate (N / 1494) |
|---|---:|---:|---:|---:|
| Long  | 47 | 0.4427 | **2.120** | 3.15% |
| Short | 50 | 0.4909 | **2.525** | 3.35% |

## v5.1 gates re-applied per-direction

`docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` lines 102-145 define:
- PROMOTE iff `t ≥ 3.0`
- NEEDS-MORE iff `t ∈ [2.5, 3.0)`
- KILL iff `t < 2.5`
- KILL iff `N_IS_on < 50`
- KILL iff `fire_rate ∉ [0.05, 0.95]` (extreme)
- KILL iff `ExpR ≤ 0`

| Side | t | N | Fire | Verdict as standalone |
|---|---:|---:|---:|---|
| Long | 2.120 | 47 | 3.15% | **KILL** — t<2.5 AND N<50 AND fire<5% |
| Short | 2.525 | 50 | 3.35% | **KILL** — fire<5% (extreme); t in NEEDS-MORE band on its own |

Both directions KILL as standalone lanes. The pooled t=3.300 is therefore a
sample-doubling artifact, not evidence of edge:

> sqrt(2) · mean(long_t, short_t) = sqrt(2) · 2.32 ≈ 3.28 ≈ pooled 3.30

That is the canonical signature of pooling inflation when the underlying
per-direction effects share sign but are individually sub-threshold.

## Why this is a revocation, not a NEEDS-MORE

- A deployable single-direction lane requires `fire_rate ≥ 5%` (TEMPLATE v5.1
  line 115). Both directions fire under 4% of the universe. No fresh data
  changes the cell's structural rarity.
- pooled-finding-rule (`/.claude/rules/pooled-finding-rule.md`) requires
  per-direction breakdown for any pooled claim that reaches heavyweight. Both
  per-direction subsets fail the deployment gate independent of the pooled
  framing. Heavyweight Chordia would reject on `per_cell_breakdown` grounds.
- The pooled framing also fails the operator-deployment test: a single
  pooled "lane" cannot route long+short simultaneously at a 6.5% combined
  fire-rate unless both sub-lanes are individually deployable, which neither
  is here.

## What this does NOT claim

- Does NOT claim ORB_VOL_16K as a filter family is dead. Sibling cells
  (other sessions, other RRs, other instruments, other apertures) are out
  of scope.
- Does NOT claim COMEX_SETTLE × E2 × RR2.0 is dead in general — only this
  exact filter-cell.
- Does NOT mutate `chordia_audit_log.yaml`, `validated_setups`, or
  `lane_allocation.json`.

## Status routing for the queue scanner

After this sidecar lands, `scripts/research/fast_lane_promote_queue.py`
classifies the original PROMOTE as **REVOKED** (sidecar precedence per
`classify()` in `scripts/research/fast_lane_promote_queue.py`).

A future hand-rebuild attempt that omits this sidecar will be caught by
drift check #157 (cache hand-edit detection).
