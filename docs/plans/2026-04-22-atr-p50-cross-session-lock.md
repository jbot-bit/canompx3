# ATR_P50 Cross-Session Lock

**Date:** 2026-04-22
**Status:** LOCKED
**Owner:** Codex

## Why this path now

This is the highest-EV next research step after stepping back from the
recent Codex branch stack.

The merged repo truth says:

- `L3 COMEX_SETTLE ORB_G5` is the only clear deployed behavioral lane on the
  current book, per
  [2026-04-21-orb-g5-deployed-lane-arithmetic-check.md](../audit/results/2026-04-21-orb-g5-deployed-lane-arithmetic-check.md).
- `L1 EUROPE_FLOW ORB_G5` is a cost gate and ATR-normalized replacement is not
  worth pursuing, per
  [2026-04-21-l1-europe-flow-filter-diagnostic.md](../audit/results/2026-04-21-l1-europe-flow-filter-diagnostic.md).
- `ATR_P50` remains the only deployed non-size, non-explicit-cost filter class
  with unresolved mixed signal on the current book.
- The corrected aperture audit says the live `L2` question is properly an
  `O15` question, not the old buggy `O5` framing, per
  [2026-04-21-correction-aperture-audit-rerun.md](../audit/results/2026-04-21-correction-aperture-audit-rerun.md).
- The allocator audit and session handover both show ATR_P50 already exists on
  `SINGAPORE_OPEN O15` and `SINGAPORE_OPEN O30`, and those two apertures are
  correlated but not identical, per
  [2026-04-20-allocator-scored-tail-audit.md](../audit/results/2026-04-20-allocator-scored-tail-audit.md)
  and
  [2026-04-19-session-handover.md](../handoffs/2026-04-19-session-handover.md).

That makes the right next question:

> Is `ATR_P50` a real cross-session MNQ filter class on the aperture surface it
> already occupies (`O15`, `O30`), or is it just a lane-local SINGAPORE_OPEN
> artefact?

This is a better use of time than extending the freshest cross-asset branch,
because it attacks a portfolio-level question the current book still needs
answered: whether there is a second distinct behavioral filter family beyond
COMEX `ORB_G5`.

## What this is NOT

This is **not**:

- a threshold search over `ATR_P30/P50/P70`
- a fresh aperture discovery over `O5/O15/O30`
- an RR search over `1.0/1.5/2.0`
- a role search over `R1/R3/R8`
- an OOS tuning exercise
- a new deployment claim

All of those would widen scope or mix role questions before the class-level
question is answered.

## Fixed scope

- Instrument: `MNQ`
- Sessions: all 12 enabled MNQ sessions present canonically at the frozen
  geometry
- Geometry: `entry_model=E2`, `confirm_bars=1`, `rr_target=1.5`
- Apertures: `O15` and `O30` as parallel fixed substudies
- Filter under test: canonical `ATR_P50` only
- Role: `R1` binary overlay only
- IS window: `trading_day < 2026-01-01`
- OOS window: `trading_day >= 2026-01-01`, descriptive only

Why `O15 + O30` and not `O5`:

- `O15` is the corrected live `L2` geometry.
- `O30` is already in the repo’s ATR_P50 surface and was the selected
  SINGAPORE_OPEN ATR_P50 lane in the allocator scored audit.
- The repo does **not** present `O5` as the ATR_P50 surface for this family.
  Pulling `O5` in here would be a new aperture expansion, not an honest
  replication of the existing ATR_P50 footprint.

Canonical session universe at the fixed `O15` geometry, queried on 2026-04-22
from `orb_outcomes` pre-holdout:

- `BRISBANE_1025` `N=1721`
- `CME_PRECLOSE` `N=201`
- `CME_REOPEN` `N=506`
- `COMEX_SETTLE` `N=1498`
- `EUROPE_FLOW` `N=1718`
- `LONDON_METALS` `N=1709`
- `NYSE_CLOSE` `N=231`
- `NYSE_OPEN` `N=1393`
- `SINGAPORE_OPEN` `N=1719`
- `TOKYO_OPEN` `N=1721`
- `US_DATA_1000` `N=1495`
- `US_DATA_830` `N=1639`

Canonical session universe at the fixed `O30` geometry, queried on 2026-04-22
from `orb_outcomes` pre-holdout:

- `BRISBANE_1025` `N=1715`
- `CME_PRECLOSE` `N=40`
- `CME_REOPEN` `N=333`
- `COMEX_SETTLE` `N=1222`
- `EUROPE_FLOW` `N=1699`
- `LONDON_METALS` `N=1677`
- `NYSE_CLOSE` `N=131`
- `NYSE_OPEN` `N=1037`
- `SINGAPORE_OPEN` `N=1707`
- `TOKYO_OPEN` `N=1705`
- `US_DATA_1000` `N=1181`
- `US_DATA_830` `N=1611`

The full pre-holdout sample at `O15` is `2019-05-06` through
`2025-12-31`, `N=15551`.
The full pre-holdout sample at `O30` is `2019-05-06` through
`2025-12-31`, `N=14058`.

## Why the old kills do not bind

- `L1 ATR-normalized replacement` does not bind because that was a size-family
  rescue on a cost-gate lane. This is a different filter class and a different
  question.
- `Cross-session pre-break null` does not bind because that killed a specific
  `pre_velocity` direction-alignment framing, not vol-regime gating.
- `OVNRNG allocator/router` kill does not bind because this is not routing or
  portfolio allocation.
- `Pooled ML smoothing` dead-path does not bind because this is a fixed,
  univariate, canonical filter replication audit.

## Why this is anti-tunnel

This path widens the lens in the right way:

- away from one lane to the full MNQ session set
- away from size-family rehashes to a different mechanism class
- away from current-winner overfitting to a class-level replication test
- away from one role per feature family only after freezing the role first
- away from silently privileging one ATR aperture without admitting the repo
  already has two live ATR_P50 apertures

If the family fails here, that is informative and shuts down more ATR_P50
storytelling. If it survives at one aperture only, that still matters because
it tells us the class is geometry-sensitive rather than universal. If it
survives at both, only then is it worth asking whether the class deserves
lane-local deployment or size-modifier follow-up.

## Next artifact

The only allowed next artifact after this lock is:

- `docs/audit/hypotheses/2026-04-22-mnq-atr-p50-cross-session-generalization-v1.yaml`

No runner before the prereg is committed and SHA-stamped.
