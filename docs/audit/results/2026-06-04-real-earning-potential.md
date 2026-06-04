# Real Earning Potential — Full Deployable Universe (not the 3-lane C11 book)

**Date:** 2026-06-04 · **Mode:** read-only audit · **DB cutoff:** last_trade_day 2025-12-31
**Author session:** C11-throttle → earning-potential pivot (operator: "surely this project
has more potential than $6k/yr")

## Scope

Operator challenged a $6k/yr figure as tunnel-visioned. It was: $6k = ONE auto-profile
(`topstep_50k_mnq_auto`), a 3-lane MNQ book. This audit sizes the HONEST earning potential
of the full deployable universe via the canonical correlation-aware portfolio builder —
no naive sums, no 3-lane ceiling, bias-audited, survival-tested.

## Decision / Verdict

**Honest range: ~$7.7k (self-funded 1 acct) → $19.3k/account survival-PROVEN (Topstep $100k)
→ ~$96k/yr projected (×5 copies, correlation untested).** The bottleneck is **account
capacity (slots × copies × DD budget), NOT edge.** Same 848-setup pool yields $0 on Topstep
$50k and $19.3k survivable on Topstep $100k.

### Canonical builder books (`prop_portfolio.select_for_profile`, 848-pool, corr-pruned, slot/DD-capped)

| Account | Copies | Lanes | $/acct/yr | Total/yr | Per-acct survival | Status |
|---|---:|---:|---:|---:|---|---|
| topstep_50k | 5 | 0 | $0 | $0 | rejects all (DD) | NO-GO |
| topstep_50k_mnq_auto (C11) | 1 | 7 | $5,530 | $5,530 | strict_gate=False | NO-GO |
| bulenox_50k | 3 | 5 | $4,620 | $13,861 | 100%, strict=True | survives |
| self_funded_tradovate | 1 | 10 | $7,700 | $7,700 | n/a (not prop sim) | deployable |
| **topstep_100k_type_a** | 5 | 14 | **$19,296** | **$96,479** | **99.71%, strict=True, p95 DD $2,026<$2,400** | per-acct PROVEN; ×5 projected |

## Evidence

- **Universe (Step 1):** 848 deployable validated setups (MNQ 787 / MES 48 / MGC 13),
  528 family heads, all status=active. `deployable_validated_setups` view.
- **Independence (Step 2):** 528 heads collapse to **~22 (instrument×session×entry) clusters,
  11 distinct sessions, 2 entry models**. Same-session same-instrument lanes ~fully correlated
  (same ORB, same bars). Effective independent lanes ≈ 11 sessions × 3 instruments ≈ **15–22**,
  NOT 848. MES/MGC are the high-edge diversifiers (MGC expR 0.43–0.66 vs MNQ ~0.10) and are
  under-used (book was 93% MNQ).
- **Builder (Step 3):** picks ONE lane/session, caps at max_slots; 848→7 (C11 profile),
  848→10 (self-funded), 848→14 ($100k). $100k pulls in MGC LONDON_METALS (effExpR 0.659) + MES.
- **Survival (canonical `evaluate_profile_survival`, n_paths=10000, write_state=False):**
  topstep_100k op-survival 99.71%, trailing-DD breach 0.29%, daily-loss breach 0%,
  p50 DD $1,030 / p95 $2,026 / hist-max $1,690 vs $2,400 budget. bulenox 100%. C11 strict=False.

## Distortion comparison (Step 5)

- **$6k (3-lane C11):** one MNQ-only auto-profile, 1 account. Pigeonhole. Too pessimistic.
- **$1.29M (naive sum of 848):** assumes 848 independent edges; they're ~15–22 correlated lanes. Fantasy.
- **$7.7k–$96k (builder books):** real — de-correlated, slot/DD-capped, copy-multiplied, survival-tested.

## Highest-EV bottleneck (Step 6)

**Account capacity, not edge.** Proof: identical 848-pool → $0 (Topstep $50k) vs $19.3k
survivable (Topstep $100k). Edges sit unused (self-funded leaves 838/848 at slot cap 10).
**One change with most impact:** activate Topstep $100k Type-A (configured, 14 lanes,
99.71% survival) → proven $6k→$19k/account jump. MGC/MES diversifiers only enter as slots open.

## Limitations

- **OPEN: copies-multiplier survival untested.** Per-account $19,296 survival is PROVEN;
  ×5 → $96k assumes 5 funded accounts trading the SAME lanes on the SAME bars all survive the
  SAME bad days. `evaluate_profile_survival` is per-single-account — it does NOT model
  cross-copy correlated blow-up. This is the one gap between proven $19k and projected $96k.
- Annual $ is a **conservative floor**: 1-contract-per-lane where the builder allows more;
  computed as expR × avg_risk_$ × trades_per_year on canonical stats ($-proxy, not pnl_r).
- **Bias audit:** OOS inflation CLEARED (OOS/full expR ratio 1.03 across 848). Fitness mix
  (FIT+WATCH) implies a live haircut — WATCH lanes are decaying. expR is expectancy, not realized.
- self_funded survival sim is N/A by design (self-funded ≠ prop rule-survival construct;
  see self-funded-sizing-doctrine.md — risk-first, no prop cap).
- topstep_50k survival errored on a stale pinned MGC lane (`MGC_TOKYO_OPEN_..._CONT_S075`
  missing from validated_setups) — data-hygiene issue, separate from this finding.
- Not live-readiness: broker/journal/bracket-parity audit (`9b3fc530`) still gate any arming.

## Files

- Builder: `trading_app/prop_portfolio.py` (`select_for_profile`, `_load_strategies_and_build_books`)
- Survival: `trading_app/account_survival.py` (`evaluate_profile_survival`, `_build_rules`)
- Profiles: `trading_app/prop_profiles.py` (copies, max_slots, is_express_funded)
- DB views: `deployable_validated_setups` (848), `validated_setups` (871)

## Next action (one)

**Simulate the 5-copy correlated book for topstep_100k_type_a** (or confirm Topstep funds 5
simultaneous $100k accounts) — the only thing between proven $19.3k/acct and projected $96k/yr.
