# L6 swap retrospective — VWAP_MID_ALIGNED → ORB_G5 on US_DATA_1000

- **rebalance:** 2026-04-18 (`docs/runtime/lane_allocation.json`)
- **swap:** previous deploy `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` (commit `2a92adf1`, 2026-04-13) → current deploy `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` (commit `c5797e4e`, 2026-04-18)
- **prior swap rationale (2026-04-13 commit msg):** "VWAP beats G5 on every metric — AnnR 22.1 vs 18.9 (+17%), WFE 0.90 vs 0.70, positive years 8/8 vs 7/8"
- **scope:** read-only retrospective; uses Phase 3a + Phase 3b empirical CSVs (no new OOS reads)
- **closes:** the "L6 swap potentially EV-losing" item from the 2026-04-18 adversarial portfolio re-audit
- **Mode-A IS boundary:** `trading_day < 2026-01-01` per `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`

## The two lanes side-by-side

Both lanes profile-eligible on `topstep_50k_mnq_auto` for the 2026-04-18 rebalance. Allocator picked ORB_G5; this audit asks whether that was right.

| metric | ORB_G5 (current) | VWAP_MID_ALIGNED (prior) | source |
|---|---:|---:|---|
| **annual_r_estimate** (12mo trailing, allocator's ranking input) | `+18.8` | `+16.0` | Phase 3a CSV |
| **rank_raw** (allocator's `_effective_annual_r` order) | **#23** (selected) | #25 (excluded) | Phase 3a CSV |
| **DSR_canonical (Mode-B inputs)** | `0.0015` | `0.6601` | Phase 3a CSV |
| **rank_dsr (Mode-B)** | #25 | **#3** | Phase 3a CSV |
| **Mode-A IS sample (N)** | `800` | `436` | Phase 3b CSV |
| **Mode-A per-trade Sharpe** | `+0.0472` | `+0.1532` | Phase 3b CSV |
| **DSR_canonical (Mode-A inputs)** | `0.0009` | `0.4643` | Phase 3b CSV |
| **rank_dsr (Mode-A)** | #29 | **#5** | Phase 3b CSV |

## Verdict

**The 2026-04-18 swap to ORB_G5 is a textbook False Strategy Theorem (LdP-Bailey 2018) instance.**

ORB_G5 wins by allocator-current-ranking (`+18.8` annual_r vs `+16.0`) only because its larger Mode-A trade count (N=800 vs N=436) inflates `annual_r ≈ ExpR × trades_per_year`. On the per-trade signal-quality dimension that matters for survivorship under selection bias:

- **Per-trade Sharpe ratio:** VWAP is **3.2×** stronger (0.153 vs 0.047)
- **DSR (Mode-A canonical N_eff=21):** VWAP is **~500×** more probable to be a real edge (0.46 vs 0.0009)
- **DSR sign-test under Bailey-LdP 2014 Eq 2:** ORB_G5 sits at the noise floor; VWAP is approaching the 0.95 "real edge" threshold the literature locks at `pre_registered_criteria.md` Criterion 5

The 2026-04-13 swap commit message ("VWAP beats G5 on every metric — AnnR 22.1 vs 18.9 (+17%), WFE 0.90 vs 0.70, positive years 8/8 vs 7/8") was correct at the time of the prior swap. Five days later the rebalance reverted to ORB_G5 because the allocator's ranking objective is **raw annual_r** with no Bailey-LdP correction — exactly the issue the entire A2b-2 / Phase 3a/3b audit thread surfaced.

## Was money lost?

**Cannot conclude.** Forward observation between 2026-04-13 and 2026-04-18 is too short to compute a meaningful ExpR delta on the live VWAP deployment. The full `live_journal.db` PnL window for the VWAP variant covers ~5 trading days at this lane's session — well below the Mode-A-IS minimum N=30 floor.

What CAN be concluded:

1. **The swap is the canonical instance** of the Phase 3a/3b RANKING_MATERIAL finding. Phase 3a flagged 4-5 of 6 deployed lanes as DSR-noise-floor; this lane is one of them. The L6 swap is that abstract finding made concrete on a single lane the user can inspect.

2. **The allocator's decision was internally consistent** with its current ranking objective (raw annual_r). The decision is "wrong" only if you accept the Phase 3a/3b literature-grounded interpretation that DSR-low lanes are noise.

3. **Shape E (commit `fc05db8f`/`fc77eb9e`) ensures every future swap is auditable in real-time.** The next rebalance will record both lanes' DSR alongside annual_r in `lane_allocation.json`. If a future swap flips a DSR-high lane out for a DSR-low lane, the operator has a per-rebalance signal to challenge the decision before it goes live.

## Doctrine question (not resolved by this audit)

Should the allocator avoid swapping a DSR-high lane out for a DSR-low lane even when raw annual_r favors the swap? That is the Shape A vs Shape E doctrine question that A2b-2 §11 deferred. **This audit does NOT resolve it; it merely makes the cost concrete.**

If the user later picks Shape A (annual_r × DSR ranking), the L6 swap would have gone the other way: VWAP_MID_ALIGNED would have been kept (`16.0 × 0.66 = +10.6` vs `18.8 × 0.0015 = +0.028` — VWAP wins by ~378×). Under Shape A this swap would be impossible.

If the user stays with Shape E (current state), the L6 swap was the allocator working as designed under the current doctrine. Forward observation accumulates the data needed to revisit the doctrine in 6+ months.

## Closes

- Adversarial portfolio re-audit open item: "L6 swap (VWAP→ORB_G5) on 2026-04-18 potentially EV-losing"
- Phase 3a/3b finding RANKING_MATERIAL_PRESERVED-WITH-PARTIAL-AGREEMENT: this lane is the cleanest-cut illustration

## Provenance

- Phase 3a empirical: `docs/audit/results/2026-04-18-dsr-ranking-empirical-verification.md` + per-lane CSV
- Phase 3b Mode-A: `docs/audit/results/2026-04-18-dsr-ranking-mode-a-phase3b.md` + per-lane CSV
- Shape E scope (deferred behavioral fix): `docs/audit/hypotheses/2026-04-18-a2b-2-dsr-ranking-preregistered.md`
- Prior swap commit (2026-04-13): `2a92adf1`
- Current swap commit (2026-04-18): `c5797e4e`
- Adversarial portfolio re-audit: `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md` (cited; file exists on `main`)
- Literature: `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md` Theorem 1 + lines 80-82 (deployed lanes already flagged below noise floor on 2026-04-07)
- Shape E doctrine ref: `trading_app/dsr.py:35` "DSR is INFORMATIONAL"
