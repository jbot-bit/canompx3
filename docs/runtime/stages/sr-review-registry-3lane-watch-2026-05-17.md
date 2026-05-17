# SR-Review-Registry — 3-lane WATCH for deployed MNQ Monday-blocker

task: Add `sr_review_registry.py` WATCH entries for the three currently-ALARM deployed MNQ lanes (COMEX_SETTLE_ORB_VOL_2K, NYSE_OPEN_COST_LT12, OVNRNG_25). Without entries, `lifecycle_state.py:245-248` blocks them on Monday because `sr_status == "ALARM"` AND `sr_review is None` falls through to the default `sr_alarm` block branch.
mode: IMPLEMENTATION

## Scope Lock
- trading_app/sr_review_registry.py
- tests/test_trading_app/test_sr_review_registry.py

## Blast Radius
- trading_app/sr_review_registry.py — appends 3 dict entries (no schema change, no field added). Consumed at trading_app/lifecycle_state.py:232 (`get_sr_alarm_review`) and surfaced via strategy_states blocked-flag at lifecycle_state.py:237-248. Each entry maps `(profile_id, strategy_id)` → `SrAlarmReview(outcome="watch", reviewed_at, summary, recheck_trigger)`. No allocator state mutation; pure registry doctrine.
- tests/test_trading_app/test_sr_review_registry.py — adds 3 mutation-proof tests (one per new entry) mirroring the existing `test_nyse_rr15_costlt12_sr_alarm_has_code_backed_watch_review` grammar. Tests assert canonical figures (WFE, OOS/IS, p_value) verbatim — any future drift in `summary` strings will fail the test.
- Reads (truth): `gold.db::validated_setups` (wfe/oos_exp_r/expectancy_r/p_value/sample_size/sharpe_ann), `data/state/sr_state.json` (current_sr_stat/recent_10_mean_r/alarm_trade), `docs/audit/results/2026-05-17-mnq-deployed-lanes-regime-stratified-audit-v1.md` (descriptive R5/R6 trail). Writes: none beyond the 2 files above.

## Why this is not "doc-only"
Initial scope estimate was doc-only, but the canonical registry is Python (`trading_app/sr_review_registry.py`), not markdown — entries are `SrAlarmReview` dataclass instances that drive `lifecycle_state.py` blocked/allowed branching. This makes it production code with allocator-gate effect.

## Canonical-grounding sources for entries
Each entry cites EXACTLY (no memory paraphrase):
1. `validated_setups.wfe` — current canonical WFE
2. `validated_setups.oos_exp_r / validated_setups.expectancy_r` — OOS/IS ratio (the precedent grammar)
3. `validated_setups.p_value`, `validated_setups.sample_size`
4. `sr_state.json::payload.results[strategy_id].current_sr_stat`, `.alarm_trade`, `.recent_10_mean_r`
5. Audit MD R5 / R6 numerics for the recheck-trigger evidence trail (descriptive, not threshold)
6. Criterion 12 (`docs/institutional/pre_registered_criteria.md:210-240`) — SR-alarm interpretive chain

## VWAP_MID_ALIGNED_O15 — explicitly NOT in scope
`MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` currently has SR=CONTINUE (`current_sr_stat=0.4154`), so `lifecycle_state.py:232` never even consults the registry. Adding an entry would be inert code. The audit's R5 ExpR=-0.0019 is a regime-window descriptive stat (H1 p=0.76 and H2 p=0.43 both fail to reject regime-stability for fire-rate and ExpR respectively); canonical full-window ExpR=0.149 and OOS/IS=103% pass deploy floors. Last turn's "MODIFY → ½-size + 30-trade shadow" decision does not map to the registry schema (no sizing field) and is statistically unsupported by canonical truth. Deferred to a future allocator-stage decision (user-forbidden in this plan).

## Acceptance criteria
1. 3 entries authored in `trading_app/sr_review_registry.py` matching existing dataclass grammar.
2. 3 companion tests pass under `pytest tests/test_trading_app/test_sr_review_registry.py`.
3. Test grammar mirrors line-25 existing test (`test_nyse_rr15_costlt12_sr_alarm_has_code_backed_watch_review`): assert outcome="watch", reviewed_at, "WFE" in summary, "OOS/IS" in summary, canonical-figure substrings present, recheck_trigger contains "N>=" trade count.
4. `python pipeline/check_drift.py` still passes (130 expected, no new failures; check 65 and adjacent SR-registry-related checks should not regress).
5. Self-review pass per `.claude/rules/institutional-rigor.md` § 1.
6. `/code-review` (or `/audit-code`) adversarial pass — entries are capital-gate, run audit before commit.
