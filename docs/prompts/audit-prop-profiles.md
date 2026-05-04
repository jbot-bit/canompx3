# Prop Profile Audit Prompt

Paste this into Claude Code to run a full audit of `trading_app/prop_profiles.py` and `trading_app/prop_portfolio.py`.

---

## PROMPT

You are auditing the prop firm profiles and portfolio allocation logic for maximum profit per account. This is a READ-ONLY audit — do NOT edit any files. Present findings as a numbered list with severity (CRITICAL / HIGH / MEDIUM / LOW).

### Phase 1: Firm Rules Verification

For each firm in `PROP_FIRM_SPECS` (topstep, mffu, tradeify, apex, self_funded):

1. **Profit split accuracy.** Web-search each firm's CURRENT payout structure (April 2026). Compare to `profit_split_tiers`. Flag any stale splits — even 5% wrong compounds across hundreds of trades. Specifically check:
   - TopStep: is it still flat 90/10? Any Express vs Live distinction on splits?
   - Tradeify: still 90/10 on Select Flex? Any tiered structure introduced?
   - MFFU: still 80/20 on Core? Any Rapid/Pro tier differences we're missing?
   - Apex: still 100% on EOD PA? Any cap on early payouts?

2. **DD type accuracy.** Confirm `dd_type` for each firm. Specifically:
   - Is Apex still EOD trailing on PA plans (not intraday)?
   - Is TopStep still EOD trailing on TopStepX?
   - Has MFFU changed DD mechanics on any plan type?

3. **Daily Loss Limit (DLL).** Verify DLL values in `ACCOUNT_TIERS`. TopStep claims no DLL on TopStepX — is that still true? Apex has DLL=$1000 on 50K — confirm current value. Does Tradeify have a DLL we're missing?

4. **Contract limits.** Verify `max_contracts_micro` per tier against current firm rules. Firms change these quietly.

5. **Banned instruments.** Confirm Apex metals ban is still active. Check if any OTHER firm has added instrument restrictions since March 2026.

6. **Auto-trading policy.** Verify each firm's `auto_trading` field:
   - TopStep: still allows full automation on Express?
   - Tradeify: still allows own bots (exclusive ownership rule)?
   - Apex: still prohibits automation on PA plans?
   - MFFU: "semi" (own strategies with active monitoring) — still accurate?

7. **Consistency rules.** TopStep 40%, Apex 30% windfall — verify these haven't changed. Tradeify: confirm no consistency rule on funded Select Flex.

8. **Close times.** Verify `close_time_et` — firms sometimes shift these by 5-15 minutes. A wrong close time means the bot holds positions past forced liquidation.

### Phase 2: Account Profile Logic Audit

For each ACTIVE profile in `ACCOUNT_PROFILES`:

9. **DD budget utilization.** Run `validate_dd_budget()` and report results. Then compute: for each profile, what % of max_dd is the worst-case all-lanes-stop scenario? Flag any profile using <50% of DD budget (leaving money on the table) or >80% (too aggressive for EOD trailing). Show the math.

10. **Profit-per-account calculation.** For each active profile, compute expected annual profit:
    - Query `gold-db` MCP for each lane's strategy: get ExpR, sample_size, win_rate, trades_per_year
    - Convert R to dollars: `ExpR × avg_risk_dollars × profit_split × copies`
    - `avg_risk_dollars = median_orb_pts × stop_multiplier × point_value`
    - Sum across all lanes for total expected annual $ per profile
    - Compare profiles: which profile has the highest expected $/year? Is the allocation optimal?

11. **Lane selection optimality.** For each active profile's `daily_lanes`:
    - Query validated_setups for the same session × instrument: is the assigned strategy actually the BEST available (highest ExpR among CORE, E2, CB1, N>=100)?
    - If a better strategy exists for that slot, flag it with both ExpRs
    - Check: are there high-ExpR session×instrument combos NOT in any active profile that should be?

12. **ORB cap tightness.** For each lane's `max_orb_size_pts`:
    - Query actual ORB size distribution from `daily_features` for that session×instrument (last 12 months)
    - What percentile does the cap correspond to? (P75? P90? P95?)
    - Flag caps that are too tight (filtering >40% of trades = lost opportunity) or too loose (>P95 = not protecting DD)
    - Compute: how many trades per year does each cap filter out?

13. **Session coverage gaps.** List all sessions with CORE validated strategies (any instrument) that are NOT assigned to any active profile. For each gap, estimate the forgone annual profit.

14. **Cross-profile overlap.** Identify any strategy_id that appears in multiple active profiles. Overlap isn't necessarily bad (different firms = different accounts), but flag it for awareness — especially if the same strategy runs on the same platform (correlated DD risk).

### Phase 3: Allocation Efficiency

15. **Copies vs DD efficiency.** For multi-copy profiles (copies>1):
    - Is the copies count optimal? Each copy multiplies profit but also multiplies DD risk if using the same platform.
    - Would fewer copies at a higher tier (e.g., 3×100K instead of 5×50K) yield more $ after splits and DD constraints?

16. **Firm arbitrage.** Compare effective ExpR after profit split across firms for the same strategy:
    - `effective_expr = raw_expr × profit_split_factor`
    - If the same strategy runs on TopStep (90%) and MFFU (80%), the TopStep version is worth 12.5% more per R
    - Are strategies allocated to the highest-split firm possible given instrument bans and platform constraints?

17. **Inactive profile opportunity cost.** For each `active=False` profile:
    - What's the estimated annual profit being left on the table?
    - What's blocking activation? (Note from profile or infer from context)
    - Prioritize: which inactive profile should be activated first for maximum marginal profit?

18. **TYPE-A vs TYPE-B session fork.** The fork gives TYPE-A: CME_PRECLOSE/TOKYO_OPEN/LONDON_METALS and TYPE-B: NYSE_CLOSE/SINGAPORE_OPEN/EUROPE_FLOW.
    - Query ExpR for each fork session's lanes
    - Is the fork balanced (similar expected profit per side)?
    - Would swapping any session between types increase total expected profit?

### Phase 4: Risk & Survival Checks

19. **Worst-day simulation.** For each active profile at 1 contract:
    - If ALL lanes trigger AND all lose on the same day, what's the total $ loss?
    - What % of max_dd does that represent?
    - How many consecutive worst-days to account death?
    - For EOD trailing: does the worst day push the trailing floor past recovery?

20. **DLL compliance (Apex).** For the active Apex profile:
    - What's the worst-case single-day loss across all lanes?
    - Does it exceed the DLL ($1,500 on 100K)?
    - If yes, which lane combination triggers the breach?

21. **Consistency rule compliance.** For TopStep (40%) and Apex (30%):
    - Given the lane structure, can a single best day exceed the consistency threshold?
    - If running 5 lanes and one hits a big winner while others sit out, what's the max single-day profit as % of typical monthly profit?

22. **Monte Carlo DD estimates.** The code uses `DD_PER_CONTRACT_075X = 935.0` and `DD_PER_CONTRACT_10X = 1350.0`:
    - Are these still valid given current ORB size distributions?
    - Query actual ORB sizes for each active instrument/session and recompute expected DD per contract
    - Flag if the hardcoded estimates are >20% off from current data

### Phase 5: Code Logic Audit

23. **`_parse_strategy_id` correctness.** Test the parser against every `strategy_id` in active profiles' `daily_lanes`. Confirm it extracts the correct entry_model, rr_target, confirm_bars, filter_type, and orb_minutes. A parse error means the wrong strategy parameters get used downstream.

24. **`compute_profit_split_factor` edge cases.** Test with:
    - cumulative_profit = 0 (new account)
    - cumulative_profit = 4999 (just under TopStep's old tier boundary)
    - cumulative_profit = 100000 (well past any tier)
    - Confirm the tiered evaluation works correctly for all firms

25. **`get_lane_registry` shadow merge.** Verify the TopStep shadow lane merge into Apex manual profiles works correctly:
    - Does it correctly suppress MGC TOKYO_OPEN when MNQ TOKYO_OPEN exists?
    - Does it correctly NOT merge topstep_50k_mnq_auto lanes?
    - Does changing the active Apex profile (50K→100K) correctly update the registry?

26. **`validate_dd_budget` accuracy.** The validation uses `max_orb_size_pts` (the ORB cap) as worst-case, NOT P90 ORB. Is this correct? If a cap is set at 120 pts but 95% of trades have ORBs < 60 pts, the "worst case" is really the cap — but should the validation also show the EXPECTED case (median ORB)?

### Deliverables

Produce a single report with:
1. **Firm rules diff table:** each field, our value vs verified current value, MATCH/MISMATCH
2. **Per-profile profit estimate:** expected annual $ after splits, per lane and total
3. **Optimization recommendations:** ranked by expected $ impact, with specific strategy_ids
4. **Risk summary:** worst-day exposure, DLL compliance, consistency compliance per active profile
5. **Priority actions:** top 5 changes that would increase total portfolio profit, in order
