task: Prop-Firm EV Simulator Stage 2 — survival + reach-payout adapter (survival half only)
mode: IMPLEMENTATION

## Scope Lock
- research/prop_firm_ev_scorecard.py

scope_lock:
  - research/prop_firm_ev_scorecard.py

## Blast Radius
- research/prop_firm_ev_scorecard.py — NEW file, zero callers (Stage 3 adds the EV half + CLI; Stage 4 adds tests). Survival-adapter half only this stage.
- Reads (import-only, never edited): trading_app/account_survival.py (evaluate_profile_survival, SurvivalSummary), trading_app/prop_profiles.py (get_profile, get_account_tier, get_firm_spec), pipeline/paths.py (GOLD_DB_PATH).
- Reads gold.db READ-ONLY via evaluate_profile_survival(write_state=False, db_path=GOLD_DB_PATH). Zero mutation: write_state=False is the enforcement seam; survival state-file mtimes asserted unchanged in verification.
- Writes: none (no DB writes, no allocation file, no canonical-module edits, no broker).
- Out of scope: EV formula, Frechet min(P_s,P_r), N_eff capacity, ev_band, NO_NUMERIC_EV seam, CLI, results.md (all Stage 3); tests (Stage 4).

## Key decisions (grounded this session)
- Gap 1 RESOLVED via AskUserQuestion: resolve_profit_target returns None + reason='no_profit_target_encoded' when a firm has no encoded target (true for ALL 3 deployed profiles — topstep/tradeify_50k/bulenox have no firm_specific_rules.profit_target). It RAISES only on malformed-but-present data (a real bug), never on genuine absence. No defaulting.
- Survive-leg = operational_pass_probability (R1), surface dd_survival_probability too.
- Gap 2: reach_payout_prob clamped [0,1], flagged 'coarse' when target outside [p05,p95]. N/A this stage since all 3 targets resolve to None — but the helper is built + unit-tested in Stage 4.
- Gap 3: assert summary.profile_id == requested_id after every call.
- Gap 4: capture sizing-parity status (engine only log.warning's) onto the record.
- R3: surface each firm's dd_type from get_firm_spec so a reader sees intraday vs EOD reached the engine.
