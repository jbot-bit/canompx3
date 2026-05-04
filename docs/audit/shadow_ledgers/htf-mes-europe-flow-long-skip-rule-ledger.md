# HTF MES EUROPE_FLOW long skip-rule — Shadow Ledger

**Pre-registration:** [`docs/audit/hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml`](../hypotheses/2026-04-18-htf-mes-europe-flow-long-skip-rule-shadow.yaml)

**Canonical predicate:** `orb_EUROPE_FLOW_break_dir = 'long' AND prev_week_high IS NOT NULL AND orb_EUROPE_FLOW_high > prev_week_high`

**Fresh OOS window:** `trading_day >= 2026-04-18` (post-v1-scan peek boundary).

**Contract:** Zero capital. Observational only. Idempotent append-only ledger.
Per-day row requires both RR 1.5 and RR 2.0 trades complete (outcome NOT NULL).

RULE 3.2 directional-only verdict is an acceptable outcome at calendar cap
(2028-06-30) per YAML `review_verdict_at_cap`.

| trading_day | break_ts | orb_high | prev_week_high | break_bar_vol | rel_vol | RR1.5 outcome | RR1.5 pnl_r | RR2.0 outcome | RR2.0 pnl_r |
|---|---|---|---|---|---|---|---|---|---|
