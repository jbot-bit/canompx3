# US_DATA_1000 Long NOT_F6 — Shadow Ledger

**Pre-registration:** [`docs/audit/hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml`](../hypotheses/2026-04-20-usdata1000-long-not-f6-shadow-v1.yaml)

**Canonical predicate:** `NOT_F6_INSIDE_PDR`, where `F6_INSIDE_PDR := orb_mid > prev_day_low AND orb_mid < prev_day_high`

**Descriptor:** `F5_BELOW_PDL := orb_mid < prev_day_low`

**Fresh OOS window:** `trading_day >= 2026-04-17` because candidate-lane validation already consumed OOS through `2026-04-16`.

**Contract:** Zero capital. Observational only. Idempotent append-only ledger.
Per-day row requires both RR 1.0 and RR 1.5 trades complete.

| trading_day | orb_mid | prev_day_low | prev_day_high | descriptor_f5 | RR1.0 outcome | RR1.0 pnl_r | RR1.5 outcome | RR1.5 pnl_r |
|---|---:|---:|---:|---|---|---:|---|---:|
