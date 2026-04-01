---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Fix verified prop firm rule mismatches from audit
updated: 2026-04-01T13:00:00Z
scope_lock:
  - trading_app/prop_profiles.py
  - docs/plans/2026-04-01-prop-profile-audit-findings.md
blast_radius:
  - prop_profiles.py: ACCOUNT_TIERS data values + PROP_FIRM_SPECS close times
  - DD budget validation runs at import time
  - TYPE-B profiles use Tradeify tiers (DD budget changes)
acceptance:
  - Tradeify 100K DD = $3,000 (was $4,000)
  - Tradeify 150K DD = $4,500 (was $6,000)
  - TopStep close_time_et = 16:10 (was 16:00)
  - Tradeify close_time_et = 16:59 (was 16:00)
  - Apex consistency_rule = 0.50 (was 0.30)
  - All sources documented in audit findings
  - check_drift passes
---
