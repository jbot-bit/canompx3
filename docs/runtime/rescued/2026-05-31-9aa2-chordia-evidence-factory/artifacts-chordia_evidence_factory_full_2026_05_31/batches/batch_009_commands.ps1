# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-8k-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt08-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt10-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-no-filter-chordia-unlock-v1.draft.yaml
