# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-5-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-no-filter-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb3-x-mgc-atr70-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT08_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-cost-lt08-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-vol-4k-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R105
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr3-0-cb1-pdr-r105-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_4K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-vol-4k-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-vwap-mid-aligned-o30-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-vwap-bp-aligned-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ATR_P30_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-atr-p30-o15-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-cost-lt12-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-atr-vel-ge105-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-cost-lt08-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-g5-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-g8-chordia-unlock-v1.draft.yaml
