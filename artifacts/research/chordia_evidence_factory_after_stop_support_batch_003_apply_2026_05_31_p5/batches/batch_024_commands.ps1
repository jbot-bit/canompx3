# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-8k-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_VWAP_BP_ALIGNED_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-vwap-bp-aligned-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR2.0_CB1_ATR_P70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr2-0-cb1-atr-p70-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-atr-p50-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_50
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-ovnrng-50-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR60_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb1-x-mes-atr60-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt08-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G6_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-orb-g6-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-orb-vol-8k-o15-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-pdr-r125-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R125_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-pdr-r125-s075-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-0-cb1-pdr-r125-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_X_MGC_ATR70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-x-mgc-atr70-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G5
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-g5-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-10-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-25-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB3_PD_CLEAR_LONG_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb3-pd-clear-long-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb2-pd-clear-long-o30-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.5_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-5-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
