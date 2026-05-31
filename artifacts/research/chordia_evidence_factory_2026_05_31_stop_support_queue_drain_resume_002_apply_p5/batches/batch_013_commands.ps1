# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-x-mgc-atr70-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb2-vwap-mid-aligned-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB4_X_MES_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb4-x-mes-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-pdr-r105-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.5_CB1_PDR_R105_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-pdr-r105-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cross-nyse-momentum-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-orb-vol-16k-s075-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.0_CB1_PDR_R125
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-0-cb1-pdr-r125-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB4_VWAP_MID_ALIGNED_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb4-vwap-mid-aligned-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-atr-p30-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-orb-vol-16k-s075-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-cost-lt12-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MGC_ATR70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-x-mgc-atr70-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.5_CB1_VWAP_MID_ALIGNED
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-5-cb1-vwap-mid-aligned-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R125
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-pdr-r125-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-x-mes-atr60-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cross-nyse-momentum-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.5_CB1_COST_LT15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-5-cb1-cost-lt15-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_G8_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-5-cb1-orb-g8-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb5-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb1-x-mes-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-x-mes-atr60-s075-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_002_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr3-0-cb1-pdr-r125-chordia-unlock-v1.draft.yaml
