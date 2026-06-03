# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E1_RR1.0_CB5_ORB_G8_FAST10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb5-orb-g8-fast10-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E1_RR1.0_CB5_OVNRNG_25_FAST10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb5-ovnrng-25-fast10-o15-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR1.0_CB1_COST_LT15_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr1-0-cb1-cost-lt15-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB2_ATR70_VOL_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb2-atr70-vol-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.5_CB1_CROSS_NYSE_MOMENTUM
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-5-cb1-cross-nyse-momentum-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.5_CB1_COST_LT15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-5-cb1-cost-lt15-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G8_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-5-cb1-orb-g8-s075-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-cost-lt10-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-vol-16k-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P50
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p50-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-x-mes-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.5_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-5-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-x-mes-atr70-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-x-mes-atr60-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.5_CB1_OVNRNG_100
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-5-cb1-ovnrng-100-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT12_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt12-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-5-cb1-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb5-vwap-mid-aligned-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-vol-8k-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt10-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-50-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-0-cb1-ovnrng-100-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_X_MES_ATR60
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_resume_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-x-mes-atr60-chordia-unlock-v1.draft.yaml
