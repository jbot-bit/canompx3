# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-vwap-mid-aligned-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E1_RR1.0_CB4_X_MGC_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb4-x-mgc-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_4K_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-vol-4k-s075-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-no-filter-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-orb-g4-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-orb-g5-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E1_RR1.0_CB3_X_MGC_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb3-x-mgc-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_OVNRNG_50
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-ovnrng-50-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MGC_ATR70_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-x-mgc-atr70-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-x-mes-atr70-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-vol-16k-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_16K_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-vol-16k-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-50-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.0_CB1_ORB_VOL_8K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-0-cb1-orb-vol-8k-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT08_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-cost-lt08-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10_O15_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-cost-lt10-o15-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT08_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-cost-lt08-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT10_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-cost-lt10-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-cost-lt12-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G6_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-g6-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-16k-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-4k-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-8k-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_10_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-ovnrng-10-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_stop_support_batch_003_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-ovnrng-25-o30-s075-chordia-unlock-v1.draft.yaml
