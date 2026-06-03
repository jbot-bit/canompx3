# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-cost-lt15-o15-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-orb-g8-o15-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_OVNRNG_100_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-ovnrng-100-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-50-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-5-cb1-atr-p30-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_X_MGC_ATR70_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-x-mgc-atr70-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-100-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p70-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-50-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-0-cb1-cross-nyse-momentum-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_16K_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-vol-16k-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_4K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-4k-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_PDR_R105
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-pdr-r105-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-100-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_CROSS_NYSE_MOMENTUM
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-cross-nyse-momentum-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT08_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-cost-lt08-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-cost-lt12-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G4_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-g4-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_16K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-vol-16k-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_2K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-vol-2k-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_005_apply_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-vol-4k-o15-chordia-unlock-v1.draft.yaml
