# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G4_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g4-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g6-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g8-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_4K_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-4k-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-8k-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_10_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-10-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_25_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-25-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT12_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-cost-lt12-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_8K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-8k-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ATR_P50_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-5-cb1-atr-p50-s075-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt08-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt10-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-ovnrng-100-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-cost-lt12-s075-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-no-filter-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G4
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g4-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g5-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-10-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-25-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P30_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-atr-p30-o30-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ATR_P50_O30_S075
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr2-0-cb1-atr-p50-o30-s075-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ATR_P50_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr2-5-cb1-atr-p50-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_8K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31_stop_support_queue_drain_003_apply_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-orb-vol-8k-chordia-unlock-v1.draft.yaml
