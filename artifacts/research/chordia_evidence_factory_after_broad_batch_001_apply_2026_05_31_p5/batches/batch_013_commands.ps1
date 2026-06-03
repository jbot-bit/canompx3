# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_25
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-ovnrng-25-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-cost-lt12-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g5-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g6-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G8_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-g8-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_4K_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-vol-4k-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_VOL_8K_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-orb-vol-8k-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_10_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-10-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_25_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-25-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT15_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-cost-lt15-o15-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ORB_G8_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-0-cb1-orb-g8-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-50-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p70-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-ovnrng-50-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.0_CB1_CROSS_NYSE_MOMENTUM
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_broad_batch_001_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-0-cb1-cross-nyse-momentum-chordia-unlock-v1.draft.yaml
