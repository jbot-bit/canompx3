# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-100-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-vwap-mid-aligned-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT08
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt08-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr2-0-cb1-vwap-mid-aligned-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_CROSS_NYSE_MOMENTUM_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cross-nyse-momentum-o30-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E1_RR1.0_CB1_X_MGC_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb1-x-mgc-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-x-mes-atr70-chordia-unlock-v1.draft.yaml
