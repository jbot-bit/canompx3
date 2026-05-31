# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_OVNRNG_25
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-ovnrng-25-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_VOL_8K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-orb-vol-8k-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.5_CB1_OVNRNG_50
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-5-cb1-ovnrng-50-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-50-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt08-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt10-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-g8-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-vol-16k-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_8K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-orb-vol-8k-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_25_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-25-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV20_N20_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb2-vol-rv20-n20-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p30-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb1-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR2.0_CB1_ORB_VOL_8K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr2-0-cb1-orb-vol-8k-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB1_ATR70_VOL_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb1-atr70-vol-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB5_VOL_RV15_N20_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb5-vol-rv15-n20-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB5_VWAP_MID_ALIGNED_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb5-vwap-mid-aligned-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB2_VOL_RV15_N20_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb2-vol-rv15-n20-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-cost-lt08-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_OVNRNG_50
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-ovnrng-50-chordia-unlock-v1.draft.yaml
