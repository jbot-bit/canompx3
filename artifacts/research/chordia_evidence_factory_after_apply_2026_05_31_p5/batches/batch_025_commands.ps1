# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-nyse-close-e2-rr1-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB5_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb5-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB1_X_MES_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb1-x-mes-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR3.0_CB1_PDR_R125
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr3-0-cb1-pdr-r125-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E1_RR1.0_CB4_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e1-rr1-0-cb4-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-cme-preclose-e2-rr1-0-cb1-atr-p70-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E1_RR1.0_CB2_VWAP_MID_ALIGNED_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-us-data-1000-e1-rr1-0-cb2-vwap-mid-aligned-o15-chordia-unlock-v1.draft.yaml
