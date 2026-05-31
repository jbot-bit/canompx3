# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E1_RR1.0_CB3_ORB_G8_FAST10_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e1-rr1-0-cb3-orb-g8-fast10-o15-chordia-unlock-v1.draft.yaml
# MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G4_NOMON_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-london-metals-e2-rr1-5-cb1-orb-g4-nomon-o15-chordia-unlock-v1.draft.yaml
# MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G5_NOMON_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-london-metals-e2-rr1-5-cb1-orb-g5-nomon-o15-chordia-unlock-v1.draft.yaml
# MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G6_NOMON_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-london-metals-e2-rr1-5-cb1-orb-g6-nomon-o15-chordia-unlock-v1.draft.yaml
# MNQ_LONDON_METALS_E2_RR1.5_CB1_ORB_G8_NOMON_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-london-metals-e2-rr1-5-cb1-orb-g8-nomon-o15-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ATR_P70_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-atr-p70-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-100-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MGC_ATR70_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-0-cb1-x-mgc-atr70-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G6
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g6-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_ORB_VOL_2K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-orb-vol-2k-chordia-unlock-v1.draft.yaml
# MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G8
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-tokyo-open-e2-rr1-5-cb1-orb-g8-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_OVNRNG_100
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-ovnrng-100-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_16K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-orb-vol-16k-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_full_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-vwap-mid-aligned-chordia-unlock-v1.draft.yaml
