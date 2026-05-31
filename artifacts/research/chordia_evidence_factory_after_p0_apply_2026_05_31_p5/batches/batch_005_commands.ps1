# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_ORB_VOL_2K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr2-5-cb1-orb-vol-2k-o15-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ORB_VOL_4K_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr1-5-cb1-orb-vol-4k-o30-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-0-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_COST_LT10
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-cost-lt10-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR3.0_CB1_ORB_VOL_2K_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr3-0-cb1-orb-vol-2k-o15-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_4K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-orb-vol-4k-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_OVNRNG_100
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-ovnrng-100-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR2.5_CB1_COST_LT12_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr2-5-cb1-cost-lt12-o15-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr2-0-cb1-cross-sgp-momentum-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR1.5_CB1_COST_LT08
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr1-5-cb1-cost-lt08-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ATR_P30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-atr-p30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_4K
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-orb-vol-4k-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-0-cb1-ovnrng-100-chordia-unlock-v1.draft.yaml
# MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-singapore-open-e2-rr1-5-cb1-atr-p50-o30-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_X_MGC_ATR70
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-x-mgc-atr70-chordia-unlock-v1.draft.yaml
# MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT08
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt08-chordia-unlock-v1.draft.yaml
# MNQ_COMEX_SETTLE_E2_RR2.0_CB1_COST_LT15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_after_p0_apply_2026_05_31_p5/prereg_drafts/2026-05-31-mnq-comex-settle-e2-rr2-0-cb1-cost-lt15-chordia-unlock-v1.draft.yaml
