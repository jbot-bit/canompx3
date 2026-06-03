# Proposal-only strict replay command shard.
# Review and move draft preregs into docs/audit/hypotheses before executing against live doctrine.
# This file is not executed by the evidence factory.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-no-filter-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER_O30
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-no-filter-o30-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-no-filter-chordia-unlock-v1.draft.yaml
# MNQ_NYSE_OPEN_E2_RR1.0_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-nyse-open-e2-rr1-0-cb1-no-filter-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.5_CB1_NO_FILTER_O15
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-5-cb1-no-filter-o15-chordia-unlock-v1.draft.yaml
# MNQ_US_DATA_1000_E2_RR1.0_CB1_NO_FILTER
python research/chordia_strict_unlock_v1.py --hypothesis-file artifacts/research/chordia_evidence_factory_2026_05_31/prereg_drafts/2026-05-31-mnq-us-data-1000-e2-rr1-0-cb1-no-filter-chordia-unlock-v1.draft.yaml
