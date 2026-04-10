mode: IMPLEMENTATION
task: "Phase 2-3: Write MES+MGC Pathway B hypothesis files and run discovery for all 3 instruments"
updated: 2026-04-10T15:30:00+10:00
scope_lock:
  - docs/audit/hypotheses/2026-04-10-mes-pathway-b.yaml
  - docs/audit/hypotheses/2026-04-10-mgc-pathway-b.yaml
blast_radius: "Research artifacts only. No production code changes. Discovery writes to experimental_strategies table."
acceptance:
  - "MES Pathway B hypothesis file exists with testing_mode: individual and 5 hypotheses"
  - "MGC Pathway B hypothesis file exists with testing_mode: individual and 5 hypotheses"
  - "Discovery run complete for MNQ (existing file), MES, MGC with --testing-mode individual"
  - "Validation run complete for all 3 instruments with --testing-mode individual"
  - "Results reported: survivors per instrument"
