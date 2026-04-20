---
stage: mes-mgc-unfiltered-baseline-cross-family
mode: IMPLEMENTATION
task: "Cross-instrument extension of PR #51. Run unfiltered-baseline cross-family scan (E2 CB1, apertures 5/15/30, RR 1.0/1.5/2.0, pooled direction) on MES + MGC as one Pathway-A family with BH-FDR at K_family."
updated: "2026-04-20"
scope_lock:
  - "docs/audit/hypotheses/2026-04-20-mes-mgc-unfiltered-baseline-cross-family-v1.yaml"
  - "research/mes_mgc_unfiltered_baseline_cross_family_v1.py"
  - "docs/audit/results/2026-04-20-mes-mgc-unfiltered-baseline-cross-family-v1.md"
  - "HANDOFF.md"
  - "memory/MEMORY.md"
  - "memory/recent_findings.md"
acceptance:
  - "Pre-reg yaml committed with LOCKED status, grounded in local literature extracts (Bailey 2013 MinBTL, Chordia 2018 t≥3.0, Harvey-Liu 2015 BH-FDR, Fitschen 2013, Chan 2013)."
  - "Scan script uses pipeline.paths.GOLD_DB_PATH and trading_app.holdout_policy.HOLDOUT_SACRED_FROM (no hardcoded path / dates)."
  - "Both MES and MGC tested in ONE honest K_family with BH-FDR computed across the combined set."
  - "Result MD written with CANDIDATE_READY / RESEARCH_SURVIVOR / KILL_IS breakdown, per-instrument subsection, and full result table."
  - "python pipeline/check_drift.py exits 0."
  - "HANDOFF.md + MEMORY.md + memory/recent_findings.md updated with the finding."
---
