---
task: Multi-aperture discovery redesign — SHA gate fix + hypothesis v2 (K=8)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/phase_4_discovery_gates.py
  - trading_app/strategy_discovery.py
  - tests/test_trading_app/test_phase_4_discovery_gates.py
  - docs/audit/hypotheses/2026-04-13-mnq-wider-aperture-vol-regime-v2.yaml
blast_radius:
  - trading_app/phase_4_discovery_gates.py (add orb_minutes parameter to check_single_use)
  - trading_app/strategy_discovery.py (pass orb_minutes to check_single_use call)
  - tests/test_trading_app/test_phase_4_discovery_gates.py (2 new tests for orb_minutes-scoped check)
  - docs/audit/hypotheses/ (new hypothesis file, supersedes v1)
updated: 2026-04-13T09:30:00Z
agent: claude
---

## Design summary

SHA gate fix: `check_single_use` gains `orb_minutes` keyword parameter. When provided,
scopes the existing-rows check to matching orb_minutes. This allows one hypothesis file
covering multiple apertures to be run in separate `--orb-minutes` CLI invocations.
Same-aperture re-runs are still caught. Caller passes `orb_minutes=orb_minutes`.

New hypothesis file: K=8 (4 session-aperture pairs x 2 vol-regime filters). MinBTL=4.16yr
on 5.99yr data (30.6% headroom). Filters: ATR_P50 + ATR_P70 (aperture-invariant) for
Asian sessions, ATR_P50 + X_MES_ATR60 for US_DATA_1000. Supersedes v1 (d1b9c9cb SHA).

Adversarial review completed: LONDON_METALS dropped (correlated duplicate), sessions
narrowed from 12 to 3, cumulative K accounting documented.
