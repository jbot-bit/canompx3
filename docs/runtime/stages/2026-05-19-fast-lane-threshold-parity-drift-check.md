---
task: "Add Check #158 fast_lane_promote_threshold_parity — parity drift check for fast_lane_promote_queue.py scanner constants vs TEMPLATE-fast-lane-v5.1.yaml canonical thresholds"
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_fast_lane_promote_threshold_parity.py
  - docs/runtime/stages/2026-05-19-fast-lane-threshold-parity-drift-check.md
  - docs/runtime/decision-ledger.md
---

## Blast Radius

- `pipeline/check_drift.py` — add ONE new function `check_fast_lane_promote_threshold_parity` and register in `CHECKS`. Pure addition, no existing check touched. Drift count 157 → 158. Reads `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` (canonical source) and imports the scanner module to read its constants. Returns list of violation strings (project's standard contract).
- `tests/test_pipeline/test_check_drift_fast_lane_promote_threshold_parity.py` — NEW file, 5 mutation-probe injection tests (one per constant T_KILL_FLOOR / T_PROMOTE_FLOOR / N_FLOOR / FIRE_MIN / FIRE_MAX) + 1 clean-state pass test + 1 missing-template-file fail-closed test. Mirrors monkeypatch style of `test_check_drift_fast_lane_promote_orphans.py`. Each constant must have its own dedicated injection so the regex-alternation-sibling-coverage rule is satisfied.
- `docs/runtime/decision-ledger.md` — one entry appended at top.
- Reads: `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` (read-only); `scripts/research/fast_lane_promote_queue.py` (imported read-only).
- Writes: none at runtime; new file at commit time.
- Trading logic / DB / lane allocator: untouched. No capital-class blast radius.

## Canonical-source map (verified against TEMPLATE-fast-lane-v5.1.yaml lines 102-145)

| Scanner constant | Canonical YAML path | Canonical value | Notes |
|---|---|---|---|
| `T_KILL_FLOOR=2.5` | `screen.promote_threshold` (line 104) | 2.5 | Direct |
| `T_PROMOTE_FLOOR=3.0` | `screen.promote_threshold + screen.needs_more_band` (lines 104+113) | 3.0 | Computed `2.5 + 0.5` |
| `EXPR_FLOOR=0.0` | `screen.expr_min` (line 111) | 0.0 | Direct |
| `N_FLOOR=50` | `screen.n_IS_on_min` (line 112) | 50 | Direct |
| `FIRE_MIN=0.05` | parsed from `screen.fire_rate_gate.kill_if` line 115 | 0.05 | Regex over kill_if string |
| `FIRE_MAX=0.95` | parsed from `screen.fire_rate_gate.kill_if` line 115 | 0.95 | Regex over kill_if string |

Cleaner alternative: amend template to add explicit `fire_rate_min/max` numeric fields. **NOT TAKEN** this stage — keeps scope ≤2 production files; if regex parsing of the kill_if string proves brittle we revisit in Stage 2 (meta-registry) where the cleanup will naturally surface.

## Acceptance

1. New file `test_check_drift_fast_lane_promote_threshold_parity.py` lands with **5 mutation-probe injection tests** (one per gated constant — T_KILL_FLOOR, T_PROMOTE_FLOOR, N_FLOOR, FIRE_MIN, FIRE_MAX) + 1 clean-state pass + 1 template-missing fail-closed test. EXPR_FLOOR has no canonical-side variation (template hardcodes 0.0) — included in the parity check but covered by the clean-state test rather than dedicated injection.
2. All 7 new tests PASS via `pytest -q tests/test_pipeline/test_check_drift_fast_lane_promote_threshold_parity.py`.
3. Existing 20 scanner tests still PASS (`pytest -q tests/test_research/test_fast_lane_promote_queue.py` or equivalent).
4. `python pipeline/check_drift.py` count goes 157 → 158 and passes (modulo the pre-existing MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 trade-window violation, which is orthogonal carry-over).
5. Commit message cites `[[canonical-inline-copy-parity-bug-class]]` as the class this closes.
6. Per `institutional-rigor.md` § 2 + `adversarial-audit-gate.md`: evidence-auditor pass on the commit AFTER landing. This is a `pipeline/` touch but pure additive check with no truth-layer mutation — gate is advisory not compulsory. Running it anyway because the parity check IS the truth-layer assertion.

## Self-check (simulated)

- **Happy path:** scanner constants match template → check returns `[]`. Verified by clean-state test.
- **Drift path (per constant):** flip ONE constant on the imported scanner module via `monkeypatch.setattr` → check returns ONE violation naming that constant + the expected canonical value + the observed scanner value. Verified by 5 dedicated injection tests.
- **Failure mode:** template file missing or unparseable → return a single advisory violation `"check_fast_lane_promote_threshold_parity: failed to load TEMPLATE-fast-lane-v5.1.yaml: <reason>"` so the check fails closed (the parity *cannot* be verified). Verified by template-missing test pointing the loader at tmp_path.
- **Regex precision:** `kill_if` string is `"fire_rate < 0.05 OR fire_rate > 0.95"`. Regex `r"< ?(\d+\.\d+).*?> ?(\d+\.\d+)"` extracts (0.05, 0.95) deterministically. Validated mentally; injection tests will validate empirically.
