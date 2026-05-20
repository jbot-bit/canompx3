---
task: |
  Stage A IMPLEMENTATION — Idea Ingestion Front-Door for the Fast-Lane Chain.
  Builds scripts/research/ingest_idea.py: a CLI emitter that turns structured
  CLI arguments + interactive mechanism/literature prompts into a
  fast-lane v5.1-compliant pre-reg YAML under docs/audit/hypotheses/drafts/.
  Closes the upstream gap on the "idea in → live trade out" automation goal —
  today the chain has no entry point except hand-writing 220 lines of YAML.
  Stage A writes to drafts/ only; operator promotes to docs/audit/hypotheses/
  manually after review (same quarantine pattern as the bridge per
  memory/feedback_lhp_validator_vs_field_presence_trap_n1.md).
mode: IMPLEMENTATION
scope_lock:
  - scripts/research/ingest_idea.py
  - tests/test_research/test_ingest_idea.py
  - docs/runtime/stages/2026-05-20-stage-a-ingest-idea.md
---

## Blast Radius

- `scripts/research/ingest_idea.py` — NEW file. Zero production-code touched. Imports canonical surfaces read-only: `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`, `pipeline.dst.SESSION_CATALOG`, `trading_app.config.ALL_FILTERS/ENTRY_MODELS/E2_EXCLUDED_FILTER_PREFIXES/E2_EXCLUDED_FILTER_SUBSTRINGS`, `scripts.research.lhp.literature_index.load_corpus/citation_exists`, `scripts.research.lhp.yaml_emitter.write_draft`.
- `tests/test_research/test_ingest_idea.py` — NEW file. 3 tests: schema parity vs an existing fast-lane v5.1 pre-reg, refusal delegation (E2+banned filter, unknown filter, unknown session, unknown instrument), literature-citation existence check.
- Reads only: `docs/institutional/literature/*.md` (citation existence check), `docs/audit/hypotheses/2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.yaml` (schema parity baseline in test).
- Writes only: `docs/audit/hypotheses/drafts/<date>-<slug>-fast-lane-v1.draft.yaml`. Same boundary as `fast_lane_to_heavyweight_bridge.py`. Not capital-class — does NOT touch `chordia_audit_log.yaml`, `validated_setups`, `lane_allocation.json`, `trading_app/live/`.
- No drift check added in this stage (n=1 forcing-function ban per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`). If a parity-drift class emerges (n≥2), Stage A.1 adds Check #174.

## Predecessor and Gate Status

- Fast-lane chain (Stages 2A.1 → 2A.3): all CLOSED on main (`bdb04ca6` 2026-05-20).
- TEMPLATE-fast-lane-v5.1.yaml exists at `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`.
- 6 in-the-wild fast-lane v5.1 pre-regs from 2026-05-18 confirm output schema.
- Evidence-auditor pre-review applied 5 corrections to original design:
  1. Delegate refusal to bridge + lhp validators (no parallel inline re-encoding)
  2. No fcntl locking (Unix-only, would crash on Windows)
  3. No numeric `chordia_gate_threshold` (emit prose only, runner resolves at execution)
  4. Remove `theory_citation_pending_audit_log_approval` (loader Amendment 3.3 already fixed via `theory_grant: false`)
  5. Add `--confirm-bars` + `--direction` + `--orb-label` to CLI (required scope fields)

## Refusal Gates (all delegated to canonical sources)

| # | Gate | Delegate to |
|---|---|---|
| 1 | Instrument in ACTIVE_ORB_INSTRUMENTS | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| 2 | Session in SESSION_CATALOG | `pipeline.dst.SESSION_CATALOG` |
| 3 | Entry model in {E1, E2} (E0/E3 banned for fast-lane v5.1) | `trading_app.config.ENTRY_MODELS` + hard-coded fast-lane v5.1 set |
| 4 | Filter exists in ALL_FILTERS | `trading_app.config.ALL_FILTERS` |
| 5 | E2 + banned filter (look-ahead class) | `trading_app.config.E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS` |
| 6 | Literature citation resolves to docs/institutional/literature/ | `scripts.research.lhp.literature_index.load_corpus` + `citation_exists` |
| 7 | Mechanism text non-empty | local check (no canonical source — this is the new gate added by Stage A) |
| 8 | Direction in {pooled, long, short} | local enum (matches fast-lane v5.1 schema) |

Gates that are NOT in Stage A (deferred to bridge / static_checks / scanner):
- structural_hash graveyard match — runs at fast-lane scan time after `--run-fast-lane`, not at ingestion
- K_lane >= 2 sibling-retest — same as above
- OOS power floor — runs in the fast-lane runner itself

Rationale: Stage A is the **upstream front-door**; the graveyard / K_lane / OOS gates are scan-time gates that need the trial ledger + scan results, which don't exist yet at ingestion. The bridge already enforces them downstream.

## CLI Signature

```
python scripts/research/ingest_idea.py \
  --instrument {MES,MGC,MNQ} \
  --session <SESSION_CATALOG key> \
  --orb-minutes {5,15,30} \
  --entry {E1,E2} \
  --confirm-bars N \
  --rr <float> \
  --direction {pooled,long,short} \
  --filter <ALL_FILTERS key> \
  --mechanism "<one-line economic mechanism>" \
  --literature <slug-from-literature-corpus> \
  [--purpose "<one-line purpose statement>"]
```

All flags required except `--purpose` (defaulted to a templated string).

If any required flag missing: argparse exits 2.
If any gate fails: stderr + exit code 3, no file written.
If all gates pass: writes to `docs/audit/hypotheses/drafts/<date>-<slug>-fast-lane-v1.draft.yaml`, prints path to stdout, exit 0.

## Acceptance Criteria

1. `python scripts/research/ingest_idea.py --help` returns a non-empty CLI usage table.
2. `python pipeline/check_drift.py` still passes (count unchanged from current main — Stage A adds no drift check).
3. 3 tests pass: schema parity, refusal delegation, literature citation existence.
4. Smoke test: run ingest_idea against the same scope as an existing 2026-05-18 fast-lane pre-reg, diff the output — only `name`, `date_locked`, paths, and ID-string fields should differ; structural schema must match.
5. Drift sweep run: `python pipeline/check_drift.py` returns 0.
6. Dead code: `grep -r "ingest_idea" --include="*.py"` shows only the new file + its test.

## Out of Scope

- Auto-promoting drafts/ to docs/audit/hypotheses/ (operator decision)
- Running the fast-lane chain itself (separate command: `python scripts/tools/fast_lane_walk.py`)
- Theory-grant authoring (always emits `theory_grant: false`; operator manually upgrades after audit-log entry)
- Capital-class writes (Stage C — separate, requires `/capital-review`)
- Adding a drift check for emitter/template parity (deferred until n≥2 per `feedback_meta_tooling_n1_tunnel_2026_05_01.md`)
