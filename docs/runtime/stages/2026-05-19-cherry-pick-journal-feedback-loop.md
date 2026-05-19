# Stage — Cherry-pick journal feedback loop

task: Improvement 1 of research-fetch-improvements plan — add structured
journal write/enrich path so the cherry-pick ranker can eventually learn from
heavyweight outcomes. Closes the era-stability proxy weight=0 placeholder by
providing the data substrate; v1 keeps the weight at 0 per
`feedback_meta_tooling_n1_tunnel_2026_05_01.md`.

mode: IMPLEMENTATION

## Scope Lock

- scripts/research/cherry_pick_ranker.py
- scripts/research/cherry_pick_journal_enricher.py
- docs/runtime/cherry_pick_journal.yaml
- pipeline/check_drift.py
- tests/test_research/test_cherry_pick_journal.py
- docs/runtime/cherry_pick_journal.md

## Blast Radius

- `scripts/research/cherry_pick_ranker.py` — adds `--write-journal` flag; no
  caller of `rank_queue_entries()` changes signature; default behaviour
  unchanged for existing operators.
- `scripts/research/cherry_pick_journal_enricher.py` — NEW. Read-only over
  `docs/audit/results/`, write-only to `docs/runtime/cherry_pick_journal.yaml`.
  Zero callers.
- `docs/runtime/cherry_pick_journal.yaml` — NEW structured journal. Existing
  markdown stub stays for human reading; YAML is the data substrate.
- `pipeline/check_drift.py` — adds Check #164 (journal entry integrity).
- Reads: `docs/runtime/promote_queue.yaml`, `docs/audit/results/*.md`.
- Writes: `docs/runtime/cherry_pick_journal.yaml` (append-only).
- No capital-path mutation. No edit under `trading_app/`, `pipeline/`
  (other than check_drift.py), `docs/runtime/lane_allocation.json`,
  or `docs/runtime/chordia_audit_log.yaml`.

## Verification

1. `python pipeline/check_drift.py` passes.
2. `python -m pytest tests/test_research/test_cherry_pick_journal.py -v`
   shows all-green; injection probes confirm Check #164 catches stale entries.
3. `python scripts/research/cherry_pick_ranker.py --write-journal --dry-run`
   round-trip: confirms journal entry shape.
4. `python scripts/research/cherry_pick_journal_enricher.py --dry-run`
   confirms enricher picks up existing matching result MDs.
