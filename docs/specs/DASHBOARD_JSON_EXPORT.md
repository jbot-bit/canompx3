# Spec: dashboard.py JSON Export

## What
Add `--json` flag to `pipeline/dashboard.py` that outputs raw metrics as JSON instead of HTML.

## Why
Enable programmatic consumption of pipeline status for monitoring, diffing, and AI context loading. Currently the only output is a rendered HTML file — no machine-readable format exists.

## Current Architecture (do not change)
`dashboard.py` already separates data collection from rendering:

```
8 collect_*() functions → return dicts/lists → main() gathers them → render_dashboard() → HTML
```

| Collector | Returns | Source |
|-----------|---------|--------|
| `collect_db_metrics(db_path)` | `dict` | gold.db row counts, date ranges, DB size |
| `collect_checkpoint_progress(cp_dir)` | `dict` | Ingestion checkpoint status |
| `collect_file_inventory(dbn_dir)` | `dict` | .dbn.zst file counts |
| `collect_guardrail_status()` | `dict` | Drift check + test pass/fail |
| `collect_contract_history(db_path)` | `list[dict]` | Roll dates, volume by contract |
| `collect_data_quality(db_path)` | `dict` | OHLCV sanity, gap analysis |
| `collect_strategy_metrics(db_path)` | `dict` | Validated/experimental strategy counts |
| `collect_roadmap_status(roadmap_path)` | `list[dict]` | Phase status from ROADMAP.md |

## Implementation

### CLI change
```
python pipeline/dashboard.py --json                    # stdout
python pipeline/dashboard.py --json --output snap.json # file
```

`--json` and default HTML are mutually exclusive. When `--json` is used without `--output`, write to stdout (pipeable). When `--output` is provided with `--json`, write to that file path.

### JSON structure
```json
{
  "generated_at": "2026-02-26T14:30:00+10:00",
  "db_metrics": { ... },
  "checkpoint_progress": { ... },
  "file_inventory": { ... },
  "guardrail_status": { ... },
  "contract_history": [ ... ],
  "data_quality": { ... },
  "strategy_metrics": { ... },
  "roadmap_status": [ ... ]
}
```

Keys map 1:1 to collector function names. The dict values are the EXACT return values from each `collect_*()` function — no transformation, no filtering, no reshaping.

### Serialization
Use `json.dumps(payload, indent=2, default=str)` — the `default=str` handles any `datetime`, `Path`, or `Decimal` objects that collectors might return. No custom encoder needed.

### What NOT to do
- Do not refactor the collect functions
- Do not add new collectors
- Do not change the HTML output path
- Do not add dependencies

## Files touched
- `pipeline/dashboard.py` — add `--json` arg, branch in `main()` before `render_dashboard()` call

## Estimated size
~15 lines changed in `main()`.
