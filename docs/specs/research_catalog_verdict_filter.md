# Research Catalog — Verdict-Tag Filter

**Status:** LIVE (2026-05-12)
**Owner:** `scripts/tools/research_catalog_mcp_server.py`
**Authority:** read-only proxy over canonical research surfaces. Adds **no** truth surface.

## What this spec covers

Extends the existing `research-catalog` MCP server (`search_research_catalog` tool) with:

1. A `verdict_tags` filter parameter on `search_research_catalog`.
2. Verdict detection on every `result`-kind artifact at index time (filename stem + markdown front-matter).
3. Indexing of `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO Registry table rows as `result`-kind artifacts so they are searchable alongside `docs/audit/results/`.

No new MCP server. No new write path. No external sidecar.

## Verdict tags

The canonical vocabulary — raw-spelling -> canonical-tag map, priority
resolution order, and compound-qualifier rules — lives in
`RESEARCH_RULES.md § Verdict Token Vocabulary` (promoted 2026-05-14).
Source-of-truth is the Python constants `_VERDICT_NORMALIZE` /
`_VERDICT_PRIORITY_TOKENS` in this server; the doctrine MD section is kept
in parity by drift check `check_verdict_vocabulary_md_matches_code`
(binding, `pipeline/check_drift.py`). Additions to either constant MUST
be mirrored in the doctrine MD in the same commit.

The compound-qualifier resolution algorithm (regex word-boundary scan,
priority-order tie-break, bold-marker stripping) is owned by
`_normalize_verdict` in this server and is intentionally not duplicated
in the doctrine. Unknown tags raise `ValueError` listing the valid set —
fail-closed.

The artifact's `metadata["raw_verdict"]` field preserves the original
spelling for traceability; `metadata["verdict"]` carries the normalized
canonical tag.

For implementation detail on how detection precedence (front-matter ->
body marker -> filename stem) intersects with the vocabulary, see
§ "Detection precedence" below.

### Why this matters (silent-gap class)

A first pass used exact-match-only normalization and silently dropped 14 of 37 BLUEPRINT
NO-GO Registry rows (caught 2026-05-12). Rows with qualifier patterns (`DEAD (qualifier)`,
em-dash combos, `GUILTY`, `ARITHMETIC_ONLY`) were invisible to `verdict_tags=["NO-GO"]`
queries. The token-priority scan closes that gap. New BLUEPRINT verdict spellings should be
added to `_VERDICT_NORMALIZE` and `_VERDICT_PRIORITY_TOKENS` together — never relax the
normalizer to "any unknown string passes."

## Detection precedence (per artifact, high → low)

1. **YAML front-matter / inline scalar** — `^verdict: <X>` line on its own.
2. **Body marker (canonical production format)** — `**Verdict:** TOKEN ...` /
   `**VERDICT: TOKEN**` / `**Verdict on X:** TOKEN ...`. The detector walks
   every matching line and returns the FIRST token that `_normalize_verdict`
   recognizes — handles audit-results that carry `Verdict on <subscope>:`
   informational headers before the load-bearing summary. Verified 2026-05-12
   against 31 production audit-result files using
   `grep -E "^\*\*[Vv]erdict[^:]*:\*\*" docs/audit/results/*.md`.
3. **Filename stem suffix** — `*-nogo.md`, `*-park.md`, `*-kill.md`,
   `*-unsupported.md`, `*-decay.md`, `*-stale.md`, `*-dead.md`
   (case-insensitive). Legacy fallback only.
4. **STRATEGY_BLUEPRINT.md §5 rows** — verdict comes from the table's Verdict
   column via `_normalize_verdict` (token-priority scan).

The body-marker layer was added 2026-05-12 to close the F-1 silent-functional
gap where the front-matter-only detector matched 0 of 240 production
audit-results. The fix lifted real audit-results with kill verdicts from
0 → 17 (9 KILL + 6 PARK + 2 NO-GO).

## Behavior contract

- Default `verdict_tags=None` → no filtering, fully backwards-compatible.
- When set, only artifacts whose `metadata["verdict"]` is in the (normalized) set are returned.
- Filter applies after scope filtering, before scoring — items with score 0 are still excluded.
- Response payload echoes the normalized filter as `verdict_tags` (sorted) so callers can confirm what was applied.

## STRATEGY_BLUEPRINT.md §5 indexing rules

- The parser locates the `## 5. NO-GO Registry` heading via `_heading_windows` and extracts only that section.
- It treats markdown table rows as data: row 1 = header, row 2 = `|---|---|` separator, row 3+ = NO-GO entries.
- Each kept row produces one `result`-kind artifact:
  - `artifact_id = "blueprint-nogo-<slug>"` (slug derived from the Path column)
  - `title = "BLUEPRINT NO-GO: <path-label>"`
  - `metadata = {verdict, source: "STRATEGY_BLUEPRINT.md", section: "5. NO-GO Registry", path, raw_verdict, reopen_criteria}`
- Rows whose Verdict cell does not normalize to a recognized canonical tag are skipped (e.g., `PROMISING`).

## What this spec does NOT do

- Does **not** create a new MCP server. The recurring class-bug "5th MCP" failure modes (`feedback_mcp_local_scope_shadows_project_scope`, `feedback_mcp_partial_install_state_2026_05_01`, `feedback_mcp_env_requires_restart`) are avoided by extending the existing surface in place.
- Does **not** index `memory/` files. There is no `*-nogo.md` slug convention in `memory/`; if recurring queries hit `memory/`, that is a separate change with its own evidence-of-demand bar.
- Does **not** add a write path. Trial-ledger / verdict updates remain git commits to `docs/audit/results/` and `docs/STRATEGY_BLUEPRINT.md`.
- Does **not** change scoring. Filter is post-scope, pre-scoring; ranking is unchanged.

## Verification

8 new unit tests in `tests/test_tools/test_research_catalog_mcp_server.py`:

- `test_search_filters_by_verdict_tag`
- `test_search_no_verdict_filter_returns_all` (regression guard)
- `test_artifact_index_parses_blueprint_nogo_subsections`
- `test_verdict_detected_from_filename_stem`
- `test_verdict_detected_from_frontmatter` (asserts disagreement: front-matter
  KILL beats filename `-park` suffix — F-2 fix from 2026-05-12 review)
- `test_verdict_detected_from_body_marker_production_format` (5 production
  shapes — F-1 fix from 2026-05-12 review)
- `test_compound_verdict_qualifiers_normalize_to_kill_tag` (regression guard
  for the silent-gap class)
- `test_search_rejects_unknown_verdict_tag` (fail-closed)

Live smoke (post-fix verdict distribution against real audit-results):

| Verdict | Count |
|---------|-------|
| KILL | 9 |
| PARK | 6 |
| NO-GO | 2 |
| Other recognized non-kill (`PASS` / `WEAK` / `CONTINUE` / `EDGE_WITH_CAVEAT` / `MARGINAL` / `VALIDATED` / `CLOSED` / `FIX` / `CONDITIONAL` / `HOLDING` / `REDESIGN` / `RESEARCH_PROVISIONAL` / `DOWNSIZE` / `NULL`) | 17 |
| None (no recognized verdict marker) | 205 |

Pre-fix distribution was `{None: 240}` — F-1 silent-functional gap.

Live smoke after Claude restart (`feedback_mcp_env_requires_restart`):

```
mcp__research-catalog__search_research_catalog query="aperture" verdict_tags=["NO-GO"]
```

Expect items including BLUEPRINT-sourced NO-GO entries.
