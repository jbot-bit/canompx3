# MotherDuck MCP — proposed `.mcp.json` patch

**Status:** UNCOMMITTED. This patch is documentation only. It will be applied to
`.mcp.json` ONLY after the eval verdict is GO (see `eval-rubric.md` aggregate scoring
and `discipline-checklist.md` checks).

**Server:** [`mcp-server-motherduck`](https://github.com/motherduckdb/mcp-server-motherduck)
— a stdio MCP server that exposes a DuckDB connection. Pointed at a local file with
`--db-path`, it runs entirely offline; no MotherDuck cloud auth required.

---

## Proposed patch

Add this block under the top-level `mcpServers` key in `.mcp.json`:

```json
{
  "mcpServers": {
    "motherduck-eval": {
      "command": "uvx",
      "args": [
        "--constraints", "C:/Users/joshd/canompx3/constraints.txt",
        "mcp-server-motherduck",
        "--db-path", "C:/Users/joshd/canompx3/gold.db.eval",
        "--read-only"
      ],
      "env": {
        "HOME": "C:/Users/joshd",
        "MOTHERDUCK_TOKEN": "",
        "DUCKDB_READ_ONLY": "1"
      }
    }
  }
}
```

### Why each piece

- **`uvx` (not `pip install`)**: ephemeral isolated venv per launch, matches existing
  pattern for `code-review-graph`. No global pollution of the project venv.
- **`--constraints C:/Users/joshd/canompx3/constraints.txt`**: pins `cryptography<47`
  per `memory/feedback_mcp_venv_drift_cryptography47.md`. `cryptography==47` removed
  `hazmat.backends`, which Authlib 1.7 (transitive dep of FastMCP servers) still
  imports — without the constraint, the server crashes on first connect.
- **`--db-path C:/Users/joshd/canompx3/gold.db.eval`**: SNAPSHOT path, never the live
  `gold.db`. Per `pipeline.paths.GOLD_DB_PATH` discipline + drift check #62, the live
  DB stays untouched.
- **`--read-only`**: server-level enforcement of read-only. DuckDB will reject any
  `INSERT` / `UPDATE` / `CREATE` / `DROP` at the connection layer.
- **`MOTHERDUCK_TOKEN=""`**: explicitly empty — we are NOT using the cloud product. The
  server falls back to local-file mode when the token is empty.
- **`DUCKDB_READ_ONLY=1`**: belt-and-braces; some versions of the server honour this
  env var as a secondary read-only check.

---

## What this patch does NOT do

- Does NOT replace `gold-db` MCP. `gold-db` keeps its templated, schema-aware,
  documented surface for repeatable workflows. MotherDuck MCP is the **ad-hoc fallback
  for one-off questions**.
- Does NOT point at the live `gold.db`. If a future evaluation suggests MotherDuck for
  live queries, that's a separate decision with its own discipline check.
- Does NOT enable MotherDuck cloud sync. `MOTHERDUCK_TOKEN` stays empty.

---

## Application checklist (post-GO only)

1. Re-run all four checks in `discipline-checklist.md` — all must PASS.
2. Add the block above to `.mcp.json` (project scope), commit on a new branch.
3. **Verify no local-scope shadow:** per
   `memory/feedback_mcp_local_scope_shadows_project_scope.md`, run
   `claude mcp get motherduck-eval` and confirm scope is `project`. If `local`, run
   `claude mcp remove motherduck-eval -s local` first.
4. Restart Claude Code (`.mcp.json` env edits don't hot-reload — see
   `feedback_mcp_env_requires_restart.md`).
5. Rebuild snapshot from latest `gold.db` if more than 7 days old at activation time.
6. Smoke test: ask the server to run `SELECT COUNT(*) FROM bars_1m`. Expect
   `20,513,435` (snapshot row count as of 2026-05-01) or higher if re-snapshotted.
