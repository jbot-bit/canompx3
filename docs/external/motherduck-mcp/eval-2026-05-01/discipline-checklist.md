# MotherDuck MCP Eval — Discipline Checklist

Four institutional-rigor checks. The eval verdict can only flip to GO if every check
PASSes. Source: `.session/mcp-eval-plan-2026-05-01.md` requirement (token-reduction
threshold) + project canon (read-only DB rule, snapshot rule, cryptography pin).

---

## Check 1 — DB read-only enforcement

**Goal:** confirm the running MCP server cannot write to `gold.db.eval`. Belt-and-braces:
the server is launched with `--read-only`, but we verify by attempting a write through
the MCP tool surface.

**Procedure:**
1. With the server running and connected, ask the MCP server to execute:
   ```sql
   CREATE TABLE _readonly_probe AS SELECT 1 AS x;
   ```
2. Expected: an error returned by the MCP server, surfaced as a tool-call failure. Error
   text should mention "read-only" or "Cannot execute statement".
3. Then attempt:
   ```sql
   INSERT INTO bars_1m SELECT * FROM bars_1m LIMIT 1;
   ```
   Same expectation: tool-call failure with read-only error.
4. Verify file mtime on `gold.db.eval` is unchanged after both attempts.

**PASS** = both writes rejected AND mtime unchanged. **FAIL** = either write succeeds OR
mtime advances.

---

## Check 2 — Constraint pin (cryptography<47)

**Goal:** prevent the venv-drift class documented in
`memory/feedback_mcp_venv_drift_cryptography47.md`. `cryptography==47` removed
`hazmat.backends`; Authlib 1.7 (transitive of FastMCP) still imports it. If `uvx`
resolves cryptography unconstrained, the server crashes on first request.

**Procedure:**
1. Ensure `mcp-config-patch.md` includes `--constraints C:/Users/joshd/canompx3/constraints.txt`
   in the `args` array.
2. Confirm `constraints.txt` pins `cryptography<47`:
   ```bash
   grep -i '^cryptography' C:/Users/joshd/canompx3/constraints.txt
   ```
   Expect a line like `cryptography<47`.
3. Launch the server (or `uvx --constraints ... mcp-server-motherduck --help` as a dry
   probe) and inspect the resolved env via `uv pip list` against the ephemeral venv.
   Cryptography version must be `< 47`.
4. Run the smoke query (Check 1's first benign read: `SELECT COUNT(*) FROM bars_1m`) and
   confirm no `ImportError: cannot import name 'backends' from 'cryptography.hazmat'`.

**PASS** = constraint flag present, resolved version <47, smoke query succeeds.
**FAIL** = any of the three subchecks fail.

---

## Check 3 — Snapshot path verification (never the live DB)

**Goal:** the server must NEVER be pointed at `gold.db`. The snapshot at `gold.db.eval`
is the only acceptable target during eval.

**Procedure:**
1. Inspect the launched server's command line via `ps` (or Windows equivalent
   `Get-CimInstance Win32_Process` filtered to `mcp-server-motherduck`). Confirm the
   `--db-path` argument resolves to `C:/Users/joshd/canompx3/gold.db.eval` exactly. No
   trailing space, no symlink redirection, no `gold.db` substring without `.eval`.
2. Confirm the snapshot file exists and the live file is distinct:
   ```bash
   ls -la C:/Users/joshd/canompx3/gold.db C:/Users/joshd/canompx3/gold.db.eval
   ```
   Two separate inodes, both with non-zero size.
3. Optional but recommended: take an mtime snapshot of `gold.db` before eval kickoff
   and confirm it is unchanged after the eval session ends. The live DB must not be
   touched at any point during the eval.
4. Confirm `.gitignore` excludes `gold.db.eval` (so the snapshot can never accidentally
   land in a commit).

**PASS** = `--db-path` is the snapshot, files are distinct, live `gold.db` mtime
unchanged across the eval, `.gitignore` covers `gold.db.eval`. **FAIL** = any subcheck
fails.

---

## Check 4 — Token-reduction threshold

**Goal:** the eval premise (MotherDuck MCP saves tokens vs the raw-Python fallback) must
be empirically true on the harness. Threshold per task plan: **≥30% token reduction on
≥3 of 5 questions**, no correctness regressions.

**Procedure:**
1. For each of Q1–Q5 in `eval-rubric.md`, run BOTH paths in a fresh Claude session
   (clean context per question to avoid cache contamination):
   - **Current path:** raw Python (or curated template where applicable).
   - **MotherDuck path:** the SQL provided in the rubric, executed via the MCP tool.
2. Record actual values for `tokens_current`, `tokens_motherduck`, `time_current_s`,
   `time_motherduck_s`, and `correctness` for each question. Use the harness's
   per-message token telemetry; sum across the multi-turn loop.
3. Compute per-question reduction: `(tokens_current - tokens_motherduck) / tokens_current`.
4. Aggregate:
   - Count questions with reduction ≥ 30%. Need **≥3 of 5**.
   - Count questions where MotherDuck `correctness = FAIL`. Need **0**.
   - PARTIAL is allowed but flagged for human review.
5. Fill in the aggregate table in `eval-rubric.md` and write a brief verdict block at
   the bottom: GO / NO-GO / NEEDS-MORE-DATA, with one-line reasoning per question.

**PASS** = ≥3 questions hit ≥30% reduction AND zero `FAIL` correctness AND no PARTIAL on
a question that is the ONLY thing pushing the count to 3. **FAIL** = any of those.

---

## Aggregate gate

GO if and only if all four checks PASS. Any FAIL → eval verdict is NO-GO and
`mcp-config-patch.md` is NOT applied. Document the failure mode in a follow-up file
under this directory and decide whether the eval is retryable (e.g. retry with a smaller
`--memory-limit`) or terminal (e.g. server architecturally writes to disk).
