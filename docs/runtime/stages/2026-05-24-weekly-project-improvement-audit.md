---
task: |
  IMPLEMENTATION - Weekly Project Improvement Audit Scanner. Add a deterministic
  repo evidence collector that supports a recurring weekly Codex audit. The tool
  gathers current project health signals, emits a compact evidence packet, and
  never mutates repo-tracked files by default. The recurring automation/model
  ranks findings from that evidence; the script itself does not make judgment
  calls beyond severity normalization.
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/weekly_project_audit.py
  - tests/test_tools/test_weekly_project_audit.py
  - .codex/AUTOMATIONS.md
---

## Blast Radius

- `scripts/tools/weekly_project_audit.py` - new read-only CLI evidence collector over git, gh, project_pulse, work_queue, stage files, branch protection, and GitHub security endpoints. Stdout only by default.
- `tests/test_tools/test_weekly_project_audit.py` - mocked command/file tests for packet schema, failure handling, and no-write default behavior.
- `.codex/AUTOMATIONS.md` - add the weekly Project Improvement Audit template and exact recurring prompt.
- Reads local repo state, GitHub metadata via `gh`, and automation memory timestamps when provided. Writes nothing unless an explicit `--out` path is supplied.
- No trading logic, research thresholds, database writes, allocator/profile mutation, broker state, live runtime state, or `.claude/` changes.

## Acceptance

- `python scripts/tools/weekly_project_audit.py --format markdown` prints a compact evidence packet and exits 0 when optional GitHub endpoints are unavailable.
- `python scripts/tools/weekly_project_audit.py --format json` emits stable JSON with `generated_at`, `repo`, `git`, `prs`, `ci`, `live_readiness`, `security`, `workflow`, `carryovers`, and `recommended_attention_inputs`.
- Default run does not modify tracked files. Tests assert no writes without `--out`.
- GitHub API failures are reported as `unknown` with endpoint/error, not silently swallowed.
- Weekly automation prompt in `.codex/AUTOMATIONS.md` consumes the evidence packet and outputs: verdict, top 5 EV-ranked findings, evidence path, risk, smallest safe fix, what not to do, and one next patch scope.
