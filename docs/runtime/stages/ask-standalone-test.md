---
task: Land standalone `ask` CLI test (`tests/test_ask_standalone.py`) — follow-on from PRs #222/#223 testing the portable `~/.canompx-ask/` install
mode: TRIVIAL
scope_lock:
  - tests/test_ask_standalone.py
  - docs/runtime/stages/ask-standalone-test.md
---

## Blast Radius

- `tests/test_ask_standalone.py` — new test file (321 lines), already on disk untracked. Targets `~/.canompx-ask/ask.py` standalone CLI installed by `install_ask.py`. Auto-skips when `~/.canompx-ask/ask.py` not present (CI-safe).
- No production code touched. No `pipeline/` or `trading_app/` edits. Pure test surface.
- Reads: `~/.canompx-ask/ask.py` and `install_ask.py` via `importlib.util` only when present. All filesystem state goes through `tmp_path` per the test docstring.

## Acceptance

- `pytest tests/test_ask_standalone.py -q` passes locally (or skips cleanly on machines without the standalone install).
- File committed; `pipeline/check_drift.py` passes.

## Cleanup of stale `ask-cli` stage

Old `docs/runtime/stages/ask-cli.md` stub is from before PRs #222 and #223 shipped. Acceptance was never specified; scope file `scripts/tools/ask_chat.py` was never built (chat mode consolidated into `ask.py`). Stage is closed by the merged PRs and is deleted as part of this trivial close.
