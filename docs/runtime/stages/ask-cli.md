---
task: ADHD-friendly `ask` CLI wrapping OpenRouter runtime — standalone terminal use, multi-mode (grounded research / free chat / coding / planning), conversational not single-shot
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/ask.py
  - scripts/tools/ask_chat.py
  - ask.bat
  - tests/test_tools/test_ask_cli.py
  - docs/runtime/stages/ask-cli.md
---

## Blast Radius

- `scripts/tools/ask.py` — new file, no callers, thin wrapper over `trading_app.ai.openrouter_runtime.run_openrouter_task`
- `scripts/tools/ask_chat.py` — new file, free-chat mode that hits OpenRouter directly without the canonical packet (for coding / casual questions where the research packet is overkill)
- `ask.bat` — new file at repo root, launcher matching `codex.bat`/`claude.bat` pattern. Double-clickable, keeps window open on no-args.
- `tests/test_tools/test_ask_cli.py` — new test file, mocks httpx + checks env loading, profile aliasing, dry-run path, and free-chat path
- Reads: `.env`, `pipeline.paths.GOLD_DB_PATH`, `trading_app.ai.openrouter_runtime`. Writes: stdout/stderr only.
- No `pipeline/` or `trading_app/` modifications. Pure additive surface. Reuses canonical OpenRouter runtime + provider registry.
