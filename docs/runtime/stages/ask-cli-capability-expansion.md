---
task: Expand ask CLI OpenRouter capability surface (web search, reasoning, multimodal, fallback, stdin, model filters, history, safety wiring)
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/ask.py
  - tests/test_tools/test_ask_cli.py
  - docs/runtime/stages/ask-cli-capability-expansion.md
---

## Blast Radius

- scripts/tools/ask.py — caller-facing CLI; sole consumer is operator (`ask "..."`) and `ask.bat` launcher. No `pipeline/` or `trading_app/` import depends on it.
- tests/test_tools/test_ask_cli.py — pytest tests for the CLI. Update to cover new flags (--web/--think/--image/--pdf/--pipe/--models filters/--models-fallback).
- Reads: existing `trading_app.ai.provider_registry` (`assert_openrouter_research_profile`, `list_openrouter_research_profiles`) — no changes there. `pipeline.paths.GOLD_DB_PATH` for grounded path — unchanged.
- Writes: none beyond stdout/stderr and `.cache/openrouter_models.json` (existing) + `.cache/ask_history.jsonl` (new, REPL-only, opt-in via `/save`).
- Network: `httpx` to `https://openrouter.ai/api/v1/{models,chat/completions}` only — same surface the file already uses.

## Acceptance criteria

1. Chat path no longer hardcodes `deepseek/deepseek-chat`. Defaults to `CANOMPX3_AI_CHAT_MODEL` env, falls back to literal default.
2. Chat path passes OpenRouter `provider` block with `data_collection: deny` and `allow_fallbacks: false` by default (overridable with `--allow-fallbacks`).
3. `--web` activates the web plugin (`plugins=[{"id":"web", "max_results":N, "engine":<eng>}]`); `--web-engine`, `--web-results` configurable.
4. `--think` enables reasoning. Effort flag `--effort {minimal,low,medium,high,xhigh}`; or `--reasoning-tokens N` for max_tokens form. Mutually exclusive.
5. `--image PATH` (repeatable) and `--pdf PATH` (repeatable) attach base64-encoded files to the user message content array. Text precedes files.
6. `--models-fallback m1,m2,m3` sets top-level `models` array for OpenRouter automatic failover. `--route fallback` available without explicit array.
7. `--pipe` reads stdin as the question; if both stdin and positional question are given, positional question is the lead-in and stdin appended.
8. `--models` flag gains filters: `--tools-only`, `--reasoning-only`, `--min-ctx N`, `--max-prompt-cost X`, `--max-completion-cost X`, `--provider P`, `--refresh`.
9. REPL gains `/save NAME`, `/load NAME`, `/history` commands persisting to `.cache/ask_history/`.
10. Reasoning content (delta.reasoning / delta.reasoning_details) is streamed to stderr in dim formatting when `--think` is on.
11. All existing tests still pass. New tests cover ≥1 happy-path per new flag (dry-run / parser-level).
12. `python pipeline/check_drift.py` clean.

## Verification plan

- pytest `tests/test_tools/test_ask_cli.py -v`
- `python pipeline/check_drift.py`
- Smoke: `python scripts/tools/ask.py --dry "hello"` ; `--dry --web "x"` ; `--dry --think "x"` ; `--dry --image fake.png "x"`.
- Self-review: seven sins scan + fail-closed scan + dead-code grep on ask.py before commit.
