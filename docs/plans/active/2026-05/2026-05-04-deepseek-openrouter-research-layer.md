# DeepSeek / OpenRouter Research Layer

**Date:** 2026-05-04
**Status:** active foundation landed
**Scope:** repo-native research/planning surface only

## Goal

Use DeepSeek where it is strongest for this repo without creating a second AI
truth layer, a hidden fallback chain, or a mutation-capable sidecar.

The system must:

- start from canonical repo context
- stay grounded in repo doctrine and local literature
- remain read-only in v1
- keep model/provider settings centralized
- expose a measurable, explicit request contract rather than implicit magic

## Official-doc basis

The provider-layer assumptions for this slice are grounded in official docs,
not model self-description:

- **OpenRouter provider routing docs**
  - `provider.order`
  - `provider.allow_fallbacks`
  - `provider.require_parameters`
  - `provider.data_collection`
- **OpenRouter structured outputs docs**
  - `response_format` with JSON schema for structured profiles
- **DeepSeek official API docs**
  - OpenAI/Anthropic-compatible API surface
  - Anthropic-format compatibility path
  - tool schema compatibility and thinking-mode support

This slice does **not** assume DeepSeek should run live trading, HFT, or
execution decisions. That claim is unsupported for this repo.

## Repo-native design

The canonical extension points are the existing repo surfaces:

- `context/registry.py`
- `pipeline/system_brief.py`
- `scripts/tools/context_views.py`
- `trading_app/ai/corpus.py`
- `trading_app/mcp_server.py`
- `docs/ai-context/LOCAL_MODEL_CONTEXT.md`
- `chatgpt_bundle/00_INDEX.md`
- `docs/institutional/literature/*.md`

The DeepSeek/OpenRouter layer must consume those surfaces. It must not create:

- a separate prompt truth layer
- duplicated threshold/session/cost tables
- mutation authority
- raw SQL write authority
- hidden provider fallbacks

## Landed foundation

### 1. Canonical provider/profile registry

`trading_app/ai/provider_registry.py`

Centralizes:

- Anthropic profile pins for current Claude query flow
- OpenRouter-backed DeepSeek research profiles
- provider-routing defaults
- reasoning/response-mode settings
- profile-level allowed tool sets

DeepSeek/OpenRouter profiles require explicit model configuration through env.
The repo does **not** silently pick a model.

### 2. Generated research packet

`trading_app/ai/research_packet.py`

Builds a task-routed packet from canonical repo surfaces:

- system brief
- route doctrine and canonical owners
- generated context views
- corpus inventory
- local literature references
- read-only contract

### 3. Repo-native launcher/export path

`scripts/tools/render_ai_research_packet.py`

Outputs:

- JSON packet
- Markdown packet
- OpenRouter request scaffold

### 4. MCP-facing read-only exposure

`trading_app/mcp_server.py`

Adds a read-only `get_ai_research_packet` surface so the packet is available
through an existing canonical repo tool boundary.

### 5. Drift guard

`pipeline/check_drift.py`

Adds guardrails so OpenRouter defaults and DeepSeek model IDs do not drift into
scattered Python files outside `trading_app/ai/provider_registry.py`.

## Current boundaries

### Included

- planning
- research synthesis
- long-context repo understanding
- structured extraction scaffolding
- read-only DB/context usage

### Excluded

- live trading execution
- broker/session control
- code mutation by DeepSeek
- self-hosting
- fine-tuning
- portfolio execution agents

## Follow-on work

1. Add a small eval harness for repo-local task routing, citation quality, and
   refusal behavior.
2. Decide explicit recommended OpenRouter model IDs per profile from measured
   repo tasks, not model marketing.
3. Reduce the existing `.env` parse noise so packet rendering is cleaner in
   operator shells.
