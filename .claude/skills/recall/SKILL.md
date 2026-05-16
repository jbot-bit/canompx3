---
name: recall
description: >
  Retrospective decision-quality lookup. Given a past decision, strategy, edge,
  hypothesis, or topic ($ARGUMENTS), pull from Pinecone (history), research-catalog
  MCP (hypothesis lineage + audit verdicts), docs/audit/results (verdict trail),
  gold-db MCP (current truth for the scope), and memory feedback files (lessons),
  then synthesize ONE brief: what was decided, on what evidence, what happened
  after, and whether current state still matches the decision. Use when:
  "recall", "what's the story on X", "why did we park/kill/promote X",
  "decision history on Y", "is verdict on X still holding".
effort: high
---

# Recall — Retrospective Decision Synthesis

You are given a topic, strategy id, edge name, hypothesis, or decision in `$ARGUMENTS`. Your job is to assemble a single synthesized brief from ALL of the following surfaces (in this order), then write a tight verdict.

## Surfaces (query in parallel where possible)

1. **`/nogo` first** — delegate the kill-verdict slice (PARK/KILL/NO-GO/UNSUPPORTED/DECAY/STALE) to the existing `/nogo` skill. Do NOT re-implement that filter here. Capture its output verbatim into the "Decision trail" section.
2. **research-catalog MCP — non-kill verdicts** — call `mcp__research-catalog__search_research_catalog` WITHOUT verdict_tags to capture VALIDATED / PROMOTE / CONDITIONAL records that `/nogo` filters out.
3. **docs/audit/results/** — grep for the topic; load matching MD verdicts that may not be indexed yet.
4. **gold-db MCP** — if the topic resolves to a strategy id or instrument/session/entry-model triple, call `mcp__gold-db__get_strategy_fitness` for CURRENT fitness (FIT/WATCH/DECAY/STALE). Do NOT cite memory.
5. **memory feedback files** — grep `memory/feedback_*.md` for lessons referencing the topic.
6. **HANDOFF.md + docs/plans/** — check whether the topic appears in active work; flag if HANDOFF still points to it.
7. **pinecone-assistant (optional)** — only if surfaces 1-6 returned thin results AND the assistant is configured. Skip silently otherwise.

## Synthesis output (use this exact structure)

```
## Recall: <topic>

### Decision trail
- <date> — <verdict> (source: <file or MCP>)
- <date> — <revision/amendment> (source: ...)

### Evidence cited at decision time
- <literature/canonical-data/audit-result> — <one-line summary>

### Current truth (live query)
- Fitness: <FIT/WATCH/DECAY/STALE/N/A> (gold-db, queried <now>)
- Allocator state: <DEPLOYED/PAUSED/NOT_PRESENT> (lane_allocation.json)
- Open hypothesis: <yes/no, status> (research-catalog)

### Contradictions / staleness
- <any place where the decision and current truth disagree, OR where audit
  results contradict each other, OR where HANDOFF still references a closed
  topic>. If none, say "None found."

### Lessons captured (memory)
- <feedback_*.md slug>: <one-line>

### Verdict on the verdict
- [HOLDS] decision still aligned with current truth
- [STALE] current truth diverges — needs re-review
- [SUPERSEDED] newer verdict overrides; cite which
- [INSUFFICIENT_EVIDENCE] no canonical record found; treat as unverified
```

## Rules

- **NEVER** cite numbers from memory. Always live-query gold-db / canonical files.
- **NEVER** invent verdicts. If a surface returns nothing, write `(no record)`.
- **Volatile data** (fitness, allocator state, sample sizes) MUST come from MCP/file reads inside this call.
- If the topic is ambiguous (e.g. just an instrument name), ask ONE clarifying question before running.
- If pinecone-assistant is not configured / API key missing, skip it silently — note `pinecone: unavailable` in output.
- Output stays under ~40 lines. This is a brief, not a report.

## Anti-patterns

- Confabulating that "research-catalog says X" without actually calling the MCP.
- Quoting feedback files as if they're decision records — they are LESSONS, not verdicts. The verdict lives in `docs/audit/results/`.
- Recommending action. This skill is retrospective only. If user wants "should I do X next" → suggest a follow-up call to `/capital-review` or `/research`.

## Example invocations

- `/recall PD_theory_grant`
- `/recall NYSE_OPEN RR1.5`
- `/recall E2 break-bar look-ahead`
- `/recall why did we park MES_AUG_RR2`
