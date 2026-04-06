## Iteration: 159
## Target: trading_app/ai/corpus.py:55
## Finding: Missing corpus file returns "[MISSING:]" sentinel with no log warning — silent failure
## Classification: [mechanical]
## Blast Radius: 2 callers (query_agent.py, mcp_server.py), 1 test file (test_corpus.py)
## Invariants: load_corpus() return shape unchanged; sentinel value unchanged; callers unaffected
## Diff estimate: 3 lines (import logging + warning call)
