## Iteration: 218
## Target: trading_app/ai/provider_registry.py:203,224,248,267
## Cluster: 4 findings, types=[annotation_debt/DRY], severity=[LOW, LOW, LOW, LOW]
## Classification: [mechanical]
## Blast Radius: 1 file (provider_registry.py only); 6 callers import from this module but none reference base_url directly
## Invariants: all 4 OpenRouter profiles retain base_url="https://openrouter.ai/api/v1"; no behavior change; PROFILE_REGISTRY shape unchanged; 19 tests must pass
## Diff estimate: 6 lines (add 1 constant, replace 4 string literals)
## Doctrine cited: integrity-guardian.md § 2 (DRY on infrastructure constants); S3 pattern (avoid scattered literals)
## Findings deferred: none — single cluster, all LOW
