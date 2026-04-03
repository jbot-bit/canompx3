## Iteration: 140
## Target: pipeline/run_full_pipeline.py:123
## Finding: Hardcoded "--min-sample=30" string in step_validate subprocess command should use REGIME_MIN_SAMPLES from trading_app.config
## Classification: [judgment]
## Blast Radius: 1 production file (run_full_pipeline.py), 1 test file (test_full_pipeline.py — no change needed, test doesn't assert step_validate command contents)
## Invariants: [1] subprocess command to strategy_validator.py must pass --min-sample; [2] value passed must equal REGIME_MIN_SAMPLES; [3] no other step functions change
## Diff estimate: 2 lines
