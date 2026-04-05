## Iteration: 142
## Target: trading_app/prop_profiles.py:857
## Finding: parse_strategy_id hardcodes ("E1", "E2", "E3") instead of importing ENTRY_MODELS from trading_app.config — canonical violation
## Classification: [mechanical]
## Blast Radius: 2 callers (paper_trade_logger.py:74, prop_profiles.py:922), no behavior change since ENTRY_MODELS currently equals ["E1", "E2", "E3"]
## Invariants:
##   1. parse_strategy_id("MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100") must still return entry_model="E2"
##   2. Unknown entry model tokens must NOT be matched (behavior preserved by membership test)
##   3. No circular import (trading_app.config does not import from prop_profiles)
## Diff estimate: 3 lines (1 added import line, 1 changed line)
