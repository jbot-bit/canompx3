## Iteration: 149
## Target: trading_app/live/bot_dashboard.py:501
## Finding: api_sessions() passes date.today() (system local date) to DST resolvers instead of now_bris.date() (Brisbane date) — incorrect during NYSE_OPEN midnight crossing period
## Classification: [judgment]
## Blast Radius: 1 file (bot_dashboard.py), display-only endpoint, no production trading path
## Invariants:
##   1. api_sessions() JSON shape unchanged (sessions list + next field)
##   2. DST resolver interface resolver(date) -> tuple[int, int] unchanged
##   3. Session sort order by minutes_away unchanged
## Diff estimate: 3 lines
## Secondary fix: line 70 silent except Exception: pass on heartbeat parse failure → add log.warning [mechanical]
