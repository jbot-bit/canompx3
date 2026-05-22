task: TRIVIAL IMPLEMENTATION — silence pysignalr.transport INFO log that emits the SignalR negotiation URL containing the access_token JWT. Adds one helper `_silence_pysignalr_negotiation_log()` to `scripts/run_live_session.py` and one call site at the top of `main()` before any pysignalr import is triggered through the broker factory.
mode: CLOSED
scope_lock:
  - scripts/run_live_session.py

## Blast Radius
- scripts/run_live_session.py — adds one helper function (~20 lines including docstring) and one call line at top of main(). No production trading logic touched. No schema, config, or canonical-source change.
- Effect on logs: pysignalr.transport INFO suppressed (negotiation-URL line, State change trio, handshake lines). WARNING/ERROR from pysignalr unchanged. Orchestrator-side INFO ("Connected to ProjectX Market Hub", "Subscribed to quotes: ...") still surfaces feed connectivity.
- Reads: standard logging only. Writes: none (logger-level config only).
- Pre-flight library source verified: pysignalr/transport/websocket.py:41 declares `_logger = logging.getLogger('pysignalr.transport')`, line 320 logs the negotiation URL at INFO, line 211 logs connection-closed at WARNING (preserved).
