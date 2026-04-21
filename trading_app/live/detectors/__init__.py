"""Phase 6e monitoring detectors.

Each detector is a pure function: takes scalar runtime state + a
MonitorThresholds instance, returns a list of canonical-marker message
strings. monitor_runner (sub-step 2.i) wires detectors to PerformanceMonitor
state and dispatches messages via alert_engine.record_operator_alert().
"""
