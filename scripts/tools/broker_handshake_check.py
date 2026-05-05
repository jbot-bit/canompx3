"""Read-only TopstepX/ProjectX handshake diagnostic.

Loads canonical broker connection manager, runs auth login, lists accounts.
No orders, no writes — proves the broker pipe is open.

Usage:
    python scripts/tools/broker_handshake_check.py
"""

from __future__ import annotations

import logging
import sys

from trading_app.live.broker_connections import connection_manager
from trading_app.live.projectx.auth import BASE_URL
from trading_app.live.projectx.contract_resolver import ProjectXContracts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    print(f"\n=== TopstepX / ProjectX Handshake — base_url={BASE_URL} ===\n")

    connection_manager.load()
    listing = connection_manager.list_connections()
    if not listing:
        print("FAIL: no broker connections registered (data/broker_connections.json missing or empty).")
        return 2

    print(f"Registered connections: {len(listing)}")
    for conn in listing:
        flag = "ENABLED" if conn["enabled"] else "disabled"
        print(f"  - {conn['display_name']} [{conn['broker_type']}] {flag} status={conn['status']}")

    enabled = connection_manager.get_enabled_connections()
    if not enabled:
        print("\nFAIL: no enabled connections to test.")
        return 2

    projectx = next((c for c in enabled if c["broker_type"] == "projectx"), None)
    if projectx is None:
        print("\nFAIL: no enabled ProjectX connection.")
        return 2

    conn_id = projectx["id"]
    print(f"\n--- Connecting: {projectx['display_name']} ({conn_id}) ---")

    try:
        connection_manager.connect(conn_id)
    except Exception as exc:
        print(f"FAIL: connect() raised {type(exc).__name__}: {exc}")
        return 3

    auth = connection_manager.get_auth(conn_id)
    if auth is None:
        print("FAIL: connect() succeeded but auth is None.")
        return 3

    healthy = getattr(auth, "is_healthy", "unknown")
    print(f"Auth health: {healthy}")
    try:
        token = auth.get_token()
        print(f"Token acquired: {len(token)} chars (prefix={token[:10]}...)")
    except Exception as exc:
        print(f"FAIL: get_token() raised {type(exc).__name__}: {exc}")
        return 3

    print("\n--- Account discovery ---")
    contracts = ProjectXContracts(auth)
    try:
        accounts = contracts.resolve_all_account_ids()
    except Exception as exc:
        print(f"FAIL: resolve_all_account_ids() raised {type(exc).__name__}: {exc}")
        return 4

    if not accounts:
        print("WARN: zero accounts returned. This is the Express-account symptom.")
        print("      Auth is working, but your TopStep user has no tradable accounts attached.")
        print("      Action: start a Combine on TopStep, then re-run this script.")
        return 1

    print(f"Accounts visible to this API key: {len(accounts)}")
    for acct_id, name in accounts:
        print(f"  - id={acct_id} name={name}")

    print("\n--- Contract probe (MNQ front month) ---")
    try:
        front = contracts.resolve_front_month("MNQ")
        print(f"MNQ front month contract id: {front}")
    except Exception as exc:
        print(f"WARN: front-month resolve failed: {type(exc).__name__}: {exc}")

    print("\nPASS: broker pipe is open. signal_only path can now read live account state.")
    print("      Next step: keep signal_only=True, run multi_runner against a live lane,")
    print("      verify F-1 gate behavior, then flip signal_only=False when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
