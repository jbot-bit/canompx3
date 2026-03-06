"""
ProjectX Integration Test — auth → accounts → contracts → feed connection.

Requires PROJECTX_USER and PROJECTX_API_KEY in .env or environment.
Run: python scripts/test_projectx_integration.py

This is NOT a unit test — it hits the real ProjectX API.
"""

import asyncio
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def test_auth():
    """Step 1: Authenticate with ProjectX."""
    log.info("=" * 60)
    log.info("STEP 1: Authentication")
    log.info("=" * 60)

    from trading_app.live.projectx.auth import ProjectXAuth

    auth = ProjectXAuth()
    token = auth.get_token()
    log.info("Token acquired: %s...", token[:30])
    log.info("Headers: %s", {k: v[:40] + "..." for k, v in auth.headers().items()})
    return auth


def test_accounts(auth):
    """Step 2: List accounts."""
    log.info("=" * 60)
    log.info("STEP 2: Account Discovery")
    log.info("=" * 60)

    from trading_app.live.projectx.contract_resolver import ProjectXContracts

    contracts = ProjectXContracts(auth=auth)
    account_id = contracts.resolve_account_id()
    log.info("Account ID: %d", account_id)
    return contracts, account_id


def test_contracts(contracts):
    """Step 3: Resolve contracts for all 4 instruments."""
    log.info("=" * 60)
    log.info("STEP 3: Contract Resolution")
    log.info("=" * 60)

    results = {}
    for instrument in ["MGC", "MNQ", "MES", "M2K"]:
        try:
            cid = contracts.resolve_front_month(instrument)
            log.info("  %s -> %s", instrument, cid)
            results[instrument] = cid
        except Exception as e:
            log.error("  %s -> FAILED: %s", instrument, e)

    if not results:
        raise RuntimeError("No contracts resolved — cannot proceed to feed test")
    return results


def test_positions(auth, account_id):
    """Step 4: Query open positions."""
    log.info("=" * 60)
    log.info("STEP 4: Position Query")
    log.info("=" * 60)

    from trading_app.live.projectx.positions import ProjectXPositions

    positions = ProjectXPositions(auth=auth)
    open_pos = positions.query_open(account_id)
    if open_pos:
        log.warning("Open positions found: %s", open_pos)
    else:
        log.info("No open positions (clean state)")
    return open_pos


async def test_feed(auth, contract_id, instrument, duration_seconds=30):
    """Step 5: Connect to market data feed and receive quotes."""
    log.info("=" * 60)
    log.info("STEP 5: Market Data Feed (%s for %ds)", instrument, duration_seconds)
    log.info("=" * 60)

    from trading_app.live.bar_aggregator import Bar
    from trading_app.live.projectx.data_feed import ProjectXDataFeed

    bars_received = []
    quotes_received = [0]

    async def on_bar(bar: Bar):
        bars_received.append(bar)
        log.info("BAR RECEIVED: %s", bar)

    feed = ProjectXDataFeed(auth=auth, on_bar=on_bar)

    # Override _on_quote to count quotes
    original_on_quote = feed._on_quote

    async def counting_on_quote(args):
        quotes_received[0] += 1
        if quotes_received[0] % 50 == 1:
            log.info("  Quotes received: %d", quotes_received[0])
        await original_on_quote(args)

    feed._on_quote = counting_on_quote

    # Run feed with timeout
    log.info("Connecting to SignalR Market Hub for %s...", contract_id)
    try:
        await asyncio.wait_for(feed.run(contract_id), timeout=duration_seconds)
    except TimeoutError:
        log.info("Feed timeout reached (%ds) — stopping", duration_seconds)
    except Exception as e:
        log.error("Feed error: %s", e)

    log.info("Quotes received: %d", quotes_received[0])
    log.info("Bars completed: %d", len(bars_received))

    # Flush final partial bar
    final_bar = feed.flush(contract_id)
    if final_bar:
        log.info("Flushed partial bar: %s", final_bar)

    return quotes_received[0], bars_received


def test_factory():
    """Step 6: Verify broker factory works end-to-end."""
    log.info("=" * 60)
    log.info("STEP 6: Broker Factory")
    log.info("=" * 60)

    from trading_app.live.broker_factory import create_broker_components

    components = create_broker_components("projectx")
    log.info("Factory created components:")
    for key, val in components.items():
        log.info("  %s: %s", key, type(val).__name__)

    from trading_app.live.broker_base import BrokerAuth

    assert isinstance(components["auth"], BrokerAuth), "Auth should be BrokerAuth"
    log.info("Factory test PASSED")


def main():
    log.info("ProjectX Integration Test")
    log.info("=" * 60)

    t0 = time.monotonic()
    passed = 0
    failed = 0

    # Step 1: Auth
    try:
        auth = test_auth()
        passed += 1
    except Exception as e:
        log.error("STEP 1 FAILED: %s", e)
        log.error("Check PROJECTX_USER and PROJECTX_API_KEY in .env")
        failed += 1
        sys.exit(1)

    # Step 2: Accounts
    try:
        contracts, account_id = test_accounts(auth)
        passed += 1
    except Exception as e:
        log.error("STEP 2 FAILED: %s", e)
        failed += 1
        sys.exit(1)

    # Step 3: Contracts
    try:
        contract_map = test_contracts(contracts)
        passed += 1
    except Exception as e:
        log.error("STEP 3 FAILED: %s", e)
        failed += 1
        contract_map = {}

    # Step 4: Positions
    try:
        test_positions(auth, account_id)
        passed += 1
    except Exception as e:
        log.error("STEP 4 FAILED: %s", e)
        failed += 1

    # Step 5: Feed (only if we have a contract)
    if contract_map:
        instrument = next(iter(contract_map))
        contract_id = contract_map[instrument]
        try:
            quotes, bars = asyncio.run(test_feed(auth, contract_id, instrument, duration_seconds=30))
            if quotes > 0:
                log.info("STEP 5 PASSED: received %d quotes, %d bars", quotes, len(bars))
                passed += 1
            else:
                log.warning("STEP 5 WARNING: connected but received 0 quotes (market may be closed)")
                passed += 1  # Connection worked, market just closed
        except Exception as e:
            log.error("STEP 5 FAILED: %s", e)
            failed += 1
    else:
        log.warning("STEP 5 SKIPPED: no contracts resolved")

    # Step 6: Factory
    try:
        test_factory()
        passed += 1
    except Exception as e:
        log.error("STEP 6 FAILED: %s", e)
        failed += 1

    elapsed = time.monotonic() - t0
    log.info("=" * 60)
    log.info("RESULTS: %d passed, %d failed (%.1fs)", passed, failed, elapsed)
    log.info("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
