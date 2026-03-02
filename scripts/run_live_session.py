"""Entry point for a live trading session."""
import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from trading_app.live.session_orchestrator import SessionOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--demo", action="store_true", default=True)
    parser.add_argument("--live", action="store_true",
                        help="REAL MONEY — requires typing CONFIRM")
    parser.add_argument("--account-id", type=int, default=0)
    args = parser.parse_args()

    if args.live:
        confirm = input("⚠ LIVE MODE with real money. Type CONFIRM to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            return
        demo = False
    else:
        demo = True

    session = SessionOrchestrator(
        instrument=args.instrument, demo=demo, account_id=args.account_id
    )
    try:
        asyncio.run(session.run())
    finally:
        session.post_session()


if __name__ == "__main__":
    main()
