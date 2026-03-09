from __future__ import annotations

from getpass import getpass

from .brokers import SUPPORTED_BROKERS
from .config import BrokerCredentials, BotSettings, save_config


def run_setup_wizard() -> None:
    print("=== Universal Broker/API Setup Wizard ===")
    print("Supported brokers:")
    keys = list(SUPPORTED_BROKERS.keys())
    for idx, key in enumerate(keys, 1):
        print(f"  {idx}. {key} - {SUPPORTED_BROKERS[key]}")

    broker_index = int(input("Select broker number: ").strip()) - 1
    broker = keys[broker_index]

    api_key = input("API key (leave blank for paper mode): ").strip()
    api_secret = getpass("API secret (hidden, leave blank for paper mode): ").strip()
    endpoint = input("API endpoint/base URL (optional): ").strip()

    symbols = [s.strip().upper() for s in input("Symbols (comma separated, e.g. AAPL,MSFT,NVDA): ").split(",") if s.strip()]
    initial_cash = float(input("Initial paper cash [100000]: ").strip() or "100000")
    position_pct = float(input("Position size pct [0.1]: ").strip() or "0.1")
    timeframe = input("Timeframe [1d]: ").strip() or "1d"
    lookback = input("Lookback period [2y]: ").strip() or "2y"

    creds = BrokerCredentials(broker=broker, api_key=api_key, api_secret=api_secret, endpoint=endpoint)
    settings = BotSettings(
        symbols=symbols or ["AAPL"],
        initial_cash=initial_cash,
        position_size_pct=position_pct,
        timeframe=timeframe,
        lookback=lookback,
    )

    save_config(creds, settings)
    print("Configuration saved to stockbot_config.json")


if __name__ == "__main__":
    run_setup_wizard()
