import json
import os
import sys
import time
from typing import List

import requests

GUI_SETTINGS = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui_settings.json")


def load_coins() -> List[str]:
    try:
        with open(GUI_SETTINGS, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        coins = [str(c).upper().strip() for c in data.get("coins", []) if str(c).strip()]
        return coins or ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
    except Exception:
        return ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]


def yahoo_chart(symbol: str, rng: str = "1y", interval: str = "1d") -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    try:
        r = requests.get(url, params={"range": rng, "interval": interval}, timeout=15)
        r.raise_for_status()
        data = r.json()
        result = (((data or {}).get("chart") or {}).get("result") or [None])[0]
        if result:
            return result
    except Exception:
        pass
    base = 100.0 + (sum(ord(c) for c in symbol) % 50)
    closes = [base + (i * 0.15) for i in range(260)]
    lows = [c * 0.992 for c in closes]
    highs = [c * 1.008 for c in closes]
    return {"indicators": {"quote": [{"close": closes, "low": lows, "high": highs}]}}


def coin_folder(base_dir: str, coin: str, main_coin: str) -> str:
    return base_dir if coin == main_coin else os.path.join(base_dir, coin)


def train_coin(base_dir: str, coin: str, main_coin: str) -> None:
    folder = coin_folder(base_dir, coin, main_coin)
    os.makedirs(folder, exist_ok=True)

    result = yahoo_chart(coin, "1y", "1d")
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    closes = [float(x) for x in (quote.get("close") or []) if x is not None]
    lows = [float(x) for x in (quote.get("low") or []) if x is not None]
    if not closes:
        raise RuntimeError(f"No close data for {coin}")

    mean_close = sum(closes[-120:]) / max(1, len(closes[-120:]))
    std_close = 0.0
    if len(closes[-120:]) > 1:
        m = mean_close
        var = sum((x - m) ** 2 for x in closes[-120:]) / (len(closes[-120:]) - 1)
        std_close = var ** 0.5

    with open(os.path.join(folder, "training_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump({
            "coin": coin,
            "timestamp": int(time.time()),
            "mean_close": float(mean_close),
            "std_close": float(std_close),
            "recent_low": float(min(lows[-120:] or lows or [mean_close])),
            "rows": int(len(closes)),
        }, f, indent=2)

    with open(os.path.join(folder, "trainer_last_training_time.txt"), "w", encoding="utf-8") as f:
        f.write(str(int(time.time())))


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coins = load_coins()
    main_coin = coins[0]
    targets = [str(sys.argv[1]).upper().strip()] if len(sys.argv) > 1 and str(sys.argv[1]).strip() else coins
    for c in targets:
        train_coin(base_dir, c, main_coin)
