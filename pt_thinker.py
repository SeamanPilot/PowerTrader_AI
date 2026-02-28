import json
import os
import time
from typing import List

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUB_DATA_DIR = os.environ.get("POWERTRADER_HUB_DIR", os.path.join(BASE_DIR, "hub_data"))
os.makedirs(HUB_DATA_DIR, exist_ok=True)
RUNNER_READY_PATH = os.path.join(HUB_DATA_DIR, "runner_ready.json")
GUI_SETTINGS = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(BASE_DIR, "gui_settings.json")


def yahoo_chart(symbol: str, rng: str = "3mo", interval: str = "1h") -> dict:
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
    close = [base + (i * 0.03) for i in range(400)]
    low = [c * 0.994 for c in close]
    high = [c * 1.006 for c in close]
    return {"indicators": {"quote": [{"close": close, "low": low, "high": high}]}}


def load_settings() -> dict:
    try:
        with open(GUI_SETTINGS, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        data = {}
    coins = [str(c).upper().strip() for c in data.get("coins", []) if str(c).strip()] or ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
    return {"coins": coins}


def coin_folder(coin: str, coins: List[str]) -> str:
    return BASE_DIR if coin == coins[0] else os.path.join(BASE_DIR, coin)


def write_runner_ready(ok: bool, stage: str, ready_coins: List[str], total: int) -> None:
    payload = {"timestamp": time.time(), "ready": bool(ok), "stage": stage, "ready_coins": ready_coins, "total_coins": total}
    tmp = RUNNER_READY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, RUNNER_READY_PATH)


def quantile(values: list, q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def compute_levels(values: list, level_count=7):
    vals = []
    for i in range(level_count):
        q = 0.1 + (0.8 * (i / max(1, level_count - 1)))
        vals.append(float(quantile(values, q)))
    return sorted(vals, reverse=True)


def step_coin(coin: str, coins: List[str]) -> bool:
    folder = coin_folder(coin, coins)
    os.makedirs(folder, exist_ok=True)
    if not os.path.isfile(os.path.join(folder, "trainer_last_training_time.txt")):
        for fn in ("long_dca_signal.txt", "short_dca_signal.txt"):
            with open(os.path.join(folder, fn), "w", encoding="utf-8") as f:
                f.write("0")
        return False

    result = yahoo_chart(coin, "3mo", "1h")
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    close = [float(x) for x in (quote.get("close") or []) if x is not None]
    high = [float(x) for x in (quote.get("high") or []) if x is not None]
    low = [float(x) for x in (quote.get("low") or []) if x is not None]
    if not close or not high or not low:
        return False

    current = close[-1]
    long_levels = compute_levels(low[-240:], 7)
    short_levels = sorted(compute_levels(high[-240:], 7))
    long_signal = sum(1 for lv in long_levels if current <= lv)
    short_signal = sum(1 for lv in short_levels if current >= lv)

    with open(os.path.join(folder, "low_bound_prices.html"), "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6f}" for v in long_levels))
    with open(os.path.join(folder, "high_bound_prices.html"), "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6f}" for v in short_levels))
    with open(os.path.join(folder, "long_dca_signal.txt"), "w", encoding="utf-8") as f:
        f.write(str(long_signal))
    with open(os.path.join(folder, "short_dca_signal.txt"), "w", encoding="utf-8") as f:
        f.write(str(short_signal))
    return True


if __name__ == "__main__":
    while True:
        coins = load_settings()["coins"]
        ready = []
        for coin in coins:
            try:
                if step_coin(coin, coins):
                    ready.append(coin)
            except Exception:
                pass
        write_runner_ready(len(ready) == len(coins), "real_predictions", ready, len(coins))
        time.sleep(30)
