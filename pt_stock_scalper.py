import os
import time
import math
import json
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


"""
PowerTrader AI - Stock Scalper (Alpaca)

This script keeps the original project philosophy (simple, memory-based AI)
but applies it to stock scalping:
- Data: 1-minute bars from Alpaca market data API
- AI: instance-based nearest-neighbor predictor on recent return patterns
- Trading: market orders via Alpaca trading API (paper by default)

Env vars:
  ALPACA_API_KEY           required
  ALPACA_SECRET_KEY        required
  ALPACA_BASE_URL          optional (default paper endpoint)
  ALPACA_DATA_BASE_URL     optional

Optional tuning:
  SCALPER_SYMBOLS          comma list (default: AAPL,MSFT,NVDA,TSLA,AMD)
  SCALPER_NOTIONAL_USD     per-entry order notional (default: 150)
  SCALPER_LOOKBACK         pattern length in candles (default: 12)
  SCALPER_NEIGHBORS        k nearest neighbors (default: 25)
  SCALPER_MIN_EDGE_PCT     minimum expected edge to enter (default: 0.15)
  SCALPER_TAKE_PROFIT_PCT  take profit (default: 0.35)
  SCALPER_STOP_LOSS_PCT    stop loss (default: 0.25)
  SCALPER_MAX_HOLD_MIN     max hold time in minutes (default: 12)
  SCALPER_LOOP_SLEEP_SEC   polling interval (default: 20)
"""


@dataclass
class PositionState:
    qty: float
    entry_price: float
    entered_at: dt.datetime


class AlpacaClient:
    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY", "").strip()
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "").strip()
        if not self.api_key or not self.secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

        self.base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        self.data_url = os.environ.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        })

    def _req(self, method: str, url: str, **kwargs) -> dict:
        r = self.s.request(method, url, timeout=20, **kwargs)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        if not r.text.strip():
            return {}
        return r.json()

    def get_latest_bars(self, symbols: List[str], limit: int = 300) -> Dict[str, List[dict]]:
        syms = ",".join(symbols)
        url = f"{self.data_url}/v2/stocks/bars"
        params = {
            "symbols": syms,
            "timeframe": "1Min",
            "limit": str(limit),
            "adjustment": "raw",
            "feed": "iex",
        }
        data = self._req("GET", url, params=params)
        return data.get("bars", {})

    def submit_market_buy_notional(self, symbol: str, notional_usd: float) -> dict:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "notional": f"{notional_usd:.2f}",
        }
        return self._req("POST", url, data=json.dumps(payload))

    def submit_market_sell_qty(self, symbol: str, qty: float) -> dict:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "qty": str(max(0.0, qty)),
        }
        return self._req("POST", url, data=json.dumps(payload))


def close_prices_from_bars(bars: List[dict]) -> List[float]:
    closes = []
    for b in bars:
        c = b.get("c")
        if c is None:
            continue
        try:
            closes.append(float(c))
        except Exception:
            pass
    return closes


def build_returns(prices: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        cur = prices[i]
        if prev <= 0:
            out.append(0.0)
        else:
            out.append((cur - prev) / prev)
    return out


def predict_next_return(returns: List[float], lookback: int, k: int) -> Optional[float]:
    # Instance-based memory: nearest historical patterns, weighted by inverse distance
    if len(returns) < (lookback + 40):
        return None

    cur = returns[-lookback:]
    memory: List[Tuple[float, float]] = []

    # Build memory tuples: (distance, next_return)
    for end_idx in range(lookback, len(returns) - 1):
        hist = returns[end_idx - lookback:end_idx]
        nxt = returns[end_idx]
        dist = 0.0
        for a, b in zip(cur, hist):
            d = a - b
            dist += d * d
        memory.append((math.sqrt(dist), nxt))

    if not memory:
        return None

    memory.sort(key=lambda x: x[0])
    nearest = memory[: max(1, min(k, len(memory)))]

    num = 0.0
    den = 0.0
    for dist, nxt in nearest:
        w = 1.0 / (dist + 1e-9)
        num += nxt * w
        den += w
    if den <= 0:
        return None
    return num / den


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def run() -> None:
    symbols = [s.strip().upper() for s in os.environ.get("SCALPER_SYMBOLS", "AAPL,MSFT,NVDA,TSLA,AMD").split(",") if s.strip()]
    notional_usd = float(os.environ.get("SCALPER_NOTIONAL_USD", "150"))
    lookback = int(os.environ.get("SCALPER_LOOKBACK", "12"))
    neighbors = int(os.environ.get("SCALPER_NEIGHBORS", "25"))
    min_edge_pct = float(os.environ.get("SCALPER_MIN_EDGE_PCT", "0.15")) / 100.0
    take_profit_pct = float(os.environ.get("SCALPER_TAKE_PROFIT_PCT", "0.35")) / 100.0
    stop_loss_pct = float(os.environ.get("SCALPER_STOP_LOSS_PCT", "0.25")) / 100.0
    max_hold_min = int(os.environ.get("SCALPER_MAX_HOLD_MIN", "12"))
    sleep_sec = int(os.environ.get("SCALPER_LOOP_SLEEP_SEC", "20"))

    client = AlpacaClient()
    positions: Dict[str, PositionState] = {}

    print("Starting stock scalper...")
    print(f"Symbols: {symbols}")
    print(f"Notional/entry: ${notional_usd:.2f}")

    while True:
        try:
            bars_map = client.get_latest_bars(symbols, limit=max(300, lookback + 80))
            tnow = now_utc()

            for sym in symbols:
                bars = bars_map.get(sym, [])
                prices = close_prices_from_bars(bars)
                if len(prices) < (lookback + 50):
                    continue

                px = prices[-1]
                rets = build_returns(prices)
                pred = predict_next_return(rets, lookback, neighbors)
                if pred is None:
                    continue

                momentum = sum(rets[-3:]) / 3.0 if len(rets) >= 3 else 0.0
                pos = positions.get(sym)

                if pos is None:
                    if pred >= min_edge_pct and momentum > 0:
                        order = client.submit_market_buy_notional(sym, notional_usd)
                        filled_avg = float(order.get("filled_avg_price") or px)
                        qty = float(order.get("filled_qty") or (notional_usd / max(px, 1e-9)))
                        positions[sym] = PositionState(qty=qty, entry_price=filled_avg, entered_at=tnow)
                        print(f"[BUY ] {sym} qty={qty:.6f} entry={filled_avg:.4f} pred={pred*100:.3f}%")
                    continue

                pnl = (px - pos.entry_price) / max(pos.entry_price, 1e-9)
                age_min = (tnow - pos.entered_at).total_seconds() / 60.0
                should_exit = (
                    pnl >= take_profit_pct
                    or pnl <= -stop_loss_pct
                    or pred < 0
                    or age_min >= max_hold_min
                )

                if should_exit:
                    client.submit_market_sell_qty(sym, pos.qty)
                    print(f"[SELL] {sym} qty={pos.qty:.6f} px={px:.4f} pnl={pnl*100:.3f}% age={age_min:.1f}m")
                    positions.pop(sym, None)
                else:
                    print(f"[HOLD] {sym} px={px:.4f} pred={pred*100:.3f}% pnl={pnl*100:.3f}%")

        except Exception as e:
            print(f"loop error: {e}")

        time.sleep(max(5, sleep_sec))


if __name__ == "__main__":
    run()
