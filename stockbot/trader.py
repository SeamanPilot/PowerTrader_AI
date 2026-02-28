from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .brokers import SimulatedBroker
from .data import fetch_ohlcv
from .features import with_indicators
from .model import DeepPriceForecaster
from .strategies import default_strategy_stack


@dataclass
class TradeEvent:
    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    reason: str


class PaperTradingEngine:
    def __init__(self, symbols: list[str], initial_cash: float = 100_000, position_size_pct: float = 0.1):
        self.symbols = symbols
        self.position_size_pct = position_size_pct
        self.broker = SimulatedBroker(cash=initial_cash)
        self.model = DeepPriceForecaster()
        self.strategy = default_strategy_stack()
        self.trade_log: list[TradeEvent] = []

    def train(self, symbol: str, period: str = "2y", interval: str = "1d") -> Any:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
        feat = with_indicators(df)
        self.model.fit(feat)
        return feat

    def run_once(self, symbol: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
        feat = with_indicators(df)

        if not self.model.trained:
            self.model.fit(feat)

        strat_signal = self.strategy.generate(feat)
        pred_return = self.model.predict_next_return(feat)
        combined_score = strat_signal.signal * strat_signal.confidence + pred_return * 6

        last_price = float(feat.iloc[-1]["close"])
        alloc_cash = self.broker.cash * self.position_size_pct
        qty = round(alloc_cash / max(last_price, 1e-9), 4)

        action = "hold"
        reason = f"ensemble={strat_signal.signal}, conf={strat_signal.confidence:.2f}, pred={pred_return:.4f}"

        if combined_score > 0.2 and qty > 0:
            self.broker.buy(symbol, last_price, qty)
            action = "buy"
            self.trade_log.append(TradeEvent(datetime.utcnow().isoformat(), symbol, "buy", qty, last_price, reason))
        elif combined_score < -0.2:
            pos = self.broker.positions.get(symbol)
            if pos and pos.qty > 0:
                sell_qty = round(min(pos.qty, qty if qty > 0 else pos.qty), 4)
                if sell_qty > 0:
                    self.broker.sell(symbol, last_price, sell_qty)
                    action = "sell"
                    self.trade_log.append(TradeEvent(datetime.utcnow().isoformat(), symbol, "sell", sell_qty, last_price, reason))

        equity = self.broker.equity({symbol: last_price})
        return {
            "symbol": symbol,
            "price": last_price,
            "action": action,
            "reason": reason,
            "equity": equity,
            "cash": self.broker.cash,
            "position_qty": self.broker.positions.get(symbol).qty if symbol in self.broker.positions else 0,
            "predicted_return": pred_return,
            "strategy_confidence": strat_signal.confidence,
            "data": feat,
        }
