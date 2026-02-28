from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


SUPPORTED_BROKERS = {
    "paper": "Local simulation broker",
    "alpaca": "Alpaca Trading API",
    "interactive_brokers": "Interactive Brokers (gateway/TWS integration placeholder)",
    "tradier": "Tradier REST API placeholder",
}


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class SimulatedBroker:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def buy(self, symbol: str, price: float, qty: float) -> None:
        cost = price * qty
        if cost > self.cash:
            raise ValueError("Insufficient cash")
        pos = self.positions.get(symbol, Position())
        total_qty = pos.qty + qty
        pos.avg_price = ((pos.avg_price * pos.qty) + cost) / max(total_qty, 1e-9)
        pos.qty = total_qty
        self.positions[symbol] = pos
        self.cash -= cost

    def sell(self, symbol: str, price: float, qty: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos.qty < qty:
            raise ValueError("Insufficient position")
        pos.qty -= qty
        self.cash += price * qty

    def equity(self, market_prices: dict[str, float]) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            value += pos.qty * market_prices.get(sym, pos.avg_price)
        return value
