from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class StrategySignal:
    name: str
    signal: int
    confidence: float


class BaseStrategy:
    name = "base"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError


class SmaCrossoverStrategy(BaseStrategy):
    name = "sma_crossover"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        sig = 1 if last["sma_fast"] > last["sma_slow"] else -1
        spread = abs(last["sma_fast"] - last["sma_slow"]) / max(last["close"], 1e-9)
        return StrategySignal(self.name, sig, float(min(1.0, spread * 50)))


class RsiMeanReversionStrategy(BaseStrategy):
    name = "rsi_mean_reversion"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        rsi = float(df.iloc[-1]["rsi"])
        if rsi < 30:
            return StrategySignal(self.name, 1, (30 - rsi) / 30)
        if rsi > 70:
            return StrategySignal(self.name, -1, (rsi - 70) / 30)
        return StrategySignal(self.name, 0, 0.2)


class BreakoutStrategy(BaseStrategy):
    name = "breakout"

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        last = df.iloc[-1]
        highs = df["high"].tail(20)
        lows = df["low"].tail(20)
        if last["close"] >= highs.max():
            return StrategySignal(self.name, 1, 0.85)
        if last["close"] <= lows.min():
            return StrategySignal(self.name, -1, 0.85)
        return StrategySignal(self.name, 0, 0.3)


class EnsembleStrategy(BaseStrategy):
    name = "ensemble"

    def __init__(self, strategies: list[BaseStrategy]):
        self.strategies = strategies

    def generate(self, df: pd.DataFrame) -> StrategySignal:
        signals = [s.generate(df) for s in self.strategies]
        weighted = sum(s.signal * s.confidence for s in signals)
        confidence = min(1.0, abs(weighted) / max(len(signals), 1))
        if weighted > 0.15:
            sig = 1
        elif weighted < -0.15:
            sig = -1
        else:
            sig = 0
        return StrategySignal(self.name, sig, confidence)


def default_strategy_stack() -> EnsembleStrategy:
    return EnsembleStrategy([
        SmaCrossoverStrategy(),
        RsiMeanReversionStrategy(),
        BreakoutStrategy(),
    ])
