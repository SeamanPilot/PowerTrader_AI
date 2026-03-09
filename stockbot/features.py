from __future__ import annotations

import pandas as pd


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["returns"] = out["close"].pct_change().fillna(0)
    out["sma_fast"] = out["close"].rolling(10).mean()
    out["sma_slow"] = out["close"].rolling(30).mean()

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["bb_mid"] = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + (2 * std)
    out["bb_lower"] = out["bb_mid"] - (2 * std)
    return out.dropna()
