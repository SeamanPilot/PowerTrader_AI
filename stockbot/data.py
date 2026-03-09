from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No market data available for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]].dropna()
