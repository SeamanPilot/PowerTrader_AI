from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import CONFIG_PATH, load_config
from .trader import PaperTradingEngine


@st.cache_resource
def get_engine(symbols: list[str], initial_cash: float, position_size_pct: float) -> PaperTradingEngine:
    return PaperTradingEngine(symbols=symbols, initial_cash=initial_cash, position_size_pct=position_size_pct)


def _price_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_fast"], mode="lines", name="SMA 10"))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma_slow"], mode="lines", name="SMA 30"))
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(dash="dot")))
    fig.update_layout(title=f"{symbol} Live Paper Trading Chart", xaxis_rangeslider_visible=False, height=600)
    return fig


def run_dashboard() -> None:
    st.set_page_config(layout="wide", page_title="PowerTrader StockBot")
    st.title("PowerTrader Paper Trading Deep-Learning Stock Bot")

    if not CONFIG_PATH.exists():
        st.warning("Run setup wizard first: python -m stockbot.wizard")
        return

    creds, settings = load_config()
    st.caption(f"Broker: {creds.broker} | Timeframe: {settings.timeframe} | Lookback: {settings.lookback}")

    symbol = st.selectbox("Symbol", settings.symbols)
    engine = get_engine(settings.symbols, settings.initial_cash, settings.position_size_pct)

    if st.button("Run Trading Cycle"):
        result = engine.run_once(symbol=symbol, period=settings.lookback, interval=settings.timeframe)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Action", result["action"])
        c2.metric("Equity", f"${result['equity']:,.2f}")
        c3.metric("Cash", f"${result['cash']:,.2f}")
        c4.metric("Pred Return", f"{result['predicted_return'] * 100:.2f}%")
        st.write(result["reason"])

        fig = _price_figure(result["data"].tail(160), symbol)
        st.plotly_chart(fig, use_container_width=True)

        if engine.trade_log:
            st.subheader("Trade Log")
            st.dataframe(pd.DataFrame([t.__dict__ for t in engine.trade_log]))


if __name__ == "__main__":
    run_dashboard()
