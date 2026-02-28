from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class DeepPriceForecaster:
    """A compact feedforward deep-learning forecaster for next-close return."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
        )
        self.trained = False

    @staticmethod
    def feature_columns() -> list[str]:
        return [
            "returns",
            "sma_fast",
            "sma_slow",
            "rsi",
            "macd",
            "signal",
            "bb_upper",
            "bb_lower",
            "volume",
            "close",
        ]

    def fit(self, df: pd.DataFrame) -> None:
        feat_cols = self.feature_columns()
        x = df[feat_cols].values[:-1]
        y = (df["close"].shift(-1) / df["close"] - 1).values[:-1]
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self.trained = True

    def predict_next_return(self, df: pd.DataFrame) -> float:
        if not self.trained:
            raise RuntimeError("Model is not trained. Call fit() first.")
        row = df[self.feature_columns()].tail(1).values
        row_scaled = self.scaler.transform(row)
        pred = self.model.predict(row_scaled)[0]
        return float(np.clip(pred, -0.2, 0.2))
