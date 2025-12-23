"""
Feature Engineering for LSTM+XGBoost Hybrid Model

Creates technical indicators and lag features from OHLCV data.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class FeatureEngineer:
    """Creates features from OHLCV data for the hybrid model."""

    # Default lag periods for feature creation
    DEFAULT_LAGS = [1, 2, 3, 5, 10, 15, 30, 60]

    # Features to normalize (price-based)
    PRICE_FEATURES = ["open", "high", "low", "close"]

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
    ):
        """
        Initialize feature engineer.

        Args:
            lags: List of lag periods for creating lag features
            rsi_period: RSI calculation period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            atr_period: ATR period
        """
        self.lags = lags or self.DEFAULT_LAGS
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period

        # Store scaling parameters for inference
        self._price_scaler_params: Optional[dict] = None
        self._feature_names: Optional[List[str]] = None

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for close, high, low, volume."""
        df = df.copy()

        for lag in self.lags:
            # Price lags
            df[f"close_lag{lag}"] = df["close"].shift(lag)
            df[f"high_lag{lag}"] = df["high"].shift(lag)
            df[f"low_lag{lag}"] = df["low"].shift(lag)

            # Returns (percentage change)
            df[f"return_lag{lag}"] = df["close"].pct_change(lag)

            # Volume lag
            df[f"volume_lag{lag}"] = df["volume"].shift(lag)

        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator."""
        df = df.copy()
        rsi = RSIIndicator(close=df["close"], window=self.rsi_period)
        df["rsi"] = rsi.rsi()
        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands indicators."""
        df = df.copy()
        bb = BollingerBands(
            close=df["close"],
            window=self.bb_period,
            window_dev=self.bb_std,
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pband"] = bb.bollinger_pband()  # % position within bands
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators."""
        df = df.copy()
        macd = MACD(
            close=df["close"],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages."""
        df = df.copy()

        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            sma = SMAIndicator(close=df["close"], window=period)
            df[f"sma_{period}"] = sma.sma_indicator()

        # Exponential Moving Averages
        for period in [5, 10, 20]:
            ema = EMAIndicator(close=df["close"], window=period)
            df[f"ema_{period}"] = ema.ema_indicator()

        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        df = df.copy()

        # ATR
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.atr_period,
        )
        df["atr"] = atr.average_true_range()

        # Price range features
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
        df["open_close_range"] = (df["close"] - df["open"]) / df["open"]

        # Rolling volatility (std of returns)
        df["volatility_5"] = df["close"].pct_change().rolling(5).std()
        df["volatility_20"] = df["close"].pct_change().rolling(20).std()
        df["volatility_60"] = df["close"].pct_change().rolling(60).std()

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()

        # Volume ratios
        df["volume_sma_5"] = df["volume"].rolling(5).mean()
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio_5"] = df["volume"] / df["volume_sma_5"]
        df["volume_ratio_20"] = df["volume"] / df["volume_sma_20"]

        # OBV
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        df["obv"] = obv.on_balance_volume()

        # OBV change
        df["obv_change"] = df["obv"].pct_change()

        return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        df = df.copy()

        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
            smooth_window=3,
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = df["close"].pct_change(period) * 100

        return df

    def add_price_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features about price position relative to ranges."""
        df = df.copy()

        # Position relative to day's range
        df["price_position_day"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # Position relative to rolling high/low
        for period in [20, 60]:
            rolling_high = df["high"].rolling(period).max()
            rolling_low = df["low"].rolling(period).min()
            df[f"price_position_{period}"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-8)

        return df

    def add_target_features(self, df: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
        """
        Add target variables for training.

        Args:
            df: DataFrame with OHLCV data
            horizon: Prediction horizon in minutes (default 60 = 1 hour)

        Returns:
            DataFrame with target columns
        """
        df = df.copy()

        # Future close price
        df["target_close"] = df["close"].shift(-horizon)

        # Future high (max over next horizon minutes)
        df["target_high"] = df["high"].rolling(horizon).max().shift(-horizon + 1)

        # Future low (min over next horizon minutes)
        df["target_low"] = df["low"].rolling(horizon).min().shift(-horizon + 1)

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_targets: bool = True,
        horizon: int = 60,
    ) -> pd.DataFrame:
        """
        Create all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            include_targets: Whether to include target variables
            horizon: Prediction horizon in minutes

        Returns:
            DataFrame with all features
        """
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add all feature groups
        df = self.add_lag_features(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_macd(df)
        df = self.add_moving_averages(df)
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        df = self.add_momentum_features(df)
        df = self.add_price_position_features(df)

        if include_targets:
            df = self.add_target_features(df, horizon=horizon)

        return df

    def normalize_features(
        self,
        df: pd.DataFrame,
        fit: bool = True,
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.

        Args:
            df: DataFrame with features
            fit: Whether to fit scaling parameters (True for training)
            exclude_cols: Columns to exclude from normalization

        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        # Make a copy of exclude_cols to avoid mutating the original list
        exclude_cols = list(exclude_cols) if exclude_cols else []

        # Add target columns to exclude list
        target_cols = [c for c in df.columns if c.startswith("target_")]
        exclude_cols.extend(target_cols)

        # Get columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]

        if fit:
            # Compute and store scaling parameters
            self._price_scaler_params = {}
            for col in cols_to_normalize:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val == 0 or np.isnan(std_val):
                    std_val = 1.0
                self._price_scaler_params[col] = {"mean": mean_val, "std": std_val}

        # Apply normalization
        if self._price_scaler_params:
            for col in cols_to_normalize:
                if col in self._price_scaler_params:
                    params = self._price_scaler_params[col]
                    df[col] = (df[col] - params["mean"]) / params["std"]

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        horizon: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for LSTM training.

        Args:
            df: Raw OHLCV DataFrame
            sequence_length: Number of timesteps for LSTM input
            horizon: Prediction horizon in minutes

        Returns:
            X: Feature array of shape (n_samples, sequence_length, n_features)
            y: Target array of shape (n_samples, 3) for [close, high, low]
            feature_names: List of feature names
        """
        # Create all features
        df_features = self.create_all_features(df, include_targets=True, horizon=horizon)

        # Drop rows with NaN
        df_features = df_features.dropna()

        if len(df_features) < sequence_length + 1:
            raise ValueError(f"Not enough data: {len(df_features)} rows after feature creation")

        # Get feature columns (exclude targets and non-numeric)
        target_cols = ["target_close", "target_high", "target_low"]
        exclude_cols = target_cols + ["close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols and df_features[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        self._feature_names = feature_cols

        # Normalize features
        df_normalized = self.normalize_features(df_features, fit=True, exclude_cols=target_cols)

        # Extract arrays
        X_data = df_normalized[feature_cols].values
        y_data = df_normalized[target_cols].values

        # Create sequences
        X_sequences = []
        y_sequences = []

        for i in range(len(X_data) - sequence_length):
            X_sequences.append(X_data[i : i + sequence_length])
            y_sequences.append(y_data[i + sequence_length - 1])

        X = np.array(X_sequences)
        y = np.array(y_sequences)

        return X, y, feature_cols

    def prepare_inference_data(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
    ) -> Tuple[np.ndarray, float]:
        """
        Prepare data for inference.

        Args:
            df: Raw OHLCV DataFrame (at least sequence_length rows)
            sequence_length: Number of timesteps for LSTM input

        Returns:
            X: Feature array of shape (1, sequence_length, n_features)
            current_price: The current close price (for reference)
        """
        if len(df) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} rows, got {len(df)}")

        # Take the last sequence_length + buffer rows
        df_subset = df.tail(sequence_length + max(self.lags) + 100).copy()

        # Create features (no targets needed for inference)
        df_features = self.create_all_features(df_subset, include_targets=False)
        df_features = df_features.dropna()

        if len(df_features) < sequence_length:
            raise ValueError(f"Not enough valid data after feature creation: {len(df_features)}")

        # Get feature columns
        if self._feature_names is None:
            raise ValueError("Feature names not set. Call prepare_training_data first or load scaler params.")

        # Filter to only the features we trained on
        available_features = [f for f in self._feature_names if f in df_features.columns]
        if len(available_features) != len(self._feature_names):
            missing = set(self._feature_names) - set(available_features)
            raise ValueError(f"Missing features: {missing}")

        # Normalize using stored parameters
        df_normalized = self.normalize_features(df_features, fit=False)

        # Extract the last sequence
        X_data = df_normalized[self._feature_names].values
        X = X_data[-sequence_length:].reshape(1, sequence_length, -1)

        # Get current price
        current_price = df["close"].iloc[-1]

        return X, current_price

    def get_feature_names(self) -> List[str]:
        """Get the list of feature names."""
        return self._feature_names or []

    def get_scaler_params(self) -> dict:
        """Get scaler parameters for saving."""
        return {
            "price_scaler_params": self._price_scaler_params,
            "feature_names": self._feature_names,
        }

    def set_scaler_params(self, params: dict):
        """Set scaler parameters from loaded model."""
        self._price_scaler_params = params.get("price_scaler_params")
        self._feature_names = params.get("feature_names")

