"""Feature engineering for LSTM models"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create features from raw price data for LSTM input"""

    def __init__(self, lookback_window=60):
        self.lookback_window = lookback_window  # Number of timesteps to look back

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from price data

        Args:
            df: DataFrame with columns ['time', 'ReferenceRateUSD']

        Returns:
            DataFrame with features for each timestep
        """
        df = df.copy()

        # 1. Basic price features
        df["price"] = df["ReferenceRateUSD"]

        # 2. Returns (price changes)
        df["returns_1"] = df["price"].pct_change(1)  # 1-period return
        df["returns_5"] = df["price"].pct_change(5)  # 5-period return
        df["returns_10"] = df["price"].pct_change(10)  # 10-period return

        # 3. Moving averages
        df["sma_5"] = df["price"].rolling(window=5).mean()
        df["sma_10"] = df["price"].rolling(window=10).mean()
        df["sma_20"] = df["price"].rolling(window=20).mean()
        df["sma_50"] = df["price"].rolling(window=50).mean()

        # 4. Exponential moving averages
        df["ema_5"] = df["price"].ewm(span=5, adjust=False).mean()
        df["ema_20"] = df["price"].ewm(span=20, adjust=False).mean()

        # 5. Volatility
        df["volatility_5"] = df["returns_1"].rolling(window=5).std()
        df["volatility_20"] = df["returns_1"].rolling(window=20).std()

        # 6. Price momentum
        df["momentum_5"] = df["price"] - df["price"].shift(5)
        df["momentum_10"] = df["price"] - df["price"].shift(10)

        # 7. RSI (Relative Strength Index)
        df["rsi_14"] = self.calculate_rsi(df["price"], 14)

        # 8. Bollinger Bands
        df["bb_upper"], df["bb_lower"] = self.calculate_bollinger_bands(df["price"], 20)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]

        # 9. Time features
        if "time" in df.columns:
            df["hour"] = df["time"].dt.hour
            df["day_of_week"] = df["time"].dt.dayofweek
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # 10. Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f"price_lag_{lag}"] = df["price"].shift(lag)

        # Fill NaN values - use forward fill then backward fill
        df = df.ffill().bfill().fillna(0)
        
        # Ensure all numeric columns are float type
        for col in df.columns:
            if col not in ['time', 'asset']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(0).astype(np.float32)
                except:
                    pass

        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def prepare_sequences(self, df: pd.DataFrame, feature_cols):
        """
        Prepare sequences for LSTM input

        Args:
            df: DataFrame with features
            feature_cols: List of column names to use as features

        Returns:
            numpy array of shape (num_sequences, lookback_window, num_features)
        """
        # Get numeric features only
        features = df[feature_cols].values.astype(np.float32)

        sequences = []
        for i in range(len(features) - self.lookback_window):
            seq = features[i : i + self.lookback_window]
            sequences.append(seq)

        # Return as float32 array
        return np.array(sequences, dtype=np.float32)

