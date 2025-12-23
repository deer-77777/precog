"""
Data Preprocessor for LSTM Training

Handles data normalization, sequence creation, and train/validation splitting.
Supports 3-output prediction: (price, min_price, max_price) for interval prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class PriceDataset(Dataset):
    """
    PyTorch Dataset for price prediction sequences.
    
    Creates sequences of (X, y) pairs where:
        X: sequence of features (OHLCV) of length sequence_length
        y: target values - either (price,) or (price, min, max)
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences of shape (num_samples, sequence_length, num_features)
            targets: Target values of shape (num_samples, 1) or (num_samples, 3)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class DataPreprocessor:
    """
    Preprocessor for converting raw price data into LSTM-ready sequences.
    
    Features used: Open, High, Low, Close, Volume (OHLCV)
    
    Supports two modes:
    - predict_interval=False: predicts only close price (1 output)
    - predict_interval=True: predicts close price, min price, max price (3 outputs)
    """
    
    FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
    TARGET_COLUMN = "close"
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        predict_interval: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Length of input sequences (lookback window)
            prediction_horizon: How many steps ahead to predict
            feature_columns: Columns to use as features
            target_column: Column to predict (close price)
            predict_interval: If True, also predict min/max for interval
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns or self.FEATURE_COLUMNS
        self.target_column = target_column or self.TARGET_COLUMN
        self.predict_interval = predict_interval
        
        # Scalers for features and targets
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        # Use a single scaler for all price targets (close, min, max are all prices)
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fitted flag
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit scalers on the data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            self
        """
        features = df[self.feature_columns].values
        
        # For target scaler, fit on all price data (close, high, low)
        # This ensures consistent scaling for price, min, and max
        all_prices = df[["close", "high", "low"]].values.flatten().reshape(-1, 1)
        
        self.feature_scaler.fit(features)
        self.target_scaler.fit(all_prices)
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data into sequences.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (sequences, targets)
            - If predict_interval=False: targets shape is (num_samples, 1)
            - If predict_interval=True: targets shape is (num_samples, 3) for [price, min, max]
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Scale features
        features = df[self.feature_columns].values
        scaled_features = self.feature_scaler.transform(features)
        
        # Scale price columns for targets
        close_prices = df["close"].values.reshape(-1, 1)
        low_prices = df["low"].values.reshape(-1, 1)
        high_prices = df["high"].values.reshape(-1, 1)
        
        scaled_close = self.target_scaler.transform(close_prices).flatten()
        scaled_low = self.target_scaler.transform(low_prices).flatten()
        scaled_high = self.target_scaler.transform(high_prices).flatten()
        
        # Create sequences
        sequences, sequence_targets = self._create_sequences(
            scaled_features, scaled_close, scaled_low, scaled_high, df
        )
        
        return sequences, sequence_targets
    
    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (sequences, targets)
        """
        self.fit(df)
        return self.transform(df)
    
    def _create_sequences(
        self,
        features: np.ndarray,
        scaled_close: np.ndarray,
        scaled_low: np.ndarray,
        scaled_high: np.ndarray,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from scaled data.
        
        Args:
            features: Scaled feature array
            scaled_close: Scaled close prices
            scaled_low: Scaled low prices
            scaled_high: Scaled high prices
            df: Original dataframe (for getting raw min/max in horizon window)
        
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Target window: from end of input sequence to prediction_horizon steps ahead
            horizon_start = i + self.sequence_length
            horizon_end = i + self.sequence_length + self.prediction_horizon
            
            if self.predict_interval:
                # Get the close price at the END of the horizon
                target_price = scaled_close[horizon_end - 1]
                
                # Get MIN (lowest low) during the entire horizon window
                horizon_lows = df["low"].iloc[horizon_start:horizon_end].values
                min_price_raw = horizon_lows.min()
                target_min = self.target_scaler.transform([[min_price_raw]])[0, 0]
                
                # Get MAX (highest high) during the entire horizon window
                horizon_highs = df["high"].iloc[horizon_start:horizon_end].values
                max_price_raw = horizon_highs.max()
                target_max = self.target_scaler.transform([[max_price_raw]])[0, 0]
                
                # Target: [price, min, max]
                target = np.array([target_price, target_min, target_max])
            else:
                # Original behavior: just the close price
                target_price = scaled_close[horizon_end - 1]
                target = np.array([target_price])
            
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Convert scaled target back to original scale.
        
        Args:
            scaled_target: Scaled target values of shape (n, 1) or (n, 3)
        
        Returns:
            Original scale target values
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        original_shape = scaled_target.shape
        
        # Flatten to 2D for inverse transform
        if scaled_target.ndim == 1:
            scaled_target = scaled_target.reshape(-1, 1)
        elif scaled_target.ndim == 2 and scaled_target.shape[1] > 1:
            # For 3-output case, transform each column separately
            result = np.zeros_like(scaled_target)
            for i in range(scaled_target.shape[1]):
                col = scaled_target[:, i:i+1]
                result[:, i:i+1] = self.target_scaler.inverse_transform(col)
            return result
        
        return self.target_scaler.inverse_transform(scaled_target)
    
    def prepare_for_prediction(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Prepare the most recent data for making a prediction.
        
        Args:
            df: DataFrame with recent OHLCV data (at least sequence_length rows)
        
        Returns:
            Scaled sequence ready for model input
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows, got {len(df)}"
            )
        
        # Get the most recent sequence
        features = df[self.feature_columns].iloc[-self.sequence_length:].values
        scaled_features = self.feature_scaler.transform(features)
        
        return scaled_features.reshape(1, self.sequence_length, len(self.feature_columns))
    
    def save(self, filepath: str):
        """Save preprocessor state to file"""
        state = {
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "predict_interval": self.predict_interval,
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "is_fitted": self._is_fitted,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Load preprocessor from file"""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            sequence_length=state["sequence_length"],
            prediction_horizon=state["prediction_horizon"],
            feature_columns=state["feature_columns"],
            target_column=state["target_column"],
            predict_interval=state.get("predict_interval", False),  # Backward compatible
        )
        preprocessor.feature_scaler = state["feature_scaler"]
        preprocessor.target_scaler = state["target_scaler"]
        preprocessor._is_fitted = state["is_fitted"]
        
        return preprocessor


def create_data_loaders(
    df: pd.DataFrame,
    preprocessor: DataPreprocessor,
    batch_size: int = 32,
    validation_split: float = 0.1,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataPreprocessor]:
    """
    Create train and validation data loaders.
    
    Args:
        df: DataFrame with OHLCV data
        preprocessor: DataPreprocessor instance
        batch_size: Batch size for data loaders
        validation_split: Fraction of data for validation
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, fitted_preprocessor)
    """
    # Fit and transform
    sequences, targets = preprocessor.fit_transform(df)
    
    # Split data (time series - no random split, use last portion for validation)
    split_idx = int(len(sequences) * (1 - validation_split))
    
    train_sequences = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_sequences = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    # Create datasets
    train_dataset = PriceDataset(train_sequences, train_targets)
    val_dataset = PriceDataset(val_sequences, val_targets)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, val_loader, preprocessor


def create_incremental_loader(
    df: pd.DataFrame,
    preprocessor: DataPreprocessor,
    batch_size: int = 32,
) -> DataLoader:
    """
    Create data loader for incremental training with new data.
    
    Args:
        df: DataFrame with new OHLCV data
        preprocessor: Already fitted DataPreprocessor
        batch_size: Batch size
    
    Returns:
        DataLoader for incremental training
    """
    if not preprocessor._is_fitted:
        raise ValueError("Preprocessor must be fitted before incremental training")
    
    sequences, targets = preprocessor.transform(df)
    
    dataset = PriceDataset(sequences, targets)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


class MultiAssetPreprocessor:
    """
    Preprocessor for multiple assets (BTC, ETH, TAO).
    
    Maintains separate preprocessors for each asset.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.preprocessors: Dict[str, DataPreprocessor] = {}
    
    def fit(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> "MultiAssetPreprocessor":
        """
        Fit preprocessors for all assets.
        
        Args:
            data: Dictionary mapping asset symbol to DataFrame
        
        Returns:
            self
        """
        for symbol, df in data.items():
            preprocessor = DataPreprocessor(
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
            )
            preprocessor.fit(df)
            self.preprocessors[symbol] = preprocessor
        
        return self
    
    def transform(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform data for all assets.
        
        Args:
            data: Dictionary mapping asset symbol to DataFrame
        
        Returns:
            Dictionary mapping asset to (sequences, targets) tuple
        """
        result = {}
        for symbol, df in data.items():
            if symbol not in self.preprocessors:
                raise ValueError(f"No preprocessor fitted for {symbol}")
            result[symbol] = self.preprocessors[symbol].transform(df)
        return result
    
    def get_preprocessor(self, symbol: str) -> DataPreprocessor:
        """Get preprocessor for a specific asset"""
        if symbol not in self.preprocessors:
            raise ValueError(f"No preprocessor for {symbol}")
        return self.preprocessors[symbol]
    
    def save(self, directory: str):
        """Save all preprocessors"""
        os.makedirs(directory, exist_ok=True)
        for symbol, preprocessor in self.preprocessors.items():
            filepath = os.path.join(directory, f"{symbol}_preprocessor.pkl")
            preprocessor.save(filepath)
    
    @classmethod
    def load(cls, directory: str) -> "MultiAssetPreprocessor":
        """Load all preprocessors from directory"""
        instance = cls()
        
        for filename in os.listdir(directory):
            if filename.endswith("_preprocessor.pkl"):
                symbol = filename.replace("_preprocessor.pkl", "")
                filepath = os.path.join(directory, filename)
                instance.preprocessors[symbol] = DataPreprocessor.load(filepath)
        
        if instance.preprocessors:
            first_preprocessor = next(iter(instance.preprocessors.values()))
            instance.sequence_length = first_preprocessor.sequence_length
            instance.prediction_horizon = first_preprocessor.prediction_horizon
        
        return instance

