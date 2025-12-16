"""
Data Preprocessor for LSTM Training

Handles data normalization, sequence creation, and train/validation splitting.
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
        y: target price (close price at next time step)
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
            targets: Target values of shape (num_samples, 1)
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
    """
    
    FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
    TARGET_COLUMN = "close"
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Length of input sequences (lookback window)
            prediction_horizon: How many steps ahead to predict
            feature_columns: Columns to use as features
            target_column: Column to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns or self.FEATURE_COLUMNS
        self.target_column = target_column or self.TARGET_COLUMN
        
        # Scalers for each feature
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
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
        targets = df[[self.target_column]].values
        
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        
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
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Scale features
        features = df[self.feature_columns].values
        targets = df[[self.target_column]].values
        
        scaled_features = self.feature_scaler.transform(features)
        scaled_targets = self.target_scaler.transform(targets)
        
        # Create sequences
        sequences, sequence_targets = self._create_sequences(
            scaled_features, scaled_targets
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
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from scaled data.
        
        Args:
            features: Scaled feature array
            targets: Scaled target array
        
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        sequence_targets = []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = features[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Target (price at prediction_horizon steps after sequence ends)
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            target = targets[target_idx]
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def inverse_transform_target(self, scaled_target: np.ndarray) -> np.ndarray:
        """
        Convert scaled target back to original scale.
        
        Args:
            scaled_target: Scaled target values
        
        Returns:
            Original scale target values
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted.")
        
        # Ensure 2D shape
        if scaled_target.ndim == 1:
            scaled_target = scaled_target.reshape(-1, 1)
        
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

