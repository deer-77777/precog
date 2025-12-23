"""
Hybrid LSTM+XGBoost Model for Cryptocurrency Price Prediction

Two-stage architecture:
1. LSTM extracts temporal embeddings from sequential features
2. XGBoost predicts [close, high, low] from LSTM embeddings

Based on arXiv:2506.22055 "LSTM+XGBoost" methodology.
"""

import json
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

# TensorFlow with reduced verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

from precog.miners.models.feature_engineering import FeatureEngineer


class LSTMFeatureExtractor:
    """
    LSTM-based feature extractor that learns temporal patterns.

    Architecture:
        Input: (batch, sequence_length, n_features)
        LSTM(50, return_sequences=True)
        LSTM(50)
        Dense(25, activation='relu')  -> Temporal embedding
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 100,
        lstm_units_1: int = 50,
        lstm_units_2: int = 50,
        embedding_dim: int = 25,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            sequence_length: Number of timesteps in input sequence
            n_features: Number of features per timestep
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            embedding_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        self.model: Optional[Model] = None
        self.feature_extractor: Optional[Model] = None
        self._is_built = False

    def build(self, n_features: Optional[int] = None):
        """Build the LSTM model using Functional API for Keras 3.x compatibility."""
        if n_features is not None:
            self.n_features = n_features

        # Use Functional API for better compatibility with Keras 3.x
        inputs = Input(shape=(self.sequence_length, self.n_features), name="input")

        # LSTM layers
        x = LSTM(self.lstm_units_1, return_sequences=True, name="lstm_1")(inputs)
        x = Dropout(self.dropout_rate, name="dropout_1")(x)
        x = LSTM(self.lstm_units_2, return_sequences=False, name="lstm_2")(x)
        x = Dropout(self.dropout_rate, name="dropout_2")(x)

        # Embedding layer (this is what we extract for XGBoost)
        embedding = Dense(self.embedding_dim, activation="relu", name="embedding")(x)

        # Output layer for auxiliary training loss
        outputs = Dense(3, name="output")(embedding)

        # Full model for training
        self.model = Model(inputs=inputs, outputs=outputs, name="lstm_model")
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )

        # Feature extractor (outputs embedding layer)
        self.feature_extractor = Model(
            inputs=inputs,
            outputs=embedding,
            name="feature_extractor",
        )

        self._is_built = True
        bt.logging.debug(f"LSTM model built: input shape = ({self.sequence_length}, {self.n_features})")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X: Input features of shape (n_samples, sequence_length, n_features)
            y: Targets of shape (n_samples, 3) for [close, high, low]
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)

        Returns:
            Training history dictionary
        """
        if not self._is_built:
            self.build(n_features=X.shape[2])

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=verbose,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=verbose,
            ),
        ]

        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )

        return history.history

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract temporal embeddings from input sequences.

        Args:
            X: Input of shape (n_samples, sequence_length, n_features)

        Returns:
            Embeddings of shape (n_samples, embedding_dim)
        """
        if not self._is_built or self.feature_extractor is None:
            raise ValueError("Model not built. Call build() or train() first.")

        return self.feature_extractor.predict(X, verbose=0)

    def save(self, path: str):
        """Save the LSTM model."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load a saved LSTM model."""
        self.model = load_model(path)

        # Update dimensions from loaded model
        input_shape = self.model.input_shape
        self.sequence_length = input_shape[1]
        self.n_features = input_shape[2]

        # Get embedding layer output shape
        embedding_layer = self.model.get_layer("embedding")
        self.embedding_dim = embedding_layer.output.shape[-1]

        # Rebuild feature extractor using the loaded model's layers
        self.feature_extractor = Model(
            inputs=self.model.input,
            outputs=embedding_layer.output,
            name="feature_extractor",
        )

        self._is_built = True


class HybridLSTMXGBoost:
    """
    Hybrid LSTM+XGBoost model for price prediction.

    Stage 1: LSTM extracts temporal embeddings
    Stage 2: XGBoost predicts [close, high, low] from embeddings
    """

    def __init__(
        self,
        asset: str = "btc",
        sequence_length: int = 60,
        lstm_units: int = 50,
        embedding_dim: int = 25,
        xgb_n_estimators: int = 200,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
    ):
        """
        Initialize hybrid model.

        Args:
            asset: Asset name (btc, eth, tao)
            sequence_length: LSTM sequence length
            lstm_units: Units per LSTM layer
            embedding_dim: LSTM output embedding dimension
            xgb_n_estimators: Number of XGBoost trees
            xgb_max_depth: Maximum tree depth
            xgb_learning_rate: XGBoost learning rate
        """
        self.asset = asset.lower()
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate

        # Model components
        self.lstm_extractor: Optional[LSTMFeatureExtractor] = None
        self.xgb_model: Optional[MultiOutputRegressor] = None
        self.feature_engineer: Optional[FeatureEngineer] = None

        # Metadata
        self.version = "1.0"
        self.trained_at: Optional[str] = None
        self.training_samples: int = 0
        self.training_metrics: Dict[str, float] = {}

        self._lock = threading.Lock()

    def train(
        self,
        df: "pd.DataFrame",
        lstm_epochs: int = 20,
        lstm_batch_size: int = 32,
        validation_split: float = 0.1,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Train the hybrid model on OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            lstm_epochs: Number of LSTM training epochs
            lstm_batch_size: LSTM batch size
            validation_split: Validation split ratio
            verbose: Verbosity level

        Returns:
            Dictionary with training metrics
        """
        import pandas as pd

        bt.logging.info(f"Training hybrid model for {self.asset}...")

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()

        # Prepare training data
        bt.logging.debug("Preparing training data...")
        X, y, feature_names = self.feature_engineer.prepare_training_data(
            df,
            sequence_length=self.sequence_length,
            horizon=60,  # 1-hour ahead prediction
        )

        bt.logging.info(f"Training data shape: X={X.shape}, y={y.shape}")
        self.training_samples = len(X)

        # Stage 1: Train LSTM
        bt.logging.info("Stage 1: Training LSTM feature extractor...")
        self.lstm_extractor = LSTMFeatureExtractor(
            sequence_length=self.sequence_length,
            n_features=X.shape[2],
            lstm_units_1=self.lstm_units,
            lstm_units_2=self.lstm_units,
            embedding_dim=self.embedding_dim,
        )

        lstm_history = self.lstm_extractor.train(
            X,
            y,
            epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

        # Extract embeddings for XGBoost training
        bt.logging.info("Extracting LSTM embeddings...")
        embeddings = self.lstm_extractor.extract_features(X)
        bt.logging.debug(f"Embeddings shape: {embeddings.shape}")

        # Stage 2: Train XGBoost
        bt.logging.info("Stage 2: Training XGBoost regressor...")

        # Use MultiOutputRegressor for predicting [close, high, low]
        base_xgb = xgb.XGBRegressor(
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
            random_state=42,
        )

        self.xgb_model = MultiOutputRegressor(base_xgb)
        self.xgb_model.fit(embeddings, y)

        # Calculate training metrics
        y_pred = self.xgb_model.predict(embeddings)
        mse = np.mean((y - y_pred) ** 2, axis=0)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred), axis=0)

        self.training_metrics = {
            "lstm_final_loss": float(lstm_history.get("loss", [0])[-1]),
            "lstm_final_val_loss": float(lstm_history.get("val_loss", [0])[-1]),
            "xgb_rmse_close": float(rmse[0]),
            "xgb_rmse_high": float(rmse[1]),
            "xgb_rmse_low": float(rmse[2]),
            "xgb_mae_close": float(mae[0]),
            "xgb_mae_high": float(mae[1]),
            "xgb_mae_low": float(mae[2]),
        }

        self.trained_at = datetime.utcnow().isoformat()

        bt.logging.success(
            f"Training complete for {self.asset}: "
            f"RMSE(close)={rmse[0]:.2f}, RMSE(high)={rmse[1]:.2f}, RMSE(low)={rmse[2]:.2f}"
        )

        return self.training_metrics

    def predict(self, df: "pd.DataFrame") -> Dict[str, float]:
        """
        Make predictions using the hybrid model.

        Args:
            df: DataFrame with recent OHLCV data (at least sequence_length rows)

        Returns:
            Dictionary with 'close', 'high', 'low' predictions
        """
        with self._lock:
            if self.lstm_extractor is None or self.xgb_model is None:
                raise ValueError("Model not trained. Call train() first.")

            if self.feature_engineer is None:
                raise ValueError("Feature engineer not initialized.")

            # Prepare inference data
            X, current_price = self.feature_engineer.prepare_inference_data(
                df,
                sequence_length=self.sequence_length,
            )

            # Stage 1: Extract LSTM embedding
            embedding = self.lstm_extractor.extract_features(X)

            # Stage 2: XGBoost prediction
            prediction = self.xgb_model.predict(embedding)[0]

            return {
                "close": float(prediction[0]),
                "high": float(prediction[1]),
                "low": float(prediction[2]),
                "current_price": float(current_price),
            }

    def save(self, directory: str):
        """
        Save the complete hybrid model.

        Args:
            directory: Directory to save model files
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save LSTM model
        lstm_path = path / "lstm_model.keras"
        if self.lstm_extractor is not None:
            self.lstm_extractor.save(str(lstm_path))

        # Save XGBoost model
        xgb_path = path / "xgb_model.pkl"
        if self.xgb_model is not None:
            with open(xgb_path, "wb") as f:
                pickle.dump(self.xgb_model, f)

        # Save feature engineer params
        fe_path = path / "feature_engineer.json"
        if self.feature_engineer is not None:
            with open(fe_path, "w") as f:
                json.dump(self.feature_engineer.get_scaler_params(), f, default=str)

        # Save metadata
        metadata = {
            "asset": self.asset,
            "version": self.version,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_units,
            "embedding_dim": self.embedding_dim,
            "xgb_n_estimators": self.xgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "xgb_learning_rate": self.xgb_learning_rate,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "training_metrics": self.training_metrics,
        }

        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        bt.logging.info(f"Model saved to {directory}")

    def load(self, directory: str):
        """
        Load a saved hybrid model.

        Args:
            directory: Directory containing model files
        """
        path = Path(directory)

        if not path.exists():
            raise ValueError(f"Model directory not found: {directory}")

        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.asset = metadata["asset"]
        self.version = metadata["version"]
        self.sequence_length = metadata["sequence_length"]
        self.lstm_units = metadata["lstm_units"]
        self.embedding_dim = metadata["embedding_dim"]
        self.xgb_n_estimators = metadata["xgb_n_estimators"]
        self.xgb_max_depth = metadata["xgb_max_depth"]
        self.xgb_learning_rate = metadata["xgb_learning_rate"]
        self.trained_at = metadata["trained_at"]
        self.training_samples = metadata["training_samples"]
        self.training_metrics = metadata["training_metrics"]

        # Load LSTM model
        lstm_path = path / "lstm_model.keras"
        self.lstm_extractor = LSTMFeatureExtractor(
            sequence_length=self.sequence_length,
            embedding_dim=self.embedding_dim,
        )
        self.lstm_extractor.load(str(lstm_path))

        # Load XGBoost model
        xgb_path = path / "xgb_model.pkl"
        with open(xgb_path, "rb") as f:
            self.xgb_model = pickle.load(f)

        # Load feature engineer params
        fe_path = path / "feature_engineer.json"
        self.feature_engineer = FeatureEngineer()
        with open(fe_path, "r") as f:
            fe_params = json.load(f)
            self.feature_engineer.set_scaler_params(fe_params)

        bt.logging.info(f"Model loaded from {directory} (trained at {self.trained_at})")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metrics."""
        return {
            "asset": self.asset,
            "version": self.version,
            "sequence_length": self.sequence_length,
            "lstm_units": self.lstm_units,
            "embedding_dim": self.embedding_dim,
            "xgb_n_estimators": self.xgb_n_estimators,
            "xgb_max_depth": self.xgb_max_depth,
            "trained_at": self.trained_at,
            "training_samples": self.training_samples,
            "training_metrics": self.training_metrics,
            "is_trained": self.lstm_extractor is not None and self.xgb_model is not None,
        }

