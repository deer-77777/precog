"""
Model Manager for LSTM+XGBoost Hybrid Model

Handles:
- Multi-asset model management
- Atomic model swaps during retraining
- Model persistence and versioning
- Background retraining scheduler
"""

import asyncio
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import bittensor as bt

from precog.miners.models.binance_data import BinanceDataFetcher, get_binance_fetcher
from precog.miners.models.hybrid_model import HybridLSTMXGBoost


class ModelManager:
    """
    Manages hybrid models for multiple assets.

    Features:
    - Loads/saves models per asset
    - Atomic model swaps (new model replaces old only on success)
    - Periodic retraining in background thread
    - Fallback to older models if training fails
    """

    # Default assets to manage
    DEFAULT_ASSETS = ["btc", "eth", "tao"]

    # Default model directory
    DEFAULT_MODEL_DIR = "models/lstm_xgb"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        assets: Optional[List[str]] = None,
        binance_api_key: str = "",
        binance_api_secret: str = "",
        training_days: int = 14,
        retrain_interval_hours: float = 4.0,
        lstm_epochs: int = 20,
        auto_start_retraining: bool = False,
    ):
        """
        Initialize the model manager.

        Args:
            model_dir: Directory to store model files
            assets: List of assets to manage (default: btc, eth, tao)
            binance_api_key: Binance API key
            binance_api_secret: Binance API secret
            training_days: Days of historical data for training
            retrain_interval_hours: Hours between retraining cycles
            lstm_epochs: Epochs for LSTM training
            auto_start_retraining: Whether to auto-start background retraining
        """
        self.model_dir = Path(model_dir or self.DEFAULT_MODEL_DIR)
        self.assets = [a.lower() for a in (assets or self.DEFAULT_ASSETS)]
        self.training_days = training_days
        self.retrain_interval_hours = retrain_interval_hours
        self.lstm_epochs = lstm_epochs

        # Initialize data fetcher
        self.data_fetcher = get_binance_fetcher(binance_api_key, binance_api_secret)

        # Model storage
        self._models: Dict[str, HybridLSTMXGBoost] = {}
        self._model_lock = threading.RLock()

        # Training state
        self._last_training_time: Dict[str, datetime] = {}
        self._training_in_progress: Dict[str, bool] = {}
        self._training_errors: Dict[str, str] = {}

        # Background retraining
        self._retrain_thread: Optional[threading.Thread] = None
        self._stop_retraining = threading.Event()

        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Load existing models
        self._load_all_models()

        # Start background retraining if requested
        if auto_start_retraining:
            self.start_background_retraining()

    def _get_model_path(self, asset: str, version: str = "current") -> Path:
        """Get the path for a model version."""
        return self.model_dir / f"{asset}_{version}"

    def _load_all_models(self):
        """Load all available models from disk."""
        for asset in self.assets:
            try:
                self.load_model(asset)
            except Exception as e:
                bt.logging.warning(f"Could not load model for {asset}: {e}")
                self._models[asset] = None

    def load_model(self, asset: str) -> bool:
        """
        Load a model for an asset from disk.

        Args:
            asset: Asset name

        Returns:
            True if model was loaded successfully
        """
        asset = asset.lower()
        model_path = self._get_model_path(asset, "current")

        if not model_path.exists():
            bt.logging.info(f"No saved model found for {asset}")
            return False

        try:
            model = HybridLSTMXGBoost(asset=asset)
            model.load(str(model_path))

            with self._model_lock:
                self._models[asset] = model

            bt.logging.info(f"Loaded model for {asset} (trained at {model.trained_at})")
            return True

        except Exception as e:
            bt.logging.error(f"Error loading model for {asset}: {e}")
            return False

    def save_model(self, asset: str) -> bool:
        """
        Save the current model for an asset.

        Args:
            asset: Asset name

        Returns:
            True if model was saved successfully
        """
        asset = asset.lower()

        with self._model_lock:
            model = self._models.get(asset)

        if model is None:
            bt.logging.warning(f"No model to save for {asset}")
            return False

        try:
            # Save to temp directory first
            temp_path = self._get_model_path(asset, "temp")
            model.save(str(temp_path))

            # Atomic swap: backup current, move temp to current
            current_path = self._get_model_path(asset, "current")
            backup_path = self._get_model_path(asset, "backup")

            if current_path.exists():
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.move(str(current_path), str(backup_path))

            shutil.move(str(temp_path), str(current_path))

            bt.logging.info(f"Saved model for {asset}")
            return True

        except Exception as e:
            bt.logging.error(f"Error saving model for {asset}: {e}")
            return False

    def train_model(
        self,
        asset: str,
        days: Optional[int] = None,
        epochs: Optional[int] = None,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Train or retrain a model for an asset.

        Args:
            asset: Asset name
            days: Days of historical data (default: self.training_days)
            epochs: Training epochs (default: self.lstm_epochs)
            verbose: Training verbosity

        Returns:
            Training metrics dictionary
        """
        asset = asset.lower()
        days = days or self.training_days
        epochs = epochs or self.lstm_epochs

        # Mark training in progress
        self._training_in_progress[asset] = True
        self._training_errors[asset] = ""

        try:
            bt.logging.info(f"Starting training for {asset} with {days} days of data...")

            # Fetch training data
            data_dict = self.data_fetcher.get_training_data(
                assets=[asset],
                days=days,
            )

            if asset not in data_dict or data_dict[asset].empty:
                raise ValueError(f"No training data available for {asset}")

            df = data_dict[asset]
            bt.logging.info(f"Fetched {len(df)} data points for {asset}")

            # Create and train new model
            model = HybridLSTMXGBoost(asset=asset)
            metrics = model.train(
                df,
                lstm_epochs=epochs,
                verbose=verbose,
            )

            # Atomic swap: only replace if training succeeded
            with self._model_lock:
                self._models[asset] = model

            # Save to disk
            self.save_model(asset)

            # Update training time
            self._last_training_time[asset] = datetime.utcnow()

            bt.logging.success(f"Training complete for {asset}")
            return metrics

        except Exception as e:
            self._training_errors[asset] = str(e)
            bt.logging.error(f"Training failed for {asset}: {e}")
            raise

        finally:
            self._training_in_progress[asset] = False

    def predict(self, asset: str, df: "pd.DataFrame" = None) -> Optional[Dict[str, float]]:
        """
        Make a prediction for an asset.

        Args:
            asset: Asset name
            df: Optional DataFrame with OHLCV data. If None, fetches latest.

        Returns:
            Dictionary with 'close', 'high', 'low' predictions, or None if no model
        """
        import pandas as pd

        asset = asset.lower()

        with self._model_lock:
            model = self._models.get(asset)

        if model is None:
            bt.logging.warning(f"No model available for {asset}")
            return None

        try:
            # Fetch data if not provided
            if df is None:
                df = self.data_fetcher.get_latest_klines(
                    asset=asset,
                    lookback=200,  # Extra buffer for feature calculation
                )

            if df.empty:
                bt.logging.warning(f"No data available for {asset}")
                return None

            return model.predict(df)

        except Exception as e:
            bt.logging.error(f"Prediction error for {asset}: {e}")
            return None

    def predict_all(self) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for all managed assets.

        Returns:
            Dictionary mapping asset name to predictions
        """
        results = {}
        for asset in self.assets:
            prediction = self.predict(asset)
            if prediction is not None:
                results[asset] = prediction
        return results

    def needs_retraining(self, asset: str) -> bool:
        """Check if an asset model needs retraining."""
        asset = asset.lower()

        # No model exists
        if self._models.get(asset) is None:
            return True

        # Check last training time
        last_time = self._last_training_time.get(asset)
        if last_time is None:
            # Check model's trained_at
            model = self._models.get(asset)
            if model and model.trained_at:
                try:
                    last_time = datetime.fromisoformat(model.trained_at)
                    self._last_training_time[asset] = last_time
                except (ValueError, TypeError):
                    return True
            else:
                return True

        # Check if interval has passed
        interval = timedelta(hours=self.retrain_interval_hours)
        return datetime.utcnow() - last_time > interval

    def _retraining_loop(self):
        """Background retraining loop."""
        bt.logging.info("Background retraining loop started")

        while not self._stop_retraining.is_set():
            for asset in self.assets:
                if self._stop_retraining.is_set():
                    break

                if self._training_in_progress.get(asset, False):
                    continue

                if self.needs_retraining(asset):
                    try:
                        bt.logging.info(f"Background retraining triggered for {asset}")
                        self.train_model(asset)
                    except Exception as e:
                        bt.logging.error(f"Background retraining failed for {asset}: {e}")

            # Wait before next check (check every 5 minutes)
            self._stop_retraining.wait(timeout=300)

        bt.logging.info("Background retraining loop stopped")

    def start_background_retraining(self):
        """Start the background retraining thread."""
        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            bt.logging.warning("Background retraining already running")
            return

        self._stop_retraining.clear()
        self._retrain_thread = threading.Thread(
            target=self._retraining_loop,
            name="ModelRetraining",
            daemon=True,
        )
        self._retrain_thread.start()
        bt.logging.info("Background retraining started")

    def stop_background_retraining(self):
        """Stop the background retraining thread."""
        if self._retrain_thread is None:
            return

        self._stop_retraining.set()
        self._retrain_thread.join(timeout=10)
        self._retrain_thread = None
        bt.logging.info("Background retraining stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get the status of all models."""
        status = {
            "model_dir": str(self.model_dir),
            "retrain_interval_hours": self.retrain_interval_hours,
            "training_days": self.training_days,
            "background_retraining_active": (
                self._retrain_thread is not None and self._retrain_thread.is_alive()
            ),
            "models": {},
        }

        for asset in self.assets:
            model = self._models.get(asset)
            status["models"][asset] = {
                "loaded": model is not None,
                "trained_at": model.trained_at if model else None,
                "training_samples": model.training_samples if model else 0,
                "training_metrics": model.training_metrics if model else {},
                "training_in_progress": self._training_in_progress.get(asset, False),
                "last_error": self._training_errors.get(asset, ""),
                "needs_retraining": self.needs_retraining(asset),
            }

        return status

    def ensure_models_ready(self, assets: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Ensure models are ready for the specified assets.
        Trains any missing or stale models.

        Args:
            assets: List of assets to check (default: all)

        Returns:
            Dictionary mapping asset to ready status
        """
        assets = [a.lower() for a in (assets or self.assets)]
        ready = {}

        for asset in assets:
            if self._models.get(asset) is not None:
                ready[asset] = True
            else:
                try:
                    self.train_model(asset)
                    ready[asset] = True
                except Exception as e:
                    bt.logging.error(f"Failed to prepare model for {asset}: {e}")
                    ready[asset] = False

        return ready


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(**kwargs) -> ModelManager:
    """Get or create the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(**kwargs)
    return _model_manager


def set_model_manager(manager: ModelManager):
    """Set the global model manager instance."""
    global _model_manager
    _model_manager = manager

