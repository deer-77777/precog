"""
LSTM Miner - Forward Function for Price Prediction

This module provides the forward function for the miner.
It loads trained LSTM models from disk and makes predictions.

The trainer runs separately (lstm_trainer.py) and saves models to disk.
This miner loads the models and uses them for prediction.

The model outputs 3 values: (predicted_price, predicted_min, predicted_max)
for direct interval prediction instead of heuristic-based intervals.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import bittensor as bt
except ImportError:
    class bt:
        class logging:
            @staticmethod
            def info(msg): print(f"[INFO] {msg}")
            @staticmethod
            def debug(msg): print(f"[DEBUG] {msg}")
            @staticmethod
            def warning(msg): print(f"[WARN] {msg}")
            @staticmethod
            def error(msg): print(f"[ERROR] {msg}")
            @staticmethod
            def success(msg): print(f"[SUCCESS] {msg}")
            @staticmethod
            def trace(msg): print(f"[TRACE] {msg}")

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.binance_data import BinanceData, precog_to_binance_symbol
from precog.miners.models import (
    LSTMConfig,
    PricePredictorLSTM,
    DataPreprocessor,
    ModelManager,
)


class LSTMPredictor:
    """
    LSTM-based price predictor.
    
    Loads trained models from disk and makes predictions.
    Models are trained separately by lstm_trainer.py
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        """
        Initialize predictor.
        
        Args:
            config: LSTM configuration
        """
        self.config = config or LSTMConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        bt.logging.info(f"LSTM Predictor initialized on {self.device}")
        
        # Binance data fetcher for getting recent data for prediction
        self.binance = BinanceData()
        
        # Model manager for loading models
        self.model_manager = ModelManager(
            base_path=self.config.management.model_save_path,
            model_prefix=self.config.management.model_prefix,
            keep_old_models=self.config.management.keep_old_models,
        )
        
        # Cache for loaded models
        self._models: Dict[str, PricePredictorLSTM] = {}
        self._preprocessors: Dict[str, DataPreprocessor] = {}
        self._model_versions: Dict[str, int] = {}
    
    def _load_model_if_needed(self, symbol: str) -> bool:
        """
        Load model from disk if not cached or if newer version available.
        
        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
        
        Returns:
            True if model is available
        """
        try:
            # Check if model exists on disk
            if not self.model_manager.model_exists(symbol):
                bt.logging.warning(f"No trained model found for {symbol}")
                return False
            
            # Get current model info
            model_info = self.model_manager.get_model_info(symbol)
            disk_version = model_info.get("version", 0) if model_info else 0
            cached_version = self._model_versions.get(symbol, -1)
            
            # Load if not cached or newer version on disk
            if symbol not in self._models or disk_version > cached_version:
                bt.logging.info(f"Loading model for {symbol} (version {disk_version})")
                
                model, preprocessor, metadata = self.model_manager.load_model(
                    symbol, str(self.device)
                )
                
                self._models[symbol] = model
                self._preprocessors[symbol] = preprocessor
                self._model_versions[symbol] = disk_version
                
                bt.logging.info(f"Model loaded for {symbol}")
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Failed to load model for {symbol}: {e}")
            return False
    
    def predict(self, asset: str) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """
        Make price prediction for an asset (1 hour ahead).
        
        The model outputs 3 values: (price, min_price, max_price)
        This provides learned interval prediction instead of heuristic-based.
        
        Args:
            asset: Precog asset name (e.g., "btc", "eth", "tao_bittensor")
        
        Returns:
            Tuple of (predicted_price, (lower_bound, upper_bound))
        """
        # Convert to Binance symbol
        symbol = precog_to_binance_symbol(asset)
        
        # Load model if needed
        if not self._load_model_if_needed(symbol):
            return None, None
        
        try:
            model = self._models[symbol]
            preprocessor = self._preprocessors[symbol]
            
            # Fetch recent data for prediction input
            # Need sequence_length data points
            df = self.binance.get_recent_klines(
                symbol=symbol,
                interval=self.config.data.data_interval,
                limit=self.config.model.sequence_length + 10,  # Extra buffer
            )
            
            if df.empty or len(df) < self.config.model.sequence_length:
                bt.logging.warning(f"Insufficient data for {symbol} prediction")
                return None, None
            
            # Prepare input sequence
            input_seq = preprocessor.prepare_for_prediction(df)
            input_tensor = torch.FloatTensor(input_seq).to(self.device)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                scaled_prediction, _ = model(input_tensor)
            
            # Convert back to original scale
            # Shape: (1, 3) for [price, min, max] or (1, 1) for price only
            raw_prediction = preprocessor.inverse_transform_target(
                scaled_prediction.cpu().numpy()
            )
            
            # Check if this is a 3-output model (interval prediction)
            if raw_prediction.shape[1] == 3:
                # Model outputs: [price, min_price, max_price]
                prediction = float(raw_prediction[0, 0])
                lower_bound = float(raw_prediction[0, 1])
                upper_bound = float(raw_prediction[0, 2])
                
                # Ensure bounds are valid (min <= price <= max)
                lower_bound = min(lower_bound, prediction)
                upper_bound = max(upper_bound, prediction)
                
                # Ensure lower < upper
                if lower_bound >= upper_bound:
                    # Fallback: add small margin
                    margin = prediction * 0.02
                    lower_bound = prediction - margin
                    upper_bound = prediction + margin
                
                bt.logging.debug(
                    f"{symbol}: Learned interval - price={prediction:.2f}, "
                    f"min={lower_bound:.2f}, max={upper_bound:.2f}"
                )
            else:
                # Single-output model: use volatility-based heuristic (fallback)
                prediction = float(raw_prediction[0, 0])
            
            # Calculate prediction interval based on volatility
            recent_prices = df["close"].tail(self.config.model.sequence_length)
            volatility = recent_prices.pct_change().std()
            
            # 95% confidence interval
            horizon_factor = np.sqrt(self.config.model.prediction_horizon)
            margin = prediction * volatility * 2 * horizon_factor
            
            # Ensure reasonable bounds
            min_margin = prediction * 0.01
            max_margin = prediction * 0.10
            margin = max(min_margin, min(margin, max_margin))
            
            lower_bound = prediction - margin
            upper_bound = prediction + margin
                
            bt.logging.debug(
                f"{symbol}: Heuristic interval (fallback) - price={prediction:.2f}, "
                f"margin={margin:.2f}"
            )
            
            return float(prediction), (float(lower_bound), float(upper_bound))
            
        except Exception as e:
            bt.logging.error(f"Prediction failed for {symbol}: {e}")
            import traceback
            bt.logging.trace(traceback.format_exc())
            return None, None


# Global predictor instance (lazy initialization)
_predictor: Optional[LSTMPredictor] = None


def _get_predictor() -> LSTMPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = LSTMPredictor()
    return _predictor


async def forward(synapse: Challenge, cm: CMData = None) -> Challenge:
    """
    Forward function for LSTM price prediction.
    
    Compatible with the existing miner infrastructure.
    Loads models from disk (trained by lstm_trainer.py).
    
    Args:
        synapse: Challenge synapse with timestamp and assets
        cm: CMData instance (not used, kept for compatibility)
    
    Returns:
        Challenge synapse with predictions
    """
    total_start_time = time.perf_counter()
    
    # Get predictor
    predictor = _get_predictor()
    
    # Get assets to predict
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]
    
    bt.logging.info(
        f"👈 Received LSTM prediction request from: {synapse.dendrite.hotkey} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )
    
    predictions = {}
    intervals = {}
    
    for asset in assets:
        try:
            prediction, interval = predictor.predict(asset)
            
            if prediction is not None:
                predictions[asset] = prediction
                intervals[asset] = list(interval) if interval else None
                
                bt.logging.info(
                    f"{asset}: LSTM Prediction=${prediction:.2f} | "
                    f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
                )
            else:
                bt.logging.warning(f"No LSTM prediction available for {asset}")
                
        except Exception as e:
            bt.logging.error(f"LSTM prediction failed for {asset}: {e}")
    
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total LSTM forward call took: {total_time:.3f} seconds")
    
    if synapse.predictions:
        bt.logging.success(f"LSTM Predictions complete for {list(predictions.keys())}")
    else:
        bt.logging.info("No LSTM predictions for this request.")
    
    return synapse
