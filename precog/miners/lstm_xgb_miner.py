"""
LSTM+XGBoost Hybrid Miner for Precog Subnet

This miner uses a hybrid LSTM+XGBoost model to predict cryptocurrency prices.
It replaces the simple volatility-based baseline with a trained ML model.

Usage:
    1. Set FORWARD_FUNCTION=lstm_xgb_miner in .env.miner
    2. Run: make miner ENV_FILE=.env.miner

The model:
    - Fetches 1m OHLCV data from Binance
    - Creates technical features (lags, RSI, BB, MACD, etc.)
    - LSTM extracts temporal embeddings
    - XGBoost predicts [close, high, low] for 1-hour ahead
    - Retrains automatically every 4 hours
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData

# Import hybrid model components
from precog.miners.models.model_manager import ModelManager, get_model_manager

# Global model manager instance
_model_manager: Optional[ModelManager] = None

# Configuration from environment
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
MODEL_DIR = os.getenv("LSTM_XGB_MODEL_DIR", "models/lstm_xgb")
TRAINING_DAYS = int(os.getenv("LSTM_XGB_TRAINING_DAYS", "14"))
RETRAIN_INTERVAL_HOURS = float(os.getenv("LSTM_XGB_RETRAIN_HOURS", "1.0"))
LSTM_EPOCHS = int(os.getenv("LSTM_XGB_EPOCHS", "20"))


def get_manager() -> ModelManager:
    """Get or initialize the global model manager."""
    global _model_manager

    if _model_manager is None:
        bt.logging.info("Initializing LSTM+XGBoost Model Manager...")
        _model_manager = ModelManager(
            model_dir=MODEL_DIR,
            assets=["btc", "eth", "tao"],
            binance_api_key=BINANCE_API_KEY,
            binance_api_secret=BINANCE_API_SECRET,
            training_days=TRAINING_DAYS,
            retrain_interval_hours=RETRAIN_INTERVAL_HOURS,
            lstm_epochs=LSTM_EPOCHS,
            auto_start_retraining=True,  # Enable background retraining
        )
        bt.logging.info("Model Manager initialized")

        # Log initial status
        status = _model_manager.get_status()
        for asset, info in status["models"].items():
            if info["loaded"]:
                bt.logging.info(
                    f"  {asset.upper()}: Loaded (trained at {info['trained_at']}, "
                    f"samples={info['training_samples']})"
                )
            else:
                bt.logging.warning(f"  {asset.upper()}: Not loaded - will train on first request")

    return _model_manager


def calculate_prediction_interval_ml(
    close_pred: float,
    high_pred: float,
    low_pred: float,
    current_price: float,
) -> Tuple[float, float]:
    """
    Calculate prediction interval from ML model outputs.

    The model predicts high and low bounds directly, but we apply
    some sanity checks and adjustments.

    Args:
        close_pred: Predicted close price
        high_pred: Predicted high price
        low_pred: Predicted low price
        current_price: Current market price

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Ensure bounds are ordered correctly
    lower = min(low_pred, high_pred, close_pred)
    upper = max(low_pred, high_pred, close_pred)

    # Minimum spread: at least 0.5% of close price
    min_spread = close_pred * 0.005
    if upper - lower < min_spread:
        spread_adjustment = (min_spread - (upper - lower)) / 2
        upper += spread_adjustment
        lower -= spread_adjustment

    # Maximum spread: cap at 5% of close price to avoid crazy wide bounds
    max_spread = close_pred * 0.05
    if upper - lower > max_spread:
        center = (upper + lower) / 2
        upper = center + max_spread / 2
        lower = center - max_spread / 2

    # Ensure bounds are positive
    lower = max(lower, close_pred * 0.9)  # Floor at -10% of prediction
    upper = max(upper, lower + min_spread)

    return lower, upper


def fallback_prediction(
    current_price: float,
    historical_prices: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Fallback prediction when ML model is unavailable.

    Uses simple volatility-based estimation similar to base_miner.

    Args:
        current_price: Current market price
        historical_prices: Optional historical price series for volatility

    Returns:
        Dictionary with close, high, low predictions
    """
    if historical_prices is not None and len(historical_prices) > 10:
        # Calculate hourly volatility
        returns = historical_prices.pct_change().dropna()
        volatility = returns.std() * 2.58  # 99% confidence
        margin = current_price * max(0.02, min(volatility, 0.10))
    else:
        # Fallback: 3% margin
        margin = current_price * 0.03

    return {
        "close": current_price,
        "high": current_price + margin,
        "low": current_price - margin,
    }


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """
    Forward function for the LSTM+XGBoost hybrid miner.

    This is called by the miner when a validator sends a prediction request.

    Args:
        synapse: Challenge synapse with timestamp and assets
        cm: CoinMetrics data client (for fallback)

    Returns:
        Challenge synapse with predictions and intervals filled
    """
    total_start_time = time.perf_counter()

    # Get list of assets to predict
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]

    bt.logging.info(
        f"👈 LSTM+XGBoost: Received request from {synapse.dendrite.hotkey} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )

    # Get model manager
    try:
        manager = get_manager()
    except Exception as e:
        bt.logging.error(f"Failed to initialize model manager: {e}")
        manager = None

    predictions = {}
    intervals = {}

    for asset in assets:
        try:
            # Try ML prediction first
            if manager is not None:
                ml_pred = manager.predict(asset)

                if ml_pred is not None:
                    close_pred = ml_pred["close"]
                    high_pred = ml_pred["high"]
                    low_pred = ml_pred["low"]
                    current_price = ml_pred["current_price"]

                    # Calculate interval bounds
                    lower, upper = calculate_prediction_interval_ml(
                        close_pred, high_pred, low_pred, current_price
                    )

                    predictions[asset] = close_pred
                    intervals[asset] = [lower, upper]

                    bt.logging.info(
                        f"  {asset.upper()}: ML Prediction=${close_pred:.2f} "
                        f"| Interval=[${lower:.2f}, ${upper:.2f}] "
                        f"| Current=${current_price:.2f}"
                    )
                    continue

            # Fallback to volatility-based if ML unavailable
            bt.logging.warning(f"  {asset.upper()}: Using fallback prediction (no ML model)")

            # Get current price from Binance or CoinMetrics
            current_price = None

            if manager is not None:
                current_price = manager.data_fetcher.get_current_price(asset)

            if current_price is None:
                # Try CoinMetrics fallback
                from precog.utils.timestamp import get_before, to_datetime, to_str

                provided_timestamp = to_datetime(synapse.timestamp)
                start_timestamp = get_before(synapse.timestamp, hours=1)

                cm_data = cm.get_CM_ReferenceRate(
                    assets=[asset],
                    start=to_str(start_timestamp),
                    end=to_str(provided_timestamp),
                    frequency="1s",
                )

                if not cm_data.empty:
                    asset_data = cm_data[cm_data["asset"] == asset]
                    if not asset_data.empty:
                        current_price = float(asset_data["ReferenceRateUSD"].iloc[-1])

            if current_price is None:
                bt.logging.error(f"  {asset.upper()}: Could not get current price")
                continue

            # Use fallback
            fallback = fallback_prediction(current_price)
            predictions[asset] = fallback["close"]
            intervals[asset] = [fallback["low"], fallback["high"]]

            bt.logging.info(
                f"  {asset.upper()}: Fallback Prediction=${fallback['close']:.2f} "
                f"| Interval=[${fallback['low']:.2f}, ${fallback['high']:.2f}]"
            )

        except Exception as e:
            bt.logging.error(f"  {asset.upper()}: Error during prediction: {e}")
            continue

    synapse.predictions = predictions
    synapse.intervals = intervals

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ LSTM+XGBoost forward call took: {total_time:.3f} seconds")

    if synapse.predictions:
        bt.logging.success(f"✅ Predictions complete for {list(predictions.keys())}")
    else:
        bt.logging.warning("⚠️ No predictions generated")

    return synapse


async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    """Async wrapper for forward function."""
    return await forward(synapse, cm)

