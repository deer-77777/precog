"""LSTM-based miner for Precog subnet"""

import time
from pathlib import Path

import bittensor as bt
import numpy as np
import pandas as pd
import torch

from precog.miners.models.feature_engineer import FeatureEngineer
from precog.miners.models.lstm_model import PriceLSTM
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str


class LSTMPredictor:
    """LSTM-based price predictor"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.feature_engineer = FeatureEngineer(lookback_window=60)
        self.feature_cols = {}
        self.model_info = {}

        bt.logging.info(f"LSTMPredictor initialized on device: {self.device}")

        # Load trained models
        self.load_models()

    def load_models(self):
        """Load pre-trained LSTM models"""
        for asset in ["btc", "eth", "tao_bittensor"]:
            model_path = Path(f"precog/miners/models/trained_weights/{asset}_lstm.pth")

            if not model_path.exists():
                bt.logging.warning(f"⚠️  Model for {asset} not found at {model_path}")
                bt.logging.warning(f"    Run training first: python3 precog/miners/scripts/train_lstm.py")
                continue

            try:
                checkpoint = torch.load(model_path, map_location=self.device)

                # Initialize model
                model = PriceLSTM(input_size=checkpoint["input_size"], hidden_size=128, num_layers=2)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()

                self.models[asset] = model
                self.feature_cols[asset] = checkpoint["feature_cols"]
                self.model_info[asset] = {
                    "epoch": checkpoint.get("epoch", "unknown"),
                    "val_loss": checkpoint.get("val_loss", "unknown"),
                    "val_mape": checkpoint.get("val_mape", "unknown"),
                }

                bt.logging.success(
                    f"✓ Loaded LSTM model for {asset} "
                    f"(epoch {self.model_info[asset]['epoch']}, "
                    f"MAPE: {self.model_info[asset]['val_mape']:.4f}%)"
                )
            except Exception as e:
                bt.logging.error(f"❌ Failed to load model for {asset}: {e}")

        if not self.models:
            bt.logging.warning("⚠️  No LSTM models loaded! Will use fallback predictions.")
            bt.logging.warning("    Train models first: python3 precog/miners/scripts/train_lstm.py")

    def predict(self, asset, data_df):
        """Make prediction for an asset"""
        if asset not in self.models:
            bt.logging.debug(f"No LSTM model for {asset}, using fallback")
            return None, None

        try:
            # Create features
            df_features = self.feature_engineer.create_features(data_df)

            # Prepare sequence
            sequences = self.feature_engineer.prepare_sequences(df_features, self.feature_cols[asset])

            if len(sequences) == 0:
                bt.logging.warning(f"Not enough data for {asset} (need at least 60 minutes)")
                return None, None

            # Get last sequence
            last_sequence = sequences[-1:]

            # Predict
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
                prediction = self.models[asset](sequence_tensor)
                predicted_price = prediction.cpu().numpy()[0][0]

            # Calculate interval using recent volatility
            recent_prices = df_features["ReferenceRateUSD"].tail(60)
            returns = recent_prices.pct_change().dropna()

            if len(returns) > 0:
                volatility = returns.std()

                # Confidence interval (2.58 std = 99% confidence)
                margin = predicted_price * volatility * 2.58

                # Apply safety bounds (2% min, 30% max)
                margin = max(predicted_price * 0.02, min(margin, predicted_price * 0.30))
            else:
                # Fallback margin
                margin = predicted_price * 0.10

            lower_bound = predicted_price - margin
            upper_bound = predicted_price + margin

            return predicted_price, [lower_bound, upper_bound]

        except Exception as e:
            bt.logging.error(f"Prediction error for {asset}: {e}")
            import traceback

            bt.logging.debug(traceback.format_exc())
            return None, None

    def fallback_prediction(self, asset_data):
        """Fallback to naive prediction when LSTM not available"""
        last_price = float(asset_data["ReferenceRateUSD"].iloc[-1])

        # Calculate volatility for interval
        recent_prices = asset_data["ReferenceRateUSD"].tail(60)
        if len(recent_prices) > 1:
            returns = recent_prices.pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std()
                margin = last_price * volatility * 2.58
                margin = max(last_price * 0.02, min(margin, last_price * 0.30))
            else:
                margin = last_price * 0.10
        else:
            margin = last_price * 0.10

        return last_price, [last_price - margin, last_price + margin]


# Global predictor instance (initialized once)
predictor = None


def initialize_predictor():
    """Initialize the global predictor instance"""
    global predictor
    if predictor is None:
        predictor = LSTMPredictor()
    return predictor


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """LSTM-based forward function for miner"""
    total_start_time = time.perf_counter()

    # Initialize predictor if not already done
    global predictor
    if predictor is None:
        predictor = initialize_predictor()

    # Get list of assets to predict
    assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in assets]

    bt.logging.info(
        f"👈 LSTM prediction request from {synapse.dendrite.hotkey} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )

    # Fetch last 4 hours of data (for feature engineering)
    provided_timestamp = to_datetime(synapse.timestamp)
    start_timestamp = get_before(
        synapse.timestamp, hours=4, minutes=0, seconds=0
    )  # 4 hours for features

    all_data = cm.get_CM_ReferenceRate(
        assets=assets, start=to_str(start_timestamp), end=to_str(provided_timestamp), frequency="1m"
    )

    predictions = {}
    intervals = {}

    if not all_data.empty:
        for asset in assets:
            # Filter data for this asset
            asset_data = all_data[all_data["asset"] == asset]

            if not asset_data.empty:
                # Try LSTM prediction first
                pred, interval = predictor.predict(asset, asset_data)

                if pred is not None:
                    # LSTM prediction successful
                    predictions[asset] = float(pred)
                    intervals[asset] = [float(interval[0]), float(interval[1])]
                    bt.logging.success(
                        f"🧠 {asset}: LSTM Prediction=${pred:.2f} | "
                        f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
                    )
                else:
                    # Fallback to naive prediction
                    pred, interval = predictor.fallback_prediction(asset_data)
                    predictions[asset] = float(pred)
                    intervals[asset] = [float(interval[0]), float(interval[1])]
                    bt.logging.info(
                        f"📊 {asset}: Fallback Prediction=${pred:.2f} | "
                        f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
                    )
            else:
                bt.logging.warning(f"No data for {asset} in response")
    else:
        bt.logging.warning("No data fetched from API")

    synapse.predictions = predictions
    synapse.intervals = intervals

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️  Total forward call took: {total_time:.3f} seconds")

    if synapse.predictions:
        bt.logging.success(f"✅ Predictions complete for {list(predictions.keys())}")
    else:
        bt.logging.info("No predictions for this request.")

    return synapse

