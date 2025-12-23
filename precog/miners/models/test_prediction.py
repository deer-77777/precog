#!/usr/bin/env python3
"""
Test Script for LSTM+XGBoost Hybrid Model Predictions

Run this to verify the model works locally without starting the miner.

Usage:
    # Quick test (uses existing models or trains new ones)
    python -m precog.miners.models.test_prediction

    # Test specific assets
    python -m precog.miners.models.test_prediction --assets btc eth

    # Force retrain before testing
    python -m precog.miners.models.test_prediction --train --days 7 --epochs 10

    # Just fetch current prices (no model needed)
    python -m precog.miners.models.test_prediction --prices-only
"""

import argparse
import os
import sys
from datetime import datetime

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_prediction(asset: str, prediction: dict):
    """Print a formatted prediction."""
    current = prediction.get("current_price", 0)
    close = prediction.get("close", 0)
    high = prediction.get("high", 0)
    low = prediction.get("low", 0)

    # Calculate bounds for interval
    lower_bound = min(low, close)
    upper_bound = max(high, close)

    # Calculate predicted change
    if current > 0:
        change_pct = (close - current) / current * 100
        change_str = f"{change_pct:+.2f}%"
    else:
        change_str = "N/A"

    print(f"\n  {asset.upper()} Prediction (1-hour ahead):")
    print(f"  {'-'*40}")
    print(f"  Current Price:      ${current:>12,.2f}")
    print(f"  {'-'*40}")
    print(f"  Predicted Close:    ${close:>12,.2f}  ({change_str})")
    print(f"  Predicted High:     ${high:>12,.2f}")
    print(f"  Predicted Low:      ${low:>12,.2f}")
    print(f"  {'-'*40}")
    print(f"  Interval (bounds):  [${lower_bound:,.2f}, ${upper_bound:,.2f}]")
    print(f"  Interval Width:     ${upper_bound - lower_bound:,.2f} ({(upper_bound - lower_bound) / close * 100:.2f}%)")


def fetch_current_prices(assets: list):
    """Fetch and display current prices from Binance."""
    from precog.miners.models.binance_data import get_binance_fetcher

    print_header("Current Prices from Binance")

    fetcher = get_binance_fetcher()

    for asset in assets:
        try:
            price = fetcher.get_current_price(asset)
            if price:
                print(f"  {asset.upper()}: ${price:,.2f}")
            else:
                print(f"  {asset.upper()}: Unable to fetch price")
        except Exception as e:
            print(f"  {asset.upper()}: Error - {e}")


def test_predictions(
    assets: list,
    model_dir: str = "models/lstm_xgb",
    train_first: bool = False,
    train_days: int = 14,
    train_epochs: int = 20,
):
    """Test model predictions for given assets."""
    from precog.miners.models.model_manager import ModelManager

    print_header("LSTM+XGBoost Model Test")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Assets: {', '.join(a.upper() for a in assets)}")
    print(f"  Model Dir: {model_dir}")

    # Initialize manager
    manager = ModelManager(
        model_dir=model_dir,
        assets=assets,
        auto_start_retraining=False,
    )

    # Check model status
    print_header("Model Status")
    status = manager.get_status()

    for asset in assets:
        info = status["models"].get(asset, {})
        loaded = info.get("loaded", False)
        trained_at = info.get("trained_at", "Never")
        samples = info.get("training_samples", 0)

        if loaded:
            print(f"  {asset.upper()}: ✅ Loaded (trained: {trained_at}, samples: {samples})")
        else:
            print(f"  {asset.upper()}: ❌ Not loaded")

    # Train if requested or if no models exist
    needs_training = train_first or any(
        not status["models"].get(a, {}).get("loaded", False) for a in assets
    )

    if needs_training:
        print_header(f"Training Models ({train_days} days, {train_epochs} epochs)")
        print("  This may take a few minutes...")

        for asset in assets:
            if train_first or not status["models"].get(asset, {}).get("loaded", False):
                try:
                    print(f"\n  Training {asset.upper()}...")
                    metrics = manager.train_model(
                        asset=asset,
                        days=train_days,
                        epochs=train_epochs,
                        verbose=1,
                    )
                    print(f"  ✅ {asset.upper()}: RMSE=${metrics['xgb_rmse_close']:.2f}")
                except Exception as e:
                    print(f"  ❌ {asset.upper()}: Training failed - {e}")

    # Make predictions
    print_header("Predictions")

    results = {}
    for asset in assets:
        try:
            prediction = manager.predict(asset)
            if prediction:
                results[asset] = prediction
                print_prediction(asset, prediction)
            else:
                print(f"\n  {asset.upper()}: ⚠️ No prediction (model not available)")
        except Exception as e:
            print(f"\n  {asset.upper()}: ❌ Error - {e}")

    # Summary in format matching synapse output
    if results:
        print_header("Synapse Output Format")
        print("\n  synapse.predictions = {")
        for asset, pred in results.items():
            print(f"      '{asset}': {pred['close']:.2f},")
        print("  }")

        print("\n  synapse.intervals = {")
        for asset, pred in results.items():
            lower = min(pred["low"], pred["close"])
            upper = max(pred["high"], pred["close"])
            print(f"      '{asset}': [{lower:.2f}, {upper:.2f}],")
        print("  }")

    print("\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test LSTM+XGBoost predictions locally"
    )

    parser.add_argument(
        "--assets",
        nargs="+",
        default=["btc", "eth", "tao"],
        help="Assets to test (default: btc eth tao)",
    )

    parser.add_argument(
        "--model-dir",
        default="models/lstm_xgb",
        help="Model directory (default: models/lstm_xgb)",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Force training before testing",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Training days if --train is set (default: 14)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs if --train is set (default: 20)",
    )

    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Only fetch current prices (no model prediction)",
    )

    args = parser.parse_args()
    assets = [a.lower() for a in args.assets]

    if args.prices_only:
        fetch_current_prices(assets)
    else:
        test_predictions(
            assets=assets,
            model_dir=args.model_dir,
            train_first=args.train,
            train_days=args.days,
            train_epochs=args.epochs,
        )


if __name__ == "__main__":
    main()

