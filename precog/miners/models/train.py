"""
Training Script for LSTM+XGBoost Hybrid Model

Usage:
    python -m precog.miners.models.train --assets btc eth tao --days 14 --epochs 20

This script trains hybrid models for the specified assets and saves them
to the model directory for use by the miner.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List

# Set TensorFlow logging level before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import bittensor as bt


def setup_logging(level: str = "info"):
    """Setup bittensor logging."""
    import logging

    log_levels = {
        "trace": logging.DEBUG,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=log_levels.get(level.lower(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_models(
    assets: List[str],
    days: int = 14,
    epochs: int = 20,
    model_dir: str = "models/lstm_xgb",
    binance_api_key: str = "",
    binance_api_secret: str = "",
    verbose: int = 1,
):
    """
    Train hybrid models for the specified assets.

    Args:
        assets: List of assets to train (btc, eth, tao)
        days: Days of historical data for training
        epochs: Number of LSTM training epochs
        model_dir: Directory to save models
        binance_api_key: Binance API key (optional for public data)
        binance_api_secret: Binance API secret
        verbose: Training verbosity (0=silent, 1=progress, 2=detailed)
    """
    from precog.miners.models.model_manager import ModelManager

    print(f"\n{'='*60}")
    print("LSTM+XGBoost Hybrid Model Training")
    print(f"{'='*60}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Training days: {days}")
    print(f"LSTM epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"{'='*60}\n")

    # Initialize manager
    manager = ModelManager(
        model_dir=model_dir,
        assets=assets,
        binance_api_key=binance_api_key,
        binance_api_secret=binance_api_secret,
        training_days=days,
        lstm_epochs=epochs,
        auto_start_retraining=False,  # Manual training
    )

    results = {}

    for asset in assets:
        print(f"\n{'='*40}")
        print(f"Training model for {asset.upper()}")
        print(f"{'='*40}")

        try:
            metrics = manager.train_model(
                asset=asset,
                days=days,
                epochs=epochs,
                verbose=verbose,
            )

            results[asset] = {
                "success": True,
                "metrics": metrics,
            }

            print(f"\n✅ {asset.upper()} training complete:")
            print(f"   RMSE (close): ${metrics['xgb_rmse_close']:.2f}")
            print(f"   RMSE (high):  ${metrics['xgb_rmse_high']:.2f}")
            print(f"   RMSE (low):   ${metrics['xgb_rmse_low']:.2f}")
            print(f"   MAE (close):  ${metrics['xgb_mae_close']:.2f}")

        except Exception as e:
            results[asset] = {
                "success": False,
                "error": str(e),
            }
            print(f"\n❌ {asset.upper()} training failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")

    successful = [a for a, r in results.items() if r["success"]]
    failed = [a for a, r in results.items() if not r["success"]]

    print(f"Successful: {len(successful)}/{len(assets)}")
    for asset in successful:
        metrics = results[asset]["metrics"]
        print(f"  ✅ {asset.upper()}: RMSE=${metrics['xgb_rmse_close']:.2f}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(assets)}")
        for asset in failed:
            print(f"  ❌ {asset.upper()}: {results[asset]['error']}")

    print(f"\nModels saved to: {model_dir}")
    print(f"{'='*60}\n")

    return results


def test_predictions(
    assets: List[str],
    model_dir: str = "models/lstm_xgb",
):
    """
    Test predictions with trained models.

    Args:
        assets: List of assets to test
        model_dir: Directory containing trained models
    """
    from precog.miners.models.model_manager import ModelManager

    print(f"\n{'='*60}")
    print("Testing Model Predictions")
    print(f"{'='*60}\n")

    manager = ModelManager(
        model_dir=model_dir,
        assets=assets,
        auto_start_retraining=False,
    )

    for asset in assets:
        print(f"\n{asset.upper()}:")

        try:
            prediction = manager.predict(asset)

            if prediction is not None:
                print(f"  Current Price: ${prediction['current_price']:,.2f}")
                print(f"  Predicted Close (1h): ${prediction['close']:,.2f}")
                print(f"  Predicted High:  ${prediction['high']:,.2f}")
                print(f"  Predicted Low:   ${prediction['low']:,.2f}")

                # Calculate predicted change
                change = (prediction["close"] - prediction["current_price"]) / prediction["current_price"] * 100
                print(f"  Predicted Change: {change:+.2f}%")
            else:
                print(f"  ⚠️ No prediction available (model not loaded)")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train LSTM+XGBoost hybrid models for crypto price prediction"
    )

    parser.add_argument(
        "--assets",
        nargs="+",
        default=["btc", "eth", "tao"],
        help="Assets to train (default: btc eth tao)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Days of historical data for training (default: 14)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of LSTM training epochs (default: 20)",
    )

    parser.add_argument(
        "--model-dir",
        default="models/lstm_xgb",
        help="Directory to save models (default: models/lstm_xgb)",
    )

    parser.add_argument(
        "--binance-api-key",
        default=os.getenv("BINANCE_API_KEY", ""),
        help="Binance API key (optional, uses env var BINANCE_API_KEY)",
    )

    parser.add_argument(
        "--binance-api-secret",
        default=os.getenv("BINANCE_API_SECRET", ""),
        help="Binance API secret (optional, uses env var BINANCE_API_SECRET)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Training verbosity: 0=silent, 1=progress, 2=detailed (default: 1)",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test predictions after training",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test predictions (skip training)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Normalize asset names
    assets = [a.lower() for a in args.assets]

    if args.test_only:
        test_predictions(assets, args.model_dir)
    else:
        results = train_models(
            assets=assets,
            days=args.days,
            epochs=args.epochs,
            model_dir=args.model_dir,
            binance_api_key=args.binance_api_key,
            binance_api_secret=args.binance_api_secret,
            verbose=args.verbose,
        )

        if args.test:
            test_predictions(assets, args.model_dir)

        # Exit with error if any training failed
        if any(not r["success"] for r in results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()

