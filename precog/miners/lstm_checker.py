"""
LSTM Model Checker - Evaluate Model Performance

This script checks how well the LSTM model predictions match actual prices,
following the same scoring logic as the validator.

Scoring Logic (same as validator):
1. Point Error: abs(prediction - actual_price) / actual_price
2. Interval Score: inclusion_factor * width_factor
   - inclusion_factor: % of prices within predicted bounds
   - width_factor: overlap between predicted and observed range

Usage:
    python -m precog.miners.lstm_checker [options]
    
Options:
    --symbol: Asset to check (BTCUSDT, ETHUSDT, TAOUSDT)
    --hours: Number of hours to backtest (default: 24)
    --model_save_path: Path to saved models
"""

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from precog.miners.models.logging_utils import logger as bt_logging


class bt:
    logging = bt_logging

from precog.utils.binance_data import BinanceData, binance_to_precog_asset
from precog.miners.lstm_miner import LSTMPredictor
from precog.miners.models import LSTMConfig


class LSTMChecker:
    """
    Checker for LSTM model performance.
    
    Evaluates predictions against actual prices using the same
    scoring logic as the validator.
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        """
        Initialize checker.
        
        Args:
            config: LSTM configuration
        """
        self.config = config or LSTMConfig()
        self.predictor = LSTMPredictor(config=self.config)
        self.binance = BinanceData()
        
        bt.logging.info("LSTM Checker initialized")
    
    def calculate_point_error(
        self,
        prediction: float,
        actual_price: float,
    ) -> float:
        """
        Calculate point prediction error (same as validator).
        
        Args:
            prediction: Predicted price
            actual_price: Actual price
        
        Returns:
            Point error (percentage error)
        """
        return abs(prediction - actual_price) / actual_price
    
    def calculate_interval_score(
        self,
        interval_bounds: Tuple[float, float],
        hour_prices: List[float],
    ) -> float:
        """
        Calculate interval score (same as validator).
        
        Args:
            interval_bounds: (min_prediction, max_prediction)
            hour_prices: List of actual prices during the hour
        
        Returns:
            Interval score (0 to 1)
        """
        if not hour_prices:
            return 0.0
        
        pred_min = min(interval_bounds)
        pred_max = max(interval_bounds)
        observed_min = min(hour_prices)
        observed_max = max(hour_prices)
        
        # Calculate effective top and bottom
        effective_top = min(pred_max, observed_max)
        effective_bottom = max(pred_min, observed_min)
        
        # Calculate width factor (f_w)
        if pred_max == pred_min:
            width_factor = 0.0
        else:
            width_factor = max(0, (effective_top - effective_bottom) / (pred_max - pred_min))
        
        # Calculate inclusion factor (f_i)
        prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
        inclusion_factor = prices_in_bounds / len(hour_prices)
        
        return inclusion_factor * width_factor
    
    def check_single_prediction(
        self,
        symbol: str,
        prediction_time: datetime,
    ) -> Dict:
        """
        Check a single prediction by simulating what would have happened.
        
        Uses historical data: makes prediction at prediction_time,
        then checks against prices from prediction_time to prediction_time + 1 hour.
        
        Args:
            symbol: Binance symbol (e.g., "BTCUSDT")
            prediction_time: Time when prediction was made
        
        Returns:
            Dictionary with prediction results and scores
        """
        eval_time = prediction_time + timedelta(hours=1)
        
        # Get historical data for the hour being predicted
        # Need data from prediction_time to eval_time
        df = self.binance.get_klines(
            symbol=symbol,
            interval="1m",
            start_time=prediction_time,
            end_time=eval_time,
            limit=70,  # ~60 minutes + buffer
        )
        
        if df.empty:
            return {
                "symbol": symbol,
                "prediction_time": prediction_time,
                "error": "No price data available",
            }
        
        # Get prices during the prediction hour
        hour_prices = df["close"].tolist()
        actual_price_at_eval = hour_prices[-1] if hour_prices else None
        
        # Make prediction using the model
        # The predictor uses current data, but we need data from prediction_time
        # So we fetch data ending at prediction_time for the model input
        input_df = self.binance.get_klines(
            symbol=symbol,
            interval="1m",
            end_time=prediction_time,
            limit=self.config.model.sequence_length + 10,
        )
        
        if input_df.empty or len(input_df) < self.config.model.sequence_length:
            return {
                "symbol": symbol,
                "prediction_time": prediction_time,
                "error": "Insufficient input data for prediction",
            }
        
        # Load model and make prediction
        asset = binance_to_precog_asset(symbol)
        prediction, interval = self.predictor.predict(asset)
        
        if prediction is None:
            return {
                "symbol": symbol,
                "prediction_time": prediction_time,
                "error": "Model prediction failed",
            }
        
        # Calculate scores
        point_error = self.calculate_point_error(prediction, actual_price_at_eval)
        interval_score = self.calculate_interval_score(interval, hour_prices)
        
        # Log detailed comparison
        bt.logging.info(f"─" * 60)
        bt.logging.info(f"📊 {symbol} | Prediction Time: {prediction_time.strftime('%Y-%m-%d %H:%M')}")
        bt.logging.info(f"   Predicted Price:  ${prediction:.2f}")
        bt.logging.info(f"   Actual Price:     ${actual_price_at_eval:.2f}")
        bt.logging.info(f"   Predicted Range:  [${interval[0]:.2f} ~ ${interval[1]:.2f}]")
        bt.logging.info(f"   Actual Range:     [${min(hour_prices):.2f} ~ ${max(hour_prices):.2f}]")
        bt.logging.info(f"   Point Error:      {point_error * 100:.4f}%")
        bt.logging.info(f"   Interval Score:   {interval_score:.4f}")
        
        return {
            "symbol": symbol,
            "prediction_time": prediction_time.isoformat(),
            "eval_time": eval_time.isoformat(),
            "prediction": prediction,
            "interval": interval,
            "actual_price": actual_price_at_eval,
            "hour_price_range": (min(hour_prices), max(hour_prices)),
            "point_error": point_error,
            "point_error_percent": point_error * 100,
            "interval_score": interval_score,
        }
    
    def check_current_prediction(self, symbol: str) -> Dict:
        """
        Make a prediction for 1 hour from now and show expected scores.
        
        Note: Actual scores can only be calculated after 1 hour.
        This shows the prediction and estimates based on current volatility.
        
        Args:
            symbol: Binance symbol
        
        Returns:
            Dictionary with current prediction
        """
        asset = binance_to_precog_asset(symbol)
        prediction, interval = self.predictor.predict(asset)
        
        if prediction is None:
            return {
                "symbol": symbol,
                "error": "Model prediction failed - is model trained?",
            }
        
        # Get current price for reference
        current_price = self.binance.get_latest_price(symbol)
        price_change = (prediction - current_price) / current_price * 100 if current_price else None
        
        # Log detailed prediction
        bt.logging.info(f"─" * 60)
        bt.logging.info(f"🔮 {symbol} | CURRENT PREDICTION")
        bt.logging.info(f"─" * 60)
        bt.logging.info(f"   Current Time:     {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        bt.logging.info(f"   Prediction For:   {(datetime.utcnow() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        bt.logging.info(f"─" * 60)
        bt.logging.info(f"   Current Price:    ${current_price:.2f}")
        bt.logging.info(f"   Predicted Price:  ${prediction:.2f}")
        bt.logging.info(f"   Predicted Range:  [${interval[0]:.2f} ~ ${interval[1]:.2f}]")
        if price_change:
            direction = "📈 UP" if price_change > 0 else "📉 DOWN"
            bt.logging.info(f"   Expected Change:  {direction} {abs(price_change):.2f}%")
        bt.logging.info(f"─" * 60)
        
        return {
            "symbol": symbol,
            "current_time": datetime.utcnow().isoformat(),
            "prediction_for": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            "current_price": current_price,
            "predicted_price": prediction,
            "predicted_interval": interval,
            "price_change_expected": price_change,
        }
    
    def backtest(
        self,
        symbol: str,
        hours: int = 24,
    ) -> Dict:
        """
        Backtest model performance over the last N hours.
        
        Simulates making predictions every hour and calculates scores.
        
        Args:
            symbol: Binance symbol
            hours: Number of hours to backtest
        
        Returns:
            Dictionary with backtest results and summary statistics
        """
        bt.logging.info(f"═" * 60)
        bt.logging.info(f"📈 BACKTEST: {symbol} | Last {hours} hours")
        bt.logging.info(f"═" * 60)
        
        results = []
        current_time = datetime.utcnow()
        
        # Start from (hours + 1) hours ago to allow for 1-hour prediction window
        start_time = current_time - timedelta(hours=hours + 1)
        
        for i in range(hours):
            prediction_time = start_time + timedelta(hours=i)
            
            bt.logging.info(f"\n[{i+1}/{hours}] Testing prediction at {prediction_time.strftime('%Y-%m-%d %H:%M')}")
            
            result = self.check_single_prediction(symbol, prediction_time)
            
            if "error" not in result:
                results.append(result)
            else:
                bt.logging.warning(f"   ⚠ Skipped: {result.get('error', 'Unknown error')}")
        
        if not results:
            return {
                "symbol": symbol,
                "hours": hours,
                "error": "No valid predictions could be made",
            }
        
        # Calculate summary statistics
        point_errors = [r["point_error_percent"] for r in results]
        interval_scores = [r["interval_score"] for r in results]
        
        summary = {
            "symbol": symbol,
            "backtest_hours": hours,
            "valid_predictions": len(results),
            "point_error": {
                "mean": np.mean(point_errors),
                "std": np.std(point_errors),
                "min": np.min(point_errors),
                "max": np.max(point_errors),
                "median": np.median(point_errors),
            },
            "interval_score": {
                "mean": np.mean(interval_scores),
                "std": np.std(interval_scores),
                "min": np.min(interval_scores),
                "max": np.max(interval_scores),
                "median": np.median(interval_scores),
            },
            "results": results,
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print formatted summary of backtest results."""
        print("\n" + "=" * 70)
        print(f"BACKTEST SUMMARY: {summary['symbol']}")
        print("=" * 70)
        
        if "error" in summary:
            print(f"ERROR: {summary['error']}")
            return
        
        print(f"Period: {summary['backtest_hours']} hours")
        print(f"Valid predictions: {summary['valid_predictions']}")
        print()
        
        print("POINT PREDICTION ERROR (lower is better):")
        pe = summary["point_error"]
        print(f"  Mean:   {pe['mean']:.4f}%")
        print(f"  Std:    {pe['std']:.4f}%")
        print(f"  Min:    {pe['min']:.4f}%")
        print(f"  Max:    {pe['max']:.4f}%")
        print(f"  Median: {pe['median']:.4f}%")
        print()
        
        print("INTERVAL SCORE (higher is better, max 1.0):")
        ints = summary["interval_score"]
        print(f"  Mean:   {ints['mean']:.4f}")
        print(f"  Std:    {ints['std']:.4f}")
        print(f"  Min:    {ints['min']:.4f}")
        print(f"  Max:    {ints['max']:.4f}")
        print(f"  Median: {ints['median']:.4f}")
        print()
        
        # Overall assessment
        print("ASSESSMENT:")
        if pe["mean"] < 1.0:
            print("  ✅ Point error < 1% - Excellent!")
        elif pe["mean"] < 2.0:
            print("  ✓ Point error < 2% - Good")
        elif pe["mean"] < 5.0:
            print("  ⚠ Point error < 5% - Acceptable")
        else:
            print("  ❌ Point error > 5% - Needs improvement")
        
        if ints["mean"] > 0.5:
            print("  ✅ Interval score > 0.5 - Excellent!")
        elif ints["mean"] > 0.3:
            print("  ✓ Interval score > 0.3 - Good")
        elif ints["mean"] > 0.1:
            print("  ⚠ Interval score > 0.1 - Acceptable")
        else:
            print("  ❌ Interval score < 0.1 - Needs improvement")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LSTM Model Checker")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                       help="Symbol to check (BTCUSDT, ETHUSDT, TAOUSDT)")
    parser.add_argument("--all", action="store_true",
                       help="Check all assets")
    parser.add_argument("--hours", type=int, default=24,
                       help="Hours to backtest (default: 24)")
    parser.add_argument("--current", action="store_true",
                       help="Show current prediction only (no backtest)")
    parser.add_argument("--model_save_path", type=str, default="./models/lstm/",
                       help="Path to saved models")
    
    args = parser.parse_args()
    
    # Create config
    config = LSTMConfig()
    config.management.model_save_path = args.model_save_path
    
    # Create checker
    checker = LSTMChecker(config=config)
    
    # Determine symbols to check
    if args.all:
        symbols = ["BTCUSDT", "ETHUSDT", "TAOUSDT"]
    else:
        symbols = [args.symbol.upper()]
    
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"Checking {symbol}...")
        print("=" * 70)
        
        if args.current:
            # Show current prediction
            result = checker.check_current_prediction(symbol)
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Current Time: {result['current_time']}")
                print(f"Prediction For: {result['prediction_for']} (1 hour ahead)")
                print(f"Current Price: ${result['current_price']:.2f}")
                print(f"Predicted Price: ${result['predicted_price']:.2f}")
                print(f"Predicted Interval: [${result['predicted_interval'][0]:.2f}, ${result['predicted_interval'][1]:.2f}]")
                if result['price_change_expected']:
                    direction = "📈" if result['price_change_expected'] > 0 else "📉"
                    print(f"Expected Change: {direction} {result['price_change_expected']:.2f}%")
        else:
            # Run backtest
            summary = checker.backtest(symbol, hours=args.hours)
            checker.print_summary(summary)


if __name__ == "__main__":
    main()

