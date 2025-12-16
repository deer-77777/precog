#!/usr/bin/env python3
"""
Local testing script for LSTM miner
Tests the complete pipeline without deploying as a miner
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import pandas as pd

from precog.miners.lstm_miner import LSTMPredictor
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str


def test_data_fetching():
    """Test 1: Can we fetch data from CoinMetrics API?"""
    print("\n" + "=" * 60)
    print("TEST 1: Data Fetching")
    print("=" * 60)

    try:
        cm = CMData()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        print(f"Fetching BTC data from {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}...")

        data = cm.get_CM_ReferenceRate(
            assets=["btc"], start=to_str(start_time), end=to_str(end_time), frequency="1m"
        )

        if data.empty:
            print("❌ FAILED: No data fetched")
            return False

        print(f"✅ SUCCESS: Fetched {len(data)} data points")
        print(f"   Columns: {list(data.columns)}")
        print(f"\n   Sample data:")
        print(data.head(3).to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test 2: Can we create features?"""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Engineering")
    print("=" * 60)

    try:
        from precog.miners.models.feature_engineer import FeatureEngineer

        cm = CMData()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)

        print("Fetching data for feature engineering...")
        data = cm.get_CM_ReferenceRate(
            assets=["btc"], start=to_str(start_time), end=to_str(end_time), frequency="1m"
        )

        if data.empty:
            print("❌ FAILED: No data")
            return False

        print(f"Creating features from {len(data)} data points...")
        fe = FeatureEngineer(lookback_window=60)
        df_features = fe.create_features(data)

        feature_cols = [col for col in df_features.columns if col not in ["time", "asset", "ReferenceRateUSD"]]

        print(f"✅ SUCCESS: Created {len(feature_cols)} features")
        print(f"\n   Features: {', '.join(feature_cols[:10])}")
        if len(feature_cols) > 10:
            print(f"   ... and {len(feature_cols) - 10} more")

        print(f"\n   Sample features (last row):")
        print(df_features[feature_cols[:5]].tail(1).to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_loading():
    """Test 3: Can we load trained models?"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Loading")
    print("=" * 60)

    try:
        predictor = LSTMPredictor()

        if not predictor.models:
            print("⚠️  WARNING: No trained models found")
            print("   Models need to be trained first:")
            print("   python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10")
            return False

        print(f"✅ SUCCESS: Loaded {len(predictor.models)} models")
        for asset, model in predictor.models.items():
            info = predictor.model_info.get(asset, {})
            print(f"   - {asset}: epoch {info.get('epoch', '?')}, MAPE {info.get('val_mape', '?'):.4f}%")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prediction():
    """Test 4: Can we make predictions?"""
    print("\n" + "=" * 60)
    print("TEST 4: LSTM Prediction")
    print("=" * 60)

    try:
        predictor = LSTMPredictor()

        if not predictor.models:
            print("⚠️  SKIPPED: No models loaded")
            return False

        cm = CMData()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        for asset in ["btc", "eth", "tao_bittensor"]:
            if asset not in predictor.models:
                print(f"\n   Skipping {asset} (no model)")
                continue

            print(f"\n   Testing {asset.upper()}:")
            print("   " + "-" * 56)

            data = cm.get_CM_ReferenceRate(
                assets=[asset], start=to_str(start_time), end=to_str(end_time), frequency="1m"
            )

            if data.empty:
                print(f"   ❌ No data for {asset}")
                continue

            current_price = data["ReferenceRateUSD"].iloc[-1]
            print(f"   Current price: ${current_price:,.2f}")

            # Make prediction
            pred, interval = predictor.predict(asset, data)

            if pred is not None:
                diff = pred - current_price
                diff_pct = (diff / current_price) * 100

                print(f"   Predicted:     ${pred:,.2f}")
                print(f"   Difference:    ${diff:+,.2f} ({diff_pct:+.2f}%)")
                print(f"   Interval:      [${interval[0]:,.2f}, ${interval[1]:,.2f}]")
                print(f"   Width:         ${interval[1] - interval[0]:,.2f}")

                # Sanity checks
                if abs(diff_pct) > 10:
                    print(f"   ⚠️  Large difference: {diff_pct:.2f}%")
                elif interval[0] > current_price or interval[1] < current_price:
                    print(f"   ⚠️  Current price outside interval")
                else:
                    print(f"   ✅ Prediction looks reasonable")
            else:
                print(f"   ❌ Prediction failed")
                return False

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_forward_function():
    """Test 5: Test the complete forward function (as validators will call it)"""
    print("\n" + "=" * 60)
    print("TEST 5: Complete Forward Function")
    print("=" * 60)

    try:
        from precog.miners import lstm_miner

        # Create a mock synapse (simulating validator request)
        current_time = datetime.now()
        timestamp_str = to_str(current_time)

        print(f"Simulating validator request at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Create mock synapse with proper dendrite
        import bittensor as bt
        
        synapse = Challenge(timestamp=timestamp_str, assets=["btc", "eth", "tao_bittensor"])
        # Set dendrite to None to skip validation in testing
        synapse.dendrite = None

        cm = CMData()

        print("Calling forward function...")

        # Call the forward function (async)
        import asyncio

        result = asyncio.run(lstm_miner.forward(synapse, cm))

        print(f"\n✅ SUCCESS: Forward function executed")
        print(f"   Predictions: {result.predictions}")
        print(f"   Intervals:   {result.intervals}")

        # Validate results
        if not result.predictions:
            print("   ⚠️  No predictions returned")
            return False

        for asset in result.predictions:
            pred = result.predictions[asset]
            interval = result.intervals[asset]
            print(f"\n   {asset}:")
            print(f"      Prediction: ${pred:,.2f}")
            print(f"      Interval:   [${interval[0]:,.2f}, ${interval[1]:,.2f}]")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance():
    """Test 6: Measure performance metrics"""
    print("\n" + "=" * 60)
    print("TEST 6: Performance Metrics")
    print("=" * 60)

    try:
        import time

        predictor = LSTMPredictor()

        if not predictor.models:
            print("⚠️  SKIPPED: No models loaded")
            return False

        cm = CMData()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        # Test with BTC
        data = cm.get_CM_ReferenceRate(assets=["btc"], start=to_str(start_time), end=to_str(end_time), frequency="1m")

        if data.empty or "btc" not in predictor.models:
            print("⚠️  SKIPPED: No BTC data or model")
            return False

        # Measure inference time
        times = []
        for i in range(10):
            start = time.perf_counter()
            pred, interval = predictor.predict("btc", data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"✅ Inference time (10 runs):")
        print(f"   Average: {avg_time*1000:.2f}ms")
        print(f"   Min:     {min_time*1000:.2f}ms")
        print(f"   Max:     {max_time*1000:.2f}ms")

        if avg_time > 1.0:
            print(f"   ⚠️  Slow inference (>{1.0}s avg)")
        else:
            print(f"   ✅ Good performance (<1s avg)")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LSTM MINER LOCAL TESTING")
    print("=" * 60)
    print("\nThis will test your LSTM implementation locally")
    print("before deploying as a miner.\n")

    tests = [
        ("Data Fetching", test_data_fetching),
        ("Feature Engineering", test_feature_engineering),
        ("Model Loading", test_model_loading),
        ("LSTM Prediction", test_prediction),
        ("Forward Function", test_forward_function),
        ("Performance", test_performance),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️  Testing interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your LSTM miner is ready to deploy.")
        print("\nNext steps:")
        print("1. Make sure you're satisfied with the model accuracy")
        print("2. Consider training on more data (30+ days)")
        print("3. Deploy: make miner_lstm ENV_FILE=.env.miner")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before deploying.")
        if not results.get("Model Loading", False):
            print("\n💡 Tip: Train models first:")
            print("   python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

