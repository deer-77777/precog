#!/usr/bin/env python3
"""Test script for LSTM predictor"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

from precog.miners.lstm_miner import LSTMPredictor
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str


def test_lstm_predictor():
    """Test the LSTM predictor with real data"""

    print("=" * 60)
    print("Testing LSTM Predictor")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing predictor...")
    predictor = LSTMPredictor()

    if not predictor.models:
        print("\n❌ No models loaded!")
        print("Train models first:")
        print("  python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10")
        return

    print(f"\n✓ Loaded {len(predictor.models)} models")

    # Fetch recent data
    print("\n2. Fetching recent data...")
    cm = CMData()
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=4)

    for asset in ["btc", "eth", "tao_bittensor"]:
        if asset not in predictor.models:
            print(f"\nSkipping {asset} (no model trained)")
            continue

        print(f"\n{'='*60}")
        print(f"Testing {asset.upper()}")
        print(f"{'='*60}")

        try:
            data = cm.get_CM_ReferenceRate(assets=[asset], start=to_str(start_time), end=to_str(end_time), frequency="1m")

            if data.empty:
                print(f"❌ No data fetched for {asset}")
                continue

            print(f"✓ Fetched {len(data)} data points")

            # Make prediction
            print("\n3. Making prediction...")
            pred, interval = predictor.predict(asset, data)

            if pred is not None:
                current_price = data["ReferenceRateUSD"].iloc[-1]
                price_diff = pred - current_price
                price_diff_pct = (price_diff / current_price) * 100

                print(f"\n📊 Results:")
                print(f"   Current price:     ${current_price:,.2f}")
                print(f"   Predicted price:   ${pred:,.2f}")
                print(f"   Difference:        ${price_diff:+,.2f} ({price_diff_pct:+.2f}%)")
                print(f"   Interval:          [${interval[0]:,.2f}, ${interval[1]:,.2f}]")
                print(f"   Interval width:    ${interval[1] - interval[0]:,.2f}")

                # Check if prediction is reasonable
                if abs(price_diff_pct) > 10:
                    print(f"\n⚠️  Warning: Large price difference ({price_diff_pct:.2f}%)")
                else:
                    print(f"\n✅ Prediction looks reasonable")
            else:
                print(f"\n❌ Prediction failed for {asset}")

        except Exception as e:
            print(f"\n❌ Error testing {asset}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_lstm_predictor()

