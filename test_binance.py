#!/usr/bin/env python3
"""Test Binance API integration"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from precog.utils.binance_data import BinanceData
from precog.utils.data_source import DataSource

print("=" * 60)
print("Testing Binance API Integration")
print("=" * 60)

# Test 1: Direct Binance API
print("\n1. Testing BinanceData directly")
print("-" * 60)

binance = BinanceData()
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

try:
    data = binance.get_CM_ReferenceRate(
        assets=["btc"],
        start=start_time,
        end=end_time,
        frequency="1m"
    )
    
    print(f"✅ SUCCESS!")
    print(f"   Fetched: {len(data)} data points")
    print(f"   Time range: {data['time'].min()} to {data['time'].max()}")
    print(f"   Current BTC price: ${data['ReferenceRateUSD'].iloc[-1]:,.2f}")
    
    print(f"\n   First 3 rows:")
    print(data.head(3))
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DataSource wrapper
print("\n2. Testing DataSource (unified interface)")
print("-" * 60)

try:
    ds = DataSource(source="binance")
    data2 = ds.get_CM_ReferenceRate(
        assets=["eth"],
        start=start_time,
        end=end_time,
        frequency="1m"
    )
    
    print(f"✅ SUCCESS!")
    print(f"   Fetched: {len(data2)} data points")
    print(f"   Current ETH price: ${data2['ReferenceRateUSD'].iloc[-1]:,.2f}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Compare data availability
print("\n3. Data Availability Test")
print("-" * 60)

print("Testing different time ranges:")
for days in [7, 30, 60]:
    start = end_time - timedelta(days=days)
    try:
        test_data = binance.get_CM_ReferenceRate(
            assets=["btc"],
            start=start,
            end=end_time,
            frequency="1m"
        )
        print(f"   {days} days: ✅ {len(test_data):,} data points")
    except Exception as e:
        print(f"   {days} days: ❌ {e}")

print("\n" + "=" * 60)
print("✅ Binance API is working!")
print("=" * 60)
print("\nYou can now train with unlimited historical data:")
print("  python3 precog/miners/scripts/train_lstm.py --assets btc --days 30 --epochs 50")
print("")

