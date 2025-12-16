# 📊 Precog Data Structure Reference

## Quick Answer

**Yes, you're building a MODEL-BASED project!** 

The base miner doesn't use any model (just statistics), but **you will create an LSTM model** to predict prices.

---

## 🗂️ What Data Fields Are Available?

### **Method 1: `get_CM_ReferenceRate()` - Simple Price Data**

This returns **3 fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `asset` | string | Asset symbol | `'btc'`, `'eth'`, `'tao_bittensor'` |
| `time` | datetime | Timestamp (UTC) | `2025-01-15 14:30:00+00:00` |
| `ReferenceRateUSD` | float | Price in USD | `95234.50` |

**Example DataFrame:**
```python
     asset                      time  ReferenceRateUSD
0      btc  2025-01-15 14:00:00+00:00          95234.50
1      btc  2025-01-15 14:01:00+00:00          95245.20
2      btc  2025-01-15 14:02:00+00:00          95198.75
3      btc  2025-01-15 14:03:00+00:00          95267.30
4      btc  2025-01-15 14:04:00+00:00          95289.40
...
```

**Usage:**
```python
from precog.utils.cm_data import CMData

cm = CMData()
data = cm.get_CM_ReferenceRate(
    assets=['btc', 'eth'],
    start='2025-01-15T14:00:00.000000Z',
    end='2025-01-15T18:00:00.000000Z',
    frequency='1m'  # Options: '1s', '1m', '5m', '1h', '1d'
)

print(data.head())
#      asset                      time  ReferenceRateUSD
# 0      btc  2025-01-15 14:00:00+00:00          95234.50
# 1      btc  2025-01-15 14:01:00+00:00          95245.20
```

---

### **Method 2: `get_pair_candles()` - OHLC Candle Data (MORE FEATURES!)**

This returns **6 fields** - much better for ML models!

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `pair` | string | Trading pair | `'btc-usd'`, `'eth-usd'` |
| `time` | datetime | Candle timestamp | `2025-01-15 14:00:00+00:00` |
| `price_open` | float | Opening price | `95234.50` |
| `price_close` | float | Closing price | `95245.20` |
| `price_high` | float | Highest price | `95280.00` |
| `price_low` | float | Lowest price | `95200.00` |

**Example DataFrame:**
```python
      pair                      time  price_open  price_close  price_high  price_low
0  btc-usd  2025-01-15 14:00:00+00:00    95234.50     95245.20    95280.00   95200.00
1  btc-usd  2025-01-15 14:05:00+00:00    95245.20     95198.75    95250.00   95180.00
2  btc-usd  2025-01-15 14:10:00+00:00    95198.75     95267.30    95290.00   95195.00
```

**Usage:**
```python
data = cm.get_pair_candles(
    pairs=['btc-usd', 'eth-usd'],
    start='2025-01-15T14:00:00.000000Z',
    end='2025-01-15T18:00:00.000000Z',
    frequency='5m'  # 5-minute candles
)
```

---

## 🧠 What Features Can You Create for LSTM?

### **Raw Features (from API):**
1. ✅ **Price** - `ReferenceRateUSD` or `price_close`
2. ✅ **Open, High, Low, Close** - From candles
3. ✅ **Time** - Hour, day of week, etc.

### **Calculated Features (you create these):**

#### **Price-based:**
```python
df['returns'] = df['price'].pct_change()           # % change
df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
```

#### **Moving Averages:**
```python
df['sma_10'] = df['price'].rolling(10).mean()      # Simple MA
df['ema_10'] = df['price'].ewm(span=10).mean()     # Exponential MA
```

#### **Volatility:**
```python
df['volatility'] = df['returns'].rolling(20).std()
df['atr'] = calculate_average_true_range(df)       # Average True Range
```

#### **Technical Indicators:**
```python
df['rsi'] = calculate_rsi(df['price'], 14)         # Relative Strength Index
df['macd'] = calculate_macd(df['price'])           # MACD
df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['price'])
```

#### **Momentum:**
```python
df['momentum'] = df['price'] - df['price'].shift(10)
df['roc'] = df['price'].pct_change(10)             # Rate of Change
```

#### **Lag Features:**
```python
df['price_lag_1'] = df['price'].shift(1)           # Previous price
df['price_lag_5'] = df['price'].shift(5)
df['price_lag_10'] = df['price'].shift(10)
```

#### **Time Features:**
```python
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)  # Cyclical encoding
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---

## 🎯 LSTM Input Format

LSTM expects data in this shape:

```
(num_samples, lookback_window, num_features)
```

**Example:**
- `num_samples`: 1000 prediction instances
- `lookback_window`: 60 (last 60 minutes)
- `num_features`: 20 (price + all calculated features)

**Shape: `(1000, 60, 20)`**

### **Visual Example:**

```python
# For one prediction:
# Look at last 60 minutes, each with 20 features

Input:
[
  [95234.50, 0.001, 95000, ...],  # Minute 1: [price, return, sma, ...]
  [95245.20, 0.002, 95010, ...],  # Minute 2
  [95198.75, -0.001, 95020, ...], # Minute 3
  ...
  [95289.40, 0.001, 95200, ...]   # Minute 60
]
    ↓
  LSTM Model
    ↓
Output: 95310.25  # Predicted price for minute 61
```

---

## 📋 Complete Example

```python
from precog.utils.cm_data import CMData
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 1. Fetch data
cm = CMData()
end_time = datetime.now()
start_time = end_time - timedelta(hours=4)

data = cm.get_CM_ReferenceRate(
    assets=['btc'],
    start=start_time.strftime('%Y-%m-%dT%H:%M:%S.000000Z'),
    end=end_time.strftime('%Y-%m-%dT%H:%M:%S.000000Z'),
    frequency='1m'
)

print(f"Fetched {len(data)} rows")
print(f"Columns: {list(data.columns)}")
print(f"\nFirst 5 rows:")
print(data.head())

# 2. Create features
data['returns'] = data['ReferenceRateUSD'].pct_change()
data['sma_10'] = data['ReferenceRateUSD'].rolling(10).mean()
data['volatility'] = data['returns'].rolling(20).std()
data['hour'] = data['time'].dt.hour

print(f"\nAfter feature engineering:")
print(f"Columns: {list(data.columns)}")
print(data.tail())

# 3. Prepare sequences for LSTM
lookback = 60
features = ['ReferenceRateUSD', 'returns', 'sma_10', 'volatility', 'hour']
data_clean = data.dropna()

sequences = []
targets = []

for i in range(len(data_clean) - lookback - 60):  # -60 for 1hr prediction
    # Input: last 60 minutes of features
    seq = data_clean[features].iloc[i:i+lookback].values
    sequences.append(seq)
    
    # Target: price 60 minutes later
    target = data_clean['ReferenceRateUSD'].iloc[i+lookback+60]
    targets.append(target)

sequences = np.array(sequences)
targets = np.array(targets)

print(f"\n📊 LSTM Input Shape: {sequences.shape}")
print(f"   - {sequences.shape[0]} samples")
print(f"   - {sequences.shape[1]} timesteps (lookback)")
print(f"   - {sequences.shape[2]} features")
print(f"\n🎯 Target Shape: {targets.shape}")
```

---

## 🚀 Next Steps

1. ✅ **Read**: `LSTM_MINER_GUIDE.md` - Complete implementation guide
2. ✅ **Create**: Feature engineering module
3. ✅ **Build**: LSTM model architecture
4. ✅ **Train**: On historical data (30+ days)
5. ✅ **Test**: Validate predictions
6. ✅ **Deploy**: As your forward function

**Your Path:**
```
Historical Data → Feature Engineering → LSTM Training → 
Prediction → Deployment → Earning Rewards!
```

Good luck! 🎉

