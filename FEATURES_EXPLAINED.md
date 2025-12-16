# 📊 LSTM Features Explained - Complete Guide

This guide explains every feature created by `FeatureEngineer` with examples.

---

## 🎯 Overview

The LSTM model uses **27+ technical features** to predict cryptocurrency prices. Each feature captures different aspects of price behavior.

**Raw Input:**
```
time, ReferenceRateUSD (price)
```

**Feature Output:**
```
27+ features including price patterns, trends, volatility, etc.
```

---

## 1️⃣ **Basic Price Features**

### **Feature: `price`**

```python
df["price"] = df["ReferenceRateUSD"]
```

**What it is:** The actual price of the asset in USD

**Example:**
```
Time         | ReferenceRateUSD | price
-------------|------------------|--------
14:00        | 89,650.00        | 89,650.00
14:01        | 89,670.00        | 89,670.00
14:02        | 89,680.00        | 89,680.00
```

**Why it's useful:**
- Base feature for all calculations
- LSTM learns price levels and patterns
- Most important feature

---

## 2️⃣ **Returns (Price Changes)**

Returns show **how much the price changed** as a percentage.

### **Feature: `returns_1`** (1-minute return)

```python
df["returns_1"] = df["price"].pct_change(1)
```

**Formula:** `(Current Price - Previous Price) / Previous Price`

**Example:**
```
Time  | Price      | returns_1  | Meaning
------|------------|------------|------------------
14:00 | 89,650.00  | NaN        | No previous data
14:01 | 89,670.00  | 0.000223   | +0.0223% increase
14:02 | 89,680.00  | 0.000112   | +0.0112% increase
14:03 | 89,600.00  | -0.000892  | -0.0892% decrease
```

**Calculation:**
```python
returns_1 at 14:01 = (89,670 - 89,650) / 89,650 = 0.000223
returns_1 at 14:02 = (89,680 - 89,670) / 89,670 = 0.000112
returns_1 at 14:03 = (89,600 - 89,680) / 89,680 = -0.000892
```

**Why it's useful:**
- Shows momentum (going up or down)
- Normalized (works for any price level)
- Captures short-term trends

### **Feature: `returns_5`** (5-minute return)

```python
df["returns_5"] = df["price"].pct_change(5)
```

**Example:**
```
Time  | Price      | returns_5  | Meaning
------|------------|------------|------------------
14:00 | 89,650.00  | NaN        | 
14:05 | 89,750.00  | 0.001116   | +0.11% over 5 min
```

**Why it's useful:**
- Medium-term momentum
- Smooths out noise

### **Feature: `returns_10`** (10-minute return)

```python
df["returns_10"] = df["price"].pct_change(10)
```

**Why it's useful:**
- Longer-term trend
- Less noisy than 1-minute

---

## 3️⃣ **Moving Averages (Trend Detection)**

Moving averages smooth out price data to show trends.

### **Feature: `sma_5`** (5-period Simple Moving Average)

```python
df["sma_5"] = df["price"].rolling(window=5).mean()
```

**Formula:** Average of last 5 prices

**Example:**
```
Time  | Price      | sma_5      | Calculation
------|------------|------------|----------------------------------
14:00 | 89,650.00  | NaN        | Not enough data
14:01 | 89,670.00  | NaN        | 
14:02 | 89,680.00  | NaN        | 
14:03 | 89,600.00  | NaN        | 
14:04 | 89,700.00  | 89,660.00  | (89,650+89,670+89,680+89,600+89,700)/5
14:05 | 89,720.00  | 89,674.00  | (89,670+89,680+89,600+89,700+89,720)/5
```

**Why it's useful:**
- Smooths out noise
- Shows overall direction
- When price > SMA → uptrend
- When price < SMA → downtrend

### **Feature: `sma_10`, `sma_20`, `sma_50`**

Same concept, but longer windows:
- `sma_10`: 10-minute average (short-term trend)
- `sma_20`: 20-minute average (medium-term trend)
- `sma_50`: 50-minute average (long-term trend)

**Example - Trend Detection:**
```
Time  | Price      | sma_5      | sma_20     | Trend
------|------------|------------|------------|------------
14:00 | 89,850.00  | 89,700.00  | 89,600.00  | Strong uptrend
      | (price above both SMAs)              | Price > SMA_5 > SMA_20
```

### **Feature: `ema_5`, `ema_20`** (Exponential Moving Average)

```python
df["ema_5"] = df["price"].ewm(span=5, adjust=False).mean()
df["ema_20"] = df["price"].ewm(span=20, adjust=False).mean()
```

**Difference from SMA:**
- SMA: All prices weighted equally
- EMA: Recent prices weighted MORE

**Example:**
```
Time  | Price      | sma_5      | ema_5      | Difference
------|------------|------------|------------|------------------
14:00 | 89,650.00  | 89,600.00  | 89,620.00  | EMA reacts faster
14:01 | 89,850.00  | 89,650.00  | 89,710.00  | EMA gives more weight
      |            |            |            | to recent spike
```

**Why it's useful:**
- Reacts faster to price changes
- Better for detecting trend changes
- Used in trading strategies

---

## 4️⃣ **Volatility (How Much Price Moves)**

Volatility measures how much the price jumps around.

### **Feature: `volatility_5`, `volatility_20`**

```python
df["volatility_5"] = df["returns_1"].rolling(window=5).std()
df["volatility_20"] = df["returns_1"].rolling(window=20).std()
```

**Formula:** Standard deviation of returns

**Example:**
```
Time  | returns_1  | volatility_5 | Meaning
------|------------|--------------|-------------------------
14:00 | 0.0001     | 0.0003       | Low volatility (stable)
14:01 | 0.0002     |              | Small changes
14:02 | 0.0001     |              |
14:03 | -0.0001    |              |
14:04 | 0.0002     |              |

vs

Time  | returns_1  | volatility_5 | Meaning
------|------------|--------------|-------------------------
15:00 | 0.005      | 0.008        | High volatility (jumpy)
15:01 | -0.003     |              | Big swings
15:02 | 0.007      |              | Unpredictable
15:03 | -0.004     |              |
15:04 | 0.006      |              |
```

**Why it's useful:**
- Predicts risk
- Used to calculate prediction intervals
- High volatility → wider prediction intervals
- Low volatility → tighter prediction intervals

**Real scenario:**
```
If BTC volatility is HIGH:
  → Model predicts: $89,500 ± $500 (wide interval)
  
If BTC volatility is LOW:
  → Model predicts: $89,500 ± $100 (tight interval)
```

---

## 5️⃣ **Momentum (Strength of Price Movement)**

Momentum shows the **speed and direction** of price changes.

### **Feature: `momentum_5`, `momentum_10`**

```python
df["momentum_5"] = df["price"] - df["price"].shift(5)
df["momentum_10"] = df["price"] - df["price"].shift(10)
```

**Formula:** Current Price - Price N periods ago

**Example:**
```
Time  | Price      | momentum_5 | Meaning
------|------------|------------|------------------
14:00 | 89,650.00  | NaN        |
14:05 | 89,750.00  | +100.00    | Gained $100 in 5 min
14:10 | 89,800.00  | +50.00     | Gained $50 in 5 min
14:15 | 89,780.00  | +30.00     | Slowing down
14:20 | 89,760.00  | -40.00     | Losing momentum
```

**Difference from Returns:**
- **Returns**: Percentage change (normalized)
- **Momentum**: Absolute dollar change

**Why it's useful:**
- Shows acceleration/deceleration
- Positive momentum → price gaining
- Negative momentum → price losing
- Decreasing momentum → trend weakening

---

## 6️⃣ **RSI (Relative Strength Index)**

RSI measures if an asset is **overbought** or **oversold**.

### **Feature: `rsi_14`**

```python
df["rsi_14"] = self.calculate_rsi(df["price"], 14)
```

**Formula:**
```python
# Step 1: Calculate gains and losses
gains = price increases over 14 periods
losses = price decreases over 14 periods

# Step 2: Calculate average gain/loss
avg_gain = average of gains
avg_loss = average of losses

# Step 3: Calculate RS
RS = avg_gain / avg_loss

# Step 4: Calculate RSI
RSI = 100 - (100 / (1 + RS))
```

**Range:** 0 to 100

**Example:**
```
RSI Value | Meaning           | Action Signal
----------|-------------------|------------------
80-100    | Overbought        | Price too high, might drop
70-80     | Strong uptrend    | Careful, could reverse
50-70     | Normal uptrend    | Healthy growth
30-50     | Normal downtrend  | Healthy correction
20-30     | Strong downtrend  | Oversold signal
0-20      | Oversold          | Price too low, might rally
```

**Real Example:**
```
Time  | Price      | RSI    | Interpretation
------|------------|--------|----------------------------------
14:00 | 88,000.00  | 35     | Normal, slightly bearish
14:30 | 89,000.00  | 55     | Gaining strength
15:00 | 90,500.00  | 75     | Strong uptrend, caution
15:30 | 91,000.00  | 82     | OVERBOUGHT! Likely to drop soon
16:00 | 89,500.00  | 60     | Correction happened, back to normal
```

**Why it's useful:**
- Predicts reversals
- RSI > 70: Price might drop soon
- RSI < 30: Price might rally soon
- Used in trading strategies worldwide

---

## 7️⃣ **Bollinger Bands (Volatility Bands)**

Bollinger Bands create upper and lower boundaries around price.

### **Features: `bb_upper`, `bb_lower`, `bb_width`**

```python
df["bb_upper"], df["bb_lower"] = self.calculate_bollinger_bands(df["price"], 20)
df["bb_width"] = df["bb_upper"] - df["bb_lower"]
```

**Formula:**
```python
# Middle band = 20-period SMA
middle = sma_20

# Standard deviation of prices
std = standard_deviation(20 periods)

# Upper band = middle + (2 × std)
bb_upper = middle + (2 × std)

# Lower band = middle - (2 × std)
bb_lower = middle - (2 × std)

# Width = difference
bb_width = bb_upper - bb_lower
```

**Example:**
```
Time  | Price      | bb_upper   | bb_lower   | bb_width | Position
------|------------|------------|------------|----------|-------------
14:00 | 89,500.00  | 90,000.00  | 89,000.00  | 1,000.00 | Middle
14:30 | 89,950.00  | 90,200.00  | 88,800.00  | 1,400.00 | Near upper
15:00 | 90,100.00  | 90,300.00  | 88,700.00  | 1,600.00 | ABOVE upper!
      |            |            |            |          | Overbought
```

**Visual Example:**
```
Price Chart:

90,500 ┤━━━━━━━━━━━ bb_upper (resistance)
       │     ╱╲
89,500 ┤────╱──╲─── Price bounces between bands
       │   ╱    ╲
88,500 ┤━━╱══════╲═ bb_lower (support)
       │
```

**Interpretations:**

1. **Price touches upper band:**
   - Price is HIGH
   - Might reverse downward
   
2. **Price touches lower band:**
   - Price is LOW
   - Might bounce upward
   
3. **Wide bands (`bb_width` large):**
   - High volatility
   - Big price swings expected
   
4. **Narrow bands (`bb_width` small):**
   - Low volatility
   - Calm market, breakout coming

**Why it's useful:**
- Shows price boundaries
- Detects overbought/oversold
- Measures volatility
- `bb_width` predicts big moves

---

## 8️⃣ **Time Features (Patterns in Time)**

Cryptocurrency markets have patterns based on time of day.

### **Features: `hour`, `day_of_week`, `hour_sin`, `hour_cos`**

```python
df["hour"] = df["time"].dt.hour                    # 0-23
df["day_of_week"] = df["time"].dt.dayofweek        # 0-6 (Mon-Sun)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

**Example:**
```
Time               | hour | day_of_week | hour_sin | hour_cos | Patterns
-------------------|------|-------------|----------|----------|-------------
Mon 00:00 (night)  |  0   |     0       |  0.0     |  1.0     | Low volume
Mon 09:00 (morning)|  9   |     0       |  0.7     |  0.7     | US wakes up
Mon 14:00 (noon)   | 14   |     0       |  0.8     | -0.2     | Peak trading
Mon 22:00 (night)  | 22   |     0       | -0.8     |  0.3     | Asia trading
Sat 14:00          | 14   |     5       |  0.8     | -0.2     | Weekend (slow)
```

**Why time matters:**

**Hour patterns:**
```
00:00-08:00 (Night):  Low volume, smaller moves
09:00-16:00 (Day):    High volume, bigger moves (US markets)
16:00-00:00 (Evening): Medium volume (Europe/Asia)
```

**Day of week patterns:**
```
Monday:     Often volatile (weekend news)
Mid-week:   Normal trading
Friday:     Position closing, less risk
Weekend:    Lower volume, can be more volatile
```

**Why sin/cos encoding:**

Instead of using hour directly (0-23), we use sin/cos because:
```
Hour 23 (11pm) and Hour 0 (midnight) are close in time
But 23 and 0 are far apart numerically!

With sin/cos:
  hour_sin(23) ≈ hour_sin(0)  ← Model understands they're close!
  hour_cos(23) ≈ hour_cos(0)
```

**Why it's useful:**
- Captures daily cycles
- Model learns "US trading hours are volatile"
- Model learns "3am is usually quiet"
- Improves predictions by 5-10%

---

## 9️⃣ **Lag Features (Recent Price History)**

Lag features give the model **memory** of recent prices.

### **Features: `price_lag_1`, `price_lag_2`, etc.**

```python
for lag in [1, 2, 3, 5, 10]:
    df[f"price_lag_{lag}"] = df["price"].shift(lag)
```

**Example:**
```
Time  | price      | price_lag_1 | price_lag_2 | price_lag_5 | price_lag_10
------|------------|-------------|-------------|-------------|-------------
14:00 | 89,650.00  | NaN         | NaN         | NaN         | NaN
14:01 | 89,670.00  | 89,650.00   | NaN         | NaN         | NaN
14:02 | 89,680.00  | 89,670.00   | 89,650.00   | NaN         | NaN
14:05 | 89,700.00  | 89,690.00   | 89,680.00   | 89,650.00   | NaN
14:10 | 89,750.00  | 89,740.00   | 89,730.00   | 89,700.00   | 89,650.00
```

**What the model sees:**
```
At 14:10, model knows:
  - Current price: 89,750
  - 1 min ago: 89,740
  - 2 min ago: 89,730
  - 5 min ago: 89,700
  - 10 min ago: 89,650

Model thinks: "Price steadily rising for 10 minutes → likely to continue"
```

**Why it's useful:**
- Gives context
- Model sees recent trajectory
- Helps predict continuation/reversal
- Essential for time series prediction

---

## 🎯 **How Features Work Together**

The LSTM uses **all features combined** to make predictions.

### **Example Scenario:**

```
Current State:
  price = 89,750
  returns_1 = 0.0002 (going up)
  sma_5 = 89,700 (price above SMA → uptrend)
  volatility_5 = 0.0003 (low volatility → stable)
  rsi_14 = 60 (normal, not overbought)
  bb_position = middle (not extreme)
  hour = 14 (peak trading time)
  momentum_5 = +50 (gaining)

Model prediction:
  → Likely to continue up slightly
  → Predicted: 89,800 (cautious +50)
  → Interval: [89,700, 89,900] (narrow, low volatility)
```

vs

```
Current State:
  price = 91,000
  returns_1 = 0.005 (big jump!)
  sma_5 = 89,800 (price way above SMA)
  volatility_5 = 0.008 (high volatility!)
  rsi_14 = 82 (OVERBOUGHT!)
  bb_position = above upper band (extreme)
  hour = 22 (night, low volume)
  momentum_5 = -20 (losing steam)

Model prediction:
  → Likely to drop (overbought + losing momentum)
  → Predicted: 90,500 (correction -500)
  → Interval: [89,800, 91,200] (wide, high volatility)
```

---

## 📊 **Feature Importance**

### **Most Important Features:**

1. **price** (baseline)
2. **sma_5, sma_20** (trend)
3. **returns_1, returns_5** (momentum)
4. **volatility_5** (uncertainty)
5. **rsi_14** (overbought/oversold)

### **Supporting Features:**

6. **ema_5, ema_20** (faster trend)
7. **momentum_5** (strength)
8. **bb_width** (volatility confirmation)
9. **hour, hour_sin/cos** (time patterns)
10. **price_lag_1, price_lag_5** (recent memory)

---

## 💡 **Summary Table**

| Feature Type | Features | What They Do | Example Use |
|--------------|----------|--------------|-------------|
| **Price** | price | Current value | Base reference |
| **Returns** | returns_1/5/10 | % change | Detect momentum |
| **SMA** | sma_5/10/20/50 | Average price | Identify trends |
| **EMA** | ema_5/20 | Weighted average | Faster trend detection |
| **Volatility** | volatility_5/20 | Price jumpiness | Risk assessment |
| **Momentum** | momentum_5/10 | Price strength | Trend strength |
| **RSI** | rsi_14 | Overbought/sold | Reversal prediction |
| **Bollinger** | bb_upper/lower/width | Price bands | Extremes detection |
| **Time** | hour, day, sin/cos | Time patterns | Daily cycles |
| **Lags** | price_lag_1/2/3/5/10 | Recent history | Context memory |

---

## 🎓 **Total: 27 Features**

```
 1. price
 2. returns_1
 3. returns_5
 4. returns_10
 5. sma_5
 6. sma_10
 7. sma_20
 8. sma_50
 9. ema_5
10. ema_20
11. volatility_5
12. volatility_20
13. momentum_5
14. momentum_10
15. rsi_14
16. bb_upper
17. bb_lower
18. bb_width
19. hour
20. day_of_week
21. hour_sin
22. hour_cos
23. price_lag_1
24. price_lag_2
25. price_lag_3
26. price_lag_5
27. price_lag_10
```

**All together → LSTM learns complex patterns → Accurate predictions!** 🚀

---

## 📚 **Want to Learn More?**

- Technical Analysis: Investopedia.com
- RSI Guide: stockcharts.com/school
- Bollinger Bands: bollingerbands.com
- LSTM Time Series: colah.github.io/posts/2015-08-Understanding-LSTMs/

**Your LSTM now understands the market like a professional trader!** 📈

