# 🧪 Local Testing Guide - LSTM Miner

Test your LSTM miner locally before deploying to production.

## 📋 Why Test Locally?

✅ **Verify model works** - Make sure predictions are reasonable  
✅ **Check performance** - Ensure fast inference (<1s)  
✅ **Debug issues** - Find problems before production  
✅ **Validate accuracy** - See if model is better than baseline  
✅ **No risk** - Test without affecting your miner ranking  

---

## 🚀 Quick Test (5 minutes)

### **Step 1: Install Dependencies**

```bash
cd /home/fang/develop/precog

# Install PyTorch
pip install torch torchvision

# Verify
python3 -c "import torch; print('PyTorch:', torch.__version__)"
```

### **Step 2: Train a Quick Model**

```bash
# Quick training (7 days data, 10 epochs, ~5 minutes)
python3 precog/miners/scripts/train_lstm.py \
    --assets btc \
    --days 7 \
    --epochs 10
```

**Expected output:**
```
============================================================
Training LSTM for BTC
============================================================

Fetching 7 days of btc data...
Fetched 10080 data points

Preparing training data...
Using 35 features
Created 9960 training sequences

Train size: 7968, Val size: 1992

============================================================
Starting training...
============================================================

Epoch   1/10 | Train Loss: 12345.67 | Val Loss: 13456.78 | Val MAPE: 2.3456%
  ✓ Saved best model to precog/miners/models/trained_weights/btc_lstm.pth
...
Epoch  10/10 | Train Loss: 10123.45 | Val Loss: 11234.56 | Val MAPE: 1.8765%

============================================================
✅ Training complete!
Best validation loss: 11234.56
Best validation MAPE: 1.8765%
============================================================
```

### **Step 3: Run Local Tests**

```bash
# Run comprehensive test suite
python3 test_local.py
```

**Expected output:**
```
============================================================
LSTM MINER LOCAL TESTING
============================================================

This will test your LSTM implementation locally
before deploying as a miner.

============================================================
TEST 1: Data Fetching
============================================================
Fetching BTC data from 14:30 to 15:30...
✅ SUCCESS: Fetched 60 data points
   Columns: ['asset', 'time', 'ReferenceRateUSD']

============================================================
TEST 2: Feature Engineering
============================================================
Fetching data for feature engineering...
Creating features from 120 data points...
✅ SUCCESS: Created 35 features
   Features: price, returns_1, returns_5, sma_5, sma_10, ...

============================================================
TEST 3: Model Loading
============================================================
✅ SUCCESS: Loaded 1 models
   - btc: epoch 9, MAPE 1.8765%

============================================================
TEST 4: LSTM Prediction
============================================================

   Testing BTC:
   --------------------------------------------------------
   Current price: $95,234.50
   Predicted:     $95,467.80
   Difference:    +$233.30 (+0.24%)
   Interval:      [$94,562.15, $96,373.45]
   Width:         $1,811.30
   ✅ Prediction looks reasonable

============================================================
TEST 5: Complete Forward Function
============================================================
Simulating validator request at 2025-01-15 15:30:00
Calling forward function...

✅ SUCCESS: Forward function executed
   Predictions: {'btc': 95467.8}
   Intervals:   {'btc': [94562.15, 96373.45]}

   btc:
      Prediction: $95,467.80
      Interval:   [$94,562.15, $96,373.45]

============================================================
TEST 6: Performance Metrics
============================================================
✅ Inference time (10 runs):
   Average: 127.45ms
   Min:     115.23ms
   Max:     156.78ms
   ✅ Good performance (<1s avg)

============================================================
TEST SUMMARY
============================================================
✅ PASS  Data Fetching
✅ PASS  Feature Engineering
✅ PASS  Model Loading
✅ PASS  LSTM Prediction
✅ PASS  Forward Function
✅ PASS  Performance

6/6 tests passed

🎉 All tests passed! Your LSTM miner is ready to deploy.

Next steps:
1. Make sure you're satisfied with the model accuracy
2. Consider training on more data (30+ days)
3. Deploy: make miner_lstm ENV_FILE=.env.miner
```

---

## 📊 Understanding Test Results

### **Test 1: Data Fetching**
- ✅ **PASS**: Can fetch data from CoinMetrics API
- ❌ **FAIL**: API connection issue, check internet

### **Test 2: Feature Engineering**
- ✅ **PASS**: Creates 35 features successfully
- ❌ **FAIL**: Code error, check feature_engineer.py

### **Test 3: Model Loading**
- ✅ **PASS**: Trained models loaded
- ⚠️ **FAIL**: No models found → Train first!

### **Test 4: LSTM Prediction**
- ✅ **PASS**: Predictions are reasonable (<10% diff)
- ⚠️ **WARNING**: Large difference (>10%) → May need retraining
- ❌ **FAIL**: Prediction error → Check logs

### **Test 5: Forward Function**
- ✅ **PASS**: Complete pipeline works
- ❌ **FAIL**: Integration error → Debug

### **Test 6: Performance**
- ✅ **PASS**: Fast inference (<1s average)
- ⚠️ **WARNING**: Slow (>1s) → May timeout with validators

---

## 🔍 Detailed Testing

### **Individual Component Tests**

#### **Test Specific Asset:**

```python
# test_btc.py
from datetime import datetime, timedelta
from precog.miners.lstm_miner import LSTMPredictor
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str

predictor = LSTMPredictor()
cm = CMData()

end_time = datetime.now()
start_time = end_time - timedelta(hours=4)

data = cm.get_CM_ReferenceRate(
    assets=['btc'],
    start=to_str(start_time),
    end=to_str(end_time),
    frequency='1m'
)

pred, interval = predictor.predict('btc', data)
print(f"Prediction: ${pred:,.2f}")
print(f"Interval: [${interval[0]:,.2f}, ${interval[1]:,.2f}]")
```

#### **Compare with Current Price:**

```python
current_price = data['ReferenceRateUSD'].iloc[-1]
predicted_price = pred

diff = predicted_price - current_price
diff_pct = (diff / current_price) * 100

print(f"\nCurrent:   ${current_price:,.2f}")
print(f"Predicted: ${predicted_price:,.2f}")
print(f"Diff:      ${diff:+,.2f} ({diff_pct:+.2f}%)")

# Good prediction: diff < 5%
# Warning: diff > 5% but < 10%
# Bad: diff > 10%
```

#### **Test Multiple Times:**

```bash
# Run test every 5 minutes for 1 hour
for i in {1..12}; do
    echo "=== Test $i/12 ==="
    python3 test_local.py
    sleep 300  # 5 minutes
done
```

---

## 🎯 Evaluating Model Quality

### **Good Model Indicators:**

✅ **MAPE < 3%** - Validation error is low  
✅ **Predictions within ±5%** - Not too far from current  
✅ **Interval captures price** - Current price inside interval  
✅ **Fast inference** - <500ms average  
✅ **Consistent** - Similar accuracy across assets  

### **Warning Signs:**

⚠️ **MAPE > 5%** - Model may be underfitting  
⚠️ **Large prediction jumps** - Price changes >10%  
⚠️ **Wide intervals** - Uncertainty too high  
⚠️ **Slow inference** - >1s (may timeout)  
⚠️ **Volatile predictions** - Jumps around too much  

### **Red Flags:**

❌ **MAPE > 10%** - Model is bad  
❌ **Predictions always high/low** - Bias issue  
❌ **Interval doesn't capture price** - Poor calibration  
❌ **Crashes or errors** - Implementation bug  

---

## 🔧 Troubleshooting

### **Problem: "No models loaded"**

**Solution:**
```bash
# Train models first
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

### **Problem: "Predictions seem wrong"**

**Check:**
1. MAPE from training - should be <3%
2. Amount of training data - more is better
3. Features being calculated correctly

**Fix:**
```bash
# Train on more data
python3 precog/miners/scripts/train_lstm.py --days 30 --epochs 50
```

### **Problem: "Slow performance"**

**Check:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Fix:**
- Use GPU if available
- Reduce model size
- Optimize feature calculation

### **Problem: "API errors"**

**Check:**
- Internet connection
- CoinMetrics API status
- Rate limits

---

## 📈 Benchmarking Against Baseline

Compare your LSTM vs base miner:

```python
# Compare predictions
from precog.miners.base_miner import forward as base_forward
from precog.miners.lstm_miner import forward as lstm_forward
from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str
from datetime import datetime
import asyncio

# Setup
class MockDendrite:
    hotkey = "test"

synapse = Challenge(
    timestamp=to_str(datetime.now()),
    assets=['btc']
)
synapse.dendrite = MockDendrite()
cm = CMData()

# Base miner prediction
base_result = asyncio.run(base_forward(synapse, cm))
print(f"Base:  ${base_result.predictions['btc']:,.2f}")

# LSTM prediction
lstm_result = asyncio.run(lstm_forward(synapse, cm))
print(f"LSTM:  ${lstm_result.predictions['btc']:,.2f}")

# Wait 1 hour and compare accuracy
# (You'd need to actually wait and check real prices)
```

---

## ✅ Checklist Before Deployment

Before deploying as a miner, verify:

- [ ] All 6 tests pass in `test_local.py`
- [ ] MAPE < 3% on validation set
- [ ] Predictions are within ±5% of current price
- [ ] Inference time < 1 second
- [ ] Intervals capture current price
- [ ] Tested on all assets (BTC, ETH, TAO)
- [ ] No errors in logs
- [ ] Compared with baseline miner
- [ ] Satisfied with model quality

---

## 🚀 Ready to Deploy?

If all tests pass:

```bash
# Deploy as miner
make miner_lstm ENV_FILE=.env.miner

# Monitor logs
pm2 logs miner

# Check status
pm2 status
```

---

## 📊 Production Monitoring

After deployment, monitor:

```bash
# View predictions in logs
pm2 logs miner | grep "LSTM Prediction"

# Check for errors
pm2 logs miner | grep "ERROR"

# Monitor response times
pm2 logs miner | grep "forward call took"
```

**What to look for:**
- Predictions being made every 5 minutes
- No timeout errors
- Reasonable prediction values
- Fast response times (<1s)

---

## 💡 Tips

1. **Start small**: Test with 7 days first, then scale to 30 days
2. **Monitor closely**: Watch logs for first hour after deployment
3. **Compare**: Track your rank vs baseline miner
4. **Iterate**: Retrain weekly with new data
5. **Document**: Keep notes on what works

---

**Good luck with your testing!** 🎉

Your LSTM miner is ready when all tests pass and you're satisfied with the accuracy!

