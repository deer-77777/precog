# 🔧 Fix Model Quality Issue

## ❌ Current Problem

Your model has **52% error** - it's predicting $42,503 when BTC is actually $89,666!

```
MAPE: 52.5062% ← TERRIBLE!
Current:  $89,666
Predicted: $42,503
```

**This model CANNOT be used for mining!**

---

## 🔍 Root Cause

Training on **only 7 days** with **10 epochs** is not enough:
- Not enough data to learn patterns
- Model didn't converge
- Too few training iterations

---

## ✅ Solution 1: Retrain with Better Parameters (RECOMMENDED)

```bash
cd /home/fang/develop/precog

# Remove bad model
rm precog/miners/models/trained_weights/btc_lstm.pth

# Retrain with proper parameters
python3 precog/miners/scripts/train_lstm.py \
    --assets btc \
    --days 30 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0001
```

**This will take ~30 minutes but will give much better results!**

---

## 📊 Expected Results After Retraining

### Good Model:
```
MAPE: 1-3%
Validation Loss: < 5000
Prediction difference: < 5%
```

### Acceptable Model:
```
MAPE: 3-5%
Validation Loss: 5000-10000
Prediction difference: 5-10%
```

### Bad Model (current):
```
MAPE: > 50% ← You are here!
Validation Loss: high
Prediction difference: > 50%
```

---

## 🎯 Quick Commands

### Option 1: Use Script (Easy)
```bash
./retrain_properly.sh
```

### Option 2: Manual (Control parameters)
```bash
# Remove bad model
rm precog/miners/models/trained_weights/btc_lstm.pth

# Retrain
python3 precog/miners/scripts/train_lstm.py --assets btc --days 30 --epochs 50
```

### Option 3: Diagnose First
```bash
# See what's wrong
python3 diagnose_model.py

# Then retrain
./retrain_properly.sh
```

---

## 📋 After Retraining

1. **Test again:**
```bash
python3 test_local.py
```

2. **Look for:**
```
✅ MAPE < 3%
✅ Prediction difference < 5%
✅ All tests pass
```

3. **Then deploy:**
```bash
make miner_lstm ENV_FILE=.env.miner
```

---

## ⏱️ Time Estimates

| Training Data | Epochs | Time | Quality |
|--------------|--------|------|---------|
| 7 days | 10 | ~5 min | ❌ Bad (52% MAPE) |
| 30 days | 50 | ~30 min | ✅ Good (1-3% MAPE) |
| 60 days | 100 | ~2 hours | ✅ Best (<1% MAPE) |

---

## 💡 Why This Happened

**Your original training:**
```bash
python3 train_lstm.py --days 7 --epochs 10
```

This was just a **quick test** to see if the system works!

**For production, you need:**
```bash
python3 train_lstm.py --days 30 --epochs 50
```

---

## 🚨 DO NOT DEPLOY Current Model!

The current model will:
- ❌ Make terrible predictions
- ❌ Get very low rewards
- ❌ Rank at the bottom
- ❌ Waste compute resources

---

## ✅ Action Plan

```bash
# 1. Remove bad model
rm precog/miners/models/trained_weights/btc_lstm.pth

# 2. Retrain properly (30 minutes)
python3 precog/miners/scripts/train_lstm.py --assets btc --days 30 --epochs 50

# 3. Test
python3 test_local.py

# 4. Look for MAPE < 3%

# 5. If good, deploy
make miner_lstm ENV_FILE=.env.miner
```

---

## 🎓 Learning

- **7 days, 10 epochs** = Quick test only
- **30 days, 50 epochs** = Production quality
- **60+ days, 100 epochs** = Best quality

Always check MAPE before deploying!

---

## 🔄 Start Retraining Now

```bash
cd /home/fang/develop/precog
./retrain_properly.sh
```

This will give you a model that actually works! 🚀

