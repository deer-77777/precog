# 🚀 Quick Reference Guide - Model Updating

## 📋 Three Ways to Update Your Model

### **1️⃣ Manual Retraining** (Simplest)

Run whenever you want to update the model:

```bash
cd /home/fang/develop/precog

# Train all assets (BTC + ETH + TAO)
./retrain_all.sh 30 50

# Or train individually
./retrain_schedule.sh btc 30 50
./retrain_schedule.sh eth 30 50
./retrain_schedule.sh tao_bittensor 30 50
```

**What it does:**
- Fetches 30 days of data from Binance
- Trains for 50 epochs
- Validates new model vs current model
- Deploys if better
- Archives old model

**When to use:**
- You want full control
- Testing different parameters
- First time training

---

### **2️⃣ Automated Retraining** (Recommended) ⭐

Set up once, forget about it:

```bash
cd /home/fang/develop/precog
./setup_cron.sh
```

**Choose assets:**
- **BTC only**: Single asset
- **BTC + ETH**: Two major assets
- **BTC + ETH + TAO**: All assets (recommended)

**Choose your schedule:**
- **Conservative**: Every 7 days (staggered by 2 hours per asset)
- **Moderate**: Every 3 days (staggered by 1 hour per asset) (recommended)
- **Aggressive**: Every 2 days + daily fine-tuning (staggered)

**What it does:**
- Runs automatically at scheduled times (e.g., 2am, 3am, 4am for each asset)
- Retrains each model with latest data
- Tests and deploys if better
- Keeps archives for each asset
- Logs everything
- Restarts miner automatically

---

### **3️⃣ Incremental Fine-Tuning** (Advanced)

Quick daily updates without full retraining:

```bash
cd /home/fang/develop/precog

# Fine-tune all assets
./incremental_train_all.sh 7 10

# Or fine-tune individually
python3 incremental_train.py --asset btc --days 7 --epochs 10
python3 incremental_train.py --asset eth --days 7 --epochs 10
python3 incremental_train.py --asset tao_bittensor --days 7 --epochs 10
```

**What it does:**
- Loads existing model(s)
- Fine-tunes with last 7 days
- Uses lower learning rate
- Faster than full retraining (5-10 minutes per asset)

**When to use:**
- Daily updates between full retrains
- Quick model refresh
- Market conditions changing rapidly

---

## 🛠️ Common Commands

### **Check Model Status**

```bash
# View model metadata for all assets
cat precog/miners/models/trained_weights/btc_lstm_metadata.json
cat precog/miners/models/trained_weights/eth_lstm_metadata.json
cat precog/miners/models/trained_weights/tao_bittensor_lstm_metadata.json

# Or view all at once
for asset in btc eth tao_bittensor; do
    echo "=== $asset ==="
    cat precog/miners/models/trained_weights/${asset}_lstm_metadata.json 2>/dev/null || echo "No metadata"
    echo ""
done

# View training logs
tail -f logs/auto_retrain_$(date +%Y%m%d).log

# Check cron logs
tail -f logs/cron.log
```

### **Test Current Model**

```bash
# Test locally (tests all forward functions including LSTM)
python3 test_local.py

# Or test specific models
python3 precog/miners/scripts/test_lstm.py --asset btc
python3 precog/miners/scripts/test_lstm.py --asset eth
python3 precog/miners/scripts/test_lstm.py --asset tao_bittensor
```

### **Manual Training Commands**

```bash
# Train all assets at once (easiest)
./retrain_all.sh 30 50

# Or train individually from scratch
python3 precog/miners/scripts/train_lstm.py \
    --assets btc \
    --days 30 \
    --epochs 50 \
    --use-binance

python3 precog/miners/scripts/train_lstm.py \
    --assets eth \
    --days 30 \
    --epochs 50 \
    --use-binance

python3 precog/miners/scripts/train_lstm.py \
    --assets tao_bittensor \
    --days 30 \
    --epochs 50 \
    --use-binance

# Longer training (60 days, 100 epochs) for all assets
./retrain_all.sh 60 100

# Train with loop (if retrain_all.sh not available)
for asset in btc eth tao_bittensor; do
    python3 precog/miners/scripts/train_lstm.py \
        --assets $asset \
        --days 30 \
        --epochs 50 \
        --use-binance
done
```

### **View Model Archives**

```bash
# List all archived models (all assets)
ls -lh precog/miners/models/model_archives/

# View by asset
ls -lh precog/miners/models/model_archives/btc_*
ls -lh precog/miners/models/model_archives/eth_*
ls -lh precog/miners/models/model_archives/tao_bittensor_*

# View model with timestamp
# btc_lstm_20241215_020000.pth = trained on Dec 15 at 2am
```

### **Restore Old Model**

```bash
# If new model is bad, restore from archive (example for BTC)
cp precog/miners/models/model_archives/btc_lstm_20241210_020000.pth \
   precog/miners/models/trained_weights/btc_lstm.pth

# Restore for ETH
cp precog/miners/models/model_archives/eth_lstm_20241210_020000.pth \
   precog/miners/models/trained_weights/eth_lstm.pth

# Restore for TAO
cp precog/miners/models/model_archives/tao_bittensor_lstm_20241210_020000.pth \
   precog/miners/models/trained_weights/tao_bittensor_lstm.pth

# Restart miner to use restored models
pm2 restart miner_lstm
```

---

## 📊 Monitoring Model Health

### **Check MAPE (Lower is Better)**

Target: **< 3% MAPE**

```
1-2%   = Excellent ⭐⭐⭐
2-3%   = Good ✅
3-5%   = Acceptable ⚠️
5-10%  = Poor ❌ (retrain soon)
> 10%  = Terrible 💀 (retrain immediately!)
```

### **When to Retrain**

Retrain if:
- ✅ MAPE > 3% for several hours
- ✅ Last training was > 7 days ago
- ✅ Market volatility doubled
- ✅ Major news/events happened
- ✅ Predictions consistently off by > $200

### **Signs Model is Healthy**

- ✅ MAPE stable around 1-2%
- ✅ Prediction difference < $100
- ✅ Interval width reasonable ($200-$400)
- ✅ No sudden spikes in error

---

## 🎯 Recommended Schedule

### **For BTC/ETH (High Activity)**

```
Every 3 days:  Full retrain (30 days, 50 epochs)
Daily:         Check MAPE
Weekly:        Review logs and archives
```

### **For TAO (Lower Activity)**

```
Every 7 days:  Full retrain (30 days, 50 epochs)
Weekly:        Check MAPE
Monthly:       Review logs
```

---

## 🔧 Troubleshooting

### **Problem: Cron job not running**

```bash
# Check if cron is running
systemctl status cron

# Check crontab
crontab -l

# View cron logs
grep CRON /var/log/syslog

# Test script manually
./retrain_schedule.sh btc 30 50
```

### **Problem: New model worse than old**

The script automatically keeps the old model. Check logs:

```bash
tail -50 logs/auto_retrain_$(date +%Y%m%d).log
```

If this happens repeatedly:
- Try more training data (60-90 days)
- Try more epochs (100)
- Check if market conditions changed dramatically

### **Problem: Training fails**

```bash
# Check Python packages
pip list | grep torch
pip list | grep pandas

# Reinstall if needed
pip install -e .

# Check Binance API
python3 test_binance.py

# Check available disk space
df -h
```

### **Problem: Model file corrupted**

```bash
# Restore from archive
ls -lt precog/miners/models/model_archives/
cp precog/miners/models/model_archives/btc_lstm_YYYYMMDD_HHMMSS.pth \
   precog/miners/models/trained_weights/btc_lstm.pth

# Or retrain from scratch
./retrain_schedule.sh btc 30 50
```

---

## 📁 File Locations

```
precog/
├── auto_retrain.py              # Automated retraining script
├── retrain_schedule.sh          # Cron wrapper script
├── incremental_train.py         # Fine-tuning script
├── setup_cron.sh                # Setup automation
│
├── logs/
│   ├── auto_retrain_YYYYMMDD.log  # Daily training logs
│   └── cron.log                    # Cron execution logs
│
└── precog/miners/models/
    ├── trained_weights/
    │   ├── btc_lstm.pth           # Current model
    │   ├── eth_lstm.pth           # Current model
    │   ├── tao_bittensor_lstm.pth # Current model
    │   └── btc_lstm_metadata.json # Model info
    │
    └── model_archives/
        ├── btc_lstm_20241215_020000.pth  # Backup 1
        ├── btc_lstm_20241212_020000.pth  # Backup 2
        └── ...                            # (keeps last 5)
```

---

## 🎓 Summary

| Method | Frequency | Setup | Best For |
|--------|-----------|-------|----------|
| **Manual** | When needed | None | Testing, learning |
| **Automated** ⭐ | 3-7 days | One-time | Production |
| **Incremental** | Daily | Cron | Advanced users |

**Recommended workflow:**
1. ✅ Set up automated retraining (every 3 days)
2. ✅ Monitor MAPE weekly
3. ✅ Review logs monthly
4. ✅ Manual retrain only if issues

---

## 💡 Pro Tips

1. **Always keep backups**: Archives are your safety net
2. **Monitor MAPE**: Track it in a spreadsheet or dashboard
3. **Test before deploy**: `test_local.py` is your friend
4. **Log everything**: Logs help debug issues later
5. **Start conservative**: Weekly retraining, then speed up if needed

---

## 📞 Quick Help

```bash
# Setup automation (interactive)
./setup_cron.sh

# Manual retrain all assets
./retrain_all.sh 30 50

# Manual retrain single asset
./retrain_schedule.sh btc 30 50

# Quick fine-tune all assets
./incremental_train_all.sh 7 10

# Quick fine-tune single asset
python3 incremental_train.py --asset btc

# Test current models
python3 test_local.py

# View logs
tail -f logs/auto_retrain_$(date +%Y%m%d).log

# Check model info (all assets)
for asset in btc eth tao_bittensor; do
    echo "=== $asset ==="
    cat precog/miners/models/trained_weights/${asset}_lstm_metadata.json 2>/dev/null
done

# Restart miner
pm2 restart miner_lstm
```

**Your model stays fresh, your predictions stay accurate!** 🚀

