# 🔧 Fix Import Errors - Complete Guide

## ❌ The Error You're Seeing

```
ModuleNotFoundError: No module named 'precog'
```

This happens because Python can't find the modules. Here are the solutions:

---

## ✅ Solution 1: Install the Package (RECOMMENDED)

This is the cleanest solution - install the precog package in development mode:

```bash
cd /home/fang/develop/precog

# Install with pip in editable mode
pip install -e .
```

**This will:**
- Install all dependencies (numpy, pandas, torch, etc.)
- Make `precog` module importable
- Allow you to edit code without reinstalling

**Then run:**
```bash
# Now this will work!
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

---

## ✅ Solution 2: Use PYTHONPATH (Quick Fix)

If you don't want to install, set PYTHONPATH:

```bash
cd /home/fang/develop/precog

# Set PYTHONPATH and run
PYTHONPATH=/home/fang/develop/precog python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

**Or make it persistent:**
```bash
export PYTHONPATH=/home/fang/develop/precog:$PYTHONPATH
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

---

## ✅ Solution 3: Install Dependencies First

The scripts need these packages:

```bash
# Install PyTorch
pip install torch torchvision

# Install other dependencies (already in requirements)
pip install numpy pandas bittensor coinmetrics-api-client
```

---

## 🚀 Complete Setup (Do This Once)

```bash
cd /home/fang/develop/precog

# Step 1: Install the package in development mode
pip install -e .

# Step 2: Verify installation
python3 -c "from precog.miners.models.feature_engineer import FeatureEngineer; print('✅ Success!')"

# Step 3: Install PyTorch
pip install torch torchvision

# Step 4: Run training
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

---

## 📋 What I've Already Fixed

I've updated these files to help with imports:
- ✅ `precog/__init__.py` - Now handles missing metadata
- ✅ `precog/miners/scripts/train_lstm.py` - Adds path automatically
- ✅ `precog/miners/scripts/test_lstm.py` - Adds path automatically
- ✅ `test_local.py` - Adds path automatically

---

## 🧪 Test If It's Working

```bash
# Test 1: Can Python find precog?
python3 -c "import sys; sys.path.insert(0, '/home/fang/develop/precog'); import precog; print(f'✅ Precog version: {precog.__version__}')"

# Test 2: Can we import our modules?
python3 -c "import sys; sys.path.insert(0, '/home/fang/develop/precog'); from precog.miners.models.feature_engineer import FeatureEngineer; print('✅ Import successful!')"

# Test 3: Is PyTorch installed?
python3 -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
```

---

## 💡 Recommended: Install the Package

The cleanest way is to install in development mode:

```bash
cd /home/fang/develop/precog
pip install -e .
```

This command:
1. Installs all dependencies from pyproject.toml
2. Makes `precog` importable everywhere
3. Allows you to edit code without reinstalling
4. Is how professional Python projects work

**After this, everything will work!**

---

## 🎯 Quick Commands (Copy & Paste)

```bash
# Full setup (run once)
cd /home/fang/develop/precog
pip install -e .
pip install torch torchvision

# Train model
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10

# Test
python3 test_local.py

# Deploy
make miner_lstm ENV_FILE=.env.miner
```

---

## ❓ Still Having Issues?

### Issue: "pip command not found"
```bash
sudo apt install python3-pip
```

### Issue: "Permission denied"
```bash
pip install --user -e .
```

### Issue: "Package conflicts"
```bash
pip install --force-reinstall -e .
```

---

## ✅ Bottom Line

**Just run this:**
```bash
cd /home/fang/develop/precog
pip install -e .
pip install torch torchvision
```

Then everything will work! 🎉

