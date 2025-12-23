# Precog Miner Commands Reference

This document provides a comprehensive guide to all miner commands and configuration options.

## Table of Contents
- [1. Quick Start](#1-quick-start)
- [2. Running the Miner](#2-running-the-miner)
- [3. Training LSTM Models](#3-training-lstm-models)
- [4. Environment Configuration](#4-environment-configuration)
- [5. Miner Configuration Options](#5-miner-configuration-options)
- [6. LSTM Trainer Configuration Options](#6-lstm-trainer-configuration-options)
- [7. Comparison: base_miner vs lstm_miner](#7-comparison-base_miner-vs-lstm_miner)

---

## 1. Quick Start

### Using Base Miner (No Training Required)

```bash
# Configure .env.miner
cp .env.miner.example .env.miner
# Edit .env.miner with your wallet details

# Run miner
make miner ENV_FILE=.env.miner
```

### Using LSTM Miner (Machine Learning)

```bash
# Step 1: Train models first
source .venv/bin/activate
python -m precog.miners.lstm_trainer --run_once

# Step 2: Update .env.miner
# Set FORWARD_FUNCTION=lstm_miner

# Step 3: Run miner
make miner ENV_FILE=.env.miner
```

---

## 2. Running the Miner

### Basic Commands

```bash
# Run miner with BASE_MINER (uses CoinMetrics)
python -m precog.miners.miner \
    --wallet.name <YOUR_COLDKEY> \
    --wallet.hotkey <YOUR_HOTKEY> \
    --netuid <NETUID> \
    --forward_function base_miner

# Run miner with LSTM_MINER (uses trained LSTM models)
python -m precog.miners.miner \
    --wallet.name <YOUR_COLDKEY> \
    --wallet.hotkey <YOUR_HOTKEY> \
    --netuid <NETUID> \
    --forward_function lstm_miner
```

### Using Makefile

```bash
# Run with default settings from .env.miner
make miner ENV_FILE=.env.miner
```

### Full Example with All Options

```bash
python -m precog.miners.miner \
    --wallet.name my_coldkey \
    --wallet.hotkey my_hotkey \
    --netuid 55 \
    --subtensor.network finney \
    --forward_function lstm_miner \
    --neuron.device cuda \
    --logging.level debug \
    --print_cadence 30 \
    --lstm.model_save_path ./models/lstm/
```

---

## 3. Training LSTM Models

### Basic Training Commands

```bash
# Activate virtual environment first
source .venv/bin/activate

# Train once and exit
python -m precog.miners.lstm_trainer --run_once

# Train with scheduled retraining (daily at midnight UTC)
python -m precog.miners.lstm_trainer

# Train with custom schedule
python -m precog.miners.lstm_trainer \
    --retrain_interval_days 1 \
    --retrain_hour_utc 6
```

### Full Training Example

```bash
python -m precog.miners.lstm_trainer \
    --historical_days 60 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --hidden_size 128 \
    --num_layers 2 \
    --dropout 0.2 \
    --early_stopping_patience 10 \
    --model_save_path ./models/lstm/ \
    --device cuda \
    --run_once
```

### Quick Training (for testing)

```bash
python -m precog.miners.lstm_trainer \
    --historical_days 7 \
    --epochs 10 \
    --run_once
```

### Running Scheduled Training (Production)

For production, run the trainer continuously to retrain daily:

```bash
# Terminal 1: Start trainer (retrains daily at midnight UTC)
python -m precog.miners.lstm_trainer \
    --retrain_interval_days 1 \
    --retrain_hour_utc 0

# Terminal 2: Start miner
make miner ENV_FILE=.env.miner
```

---

## 4. Environment Configuration

### .env.miner Configuration

```bash
# Network Configuration
# Options: localnet, testnet, finney
NETWORK=finney

# Wallet Configuration
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey

# Node Configuration
MINER_NAME=miner
# This port must be open to accept incoming TCP connections.
MINER_PORT=8092

# Miner Settings
TIMEOUT=16
VPERMIT_TAO_LIMIT=2

# Forward Function
# Options: base_miner, lstm_miner
FORWARD_FUNCTION=base_miner

# Logging
# Options: info, debug, trace
LOGGING_LEVEL=debug

# Local Subtensor Configuration
# Only used if you run your own subtensor node
LOCALNET=ws://127.0.0.1:9945

# LSTM Model Configuration (only needed for lstm_miner)
# Path where LSTM models are saved
LSTM_MODEL_SAVE_PATH=./models/lstm/
```

### Switching Between Miners

To switch from `base_miner` to `lstm_miner`:

1. Train the LSTM models first:
   ```bash
   python -m precog.miners.lstm_trainer --run_once
   ```

2. Update `.env.miner`:
   ```bash
   FORWARD_FUNCTION=lstm_miner
   ```

3. Restart the miner:
   ```bash
   make miner ENV_FILE=.env.miner
   ```

---

## 5. Miner Configuration Options

### Network & Wallet

| Option | Default | Description |
|--------|---------|-------------|
| `--wallet.name` | - | Coldkey wallet name |
| `--wallet.hotkey` | - | Hotkey name |
| `--netuid` | 1 | Subnet network UID (55 for mainnet) |
| `--subtensor.network` | finney | Network: finney, test, local |
| `--subtensor.chain_endpoint` | - | Custom chain endpoint URL |

### Miner Behavior

| Option | Default | Description |
|--------|---------|-------------|
| `--forward_function` | base_miner | Which miner to use: `base_miner` or `lstm_miner` |
| `--neuron.device` | cuda | Device: cuda or cpu |
| `--print_cadence` | 12 | How often to print stats (seconds) |
| `--mock` | false | Mock mode for testing |

### LSTM Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--lstm.model_save_path` | ./models/lstm/ | Path to saved LSTM models |

### Blacklist Settings

| Option | Default | Description |
|--------|---------|-------------|
| `--blacklist.force_validator_permit` | true | Only accept requests from validators |
| `--blacklist.allow_non_registered` | false | Accept queries from non-registered entities |

### Logging

| Option | Default | Description |
|--------|---------|-------------|
| `--logging.level` | info | Log level: info, debug, trace |
| `--wandb.off` | false | Disable wandb logging |
| `--wandb.offline` | false | Run wandb in offline mode |

---

## 6. LSTM Trainer Configuration Options

### Data Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--historical_days` | 60 | Days of historical data to fetch |
| `--data_interval` | 1m | Candlestick interval: 1m, 5m, 15m, 1h |

### Model Architecture

| Option | Default | Description |
|--------|---------|-------------|
| `--sequence_length` | 60 | Input sequence length (minutes) |
| `--prediction_horizon` | 60 | Prediction horizon (60 = 1 hour ahead) |
| `--hidden_size` | 128 | LSTM hidden layer size |
| `--num_layers` | 2 | Number of LSTM layers |
| `--dropout` | 0.2 | Dropout rate for regularization |

### Training Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Maximum training epochs |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 0.001 | Learning rate |
| `--validation_split` | 0.1 | Validation data ratio (10%) |
| `--early_stopping_patience` | 10 | Stop if no improvement for N epochs |

### Schedule Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--retrain_interval_days` | 1 | Days between retraining |
| `--retrain_hour_utc` | 0 | Hour (UTC) to run training |
| `--run_once` | false | Train once and exit (no scheduling) |

### Model Management

| Option | Default | Description |
|--------|---------|-------------|
| `--model_save_path` | ./models/lstm/ | Where to save trained models |
| `--keep_old_models` | 5 | Number of archived models to keep |

### Runtime Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | cuda | Training device: cuda or cpu |
| `--random_seed` | 42 | Random seed for reproducibility |
| `--quiet` | false | Reduce logging output |

---

## 7. Comparison: base_miner vs lstm_miner

| Feature | base_miner | lstm_miner |
|---------|------------|------------|
| **Data Source** | CoinMetrics API | Binance API |
| **Prediction Method** | Latest price (no ML) | LSTM neural network |
| **Interval Method** | Volatility heuristic | Learned from historical data |
| **Requires Training** | ❌ No | ✅ Yes |
| **Requires API Key** | CoinMetrics (optional) | ❌ No |
| **GPU Recommended** | ❌ No | ✅ Yes (for training) |
| **Setup Complexity** | Simple | Moderate |

### When to Use Each

**Use `base_miner` when:**
- You want a simple, quick setup
- You don't have GPU access
- You're testing or getting started

**Use `lstm_miner` when:**
- You want machine learning-based predictions
- You have GPU access for training
- You want learned interval predictions (not heuristics)

---

## Compute Requirements

| Component | Miner (base) | Miner (LSTM) |
|-----------|--------------|--------------|
| RAM | 8gb | 16gb |
| vCPUs | 2 | 4 |
| GPU | ❌ Not required | ✅ Recommended for training |
| Storage | 10gb | 20gb (for models) |

---

## Troubleshooting

### LSTM Model Not Found

```
[WARN] No trained model found for BTCUSDT
```

**Solution:** Train models first:
```bash
python -m precog.miners.lstm_trainer --run_once
```

### CUDA Out of Memory

**Solution:** Reduce batch size or use CPU:
```bash
python -m precog.miners.lstm_trainer --batch_size 16 --device cpu --run_once
```

### Early Stopping Triggered Too Soon

**Solution:** Increase patience:
```bash
python -m precog.miners.lstm_trainer --early_stopping_patience 20 --run_once
```

### Insufficient Data for Prediction

```
[WARN] Insufficient data for BTCUSDT prediction
```

**Solution:** Check Binance API connectivity and ensure the model's `sequence_length` data points are available.

---

## File Structure

```
precog/
├── miners/
│   ├── miner.py              # Main miner entry point
│   ├── base_miner.py         # CoinMetrics-based miner
│   ├── lstm_miner.py         # LSTM-based miner
│   ├── lstm_trainer.py       # LSTM model trainer
│   └── models/
│       ├── lstm_model.py     # LSTM neural network
│       ├── lstm_config.py    # Configuration classes
│       ├── data_preprocessor.py  # Data preprocessing
│       ├── trainer.py        # Training pipeline
│       └── model_manager.py  # Model versioning
├── models/
│   └── lstm/
│       ├── current/          # Current trained models
│       │   ├── BTCUSDT/
│       │   ├── ETHUSDT/
│       │   └── TAOUSDT/
│       ├── archive/          # Archived old models
│       └── checkpoints/      # Training checkpoints
└── .env.miner                # Miner configuration
```

