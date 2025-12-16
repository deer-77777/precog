# LSTM Models for Price Prediction

This directory contains the LSTM-based price prediction system for the Precog miner.

## 📁 Directory Structure

```
models/
├── __init__.py              # Package initialization
├── feature_engineer.py      # Feature engineering (technical indicators)
├── lstm_model.py            # LSTM architecture
├── trained_weights/         # Saved model checkpoints
│   ├── btc_lstm.pth
│   ├── eth_lstm.pth
│   └── tao_lstm.pth
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install PyTorch

```bash
# CPU version
pip install torch torchvision

# Or GPU version (if you have CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Train Models

Train models for all assets (recommended: 30 days of data):

```bash
cd /home/fang/develop/precog

# Train all assets (will take time!)
python3 precog/miners/scripts/train_lstm.py --days 30 --epochs 50

# Or train single asset for testing
python3 precog/miners/scripts/train_lstm.py --assets btc --days 7 --epochs 10
```

**Training options:**
- `--assets`: Assets to train (btc, eth, tao_bittensor)
- `--days`: Days of historical data (more = better, but slower)
- `--epochs`: Training epochs (50 is good default)
- `--batch_size`: Batch size (32 is default)
- `--lookback`: Minutes to look back (60 = 1 hour)
- `--forecast_horizon`: Minutes to forecast (60 = 1 hour)

### 3. Test Predictions

```bash
# Test your trained models
python3 precog/miners/scripts/test_lstm.py
```

### 4. Deploy Miner

Add to your `.env.miner`:
```bash
FORWARD_FUNCTION=lstm_miner
```

Update `Makefile`:
```makefile
miner_lstm:
	pm2 start --name $(MINER_NAME) python3 -- precog/miners/miner.py \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(MINER_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(MINER_PORT) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL) \
		--timeout $(TIMEOUT) \
		--vpermit_tao_limit $(VPERMIT_TAO_LIMIT) \
		--forward_function lstm_miner
```

Run:
```bash
make miner_lstm ENV_FILE=.env.miner
```

## 🧠 How It Works

### Feature Engineering

The `FeatureEngineer` class creates 30+ technical features:

- **Price features**: returns, log returns
- **Moving averages**: SMA (5, 10, 20, 50), EMA (5, 20)
- **Volatility**: rolling std (5, 20 periods)
- **Momentum**: 5, 10 period momentum
- **RSI**: Relative Strength Index (14 periods)
- **Bollinger Bands**: upper, lower, width
- **Time features**: hour, day of week (cyclical encoding)
- **Lag features**: previous prices (1, 2, 3, 5, 10 periods)

### LSTM Architecture

```
Input: (batch_size, 60 timesteps, 30+ features)
   ↓
LSTM Layer 1 (128 hidden units)
   ↓
LSTM Layer 2 (128 hidden units)
   ↓
Fully Connected (64 units)
   ↓
ReLU + Dropout
   ↓
Output Layer (1 unit) → Predicted Price
```

### Training Process

1. Fetch historical data (1-minute frequency)
2. Create features for each timestep
3. Prepare sequences (60 minutes lookback)
4. Train with MSE loss
5. Validate and save best model
6. Early stopping if no improvement

### Prediction Process

1. Fetch last 4 hours of data
2. Create features
3. Extract last 60-minute sequence
4. Pass through LSTM
5. Get price prediction
6. Calculate interval using volatility

## 📊 Model Performance

Expected performance metrics:

- **MAPE**: 1-3% on validation set
- **Response time**: < 1 second per prediction
- **Improvement over baseline**: 10-30%

## 🔧 Customization

### Modify Features

Edit `feature_engineer.py`:

```python
def create_features(self, df):
    # Add your custom features here
    df['my_custom_feature'] = calculate_custom_indicator(df['price'])
    return df
```

### Modify Architecture

Edit `lstm_model.py`:

```python
class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3):  # Larger network
        # ... modify architecture
```

### Hyperparameter Tuning

```bash
# Try different configurations
python3 precog/miners/scripts/train_lstm.py \
    --assets btc \
    --days 60 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0001
```

## 📈 Tips for Better Performance

1. **More data**: Train on 30-90 days for better patterns
2. **More features**: Add volume, order book data
3. **Ensemble**: Combine LSTM with other models
4. **Regular retraining**: Retrain weekly/monthly
5. **Asset-specific tuning**: Different assets need different configs
6. **Validation**: Always validate on unseen data

## 🐛 Troubleshooting

**Issue: "No models loaded"**
- Solution: Train models first with `train_lstm.py`

**Issue: "Not enough data"**
- Solution: Ensure 60+ minutes of data available

**Issue: "Prediction error"**
- Solution: Check logs, ensure features match training

**Issue: "Poor accuracy"**
- Solution: Train longer, more data, tune hyperparameters

## 📚 References

- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Time Series Forecasting: https://arxiv.org/abs/1803.01271
- Technical Indicators: https://technical-analysis-library-in-python.readthedocs.io/

