# 🧠 Building an LSTM Miner for Precog

## 📊 Understanding the Data

### **What data is available from CoinMetrics API?**

The project uses the CoinMetrics API to fetch historical cryptocurrency prices. You have TWO data sources:

#### **1. Reference Rate Data (Simple price)**
```python
from precog.utils.cm_data import CMData

cm = CMData()
data = cm.get_CM_ReferenceRate(
    assets=['btc', 'eth', 'tao_bittensor'],
    start='2025-01-01T00:00:00.000000Z',
    end='2025-01-01T06:00:00.000000Z',
    frequency='1m'  # '1s', '1m', '5m', '1h', '1d'
)

# Returns DataFrame with columns:
# - asset: str           ('btc', 'eth', 'tao_bittensor')
# - time: datetime       (2025-01-01 00:00:00+00:00)
# - ReferenceRateUSD: float  (95234.50)
```

**Example data:**
```
asset | time                      | ReferenceRateUSD
------|---------------------------|------------------
btc   | 2025-01-01 00:00:00+00:00 | 95234.50
btc   | 2025-01-01 00:01:00+00:00 | 95245.20
btc   | 2025-01-01 00:02:00+00:00 | 95198.75
btc   | 2025-01-01 00:03:00+00:00 | 95267.30
...
```

#### **2. Candle Data (OHLC - More features!)**
```python
data = cm.get_pair_candles(
    pairs=['btc-usd', 'eth-usd'],
    start='2025-01-01T00:00:00.000000Z',
    end='2025-01-01T06:00:00.000000Z',
    frequency='5m'
)

# Returns DataFrame with columns:
# - pair: str           ('btc-usd')
# - time: datetime      (2025-01-01 00:00:00+00:00)
# - price_open: float   (95234.50)
# - price_close: float  (95245.20)
# - price_high: float   (95280.00)
# - price_low: float    (95200.00)
```

**Example data:**
```
pair    | time                      | open      | close     | high      | low
--------|---------------------------|-----------|-----------|-----------|----------
btc-usd | 2025-01-01 00:00:00+00:00 | 95234.50  | 95245.20  | 95280.00  | 95200.00
btc-usd | 2025-01-01 00:05:00+00:00 | 95245.20  | 95198.75  | 95250.00  | 95180.00
btc-usd | 2025-01-01 00:10:00+00:00 | 95198.75  | 95267.30  | 95290.00  | 95195.00
...
```

### **What features can you use for LSTM?**

**From the API:**
- ✅ Price (ReferenceRateUSD)
- ✅ Open, High, Low, Close (from candles)
- ✅ Time features (hour, day of week, etc.)

**You can calculate:**
- ✅ Returns (price changes)
- ✅ Moving averages (SMA, EMA)
- ✅ Volatility (rolling std)
- ✅ Technical indicators (RSI, MACD, Bollinger Bands)
- ✅ Volume (if available from other markets)

---

## 🏗️ LSTM Miner Architecture

### **Step 1: Project Structure**

Create these files:

```
precog/miners/
├── lstm_miner.py           # Your main forward function
├── models/
│   ├── __init__.py
│   ├── lstm_model.py       # LSTM architecture
│   ├── feature_engineer.py # Feature creation
│   └── trained_weights/
│       ├── btc_lstm.pth    # Trained BTC model
│       ├── eth_lstm.pth    # Trained ETH model
│       └── tao_lstm.pth    # Trained TAO model
└── scripts/
    └── train_lstm.py       # Training script
```

### **Step 2: Feature Engineering**

```python
# precog/miners/models/feature_engineer.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    """Create features from raw price data for LSTM input"""
    
    def __init__(self, lookback_window=60):
        self.lookback_window = lookback_window  # Number of timesteps to look back
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical features from price data
        
        Args:
            df: DataFrame with columns ['time', 'ReferenceRateUSD']
        
        Returns:
            DataFrame with features for each timestep
        """
        df = df.copy()
        
        # 1. Basic price features
        df['price'] = df['ReferenceRateUSD']
        
        # 2. Returns (price changes)
        df['returns_1'] = df['price'].pct_change(1)      # 1-period return
        df['returns_5'] = df['price'].pct_change(5)      # 5-period return
        df['returns_10'] = df['price'].pct_change(10)    # 10-period return
        
        # 3. Moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        # 4. Exponential moving averages
        df['ema_5'] = df['price'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['price'].ewm(span=20, adjust=False).mean()
        
        # 5. Volatility
        df['volatility_5'] = df['returns_1'].rolling(window=5).std()
        df['volatility_20'] = df['returns_1'].rolling(window=20).std()
        
        # 6. Price momentum
        df['momentum_5'] = df['price'] - df['price'].shift(5)
        df['momentum_10'] = df['price'] - df['price'].shift(10)
        
        # 7. RSI (Relative Strength Index)
        df['rsi_14'] = self.calculate_rsi(df['price'], 14)
        
        # 8. Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['price'], 20)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # 9. Time features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 10. Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols):
        """
        Prepare sequences for LSTM input
        
        Args:
            df: DataFrame with features
            feature_cols: List of column names to use as features
        
        Returns:
            numpy array of shape (num_sequences, lookback_window, num_features)
        """
        features = df[feature_cols].values
        
        sequences = []
        for i in range(len(features) - self.lookback_window):
            seq = features[i:i + self.lookback_window]
            sequences.append(seq)
        
        return np.array(sequences)
```

### **Step 3: LSTM Model Architecture**

```python
# precog/miners/models/lstm_model.py

import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    """LSTM model for cryptocurrency price prediction"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(PriceLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)  # Output: single price prediction
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            Predicted price (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class PriceIntervalLSTM(nn.Module):
    """LSTM model with dual outputs: point prediction + interval bounds"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(PriceIntervalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Shared layers
        self.fc_shared = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Point prediction head
        self.fc_point = nn.Linear(64, 1)
        
        # Interval prediction heads
        self.fc_lower = nn.Linear(64, 1)
        self.fc_upper = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Returns:
            point: Point prediction
            lower: Lower bound of interval
            upper: Upper bound of interval
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Shared representation
        shared = self.fc_shared(last_output)
        shared = self.relu(shared)
        shared = self.dropout(shared)
        
        # Point prediction
        point = self.fc_point(shared)
        
        # Interval bounds
        lower = self.fc_lower(shared)
        upper = self.fc_upper(shared)
        
        return point, lower, upper
```

### **Step 4: Training Script**

```python
# precog/miners/scripts/train_lstm.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_str, get_before
from precog.miners.models.lstm_model import PriceLSTM
from precog.miners.models.feature_engineer import FeatureEngineer

class PriceDataset(Dataset):
    """PyTorch dataset for price sequences"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def fetch_training_data(asset='btc', days=30):
    """Fetch historical data for training"""
    cm = CMData()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"Fetching {days} days of {asset} data...")
    data = cm.get_CM_ReferenceRate(
        assets=[asset],
        start=to_str(start_time),
        end=to_str(end_time),
        frequency='1m'  # 1-minute data
    )
    
    print(f"Fetched {len(data)} data points")
    return data


def prepare_training_data(data, lookback=60, forecast_horizon=60):
    """
    Prepare data for training
    
    Args:
        data: DataFrame with price data
        lookback: Number of minutes to look back
        forecast_horizon: Number of minutes to forecast ahead (60 = 1 hour)
    """
    # Create features
    fe = FeatureEngineer(lookback_window=lookback)
    df_features = fe.create_features(data)
    
    # Define feature columns (exclude time and target)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['time', 'asset', 'ReferenceRateUSD']]
    
    print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Prepare sequences
    sequences = fe.prepare_sequences(df_features, feature_cols)
    
    # Prepare targets (price after forecast_horizon)
    prices = df_features['ReferenceRateUSD'].values
    targets = []
    for i in range(len(sequences)):
        target_idx = i + lookback + forecast_horizon
        if target_idx < len(prices):
            targets.append(prices[target_idx])
        else:
            break
    
    # Trim sequences to match targets
    sequences = sequences[:len(targets)]
    targets = np.array(targets).reshape(-1, 1)
    
    print(f"Created {len(sequences)} training sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")
    
    return sequences, targets, feature_cols


def train_model(asset='btc', epochs=50, batch_size=32, learning_rate=0.001):
    """Train LSTM model for an asset"""
    
    print(f"\n{'='*60}")
    print(f"Training LSTM for {asset.upper()}")
    print(f"{'='*60}\n")
    
    # Fetch data
    data = fetch_training_data(asset=asset, days=30)
    
    # Prepare sequences
    sequences, targets, feature_cols = prepare_training_data(data)
    
    # Split train/validation
    split_idx = int(0.8 * len(sequences))
    X_train, X_val = sequences[:split_idx], sequences[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    # Create datasets and dataloaders
    train_dataset = PriceDataset(X_train, y_train)
    val_dataset = PriceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = PriceLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Calculate MAPE for interpretability
        mape = calculate_mape(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.2f} | "
              f"Val Loss: {val_loss:.2f} | "
              f"Val MAPE: {mape:.4f}%")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_path = Path(f'precog/miners/models/trained_weights/{asset}_lstm.pth')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_cols': feature_cols,
                'input_size': input_size,
                'epoch': epoch,
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✓ Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.2f}")
    return model


def calculate_mape(model, dataloader, device):
    """Calculate Mean Absolute Percentage Error"""
    model.eval()
    total_error = 0
    count = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            mape = torch.abs((batch_y - predictions) / batch_y).mean() * 100
            total_error += mape.item()
            count += 1
    
    return total_error / count


if __name__ == '__main__':
    # Train models for all assets
    for asset in ['btc', 'eth', 'tao_bittensor']:
        train_model(asset=asset, epochs=50, batch_size=32)
        print("\n")
```

### **Step 5: LSTM Forward Function**

```python
# precog/miners/lstm_miner.py

import torch
import numpy as np
import bittensor as bt
from pathlib import Path

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str
from precog.miners.models.lstm_model import PriceLSTM
from precog.miners.models.feature_engineer import FeatureEngineer


class LSTMPredictor:
    """LSTM-based price predictor"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.feature_engineer = FeatureEngineer(lookback_window=60)
        self.feature_cols = {}
        
        # Load trained models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained LSTM models"""
        for asset in ['btc', 'eth', 'tao_bittensor']:
            model_path = Path(f'precog/miners/models/trained_weights/{asset}_lstm.pth')
            
            if not model_path.exists():
                bt.logging.warning(f"Model for {asset} not found at {model_path}")
                continue
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Initialize model
                model = PriceLSTM(
                    input_size=checkpoint['input_size'],
                    hidden_size=128,
                    num_layers=2
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[asset] = model
                self.feature_cols[asset] = checkpoint['feature_cols']
                
                bt.logging.info(f"✓ Loaded LSTM model for {asset}")
            except Exception as e:
                bt.logging.error(f"Failed to load model for {asset}: {e}")
    
    def predict(self, asset, data_df):
        """Make prediction for an asset"""
        if asset not in self.models:
            bt.logging.warning(f"No model for {asset}, falling back to naive prediction")
            return None, None
        
        try:
            # Create features
            df_features = self.feature_engineer.create_features(data_df)
            
            # Prepare sequence
            sequences = self.feature_engineer.prepare_sequences(
                df_features, 
                self.feature_cols[asset]
            )
            
            if len(sequences) == 0:
                bt.logging.warning(f"Not enough data for {asset}")
                return None, None
            
            # Get last sequence
            last_sequence = sequences[-1:]
            
            # Predict
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
                prediction = self.models[asset](sequence_tensor)
                predicted_price = prediction.cpu().numpy()[0][0]
            
            # Calculate interval using recent volatility
            recent_prices = df_features['ReferenceRateUSD'].tail(60)
            returns = recent_prices.pct_change().dropna()
            volatility = returns.std()
            
            # Confidence interval
            margin = predicted_price * volatility * 2.58
            margin = max(predicted_price * 0.02, min(margin, predicted_price * 0.30))
            
            lower_bound = predicted_price - margin
            upper_bound = predicted_price + margin
            
            return predicted_price, [lower_bound, upper_bound]
            
        except Exception as e:
            bt.logging.error(f"Prediction error for {asset}: {e}")
            return None, None


# Global predictor instance
predictor = LSTMPredictor()


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """LSTM-based forward function"""
    
    assets = synapse.assets
    timestamp = synapse.timestamp
    
    bt.logging.info(f"👈 LSTM prediction request for {assets} at {timestamp}")
    
    # Fetch last 4 hours of data (for feature engineering)
    start_time = get_before(timestamp, hours=4)
    all_data = cm.get_CM_ReferenceRate(
        assets=assets,
        start=to_str(start_time),
        end=timestamp,
        frequency='1m'
    )
    
    predictions = {}
    intervals = {}
    
    if not all_data.empty:
        for asset in assets:
            asset_data = all_data[all_data['asset'] == asset]
            
            if not asset_data.empty:
                # LSTM prediction
                pred, interval = predictor.predict(asset, asset_data)
                
                if pred is not None:
                    predictions[asset] = float(pred)
                    intervals[asset] = [float(interval[0]), float(interval[1])]
                    bt.logging.success(
                        f"{asset}: LSTM Prediction=${pred:.2f} | "
                        f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
                    )
                else:
                    # Fallback to naive prediction
                    last_price = float(asset_data['ReferenceRateUSD'].iloc[-1])
                    predictions[asset] = last_price
                    intervals[asset] = [last_price * 0.95, last_price * 1.05]
                    bt.logging.warning(f"{asset}: Using fallback prediction")
    
    synapse.predictions = predictions
    synapse.intervals = intervals
    
    return synapse
```

---

## 🚀 How to Use

### **1. Train Your Models**

```bash
cd /home/fang/develop/precog

# Make sure you're in the right environment
# Train models (this will take time!)
python3 precog/miners/scripts/train_lstm.py
```

### **2. Test Your Predictions**

```python
# Test script
from precog.miners.lstm_miner import LSTMPredictor
from precog.utils.cm_data import CMData

predictor = LSTMPredictor()
cm = CMData()

# Fetch recent data
data = cm.get_CM_ReferenceRate(
    assets=['btc'],
    start='2025-01-01T00:00:00.000000Z',
    end='2025-01-01T04:00:00.000000Z',
    frequency='1m'
)

# Predict
pred, interval = predictor.predict('btc', data)
print(f"Prediction: ${pred:.2f}")
print(f"Interval: [${interval[0]:.2f}, ${interval[1]:.2f}]")
```

### **3. Deploy Your LSTM Miner**

Update `.env.miner`:
```bash
FORWARD_FUNCTION=lstm_miner
```

Add to `Makefile`:
```makefile
miner_lstm:
	pm2 start --name $(MINER_NAME) python3 -- precog/miners/miner.py \
		--forward_function lstm_miner \
		... (other args)
```

Run:
```bash
make miner_lstm ENV_FILE=.env.miner
```

---

## 💡 Tips for Success

1. **Start with good data**: Train on at least 30 days of data
2. **Regularization**: Use dropout to prevent overfitting
3. **Validation**: Always validate on unseen data
4. **Retraining**: Retrain models periodically (weekly/monthly)
5. **Monitoring**: Log prediction errors to improve
6. **Ensemble**: Combine LSTM with other strategies

---

## 📊 Expected Performance

A well-trained LSTM should:
- ✅ Beat naive baseline (last price)
- ✅ Capture short-term trends
- ✅ Achieve 1-3% MAPE on validation
- ✅ Rank in top 30-50% of miners initially
- ✅ Improve with more data and tuning

Good luck building your LSTM miner! 🚀

