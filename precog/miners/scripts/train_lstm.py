#!/usr/bin/env python3
"""Training script for LSTM price prediction models"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from precog.miners.models.feature_engineer import FeatureEngineer
from precog.miners.models.lstm_model import PriceLSTM
from precog.utils.data_source import DataSource
from precog.utils.timestamp import to_str


class PriceDataset(Dataset):
    """PyTorch dataset for price sequences"""

    def __init__(self, sequences, targets):
        # Ensure data is float32 numpy array
        if not isinstance(sequences, np.ndarray):
            sequences = np.array(sequences, dtype=np.float32)
        else:
            sequences = sequences.astype(np.float32)
            
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets, dtype=np.float32)
        else:
            targets = targets.astype(np.float32)
        
        # Convert to torch tensors
        self.sequences = torch.from_numpy(sequences).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def fetch_training_data(asset="btc", days=30, use_binance=True):
    """Fetch historical data for training"""
    # Use Binance by default (free, unlimited historical data)
    data_source = DataSource(source="binance" if use_binance else "coinmetrics")

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    print(f"Fetching {days} days of {asset} data...")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")

    data = data_source.get_CM_ReferenceRate(
        assets=[asset], start=to_str(start_time), end=to_str(end_time), frequency="1m"  # 1-minute data
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
    print(f"\nPreparing training data...")
    print(f"Lookback window: {lookback} minutes")
    print(f"Forecast horizon: {forecast_horizon} minutes")

    # Create features
    fe = FeatureEngineer(lookback_window=lookback)
    df_features = fe.create_features(data)

    # Define feature columns (exclude time and target, and ensure numeric only)
    feature_cols = []
    for col in df_features.columns:
        if col not in ["time", "asset", "ReferenceRateUSD"]:
            # Check if column is numeric
            if df_features[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]:
                feature_cols.append(col)
            else:
                print(f"  Skipping non-numeric column: {col} (dtype: {df_features[col].dtype})")

    print(f"Using {len(feature_cols)} features")
    print(f"Features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    # Prepare sequences
    sequences = fe.prepare_sequences(df_features, feature_cols)

    # Prepare targets (price after forecast_horizon)
    prices = df_features["ReferenceRateUSD"].values
    targets = []
    for i in range(len(sequences)):
        target_idx = i + lookback + forecast_horizon
        if target_idx < len(prices):
            targets.append(prices[target_idx])
        else:
            break

    # Trim sequences to match targets
    sequences = sequences[: len(targets)]
    targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
    
    # Convert sequences to float32 and ensure no objects
    sequences = sequences.astype(np.float32)
    
    # Validate data
    if np.any(np.isnan(sequences)) or np.any(np.isinf(sequences)):
        print("⚠️  Warning: Found NaN or Inf in sequences, replacing with 0")
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
        print("⚠️  Warning: Found NaN or Inf in targets, this shouldn't happen!")
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nCreated {len(sequences)} training sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")

    return sequences, targets, feature_cols


def calculate_mape(model, dataloader, device):
    """Calculate Mean Absolute Percentage Error"""
    model.eval()
    total_error = 0
    count = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            mape = torch.abs((batch_y - predictions) / (batch_y + 1e-10)).mean() * 100
            total_error += mape.item()
            count += 1

    return total_error / count if count > 0 else 0


def train_model(
    asset="btc", epochs=50, batch_size=32, learning_rate=0.001, lookback=60, forecast_horizon=60, days=30, use_binance=True
):
    """Train LSTM model for an asset"""

    print(f"\n{'='*60}")
    print(f"Training LSTM for {asset.upper()}")
    print(f"{'='*60}\n")

    # Fetch data
    try:
        data = fetch_training_data(asset=asset, days=days, use_binance=use_binance)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Make sure you have internet connection")
        if not use_binance:
            print("If using CoinMetrics, check API access")
        return None

    if data.empty:
        print(f"No data fetched for {asset}")
        return None

    # Prepare sequences
    try:
        sequences, targets, feature_cols = prepare_training_data(data, lookback, forecast_horizon)
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

    # Split train/validation
    split_idx = int(0.8 * len(sequences))
    X_train, X_val = sequences[:split_idx], sequences[split_idx:]
    y_train, y_val = targets[:split_idx], targets[split_idx:]

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = PriceDataset(X_train, y_train)
    val_dataset = PriceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = PriceLSTM(input_size=input_size, hidden_size=128, num_layers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    print(f"Model architecture: {input_size} features → LSTM(128, 2 layers) → FC → 1 output")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

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

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:8.2f} | "
            f"Val Loss: {val_loss:8.2f} | "
            f"Val MAPE: {mape:6.4f}%"
        )

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_path = Path(f"precog/miners/models/trained_weights/{asset}_lstm.pth")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": feature_cols,
                    "input_size": input_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mape": mape,
                    "lookback": lookback,
                    "forecast_horizon": forecast_horizon,
                },
                save_path,
            )
            print(f"  ✓ Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.2f}")
    print(f"Best validation MAPE: {mape:.4f}%")
    print(f"{'='*60}\n")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train LSTM models for cryptocurrency price prediction")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["btc", "eth", "tao_bittensor"],
        help="Assets to train models for",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window in minutes")
    parser.add_argument("--forecast_horizon", type=int, default=60, help="Forecast horizon in minutes")
    parser.add_argument("--days", type=int, default=30, help="Number of days of historical data to fetch")
    parser.add_argument(
        "--use-binance",
        action="store_true",
        default=True,
        help="Use Binance API (free, unlimited data). Default: True",
    )
    parser.add_argument(
        "--use-coinmetrics",
        action="store_true",
        help="Use CoinMetrics API instead of Binance",
    )

    args = parser.parse_args()

    # If user explicitly wants coinmetrics, use it
    use_binance = not args.use_coinmetrics

    print(f"\n{'='*60}")
    print("LSTM Training Configuration")
    print(f"{'='*60}")
    print(f"Data source: {'Binance (free)' if use_binance else 'CoinMetrics'}")
    print(f"Assets: {', '.join(args.assets)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Lookback: {args.lookback} minutes")
    print(f"Forecast horizon: {args.forecast_horizon} minutes")
    print(f"Training data: {args.days} days")
    print(f"{'='*60}\n")

    # Train models for all assets
    for asset in args.assets:
        try:
            train_model(
                asset=asset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lookback=args.lookback,
                forecast_horizon=args.forecast_horizon,
                days=args.days,
                use_binance=use_binance,
            )
        except Exception as e:
            print(f"\n❌ Error training model for {asset}: {e}")
            import traceback

            traceback.print_exc()

        print("\n")


if __name__ == "__main__":
    main()

