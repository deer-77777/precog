#!/usr/bin/env python3
"""
Incremental Model Training (Fine-tuning)

This script loads an existing model and fine-tunes it with recent data.
Faster than full retraining, good for daily updates.
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import numpy as np
import bittensor as bt

from precog.utils.data_source import DataSource
from precog.utils.timestamp import get_now
from precog.miners.models.feature_engineer import FeatureEngineer
from precog.miners.models.lstm_model import LSTMPredictor


def incremental_train(asset="btc", days=7, epochs=10, learning_rate=0.0001):
    """
    Fine-tune existing model with recent data
    
    Args:
        asset: Asset to train (btc, eth, tao_bittensor)
        days: Days of recent data to use for fine-tuning
        epochs: Number of training epochs
        learning_rate: Lower learning rate for fine-tuning
    """
    bt.logging.info("=" * 60)
    bt.logging.info(f"🔄 INCREMENTAL TRAINING - {asset.upper()}")
    bt.logging.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    bt.logging.info("=" * 60)
    
    model_path = f"precog/miners/models/trained_weights/{asset}_lstm.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        bt.logging.error(f"❌ Model not found: {model_path}")
        bt.logging.error("   Please run full training first!")
        return False
    
    try:
        # 1. Load existing model
        bt.logging.info(f"📂 Loading existing model: {model_path}")
        model = LSTMPredictor()
        model.load_state_dict(torch.load(model_path))
        model.train()  # Set to training mode
        
        # 2. Fetch recent data
        bt.logging.info(f"📊 Fetching last {days} days of {asset.upper()} data...")
        data_source = DataSource(use_binance=True)
        end_time = get_now()
        start_time = end_time - timedelta(days=days)
        
        df = data_source.get_historical_data(
            assets=asset,
            start=start_time,
            end=end_time,
            frequency="1m"
        )
        
        if df.empty:
            bt.logging.error("❌ No data fetched")
            return False
        
        bt.logging.info(f"✅ Fetched {len(df)} data points")
        
        # 3. Create features
        bt.logging.info("🔧 Engineering features...")
        feature_engineer = FeatureEngineer(lookback_window=60)
        df_features = feature_engineer.create_features(df)
        
        # Get feature columns
        feature_cols = [col for col in df_features.columns 
                       if col not in ['time', 'asset', 'ReferenceRateUSD']]
        
        bt.logging.info(f"📊 Created {len(feature_cols)} features")
        
        # 4. Prepare sequences
        bt.logging.info("🔄 Preparing sequences...")
        sequences = feature_engineer.prepare_sequences(df_features, feature_cols)
        
        if len(sequences) == 0:
            bt.logging.error("❌ Not enough data for training sequences")
            return False
        
        # 5. Prepare training data
        X_train = []
        y_point_train = []
        y_lower_train = []
        y_upper_train = []
        
        lookback = 60
        forecast_horizon = 60  # 1 hour ahead
        
        for i in range(len(sequences) - forecast_horizon):
            X_train.append(sequences[i])
            
            # Target: price 1 hour ahead
            future_idx = i + lookback + forecast_horizon
            if future_idx < len(df):
                actual_price = df.iloc[future_idx]['ReferenceRateUSD']
                
                # Calculate volatility for interval estimation
                recent_prices = df.iloc[max(0, future_idx-20):future_idx]['ReferenceRateUSD'].values
                if len(recent_prices) > 1:
                    volatility = np.std(recent_prices)
                else:
                    volatility = actual_price * 0.01
                
                # Point prediction
                y_point_train.append(actual_price)
                
                # Interval predictions (±2 std)
                y_lower_train.append(actual_price - 2 * volatility)
                y_upper_train.append(actual_price + 2 * volatility)
        
        if len(X_train) == 0:
            bt.logging.error("❌ No training samples created")
            return False
        
        # Convert to tensors
        X_train = torch.FloatTensor(np.array(X_train, dtype=np.float32))
        y_point = torch.FloatTensor(y_point_train).unsqueeze(1)
        y_lower = torch.FloatTensor(y_lower_train).unsqueeze(1)
        y_upper = torch.FloatTensor(y_upper_train).unsqueeze(1)
        
        bt.logging.info(f"📊 Training samples: {len(X_train)}")
        
        # 6. Fine-tune model
        bt.logging.info(f"🎓 Fine-tuning for {epochs} epochs (LR: {learning_rate})...")
        
        # Use lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion_point = nn.MSELoss()
        criterion_interval = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            
            # Forward pass
            pred_point, pred_lower, pred_upper = model(X_train)
            
            # Calculate losses
            loss_point = criterion_point(pred_point, y_point)
            loss_lower = criterion_interval(pred_lower, y_lower)
            loss_upper = criterion_interval(pred_upper, y_upper)
            
            # Combined loss
            loss = loss_point + 0.3 * loss_lower + 0.3 * loss_upper
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate MAPE
            with torch.no_grad():
                mape = torch.mean(torch.abs((y_point - pred_point) / y_point)) * 100
            
            if (epoch + 1) % 2 == 0:
                bt.logging.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Loss: {loss.item():.4f} "
                    f"MAPE: {mape.item():.2f}%"
                )
        
        # 7. Save fine-tuned model
        bt.logging.info(f"💾 Saving fine-tuned model...")
        torch.save(model.state_dict(), model_path)
        
        # 8. Update metadata
        metadata_path = f"precog/miners/models/trained_weights/{asset}_lstm_metadata.json"
        import json
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        metadata.update({
            "last_incremental_train": datetime.now().isoformat(),
            "incremental_days": days,
            "incremental_epochs": epochs,
            "incremental_lr": learning_rate,
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        bt.logging.info("=" * 60)
        bt.logging.info(f"✅ INCREMENTAL TRAINING COMPLETED")
        bt.logging.info(f"📊 Final MAPE: {mape.item():.2f}%")
        bt.logging.info("=" * 60)
        
        return True
        
    except Exception as e:
        bt.logging.error(f"❌ Error during incremental training: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Incremental LSTM Model Training")
    parser.add_argument("--asset", type=str, default="btc",
                       choices=["btc", "eth", "tao_bittensor"],
                       help="Asset to train")
    parser.add_argument("--days", type=int, default=7,
                       help="Days of recent data (default: 7)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs (default: 10)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate (default: 0.0001)")
    
    args = parser.parse_args()
    
    success = incremental_train(
        asset=args.asset,
        days=args.days,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

