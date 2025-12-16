#!/usr/bin/env python3
"""
Automated Model Retraining System

This script:
1. Fetches latest data from Binance
2. Trains a new LSTM model
3. Validates its performance
4. Deploys it if it's better than the current model
"""

import sys
import os
from datetime import datetime, timedelta
import shutil
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

import torch
import bittensor as bt

from precog.utils.data_source import DataSource
from precog.utils.timestamp import get_now
from precog.miners.models.feature_engineer import FeatureEngineer
from precog.miners.models.lstm_model import LSTMPredictor


class ModelRetrainer:
    def __init__(self, asset="btc", days=30, epochs=50):
        self.asset = asset
        self.days = days
        self.epochs = epochs
        self.model_dir = "precog/miners/models/trained_weights"
        self.archive_dir = "precog/miners/models/model_archives"
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        self.current_model_path = f"{self.model_dir}/{asset}_lstm.pth"
        self.new_model_path = f"{self.model_dir}/{asset}_lstm_new.pth"
        
    def archive_current_model(self):
        """Archive the current model with timestamp"""
        if os.path.exists(self.current_model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"{self.archive_dir}/{self.asset}_lstm_{timestamp}.pth"
            shutil.copy(self.current_model_path, archive_path)
            bt.logging.info(f"✅ Archived current model to: {archive_path}")
            
            # Keep only last 5 archives to save space
            self._cleanup_old_archives()
    
    def _cleanup_old_archives(self):
        """Keep only the 5 most recent model archives"""
        archives = sorted([
            os.path.join(self.archive_dir, f) 
            for f in os.listdir(self.archive_dir)
            if f.startswith(f"{self.asset}_lstm_") and f.endswith(".pth")
        ])
        
        # Remove oldest archives if more than 5
        if len(archives) > 5:
            for old_archive in archives[:-5]:
                os.remove(old_archive)
                bt.logging.info(f"🗑️ Removed old archive: {old_archive}")
    
    def train_new_model(self):
        """Train a new model with latest data"""
        bt.logging.info(f"🚀 Starting retraining for {self.asset}...")
        bt.logging.info(f"📅 Using {self.days} days of data, {self.epochs} epochs")
        
        try:
            # Run training script
            cmd = [
                "python3", "precog/miners/scripts/train_lstm.py",
                "--assets", self.asset,
                "--days", str(self.days),
                "--epochs", str(self.epochs),
                "--use-binance",
                "--learning-rate", "0.0005"
            ]
            
            bt.logging.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                bt.logging.error(f"❌ Training failed: {result.stderr}")
                return False
            
            bt.logging.info(f"✅ Training completed successfully")
            
            # Rename to new model path
            if os.path.exists(self.current_model_path):
                shutil.move(self.current_model_path, self.new_model_path)
            
            return True
            
        except Exception as e:
            bt.logging.error(f"❌ Training error: {e}")
            return False
    
    def validate_model(self, model_path):
        """
        Test model performance on recent data
        Returns: MAPE (Mean Absolute Percentage Error)
        """
        try:
            bt.logging.info(f"🔍 Validating model: {model_path}")
            
            # Load model
            model = LSTMPredictor()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            # Fetch recent test data (last 7 days)
            data_source = DataSource(use_binance=True)
            end_time = get_now()
            start_time = end_time - timedelta(days=7)
            
            df = data_source.get_historical_data(
                assets=self.asset,
                start=start_time,
                end=end_time,
                frequency="1m"
            )
            
            if df.empty:
                bt.logging.error("❌ No validation data available")
                return float('inf')
            
            # Create features
            feature_engineer = FeatureEngineer(lookback_window=60)
            df_features = feature_engineer.create_features(df)
            
            # Get feature columns (exclude metadata)
            feature_cols = [col for col in df_features.columns 
                          if col not in ['time', 'asset', 'ReferenceRateUSD']]
            
            # Prepare sequences
            sequences = feature_engineer.prepare_sequences(df_features, feature_cols)
            
            if len(sequences) == 0:
                bt.logging.error("❌ Not enough data for validation")
                return float('inf')
            
            # Make predictions
            errors = []
            with torch.no_grad():
                for i in range(len(sequences) - 1):
                    # Input sequence
                    x = torch.FloatTensor(sequences[i]).unsqueeze(0)
                    
                    # Actual future price (1 hour ahead ≈ 60 minutes)
                    future_idx = min(i + 60, len(df) - 1)
                    actual_price = df.iloc[future_idx]['ReferenceRateUSD']
                    
                    # Predict
                    pred_point, pred_lower, pred_upper = model(x)
                    predicted_price = pred_point.item()
                    
                    # Calculate error
                    if actual_price > 0:
                        error = abs(predicted_price - actual_price) / actual_price
                        errors.append(error)
            
            # Calculate MAPE
            if errors:
                mape = (sum(errors) / len(errors)) * 100
                bt.logging.info(f"📊 Validation MAPE: {mape:.2f}%")
                return mape
            else:
                return float('inf')
                
        except Exception as e:
            bt.logging.error(f"❌ Validation error: {e}")
            return float('inf')
    
    def compare_models(self):
        """
        Compare new model vs current model
        Returns: True if new model is better
        """
        bt.logging.info("⚖️ Comparing models...")
        
        # Validate current model
        if os.path.exists(self.current_model_path):
            current_mape = self.validate_model(self.current_model_path)
            bt.logging.info(f"📊 Current model MAPE: {current_mape:.2f}%")
        else:
            bt.logging.warning("⚠️ No current model found, will use new model")
            current_mape = float('inf')
        
        # Validate new model
        if os.path.exists(self.new_model_path):
            new_mape = self.validate_model(self.new_model_path)
            bt.logging.info(f"📊 New model MAPE: {new_mape:.2f}%")
        else:
            bt.logging.error("❌ New model not found")
            return False
        
        # Check if new model is acceptable
        if new_mape > 5.0:
            bt.logging.error(f"❌ New model MAPE too high: {new_mape:.2f}% (threshold: 5%)")
            return False
        
        # Check if new model is better
        improvement = current_mape - new_mape
        if new_mape < current_mape or current_mape == float('inf'):
            bt.logging.info(f"✅ New model is better! Improvement: {improvement:.2f}%")
            return True
        else:
            bt.logging.warning(f"⚠️ New model is worse by {-improvement:.2f}%")
            
            # Allow small degradation (< 0.5%) if current model is old
            if abs(improvement) < 0.5:
                bt.logging.info("✅ Degradation is minor, accepting new model")
                return True
            else:
                return False
    
    def deploy_new_model(self):
        """Replace current model with new model"""
        try:
            # Archive current model
            self.archive_current_model()
            
            # Deploy new model
            shutil.move(self.new_model_path, self.current_model_path)
            bt.logging.info(f"🚀 Deployed new model: {self.current_model_path}")
            
            # Create metadata file
            metadata = {
                "asset": self.asset,
                "training_date": datetime.now().isoformat(),
                "training_days": self.days,
                "epochs": self.epochs,
            }
            
            metadata_path = f"{self.model_dir}/{self.asset}_lstm_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            bt.logging.error(f"❌ Deployment error: {e}")
            return False
    
    def run(self):
        """Main retraining workflow"""
        bt.logging.info("=" * 60)
        bt.logging.info(f"🔄 AUTOMATED RETRAINING - {self.asset.upper()}")
        bt.logging.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        bt.logging.info("=" * 60)
        
        # Step 1: Train new model
        if not self.train_new_model():
            bt.logging.error("❌ Training failed, aborting")
            return False
        
        # Step 2: Compare models
        if not self.compare_models():
            bt.logging.warning("⚠️ New model not better, keeping current model")
            # Clean up new model
            if os.path.exists(self.new_model_path):
                os.remove(self.new_model_path)
            return False
        
        # Step 3: Deploy new model
        if not self.deploy_new_model():
            bt.logging.error("❌ Deployment failed")
            return False
        
        bt.logging.info("=" * 60)
        bt.logging.info("✅ RETRAINING COMPLETED SUCCESSFULLY")
        bt.logging.info("=" * 60)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Automated LSTM Model Retraining")
    parser.add_argument("--asset", type=str, default="btc", 
                       choices=["btc", "eth", "tao_bittensor"],
                       help="Asset to retrain (default: btc)")
    parser.add_argument("--days", type=int, default=30,
                       help="Days of training data (default: 30)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Training epochs (default: 50)")
    
    args = parser.parse_args()
    
    # Run retraining
    retrainer = ModelRetrainer(
        asset=args.asset,
        days=args.days,
        epochs=args.epochs
    )
    
    success = retrainer.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

