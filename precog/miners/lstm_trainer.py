"""
LSTM Trainer - Standalone Training Script

This script runs separately from the miner to train LSTM models.
It retrains models every day (configurable) with historical data.

Usage:
    python -m precog.miners.lstm_trainer [options]

Options:
    --historical_days: Number of days of historical data (default: 60)
    --epochs: Number of training epochs (default: 100)
    --batch_size: Training batch size (default: 32)
    --learning_rate: Learning rate (default: 0.001)
    --retrain_interval_days: Days between retraining (default: 1)
    --model_save_path: Path to save models (default: ./models/lstm/)
    --run_once: Run training once and exit (no scheduling)
"""

import argparse
import time
import schedule
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from precog.miners.models.logging_utils import logger as bt_logging


# Use simple logger for standalone training
class bt:
    logging = bt_logging

from precog.utils.binance_data import BinanceData
from precog.miners.models import (
    LSTMConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ScheduleConfig,
    ModelManagementConfig,
    PricePredictorLSTM,
    DataPreprocessor,
    Trainer,
    ModelManager,
    create_data_loaders,
)


class LSTMModelTrainer:
    """
    Standalone trainer for LSTM price prediction models.
    
    Trains models for BTC, ETH, and TAO using Binance historical data.
    Models are saved to disk for the miner to load.
    """
    
    def __init__(self, config: LSTMConfig):
        """
        Initialize trainer.
        
        Args:
            config: LSTM configuration
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        bt.logging.info(f"LSTM Trainer initialized on {self.device}")
        
        # Initialize data fetcher
        self.binance = BinanceData()
        
        # Initialize model manager
        self.model_manager = ModelManager(
            base_path=config.management.model_save_path,
            model_prefix=config.management.model_prefix,
            keep_old_models=config.management.keep_old_models,
        )
        
        # Set random seed
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def train_model(self, symbol: str) -> bool:
        """
        Train model for a single asset.
        
        Args:
            symbol: Binance trading pair (e.g., "BTCUSDT")
        
        Returns:
            True if training succeeded
        """
        bt.logging.info(f"=" * 60)
        bt.logging.info(f"Training model for {symbol}")
        bt.logging.info(f"=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Fetch historical data
            bt.logging.info(f"Fetching {self.config.data.historical_days} days of {symbol} data...")
            df = self.binance.get_historical_klines(
                symbol=symbol,
                interval=self.config.data.data_interval,
                days=self.config.data.historical_days,
            )
            
            if df.empty:
                bt.logging.error(f"No data fetched for {symbol}")
                return False
            
            bt.logging.info(f"Fetched {len(df)} records ({df['open_time'].min()} to {df['open_time'].max()})")
            
            # Step 2: Create preprocessor with 1-hour prediction horizon
            # predict_interval=True creates 3 targets: (price, min, max)
            preprocessor = DataPreprocessor(
                sequence_length=self.config.model.sequence_length,
                prediction_horizon=self.config.model.prediction_horizon,  # 60 = 1 hour
                predict_interval=self.config.model.predict_interval,
            )
            
            # Step 3: Create data loaders
            bt.logging.info("Creating training data sequences...")
            train_loader, val_loader, preprocessor = create_data_loaders(
                df=df,
                preprocessor=preprocessor,
                batch_size=self.config.training.batch_size,
                validation_split=self.config.training.validation_split,
            )
            
            bt.logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            
            # Step 4: Create model
            model = PricePredictorLSTM(
                input_size=self.config.model.num_features,
                hidden_size=self.config.model.hidden_size,
                num_layers=self.config.model.num_layers,
                output_size=self.config.model.output_size,
                dropout=self.config.model.dropout,
                bidirectional=self.config.model.bidirectional,
            ).to(self.device)
            
            bt.logging.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
            
            # Step 5: Create trainer
            trainer = Trainer(
                model=model,
                device=str(self.device),
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                gradient_clip=self.config.training.gradient_clip,
                early_stopping_patience=self.config.training.early_stopping_patience,
                lr_scheduler_step_size=self.config.training.lr_scheduler_step_size,
                lr_scheduler_gamma=self.config.training.lr_scheduler_gamma,
                checkpoint_dir=f"{self.config.management.model_save_path}/checkpoints/{symbol}/",
            )
            
            # Step 6: Train
            bt.logging.info(f"Starting training for {self.config.training.epochs} epochs...")
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.training.epochs,
                verbose=self.config.verbose,
                checkpoint_frequency=self.config.management.checkpoint_frequency,
            )
            
            # Step 7: Evaluate
            metrics = trainer.evaluate(val_loader, preprocessor)
            bt.logging.info(f"Final metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                          f"MAPE: {metrics['mape']:.2f}%, Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
            
            # Log interval-specific metrics if available
            if 'min_mae' in metrics:
                bt.logging.info(f"Interval metrics - Min MAE: {metrics['min_mae']:.4f}, Max MAE: {metrics['max_mae']:.4f}, "
                              f"Width MAE: {metrics['width_mae']:.4f}")
                bt.logging.info(f"Interval coverage - Contains Min: {metrics['interval_contains_min']:.1f}%, "
                              f"Contains Max: {metrics['interval_contains_max']:.1f}%")
            
            # Step 8: Save model
            self.model_manager.save_model(
                model=model,
                preprocessor=preprocessor,
                asset=symbol,
                metrics=metrics,
                config=self.config.to_dict(),
                is_incremental=False,
            )
            
            elapsed = time.time() - start_time
            bt.logging.success(f"Training completed for {symbol} in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Training failed for {symbol}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
    
    def train_all_models(self):
        """Train models for all configured assets."""
        bt.logging.info(f"Starting training for all assets: {self.config.data.assets}")
        bt.logging.info(f"Configuration: {self.config.data.historical_days} days history, "
                       f"{self.config.training.epochs} epochs, "
                       f"prediction horizon: {self.config.model.prediction_horizon} minutes (1 hour)")
        
        results = {}
        total_start = time.time()
        
        for symbol in self.config.data.assets:
            success = self.train_model(symbol)
            results[symbol] = "SUCCESS" if success else "FAILED"
        
        total_elapsed = time.time() - total_start
        
        bt.logging.info("=" * 60)
        bt.logging.info("TRAINING SUMMARY")
        bt.logging.info("=" * 60)
        for symbol, status in results.items():
            bt.logging.info(f"  {symbol}: {status}")
        bt.logging.info(f"Total time: {total_elapsed:.2f}s")
        bt.logging.info("=" * 60)
        
        return results


def create_config_from_args(args) -> LSTMConfig:
    """Create LSTMConfig from command line arguments."""
    data_config = DataConfig(
        historical_days=args.historical_days,
        data_interval=args.data_interval,
    )
    
    model_config = ModelConfig(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    schedule_config = ScheduleConfig(
        retrain_interval_days=args.retrain_interval_days,
        retrain_hour_utc=args.retrain_hour_utc,
    )
    
    management_config = ModelManagementConfig(
        model_save_path=args.model_save_path,
        keep_old_models=args.keep_old_models,
    )
    
    return LSTMConfig(
        data=data_config,
        model=model_config,
        training=training_config,
        schedule=schedule_config,
        management=management_config,
        device=args.device,
        random_seed=args.random_seed,
        verbose=not args.quiet,
    )


def run_scheduled_training(trainer: LSTMModelTrainer):
    """Run training as scheduled job."""
    bt.logging.info(f"Running scheduled training at {datetime.utcnow().isoformat()}")
    trainer.train_all_models()


def main():
    print("=" * 70, flush=True)
    print("LSTM TRAINER - Starting...", flush=True)
    print("=" * 70, flush=True)
    
    parser = argparse.ArgumentParser(description="LSTM Model Trainer for Price Prediction")
    
    # Data configuration
    parser.add_argument("--historical_days", type=int, default=60,
                       help="Number of days of historical data (default: 60)")
    parser.add_argument("--data_interval", type=str, default="1m",
                       help="Data interval: 1m, 5m, 15m, 1h, etc. (default: 1m)")
    
    # Model configuration
    parser.add_argument("--sequence_length", type=int, default=60,
                       help="Input sequence length in time steps (default: 60)")
    parser.add_argument("--prediction_horizon", type=int, default=60,
                       help="Prediction horizon in time steps (default: 60 = 1 hour)")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="LSTM hidden size (default: 128)")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of LSTM layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate (default: 0.2)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--validation_split", type=float, default=0.1,
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                       help="Early stopping patience (default: 10)")
    
    # Schedule configuration
    parser.add_argument("--retrain_interval_days", type=int, default=1,
                       help="Days between retraining (default: 1)")
    parser.add_argument("--retrain_hour_utc", type=int, default=0,
                       help="Hour (UTC) to run daily training (default: 0)")
    
    # Model management
    parser.add_argument("--model_save_path", type=str, default="./models/lstm/",
                       help="Path to save models (default: ./models/lstm/)")
    parser.add_argument("--keep_old_models", type=int, default=5,
                       help="Number of old models to keep (default: 5)")
    
    # Runtime options
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu (default: cuda)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce logging output")
    parser.add_argument("--run_once", action="store_true",
                       help="Run training once and exit (no scheduling)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create trainer
    trainer = LSTMModelTrainer(config)
    
    if args.run_once:
        # Run once and exit
        bt.logging.info("Running single training session...")
        trainer.train_all_models()
    else:
        # Run immediately, then schedule daily
        bt.logging.info(f"Starting scheduled training (every {args.retrain_interval_days} day(s) at {args.retrain_hour_utc}:00 UTC)")
        
        # Run initial training
        trainer.train_all_models()
        
        # Schedule daily training
        schedule_time = f"{args.retrain_hour_utc:02d}:00"
        
        if args.retrain_interval_days == 1:
            schedule.every().day.at(schedule_time).do(run_scheduled_training, trainer)
        else:
            schedule.every(args.retrain_interval_days).days.at(schedule_time).do(run_scheduled_training, trainer)
        
        bt.logging.info(f"Training scheduled for {schedule_time} UTC every {args.retrain_interval_days} day(s)")
        bt.logging.info("Trainer running... Press Ctrl+C to stop")
        
        # Run scheduler loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            bt.logging.info("Trainer stopped by user")


if __name__ == "__main__":
    main()

