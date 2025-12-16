"""
LSTM Configuration for Price Prediction Model

All configurable parameters for the LSTM model, training, and data fetching.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    """Configuration for data fetching from Binance"""
    
    # Historical data period (in days)
    historical_days: int = 60
    
    # Data interval (Binance kline intervals: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d)
    data_interval: str = "1m"
    
    # Assets to fetch (Binance trading pairs)
    assets: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "TAOUSDT"])
    
    # Asset mapping from precog format to Binance format
    asset_mapping: dict = field(default_factory=lambda: {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "tao_bittensor": "TAOUSDT",
        "tao": "TAOUSDT",
    })
    
    # Reverse mapping for predictions
    reverse_asset_mapping: dict = field(default_factory=lambda: {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
        "TAOUSDT": "tao_bittensor",
    })


@dataclass
class ModelConfig:
    """Configuration for LSTM model architecture"""
    
    # Input sequence length (number of time steps to look back)
    # 60 minutes = 1 hour of history as input
    sequence_length: int = 60
    
    # Prediction horizon (number of time steps ahead to predict)
    # 60 minutes = 1 hour ahead prediction
    prediction_horizon: int = 60
    
    # LSTM hidden layer size
    hidden_size: int = 128
    
    # Number of LSTM layers
    num_layers: int = 2
    
    # Dropout rate for regularization
    dropout: float = 0.2
    
    # Number of features (OHLCV = 5: open, high, low, close, volume)
    num_features: int = 5
    
    # Output size (predicting close price)
    output_size: int = 1
    
    # Bidirectional LSTM
    bidirectional: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    
    # Number of epochs for training
    epochs: int = 100
    
    # Batch size
    batch_size: int = 32
    
    # Learning rate
    learning_rate: float = 0.001
    
    # Validation split ratio
    validation_split: float = 0.1
    
    # Early stopping patience (epochs)
    early_stopping_patience: int = 10
    
    # Gradient clipping value
    gradient_clip: float = 1.0
    
    # Weight decay for regularization
    weight_decay: float = 1e-5
    
    # Learning rate scheduler step size
    lr_scheduler_step_size: int = 30
    
    # Learning rate scheduler gamma
    lr_scheduler_gamma: float = 0.1


@dataclass
class ScheduleConfig:
    """Configuration for training schedule"""
    
    # Days between full retraining (default: every day)
    retrain_interval_days: int = 1
    
    # Hour of day to run training (0-23, in UTC)
    retrain_hour_utc: int = 0


@dataclass
class ModelManagementConfig:
    """Configuration for model saving and versioning"""
    
    # Base path for saving models
    model_save_path: str = "./models/lstm/"
    
    # Number of old model versions to keep
    keep_old_models: int = 5
    
    # Model filename prefix
    model_prefix: str = "lstm_price_predictor"
    
    # Save best model during training
    save_best_model: bool = True
    
    # Checkpoint frequency (epochs)
    checkpoint_frequency: int = 10


@dataclass
class LSTMConfig:
    """Master configuration combining all sub-configs"""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    management: ModelManagementConfig = field(default_factory=ModelManagementConfig)
    
    # Device configuration
    device: str = "cuda"  # or "cpu"
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Logging level
    verbose: bool = True
    
    def __post_init__(self):
        """Create model save directory if it doesn't exist"""
        os.makedirs(self.management.model_save_path, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "LSTMConfig":
        """Create LSTMConfig from a dictionary"""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        schedule_config = ScheduleConfig(**config_dict.get("schedule", {}))
        management_config = ModelManagementConfig(**config_dict.get("management", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            schedule=schedule_config,
            management=management_config,
            device=config_dict.get("device", "cuda"),
            random_seed=config_dict.get("random_seed", 42),
            verbose=config_dict.get("verbose", True),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "data": {
                "historical_days": self.data.historical_days,
                "data_interval": self.data.data_interval,
                "assets": self.data.assets,
            },
            "model": {
                "sequence_length": self.model.sequence_length,
                "prediction_horizon": self.model.prediction_horizon,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "num_features": self.model.num_features,
                "output_size": self.model.output_size,
                "bidirectional": self.model.bidirectional,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "validation_split": self.training.validation_split,
                "early_stopping_patience": self.training.early_stopping_patience,
                "gradient_clip": self.training.gradient_clip,
                "weight_decay": self.training.weight_decay,
            },
            "schedule": {
                "retrain_interval_days": self.schedule.retrain_interval_days,
                "retrain_hour_utc": self.schedule.retrain_hour_utc,
            },
            "management": {
                "model_save_path": self.management.model_save_path,
                "keep_old_models": self.management.keep_old_models,
                "model_prefix": self.management.model_prefix,
            },
            "device": self.device,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
        }


# Default configuration instance
DEFAULT_LSTM_CONFIG = LSTMConfig()
