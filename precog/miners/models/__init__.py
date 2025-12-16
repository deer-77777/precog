"""
LSTM Models Package for Price Prediction

This package contains:
- lstm_config: Configuration dataclasses
- lstm_model: LSTM neural network architecture
- data_preprocessor: Data preprocessing and sequence creation
- trainer: Training pipeline
- model_manager: Model versioning and lifecycle management

Architecture:
- Trainer (lstm_trainer.py): Runs separately, trains models daily, saves to disk
- Miner (lstm_miner.py): Loads models from disk, makes predictions
"""

from precog.miners.models.lstm_config import (
    LSTMConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ScheduleConfig,
    ModelManagementConfig,
    DEFAULT_LSTM_CONFIG,
)

from precog.miners.models.lstm_model import (
    PricePredictorLSTM,
    PricePredictorLSTMWithAttention,
    create_model,
)

from precog.miners.models.data_preprocessor import (
    DataPreprocessor,
    PriceDataset,
    MultiAssetPreprocessor,
    create_data_loaders,
    create_incremental_loader,
)

from precog.miners.models.trainer import (
    Trainer,
    EarlyStopping,
    create_trainer,
)

from precog.miners.models.model_manager import (
    ModelManager,
    MultiAssetModelManager,
)


__all__ = [
    # Config
    "LSTMConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ScheduleConfig",
    "ModelManagementConfig",
    "DEFAULT_LSTM_CONFIG",
    # Model
    "PricePredictorLSTM",
    "PricePredictorLSTMWithAttention",
    "create_model",
    # Preprocessor
    "DataPreprocessor",
    "PriceDataset",
    "MultiAssetPreprocessor",
    "create_data_loaders",
    "create_incremental_loader",
    # Trainer
    "Trainer",
    "EarlyStopping",
    "create_trainer",
    # Model Manager
    "ModelManager",
    "MultiAssetModelManager",
]

