"""
Model Manager for LSTM Price Prediction

Handles model versioning, saving, loading, and lifecycle management.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn

try:
    import bittensor as bt
except ImportError:
    class bt:
        class logging:
            @staticmethod
            def info(msg): print(f"[INFO] {msg}")
            @staticmethod
            def debug(msg): print(f"[DEBUG] {msg}")
            @staticmethod
            def warning(msg): print(f"[WARN] {msg}")
            @staticmethod
            def error(msg): print(f"[ERROR] {msg}")
            @staticmethod
            def success(msg): print(f"[SUCCESS] {msg}")

from precog.miners.models.lstm_model import PricePredictorLSTM, create_model
from precog.miners.models.data_preprocessor import DataPreprocessor


class ModelManager:
    """
    Manages LSTM model lifecycle including:
    - Saving and loading models
    - Model versioning
    - Archiving old models
    - Metadata management
    """
    
    METADATA_FILE = "model_metadata.json"
    MODEL_FILE = "model.pt"
    PREPROCESSOR_FILE = "preprocessor.pkl"
    
    def __init__(
        self,
        base_path: str = "./models/lstm/",
        model_prefix: str = "lstm_price_predictor",
        keep_old_models: int = 5,
    ):
        """
        Initialize model manager.
        
        Args:
            base_path: Base directory for model storage
            model_prefix: Prefix for model directory names
            keep_old_models: Number of old model versions to retain
        """
        self.base_path = Path(base_path)
        self.model_prefix = model_prefix
        self.keep_old_models = keep_old_models
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Current model path
        self.current_model_path = self.base_path / "current"
        
        # Archive path
        self.archive_path = self.base_path / "archive"
        self.archive_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: nn.Module,
        preprocessor: DataPreprocessor,
        asset: str,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None,
        is_incremental: bool = False,
    ) -> str:
        """
        Save model, preprocessor, and metadata.
        
        Args:
            model: Trained LSTM model
            preprocessor: Fitted data preprocessor
            asset: Asset symbol (e.g., "BTCUSDT")
            metrics: Training/validation metrics
            config: Model configuration
            is_incremental: Whether this is an incremental update
        
        Returns:
            Path to saved model
        """
        # Create asset-specific directory
        asset_path = self.current_model_path / asset
        
        # Archive existing model before saving new one (only for full training)
        if not is_incremental and asset_path.exists():
            self._archive_model(asset)
        
        asset_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_file = asset_path / self.MODEL_FILE
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": model.input_size,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "output_size": model.output_size,
                "bidirectional": model.bidirectional,
            },
        }, model_file)
        
        # Save preprocessor
        preprocessor_file = asset_path / self.PREPROCESSOR_FILE
        preprocessor.save(str(preprocessor_file))
        
        # Save metadata
        metadata = {
            "asset": asset,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "is_incremental": is_incremental,
            "version": self._get_next_version(asset),
            "metrics": metrics or {},
            "config": config or {},
        }
        
        # Load existing metadata to update version tracking
        existing_metadata = self._load_metadata(asset)
        if existing_metadata:
            if is_incremental:
                metadata["version"] = existing_metadata.get("version", 1)
                metadata["incremental_updates"] = existing_metadata.get("incremental_updates", 0) + 1
                metadata["created_at"] = existing_metadata.get("created_at")
            else:
                metadata["incremental_updates"] = 0
        
        metadata_file = asset_path / self.METADATA_FILE
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        bt.logging.success(f"Model saved for {asset} at {asset_path}")
        return str(asset_path)
    
    def load_model(
        self,
        asset: str,
        device: str = "cuda",
    ) -> Tuple[nn.Module, DataPreprocessor, Dict]:
        """
        Load model, preprocessor, and metadata.
        
        Args:
            asset: Asset symbol
            device: Device to load model on
        
        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        asset_path = self.current_model_path / asset
        
        if not asset_path.exists():
            raise FileNotFoundError(f"No model found for {asset}")
        
        # Load model
        model_file = asset_path / self.MODEL_FILE
        checkpoint = torch.load(model_file, map_location=device)
        
        model_config = checkpoint["model_config"]
        model = PricePredictorLSTM(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            output_size=model_config["output_size"],
            bidirectional=model_config.get("bidirectional", False),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        # Load preprocessor
        preprocessor_file = asset_path / self.PREPROCESSOR_FILE
        preprocessor = DataPreprocessor.load(str(preprocessor_file))
        
        # Load metadata
        metadata = self._load_metadata(asset)
        
        bt.logging.info(f"Model loaded for {asset} (version {metadata.get('version', 'N/A')})")
        
        return model, preprocessor, metadata
    
    def model_exists(self, asset: str) -> bool:
        """Check if model exists for an asset"""
        asset_path = self.current_model_path / asset
        return (asset_path / self.MODEL_FILE).exists()
    
    def _archive_model(self, asset: str):
        """Archive existing model before replacement"""
        asset_path = self.current_model_path / asset
        
        if not asset_path.exists():
            return
        
        # Load metadata to get version
        metadata = self._load_metadata(asset)
        version = metadata.get("version", 1) if metadata else 1
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Create archive directory
        archive_name = f"{self.model_prefix}_{asset}_v{version}_{timestamp}"
        archive_dest = self.archive_path / archive_name
        
        # Move current model to archive
        shutil.move(str(asset_path), str(archive_dest))
        
        bt.logging.info(f"Archived model: {archive_name}")
        
        # Clean up old archives
        self._cleanup_old_archives(asset)
    
    def _cleanup_old_archives(self, asset: str):
        """Remove old archived models beyond keep_old_models limit"""
        # Find all archives for this asset
        archives = sorted(
            [d for d in self.archive_path.iterdir() 
             if d.is_dir() and asset in d.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        
        # Remove archives beyond limit
        for archive in archives[self.keep_old_models:]:
            shutil.rmtree(archive)
            bt.logging.debug(f"Removed old archive: {archive.name}")
    
    def _load_metadata(self, asset: str) -> Optional[Dict]:
        """Load metadata for an asset"""
        metadata_file = self.current_model_path / asset / self.METADATA_FILE
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            return json.load(f)
    
    def _get_next_version(self, asset: str) -> int:
        """Get next version number for an asset"""
        metadata = self._load_metadata(asset)
        
        if metadata:
            return metadata.get("version", 0) + 1
        
        return 1
    
    def get_model_info(self, asset: str) -> Optional[Dict]:
        """Get information about current model for an asset"""
        metadata = self._load_metadata(asset)
        
        if not metadata:
            return None
        
        return {
            "asset": asset,
            "version": metadata.get("version"),
            "created_at": metadata.get("created_at"),
            "updated_at": metadata.get("updated_at"),
            "incremental_updates": metadata.get("incremental_updates", 0),
            "metrics": metadata.get("metrics", {}),
        }
    
    def list_models(self) -> List[Dict]:
        """List all available models"""
        models = []
        
        if not self.current_model_path.exists():
            return models
        
        for asset_dir in self.current_model_path.iterdir():
            if asset_dir.is_dir():
                info = self.get_model_info(asset_dir.name)
                if info:
                    models.append(info)
        
        return models
    
    def list_archives(self) -> List[Dict]:
        """List all archived models"""
        archives = []
        
        for archive_dir in self.archive_path.iterdir():
            if archive_dir.is_dir():
                metadata_file = archive_dir / self.METADATA_FILE
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    archives.append({
                        "name": archive_dir.name,
                        "asset": metadata.get("asset"),
                        "version": metadata.get("version"),
                        "created_at": metadata.get("created_at"),
                    })
        
        return sorted(archives, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def restore_from_archive(self, archive_name: str) -> bool:
        """
        Restore a model from archive.
        
        Args:
            archive_name: Name of archived model directory
        
        Returns:
            True if successful
        """
        archive_path = self.archive_path / archive_name
        
        if not archive_path.exists():
            bt.logging.error(f"Archive not found: {archive_name}")
            return False
        
        # Load archive metadata to get asset name
        metadata_file = archive_path / self.METADATA_FILE
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        asset = metadata.get("asset")
        
        # Archive current model first
        if self.model_exists(asset):
            self._archive_model(asset)
        
        # Restore from archive
        dest_path = self.current_model_path / asset
        shutil.copytree(archive_path, dest_path)
        
        bt.logging.success(f"Restored model from archive: {archive_name}")
        return True
    
    def should_full_retrain(
        self,
        asset: str,
        full_train_interval_days: int = 3,
    ) -> bool:
        """
        Check if model needs full retraining.
        
        Args:
            asset: Asset symbol
            full_train_interval_days: Days between full retraining
        
        Returns:
            True if full retraining is needed
        """
        if not self.model_exists(asset):
            return True
        
        metadata = self._load_metadata(asset)
        if not metadata:
            return True
        
        created_at = datetime.fromisoformat(metadata.get("created_at", "2000-01-01"))
        days_since_creation = (datetime.utcnow() - created_at).days
        
        return days_since_creation >= full_train_interval_days
    
    def get_last_update_time(self, asset: str) -> Optional[datetime]:
        """Get the last update time for a model"""
        metadata = self._load_metadata(asset)
        
        if not metadata:
            return None
        
        updated_at = metadata.get("updated_at")
        if updated_at:
            return datetime.fromisoformat(updated_at)
        
        return None


class MultiAssetModelManager:
    """
    Manager for multiple asset models (BTC, ETH, TAO).
    """
    
    def __init__(
        self,
        base_path: str = "./models/lstm/",
        model_prefix: str = "lstm_price_predictor",
        keep_old_models: int = 5,
    ):
        self.manager = ModelManager(
            base_path=base_path,
            model_prefix=model_prefix,
            keep_old_models=keep_old_models,
        )
        
        # Loaded models cache
        self.models: Dict[str, nn.Module] = {}
        self.preprocessors: Dict[str, DataPreprocessor] = {}
    
    def load_all_models(self, device: str = "cuda") -> Dict[str, Tuple]:
        """
        Load all available models.
        
        Returns:
            Dictionary mapping asset to (model, preprocessor, metadata) tuple
        """
        result = {}
        
        for model_info in self.manager.list_models():
            asset = model_info["asset"]
            try:
                model, preprocessor, metadata = self.manager.load_model(asset, device)
                self.models[asset] = model
                self.preprocessors[asset] = preprocessor
                result[asset] = (model, preprocessor, metadata)
            except Exception as e:
                bt.logging.error(f"Failed to load model for {asset}: {e}")
        
        return result
    
    def get_model(self, asset: str) -> Optional[nn.Module]:
        """Get loaded model for asset"""
        return self.models.get(asset)
    
    def get_preprocessor(self, asset: str) -> Optional[DataPreprocessor]:
        """Get preprocessor for asset"""
        return self.preprocessors.get(asset)
    
    def save_model(
        self,
        model: nn.Module,
        preprocessor: DataPreprocessor,
        asset: str,
        **kwargs,
    ):
        """Save model and update cache"""
        self.manager.save_model(model, preprocessor, asset, **kwargs)
        self.models[asset] = model
        self.preprocessors[asset] = preprocessor

