"""
Training Pipeline for LSTM Price Prediction Model

Handles both full training and incremental updates.
Supports 3-output models for interval prediction (price, min, max).
"""

import os
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from precog.miners.models.logging_utils import logger as bt_logging


# Create a bt-like interface using our simple logger
class bt:
    logging = bt_logging


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """
    Trainer for LSTM price prediction model.
    
    Supports:
    - Full training from scratch
    - Incremental training (fine-tuning)
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Checkpoint saving
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
        lr_scheduler_step_size: int = 30,
        lr_scheduler_gamma: float = 0.1,
        checkpoint_dir: str = "./checkpoints/",
    ):
        """
        Initialize trainer.
        
        Args:
            model: LSTM model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
            gradient_clip: Max gradient norm
            early_stopping_patience: Epochs to wait before early stopping
            lr_scheduler_step_size: Steps between LR reduction
            lr_scheduler_gamma: Factor to reduce LR
            checkpoint_dir: Directory for checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=lr_scheduler_step_size,
            gamma=lr_scheduler_gamma,
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        
        # Best model state
        self.best_model_state = None
        self.best_val_loss = float("inf")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(batch_x)
            
            # Calculate loss
            loss = self.criterion(output, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output, _ = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
        checkpoint_frequency: int = 10,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress
            checkpoint_frequency: Save checkpoint every N epochs
        
        Returns:
            Training history
        """
        bt.logging.info(f"Starting training for {epochs} epochs on {self.device}")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            # Checkpoint
            if (epoch + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(epoch + 1)
            
            # Logging
            epoch_time = time.time() - epoch_start
            if verbose:
                bt.logging.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                bt.logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        bt.logging.success(f"Training completed in {total_time:.2f}s")
        bt.logging.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def incremental_train(
        self,
        train_loader: DataLoader,
        epochs: int = 5,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Incremental training (fine-tuning) with new data.
        
        Uses a smaller learning rate and fewer epochs than full training.
        
        Args:
            train_loader: Data loader with new data
            epochs: Number of epochs for incremental training
            learning_rate: Learning rate (smaller than full training)
            verbose: Whether to print progress
        
        Returns:
            Training history for this incremental update
        """
        # Adjust learning rate for incremental training
        if learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate
        
        bt.logging.info(f"Starting incremental training for {epochs} epochs")
        start_time = time.time()
        
        incremental_history = {
            "train_loss": [],
        }
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            incremental_history["train_loss"].append(train_loss)
            
            if verbose:
                bt.logging.trace(
                    f"Incremental Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f}"
                )
        
        total_time = time.time() - start_time
        bt.logging.info(f"Incremental training completed in {total_time:.2f}s")
        
        return incremental_history
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        
        filepath = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )
        torch.save(checkpoint, filepath)
        bt.logging.trace(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        
        bt.logging.info(f"Loaded checkpoint from {filepath}")
        return checkpoint["epoch"]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make prediction.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
        
        Returns:
            Predictions tensor
        """
        self.model.eval()
        x = x.to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(x)
        
        return output.cpu()
    
    def evaluate(
        self,
        val_loader: DataLoader,
        preprocessor: Any = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Supports both single-output (price only) and 3-output (price, min, max) models.
        
        Args:
            val_loader: Validation data loader
            preprocessor: Optional preprocessor for inverse transform
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                
                output, _ = self.model(batch_x)
                
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(batch_y.numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # If preprocessor provided, inverse transform to original scale
        if preprocessor is not None:
            predictions = preprocessor.inverse_transform_target(predictions)
            targets = preprocessor.inverse_transform_target(targets)
        
        # Check if this is a 3-output model (price, min, max)
        is_interval_model = predictions.ndim == 2 and predictions.shape[1] == 3
        
        if is_interval_model:
            # Extract components: [price, min, max]
            pred_price = predictions[:, 0]
            pred_min = predictions[:, 1]
            pred_max = predictions[:, 2]
            
            target_price = targets[:, 0]
            target_min = targets[:, 1]
            target_max = targets[:, 2]
            
            # Price metrics
            price_mse = np.mean((pred_price - target_price) ** 2)
            price_rmse = np.sqrt(price_mse)
            price_mae = np.mean(np.abs(pred_price - target_price))
            price_mape = np.mean(np.abs((target_price - pred_price) / target_price)) * 100
            
            # Min/Max metrics
            min_mae = np.mean(np.abs(pred_min - target_min))
            max_mae = np.mean(np.abs(pred_max - target_max))
            
            # Interval coverage: how often does predicted interval contain actual min/max
            interval_contains_min = np.mean(pred_min <= target_min)
            interval_contains_max = np.mean(pred_max >= target_max)
            
            # Interval width accuracy (how close is predicted width to actual width)
            pred_width = pred_max - pred_min
            actual_width = target_max - target_min
            width_mae = np.mean(np.abs(pred_width - actual_width))
            
            # Direction accuracy (based on price)
            if len(target_price) > 1:
                pred_direction = np.sign(pred_price[1:] - pred_price[:-1])
                actual_direction = np.sign(target_price[1:] - target_price[:-1])
                direction_accuracy = np.mean(pred_direction == actual_direction) * 100
            else:
                direction_accuracy = 0
            
            return {
                "mse": float(price_mse),
                "rmse": float(price_rmse),
                "mae": float(price_mae),
                "mape": float(price_mape),
                "direction_accuracy": float(direction_accuracy),
                # Interval-specific metrics
                "min_mae": float(min_mae),
                "max_mae": float(max_mae),
                "interval_contains_min": float(interval_contains_min * 100),
                "interval_contains_max": float(interval_contains_max * 100),
                "width_mae": float(width_mae),
            }
        else:
            # Single-output model (original behavior)
            # Flatten if needed
            if predictions.ndim == 2:
                predictions = predictions.flatten()
            if targets.ndim == 2:
                targets = targets.flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        # Direction accuracy
        if len(targets) > 1:
            pred_direction = np.sign(predictions[1:] - predictions[:-1])
            actual_direction = np.sign(targets[1:] - targets[:-1])
            direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        else:
            direction_accuracy = 0
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
        }


def create_trainer(
    model: nn.Module,
    config: dict,
) -> Trainer:
    """
    Factory function to create trainer from config.
    
    Args:
        model: LSTM model
        config: Training configuration
    
    Returns:
        Configured Trainer instance
    """
    return Trainer(
        model=model,
        device=config.get("device", "cuda"),
        learning_rate=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 1e-5),
        gradient_clip=config.get("gradient_clip", 1.0),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        lr_scheduler_step_size=config.get("lr_scheduler_step_size", 30),
        lr_scheduler_gamma=config.get("lr_scheduler_gamma", 0.1),
        checkpoint_dir=config.get("checkpoint_dir", "./checkpoints/"),
    )

