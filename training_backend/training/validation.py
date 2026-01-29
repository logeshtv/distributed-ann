"""Walk-forward validation for time series models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from .trainer import Trainer


@dataclass
class ValidationResult:
    """Result from a single validation window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_loss: float
    val_loss: float
    metrics: Dict[str, float]


class WalkForwardValidator:
    """
    Walk-forward validation for time series.
    
    Trains on rolling windows and tests on subsequent periods
    to simulate real trading conditions.
    """
    
    def __init__(
        self,
        model_class: type,
        model_config: dict,
        loss_fn: nn.Module,
        train_years: float = 2.0,
        test_months: int = 3,
        n_windows: int = 10,
        device: str = "cuda"
    ):
        self.model_class = model_class
        self.model_config = model_config
        self.loss_fn = loss_fn
        self.train_days = int(train_years * 365)
        self.test_days = test_months * 30
        self.n_windows = n_windows
        self.device = device
        self.results: List[ValidationResult] = []
    
    def create_windows(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create train/test splits for walk-forward validation."""
        df = df.sort_values('timestamp')
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        windows = []
        current_start = min_date
        
        for i in range(self.n_windows):
            train_end = current_start + timedelta(days=self.train_days)
            test_end = train_end + timedelta(days=self.test_days)
            
            if test_end > max_date:
                break
            
            train_df = df[(df['timestamp'] >= current_start) & (df['timestamp'] < train_end)]
            test_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)]
            
            if len(train_df) > 100 and len(test_df) > 10:
                windows.append((train_df, test_df))
            
            current_start = train_end
        
        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows
    
    def validate_window(
        self,
        window_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50
    ) -> ValidationResult:
        """Train and validate on a single window."""
        model = self.model_class(**self.model_config).to(self.device)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            loss_fn=self.loss_fn,
            device=self.device
        )
        
        history = trainer.train(epochs=epochs, patience=10, save_best=False)
        val_metrics = trainer.validate()
        
        result = ValidationResult(
            window_id=window_id,
            train_start=datetime.now(),
            train_end=datetime.now(),
            test_start=datetime.now(),
            test_end=datetime.now(),
            train_loss=history['train_loss'][-1] if history['train_loss'] else 0,
            val_loss=val_metrics['val_loss'],
            metrics=val_metrics
        )
        
        return result
    
    def run_validation(
        self,
        df: pd.DataFrame,
        create_loader_fn,
        epochs: int = 50
    ) -> Dict[str, float]:
        """Run full walk-forward validation."""
        windows = self.create_windows(df)
        
        all_metrics = []
        for i, (train_df, test_df) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}")
            
            train_loader = create_loader_fn(train_df, shuffle=True)
            test_loader = create_loader_fn(test_df, shuffle=False)
            
            result = self.validate_window(i, train_loader, test_loader, epochs)
            self.results.append(result)
            all_metrics.append(result.metrics)
            
            logger.info(f"Window {i+1} - Val Loss: {result.val_loss:.4f}")
        
        # Aggregate metrics
        aggregated = {}
        if all_metrics:
            for key in all_metrics[0]:
                values = [m[key] for m in all_metrics]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
        
        aggregated['n_windows'] = len(windows)
        
        return aggregated
    
    def get_summary(self) -> pd.DataFrame:
        """Get validation results summary."""
        if not self.results:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'window': r.window_id,
                'train_loss': r.train_loss,
                'val_loss': r.val_loss,
                **r.metrics
            }
            for r in self.results
        ])
