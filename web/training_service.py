"""Training service for managing ML training jobs."""

import asyncio
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import json

from loguru import logger


@dataclass
class TrainingProgress:
    """Training progress state."""
    status: str = "idle"  # idle, preparing, training, completed, failed, cancelled
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    epoch_time: float = 0.0
    eta_seconds: float = 0.0
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    best_val_loss: float = float('inf')
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "train_loss": round(self.train_loss, 4),
            "val_loss": round(self.val_loss, 4),
            "accuracy": round(self.accuracy * 100, 2),
            "epoch_time": round(self.epoch_time, 1),
            "eta_seconds": round(self.eta_seconds, 0),
            "message": self.message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "best_val_loss": round(self.best_val_loss, 4) if self.best_val_loss != float('inf') else None,
            "model_path": self.model_path,
            "progress_percent": self._calculate_progress()
        }
    
    def _calculate_progress(self) -> float:
        if self.total_epochs == 0:
            return 0
        epoch_progress = (self.current_epoch - 1) / self.total_epochs if self.current_epoch > 0 else 0
        if self.total_batches > 0 and self.current_batch > 0:
            batch_progress = (self.current_batch / self.total_batches) / self.total_epochs
            return min((epoch_progress + batch_progress) * 100, 100)
        return epoch_progress * 100


class TrainingService:
    """Manages training jobs and broadcasts progress."""
    
    def __init__(self):
        self.progress = TrainingProgress()
        self.cancel_requested = False
        self.training_thread: Optional[threading.Thread] = None
        self.websocket_clients: list = []
        self.logs: list = []
        self._lock = threading.Lock()
    
    @property
    def is_training(self) -> bool:
        return self.progress.status in ["preparing", "training"]
    
    def add_log(self, message: str):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        with self._lock:
            self.logs.append(log_entry)
            # Keep last 100 logs
            if len(self.logs) > 100:
                self.logs = self.logs[-100:]
    
    def get_logs(self) -> list:
        """Get all logs."""
        with self._lock:
            return list(self.logs)
    
    def request_cancel(self):
        """Request training cancellation."""
        self.cancel_requested = True
        self.add_log("⚠️ Cancellation requested...")
    
    async def broadcast_progress(self):
        """Broadcast progress to all connected WebSocket clients."""
        if not self.websocket_clients:
            return
        
        data = json.dumps({
            "type": "progress",
            "data": self.progress.to_dict(),
            "logs": self.get_logs()[-20:]  # Last 20 logs
        })
        
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_clients.remove(ws)
    
    def start_training(
        self,
        data_path: str,
        epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 5e-4,
        patience: int = 15,
        sequence_length: int = 60,
        broadcast_callback: Optional[Callable] = None
    ):
        """Start training in a background thread."""
        if self.is_training:
            raise RuntimeError("Training already in progress")
        
        self.cancel_requested = False
        self.logs = []
        self.progress = TrainingProgress(
            status="preparing",
            total_epochs=epochs,
            started_at=datetime.now().isoformat(),
            message="Preparing data..."
        )
        
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(data_path, epochs, batch_size, learning_rate, patience, sequence_length, broadcast_callback),
            daemon=True
        )
        self.training_thread.start()
    
    def _run_training(
        self,
        data_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        sequence_length: int,
        broadcast_callback: Optional[Callable]
    ):
        """Run training loop (called in background thread)."""
        import sys
        from pathlib import Path
        
        # Add project to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        try:
            self.add_log("🚀 Starting training pipeline...")
            self.add_log(f"📊 Config: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
            
            # Import dependencies
            import torch
            from config.settings import settings
            from config.model_config import ModelConfig
            from data.features import FeatureEngineer
            from data.dataset import TradingDataset, create_data_loaders
            from models.trading_nn import TradingNeuralNetwork
            from training.trainer import Trainer
            from training.losses import MultiTaskLoss
            
            import pandas as pd
            import numpy as np
            import warnings
            warnings.filterwarnings('ignore')
            
            # Prepare data
            self._update_progress(message="Loading data files...")
            if broadcast_callback:
                asyncio.run(broadcast_callback())
            
            data_path = Path(data_path)
            all_data = []
            
            for subdir in ["stocks", "crypto"]:
                dir_path = data_path / subdir
                if dir_path.exists():
                    for parquet_file in dir_path.glob("*.parquet"):
                        self.add_log(f"📁 Loading {parquet_file.name}...")
                        df = pd.read_parquet(parquet_file)
                        all_data.append(df)
            
            if not all_data and data_path.is_file():
                all_data.append(pd.read_parquet(data_path))
            
            if not all_data:
                raise RuntimeError(f"No parquet files found in {data_path}")
            
            df = pd.concat(all_data, ignore_index=True)
            self.add_log(f"📈 Loaded {len(df):,} rows total")
            
            # Process timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            else:
                raise ValueError("Data must have 'timestamp' column")
            
            # Feature engineering
            self._update_progress(message="Engineering features...")
            if broadcast_callback:
                asyncio.run(broadcast_callback())
            
            fe = FeatureEngineer()
            processed = []
            symbols = df['symbol'].unique()
            self.add_log(f"⚙️ Processing {len(symbols)} symbols...")
            
            for i, symbol in enumerate(symbols):
                if self.cancel_requested:
                    self._update_progress(status="cancelled", message="Training cancelled by user")
                    return
                
                symbol_df = df[df['symbol'] == symbol].copy()
                symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
                
                if len(symbol_df) < 500:
                    continue
                
                try:
                    symbol_df = fe.add_all_features(symbol_df)
                    processed.append(symbol_df)
                except Exception:
                    continue
                
                if (i + 1) % 50 == 0:
                    self.add_log(f"   Processed {i+1}/{len(symbols)} symbols")
            
            if not processed:
                raise RuntimeError("No symbols processed successfully")
            
            df = pd.concat(processed, ignore_index=True)
            self.add_log(f"✅ Feature engineering complete: {len(df):,} rows")
            
            # Create labels
            self._update_progress(message="Creating labels...")
            
            for h in [1, 4, 24]:
                df[f'target_return_{h}'] = df.groupby('symbol')['close'].transform(
                    lambda x: x.shift(-h) / x - 1
                )
                df[f'target_direction_{h}'] = 1
                df.loc[df[f'target_return_{h}'] > 0.005, f'target_direction_{h}'] = 2
                df.loc[df[f'target_return_{h}'] < -0.005, f'target_direction_{h}'] = 0
            
            df['target_return'] = df['target_return_1']
            df['target_direction'] = df['target_direction_1']
            
            target_cols = ['target_return', 'target_direction']
            df = df.dropna(subset=target_cols)
            
            # Get feature columns
            exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
            exclude += [col for col in df.columns if col.startswith('target_')]
            feature_cols = [c for c in df.columns if c not in exclude]
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
            
            self.add_log(f"📊 Using {len(feature_cols)} features")
            
            # Clean data
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df[feature_cols] = df[feature_cols].fillna(0)
            df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            
            # Split data
            n = len(df)
            train_end = int(n * 0.80)
            val_end = int(n * 0.90)
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            self.add_log(f"📊 Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
            
            # Normalize
            self._update_progress(message="Normalizing features...")
            for col in feature_cols:
                mean = train_df[col].mean()
                std = train_df[col].std()
                if pd.notna(std) and std > 0:
                    train_df.loc[:, col] = (train_df[col] - mean) / std
                    val_df.loc[:, col] = (val_df[col] - mean) / std
                    test_df.loc[:, col] = (test_df[col] - mean) / std
            
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            # Create datasets
            config = ModelConfig(
                num_features=len(feature_cols),
                sequence_length=sequence_length,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            train_dataset = TradingDataset.from_dataframe(
                train_df, feature_cols, target_cols, config.sequence_length
            )
            val_dataset = TradingDataset.from_dataframe(
                val_df, feature_cols, target_cols, config.sequence_length
            )
            
            # Detect environment and maximize CPU utilization
            import os
            is_container = os.path.exists('/.dockerenv') or os.environ.get('RAILWAY_ENVIRONMENT')
            
            if is_container:
                num_workers = 0  # Container: avoid shared memory issues
            else:
                # Local Mac: use most CPU cores for data loading
                cpu_count = os.cpu_count() or 8
                num_workers = max(1, cpu_count - 2)  # Leave 2 cores for system
            
            self.add_log(f"🔧 DataLoader workers: {num_workers} (of {os.cpu_count()} CPUs)")
            
            loaders = create_data_loaders(
                train_dataset, val_dataset, None,
                batch_size=config.batch_size,
                num_workers=num_workers
            )
            
            # Create model - Optimize for M4 Pro MPS
            if torch.cuda.is_available():
                device = "cuda"
                self.add_log(f"🖥️ Using NVIDIA CUDA GPU")
            elif torch.backends.mps.is_available():
                device = "mps"
                # Enable MPS fallback for unsupported ops
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable cache limit
                self.add_log(f"🖥️ Using Apple M4 Pro GPU (MPS) - Full Memory Mode")
            else:
                device = "cpu"
                self.add_log(f"🖥️ Using CPU (GPU not available)")
            
            model = TradingNeuralNetwork(
                input_dim=config.num_features,
                xlstm_hidden=512,
                xlstm_layers=3,
                transformer_dim=256,
                transformer_heads=8,
                transformer_layers=3,
                use_position_state=False,
                dropout=0.3
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            self.add_log(f"🧠 Model parameters: {total_params:,}")
            
            # Compile model for 2x speedup (PyTorch 2.0+)
            if device != "mps":  # torch.compile not fully supported on MPS yet
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    self.add_log("⚡ Model compiled with torch.compile (2x speedup)")
                except Exception as e:
                    self.add_log(f"ℹ️ torch.compile skipped: {e}")
            else:
                self.add_log("⚡ Using MPS GPU acceleration (native Metal)")
            
            # Setup training
            loss_fn = MultiTaskLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            trainer = Trainer(
                model=model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                loss_fn=loss_fn,
                device=device,
                checkpoint_dir=settings.MODELS_DIR,
                optimizer=optimizer,
                scheduler=scheduler
            )
            
            # Calculate accumulation
            target_batch_size = 2048
            accumulation_steps = max(1, target_batch_size // batch_size)
            
            self.add_log(f"🎯 Effective batch: {batch_size * accumulation_steps}")
            
            # Training loop with progress callbacks
            self._update_progress(status="training", message="Training started!")
            if broadcast_callback:
                asyncio.run(broadcast_callback())
            
            total_batches = len(trainer.train_loader)
            self.progress.total_batches = total_batches
            
            epoch_times = []
            
            for epoch in range(epochs):
                if self.cancel_requested:
                    self._update_progress(status="cancelled", message="Training cancelled by user")
                    self.add_log("⚠️ Training cancelled")
                    return
                
                epoch_start = time.time()
                self.progress.current_epoch = epoch + 1
                self.progress.current_batch = 0
                
                # Train epoch with batch callbacks
                model.train()
                total_loss = 0.0
                trainer.optimizer.zero_grad()
                
                for i, batch_data in enumerate(trainer.train_loader):
                    if self.cancel_requested:
                        self._update_progress(status="cancelled", message="Training cancelled")
                        return
                    
                    # Async transfer to GPU (non_blocking allows CPU to continue)
                    features = batch_data['features'].to(device, non_blocking=True)
                    targets_tensor = batch_data['targets'].to(device, non_blocking=True)
                    
                    targets = {
                        'target_return': targets_tensor[:, 0],
                        'target_direction': targets_tensor[:, 1].long()
                    }
                    
                    outputs = model(features)
                    losses = loss_fn(outputs, targets)
                    loss = losses['total_loss']
                    
                    if isinstance(loss, float):
                        loss = torch.tensor(loss, device=device, requires_grad=True)
                    
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()
                    
                    current_loss = loss.item() * accumulation_steps
                    total_loss += current_loss
                    
                    # Update batch progress
                    self.progress.current_batch = i + 1
                    self.progress.train_loss = total_loss / (i + 1)
                    
                    # Broadcast every 10 batches
                    if i % 10 == 0 and broadcast_callback:
                        asyncio.run(broadcast_callback())
                
                train_loss = total_loss / len(trainer.train_loader)
                
                # Validation
                val_metrics = trainer.validate()
                val_loss = val_metrics['val_loss']
                accuracy = val_metrics['direction_accuracy']
                
                # Update scheduler
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                # Calculate ETA
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = epochs - (epoch + 1)
                eta = avg_epoch_time * remaining_epochs
                
                # Update progress
                self.progress.train_loss = train_loss
                self.progress.val_loss = val_loss
                self.progress.accuracy = accuracy
                self.progress.epoch_time = epoch_time
                self.progress.eta_seconds = eta
                
                # Log epoch
                eta_str = f"{int(eta//3600)}h {int((eta%3600)//60)}m" if eta > 3600 else f"{int(eta//60)}m {int(eta%60)}s"
                self.add_log(
                    f"📈 Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Acc: {accuracy:.1%} | ETA: {eta_str}"
                )
                
                # Save best model
                if val_loss < self.progress.best_val_loss:
                    self.progress.best_val_loss = val_loss
                    trainer.patience_counter = 0
                    trainer.save_checkpoint('best_model.pt')
                    self.add_log(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
                else:
                    trainer.patience_counter += 1
                    if trainer.patience_counter >= patience:
                        self.add_log(f"⏹️ Early stopping at epoch {epoch+1}")
                        break
                
                if broadcast_callback:
                    asyncio.run(broadcast_callback())
            
            # Training complete
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_path = settings.MODELS_DIR / f"production_model_{timestamp}.pt"
            torch.save(model.state_dict(), final_path)
            
            self.progress.model_path = str(final_path)
            self._update_progress(
                status="completed",
                message=f"Training complete! Model saved.",
                completed_at=datetime.now().isoformat()
            )
            self.add_log(f"✅ Training complete! Model: {final_path.name}")
            
            if broadcast_callback:
                asyncio.run(broadcast_callback())
            
        except Exception as e:
            self._update_progress(status="failed", message=f"Error: {str(e)}")
            self.add_log(f"❌ Training failed: {str(e)}")
            logger.exception("Training failed")
            if broadcast_callback:
                asyncio.run(broadcast_callback())
    
    def _update_progress(self, **kwargs):
        """Update progress fields."""
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)


# Global training service instance
training_service = TrainingService()
