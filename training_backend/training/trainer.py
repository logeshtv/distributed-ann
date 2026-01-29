"""Training loop and utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, Optional, Callable
from pathlib import Path
import time
from loguru import logger

from .losses import MultiTaskLoss


class Trainer:
    """Training loop for trading model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.loss_fn = loss_fn or MultiTaskLoss()
        
        # Use provided optimizer or default to AdamW
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
        # Use provided scheduler or default to CosineAnnealingWarmRestarts
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
            
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, accumulation_steps: int = 1) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        from tqdm import tqdm
        
        self.optimizer.zero_grad()
        
        # Add progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")
        
        for i, batch in pbar:
            features = batch['features'].to(self.device)
            targets_tensor = batch['targets'].to(self.device)
            
            # Convert targets tensor to dict expected by loss function
            targets = {
                'target_return': targets_tensor[:, 0],
                'target_direction': targets_tensor[:, 1].long()
            }
            
            outputs = self.model(features)
            losses = self.loss_fn(outputs, targets)
            loss = losses['total_loss']
            
            # Handle case where loss is float (from empty loss)
            if isinstance(loss, float):
                loss = torch.tensor(loss, device=self.device, requires_grad=True)
            
            # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            # Step optimizer every 'accumulation_steps' batches
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Accumulate unscaled loss for logging
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct_direction = 0
        total_samples = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            targets_tensor = batch['targets'].to(self.device)
            
            # Convert targets tensor to dict
            targets = {
                'target_return': targets_tensor[:, 0],
                'target_direction': targets_tensor[:, 1].long()
            }
            
            outputs = self.model(features)
            losses = self.loss_fn(outputs, targets)
            loss = losses['total_loss']
            
            if isinstance(loss, float):
                total_loss += loss
            else:
                total_loss += loss.item()
            
            if 'direction_logits' in outputs:
                pred_dir = outputs['direction_logits'].argmax(dim=-1)
                correct_direction += (pred_dir == targets['target_direction']).sum().item()
                total_samples += len(pred_dir)
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'direction_accuracy': correct_direction / max(total_samples, 1)
        }
    
    def train(
        self,
        epochs: int = 100,
        patience: int = 15,
        save_best: bool = True,
        accumulation_steps: int = 1
    ) -> Dict[str, list]:
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs with accumulation steps: {accumulation_steps}")
        
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = self.train_epoch(accumulation_steps=accumulation_steps)
            val_metrics = self.validate()
            val_loss = val_metrics['val_loss']
            
            # Step scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Acc: {val_metrics['direction_accuracy']:.2%} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_best:
                    self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path}")
