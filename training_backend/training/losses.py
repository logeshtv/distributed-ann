"""Multi-task loss functions for trading model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Supports:
    - Price prediction (MSE/Huber)
    - Direction classification (CrossEntropy)
    - Position recommendation (CrossEntropy)
    - Volatility prediction (MSE)
    - Risk signal (MSE)
    """
    
    def __init__(
        self,
        price_weight: float = 0.30,
        direction_weight: float = 0.20,
        position_weight: float = 0.20,
        volatility_weight: float = 0.15,
        risk_weight: float = 0.10,
        confidence_weight: float = 0.05,
        use_huber: bool = True,
        huber_delta: float = 0.02
    ):
        super().__init__()
        
        self.weights = {
            'price': price_weight,
            'direction': direction_weight,
            'position': position_weight,
            'volatility': volatility_weight,
            'risk': risk_weight,
            'confidence': confidence_weight
        }
        
        self.huber = nn.HuberLoss(delta=huber_delta) if use_huber else nn.MSELoss()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions dict
            targets: Target values dict
            
        Returns:
            Dict with total loss and component losses
        """
        losses = {}
        total_loss = 0.0
        
        # Price prediction losses
        for key in predictions:
            if key.startswith('price_'):
                horizon = key.replace('price_', '')
                target_key = f'target_return_{horizon}'
                if target_key in targets:
                    loss = self.huber(predictions[key], targets[target_key])
                    losses[f'loss_{key}'] = loss
                    total_loss += loss * self.weights['price'] / 3
        
        # Direction classification
        if 'direction_logits' in predictions and 'target_direction' in targets:
            loss = self.ce(predictions['direction_logits'], targets['target_direction'].long())
            losses['loss_direction'] = loss
            total_loss += loss * self.weights['direction']
        
        # Position recommendation
        if 'position_logits' in predictions and 'target_position' in targets:
            loss = self.ce(predictions['position_logits'], targets['target_position'].long())
            losses['loss_position'] = loss
            total_loss += loss * self.weights['position']
        
        # Volatility prediction
        if 'volatility' in predictions and 'target_volatility' in targets:
            loss = self.mse(predictions['volatility'], targets['target_volatility'])
            losses['loss_volatility'] = loss
            total_loss += loss * self.weights['volatility']
        
        # Risk signal
        if 'risk_signal' in predictions and 'target_risk' in targets:
            loss = self.mse(predictions['risk_signal'], targets['target_risk'])
            losses['loss_risk'] = loss
            total_loss += loss * self.weights['risk']
        
        losses['total_loss'] = total_loss
        return losses


class TradingLoss(nn.Module):
    """Simplified trading loss for returns prediction."""
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        pred_return: torch.Tensor,
        pred_direction: torch.Tensor,
        target_return: torch.Tensor,
        target_direction: torch.Tensor
    ) -> torch.Tensor:
        mse_loss = self.mse(pred_return, target_return)
        ce_loss = self.ce(pred_direction, target_direction.long())
        return self.alpha * mse_loss + (1 - self.alpha) * ce_loss


class DirectionalLoss(nn.Module):
    """Loss that penalizes wrong direction predictions more heavily."""
    
    def __init__(self, direction_penalty: float = 2.0):
        super().__init__()
        self.direction_penalty = direction_penalty
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss = self.mse(pred, target)
        wrong_direction = (pred * target) < 0
        penalty = torch.where(wrong_direction, self.direction_penalty, 1.0)
        return (base_loss * penalty).mean()
