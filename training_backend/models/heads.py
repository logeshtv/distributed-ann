"""
Multi-task output heads for trading predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class PriceHead(nn.Module):
    """Price prediction head for multiple horizons."""
    
    def __init__(self, input_dim: int, horizons: List[int] = [1, 4, 24], dropout: float = 0.1):
        super().__init__()
        self.horizons = horizons
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1)
            ) for _ in horizons
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f'price_{h}h': head(x).squeeze(-1) for h, head in zip(self.horizons, self.heads)}


class DirectionHead(nn.Module):
    """Direction classification head (up/neutral/down)."""
    
    def __init__(self, input_dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PositionHead(nn.Module):
    """Position recommendation head (buy/hold/sell)."""
    
    def __init__(self, input_dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'position_logits': self.classifier(x),
            'confidence': self.confidence(x).squeeze(-1)
        }


class VolatilityHead(nn.Module):
    """Volatility prediction head."""
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x).squeeze(-1)


class RiskHead(nn.Module):
    """Risk signal head."""
    
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x).squeeze(-1)


class MultiTaskHead(nn.Module):
    """Combined multi-task output head."""
    
    def __init__(
        self,
        input_dim: int,
        price_horizons: List[int] = [1, 4, 24],
        direction_classes: int = 3,
        position_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.price_head = PriceHead(input_dim, price_horizons, dropout)
        self.direction_head = DirectionHead(input_dim, direction_classes, dropout)
        self.position_head = PositionHead(input_dim, position_classes, dropout)
        self.volatility_head = VolatilityHead(input_dim, dropout)
        self.risk_head = RiskHead(input_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        outputs.update(self.price_head(x))
        outputs['direction_logits'] = self.direction_head(x)
        outputs.update(self.position_head(x))
        outputs['volatility'] = self.volatility_head(x)
        outputs['risk_signal'] = self.risk_head(x)
        return outputs
