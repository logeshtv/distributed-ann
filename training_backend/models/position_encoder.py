"""
Position State Encoder - Encodes portfolio state into embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class PositionState:
    """Single position state."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    hold_duration: int
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.quantity, self.entry_price, self.current_price,
            self.unrealized_pnl, self.hold_duration,
            self.delta, self.gamma, self.theta, self.vega,
            (self.current_price - self.entry_price) / (self.entry_price + 1e-8),
            self.unrealized_pnl / (abs(self.entry_price * self.quantity) + 1e-8),
        ], dtype=torch.float32)


class PositionStateEncoder(nn.Module):
    """Encode portfolio positions into dense embeddings."""
    
    def __init__(
        self,
        embedding_dim: int = 320,
        max_positions: int = 20,
        position_features: int = 11,
        fund_status_features: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        
        hidden_dim = embedding_dim // 2
        self.position_encoder = nn.Sequential(
            nn.Linear(position_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        self.position_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.position_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.fund_encoder = nn.Sequential(
            nn.Linear(fund_status_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        
        self.empty_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
    
    def forward(
        self,
        position_features: torch.Tensor,
        fund_status: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = position_features.size(0)
        num_positions = position_features.size(1)
        
        position_encoded = self.position_encoder(position_features)
        
        if position_mask is None:
            position_mask = torch.ones(batch_size, num_positions, device=position_features.device)
        
        has_positions = position_mask.sum(dim=1) > 0
        query = self.position_query.expand(batch_size, -1, -1)
        
        position_aggregated, _ = self.position_attention(
            query, position_encoded, position_encoded,
            key_padding_mask=~position_mask.bool()
        )
        position_aggregated = position_aggregated.squeeze(1)
        
        position_aggregated = torch.where(
            has_positions.unsqueeze(-1),
            position_aggregated,
            self.empty_embedding.expand(batch_size, -1)
        )
        
        fund_encoded = self.fund_encoder(fund_status)
        combined = torch.cat([position_aggregated, fund_encoded], dim=-1)
        return self.combiner(combined)


class StateIntegrationModule(nn.Module):
    """Integrate position state with market features."""
    
    def __init__(self, market_dim: int, state_dim: int, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.market_proj = nn.Linear(market_dim, output_dim)
        self.state_proj = nn.Linear(state_dim, output_dim)
        self.cross_attention = nn.MultiheadAttention(output_dim, 4, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.Sigmoid())
        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim))
    
    def forward(self, market_features: torch.Tensor, state_embedding: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = market_features.shape
        market_proj = self.market_proj(market_features)
        state_proj = self.state_proj(state_embedding).unsqueeze(1).expand(-1, seq_len, -1)
        attn_output, _ = self.cross_attention(market_proj, state_proj, state_proj)
        gate = self.gate(torch.cat([market_proj, attn_output], dim=-1))
        return self.output(gate * market_proj + (1 - gate) * attn_output)
