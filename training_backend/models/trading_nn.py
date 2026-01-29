"""
Main Trading Neural Network - xLSTM-Transformer Hybrid.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .xlstm import xLSTM
from .transformer import TransformerEncoder
from .fusion import FusionLayer, TemporalFusion
from .position_encoder import PositionStateEncoder, StateIntegrationModule
from .heads import MultiTaskHead


class TradingNeuralNetwork(nn.Module):
    """
    xLSTM-Transformer Hybrid Network for Trading.
    
    Architecture:
    1. Dual-path encoding (xLSTM + Transformer)
    2. Fusion layer to combine representations
    3. Position state integration
    4. Multi-task output heads
    """
    
    def __init__(
        self,
        input_dim: int = 60,
        xlstm_hidden: int = 512,
        xlstm_layers: int = 2,
        transformer_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        fusion_dim: int = 256,
        position_dim: int = 320,
        max_positions: int = 20,
        price_horizons: List[int] = [1, 4, 24],
        dropout: float = 0.3,
        use_position_state: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_position_state = use_position_state
        
        # xLSTM encoder
        self.xlstm = xLSTM(
            input_size=input_dim,
            hidden_size=xlstm_hidden,
            num_layers=xlstm_layers,
            dropout=dropout
        )
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            input_size=input_dim,
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout / 3
        )
        
        # Fusion layer
        self.fusion = FusionLayer(
            xlstm_dim=self.xlstm.output_size,
            transformer_dim=self.transformer.output_size,
            output_dim=fusion_dim,
            dropout=dropout / 2
        )
        
        # Position state encoder
        if use_position_state:
            self.position_encoder = PositionStateEncoder(
                embedding_dim=position_dim,
                max_positions=max_positions
            )
            self.state_integration = StateIntegrationModule(
                market_dim=fusion_dim,
                state_dim=position_dim,
                output_dim=fusion_dim
            )
        
        # Temporal aggregation
        self.temporal_fusion = TemporalFusion(
            input_dim=fusion_dim,
            output_dim=fusion_dim
        )
        
        # Multi-task output heads
        self.heads = MultiTaskHead(
            input_dim=fusion_dim,
            price_horizons=price_horizons,
            dropout=dropout / 2
        )
    
    def forward(
        self,
        x: torch.Tensor,
        position_features: Optional[torch.Tensor] = None,
        fund_status: Optional[torch.Tensor] = None,
        position_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Market features (batch, seq_len, input_dim)
            position_features: Position features (batch, max_positions, position_features)
            fund_status: Fund status (batch, fund_features)
            position_mask: Position mask (batch, max_positions)
        
        Returns:
            Dictionary of predictions
        """
        # Dual-path encoding
        xlstm_output, _ = self.xlstm(x)
        transformer_output = self.transformer(x)
        
        # Fusion
        fused = self.fusion(xlstm_output, transformer_output)
        
        # Position state integration
        if self.use_position_state and position_features is not None:
            state_embedding = self.position_encoder(
                position_features, fund_status, position_mask
            )
            fused = self.state_integration(fused, state_embedding)
        
        # Temporal aggregation
        aggregated = self.temporal_fusion(fused)
        
        # Multi-task predictions
        outputs = self.heads(aggregated)
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified prediction without position state."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate embedding for analysis."""
        xlstm_output, _ = self.xlstm(x)
        transformer_output = self.transformer(x)
        fused = self.fusion(xlstm_output, transformer_output)
        return self.temporal_fusion(fused)


def create_model(config: dict) -> TradingNeuralNetwork:
    """Create model from config dictionary."""
    return TradingNeuralNetwork(
        input_dim=config.get('input_dim', 60),
        xlstm_hidden=config.get('xlstm_hidden', 512),
        xlstm_layers=config.get('xlstm_layers', 2),
        transformer_dim=config.get('transformer_dim', 256),
        transformer_heads=config.get('transformer_heads', 8),
        transformer_layers=config.get('transformer_layers', 3),
        fusion_dim=config.get('fusion_dim', 256),
        dropout=config.get('dropout', 0.3)
    )


if __name__ == "__main__":
    # Test model
    model = TradingNeuralNetwork(input_dim=60, use_position_state=False)
    x = torch.randn(32, 60, 60)
    outputs = model(x)
    print("Model outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
