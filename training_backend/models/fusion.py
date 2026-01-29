"""
Fusion Layer for combining xLSTM and Transformer outputs.

Implements attention-based fusion and gating mechanisms
to combine sequential and global pattern representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FusionLayer(nn.Module):
    """
    Attention-based fusion of multiple encoder outputs.
    
    Combines xLSTM (sequential) and Transformer (global)
    representations using learned attention weights.
    """
    
    def __init__(
        self,
        xlstm_dim: int,
        transformer_dim: int,
        output_dim: int = 256,
        dropout: float = 0.2,
        fusion_method: str = "attention"
    ):
        """
        Initialize fusion layer.
        
        Args:
            xlstm_dim: xLSTM output dimension
            transformer_dim: Transformer output dimension
            output_dim: Fused output dimension
            dropout: Dropout probability
            fusion_method: 'attention', 'gate', or 'concat'
        """
        super().__init__()
        
        self.xlstm_dim = xlstm_dim
        self.transformer_dim = transformer_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # Projection layers
        self.xlstm_proj = nn.Linear(xlstm_dim, output_dim)
        self.transformer_proj = nn.Linear(transformer_dim, output_dim)
        
        if fusion_method == "attention":
            # Cross-attention between encoders
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(output_dim)
            
        elif fusion_method == "gate":
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
            
        elif fusion_method == "concat":
            # Simple concatenation with projection
            self.concat_proj = nn.Linear(output_dim * 2, output_dim)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        xlstm_output: torch.Tensor,
        transformer_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse xLSTM and Transformer outputs.
        
        Args:
            xlstm_output: xLSTM output (batch, seq_len, xlstm_dim)
            transformer_output: Transformer output (batch, seq_len, transformer_dim)
            
        Returns:
            Fused output (batch, seq_len, output_dim)
        """
        # Project to common dimension
        xlstm_proj = self.xlstm_proj(xlstm_output)
        transformer_proj = self.transformer_proj(transformer_output)
        
        if self.fusion_method == "attention":
            # Cross-attention: xlstm attends to transformer
            attn_output, _ = self.cross_attention(
                xlstm_proj, transformer_proj, transformer_proj
            )
            fused = self.attention_norm(xlstm_proj + attn_output)
            
        elif self.fusion_method == "gate":
            # Gating: learn to mix representations
            concat = torch.cat([xlstm_proj, transformer_proj], dim=-1)
            gate = self.gate(concat)
            fused = gate * xlstm_proj + (1 - gate) * transformer_proj
            
        elif self.fusion_method == "concat":
            # Simple concatenation
            concat = torch.cat([xlstm_proj, transformer_proj], dim=-1)
            fused = self.concat_proj(concat)
            
        else:
            # Default: average
            fused = (xlstm_proj + transformer_proj) / 2
        
        # Apply output layers
        output = self.output_layers(fused)
        
        return output


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion with multiple levels.
    
    Processes inputs at different temporal scales
    before final fusion.
    """
    
    def __init__(
        self,
        input_dims: list,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_levels: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize hierarchical fusion.
        
        Args:
            input_dims: List of input dimensions for each encoder
            hidden_dim: Hidden layer dimension
            output_dim: Final output dimension
            num_levels: Number of fusion levels
            dropout: Dropout probability
        """
        super().__init__()
        
        # Input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Hierarchical fusion layers
        self.fusion_layers = nn.ModuleList()
        for i in range(num_levels):
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * len(input_dims), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim)
                )
            )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical fusion of multiple inputs.
        
        Args:
            *inputs: Variable number of input tensors
            
        Returns:
            Fused output tensor
        """
        # Project all inputs
        projected = [
            proj(x) for proj, x in zip(self.input_projs, inputs)
        ]
        
        # Hierarchical fusion
        for fusion_layer in self.fusion_layers:
            concat = torch.cat(projected, dim=-1)
            fused = fusion_layer(concat)
            # Update for next level
            projected = [fused for _ in projected]
        
        # Final projection
        output = self.output_proj(projected[0])
        
        return output


class TemporalFusion(nn.Module):
    """
    Temporal fusion for sequence outputs.
    
    Aggregates sequence to fixed representation
    using attention pooling.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize temporal fusion.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate sequence to single representation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Aggregated tensor of shape (batch, output_dim)
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled, _ = self.attention(query, x, x)
        pooled = pooled.squeeze(1)  # (batch, input_dim)
        
        # Project to output dimension
        output = self.output_proj(pooled)
        
        return output


# Example usage
if __name__ == "__main__":
    batch_size = 32
    seq_len = 60
    xlstm_dim = 256
    transformer_dim = 256
    output_dim = 256
    
    # Create sample inputs
    xlstm_output = torch.randn(batch_size, seq_len, xlstm_dim)
    transformer_output = torch.randn(batch_size, seq_len, transformer_dim)
    
    # Test fusion layer
    fusion = FusionLayer(
        xlstm_dim=xlstm_dim,
        transformer_dim=transformer_dim,
        output_dim=output_dim,
        fusion_method="attention"
    )
    
    fused = fusion(xlstm_output, transformer_output)
    print(f"Fused output shape: {fused.shape}")
    
    # Test temporal fusion
    temporal = TemporalFusion(
        input_dim=output_dim,
        output_dim=output_dim
    )
    
    aggregated = temporal(fused)
    print(f"Aggregated output shape: {aggregated.shape}")
