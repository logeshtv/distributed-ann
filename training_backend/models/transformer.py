"""
Transformer Encoder for Trading.

Multi-head self-attention with positional encoding
for capturing global patterns in market data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time series.
    
    Adds position information to input embeddings
    to help the model understand temporal order.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Positionally encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    
    Uses learned embeddings instead of fixed sinusoidal patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.pe.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block.
    
    Contains:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize encoder block.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Attention mask
            key_padding_mask: Padding mask
            
        Returns:
            Encoded tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Full Transformer encoder for time series.
    
    Stack of encoder blocks with positional encoding
    for modeling global dependencies in sequences.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_len: int = 5000,
        learnable_pe: bool = False
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            input_size: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout probability
            activation: Activation function
            max_len: Maximum sequence length
            learnable_pe: Use learnable positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional encoding
        if learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self.output_size = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            mask: Attention mask
            key_padding_mask: Padding mask
            
        Returns:
            Encoded tensor of shape (batch, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_proj(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through encoder layers
        for layer in self.layers:
            x = layer(x, mask, key_padding_mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def get_last_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the last timestep output."""
        output = self.forward(x)
        return output[:, -1, :]
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization."""
        # Project input
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Get attention from first layer
        attn_output, attn_weights = self.layers[0].self_attn(x, x, x)
        return attn_weights


class TemporalConvTransformer(nn.Module):
    """
    Combines 1D convolutions with Transformer.
    
    Uses convolutions for local pattern extraction
    and Transformer for global dependencies.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, d_model // len(kernel_sizes), ks, padding=ks // 2),
                nn.BatchNorm1d(d_model // len(kernel_sizes)),
                nn.GELU()
            )
            for ks in kernel_sizes
        ])
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            input_size=d_model,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.output_size = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with conv + transformer."""
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)  # (batch, d_model, seq_len)
        
        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Transformer
        x = self.transformer(x)
        
        return x


# Example usage
if __name__ == "__main__":
    # Test Transformer encoder
    batch_size = 32
    seq_len = 60
    input_size = 64
    d_model = 256
    
    model = TransformerEncoder(
        input_size=input_size,
        d_model=d_model,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Last output shape: {model.get_last_output(x).shape}")
    
    # Test temporal conv transformer
    tct = TemporalConvTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=8,
        num_layers=2
    )
    
    output_tct = tct(x)
    print(f"TCT output shape: {output_tct.shape}")
