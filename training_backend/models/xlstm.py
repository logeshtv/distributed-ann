"""
xLSTM (Extended Long Short-Term Memory) Implementation.

Based on Beck et al. (2024) - "xLSTM: Extended Long Short-Term Memory"
Key features:
- Exponential gating for stable gradients
- Scalar memory cells with higher capacity
- Better performance on long sequences (>2000 timesteps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class xLSTMCell(nn.Module):
    """
    xLSTM Cell with exponential gating.
    
    Features:
    - Exponential gating mechanism
    - Scalar memory cells
    - Layer normalization for stability
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_multiplier: int = 4
    ):
        """
        Initialize xLSTM cell.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            memory_multiplier: Memory cell size multiplier
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = hidden_size * memory_multiplier
        
        # Input projections
        self.W_i = nn.Linear(input_size + hidden_size, self.memory_size)  # input gate
        self.W_f = nn.Linear(input_size + hidden_size, self.memory_size)  # forget gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # output gate
        self.W_z = nn.Linear(input_size + hidden_size, self.memory_size)  # cell input
        
        # Memory to hidden projection
        self.W_m = nn.Linear(self.memory_size, hidden_size)
        
        # Layer normalization
        self.ln_h = nn.LayerNorm(hidden_size)
        self.ln_c = nn.LayerNorm(self.memory_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Forget gate bias initialization (helps with gradient flow)
                if 'W_f' in name:
                    nn.init.ones_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of xLSTM cell.
        
        Args:
            x: Input tensor of shape (batch, input_size)
            state: Tuple of (h, c) hidden and cell states
            
        Returns:
            h_new: New hidden state
            (h_new, c_new): New state tuple
        """
        batch_size = x.size(0)
        
        # Initialize state if not provided
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.memory_size, device=x.device)
        else:
            h, c = state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        
        # Compute gates with exponential gating (clamped to prevent overflow)
        i = torch.exp(torch.clamp(self.W_i(combined), -10, 10))  # Clamped exponential input gate
        f = torch.sigmoid(self.W_f(combined))  # Sigmoid forget gate
        o = torch.sigmoid(self.W_o(combined))  # Output gate
        
        # Cell input
        z = torch.tanh(self.W_z(combined))
        
        # Update cell state (clamp to prevent explosion)
        c_new = f * c + i * z
        c_new = torch.clamp(c_new, -100, 100)
        c_new = self.ln_c(c_new)
        
        # Compute output
        m = self.W_m(c_new)
        h_new = o * torch.tanh(m)
        h_new = self.ln_h(h_new)
        
        return h_new, (h_new, c_new)


class xLSTM(nn.Module):
    """
    Multi-layer xLSTM network.
    
    Stacked xLSTM layers with residual connections
    and layer normalization for deep sequence modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        batch_first: bool = True,
        memory_multiplier: int = 4
    ):
        """
        Initialize xLSTM network.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of stacked xLSTM layers
            dropout: Dropout probability
            bidirectional: Use bidirectional processing
            batch_first: Input shape is (batch, seq, features)
            memory_multiplier: Memory cell size multiplier
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # xLSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size * self.num_directions if i > 0 else hidden_size
            self.layers.append(
                xLSTMCell(layer_input_size, hidden_size, memory_multiplier)
            )
            
            if bidirectional:
                self.layers.append(
                    xLSTMCell(layer_input_size, hidden_size, memory_multiplier)
                )
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * self.num_directions)
            for _ in range(num_layers)
        ])
        
        # Output size
        self.output_size = hidden_size * self.num_directions
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through xLSTM stack.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size) if batch_first
            state: Optional initial states
            
        Returns:
            output: Sequence outputs (batch, seq_len, hidden_size * num_directions)
            (h_n, c_n): Final states
        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_proj(x)
        
        # Initialize states for all layers
        if state is None:
            h_states = [None] * self.num_layers
            c_states = [None] * self.num_layers
        else:
            h_states = list(state[0])
            c_states = list(state[1])
        
        # Process through layers
        for layer_idx in range(self.num_layers):
            cell_idx = layer_idx * self.num_directions
            
            # Forward direction
            forward_outputs = []
            h_f, c_f = None, None
            
            for t in range(seq_len):
                h_f, (h_f, c_f) = self.layers[cell_idx](
                    x[:, t, :],
                    (h_f, c_f) if h_f is not None else None
                )
                forward_outputs.append(h_f)
            
            forward_output = torch.stack(forward_outputs, dim=1)
            
            # Backward direction (if bidirectional)
            if self.bidirectional:
                backward_outputs = []
                h_b, c_b = None, None
                
                for t in range(seq_len - 1, -1, -1):
                    h_b, (h_b, c_b) = self.layers[cell_idx + 1](
                        x[:, t, :],
                        (h_b, c_b) if h_b is not None else None
                    )
                    backward_outputs.insert(0, h_b)
                
                backward_output = torch.stack(backward_outputs, dim=1)
                x = torch.cat([forward_output, backward_output], dim=-1)
            else:
                x = forward_output
            
            # Apply dropout and layer norm
            x = self.dropout_layers[layer_idx](x)
            x = self.layer_norms[layer_idx](x)
        
        # Final states
        h_n = torch.stack([h_f] * self.num_layers)
        c_n = torch.stack([c_f] * self.num_layers)
        
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        return x, (h_n, c_n)
    
    def get_last_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the last timestep output."""
        output, _ = self.forward(x)
        return output[:, -1, :]


class xLSTMWithAttention(nn.Module):
    """
    xLSTM with self-attention mechanism.
    
    Combines xLSTM's sequential modeling with
    attention for capturing global dependencies.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.xlstm = xLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with xLSTM + attention."""
        # xLSTM encoding
        lstm_output, _ = self.xlstm(x)
        
        # Self-attention
        attn_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(lstm_output + attn_output)
        
        return output


# Example usage
if __name__ == "__main__":
    # Test xLSTM
    batch_size = 32
    seq_len = 60
    input_size = 64
    hidden_size = 128
    
    model = xLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.3
    )
    
    x = torch.randn(batch_size, seq_len, input_size)
    output, (h_n, c_n) = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state shape: {h_n.shape}")
    
    # Test with attention
    model_attn = xLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=4
    )
    
    output_attn = model_attn(x)
    print(f"Output with attention shape: {output_attn.shape}")
