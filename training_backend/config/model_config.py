"""Neural network model configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class xLSTMConfig:
    """xLSTM layer configuration."""
    input_size: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    batch_first: bool = True


@dataclass
class TransformerConfig:
    """Transformer encoder configuration."""
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    batch_first: bool = True


@dataclass
class PositionEncoderConfig:
    """Position state encoder configuration."""
    embedding_dim: int = 320
    max_positions: int = 20
    num_features_per_position: int = 16
    fund_status_dim: int = 15
    risk_metrics_dim: int = 9


@dataclass
class OutputHeadsConfig:
    """Multi-task output heads configuration."""
    # Price prediction
    price_horizons: List[int] = field(default_factory=lambda: [1, 4, 24])  # hours ahead
    
    # Classification
    direction_classes: int = 3  # up, down, neutral
    position_classes: int = 3  # buy, hold, sell
    
    # Regression outputs
    volatility_output: bool = True
    confidence_output: bool = True
    risk_signal_output: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration."""
    
    # Input configuration
    num_features: int = 60  # Number of input features
    sequence_length: int = 60  # Lookback window
    
    # Component configs
    xlstm: xLSTMConfig = field(default_factory=xLSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    position_encoder: PositionEncoderConfig = field(default_factory=PositionEncoderConfig)
    output_heads: OutputHeadsConfig = field(default_factory=OutputHeadsConfig)
    
    # Fusion layer
    fusion_dim: int = 256
    fusion_dropout: float = 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Loss weights for multi-task learning
    loss_weights: dict = field(default_factory=lambda: {
        'price': 0.30,
        'direction': 0.20,
        'volatility': 0.15,
        'position': 0.20,
        'risk': 0.10,
        'confidence': 0.05
    })
    
    # Device
    device: str = "cuda"  # or "cpu", "mps"
    
    def get_total_input_dim(self) -> int:
        """Get total input dimension after position encoding."""
        return self.num_features + self.position_encoder.embedding_dim


# Default model configuration
default_model_config = ModelConfig()
