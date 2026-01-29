"""Global settings and environment configuration."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """Global application settings."""
    
    # Project paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage")
    RAW_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage" / "raw")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage" / "processed")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage" / "models")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    
    # API Keys (from environment)
    ALPACA_API_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    ALPACA_SECRET_KEY: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    ALPACA_BASE_URL: str = field(default_factory=lambda: os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
    
    BINANCE_API_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    BINANCE_SECRET_KEY: str = field(default_factory=lambda: os.getenv("BINANCE_SECRET_KEY", ""))
    
    # Database
    DATABASE_URL: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data_storage/trading.db"))
    
    # Trading parameters
    INITIAL_CAPITAL: float = 100000.0
    PAPER_TRADING: bool = True
    
    # Data parameters
    DEFAULT_LOOKBACK_DAYS: int = 60
    SEQUENCE_LENGTH: int = 60  # 60 bars for model input
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path_attr in ['DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR', 'MODELS_DIR', 'LOGS_DIR']:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        return cls()


# Global settings instance
settings = Settings.from_env()
