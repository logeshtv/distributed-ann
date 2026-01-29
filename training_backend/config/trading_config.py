"""Trading parameters and risk management configuration."""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


class TradingMode(Enum):
    """Trading mode enumeration."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class RiskLevel(Enum):
    """Risk level for position sizing."""
    CONSERVATIVE = "conservative"  # 0.5x Kelly
    MODERATE = "moderate"          # 0.75x Kelly
    AGGRESSIVE = "aggressive"      # 1.0x Kelly
    EXTREME = "extreme"            # 1.5x Kelly


@dataclass
class RiskLimits:
    """Risk management limits."""
    # Drawdown limits (negative values)
    daily_loss_limit: float = -0.03      # -3% daily -> stop trading
    weekly_loss_limit: float = -0.10     # -10% weekly -> reduce 50%
    monthly_loss_limit: float = -0.20    # -20% monthly -> close 50%
    annual_loss_limit: float = -0.30     # -30% annual -> close all
    
    # Position limits
    max_position_size: float = 0.10      # 10% per position
    max_total_positions: int = 10
    max_leverage: float = 2.0
    max_sector_exposure: float = 0.30    # 30% per sector
    max_single_stock_exposure: float = 0.15
    
    # Stop loss
    default_stop_loss: float = 0.02      # 2% stop loss
    trailing_stop_trigger: float = 0.03  # Activate at 3% profit
    trailing_stop_distance: float = 0.015  # Trail by 1.5%


@dataclass
class ExecutionConfig:
    """Order execution configuration."""
    # Slippage and costs
    slippage_bps: float = 10.0           # 10 basis points = 0.1%
    commission_per_share: float = 0.005  # $0.005 per share
    commission_minimum: float = 1.0      # $1 minimum
    commission_maximum: float = 5.0      # $5 maximum per order
    
    # Order settings
    order_timeout_seconds: int = 60
    max_retry_attempts: int = 3
    use_limit_orders: bool = True
    limit_order_offset_bps: float = 5.0  # 5 bps from mid
    
    # Market hours (US Eastern)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0


@dataclass
class PositionSizingConfig:
    """Position sizing configuration."""
    # Kelly Criterion parameters
    use_kelly: bool = True
    kelly_fraction: float = 0.5          # Fractional Kelly (safer)
    
    # Fixed sizing fallback
    fixed_position_pct: float = 0.05     # 5% per position
    
    # Volatility-based sizing
    use_volatility_sizing: bool = True
    target_volatility: float = 0.20      # 20% annual vol target
    vol_lookback_days: int = 20


@dataclass
class SymbolUniverse:
    """Trading symbol universe."""
    # US Stocks
    us_stocks: List[str] = field(default_factory=lambda: [
        # Top tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK",
        # Consumer
        "WMT", "PG", "KO", "PEP", "COST",
        # Industrial
        "CAT", "BA", "HON", "UPS", "MMM",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI"
    ])
    
    # Cryptocurrencies
    crypto: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "AVAX/USDT",
        "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ETC/USDT"
    ])


@dataclass
class TradingConfig:
    """Complete trading configuration."""
    # Mode
    mode: TradingMode = TradingMode.PAPER
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Capital
    initial_capital: float = 100000.0
    currency: str = "USD"
    
    # Components
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    symbols: SymbolUniverse = field(default_factory=SymbolUniverse)
    
    # Strategy allocation
    strategy_allocation: Dict[str, float] = field(default_factory=lambda: {
        'trend_following': 0.35,    # 35% to trend strategies
        'mean_reversion': 0.25,     # 25% to mean reversion
        'momentum': 0.25,           # 25% to momentum
        'volatility': 0.15          # 15% to volatility strategies
    })
    
    # Rebalancing
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    rebalance_threshold: float = 0.05    # 5% drift triggers rebalance
    
    def get_kelly_multiplier(self) -> float:
        """Get Kelly criterion multiplier based on risk level."""
        multipliers = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 0.75,
            RiskLevel.AGGRESSIVE: 1.0,
            RiskLevel.EXTREME: 1.5
        }
        return multipliers[self.risk_level]


# Default trading configuration
default_trading_config = TradingConfig()
