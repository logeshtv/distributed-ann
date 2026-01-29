"""Risk management system."""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class RiskLimits:
    """Risk limit configuration."""
    daily_loss_limit: float = -0.03
    weekly_loss_limit: float = -0.10
    monthly_loss_limit: float = -0.20
    max_position_size: float = 0.10
    max_positions: int = 10
    max_leverage: float = 2.0
    max_drawdown: float = -0.25


class RiskManager:
    """
    Risk management for trading.
    
    Enforces:
    - Daily/weekly/monthly loss limits
    - Position size limits
    - Maximum positions
    - Drawdown limits
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.peak_equity = 0.0
        self.trading_halted = False
        self.halt_reason = ""
        self.last_reset = datetime.now()
    
    def update_pnl(self, pnl: float, equity: float):
        """Update P&L tracking."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.monthly_pnl += pnl
        
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def reset_daily(self):
        """Reset daily P&L."""
        self.daily_pnl = 0.0
    
    def reset_weekly(self):
        """Reset weekly P&L."""
        self.weekly_pnl = 0.0
    
    def reset_monthly(self):
        """Reset monthly P&L."""
        self.monthly_pnl = 0.0
    
    def check_trade_allowed(self, position_tracker) -> bool:
        """Check if trading is allowed based on risk limits."""
        if self.trading_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return False
        
        equity = position_tracker.get_total_value()
        initial = position_tracker.initial_capital
        
        # Check daily loss limit
        daily_return = self.daily_pnl / initial
        if daily_return <= self.limits.daily_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Daily loss limit hit: {daily_return:.2%}"
            logger.warning(self.halt_reason)
            return False
        
        # Check weekly loss limit
        weekly_return = self.weekly_pnl / initial
        if weekly_return <= self.limits.weekly_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Weekly loss limit hit: {weekly_return:.2%}"
            logger.warning(self.halt_reason)
            return False
        
        # Check monthly loss limit
        monthly_return = self.monthly_pnl / initial
        if monthly_return <= self.limits.monthly_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Monthly loss limit hit: {monthly_return:.2%}"
            logger.warning(self.halt_reason)
            return False
        
        # Check max drawdown
        if self.peak_equity > 0:
            drawdown = (equity - self.peak_equity) / self.peak_equity
            if drawdown <= self.limits.max_drawdown:
                self.trading_halted = True
                self.halt_reason = f"Max drawdown hit: {drawdown:.2%}"
                logger.warning(self.halt_reason)
                return False
        
        # Check max positions
        if len(position_tracker.positions) >= self.limits.max_positions:
            logger.info(f"Max positions reached: {self.limits.max_positions}")
            return False
        
        return True
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        volatility: float = 0.02,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.01
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            capital: Available capital
            price: Current price
            volatility: Asset volatility
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
        """
        # Kelly fraction
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly = 0.1
        
        # Use half-Kelly for safety
        kelly = max(0, min(kelly * 0.5, self.limits.max_position_size))
        
        # Adjust for volatility
        vol_adjusted = kelly / max(volatility / 0.02, 1.0)
        
        position_value = capital * vol_adjusted
        quantity = position_value / price
        
        return quantity
    
    def get_stop_loss(self, entry_price: float, volatility: float = 0.02) -> float:
        """Calculate stop loss price."""
        stop_distance = max(volatility * 2, 0.02)
        return entry_price * (1 - stop_distance)
    
    def get_take_profit(self, entry_price: float, risk_reward: float = 2.0, volatility: float = 0.02) -> float:
        """Calculate take profit price."""
        stop_distance = max(volatility * 2, 0.02)
        return entry_price * (1 + stop_distance * risk_reward)
    
    def get_status(self) -> Dict:
        """Get risk manager status."""
        return {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'monthly_pnl': self.monthly_pnl,
            'peak_equity': self.peak_equity
        }
