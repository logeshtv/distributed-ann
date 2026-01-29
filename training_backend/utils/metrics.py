"""Performance metrics for trading."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TradingMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate / periods
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        excess_returns = returns - risk_free_rate / periods
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0
        return (excess_returns.mean() / downside) * np.sqrt(periods)
    
    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, equity: pd.Series, periods: int = 252) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        annual_return = returns.mean() * periods
        mdd = abs(TradingMetrics.max_drawdown(equity))
        if mdd == 0:
            return 0
        return annual_return / mdd
    
    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return wins / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    @staticmethod
    def expectancy(trades: List[Dict]) -> float:
        """Calculate expectancy (expected value per trade)."""
        if not trades:
            return 0
        wins = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
        
        if not wins and not losses:
            return 0
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def calculate_all(
        returns: pd.Series,
        equity: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        return {
            'sharpe_ratio': TradingMetrics.sharpe_ratio(returns),
            'sortino_ratio': TradingMetrics.sortino_ratio(returns),
            'max_drawdown': TradingMetrics.max_drawdown(equity),
            'calmar_ratio': TradingMetrics.calmar_ratio(returns, equity),
            'win_rate': TradingMetrics.win_rate(trades),
            'profit_factor': TradingMetrics.profit_factor(trades),
            'expectancy': TradingMetrics.expectancy(trades),
            'total_return': (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'total_trades': len(trades)
        }
