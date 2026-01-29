"""Backtesting engine for historical simulation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from .position_tracker import PositionTracker, Position
from .risk_manager import RiskManager


@dataclass
class BacktestResult:
    """Backtest results container."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades: List[Dict]
    equity_curve: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Event-driven backtesting engine.
    
    Features:
    - Realistic slippage and commission
    - Position tracking
    - Risk management integration
    - Performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        slippage_bps: float = 10.0,
        commission_rate: float = 0.001,
        risk_manager: Optional[RiskManager] = None
    ):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps / 10000
        self.commission_rate = commission_rate
        self.risk_manager = risk_manager or RiskManager()
        
        self.position_tracker = PositionTracker(initial_capital)
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
    
    def reset(self):
        """Reset backtester state."""
        self.position_tracker = PositionTracker(self.initial_capital)
        self.trades = []
        self.equity_curve = []
    
    def apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        if side == 'buy':
            return price * (1 + self.slippage_bps)
        else:
            return price * (1 - self.slippage_bps)
    
    def calculate_commission(self, value: float) -> float:
        """Calculate commission for trade."""
        return max(value * self.commission_rate, 1.0)
    
    def execute_signal(
        self,
        timestamp: datetime,
        symbol: str,
        signal: int,  # 1=buy, 0=hold, -1=sell
        price: float,
        position_size: float = 0.1
    ) -> Optional[Dict]:
        """Execute trading signal."""
        current_position = self.position_tracker.get_position(symbol)
        
        # Check risk limits
        if not self.risk_manager.check_trade_allowed(self.position_tracker):
            return None
        
        trade = None
        
        if signal == 1 and current_position is None:  # Buy
            # Calculate position size
            capital = self.position_tracker.cash
            trade_value = capital * position_size
            exec_price = self.apply_slippage(price, 'buy')
            quantity = trade_value / exec_price
            commission = self.calculate_commission(trade_value)
            
            if capital >= trade_value + commission:
                self.position_tracker.open_position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=exec_price,
                    timestamp=timestamp
                )
                self.position_tracker.cash -= (trade_value + commission)
                
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': exec_price,
                    'commission': commission,
                    'value': trade_value
                }
                self.trades.append(trade)
        
        elif signal == -1 and current_position is not None:  # Sell
            exec_price = self.apply_slippage(price, 'sell')
            quantity = current_position.quantity
            trade_value = quantity * exec_price
            commission = self.calculate_commission(trade_value)
            
            pnl = (exec_price - current_position.entry_price) * quantity - commission
            
            self.position_tracker.close_position(symbol)
            self.position_tracker.cash += trade_value - commission
            
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': exec_price,
                'commission': commission,
                'value': trade_value,
                'pnl': pnl
            }
            self.trades.append(trade)
        
        return trade
    
    def run(
        self,
        data: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, int], int],
        position_size: float = 0.1
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: OHLCV DataFrame with features
            signal_fn: Function that returns signal given data and index
            position_size: Position size as fraction of capital
        """
        self.reset()
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row['timestamp']
            symbol = row.get('symbol', 'DEFAULT')
            price = row['close']
            
            # Get signal from strategy
            signal = signal_fn(data, i)
            
            # Execute signal
            self.execute_signal(timestamp, symbol, signal, price, position_size)
            
            # Update positions with current price
            self.position_tracker.update_prices({symbol: price})
            
            # Record equity
            equity = self.position_tracker.get_total_value()
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': self.position_tracker.cash
            })
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics."""
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) == 0:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, [], equity_df)
        
        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1
        
        # Annualized return (assuming daily data)
        n_days = len(equity_df)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # Sharpe ratio
        returns = equity_df['returns'].dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio
        neg_returns = returns[returns < 0]
        sortino = (returns.mean() / neg_returns.std()) * np.sqrt(252) if len(neg_returns) > 0 else 0
        
        # Max drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade metrics
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        total_profit = sum(t.get('pnl', 0) for t in wins)
        total_loss = abs(sum(t.get('pnl', 0) for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            trades=self.trades,
            equity_curve=equity_df,
            metrics={
                'total_profit': total_profit,
                'total_loss': total_loss,
                'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in self.trades]) if self.trades else 0
            }
        )
