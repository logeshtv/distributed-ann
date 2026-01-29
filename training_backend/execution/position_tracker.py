"""Position tracking for portfolio management."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime


@dataclass
class Position:
    """Single position in portfolio."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def hold_duration(self) -> int:
        return (datetime.now() - self.entry_time).days


class PositionTracker:
    """Track all positions in portfolio."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []
        self.realized_pnl = 0.0
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        timestamp: Optional[datetime] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Open a new position."""
        if symbol in self.positions:
            # Add to existing position
            existing = self.positions[symbol]
            total_qty = existing.quantity + quantity
            avg_price = (existing.quantity * existing.entry_price + quantity * entry_price) / total_qty
            existing.quantity = total_qty
            existing.entry_price = avg_price
            return existing
        
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=timestamp or datetime.now(),
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        self.positions[symbol] = position
        return position
    
    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> Optional[Dict]:
        """Close a position."""
        if symbol not in self.positions:
            return None
        
        position = self.positions.pop(symbol)
        exit_price = exit_price or position.current_price
        pnl = (exit_price - position.entry_price) * position.quantity
        self.realized_pnl += pnl
        
        closed = {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'entry_time': position.entry_time,
            'exit_time': datetime.now()
        }
        self.closed_positions.append(closed)
        return closed
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_total_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def get_exposure(self) -> float:
        """Get total exposure as fraction of portfolio."""
        total = self.get_total_value()
        positions_value = sum(p.market_value for p in self.positions.values())
        return positions_value / total if total > 0 else 0
    
    def check_stop_losses(self) -> List[str]:
        """Check and return symbols that hit stop loss."""
        triggered = []
        for symbol, position in self.positions.items():
            if position.stop_loss and position.current_price <= position.stop_loss:
                triggered.append(symbol)
        return triggered
    
    def check_take_profits(self) -> List[str]:
        """Check and return symbols that hit take profit."""
        triggered = []
        for symbol, position in self.positions.items():
            if position.take_profit and position.current_price >= position.take_profit:
                triggered.append(symbol)
        return triggered
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            'cash': self.cash,
            'positions_value': sum(p.market_value for p in self.positions.values()),
            'total_value': self.get_total_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.realized_pnl + self.get_unrealized_pnl(),
            'num_positions': len(self.positions),
            'exposure': self.get_exposure()
        }
