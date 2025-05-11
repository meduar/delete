"""
Portfolio Management Module
Handles portfolio tracking, position management, and performance calculation
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from src.utils.config import Config

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_id: Optional[str] = None
    
    def market_value(self, current_price: float) -> float:
        """Calculate current market value of position"""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.quantity > 0:  # Long position
            return (current_price - self.entry_price) * self.quantity
        else:  # Short position
            return (self.entry_price - current_price) * abs(self.quantity)
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis of position"""
        return abs(self.quantity * self.entry_price)

class Portfolio:
    """Portfolio management class"""
    
    def __init__(self, config: Optional[Config] = None):
        # Set default values first
        self.initial_cash = 100000
        self.max_position_size = 0.1
        self.max_daily_loss = -1000
        
        # Override with config values if provided
        if config:
            # Get values from config - make sure they are not None
            initial_cash = config.get_nested('trading', 'initial_cash', default=self.initial_cash)
            max_position_size = config.get_nested('trading', 'max_position_size', default=self.max_position_size)
            max_daily_loss = config.get_nested('trading', 'max_daily_loss', default=self.max_daily_loss)
            
            # Only update if we got valid values
            if initial_cash is not None:
                self.initial_cash = initial_cash
            if max_position_size is not None:
                self.max_position_size = max_position_size
            if max_daily_loss is not None:
                self.max_daily_loss = max_daily_loss
        
        # Initialize cash to initial_cash
        self.cash = self.initial_cash
        
        # Initialize other attributes
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # Performance tracking
        self.equity_history: List[Dict] = []
        self.peak_equity = self.initial_cash
        self.max_drawdown = 0.0
        
        # Risk management
        self.current_day = datetime.now().date()
        self.daily_trades = 0
        
    def total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        # Ensure cash is not None
        if self.cash is None:
            return 0.0
            
        positions_value = sum(
            pos.market_value(current_prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total P&L including unrealized"""
        if self.initial_cash is None:
            return 0.0
        return self.total_value(current_prices) - self.initial_cash
    
    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades"""
        return sum(trade.get('pnl', 0) for trade in self.trade_history)
    
    def unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L from open positions"""
        return sum(
            pos.unrealized_pnl(current_prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
    
    def calculate_position_size(self, signal, current_price: float) -> int:
        """Calculate position size based on risk management"""
        # Get current portfolio value
        current_prices = {signal.symbol: current_price}
        portfolio_value = self.total_value(current_prices)
        
        # Ensure we have valid values
        if portfolio_value <= 0 or self.max_position_size is None:
            return 0
        
        # Maximum position based on portfolio size
        max_value = portfolio_value * self.max_position_size
        max_units = int(max_value / current_price)
        
        # Risk-based position sizing
        if signal.metadata.get('stop_loss'):
            risk_per_unit = abs(current_price - signal.metadata['stop_loss'])
            max_risk = portfolio_value * 0.02  # 2% risk per trade
            risk_based_units = int(max_risk / risk_per_unit) if risk_per_unit > 0 else max_units
            
            return min(max_units, risk_based_units)
        
        return max_units
    
    def open_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> str:
        """Open a new position"""
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_id=position_id
        )
        
        # Check if we already have a position in this symbol
        if symbol in self.positions:
            # Close existing position first
            self.close_position(symbol, entry_price)
        
        # Add new position
        self.positions[symbol] = position
        
        # Update cash
        self.cash -= abs(quantity * entry_price)
        
        # Update daily trades
        today = datetime.now().date()
        if self.current_day != today:
            self.current_day = today
            self.daily_trades = 0
        self.daily_trades += 1
        
        return position_id
    
    def close_position(self, symbol: str, exit_price: float) -> Dict:
        """Close a position and record the trade"""
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        pnl = position.unrealized_pnl(exit_price)
        
        # Update cash
        self.cash += abs(position.quantity * exit_price)
        
        # Record trade
        trade_record = {
            'position_id': position.position_id,
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_percent': (pnl / position.cost_basis) * 100,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'hold_time': datetime.now() - position.entry_time
        }
        
        self.trade_history.append(trade_record)
        
        # Update daily P&L
        today = datetime.now().date().isoformat()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += pnl
        
        # Remove position
        del self.positions[symbol]
        
        return trade_record
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def record_equity(self, current_prices: Dict[str, float]):
        """Record current equity point"""
        equity = self.total_value(current_prices)
        
        # Update peak equity and max drawdown
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Record equity point
        self.equity_history.append({
            'timestamp': datetime.now(),
            'equity': equity,
            'cash': self.cash,
            'positions_value': equity - self.cash,
            'drawdown': drawdown
        })
    
    def get_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get portfolio summary"""
        if current_prices is None:
            current_prices = {}
        
        return {
            'total_value': self.total_value(current_prices),
            'cash': self.cash,
            'initial_cash': self.initial_cash,
            'total_pnl': self.total_pnl(current_prices),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl(current_prices),
            'positions': len(self.positions),
            'max_drawdown': self.max_drawdown,
            'daily_trades': self.daily_trades,
            'total_trades': len(self.trade_history)
        }