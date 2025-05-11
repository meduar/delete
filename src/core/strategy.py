"""
Base Strategy Module
Defines the base strategy class and Signal dataclass for trading strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class Signal:
    """Represents a trading signal"""
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    price: float
    symbol: str
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate signal after initialization"""
        if self.action not in ['buy', 'sell', 'hold']:
            raise ValueError(f"Invalid action: {self.action}")
        
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.strength}")
            
        if self.price <= 0:
            raise ValueError(f"Price must be positive, got {self.price}")

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.indicators = {}
        self.state = {}
        self.data_feed = None
        
        # Performance tracking
        self.entry_count = 0
        self.exit_count = 0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
    async def initialize(self, data_feed):
        """
        Initialize strategy with data feed
        
        Args:
            data_feed: Data feed instance to use for market data
        """
        self.data_feed = data_feed
        await self._setup_indicators()
        self._initialize_state()
        
    @abstractmethod
    async def _setup_indicators(self):
        """Set up technical indicators - to be implemented by subclasses"""
        pass
        
    @abstractmethod
    async def _generate_signal(self, market_data) -> Optional[Signal]:
        """Generate trading signal - to be implemented by subclasses"""
        pass
        
    def _initialize_state(self):
        """Initialize strategy state"""
        self.state = {
            'position': None,
            'entry_price': None,
            'entry_time': None,
            'last_signal': None,
            'last_signal_time': None,
            'bars_since_entry': 0,
            'bars_since_exit': 0
        }
        
    async def update(self, market_data) -> Optional[Signal]:
        """
        Update strategy with new market data
        
        Args:
            market_data: New market data point
            
        Returns:
            Trading signal or None
        """
        # Update indicators
        await self._update_indicators(market_data)
        
        # Update state
        self._update_state(market_data)
        
        # Generate signal
        signal = await self._generate_signal(market_data)
        
        # Post-process signal
        if signal:
            signal = self._post_process_signal(signal, market_data)
            
            # Update performance tracking
            self._update_performance_tracking(signal)
            
            # Log signal
            await self._log_signal(signal, market_data)
            
        return signal
    
    async def _update_indicators(self, market_data):
        """Update all indicators with new data"""
        for name, indicator in self.indicators.items():
            if hasattr(indicator, 'update'):
                indicator.update(market_data)
    
    def _update_state(self, market_data):
        """Update strategy state"""
        # Update bar counts
        if self.state['position']:
            self.state['bars_since_entry'] += 1
        else:
            self.state['bars_since_exit'] += 1
            
        # Store last market data
        self.state['last_market_data'] = market_data
    
    def _post_process_signal(self, signal: Signal, market_data) -> Signal:
        """Post-process signal before returning"""
        # Add common metadata
        if 'strategy_name' not in signal.metadata:
            signal.metadata['strategy_name'] = self.name
            
        if 'parameters' not in signal.metadata:
            signal.metadata['parameters'] = self.parameters.copy()
            
        # Store signal in state
        self.state['last_signal'] = signal
        self.state['last_signal_time'] = signal.timestamp
        
        return signal
    
    def _update_performance_tracking(self, signal: Signal):
        """Update performance tracking"""
        if signal.action == 'buy':
            self.entry_count += 1
            self.state['position'] = 'long'
            self.state['entry_price'] = signal.price
            self.state['entry_time'] = signal.timestamp
            self.state['bars_since_entry'] = 0
            
        elif signal.action == 'sell':
            if self.state['position'] == 'long':
                # Calculate P&L
                pnl = signal.price - self.state['entry_price']
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                    
                self.exit_count += 1
                self.state['position'] = None
                self.state['bars_since_exit'] = 0
    
    async def _log_signal(self, signal: Signal, market_data):
        """Log generated signal"""
        log_entry = {
            'timestamp': signal.timestamp,
            'strategy': self.name,
            'symbol': signal.symbol,
            'action': signal.action,
            'strength': signal.strength,
            'price': signal.price,
            'metadata': signal.metadata
        }
        # Here you would integrate with your logging system
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the strategy"""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'strategy_name': self.name,
            'total_entries': self.entry_count,
            'total_exits': self.exit_count,
            'total_pnl': self.total_pnl,
            'total_trades': total_trades,
            'winning_trades': self.win_count,
            'losing_trades': self.loss_count,
            'win_rate': win_rate,
            'average_pnl_per_trade': self.total_pnl / total_trades if total_trades > 0 else 0
        }
    
    def reset(self):
        """Reset strategy state and performance tracking"""
        self._initialize_state()
        self.entry_count = 0
        self.exit_count = 0
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        # Reset indicators
        for indicator in self.indicators.values():
            if hasattr(indicator, 'reset'):
                indicator.reset()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'parameters': self.parameters,
            'state': self.state,
            'performance': self.get_performance_summary()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseStrategy':
        """Create strategy instance from dictionary"""
        strategy = cls(data['name'], data['parameters'])
        strategy.state = data.get('state', {})
        return strategy