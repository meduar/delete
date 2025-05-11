"""
Abstract Broker Interface
Defines the interface for order execution brokers
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class Order:
    """Order data structure"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = 'GTC'
    
    def __post_init__(self):
        """Validate order after initialization"""
        if self.side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side: {self.side}. Must be 'BUY' or 'SELL'")
        
        if self.order_type not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
            raise ValueError(f"Invalid order_type: {self.order_type}")
        
        if self.order_type in ['LIMIT', 'STOP', 'STOP_LIMIT'] and self.price is None:
            raise ValueError(f"{self.order_type} order requires a price")
        
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")


class Broker(ABC):
    """Abstract base class for broker implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Dict:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict:
        """Get all open positions"""
        pass
    
    @abstractmethod
    async def get_orders(self) -> Dict:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    async def close_position(self, symbol: str, units: Optional[int] = None) -> Dict:
        """
        Close a position (default implementation using orders)
        
        Args:
            symbol: Symbol to close
            units: Number of units to close (None for all)
        
        Returns:
            Result dictionary
        """
        # Get current position
        positions = await self.get_positions()
        if symbol not in positions:
            return {'status': 'no_position'}
        
        position = positions[symbol]
        current_units = position.get('long_units', 0) - position.get('short_units', 0)
        
        if current_units == 0:
            return {'status': 'no_position'}
        
        # Determine close size
        if units is None:
            close_units = abs(current_units)
        else:
            close_units = min(abs(units), abs(current_units))
        
        # Create close order
        side = 'SELL' if current_units > 0 else 'BUY'
        
        order = Order(
            symbol=symbol,
            side=side,
            quantity=close_units,
            order_type='MARKET'
        )
        
        return await self.place_order(order)
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get a specific position"""
        positions = await self.get_positions()
        return positions.get(symbol)
    
    async def get_equity(self) -> float:
        """Get account equity"""
        account_info = await self.get_account_info()
        return account_info.get('balance', 0.0)
    
    async def get_margin_available(self) -> float:
        """Get available margin"""
        account_info = await self.get_account_info()
        return account_info.get('margin_available', 0.0)