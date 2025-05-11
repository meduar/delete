"""
Order Manager Module
Manages order lifecycle and tracking
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging

from src.execution.broker import Order  # Fixed import


class OrderManager:
    """Manage order lifecycle and tracking"""
    
    def __init__(self, broker):
        self.broker = broker
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        
    async def submit_order(self, order: Order) -> Optional[str]:
        """Submit an order to the broker"""
        try:
            result = await self.broker.place_order(order)
            order_id = result.get('id')
            
            if order_id:
                self.active_orders[order_id] = order
                self.logger.info(f"Order submitted: {order_id}")
                return order_id
            else:
                self.logger.error(f"Failed to submit order: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            success = await self.broker.cancel_order(order_id)
            
            if success and order_id in self.active_orders:
                order = self.active_orders.pop(order_id)
                self.order_history.append({
                    'order': order,
                    'status': 'CANCELLED',
                    'timestamp': datetime.now()
                })
                self.logger.info(f"Order cancelled: {order_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def update_orders(self):
        """Update status of all active orders"""
        try:
            open_orders = await self.broker.get_orders()
            
            # Check for filled or cancelled orders
            for order_id in list(self.active_orders.keys()):
                if order_id not in open_orders:
                    # Order was filled or cancelled
                    order = self.active_orders.pop(order_id)
                    self.order_history.append({
                        'order': order,
                        'status': 'FILLED',  # We assume filled if not in open orders
                        'timestamp': datetime.now()
                    })
                    self.logger.info(f"Order completed: {order_id}")
        
        except Exception as e:
            self.logger.error(f"Error updating orders: {e}")
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get pending orders, optionally filtered by symbol"""
        if symbol:
            return [order for order in self.active_orders.values() if order.symbol == symbol]
        return list(self.active_orders.values())
    
    def get_order_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get order history"""
        if limit:
            return self.order_history[-limit:]
        return self.order_history.copy()