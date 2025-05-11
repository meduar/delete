"""
OANDA Broker Implementation
Handles order execution and position management through OANDA API
"""

import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.exceptions import V20Error
from typing import Dict, List, Optional
import logging
import asyncio

from src.execution.broker import Broker, Order


class OandaBroker(Broker):
    """OANDA broker implementation for order execution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key')
        self.account_id = config.get('account_id')
        self.environment = config.get('environment', 'practice')
        
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize OANDA API client"""
        try:
            if self.environment == 'practice':
                self.client = oandapyV20.API(
                    access_token=self.api_key,
                    environment='practice'
                )
            else:
                self.client = oandapyV20.API(
                    access_token=self.api_key,
                    environment='live'
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize OANDA client: {e}")
            raise
    
    async def connect(self) -> bool:
        """Connect to OANDA broker"""
        try:
            # Test connection by getting account info
            await self.get_account_info()
            self.logger.info("Connected to OANDA broker")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to OANDA broker: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OANDA broker"""
        # OANDA doesn't require explicit disconnection
        self.logger.info("Disconnected from OANDA broker")
    
    async def place_order(self, order: Order) -> Dict:
        """
        Place an order with OANDA
        
        Args:
            order: Order object with details
        
        Returns:
            Dictionary with order response
        """
        try:
            # Convert to OANDA format
            oanda_order = self._create_oanda_order(order)
            
            # Place order
            request = orders.OrderCreate(accountID=self.account_id, data=oanda_order)
            
            # Execute in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            # Parse response
            if 'orderCreateTransaction' in response:
                order_id = response['orderCreateTransaction']['id']
                self.logger.info(f"Order placed successfully: {order_id}")
                
                return {
                    'id': order_id,
                    'status': 'PENDING',
                    'transaction': response['orderCreateTransaction']
                }
            
            raise ValueError(f"Unexpected response format: {response}")
            
        except V20Error as e:
            self.logger.error(f"OANDA API error placing order: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            request = orders.OrderCancel(accountID=self.account_id, orderID=order_id)
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            if 'orderCancelTransaction' in response:
                self.logger.info(f"Order cancelled successfully: {order_id}")
                return True
            
            return False
            
        except V20Error as e:
            self.logger.error(f"OANDA API error cancelling order: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_positions(self) -> Dict:
        """Get all open positions"""
        try:
            request = positions.PositionList(accountID=self.account_id)
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            if 'positions' in response:
                return {
                    pos['instrument']: {
                        'long_units': int(pos['long']['units']),
                        'short_units': int(pos['short']['units']),
                        'unrealized_pl': float(pos['unrealizedPL'])
                    }
                    for pos in response['positions']
                    if int(pos['long']['units']) != 0 or int(pos['short']['units']) != 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    async def get_orders(self) -> Dict:
        """Get all open orders"""
        try:
            request = orders.OrderList(accountID=self.account_id, params={'state': 'PENDING'})
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            if 'orders' in response:
                return {
                    order['id']: {
                        'instrument': order['instrument'],
                        'type': order['type'],
                        'state': order['state'],
                        'units': order['units'],
                        'price': order.get('price'),
                        'time_in_force': order.get('timeInForce')
                    }
                    for order in response['orders']
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return {}
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            request = accounts.AccountSummary(accountID=self.account_id)
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            if 'account' in response:
                account = response['account']
                return {
                    'balance': float(account['balance']),
                    'margin_available': float(account['marginAvailable']),
                    'margin_used': float(account['marginUsed']),
                    'unrealized_pl': float(account['unrealizedPL']),
                    'currency': account['currency'],
                    'leverage': float(account.get('leverage', 1))
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    async def close_position(self, instrument: str, units: int = None) -> Dict:
        """
        Close a position
        
        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            units: Number of units to close (None for all)
        
        Returns:
            Dictionary with close response
        """
        try:
            # If no units specified, close entire position
            if units is None:
                current_positions = await self.get_positions()
                if instrument in current_positions:
                    long_units = current_positions[instrument]['long_units']
                    short_units = current_positions[instrument]['short_units']
                else:
                    return {'status': 'no_position'}
            else:
                long_units = units if units > 0 else 0
                short_units = abs(units) if units < 0 else 0
            
            # Close position
            close_data = {}
            if long_units > 0:
                close_data['longUnits'] = 'ALL' if units is None else str(long_units)
            if short_units > 0:
                close_data['shortUnits'] = 'ALL' if units is None else str(short_units)
            
            if close_data:
                request = positions.PositionClose(
                    accountID=self.account_id,
                    instrument=instrument,
                    data=close_data
                )
                
                # Execute in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.client.request, request)
                
                self.logger.info(f"Position closed for {instrument}")
                return response
            
            return {'status': 'no_action'}
            
        except Exception as e:
            self.logger.error(f"Error closing position for {instrument}: {e}")
            raise
    
    def _create_oanda_order(self, order: Order) -> Dict:
        """Convert Order object to OANDA format"""
        # Base order structure
        oanda_order = {
            'order': {
                'type': order.order_type,
                'instrument': order.symbol.replace('/', '_'),
                'units': str(order.quantity if order.side == 'BUY' else -order.quantity),
                'timeInForce': order.time_in_force
            }
        }
        
        # Add price for limit orders
        if order.order_type == 'LIMIT' and order.price:
            oanda_order['order']['price'] = str(order.price)
        
        # Add stop loss if specified
        if order.stop_loss:
            oanda_order['order']['stopLossOnFill'] = {
                'price': str(order.stop_loss),
                'timeInForce': 'GTC'
            }
        
        # Add take profit if specified
        if order.take_profit:
            oanda_order['order']['takeProfitOnFill'] = {
                'price': str(order.take_profit),
                'timeInForce': 'GTC'
            }
        
        return oanda_order
    
    async def modify_order(self, order_id: str, new_price: float) -> bool:
        """Modify an existing order"""
        try:
            # First get the order details
            request = orders.OrderDetails(accountID=self.account_id, orderID=order_id)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.client.request, request)
            
            if 'order' in response:
                old_order = response['order']
                
                # Create modified order request
                new_order_data = {
                    'order': {
                        'type': old_order['type'],
                        'instrument': old_order['instrument'],
                        'units': old_order['units'],
                        'price': str(new_price),
                        'timeInForce': old_order['timeInForce']
                    }
                }
                
                # Cancel old order and create new one
                await self.cancel_order(order_id)
                response = await self.place_order(Order(
                    symbol=old_order['instrument'].replace('_', '/'),
                    side='BUY' if int(old_order['units']) > 0 else 'SELL',
                    quantity=abs(int(old_order['units'])),
                    order_type=old_order['type'],
                    price=new_price,
                    time_in_force=old_order['timeInForce']
                ))
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False


# Utility functions for working with OANDA

def format_oanda_price(price: float, instrument: str) -> str:
    """Format price according to OANDA's requirements"""
    # JPY pairs use 3 decimal places, others use 5
    if 'JPY' in instrument:
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"


def parse_oanda_time(time_str: str) -> 'datetime':
    """Parse OANDA's time format to datetime"""
    from datetime import datetime
    # OANDA uses ISO 8601 format: YYYY-MM-DDTHH:MM:SS.SSSSSSSSSZ
    return datetime.fromisoformat(time_str.replace('Z', '+00:00'))


def calculate_position_size(
    balance: float,
    risk_percentage: float,
    entry_price: float,
    stop_loss_price: float,
    instrument: str
) -> int:
    """
    Calculate position size based on risk
    
    Args:
        balance: Account balance
        risk_percentage: Risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        instrument: Trading instrument
    
    Returns:
        Position size in units
    """
    risk_amount = balance * risk_percentage
    pip_risk = abs(entry_price - stop_loss_price)
    
    # Get pip value
    if 'JPY' in instrument:
        pip_value = 0.01 * 1000  # For mini lots
    else:
        pip_value = 0.0001 * 1000  # For mini lots
    
    # Calculate position size
    position_size = risk_amount / (pip_risk / pip_value)
    
    # Round to nearest mini lot
    return int(round(position_size / 1000) * 1000)


class OandaOrderBuilder:
    """Helper class to build OANDA orders"""
    
    def __init__(self, instrument: str):
        self.instrument = instrument.replace('/', '_')
        self.order_data = {
            'order': {
                'type': 'MARKET',
                'instrument': self.instrument,
                'timeInForce': 'FOK'
            }
        }
    
    def set_type(self, order_type: str) -> 'OandaOrderBuilder':
        """Set order type (MARKET, LIMIT, STOP, etc.)"""
        self.order_data['order']['type'] = order_type
        return self
    
    def set_units(self, units: int) -> 'OandaOrderBuilder':
        """Set order units (positive for buy, negative for sell)"""
        self.order_data['order']['units'] = str(units)
        return self
    
    def set_price(self, price: float) -> 'OandaOrderBuilder':
        """Set limit/stop price"""
        self.order_data['order']['price'] = format_oanda_price(price, self.instrument)
        return self
    
    def set_stop_loss(self, price: float) -> 'OandaOrderBuilder':
        """Set stop loss price"""
        self.order_data['order']['stopLossOnFill'] = {
            'price': format_oanda_price(price, self.instrument),
            'timeInForce': 'GTC'
        }
        return self
    
    def set_take_profit(self, price: float) -> 'OandaOrderBuilder':
        """Set take profit price"""
        self.order_data['order']['takeProfitOnFill'] = {
            'price': format_oanda_price(price, self.instrument),
            'timeInForce': 'GTC'
        }
        return self
    
    def set_trailing_stop(self, distance: float) -> 'OandaOrderBuilder':
        """Set trailing stop distance"""
        self.order_data['order']['trailingStopLossOnFill'] = {
            'distance': format_oanda_price(distance, self.instrument),
            'timeInForce': 'GTC'
        }
        return self
    
    def set_time_in_force(self, tif: str) -> 'OandaOrderBuilder':
        """Set time in force (GTC, IOC, FOK, etc.)"""
        self.order_data['order']['timeInForce'] = tif
        return self
    
    def build(self) -> Dict:
        """Build and return the order data"""
        return self.order_data


class OandaPositionManager:
    """Helper class for managing OANDA positions"""
    
    def __init__(self, broker: OandaBroker):
        self.broker = broker
        self.logger = logging.getLogger(__name__)
    
    async def close_partial_position(
        self, 
        instrument: str, 
        percentage: float
    ) -> Dict:
        """
        Close a percentage of a position
        
        Args:
            instrument: Trading instrument
            percentage: Percentage to close (0.0 to 1.0)
        """
        positions = await self.broker.get_positions()
        
        if instrument not in positions:
            return {'status': 'no_position'}
        
        position = positions[instrument]
        long_units = position['long_units']
        short_units = position['short_units']
        
        units_to_close = 0
        if long_units > 0:
            units_to_close = int(long_units * percentage)
        elif short_units > 0:
            units_to_close = -int(short_units * percentage)
        
        if units_to_close != 0:
            return await self.broker.close_position(instrument, units_to_close)
        
        return {'status': 'no_units_to_close'}
    
    async def get_position_pnl(self, instrument: str) -> float:
        """Get unrealized P&L for a position"""
        positions = await self.broker.get_positions()
        
        if instrument in positions:
            return positions[instrument]['unrealized_pl']
        
        return 0.0
    
    async def set_position_stop_loss(
        self, 
        instrument: str, 
        stop_loss_price: float
    ) -> bool:
        """Set or update stop loss for a position"""
        try:
            # Get current position
            positions = await self.broker.get_positions()
            
            if instrument not in positions:
                return False
            
            position = positions[instrument]
            long_units = position['long_units']
            short_units = position['short_units']
            
            # Create order to set stop loss
            units = long_units if long_units > 0 else -short_units
            
            order_data = {
                'order': {
                    'type': 'STOP',
                    'instrument': instrument,
                    'units': str(-units),  # Opposite direction
                    'price': format_oanda_price(stop_loss_price, instrument),
                    'timeInForce': 'GTC',
                    'positionFill': 'REDUCE_ONLY'
                }
            }
            
            # Place stop loss order
            await self.broker.place_order(order_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting stop loss for {instrument}: {e}")
            return False
    
    async def set_position_take_profit(
        self, 
        instrument: str, 
        take_profit_price: float
    ) -> bool:
        """Set or update take profit for a position"""
        try:
            # Get current position
            positions = await self.broker.get_positions()
            
            if instrument not in positions:
                return False
            
            position = positions[instrument]
            long_units = position['long_units']
            short_units = position['short_units']
            
            # Create order to set take profit
            units = long_units if long_units > 0 else -short_units
            
            order_data = {
                'order': {
                    'type': 'LIMIT',
                    'instrument': instrument,
                    'units': str(-units),  # Opposite direction
                    'price': format_oanda_price(take_profit_price, instrument),
                    'timeInForce': 'GTC',
                    'positionFill': 'REDUCE_ONLY'
                }
            }
            
            # Place take profit order
            await self.broker.place_order(order_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting take profit for {instrument}: {e}")
            return False


# Error handling and retry logic

class OandaRetryHandler:
    """Handle retries for OANDA API calls"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except V20Error as e:
                last_exception = e
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise
                
                # Calculate delay
                delay = self.backoff_factor ** attempt
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed. "
                    f"Retrying in {delay}s: {e}"
                )
                
                await asyncio.sleep(delay)
            except Exception as e:
                last_exception = e
                self.logger.error(f"Non-retryable error: {e}")
                raise
        
        # All retries exhausted
        raise last_exception
    
    def _is_retryable_error(self, error: V20Error) -> bool:
        """Check if error is retryable"""
        retryable_codes = [
            429,  # Too many requests
            500,  # Internal server error
            502,  # Bad gateway
            503,  # Service unavailable
            504   # Gateway timeout
        ]
        
        if hasattr(error, 'code'):
            return error.code in retryable_codes
        
        # Assume retryable if we can't determine the code
        return True