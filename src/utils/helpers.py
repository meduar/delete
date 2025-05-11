"""
Utility helper functions for the trading framework
"""

import asyncio
import time
from functools import wraps
from typing import Callable, Any, TypeVar
import logging

T = TypeVar('T')

def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Async retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay between retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    Works with both sync and async functions
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

class RateLimiter:
    """
    Rate limiter for API calls
    Ensures API calls don't exceed specified rate
    """
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
        self._lock = asyncio.Lock()
        
    async def wait(self):
        """Wait if necessary to respect rate limit"""
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                await asyncio.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await self.wait()
            return await func(*args, **kwargs)
        return wrapper

def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """
    Format currency amount for display
    
    Args:
        amount: The amount to format
        currency: Currency code (e.g., "USD", "EUR")
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    symbol_map = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "CHF": "CHF ",
        "CNY": "¥"
    }
    
    symbol = symbol_map.get(currency, f"{currency} ")
    
    if currency == "JPY":
        decimals = 0
    
    if symbol in ["$", "€", "£", "¥"]:
        return f"{symbol}{amount:,.{decimals}f}"
    else:
        return f"{symbol}{amount:,.{decimals}f}"

def calculate_pip_value(symbol: str, lot_size: float = 100000) -> float:
    """
    Calculate pip value for a given symbol
    
    Args:
        symbol: Currency pair (e.g., "EUR_USD")
        lot_size: Size of one lot (default 100,000)
    
    Returns:
        Pip value in the account currency
    """
    # This is a simplified calculation
    # In a real implementation, you'd need current exchange rates
    
    major_pairs = {
        "EUR_USD": 10.0,
        "GBP_USD": 10.0,
        "AUD_USD": 10.0,
        "NZD_USD": 10.0,
        "USD_CAD": 7.5,
        "USD_CHF": 10.5,
        "USD_JPY": 0.09,
    }
    
    return major_pairs.get(symbol, 10.0) * (lot_size / 100000)

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage value for display
    
    Args:
        value: The percentage value (e.g., 0.05 for 5%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division by zero
    
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator

class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by temporarily disabling operations
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        
    def call(self, func: Callable, *args, **kwargs):
        """
        Call function with circuit breaker protection
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Result of function call
        
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit is open
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if we were half open
            if self.state == "HALF_OPEN":
                self._reset()
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logging.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e
            
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

class PriceFormatter:
    """Format prices according to instrument specifications"""
    
    @staticmethod
    def format_price(price: float, symbol: str) -> str:
        """
        Format price according to symbol conventions
        
        Args:
            price: The price to format
            symbol: Currency pair or instrument symbol
        
        Returns:
            Formatted price string
        """
        # JPY pairs typically have 3 decimal places
        if any(x in symbol for x in ["JPY", "JPY_", "_JPY"]):
            return f"{price:.3f}"
        
        # Most pairs have 5 decimal places
        return f"{price:.5f}"
    
    @staticmethod
    def get_pip_size(symbol: str) -> float:
        """
        Get pip size for a symbol
        
        Args:
            symbol: Currency pair symbol
        
        Returns:
            Pip size (e.g., 0.00001 for most pairs, 0.001 for JPY pairs)
        """
        if any(x in symbol for x in ["JPY", "JPY_", "_JPY"]):
            return 0.001
        return 0.00001

def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol is in correct format
    
    Args:
        symbol: Symbol to validate (e.g., "EUR_USD")
    
    Returns:
        True if valid, False otherwise
    """
    # Basic validation - should have two 3-letter currency codes separated by underscore
    parts = symbol.split('_')
    if len(parts) != 2:
        return False
    
    return all(len(part) == 3 and part.isalpha() for part in parts)

def calculate_position_value(price: float, quantity: int, contract_size: float = 100000) -> float:
    """
    Calculate the total value of a position
    
    Args:
        price: Current price
        quantity: Number of units
        contract_size: Size of one contract/lot
    
    Returns:
        Total position value
    """
    return abs(price * quantity * contract_size)

class Singleton(type):
    """Singleton metaclass"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]