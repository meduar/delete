"""
Data Feed Module
Unified data feed interface for multiple providers
"""

import asyncio
from typing import Dict, List, Optional, AsyncIterator, Union
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import the base MarketData class
from dataclasses import dataclass


@dataclass
class MarketData:
    """Standard market data format used throughout the framework"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class DataFeed:
    """Unified data feed interface for multiple providers"""
    
    def __init__(self, provider_config: Dict):
        self.provider_type = provider_config.get('type', 'oanda')
        self.provider_config = provider_config
        self.provider = None
        self.is_connected = False
        self.logger = logging.getLogger(__name__)
        
    def _create_provider(self):
        """Factory method to create the appropriate data provider"""
        try:
            if self.provider_config.get('type') == 'oanda' and self.provider_config.get('api_key') and self.provider_config.get('account_id'):
                from src.data.providers.oanda import OandaDataProvider
                return OandaDataProvider(self.provider_config)
            elif self.provider_config.get('type') == 'mock':
                # Return mock provider directly for testing
                return MockDataProvider(self.provider_config)
            else:
                self.logger.warning(f"Using mock provider (missing OANDA credentials or type)")
                return MockDataProvider(self.provider_config)
        except ImportError as e:
            self.logger.error(f"Failed to import provider: {e}")
            # Create a mock provider for testing
            return MockDataProvider(self.provider_config)
        except Exception as e:
            self.logger.error(f"Error creating provider: {e}")
            # Fallback to mock provider
            return MockDataProvider(self.provider_config)
    
    async def connect(self):
        """Connect to the data provider"""
        if not self.provider:
            self.provider = self._create_provider()
        
        try:
            await self.provider.connect()
            self.is_connected = True
            self.logger.info(f"Connected to {self.provider_type} data provider")
        except Exception as e:
            self.logger.error(f"Failed to connect to data provider: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the data provider"""
        if self.provider and self.is_connected:
            await self.provider.disconnect()
            self.is_connected = False
            self.logger.info(f"Disconnected from {self.provider_type} data provider")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "M1"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading symbol (e.g., "EUR_USD")
            start_date: Start date for historical data
            end_date: End date for historical data
            granularity: Time granularity (e.g., "M1", "M5", "H1", "D")
        
        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            return await self.provider.get_historical_data(
                symbol, start_date, end_date, granularity
            )
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise
    
    async def stream_live_data(self, symbols: List[str]) -> AsyncIterator[MarketData]:
        """
        Stream live market data
        
        Args:
            symbols: List of symbols to stream
        
        Yields:
            MarketData objects as they arrive
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            async for data in self.provider.stream_live_data(symbols):
                # Convert provider-specific format to our MarketData format
                yield self._convert_to_market_data(data)
        except Exception as e:
            self.logger.error(f"Failed to stream live data: {e}")
            raise
    
    def _convert_to_market_data(self, data: Dict) -> MarketData:
        """Convert provider-specific data format to MarketData"""
        return MarketData(
            timestamp=data.get('timestamp', datetime.now()),
            symbol=data.get('symbol', ''),
            open=data.get('open', data.get('close', 0.0)),
            high=data.get('high', data.get('close', 0.0)),
            low=data.get('low', data.get('close', 0.0)),
            close=data.get('close', 0.0),
            volume=data.get('volume', 0.0)
        )
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        if not self.is_connected:
            await self.connect()
        
        try:
            return await self.provider.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        if not self.is_connected:
            await self.connect()
        
        try:
            return await self.provider.get_symbols()
        except Exception as e:
            self.logger.error(f"Failed to get symbols: {e}")
            return []
    
    async def get_market_status(self, symbol: str) -> Dict:
        """Get market status for a symbol"""
        if not self.is_connected:
            await self.connect()
        
        try:
            return await self.provider.get_market_status(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get market status for {symbol}: {e}")
            return {}


class MockDataProvider:
    """Mock data provider for testing when real providers aren't available"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Mock connection"""
        self.logger.info("Mock data provider connected")
        
    async def disconnect(self):
        """Mock disconnection"""
        self.logger.info("Mock data provider disconnected")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "M1"
    ) -> pd.DataFrame:
        """Generate mock historical data"""
        
        # Generate date range
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self._get_frequency(granularity)
        )
        
        # Generate mock OHLCV data with more realistic trends
        np.random.seed(42)
        base_price = 1.1000
        data = []
        
        # Create trending periods for more realistic signals
        trend_period = 100  # bars per trend
        trend_strength = 0.0005  # trend strength per bar
        
        for i, timestamp in enumerate(date_range):
            # Create trending behavior
            trend_phase = (i // trend_period) % 4  # 4 different trend phases
            
            if trend_phase == 0:  # Uptrend
                trend = trend_strength
            elif trend_phase == 1:  # Sideways
                trend = 0
            elif trend_phase == 2:  # Downtrend
                trend = -trend_strength
            else:  # Sideways
                trend = 0
            
            # Add trend to base price
            base_price += trend + np.random.normal(0, 0.0001)
            
            # Generate OHLC
            high = base_price + abs(np.random.normal(0, 0.0002))
            low = base_price - abs(np.random.normal(0, 0.0002))
            open_price = low + np.random.random() * (high - low)
            close = low + np.random.random() * (high - low)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(100, 1000)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Generated {len(df)} mock data points for {symbol}")
        return df
    
    async def stream_live_data(self, symbols: List[str]) -> AsyncIterator[Dict]:
        """Generate mock live data stream"""
        
        base_prices = {symbol: 1.1000 for symbol in symbols}
        
        while True:
            for symbol in symbols:
                # Generate mock tick
                change = np.random.normal(0, 0.00005)
                base_prices[symbol] += change
                
                yield {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'close': base_prices[symbol],
                    'volume': np.random.randint(10, 100)
                }
            
            await asyncio.sleep(0.1)  # 100ms between ticks
    
    async def get_current_price(self, symbol: str) -> float:
        """Get mock current price"""
        return 1.1000 + np.random.normal(0, 0.001)
    
    async def get_symbols(self) -> List[str]:
        """Return mock symbol list"""
        return ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'USD_CHF', 'USD_CAD']
    
    async def get_market_status(self, symbol: str) -> Dict:
        """Return mock market status"""
        return {
            'symbol': symbol,
            'status': 'open',
            'tradeable': True,
            'last_update': datetime.now().isoformat()
        }
    
    def _get_frequency(self, granularity: str) -> str:
        """Convert OANDA granularity to pandas frequency"""
        frequency_map = {
            'S5': '5s',    # 5 seconds
            'S10': '10s',  # 10 seconds
            'S15': '15s',  # 15 seconds
            'S30': '30s',  # 30 seconds
            'M1': '1min',  # 1 minute
            'M2': '2min',  # 2 minutes
            'M4': '4min',  # 4 minutes
            'M5': '5min',  # 5 minutes
            'M10': '10min',# 10 minutes
            'M15': '15min',# 15 minutes
            'M30': '30min',# 30 minutes
            'H1': '1h',    # 1 hour
            'H2': '2h',    # 2 hours
            'H3': '3h',    # 3 hours
            'H4': '4h',    # 4 hours
            'H6': '6h',    # 6 hours
            'H8': '8h',    # 8 hours
            'H12': '12h',  # 12 hours
            'D': '1D',     # 1 day
            'W': '1W',     # 1 week
            'M': '1M'      # 1 month
        }
        
        return frequency_map.get(granularity, '1min')


# Import numpy for mock data generation
try:
    import numpy as np
except ImportError:
    # Fallback to random if numpy not available
    import random
    
    class np:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def seed(s):
                    random.seed(s)
                
                @staticmethod
                def normal(mean, std):
                    # Simple Box-Muller transform
                    u1, u2 = random.random(), random.random()
                    z = ((-2 * math.log(u1)) ** 0.5) * math.cos(2 * math.pi * u2)
                    return mean + std * z
                
                @staticmethod
                def randint(low, high):
                    return random.randint(low, high)
                
                @staticmethod
                def random():
                    return random.random()
            
            return Random()
        
        random = random()