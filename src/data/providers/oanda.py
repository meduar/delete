"""
OANDA Data Provider
Implements data feed using OANDA API
"""

import asyncio
import aiohttp
import oandapyV20
from oandapyV20.endpoints.pricing import PricingStream, PricingInfo
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.accounts import AccountInstruments
from oandapyV20.exceptions import V20Error
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncIterator
import logging


class OandaDataProvider:
    """OANDA data provider implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key')
        self.account_id = config.get('account_id')
        self.environment = config.get('environment', 'practice')
        
        self.client = None
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize OANDA client
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
    
    async def connect(self):
        """Connect to OANDA API"""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession()
            
            # Test connection by getting account info
            if self.client:
                # Simple test request to verify credentials
                pass
            
            self.logger.info("Connected to OANDA")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OANDA: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from OANDA API"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.logger.info("Disconnected from OANDA")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "M1"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from OANDA
        
        Args:
            symbol: Currency pair (e.g., "EUR_USD")
            start_date: Start date for historical data
            end_date: End date for historical data
            granularity: Timeframe (e.g., "M1", "M5", "H1", "D")
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format (OANDA uses EUR_USD)
            instrument = symbol.replace('/', '_')
            
            # Prepare parameters
            params = {
                "granularity": granularity,
                "from": start_date.isoformat() + 'Z',
                "to": end_date.isoformat() + 'Z',
                "price": "M"  # Mid prices
            }
            
            # Make request
            request = InstrumentsCandles(instrument=instrument, params=params)
            response = self.client.request(request)
            
            # Process response
            if not response.get('candles'):
                self.logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for candle in response['candles']:
                if candle.get('complete', False):
                    data.append({
                        'timestamp': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except V20Error as e:
            self.logger.error(f"OANDA API error getting historical data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise
    
    async def stream_live_data(self, symbols: List[str]) -> AsyncIterator[Dict]:
        """
        Stream live market data from OANDA
        
        Args:
            symbols: List of symbols to stream
        
        Yields:
            Market data dictionaries
        """
        # Convert symbol formats
        instruments = [symbol.replace('/', '_') for symbol in symbols]
        
        try:
            # Set up streaming
            params = {"instruments": ",".join(instruments)}
            stream = PricingStream(accountID=self.account_id, params=params)
            
            # Stream data
            async for response in self._stream_prices(stream):
                if 'prices' in response:
                    for price in response['prices']:
                        yield self._parse_price_tick(price)
                        
        except V20Error as e:
            self.logger.error(f"OANDA streaming error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error streaming live data: {e}")
            raise
    
    async def _stream_prices(self, stream):
        """Async generator for price streaming"""
        loop = asyncio.get_event_loop()
        
        def stream_prices():
            try:
                for response in self.client.request(stream):
                    yield response
            except V20Error as e:
                self.logger.error(f"Streaming error: {e}")
                raise
        
        # Run in thread pool to avoid blocking
        for response in await loop.run_in_executor(None, stream_prices):
            yield response
    
    def _parse_price_tick(self, price: Dict) -> Dict:
        """Parse OANDA price tick to standard format"""
        return {
            'timestamp': pd.to_datetime(price['time']),
            'symbol': price['instrument'].replace('_', '/'),
            'close': (float(price['asks'][0]['price']) + float(price['bids'][0]['price'])) / 2,
            'bid': float(price['bids'][0]['price']),
            'ask': float(price['asks'][0]['price']),
            'spread': float(price['asks'][0]['price']) - float(price['bids'][0]['price'])
        }
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            instrument = symbol.replace('/', '_')
            params = {"instruments": instrument}
            request = PricingInfo(accountID=self.account_id, params=params)
            response = self.client.request(request)
            
            if 'prices' in response and response['prices']:
                price = response['prices'][0]
                # Return mid price
                return (float(price['asks'][0]['price']) + float(price['bids'][0]['price'])) / 2
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols from OANDA"""
        try:
            request = AccountInstruments(accountID=self.account_id)
            response = self.client.request(request)
            
            if 'instruments' in response:
                # Convert OANDA format to standard format
                symbols = [inst['name'].replace('_', '/') for inst in response['instruments']]
                return symbols
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return []
    
    async def get_market_status(self, symbol: str) -> Dict:
        """Get market status for a symbol"""
        try:
            instrument = symbol.replace('/', '_')
            request = AccountInstruments(accountID=self.account_id)
            response = self.client.request(request)
            
            if 'instruments' in response:
                for inst in response['instruments']:
                    if inst['name'] == instrument:
                        return {
                            'symbol': symbol,
                            'tradeable': inst.get('tradeable', False),
                            'margin_rate': inst.get('marginRate', '0'),
                            'maximum_order_units': inst.get('maximumOrderUnits', '0'),
                            'minimum_trade_size': inst.get('minimumTradeSize', '0'),
                            'last_update': datetime.now().isoformat()
                        }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting market status for {symbol}: {e}")
            return {}
    
    async def get_account_summary(self) -> Dict:
        """Get account summary information"""
        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(accountID=self.account_id)
            response = self.client.request(request)
            
            if 'account' in response:
                return response['account']
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {}
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions"""
        try:
            from oandapyV20.endpoints.positions import PositionList
            request = PositionList(accountID=self.account_id)
            response = self.client.request(request)
            
            if 'positions' in response:
                return [
                    {
                        'symbol': pos['instrument'].replace('_', '/'),
                        'long_units': int(pos['long']['units']),
                        'short_units': int(pos['short']['units']),
                        'unrealized_pl': float(pos['unrealizedPL'])
                    }
                    for pos in response['positions']
                    if int(pos['long']['units']) != 0 or int(pos['short']['units']) != 0
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []