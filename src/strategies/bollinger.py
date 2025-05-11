"""
Bollinger Bands Trading Strategy
Mean reversion strategy using Bollinger Bands with various confirmation indicators
"""

from src.core.strategy import BaseStrategy, Signal
from src.analysis.indicators import BollingerBands, RSI, ATR, SMA
from datetime import datetime
from typing import Optional

class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands breakout/reversion strategy with multiple approaches
    
    Strategy Logic:
    1. Mean Reversion: Buy at lower band, sell at upper band
    2. Breakout: Buy/sell on band penetration 
    3. Squeeze: Trade on volatility expansion
    4. RSI Confirmation: Use RSI to confirm overbought/oversold
    """
    
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        
        # Default parameters
        self.default_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'atr_period': 14,
            'mode': 'reversion',  # 'reversion', 'breakout', 'squeeze'
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'squeeze_threshold': 0.02,  # For squeeze detection
            'stop_loss_atr': 1.5,
            'take_profit_atr': 2.5
        }
        
        # Update with provided parameters
        self.params = {**self.default_params, **parameters}
        
    async def _setup_indicators(self):
        """Initialize technical indicators"""
        self.indicators = {
            # Main Bollinger Bands
            'bollinger': BollingerBands(
                period=self.params['bb_period'],
                std_dev=self.params['bb_std_dev']
            ),
            
            # RSI for confirmation
            'rsi': RSI(period=self.params['rsi_period']),
            
            # ATR for stop loss calculation
            'atr': ATR(period=self.params['atr_period']),
            
            # SMA for trend direction
            'sma': SMA(period=self.params['bb_period']),
            
            # Secondary Bollinger Bands for squeeze detection
            'bollinger_tight': BollingerBands(
                period=self.params['bb_period'],
                std_dev=1.0
            ) if self.params['mode'] == 'squeeze' else None
        }
        
        # State tracking
        self.state = {
            'last_signal': None,
            'squeeze_detected': False,
            'trend_direction': None
        }
    
    async def _generate_signal(self, market_data) -> Optional[Signal]:
        """Generate trading signal based on Bollinger Band analysis"""
        
        # Get indicator values
        bb = self.indicators['bollinger']
        rsi = self.indicators['rsi']
        atr = self.indicators['atr']
        sma = self.indicators['sma']
        
        # Check if we have enough data
        if not all([bb.upper, bb.lower, bb.middle, rsi.value, atr.value, sma.value]):
            return Signal(
                action='hold',
                strength=0.0,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={'reason': 'Insufficient data'}
            )
        
        # Determine trading mode
        if self.params['mode'] == 'reversion':
            return await self._mean_reversion_signal(market_data)
        elif self.params['mode'] == 'breakout':
            return await self._breakout_signal(market_data)
        elif self.params['mode'] == 'squeeze':
            return await self._squeeze_signal(market_data)
        else:
            raise ValueError(f"Unknown mode: {self.params['mode']}")
    
    async def _mean_reversion_signal(self, market_data) -> Optional[Signal]:
        """Generate mean reversion signals"""
        bb = self.indicators['bollinger']
        rsi = self.indicators['rsi']
        atr = self.indicators['atr']
        
        # Calculate band position
        price = market_data.close
        band_width = bb.upper - bb.lower
        band_position = (price - bb.lower) / band_width if band_width > 0 else 0.5
        
        # Buy signal: Price at or below lower band with oversold RSI
        if price <= bb.lower and rsi.value <= self.params['rsi_oversold']:
            stop_loss = price - (atr.value * self.params['stop_loss_atr'])
            take_profit = bb.middle  # Target middle band
            
            strength = min((self.params['rsi_oversold'] - rsi.value) / 10, 1.0)
            
            return Signal(
                action='buy',
                strength=max(strength, 0.5),
                price=price,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'reason': 'Mean Reversion Buy',
                    'band_position': band_position,
                    'rsi': rsi.value,
                    'bb_lower': bb.lower,
                    'bb_upper': bb.upper,
                    'bb_middle': bb.middle,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            )
        
        # Sell signal: Price at or above upper band with overbought RSI
        elif price >= bb.upper and rsi.value >= self.params['rsi_overbought']:
            stop_loss = price + (atr.value * self.params['stop_loss_atr'])
            take_profit = bb.middle  # Target middle band
            
            strength = min((rsi.value - self.params['rsi_overbought']) / 10, 1.0)
            
            return Signal(
                action='sell',
                strength=max(strength, 0.5),
                price=price,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'reason': 'Mean Reversion Sell',
                    'band_position': band_position,
                    'rsi': rsi.value,
                    'bb_lower': bb.lower,
                    'bb_upper': bb.upper,
                    'bb_middle': bb.middle,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            )
        
        # No signal
        return Signal(
            action='hold',
            strength=0.0,
            price=price,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={
                'reason': 'No reversion signal',
                'band_position': band_position,
                'rsi': rsi.value
            }
        )
    
    async def _breakout_signal(self, market_data) -> Optional[Signal]:
        """Generate breakout signals"""
        bb = self.indicators['bollinger']
        rsi = self.indicators['rsi']
        atr = self.indicators['atr']
        sma = self.indicators['sma']
        
        price = market_data.close
        trend_direction = 'up' if price > sma.value else 'down'
        
        # Bullish breakout: Price breaks above upper band with trend
        if price > bb.upper and trend_direction == 'up' and rsi.value < 80:
            stop_loss = bb.upper - (atr.value * self.params['stop_loss_atr'])
            take_profit = price + (atr.value * self.params['take_profit_atr'])
            
            # Strength based on how far above the band and RSI momentum
            strength = min((price - bb.upper) / bb.upper + rsi.value / 100, 1.0)
            
            return Signal(
                action='buy',
                strength=max(strength, 0.6),
                price=price,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'reason': 'Bullish Breakout',
                    'trend': trend_direction,
                    'rsi': rsi.value,
                    'bb_upper': bb.upper,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            )
        
        # Bearish breakout: Price breaks below lower band with trend
        elif price < bb.lower and trend_direction == 'down' and rsi.value > 20:
            stop_loss = bb.lower + (atr.value * self.params['stop_loss_atr'])
            take_profit = price - (atr.value * self.params['take_profit_atr'])
            
            strength = min((bb.lower - price) / bb.lower + (100 - rsi.value) / 100, 1.0)
            
            return Signal(
                action='sell',
                strength=max(strength, 0.6),
                price=price,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'reason': 'Bearish Breakout',
                    'trend': trend_direction,
                    'rsi': rsi.value,
                    'bb_lower': bb.lower,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
            )
        
        # No signal
        return Signal(
            action='hold',
            strength=0.0,
            price=price,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={
                'reason': 'No breakout signal',
                'trend': trend_direction,
                'rsi': rsi.value
            }
        )
    
    async def _squeeze_signal(self, market_data) -> Optional[Signal]:
        """Generate squeeze signals - trade on volatility expansion"""
        bb = self.indicators['bollinger']
        bb_tight = self.indicators['bollinger_tight']
        rsi = self.indicators['rsi']
        atr = self.indicators['atr']
        sma = self.indicators['sma']
        
        if not bb_tight:
            return Signal(action='hold', strength=0.0, price=market_data.close,
                         symbol=market_data.symbol, timestamp=market_data.timestamp,
                         metadata={'reason': 'Squeeze mode requires bollinger_tight'})
        
        price = market_data.close
        
        # Calculate band width ratio for squeeze detection
        wide_band_width = bb.upper - bb.lower
        tight_band_width = bb_tight.upper - bb_tight.lower
        band_ratio = wide_band_width / tight_band_width if tight_band_width > 0 else 0
        
        # Detect squeeze (Bollinger Bands contracting)
        if band_ratio < (2.0 + self.params['squeeze_threshold']):
            if not self.state['squeeze_detected']:
                self.state['squeeze_detected'] = True
                # No trade during squeeze, wait for expansion
                return Signal(
                    action='hold',
                    strength=0.0,
                    price=price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Squeeze detected - waiting for expansion',
                        'band_ratio': band_ratio,
                        'squeeze': True
                    }
                )
        
        # Squeeze expansion detected
        elif self.state['squeeze_detected'] and band_ratio > (2.0 + self.params['squeeze_threshold']):
            self.state['squeeze_detected'] = False
            
            # Determine trade direction based on price action and RSI
            trend_direction = 'up' if price > sma.value else 'down'
            
            # Bullish expansion
            if price > bb.middle and rsi.value > 50:
                stop_loss = bb.middle - (atr.value * self.params['stop_loss_atr'])
                take_profit = price + (atr.value * self.params['take_profit_atr'])
                
                return Signal(
                    action='buy',
                    strength=0.8,
                    price=price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Bullish Squeeze Expansion',
                        'band_ratio': band_ratio,
                        'trend': trend_direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                )
            
            # Bearish expansion
            elif price < bb.middle and rsi.value < 50:
                stop_loss = bb.middle + (atr.value * self.params['stop_loss_atr'])
                take_profit = price - (atr.value * self.params['take_profit_atr'])
                
                return Signal(
                    action='sell',
                    strength=0.8,
                    price=price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Bearish Squeeze Expansion',
                        'band_ratio': band_ratio,
                        'trend': trend_direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    }
                )
        
        # No signal
        return Signal(
            action='hold',
            strength=0.0,
            price=price,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={
                'reason': 'No squeeze expansion signal',
                'band_ratio': band_ratio,
                'squeeze': self.state['squeeze_detected']
            }
        )
    
    async def _check_exit_conditions(self, market_data) -> Optional[Signal]:
        """Check if current position should be exited"""
        if not hasattr(self, 'position_info'):
            return None
        
        bb = self.indicators['bollinger']
        price = market_data.close
        
        # Check if we've reached profit targets or stop loss
        if self.position_info['direction'] == 'long':
            # Exit long position
            if price >= bb.middle or price <= self.position_info['stop_loss']:
                return Signal(
                    action='sell',
                    strength=1.0,
                    price=price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Exit Long Position',
                        'entry_price': self.position_info['entry_price'],
                        'current_price': price
                    }
                )
        
        elif self.position_info['direction'] == 'short':
            # Exit short position
            if price <= bb.middle or price >= self.position_info['stop_loss']:
                return Signal(
                    action='buy',  # Buy to close short
                    strength=1.0,
                    price=price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Exit Short Position',
                        'entry_price': self.position_info['entry_price'],
                        'current_price': price
                    }
                )
        
        return None

# Additional utility class for Bollinger Band analysis
class BollingerBandAnalyzer:
    """Utility class for advanced Bollinger Band analysis"""
    
    @staticmethod
    def calculate_band_width(bb: BollingerBands) -> float:
        """Calculate Bollinger Band width percentage"""
        if bb.upper and bb.lower and bb.middle:
            width = (bb.upper - bb.lower) / bb.middle * 100
            return width
        return 0.0
    
    @staticmethod
    def calculate_percent_b(price: float, bb: BollingerBands) -> float:
        """Calculate %B indicator (position within bands)"""
        if bb.upper and bb.lower:
            return (price - bb.lower) / (bb.upper - bb.lower) * 100
        return 50.0
    
    @staticmethod
    def detect_squeeze(current_width: float, historical_widths: list, 
                      lookback: int = 20, threshold: float = 0.9) -> bool:
        """Detect Bollinger Band squeeze"""
        if len(historical_widths) < lookback:
            return False
        
        recent_avg = sum(historical_widths[-lookback:]) / lookback
        return current_width < recent_avg * threshold
    
    @staticmethod
    def identify_band_walk(prices: list, bb_uppers: list, bb_lowers: list, 
                          min_periods: int = 3) -> str:
        """Identify if price is walking along a band"""
        if len(prices) < min_periods:
            return 'none'
        
        # Check upper band walk
        upper_walk = sum(1 for i in range(-min_periods, 0) 
                        if prices[i] >= bb_uppers[i] * 0.95)
        
        # Check lower band walk
        lower_walk = sum(1 for i in range(-min_periods, 0) 
                        if prices[i] <= bb_lowers[i] * 1.05)
        
        if upper_walk >= min_periods * 0.7:
            return 'upper'
        elif lower_walk >= min_periods * 0.7:
            return 'lower'
        
        return 'none'