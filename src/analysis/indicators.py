"""
Technical Analysis Indicators
Complete implementation of all required indicators
"""

import numpy as np
from collections import deque
from typing import Optional, List
from dataclasses import dataclass


class BaseIndicator:
    """Base class for all technical indicators"""
    
    def __init__(self, period: int):
        self.period = period
        self.data = deque(maxlen=period + 1)
        self.value: Optional[float] = None
        
    def update(self, price: float) -> Optional[float]:
        """Update indicator with new price data"""
        self.data.append(price)
        self.value = self._calculate()
        return self.value
        
    def _calculate(self) -> Optional[float]:
        """Calculate indicator value - to be implemented by subclasses"""
        raise NotImplementedError


class SMA(BaseIndicator):
    """Simple Moving Average"""
    
    def _calculate(self) -> Optional[float]:
        if len(self.data) < self.period:
            return None
        return sum(list(self.data)[-self.period:]) / self.period


class EMA(BaseIndicator):
    """Exponential Moving Average"""
    
    def __init__(self, period: int):
        super().__init__(period)
        self.multiplier = 2 / (period + 1)
        
    def _calculate(self) -> Optional[float]:
        if len(self.data) < 1:
            return None
            
        current_price = self.data[-1]
        
        if self.value is None:
            # First calculation - use SMA
            if len(self.data) >= self.period:
                return sum(list(self.data)[-self.period:]) / self.period
            return None
        
        # EMA calculation
        return (current_price * self.multiplier) + (self.value * (1 - self.multiplier))


class ATR(BaseIndicator):
    """Average True Range"""
    
    def __init__(self, period: int):
        super().__init__(period)
        self.highs = deque(maxlen=period + 1)
        self.lows = deque(maxlen=period + 1)
        self.closes = deque(maxlen=period + 1)
        
    def update(self, market_data) -> Optional[float]:
        """Update ATR with OHLC data"""
        self.highs.append(market_data.high)
        self.lows.append(market_data.low)
        self.closes.append(market_data.close)
        self.value = self._calculate()
        return self.value
        
    def _calculate(self) -> Optional[float]:
        if len(self.closes) < 2:
            return None
            
        true_ranges = []
        for i in range(1, len(self.closes)):
            high_low = self.highs[i] - self.lows[i]
            high_close = abs(self.highs[i] - self.closes[i-1])
            low_close = abs(self.lows[i] - self.closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
            
        if len(true_ranges) >= self.period:
            return sum(true_ranges[-self.period:]) / self.period
        return None


class BollingerBands:
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.sma = SMA(period)
        self.data = deque(maxlen=period + 1)
        
        self.upper: Optional[float] = None
        self.middle: Optional[float] = None
        self.lower: Optional[float] = None
        
    def update(self, price: float):
        """Update Bollinger Bands with new price"""
        self.data.append(price)
        self.middle = self.sma.update(price)
        
        if self.middle is None or len(self.data) < self.period:
            return
            
        # Calculate standard deviation
        prices = list(self.data)[-self.period:]
        std = np.std(prices)
        
        self.upper = self.middle + (std * self.std_dev)
        self.lower = self.middle - (std * self.std_dev)


class RSI(BaseIndicator):
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        super().__init__(period)
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.prev_price: Optional[float] = None
        
    def update(self, price: float) -> Optional[float]:
        """Update RSI with new price"""
        if self.prev_price is not None:
            change = price - self.prev_price
            if change > 0:
                self.gains.append(change)
                self.losses.append(0)
            else:
                self.gains.append(0)
                self.losses.append(abs(change))
        
        self.prev_price = price
        self.value = self._calculate()
        return self.value
        
    def _calculate(self) -> Optional[float]:
        if len(self.gains) < self.period:
            return None
            
        avg_gain = sum(self.gains) / self.period
        avg_loss = sum(self.losses) / self.period
        
        if avg_loss == 0:
            return 100
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class ROC(BaseIndicator):
    """Rate of Change (Momentum) indicator"""
    
    def __init__(self, period: int = 12):
        super().__init__(period)
        
    def _calculate(self) -> Optional[float]:
        if len(self.data) < self.period + 1:
            return None
            
        current_price = self.data[-1]
        previous_price = self.data[-self.period - 1]
        
        if previous_price == 0:
            return 0
            
        return ((current_price - previous_price) / previous_price) * 100


class MACD:
    """Moving Average Convergence Divergence"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.signal_ema = EMA(signal_period)
        
        self.macd_line: Optional[float] = None
        self.signal_line: Optional[float] = None
        self.histogram: Optional[float] = None
        
    def update(self, price: float):
        """Update MACD with new price"""
        fast_value = self.fast_ema.update(price)
        slow_value = self.slow_ema.update(price)
        
        if fast_value is None or slow_value is None:
            return
            
        self.macd_line = fast_value - slow_value
        self.signal_line = self.signal_ema.update(self.macd_line)
        
        if self.signal_line is not None:
            self.histogram = self.macd_line - self.signal_line


class Stochastic:
    """Stochastic Oscillator"""
    
    def __init__(self, period: int = 14, k_period: int = 3, d_period: int = 3):
        self.period = period
        self.k_sma = SMA(k_period)
        self.d_sma = SMA(d_period)
        
        self.highs = deque(maxlen=period)
        self.lows = deque(maxlen=period)
        self.closes = deque(maxlen=period)
        
        self.k_percent: Optional[float] = None
        self.d_percent: Optional[float] = None
        
    def update(self, market_data):
        """Update Stochastic with OHLC data"""
        self.highs.append(market_data.high)
        self.lows.append(market_data.low)
        self.closes.append(market_data.close)
        
        if len(self.closes) < self.period:
            return
            
        highest_high = max(self.highs)
        lowest_low = min(self.lows)
        current_close = self.closes[-1]
        
        if highest_high == lowest_low:
            raw_k = 50
        else:
            raw_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
        self.k_percent = self.k_sma.update(raw_k)
        
        if self.k_percent is not None:
            self.d_percent = self.d_sma.update(self.k_percent)


class CrossDetector:
    """Detect crossovers between two series"""
    
    def __init__(self):
        self.prev_fast: Optional[float] = None
        self.prev_slow: Optional[float] = None
        self.bullish_cross = False
        self.bearish_cross = False
        
    def update(self, fast: float, slow: float):
        """Update with new values and detect crosses"""
        self.bullish_cross = False
        self.bearish_cross = False
        
        if self.prev_fast is not None and self.prev_slow is not None:
            # Check for bullish cross (fast crosses above slow)
            if self.prev_fast <= self.prev_slow and fast > slow:
                self.bullish_cross = True
                
            # Check for bearish cross (fast crosses below slow)
            elif self.prev_fast >= self.prev_slow and fast < slow:
                self.bearish_cross = True
        
        self.prev_fast = fast
        self.prev_slow = slow
        
    def crossed_above(self, value: float) -> bool:
        """Check if the fast line crossed above the given value"""
        if self.prev_fast is None:
            return False
        return self.prev_fast <= value and self.prev_fast > value
        
    def crossed_below(self, value: float) -> bool:
        """Check if the fast line crossed below the given value"""
        if self.prev_fast is None:
            return False
        return self.prev_fast >= value and self.prev_fast < value


class VolumeWeightedAveragePrice:
    """Volume Weighted Average Price"""
    
    def __init__(self):
        self.volume_price_sum = 0
        self.volume_sum = 0
        self.vwap: Optional[float] = None
        
    def update(self, market_data):
        """Update VWAP with new price and volume data"""
        price = market_data.close
        volume = market_data.volume
        
        self.volume_price_sum += price * volume
        self.volume_sum += volume
        
        if self.volume_sum > 0:
            self.vwap = self.volume_price_sum / self.volume_sum


class OrderBookAnalyzer:
    """Order book analysis for support/resistance levels"""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.high_volume_levels: List[float] = []
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []
        
    def analyze_volume_profile(self, price_volume_data: List[tuple]):
        """Analyze volume profile to find high volume nodes"""
        # Sort by volume (descending)
        sorted_data = sorted(price_volume_data, key=lambda x: x[1], reverse=True)
        
        # Find top volume levels
        top_volume_levels = sorted_data[:5]
        self.high_volume_levels = [level[0] for level in top_volume_levels]
        
        # Classify as support or resistance based on current price
        if price_volume_data:
            current_price = price_volume_data[-1][0]
            
            self.support_levels = [
                level for level in self.high_volume_levels 
                if level < current_price
            ]
            
            self.resistance_levels = [
                level for level in self.high_volume_levels 
                if level > current_price
            ]