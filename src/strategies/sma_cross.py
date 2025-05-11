"""
Simple Moving Average Crossover Strategy
Classic trend-following strategy using two moving averages
"""

from typing import Optional
from datetime import datetime

from src.core.strategy import BaseStrategy, Signal
from src.analysis.indicators import SMA, ATR, CrossDetector


class SmaCrossStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy
    
    Strategy Logic:
    - Buy when fast SMA crosses above slow SMA (Golden Cross)
    - Sell when fast SMA crosses below slow SMA (Death Cross)
    - Use ATR for stop loss and take profit levels
    """
    
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        
        # Default parameters
        self.default_params = {
            'fast_period': 10,
            'slow_period': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'symbol': 'EUR_USD'
        }
        
        # Update with provided parameters
        self.parameters = {**self.default_params, **parameters}
        
    async def _setup_indicators(self):
        """Initialize technical indicators"""
        self.indicators = {
            'fast_sma': SMA(period=self.parameters['fast_period']),
            'slow_sma': SMA(period=self.parameters['slow_period']),
            'atr': ATR(period=self.parameters['atr_period']),
            'cross_detector': CrossDetector()
        }
        
        # Initialize state
        self.state = {
            'position': None,
            'entry_price': None,
            'entry_time': None,
            'last_signal': None
        }
    
    async def _update_indicators(self, market_data):
        """Update all indicators with new data - override parent method"""
        # Update price-based indicators (SMA)
        self.indicators['fast_sma'].update(market_data.close)
        self.indicators['slow_sma'].update(market_data.close)
        
        # Update OHLC-based indicators (ATR)
        self.indicators['atr'].update(market_data)
        
        # Update cross detector with current SMA values
        fast_sma = self.indicators['fast_sma'].value
        slow_sma = self.indicators['slow_sma'].value
        
        if fast_sma is not None and slow_sma is not None:
            self.indicators['cross_detector'].update(fast_sma, slow_sma)
    
    async def _generate_signal(self, market_data) -> Optional[Signal]:
        """Generate trading signal based on SMA crossover"""
        
        # Get current indicator values
        fast_sma = self.indicators['fast_sma'].value
        slow_sma = self.indicators['slow_sma'].value
        atr = self.indicators['atr'].value
        cross_detector = self.indicators['cross_detector']
        
        # Check if we have enough data
        if fast_sma is None or slow_sma is None or atr is None:
            return Signal(
                action='hold',
                strength=0.0,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={'reason': 'Insufficient data'}
            )
        
        # Golden Cross - Bullish signal
        if cross_detector.bullish_cross:
            stop_loss = market_data.close - (atr * self.parameters['stop_loss_atr'])
            take_profit = market_data.close + (atr * self.parameters['take_profit_atr'])
            
            # Calculate signal strength based on gap between SMAs
            gap_pct = (fast_sma - slow_sma) / slow_sma * 100
            strength = min(max(gap_pct / 2, 0.5), 1.0)
            
            return Signal(
                action='buy',
                strength=strength,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'atr': atr,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'Golden Cross',
                    'gap_pct': gap_pct
                }
            )
        
        # Death Cross - Bearish signal
        elif cross_detector.bearish_cross:
            stop_loss = market_data.close + (atr * self.parameters['stop_loss_atr'])
            take_profit = market_data.close - (atr * self.parameters['take_profit_atr'])
            
            # Calculate signal strength based on gap between SMAs
            gap_pct = (slow_sma - fast_sma) / slow_sma * 100
            strength = min(max(gap_pct / 2, 0.5), 1.0)
            
            return Signal(
                action='sell',
                strength=strength,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'fast_sma': fast_sma,
                    'slow_sma': slow_sma,
                    'atr': atr,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': 'Death Cross',
                    'gap_pct': gap_pct
                }
            )
        
        # No signal - Hold
        return Signal(
            action='hold',
            strength=0.0,
            price=market_data.close,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={
                'fast_sma': fast_sma,
                'slow_sma': slow_sma,
                'atr': atr,
                'reason': 'No crossover detected'
            }
        )
    
    async def _check_exit_conditions(self, market_data) -> Optional[Signal]:
        """Check if current position should be exited"""
        if self.state['position'] is None:
            return None
        
        current_price = market_data.close
        entry_price = self.state['entry_price']
        atr = self.indicators['atr'].value
        
        if atr is None:
            return None
        
        # Check stop loss and take profit levels
        if self.state['position'] == 'long':
            # Long position exit conditions
            stop_loss = entry_price - (atr * self.parameters['stop_loss_atr'])
            take_profit = entry_price + (atr * self.parameters['take_profit_atr'])
            
            if current_price <= stop_loss:
                return Signal(
                    action='sell',
                    strength=1.0,
                    price=current_price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Stop Loss Hit',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'pnl': current_price - entry_price
                    }
                )
            elif current_price >= take_profit:
                return Signal(
                    action='sell',
                    strength=1.0,
                    price=current_price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Take Profit Hit',
                        'entry_price': entry_price,
                        'take_profit': take_profit,
                        'pnl': current_price - entry_price
                    }
                )
        
        elif self.state['position'] == 'short':
            # Short position exit conditions
            stop_loss = entry_price + (atr * self.parameters['stop_loss_atr'])
            take_profit = entry_price - (atr * self.parameters['take_profit_atr'])
            
            if current_price >= stop_loss:
                return Signal(
                    action='buy',  # Buy to close short
                    strength=1.0,
                    price=current_price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Stop Loss Hit',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'pnl': entry_price - current_price
                    }
                )
            elif current_price <= take_profit:
                return Signal(
                    action='buy',  # Buy to close short
                    strength=1.0,
                    price=current_price,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'reason': 'Take Profit Hit',
                        'entry_price': entry_price,
                        'take_profit': take_profit,
                        'pnl': entry_price - current_price
                    }
                )
        
        return None
    
    async def update(self, market_data) -> Optional[Signal]:
        """
        Update strategy with new market data
        Override parent method to include exit checks
        """
        # First check exit conditions for existing position
        exit_signal = await self._check_exit_conditions(market_data)
        if exit_signal:
            # Update position state on exit
            self.state['position'] = None
            self.state['entry_price'] = None
            self.state['entry_time'] = None
            return exit_signal
        
        # Call parent update for indicator updates and signal generation
        signal = await super().update(market_data)
        
        # Update position state on entry
        if signal and signal.action in ['buy', 'sell']:
            if signal.action == 'buy':
                self.state['position'] = 'long'
            elif signal.action == 'sell':
                self.state['position'] = 'short'
            
            self.state['entry_price'] = signal.price
            self.state['entry_time'] = signal.timestamp
        
        return signal


# Variations of SMA Strategy

class AdaptiveSmaCrossStrategy(SmaCrossStrategy):
    """
    Adaptive SMA Strategy that adjusts periods based on volatility
    """
    
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        
        # Additional parameters for adaptive behavior
        self.parameters.update({
            'volatility_lookback': 20,
            'fast_range': (5, 15),
            'slow_range': (20, 40)
        })
    
    async def _setup_indicators(self):
        """Initialize indicators with dynamic periods"""
        await super()._setup_indicators()
        
        # Add volatility indicator
        from src.analysis.indicators import ROC
        self.indicators['volatility'] = ROC(period=self.parameters['volatility_lookback'])
    
    async def _adjust_periods_based_on_volatility(self):
        """Adjust SMA periods based on market volatility"""
        volatility = self.indicators['volatility'].value
        
        if volatility is not None:
            # Higher volatility -> shorter periods (more responsive)
            # Lower volatility -> longer periods (more stable)
            
            fast_range = self.parameters['fast_range']
            slow_range = self.parameters['slow_range']
            
            # Map volatility to period ranges
            normalized_vol = min(max(abs(volatility), 0), 10) / 10  # Normalize to 0-1
            
            new_fast_period = int(fast_range[0] + (fast_range[1] - fast_range[0]) * (1 - normalized_vol))
            new_slow_period = int(slow_range[0] + (slow_range[1] - slow_range[0]) * (1 - normalized_vol))
            
            # Update indicators if periods changed significantly
            if abs(new_fast_period - self.parameters['fast_period']) > 2:
                self.parameters['fast_period'] = new_fast_period
                self.indicators['fast_sma'] = SMA(period=new_fast_period)
            
            if abs(new_slow_period - self.parameters['slow_period']) > 2:
                self.parameters['slow_period'] = new_slow_period
                self.indicators['slow_sma'] = SMA(period=new_slow_period)


class TripleSmaCrossStrategy(BaseStrategy):
    """
    Triple SMA Strategy using three moving averages
    More sophisticated than double SMA
    """
    
    def __init__(self, name: str, parameters: dict):
        super().__init__(name, parameters)
        
        # Default parameters
        self.default_params = {
            'fast_period': 5,
            'medium_period': 15,
            'slow_period': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'confirm_strength': True
        }
        
        # Update with provided parameters
        self.parameters = {**self.default_params, **parameters}
    
    async def _setup_indicators(self):
        """Initialize three SMAs and ATR"""
        self.indicators = {
            'fast_sma': SMA(period=self.parameters['fast_period']),
            'medium_sma': SMA(period=self.parameters['medium_period']),
            'slow_sma': SMA(period=self.parameters['slow_period']),
            'atr': ATR(period=self.parameters['atr_period']),
            'cross_detector_fast_medium': CrossDetector(),
            'cross_detector_medium_slow': CrossDetector()
        }
    
    async def _generate_signal(self, market_data) -> Optional[Signal]:
        """Generate signal based on triple SMA alignment"""
        
        # Get indicator values
        fast_sma = self.indicators['fast_sma'].value
        medium_sma = self.indicators['medium_sma'].value
        slow_sma = self.indicators['slow_sma'].value
        atr = self.indicators['atr'].value
        
        # Check if we have enough data
        if any(v is None for v in [fast_sma, medium_sma, slow_sma, atr]):
            return Signal(
                action='hold',
                strength=0.0,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={'reason': 'Insufficient data'}
            )
        
        # Update cross detectors
        self.indicators['cross_detector_fast_medium'].update(fast_sma, medium_sma)
        self.indicators['cross_detector_medium_slow'].update(medium_sma, slow_sma)
        
        # Check for bullish alignment: Fast > Medium > Slow
        if fast_sma > medium_sma > slow_sma:
            # Look for entry on fast-medium cross
            if self.indicators['cross_detector_fast_medium'].bullish_cross:
                strength = self._calculate_alignment_strength(fast_sma, medium_sma, slow_sma)
                
                return Signal(
                    action='buy',
                    strength=strength,
                    price=market_data.close,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'fast_sma': fast_sma,
                        'medium_sma': medium_sma,
                        'slow_sma': slow_sma,
                        'reason': 'Triple SMA Bullish Alignment',
                        'stop_loss': market_data.close - (atr * self.parameters['stop_loss_atr']),
                        'take_profit': market_data.close + (atr * self.parameters['take_profit_atr'])
                    }
                )
        
        # Check for bearish alignment: Fast < Medium < Slow
        elif fast_sma < medium_sma < slow_sma:
            # Look for entry on fast-medium cross
            if self.indicators['cross_detector_fast_medium'].bearish_cross:
                strength = self._calculate_alignment_strength(slow_sma, medium_sma, fast_sma)
                
                return Signal(
                    action='sell',
                    strength=strength,
                    price=market_data.close,
                    symbol=market_data.symbol,
                    timestamp=market_data.timestamp,
                    metadata={
                        'fast_sma': fast_sma,
                        'medium_sma': medium_sma,
                        'slow_sma': slow_sma,
                        'reason': 'Triple SMA Bearish Alignment',
                        'stop_loss': market_data.close + (atr * self.parameters['stop_loss_atr']),
                        'take_profit': market_data.close - (atr * self.parameters['take_profit_atr'])
                    }
                )
        
        # No signal
        return Signal(
            action='hold',
            strength=0.0,
            price=market_data.close,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={
                'fast_sma': fast_sma,
                'medium_sma': medium_sma,
                'slow_sma': slow_sma,
                'reason': 'No triple SMA alignment'
            }
        )
    
    def _calculate_alignment_strength(self, top: float, middle: float, bottom: float) -> float:
        """Calculate strength based on SMA separation"""
        # Calculate gaps between SMAs
        gap1 = abs(top - middle) / middle
        gap2 = abs(middle - bottom) / bottom
        
        # Stronger signal when SMAs are well separated
        max_gap = 0.02  # 2% maximum gap for full strength
        avg_gap = (gap1 + gap2) / 2
        
        strength = min(avg_gap / max_gap, 1.0)
        return max(strength, 0.5)  # Minimum strength of 0.5