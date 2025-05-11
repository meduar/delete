from src.core.strategy import BaseStrategy, Signal
from src.analysis.indicators import RSI, ROC, EMA
from datetime import datetime

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    async def _setup_indicators(self):
        """Initialize momentum indicators"""
        self.indicators = {
            'rsi': RSI(period=self.parameters['rsi_period']),
            'roc': ROC(period=self.parameters['roc_period']),
            'ema': EMA(period=self.parameters['ema_period'])
        }
        
    async def _generate_signal(self, market_data) -> Signal:
        """Generate signal based on momentum indicators"""
        rsi = self.indicators['rsi'].value
        roc = self.indicators['roc'].value
        ema = self.indicators['ema'].value
        
        # Check if we have enough data
        if rsi is None or roc is None or ema is None:
            return Signal(
                action='hold',
                strength=0.0,
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={}
            )
        
        # Strong momentum bullish signal
        if (rsi < 70 and rsi > 50 and 
            roc > self.parameters['roc_threshold'] and 
            market_data.close > ema):
            
            return Signal(
                action='buy',
                strength=min(roc / 10, 1.0),  # Strength based on ROC
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'rsi': rsi,
                    'roc': roc,
                    'ema': ema,
                    'reason': 'Strong Bullish Momentum'
                }
            )
        
        # Momentum reversal bearish signal
        elif (rsi > 70 and roc < -self.parameters['roc_threshold'] and 
              market_data.close < ema):
            
            return Signal(
                action='sell',
                strength=min(abs(roc) / 10, 1.0),
                price=market_data.close,
                symbol=market_data.symbol,
                timestamp=market_data.timestamp,
                metadata={
                    'rsi': rsi,
                    'roc': roc,
                    'ema': ema,
                    'reason': 'Momentum Reversal'
                }
            )
        
        return Signal(
            action='hold',
            strength=0.0,
            price=market_data.close,
            symbol=market_data.symbol,
            timestamp=market_data.timestamp,
            metadata={'rsi': rsi, 'roc': roc, 'ema': ema}
        )
