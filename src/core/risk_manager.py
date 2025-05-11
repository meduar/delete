from typing import Dict, Optional
from src.core.strategy import Signal
from src.core.portfolio import Portfolio
from src.utils.config import Config
import logging

class RiskManager:
    """Risk management system for trading operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_position_size = config.get_nested('trading', 'max_position_size', 0.1)
        self.max_daily_loss = config.get_nested('trading', 'max_daily_loss', -1000)
        self.max_trades_per_day = config.get_nested('trading', 'max_trades_per_day', 10)
        self.max_portfolio_drawdown = config.get_nested('trading', 'max_portfolio_drawdown', 0.15)
        
        # Current day tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_date = None
        
    async def evaluate_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
        market_data: Dict
    ) -> Optional[Signal]:
        """Evaluate and potentially modify a trading signal based on risk rules"""
        
        # Check if signal should be allowed
        if not await self._check_signal_allowed(signal, portfolio, market_data):
            return None
        
        # Adjust position size if needed
        approved_signal = await self._adjust_position_size(signal, portfolio, market_data)
        
        return approved_signal
    
    async def _check_signal_allowed(
        self,
        signal: Signal,
        portfolio: Portfolio,
        market_data: Dict
    ) -> bool:
        """Check if a signal should be allowed based on risk rules"""
        
        # Check daily trade limit
        current_date = signal.timestamp.date()
        if self.last_trade_date != current_date:
            # Reset daily counters
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date
        
        if self.daily_trades >= self.max_trades_per_day:
            self.logger.warning(f"Daily trade limit reached: {self.daily_trades}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            self.logger.warning(f"Daily loss limit exceeded: {self.daily_pnl}")
            return False
        
        # Check portfolio drawdown
        current_prices = {signal.symbol: market_data.get('close', signal.price)}
        total_value = portfolio.total_value(current_prices)
        peak_value = max(total_value, getattr(self, '_peak_value', portfolio.initial_cash))
        self._peak_value = peak_value
        
        drawdown = (peak_value - total_value) / peak_value
        if drawdown > self.max_portfolio_drawdown:
            self.logger.warning(f"Portfolio drawdown limit exceeded: {drawdown:.2%}")
            return False
        
        # Check position concentration
        if signal.action == 'buy':
            position_value = signal.price * portfolio.calculate_position_size(signal, signal.price)
            position_pct = position_value / total_value
            
            if position_pct > self.max_position_size:
                self.logger.warning(f"Position size limit exceeded: {position_pct:.2%}")
                return False
        
        return True
    
    async def _adjust_position_size(
        self,
        signal: Signal,
        portfolio: Portfolio,
        market_data: Dict
    ) -> Signal:
        """Adjust position size based on risk parameters"""
        
        if signal.action != 'buy':
            return signal
        
        # Calculate risk-adjusted position size
        current_prices = {signal.symbol: market_data.get('close', signal.price)}
        total_value = portfolio.total_value(current_prices)
        
        # Use Kelly Criterion if we have historical win rate
        risk_per_trade = self._calculate_risk_per_trade(signal, portfolio)
        
        # Adjust signal metadata with risk-adjusted size
        risk_adjusted_metadata = signal.metadata.copy()
        risk_adjusted_metadata['risk_per_trade'] = risk_per_trade
        
        return Signal(
            action=signal.action,
            strength=signal.strength,
            price=signal.price,
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            metadata=risk_adjusted_metadata
        )
    
    def _calculate_risk_per_trade(self, signal: Signal, portfolio: Portfolio) -> float:
        """Calculate risk per trade using Kelly Criterion or fixed percentage"""
        
        # Default to 2% risk per trade
        base_risk = 0.02
        
        # If we have stop loss information, use it
        if signal.metadata.get('stop_loss'):
            risk_distance = abs(signal.price - signal.metadata['stop_loss'])
            risk_pct = risk_distance / signal.price
            
            # Adjust based on Kelly criterion (simplified)
            win_rate = getattr(portfolio, 'historical_win_rate', 0.5)
            avg_win = getattr(portfolio, 'avg_win', 1.0)
            avg_loss = getattr(portfolio, 'avg_loss', 1.0)
            
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                # Cap Kelly fraction to prevent over-leverage
                kelly_fraction = min(max(kelly_fraction, 0), 0.25)
                return min(base_risk, kelly_fraction)
        
        return base_risk
    
    def record_trade_result(self, pnl: float):
        """Record the result of a completed trade"""
        self.daily_pnl += pnl
        self.daily_trades += 1