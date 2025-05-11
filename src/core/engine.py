"""
Core Trading Engine
Orchestrates the entire trading system
"""

import asyncio
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from src.utils.config import Config
from src.core.portfolio import Portfolio
from src.core.risk_manager import RiskManager
from src.data.feed import DataFeed, MarketData
from src.execution.broker import Broker, Order
from src.execution.order_manager import OrderManager
from src.execution.oanda_broker import OandaBroker
from src.core.strategy import BaseStrategy
from src.monitoring.logger import TradingLogger
from src.monitoring.metrics import MetricsCollector
from src.analysis.backtester import Backtester


class TradingEngine:
    """Main trading engine that orchestrates all components"""
    
    def __init__(self, config_path="config/settings.yaml"):
        self.config = Config(config_path)
        
        # Initialize components
        self.data_feed = None
        self.broker = None
        self.strategy = None
        self.portfolio = Portfolio(self.config)
        self.risk_manager = RiskManager(self.config)
        self.order_manager = None
        
        # Monitoring
        self.logger = TradingLogger(
            log_level=self.config.get_nested('monitoring', 'logging', 'level', 'INFO'),
            log_dir=self.config.get_nested('monitoring', 'logging', 'dir', 'logs')
        )
        self.metrics = MetricsCollector()
        
        # State
        self.is_running = False
        self.last_update = None
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Setup data feed
            await self.setup_data_feed()
            
            # Setup broker
            await self.setup_broker()
            
            # Setup order manager
            self.order_manager = OrderManager(self.broker)
            
            # Load strategies
            await self.setup_strategies()
            
            self.logger.logger.info("Trading engine initialized successfully")
            
        except Exception as e:
            self.logger.logger.error(f"Failed to initialize trading engine: {e}")
            raise
    
    async def setup_data_feed(self):
        """Setup data feed from configuration"""
        data_config = self.config.get_nested('data', 'provider', {})
        self.data_feed = DataFeed(data_config)
        await self.data_feed.connect()
        
    async def setup_broker(self):
        """Setup broker from configuration"""
        broker_config = self.config.get_nested('execution', 'broker', {})
        
        if broker_config.get('type') == 'oanda':
            self.broker = OandaBroker(broker_config)
            await self.broker.connect()
        else:
            raise ValueError(f"Unsupported broker type: {broker_config.get('type')}")
        
    async def setup_strategies(self):
        """Setup trading strategies from configuration"""
        # This would be expanded based on strategy configuration
        pass
    
    async def run(self, mode="live", **kwargs):
        """
        Run the trading engine
        
        Args:
            mode: Running mode ('live', 'backtest', 'paper')
            **kwargs: Additional parameters based on mode
        """
        self.is_running = True
        
        try:
            if mode == "backtest":
                return await self.run_backtest(**kwargs)
            elif mode == "live":
                return await self.run_live(**kwargs)
            elif mode == "paper":
                return await self.run_paper_trading(**kwargs)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            self.logger.logger.error(f"Error running trading engine: {e}")
            raise
        finally:
            self.is_running = False
            await self.shutdown()
    
    async def run_backtest(self, start_date: datetime, end_date: datetime, **kwargs):
        """Run backtest mode"""
        if not self.strategy:
            raise ValueError("No strategy configured for backtest")
        
        backtester = Backtester(
            initial_cash=self.portfolio.initial_cash
        )
        
        return await backtester.run(
            strategy=self.strategy,
            data_feed=self.data_feed,
            start_date=start_date,
            end_date=end_date,
            commission=kwargs.get('commission', 0.001)
        )
    
    async def run_live(self, symbols: List[str] = None, **kwargs):
        """Run live trading mode"""
        if not symbols:
            symbols = self.config.get_nested('trading', 'symbols', ['EUR_USD'])
        
        self.logger.logger.info(f"Starting live trading for symbols: {symbols}")
        
        # Start the main trading loop
        await self._main_trading_loop(symbols)
    
    async def run_paper_trading(self, symbols: List[str] = None, **kwargs):
        """Run paper trading mode (live data, simulated orders)"""
        # Similar to live trading but with mock broker
        # TODO: Implement paper trading logic
        pass
    
    async def _main_trading_loop(self, symbols: List[str]):
        """Main trading loop for live/paper trading"""
        try:
            # Stream live data
            async for market_data in self.data_feed.stream_live_data(symbols):
                # Update strategy
                if self.strategy:
                    signal = await self.strategy.update(market_data)
                    
                    if signal and signal.action != 'hold':
                        # Process signal through risk manager
                        approved_signal = await self.risk_manager.evaluate_signal(
                            signal, self.portfolio, market_data.__dict__
                        )
                        
                        if approved_signal:
                            # Execute trade
                            await self._execute_signal(approved_signal)
                
                # Update metrics
                self._update_metrics(market_data)
                
                # Check stop conditions
                if not self.is_running:
                    break
                    
                self.last_update = datetime.now()
                
        except Exception as e:
            self.logger.logger.error(f"Error in main trading loop: {e}")
            raise
    
    async def _execute_signal(self, signal):
        """Execute a trading signal"""
        try:
            # Log signal
            self.logger.log_signal({
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'action': signal.action,
                'strength': signal.strength,
                'price': signal.price,
                'metadata': signal.metadata
            })
            
            # Create order from signal
            order = self._create_order_from_signal(signal)
            
            # Submit order
            if order:
                order_id = await self.order_manager.submit_order(order)
                
                if order_id:
                    # Log successful order
                    self.logger.log_trade({
                        'timestamp': signal.timestamp,
                        'symbol': signal.symbol,
                        'action': signal.action,
                        'price': signal.price,
                        'order_id': order_id
                    })
                    
                    # Update metrics
                    self.metrics.increment_counter('orders.placed')
                
        except Exception as e:
            self.logger.logger.error(f"Error executing signal: {e}")
    
    def _create_order_from_signal(self, signal):
        """Create an order from a trading signal"""
        # Calculate position size
        position_size = self.portfolio.calculate_position_size(signal, signal.price)
        
        if position_size <= 0:
            return None
        
        # Create order
        order = Order(
            symbol=signal.symbol,
            side='BUY' if signal.action == 'buy' else 'SELL',
            quantity=position_size,
            order_type='MARKET',
            price=signal.price,
            stop_loss=signal.metadata.get('stop_loss'),
            take_profit=signal.metadata.get('take_profit')
        )
        
        return order
    
    def _update_metrics(self, market_data: MarketData):
        """Update system metrics"""
        # Record market data metrics
        self.metrics.record_metric('market.close_price', market_data.close, 
                                 tags={'symbol': market_data.symbol})
        
        # Calculate portfolio value
        current_prices = {market_data.symbol: market_data.close}
        portfolio_value = self.portfolio.total_value(current_prices)
        
        self.metrics.record_metric('portfolio.value', portfolio_value)
        self.metrics.record_metric('portfolio.cash', self.portfolio.cash)
        
        # Record system metrics
        system_metrics = self.metrics.get_system_metrics()
        for name, value in system_metrics.items():
            self.metrics.record_metric(name, value)
    
    async def shutdown(self):
        """Gracefully shutdown the trading engine"""
        self.logger.logger.info("Shutting down trading engine...")
        
        self.is_running = False
        
        try:
            # Close all positions if configured
            if self.config.get_nested('trading', 'close_on_shutdown', False):
                await self._close_all_positions()
            
            # Cancel pending orders
            if self.order_manager:
                await self._cancel_all_orders()
            
            # Disconnect from broker
            if self.broker:
                await self.broker.disconnect()
            
            # Disconnect from data feed
            if self.data_feed:
                await self.data_feed.disconnect()
            
            # Export final metrics
            self.metrics.export_metrics('logs/final_metrics.json')
            
            self.logger.logger.info("Trading engine shutdown complete")
            
        except Exception as e:
            self.logger.logger.error(f"Error during shutdown: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            if not self.broker:
                return
            
            positions = await self.broker.get_positions()
            
            for symbol, position_info in positions.items():
                long_units = position_info.get('long_units', 0)
                short_units = position_info.get('short_units', 0)
                
                if long_units > 0 or short_units > 0:
                    await self.broker.close_position(symbol)
                    self.logger.logger.info(f"Closed position for {symbol}")
                    
        except Exception as e:
            self.logger.logger.error(f"Error closing all positions: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            for order_id in list(self.order_manager.active_orders.keys()):
                await self.order_manager.cancel_order(order_id)
                
        except Exception as e:
            self.logger.logger.error(f"Error cancelling orders: {e}")
    
    async def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'portfolio_value': self.portfolio.total_value({}),
            'open_positions': len(self.portfolio.positions),
            'pending_orders': len(self.order_manager.active_orders) if self.order_manager else 0,
            'strategy': self.strategy.name if self.strategy else None
        }
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and orders immediately"""
        self.logger.logger.warning("EMERGENCY STOP TRIGGERED")
        
        # Cancel all orders
        if self.order_manager:
            await self._cancel_all_orders()
        
        # Close all positions
        await self._close_all_positions()
        
        # Stop engine
        self.is_running = False
        
        self.logger.logger.warning("Emergency stop completed")


class LiveTrader:
    """Live trading implementation"""
    
    def __init__(
        self,
        data_feed: DataFeed,
        broker: Broker,
        strategy: BaseStrategy,
        risk_manager: RiskManager
    ):
        self.data_feed = data_feed
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
    
    async def run(self):
        """Run live trading loop"""
        symbols = ['EUR_USD']  # TODO: Make configurable
        
        try:
            async for market_data in self.data_feed.stream_live_data(symbols):
                # Update strategy
                signal = await self.strategy.update(market_data)
                
                # Process signal
                if signal and signal.action != 'hold':
                    await self._process_signal(signal, market_data)
                    
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}")
            raise
    
    async def _process_signal(self, signal, market_data):
        """Process a trading signal in live mode"""
        # Risk management check
        # Order placement
        # Position management
        pass