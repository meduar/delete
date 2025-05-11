"""
Backtesting engine for trading strategies
Supports comprehensive backtesting with performance analysis
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.core.strategy import BaseStrategy, Signal
from src.core.portfolio import Portfolio
from src.data.feed import DataFeed
from src.analysis.performance import PerformanceAnalyzer


@dataclass
class BacktestResult:
    """Container for backtest results"""
    performance_metrics: Dict[str, float]
    equity_curve: pd.DataFrame
    trades: List[Dict[str, Any]]
    drawdown_analysis: Dict[str, float]
    strategy_metrics: Dict[str, Any]
    execution_time: float


class Backtester:
    """Advanced backtesting engine"""
    
    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.logger = logging.getLogger(__name__)
        
        # Reset attributes for each run
        self.reset()
    
    def reset(self):
        """Reset backtester for a new run"""
        self.portfolio = Portfolio()
        self.portfolio.cash = self.initial_cash
        self.portfolio.initial_cash = self.initial_cash
        self.equity_curve = []
        self.trades = []
        self.daily_returns = []
        self.positions_history = []
        
        # Keep track of open positions for proper exit handling
        self.open_positions = {}  # symbol -> position_info
        
    async def run(
        self,
        strategy: BaseStrategy,
        data_feed: DataFeed,
        start_date: datetime,
        end_date: datetime,
        commission: float = 0.001,
        slippage: float = 0.0001
    ) -> BacktestResult:
        """
        Run backtest for a given strategy
        
        Args:
            strategy: Trading strategy to test
            data_feed: Data feed for market data
            start_date: Backtest start date
            end_date: Backtest end date
            commission: Commission rate (e.g., 0.001 for 0.1%)
            slippage: Slippage rate (e.g., 0.0001 for 0.01%)
        
        Returns:
            BacktestResult object containing all results
        """
        start_time = datetime.now()
        self.commission = commission
        self.slippage = slippage
        
        try:
            # Initialize strategy
            await strategy.initialize(data_feed)
            
            # Get historical data
            data = await data_feed.get_historical_data(
                symbol=strategy.parameters.get('symbol', 'EUR_USD'),
                start_date=start_date,
                end_date=end_date,
                granularity="M1"
            )
            
            if data.empty:
                raise ValueError("No data received for backtesting")
            
            # Main backtesting loop
            prev_date = None
            for idx, row in data.iterrows():
                market_data = self._create_market_data(row, idx)
                
                # Update strategy and get signal
                signal = await strategy.update(market_data)
                
                # Process signal
                if signal and signal.action != 'hold':
                    await self._process_signal(signal, market_data, strategy)
                
                # Update portfolio value and equity curve
                current_prices = {signal.symbol if signal else strategy.parameters.get('symbol', 'EUR_USD'): market_data.close}
                self._update_equity_curve(idx, current_prices)
                
                # Track daily returns for analysis
                current_date = idx.date() if hasattr(idx, 'date') else datetime.now().date()
                if prev_date != current_date:
                    if prev_date is not None:
                        daily_return = self._calculate_daily_return()
                        self.daily_returns.append(daily_return)
                    prev_date = current_date
            
            # Close any remaining open positions at the end
            await self._close_all_positions(market_data, strategy)
            
            # Calculate final metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return await self._compile_results(strategy, execution_time)
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            raise
    
    def _create_market_data(self, row: pd.Series, timestamp) -> 'MarketData':
        """Convert pandas row to MarketData object"""
        from src.core.engine import MarketData
        
        return MarketData(
            timestamp=timestamp,
            symbol=getattr(row, 'symbol', 'EUR_USD'),
            open=row.get('open', row['close']),
            high=row.get('high', row['close']),
            low=row.get('low', row['close']),
            close=row['close'],
            volume=row.get('volume', 0)
        )
    
    async def _process_signal(self, signal: Signal, market_data, strategy: BaseStrategy):
        """Process a trading signal with proper entry/exit handling"""
        try:
            symbol = signal.symbol
            action = signal.action
            price = market_data.close
            
            # Calculate position size and cost
            position_size = self.portfolio.calculate_position_size(signal, price)
            cost = position_size * price * (1 + self.commission + self.slippage)
            
            # Handle different signal actions
            if action == 'buy':
                # Check if this is closing a short position or opening new long
                if symbol in self.open_positions and self.open_positions[symbol]['direction'] == 'short':
                    # Closing short position
                    await self._close_position(symbol, price, 'buy_to_cover', signal, market_data, strategy)
                else:
                    # Opening long position (or adding to existing long)
                    if cost <= self.portfolio.cash:
                        await self._open_position(symbol, price, position_size, 'long', signal, market_data, strategy)
            
            elif action == 'sell':
                # Check if this is closing a long position or opening new short
                if symbol in self.open_positions and self.open_positions[symbol]['direction'] == 'long':
                    # Closing long position
                    await self._close_position(symbol, price, 'sell_to_close', signal, market_data, strategy)
                else:
                    # Opening short position
                    await self._open_position(symbol, price, position_size, 'short', signal, market_data, strategy)
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
    
    async def _open_position(self, symbol: str, price: float, quantity: int, direction: str, 
                           signal: Signal, market_data, strategy: BaseStrategy):
        """Open a new position"""
        entry_price = price * (1 + self.slippage)
        
        # Calculate cost
        cost = quantity * entry_price
        commission = cost * self.commission
        total_cost = cost + commission
        
        # Check if we have enough cash
        if total_cost <= self.portfolio.cash:
            # Update portfolio
            if direction == 'long':
                position_id = self.portfolio.open_position(
                    symbol=symbol,
                    entry_price=entry_price,
                    quantity=quantity,
                    stop_loss=signal.metadata.get('stop_loss'),
                    take_profit=signal.metadata.get('take_profit')
                )
            else:  # short
                position_id = self.portfolio.open_position(
                    symbol=symbol,
                    entry_price=entry_price,
                    quantity=-quantity,  # Negative for short
                    stop_loss=signal.metadata.get('stop_loss'),
                    take_profit=signal.metadata.get('take_profit')
                )
            
            # Track position
            self.open_positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': signal.timestamp,
                'position_id': position_id
            }
            
            # Record trade entry
            self.trades.append({
                'timestamp': signal.timestamp,
                'symbol': symbol,
                'action': 'buy' if direction == 'long' else 'sell',
                'price': entry_price,
                'shares': quantity,
                'commission': commission,
                'position_id': position_id,
                'entry': True,
                'strategy': signal.metadata.get('strategy_name', strategy.name),
                'signal_strength': signal.strength,
                'reason': signal.metadata.get('reason', '')
            })
    
    async def _close_position(self, symbol: str, price: float, close_type: str, 
                            signal: Signal, market_data, strategy: BaseStrategy):
        """Close an existing position"""
        if symbol not in self.open_positions:
            return
        
        position_info = self.open_positions[symbol]
        exit_price = price * (1 - self.slippage)
        
        # Get position from portfolio
        portfolio_position = self.portfolio.positions.get(symbol)
        if not portfolio_position:
            return
        
        # Calculate P&L
        entry_price = position_info['entry_price']
        quantity = position_info['quantity']
        
        if position_info['direction'] == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity
        
        # Close position in portfolio
        try:
            trade_result = self.portfolio.close_position(symbol, exit_price)
            commission = abs(portfolio_position.quantity) * exit_price * self.commission
            net_pnl = pnl - commission
            
            # Record trade exit
            self.trades.append({
                'timestamp': signal.timestamp,
                'symbol': symbol,
                'action': 'sell' if position_info['direction'] == 'long' else 'buy',
                'price': exit_price,
                'shares': abs(quantity),
                'commission': commission,
                'position_id': position_info['position_id'],
                'pnl': net_pnl,
                'pnl_percent': (net_pnl / (entry_price * abs(quantity))) * 100,
                'entry': False,
                'strategy': signal.metadata.get('strategy_name', strategy.name),
                'signal_strength': signal.strength,
                'hold_time': (signal.timestamp - position_info['entry_time']).total_seconds(),
                'reason': signal.metadata.get('reason', ''),
                'entry_price': entry_price
            })
            
            # Remove from tracking
            del self.open_positions[symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    async def _close_all_positions(self, market_data, strategy: BaseStrategy):
        """Close all open positions at the end of backtest"""
        for symbol in list(self.open_positions.keys()):
            # Create a fake exit signal
            fake_signal = Signal(
                action='sell' if self.open_positions[symbol]['direction'] == 'long' else 'buy',
                strength=1.0,
                price=market_data.close,
                symbol=symbol,
                timestamp=market_data.timestamp,
                metadata={'reason': 'End of backtest'}
            )
            
            await self._close_position(symbol, market_data.close, 'end_of_backtest', fake_signal, market_data, strategy)
    
    def _update_equity_curve(self, timestamp, current_prices: Dict[str, float]):
        """Update equity curve with current portfolio value"""
        total_value = self.portfolio.total_value(current_prices)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_value,
            'cash': self.portfolio.cash,
            'positions_value': total_value - self.portfolio.cash,
            'drawdown': self._calculate_drawdown(total_value)
        })
    
    def _calculate_daily_return(self) -> float:
        """Calculate daily return"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        today_equity = self.equity_curve[-1]['equity']
        yesterday_equity = self.equity_curve[-2]['equity']
        
        return (today_equity - yesterday_equity) / yesterday_equity
    
    def _calculate_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown"""
        peak_equity = max([point['equity'] for point in self.equity_curve] + [current_equity])
        
        if peak_equity == 0:
            return 0.0
        
        return (peak_equity - current_equity) / peak_equity
    
    async def _compile_results(self, strategy: BaseStrategy, execution_time: float) -> BacktestResult:
        """Compile all results into BacktestResult object"""
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        performance_analyzer = PerformanceAnalyzer()
        metrics = performance_analyzer.calculate_metrics(
            equity_curve=equity_df,
            trades=self.trades,
            initial_capital=self.initial_cash
        )
        
        # Calculate drawdown analysis
        drawdown_analysis = self._analyze_drawdowns(equity_df)
        
        # Get strategy-specific metrics
        strategy_metrics = strategy.get_performance_summary()
        
        return BacktestResult(
            performance_metrics=metrics,
            equity_curve=equity_df,
            trades=self.trades,
            drawdown_analysis=drawdown_analysis,
            strategy_metrics=strategy_metrics,
            execution_time=execution_time
        )
    
    def _analyze_drawdowns(self, equity_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze drawdown statistics"""
        drawdowns = equity_df['drawdown']
        
        return {
            'max_drawdown': drawdowns.max(),
            'avg_drawdown': drawdowns.mean(),
            'drawdown_duration': self._calculate_avg_drawdown_duration(equity_df),
            'recovery_factor': self._calculate_recovery_factor(equity_df),
            'underwater_periods': len([d for d in drawdowns if d > 0.05])  # Periods with >5% drawdown
        }
    
    def _calculate_avg_drawdown_duration(self, equity_df: pd.DataFrame) -> float:
        """Calculate average drawdown duration in days"""
        # Simplified implementation
        drawdowns = equity_df['drawdown']
        in_drawdown = drawdowns > 0
        
        drawdown_periods = []
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                duration = (equity_df.index[i] - equity_df.index[start_idx]).days
                drawdown_periods.append(duration)
                start_idx = None
        
        return np.mean(drawdown_periods) if drawdown_periods else 0.0
    
    def _calculate_recovery_factor(self, equity_df: pd.DataFrame) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        total_return = (equity_df['equity'].iloc[-1] - equity_df['equity'].iloc[0]) / equity_df['equity'].iloc[0]
        max_drawdown = equity_df['drawdown'].max()
        
        if max_drawdown == 0:
            return float('inf')
        
        return total_return / max_drawdown


class OptimizationBacktester(Backtester):
    """Backtester with parameter optimization capabilities"""
    
    def __init__(self, initial_cash: float = 100000):
        super().__init__(initial_cash)
        self.optimization_results = []
    
    async def optimize_strategy(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        data_feed: DataFeed,
        start_date: datetime,
        end_date: datetime,
        optimization_metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_class: Strategy class to optimize
            param_grid: Dictionary of parameter names to list of values to test
            data_feed: Data feed for market data
            start_date: Backtest start date
            end_date: Backtest end date
            optimization_metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        
        Returns:
            Dictionary containing best parameters and results
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        best_metric = float('-inf')
        best_params = None
        best_result = None
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            strategy = strategy_class(f"optimize_{i}", params)
            
            try:
                result = await self.run(strategy, data_feed, start_date, end_date)
                metric_value = result.performance_metrics.get(optimization_metric, 0)
                
                self.optimization_results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'result': result
                })
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result
                    
                self.logger.info(f"Tested params {params}: {optimization_metric}={metric_value:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error testing params {params}: {str(e)}")
        
        return {
            'best_parameters': best_params,
            'best_metric_value': best_metric,
            'best_result': best_result,
            'all_results': self.optimization_results
        }