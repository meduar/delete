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
                current_prices = {signal.symbol if signal else 'EUR_USD': market_data.close}
                self._update_equity_curve(idx, current_prices)
                
                # Track daily returns for analysis
                current_date = idx.date() if hasattr(idx, 'date') else datetime.now().date()
                if prev_date != current_date:
                    if prev_date is not None:
                        daily_return = self._calculate_daily_return()
                        self.daily_returns.append(daily_return)
                    prev_date = current_date
            
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
        """Process a trading signal"""
        try:
            # Calculate position size
            if signal.action == 'buy':
                position_size = self.portfolio.calculate_position_size(signal, market_data.close)
                
                # Check if we have enough cash
                cost = position_size * market_data.close * (1 + self.commission + self.slippage)
                if cost <= self.portfolio.cash:
                    # Open position
                    stop_loss = signal.metadata.get('stop_loss')
                    take_profit = signal.metadata.get('take_profit')
                    
                    position_id = self.portfolio.open_position(
                        symbol=signal.symbol,
                        entry_price=market_data.close * (1 + self.slippage),
                        quantity=position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    # Record trade entry
                    self.trades.append({
                        'timestamp': signal.timestamp,
                        'symbol': signal.symbol,
                        'action': 'buy',
                        'price': market_data.close * (1 + self.slippage),
                        'shares': position_size,
                        'commission': position_size * market_data.close * self.commission,
                        'position_id': position_id,
                        'entry': True,
                        'strategy': strategy.name,
                        'signal_strength': signal.strength
                    })
                
            elif signal.action == 'sell':
                # Close existing position
                if signal.symbol in self.portfolio.positions:
                    exit_price = market_data.close * (1 - self.slippage)
                    trade_result = self.portfolio.close_position(signal.symbol, exit_price)
                    
                    # Record trade exit
                    position = self.portfolio.positions.get(signal.symbol)
                    if position:
                        commission = abs(position.quantity) * exit_price * self.commission
                        pnl = trade_result['pnl'] - commission
                        
                        self.trades.append({
                            'timestamp': signal.timestamp,
                            'symbol': signal.symbol,
                            'action': 'sell',
                            'price': exit_price,
                            'shares': abs(position.quantity),
                            'commission': commission,
                            'position_id': position.position_id,
                            'pnl': pnl,
                            'pnl_percent': (pnl / position.cost_basis) * 100,
                            'entry': False,
                            'strategy': strategy.name,
                            'hold_time': (signal.timestamp - position.entry_time).total_seconds()
                        })
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
    
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