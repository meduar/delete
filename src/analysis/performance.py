"""
Performance Analysis Module
Calculate comprehensive trading performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging


class PerformanceAnalyzer:
    """Calculate and analyze trading performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        risk_free_rate: float = 0.02  # Annual risk-free rate
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: DataFrame with timestamp index and equity values
            trades: List of trade dictionaries
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        
        Returns:
            Dictionary containing all performance metrics
        """
        try:
            # Basic return metrics
            total_return = self._calculate_total_return(equity_curve, initial_capital)
            annualized_return = self._calculate_annualized_return(equity_curve, initial_capital)
            
            # Risk metrics
            volatility = self._calculate_volatility(equity_curve)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            
            # Risk-adjusted metrics
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve, initial_capital, risk_free_rate)
            sortino_ratio = self._calculate_sortino_ratio(equity_curve, initial_capital, risk_free_rate)
            calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
            
            # Trade statistics
            trade_stats = self._calculate_trade_statistics(trades)
            
            # Additional metrics
            profit_factor = self._calculate_profit_factor(trades)
            recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
            expectancy = self._calculate_expectancy(trades)
            
            # Combine all metrics
            metrics = {
                # Return metrics
                'total_return': total_return * 100,  # As percentage
                'annualized_return': annualized_return * 100,
                'final_equity': equity_curve['equity'].iloc[-1],
                
                # Risk metrics
                'volatility': volatility * 100,
                'max_drawdown': max_drawdown * 100,
                'average_drawdown': equity_curve['drawdown'].mean() * 100,
                
                # Risk-adjusted ratios
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                
                # Trade metrics
                'total_trades': trade_stats['total_trades'],
                'winning_trades': trade_stats['winning_trades'],
                'losing_trades': trade_stats['losing_trades'],
                'win_rate': trade_stats['win_rate'],
                'avg_winning_trade': trade_stats['avg_winning_trade'],
                'avg_losing_trade': trade_stats['avg_losing_trade'],
                'max_winning_trade': trade_stats['max_winning_trade'],
                'max_losing_trade': trade_stats['max_losing_trade'],
                'avg_trade_duration': trade_stats['avg_trade_duration'],
                
                # Additional metrics
                'profit_factor': profit_factor,
                'recovery_factor': recovery_factor,
                'expectancy': expectancy,
                'consecutive_wins': trade_stats['consecutive_wins'],
                'consecutive_losses': trade_stats['consecutive_losses']
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _calculate_total_return(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate total return"""
        if len(equity_curve) == 0:
            return 0.0
        
        final_equity = equity_curve['equity'].iloc[-1]
        return (final_equity - initial_capital) / initial_capital
    
    def _calculate_annualized_return(self, equity_curve: pd.DataFrame, initial_capital: float) -> float:
        """Calculate annualized return"""
        if len(equity_curve) < 2:
            return 0.0
        
        total_return = self._calculate_total_return(equity_curve, initial_capital)
        
        # Calculate number of years
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        if years == 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        # Annualize volatility
        daily_vol = equity_curve['returns'].std()
        return daily_vol * np.sqrt(252)  # Assuming 252 trading days per year
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) == 0:
            return 0.0
        
        equity = equity_curve['equity']
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        
        return drawdown.max()
    
    def _calculate_sharpe_ratio(
        self, 
        equity_curve: pd.DataFrame, 
        initial_capital: float, 
        risk_free_rate: float
    ) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate excess returns
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        mean_excess_return = equity_curve['returns'].mean() - risk_free_rate / 252
        std_returns = equity_curve['returns'].std()
        
        if std_returns == 0:
            return 0.0
        
        return np.sqrt(252) * mean_excess_return / std_returns
    
    def _calculate_sortino_ratio(
        self, 
        equity_curve: pd.DataFrame, 
        initial_capital: float, 
        risk_free_rate: float
    ) -> float:
        """Calculate Sortino ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate excess returns
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        mean_excess_return = equity_curve['returns'].mean() - risk_free_rate / 252
        
        # Calculate downside deviation
        negative_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else equity_curve['returns'].std()
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * mean_excess_return / downside_std
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-level statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'max_winning_trade': 0.0,
                'max_losing_trade': 0.0,
                'avg_trade_duration': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }
        
        # Filter only exit trades (complete trades)
        exit_trades = [t for t in trades if not t.get('entry', True)]
        
        if not exit_trades:
            return self._calculate_trade_statistics([])  # Return empty stats
        
        # Separate winning and losing trades
        winning_trades = [t for t in exit_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in exit_trades if t.get('pnl', 0) <= 0]
        
        # Calculate statistics
        total_trades = len(exit_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average trade metrics
        avg_winning_trade = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_losing_trade = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Extreme trade metrics
        max_winning_trade = max([t['pnl'] for t in winning_trades]) if winning_trades else 0
        max_losing_trade = min([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Average trade duration
        durations = [t.get('hold_time', 0) for t in exit_trades if 'hold_time' in t]
        avg_duration = np.mean(durations) if durations else 0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(exit_trades)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'max_winning_trade': max_winning_trade,
            'max_losing_trade': max_losing_trade,
            'avg_trade_duration': avg_duration / 3600,  # Convert to hours
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
    
    def _calculate_consecutive_trades(self, trades: List[Dict[str, Any]]) -> tuple:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return max_consecutive_wins, max_consecutive_losses
    
    def _calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate profit factor"""
        exit_trades = [t for t in trades if not t.get('entry', True)]
        
        if not exit_trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in exit_trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in exit_trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor"""
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_drawdown
    
    def _calculate_expectancy(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate expectancy per trade"""
        exit_trades = [t for t in trades if not t.get('entry', True)]
        
        if not exit_trades:
            return 0.0
        
        total_pnl = sum(t.get('pnl', 0) for t in exit_trades)
        return total_pnl / len(exit_trades)


class AdvancedPerformanceAnalyzer(PerformanceAnalyzer):
    """Extended performance analyzer with additional metrics"""
    
    def calculate_advanced_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        
        base_metrics = self.calculate_metrics(equity_curve, trades, equity_curve['equity'].iloc[0])
        
        # Add advanced metrics
        advanced_metrics = {
            'information_ratio': self._calculate_information_ratio(equity_curve, market_data),
            'treynor_ratio': self._calculate_treynor_ratio(equity_curve, market_data),
            'jensen_alpha': self._calculate_jensen_alpha(equity_curve, market_data),
            'tracking_error': self._calculate_tracking_error(equity_curve, market_data),
            'downside_capture': self._calculate_downside_capture(equity_curve, market_data),
            'upside_capture': self._calculate_upside_capture(equity_curve, market_data),
            'sterling_ratio': self._calculate_sterling_ratio(equity_curve),
            'mar_ratio': self._calculate_mar_ratio(equity_curve)
        }
        
        return {**base_metrics, **advanced_metrics}
    
    def _calculate_information_ratio(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate Information Ratio relative to benchmark"""
        # Simplified implementation
        return 0.0  # TODO: Implement with proper benchmark data
    
    def _calculate_treynor_ratio(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate Treynor Ratio"""
        # Simplified implementation
        return 0.0  # TODO: Implement with beta calculation
    
    def _calculate_jensen_alpha(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate Jensen's Alpha"""
        # Simplified implementation
        return 0.0  # TODO: Implement with CAPM
    
    def _calculate_tracking_error(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate tracking error"""
        # Simplified implementation
        return 0.0  # TODO: Implement with benchmark comparison
    
    def _calculate_downside_capture(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate downside capture ratio"""
        # Simplified implementation
        return 0.0  # TODO: Implement with benchmark down periods
    
    def _calculate_upside_capture(
        self, 
        equity_curve: pd.DataFrame, 
        market_data: pd.DataFrame
    ) -> float:
        """Calculate upside capture ratio"""
        # Simplified implementation
        return 0.0  # TODO: Implement with benchmark up periods
    
    def _calculate_sterling_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Sterling Ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        annualized_return = self._calculate_annualized_return(
            equity_curve, 
            equity_curve['equity'].iloc[0]
        )
        
        # Calculate average drawdown of worst 10%
        drawdowns = equity_curve['drawdown'].sort_values(ascending=False)
        worst_10_percent = int(len(drawdowns) * 0.1)
        avg_worst_drawdown = drawdowns.head(worst_10_percent).mean()
        
        if avg_worst_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / avg_worst_drawdown
    
    def _calculate_mar_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate MAR (Managed Account Reports) Ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        annualized_return = self._calculate_annualized_return(
            equity_curve, 
            equity_curve['equity'].iloc[0]
        )
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown