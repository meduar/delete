"""
Trading Visualization Module
Create comprehensive visualizations for trading analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging


class TradingVisualizer:
    """Create visualizations for trading analysis"""
    
    def __init__(self, style: str = 'seaborn'):
        self.logger = logging.getLogger(__name__)
        self.style = style
        self._setup_style()
    
    def _setup_style(self):
        """Set up plotting style"""
        plt.style.use(self.style if self.style in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_equity_curve(
        self, 
        equity_curve: pd.DataFrame, 
        trades: Optional[List[Dict[str, Any]]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> go.Figure:
        """
        Create interactive equity curve plot with trade markers
        
        Args:
            equity_curve: DataFrame with timestamp index and equity values
            trades: Optional list of trade dictionaries
            figsize: Figure size
        
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown', 'Daily Returns'),
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add trade markers if provided
        if trades:
            buy_trades = [t for t in trades if t['action'] == 'buy']
            sell_trades = [t for t in trades if t['action'] == 'sell']
            
            if buy_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[t['timestamp'] for t in buy_trades],
                        y=[equity_curve.loc[t['timestamp'], 'equity'] for t in buy_trades if t['timestamp'] in equity_curve.index],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=10, color='green')
                    ),
                    row=1, col=1
                )
            
            if sell_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[t['timestamp'] for t in sell_trades],
                        y=[equity_curve.loc[t['timestamp'], 'equity'] for t in sell_trades if t['timestamp'] in equity_curve.index],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=10, color='red')
                    ),
                    row=1, col=1
                )
        
        # Drawdown chart
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve['drawdown'] * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Daily returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        fig.add_trace(
            go.Bar(
                x=equity_curve.index,
                y=equity_curve['returns'] * 100,
                name='Daily Returns',
                marker_color=np.where(equity_curve['returns'] > 0, 'green', 'red')
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Trading Performance Dashboard',
            showlegend=True,
            hovermode='x unified',
            xaxis3_title='Date',
            yaxis_title='Equity ($)',
            yaxis2_title='Drawdown (%)',
            yaxis3_title='Daily Return (%)'
        )
        
        return fig
    
    def plot_performance_metrics(
        self, 
        metrics: Dict[str, float],
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Create visualization of performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Performance Metrics Dashboard', fontsize=16)
        
        # Risk-Return Scatter
        ax1 = axes[0, 0]
        ax1.scatter(metrics.get('volatility', 0), metrics.get('annualized_return', 0), 
                   s=100, color='blue', alpha=0.6)
        ax1.set_xlabel('Volatility (%)')
        ax1.set_ylabel('Annualized Return (%)')
        ax1.set_title('Risk-Return Profile')
        ax1.grid(True, alpha=0.3)
        
        # Key Metrics Bar Chart
        ax2 = axes[0, 1]
        key_metrics = {
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Sortino Ratio': metrics.get('sortino_ratio', 0),
            'Calmar Ratio': metrics.get('calmar_ratio', 0),
            'Profit Factor': metrics.get('profit_factor', 0)
        }
        bars = ax2.bar(key_metrics.keys(), key_metrics.values())
        ax2.set_title('Key Performance Ratios')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Trade Statistics
        ax3 = axes[1, 0]
        trade_stats = {
            'Total Trades': metrics.get('total_trades', 0),
            'Win Rate': metrics.get('win_rate', 0),
            'Consecutive Wins': metrics.get('consecutive_wins', 0),
            'Consecutive Losses': metrics.get('consecutive_losses', 0)
        }
        
        y_pos = np.arange(len(trade_stats))
        bars = ax3.barh(y_pos, list(trade_stats.values()))
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(trade_stats.keys())
        ax3.set_title('Trade Statistics')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f'{width:.0f}%' if 'Rate' in list(trade_stats.keys())[i] else f'{width:.0f}'
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    label,
                    ha='left', va='center')
        
        # Risk Metrics
        ax4 = axes[1, 1]
        risk_metrics = {
            'Max Drawdown': -metrics.get('max_drawdown', 0),
            'Avg Drawdown': -metrics.get('average_drawdown', 0),
            'Volatility': metrics.get('volatility', 0),
            'Recovery Factor': metrics.get('recovery_factor', 0)
        }
        
        colors = ['red' if v < 0 else 'green' for v in risk_metrics.values()]
        bars = ax4.bar(risk_metrics.keys(), risk_metrics.values(), color=colors)
        ax4.set_title('Risk Metrics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label_y = height - 0.5 if height < 0 else height + 0.5
            ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{abs(height):.1f}%' if 'Drawdown' in bar.get_label() or 'Volatility' in bar.get_label() else f'{height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        return fig
    
    def plot_trade_distribution(
        self, 
        trades: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Analyze and visualize trade distribution
        
        Args:
            trades: List of trade dictionaries
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Trade Analysis Dashboard', fontsize=16)
        
        # Filter completed trades
        completed_trades = [t for t in trades if not t.get('entry', True) and 'pnl' in t]
        
        if not completed_trades:
            self.logger.warning("No completed trades to visualize")
            return fig
        
        # Extract data
        pnl_values = [t['pnl'] for t in completed_trades]
        durations = [t.get('hold_time', 0) / 3600 for t in completed_trades]  # Convert to hours
        timestamps = [t['timestamp'] for t in completed_trades]
        
        # 1. PnL Distribution
        ax1 = axes[0, 0]
        ax1.hist(pnl_values, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('PnL ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PnL Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss by Time
        ax2 = axes[0, 1]
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        ax2.scatter(timestamps, pnl_values, c=colors, alpha=0.6)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('PnL ($)')
        ax2.set_title('Wins/Losses Over Time')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Hold Time Distribution
        ax3 = axes[0, 2]
        ax3.hist(durations, bins=30, edgecolor='black', alpha=0.7, color='blue')
        ax3.set_xlabel('Hold Time (hours)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Trade Duration Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. PnL vs Hold Time
        ax4 = axes[1, 0]
        scatter = ax4.scatter(durations, pnl_values, c=colors, alpha=0.6)
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Hold Time (hours)')
        ax4.set_ylabel('PnL ($)')
        ax4.set_title('PnL vs Hold Time')
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative PnL
        ax5 = axes[1, 1]
        cumulative_pnl = np.cumsum(pnl_values)
        ax5.plot(timestamps, cumulative_pnl, linewidth=2, color='blue')
        ax5.fill_between(timestamps, cumulative_pnl, 0, 
                        where=(np.array(cumulative_pnl) >= 0), 
                        interpolate=True, alpha=0.3, color='green')
        ax5.fill_between(timestamps, cumulative_pnl, 0, 
                        where=(np.array(cumulative_pnl) < 0), 
                        interpolate=True, alpha=0.3, color='red')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Cumulative PnL ($)')
        ax5.set_title('Cumulative Performance')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # 6. Monthly Performance
        ax6 = axes[1, 2]
        # Convert to monthly data
        df = pd.DataFrame(completed_trades)
        df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
        monthly_pnl = df.groupby('month')['pnl'].sum()
        
        bars = ax6.bar(range(len(monthly_pnl)), monthly_pnl.values, 
                      color=['green' if v > 0 else 'red' for v in monthly_pnl.values])
        ax6.set_xticks(range(len(monthly_pnl)))
        ax6.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45)
        ax6.set_ylabel('Monthly PnL ($)')
        ax6.set_title('Monthly Performance')
        ax6.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label_y = height + 5 if height > 0 else height - 5
            ax6.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'${int(height)}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Compare multiple strategies side by side
        
        Args:
            results: Dictionary with strategy names as keys and metrics as values
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Strategy Comparison Dashboard', fontsize=16)
        
        strategies = list(results.keys())
        
        # Key metrics comparison
        ax1 = axes[0, 0]
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results[s].get(metric, 0) for s in strategies]
            offset = width * (i - 1)
            bars = ax1.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_ylabel('Value')
        ax1.set_title('Key Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Risk-adjusted returns
        ax2 = axes[0, 1]
        volatilities = [results[s].get('volatility', 0) for s in strategies]
        returns = [results[s].get('annualized_return', 0) for s in strategies]
        
        for i, strategy in enumerate(strategies):
            ax2.scatter(volatilities[i], returns[i], s=100, label=strategy)
        
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Annualized Return (%)')
        ax2.set_title('Risk-Return Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Trade statistics
        ax3 = axes[1, 0]
        trade_metrics = ['total_trades', 'win_rate', 'profit_factor']
        
        for i, metric in enumerate(trade_metrics):
            values = [results[s].get(metric, 0) for s in strategies]
            offset = width * (i - 1)
            bars = ax3.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        ax3.set_ylabel('Value')
        ax3.set_title('Trading Statistics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategies, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        summary_metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                          'Win Rate (%)', 'Profit Factor', 'Total Trades']
        
        table_data = []
        for strategy in strategies:
            row = [
                f"{results[strategy].get('total_return', 0):.1f}",
                f"{results[strategy].get('sharpe_ratio', 0):.2f}",
                f"{results[strategy].get('max_drawdown', 0):.1f}",
                f"{results[strategy].get('win_rate', 0):.1f}",
                f"{results[strategy].get('profit_factor', 0):.2f}",
                f"{results[strategy].get('total_trades', 0):.0f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         rowLabels=strategies,
                         colLabels=summary_metrics,
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax4.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown_analysis(
        self,
        equity_curve: pd.DataFrame,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Detailed drawdown analysis
        
        Args:
            equity_curve: DataFrame with equity and drawdown data
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('Drawdown Analysis', fontsize=16)
        
        # Equity curve
        axes[0].plot(equity_curve.index, equity_curve['equity'], linewidth=2)
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown curve
        axes[1].fill_between(equity_curve.index, 0, equity_curve['drawdown'] * 100, 
                           color='red', alpha=0.3)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Underwater plot
        axes[2].fill_between(equity_curve.index, 0, -equity_curve['drawdown'] * 100,
                           color='blue', alpha=0.3)
        axes[2].set_ylabel('Underwater Period')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_performance_report(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        metrics: Dict[str, float],
        strategy_name: str,
        output_dir: str = "reports"
    ) -> str:
        """
        Create a comprehensive PDF performance report
        
        Args:
            equity_curve: Equity curve data
            trades: List of trades
            metrics: Performance metrics
            strategy_name: Name of the strategy
            output_dir: Directory to save the report
        
        Returns:
            Path to the generated report
        """
        import os
        from matplotlib.backends.backend_pdf import PdfPages
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f"performance_report_{strategy_name}_{timestamp}.pdf")
        
        with PdfPages(filename) as pdf:
            # Page 1: Equity curve and drawdown
            fig1 = plt.figure(figsize=(11, 8.5))
            # Create equity curve plot (simplified for PDF)
            # ... implementation ...
            pdf.savefig(fig1)
            plt.close(fig1)
            
            # Page 2: Performance metrics
            fig2 = self.plot_performance_metrics(metrics)
            pdf.savefig(fig2)
            plt.close(fig2)
            
            # Page 3: Trade analysis
            fig3 = self.plot_trade_distribution(trades)
            pdf.savefig(fig3)
            plt.close(fig3)
            
            # Page 4: Drawdown analysis
            fig4 = self.plot_drawdown_analysis(equity_curve)
            pdf.savefig(fig4)
            plt.close(fig4)
        
        return filename