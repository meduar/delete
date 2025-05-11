#!/usr/bin/env python3
"""
Backtest Runner Script
Run backtests for different strategies with comprehensive analysis
"""

import asyncio
import argparse
import sys
from datetime import datetime
from pathlib import Path
import json

from src.utils.config import Config
from src.analysis.backtester import Backtester
from src.analysis.visualizer import TradingVisualizer
from src.data.feed import DataFeed
from src.strategies import SmaCrossStrategy, BollingerStrategy, MomentumStrategy

# Strategy mapping
STRATEGY_CLASSES = {
    'sma_cross': SmaCrossStrategy,
    'bollinger': BollingerStrategy,
    'momentum': MomentumStrategy
}

async def run_backtest(args):
    """Run a comprehensive backtest"""
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize backtester
    backtester = Backtester(initial_cash=args.cash)
    
    # Get strategy class
    if args.strategy not in STRATEGY_CLASSES:
        print(f"Error: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {', '.join(STRATEGY_CLASSES.keys())}")
        sys.exit(1)
    
    strategy_class = STRATEGY_CLASSES[args.strategy]
    
    # Get strategy parameters from config
    strategy_config = config.get_nested('strategies', args.strategy, {})
    strategy_params = strategy_config.get('parameters', {})
    
    # Override with command line arguments if provided
    if args.strategy_params:
        try:
            cli_params = json.loads(args.strategy_params)
            strategy_params.update(cli_params)
        except json.JSONDecodeError as e:
            print(f"Error parsing strategy parameters: {e}")
            sys.exit(1)
    
    # Print run information
    print("\n" + "="*50)
    print(f"BACKTEST CONFIGURATION")
    print("="*50)
    print(f"Strategy: {args.strategy}")
    print(f"Symbol: {args.symbol}")
    print(f"Start Date: {args.start}")
    print(f"End Date: {args.end}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"Commission: {args.commission*100:.3f}%")
    print(f"Parameters: {json.dumps(strategy_params, indent=2)}")
    print("="*50 + "\n")
    
    # Initialize data feed
    data_config = config.get_nested('data', 'provider', {})
    data_feed = DataFeed(data_config)
    
    try:
        # Connect to data provider
        print("Connecting to data provider...")
        await data_feed.connect()
        
        # Create strategy instance
        strategy = strategy_class(args.strategy, strategy_params)
        
        # Run backtest
        print("Running backtest...")
        start_time = datetime.now()
        
        result = await backtester.run(
            strategy=strategy,
            data_feed=data_feed,
            start_date=datetime.strptime(args.start, '%Y-%m-%d'),
            end_date=datetime.strptime(args.end, '%Y-%m-%d'),
            commission=args.commission
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Execution Time: {duration:.2f} seconds")
        print(f"\nPerformance Metrics:")
        
        # Display key metrics
        metrics = result.performance_metrics
        key_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Final Equity', f"${metrics.get('final_equity', 0):.2f}")
        ]
        
        for name, value in key_metrics:
            print(f"  {name:<15} {value}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualizer = TradingVisualizer()
        
        # Equity curve with trade markers
        equity_fig = visualizer.plot_equity_curve(
            result.equity_curve, 
            result.trades
        )
        equity_file = output_dir / f"equity_curve_{args.strategy}_{args.symbol}.html"
        equity_fig.write_html(str(equity_file))
        print(f"  Equity curve saved to: {equity_file}")
        
        # Performance metrics dashboard
        metrics_fig = visualizer.plot_performance_metrics(result.performance_metrics)
        metrics_file = output_dir / f"performance_metrics_{args.strategy}_{args.symbol}.png"
        metrics_fig.savefig(str(metrics_file), dpi=300, bbox_inches='tight')
        print(f"  Performance metrics saved to: {metrics_file}")
        
        # Trade distribution analysis
        dist_fig = visualizer.plot_trade_distribution(result.trades)
        dist_file = output_dir / f"trade_distribution_{args.strategy}_{args.symbol}.png"
        dist_fig.savefig(str(dist_file), dpi=300, bbox_inches='tight')
        print(f"  Trade distribution saved to: {dist_file}")
        
        # Save detailed results to JSON
        results_file = output_dir / f"backtest_results_{args.strategy}_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        detailed_results = {
            'backtest_config': {
                'strategy': args.strategy,
                'symbol': args.symbol,
                'start_date': args.start,
                'end_date': args.end,
                'initial_cash': args.cash,
                'commission': args.commission,
                'parameters': strategy_params
            },
            'performance_metrics': metrics,
            'trades': [
                {
                    'date': trade['date'].isoformat() if hasattr(trade['date'], 'isoformat') else str(trade['date']),
                    'symbol': trade['symbol'],
                    'action': trade['action'],
                    'price': trade['price'],
                    'shares': trade['shares'],
                    'pnl': trade.get('pnl', 0),
                    'cost': trade.get('cost', 0),
                    'revenue': trade.get('revenue', 0)
                }
                for trade in result.trades
            ],
            'drawdown_analysis': result.drawdown_analysis,
            'execution_time': duration
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"  Detailed results saved to: {results_file}")
        
        # Generate summary report
        report_file = output_dir / f"backtest_report_{args.strategy}_{args.symbol}.md"
        generate_markdown_report(detailed_results, report_file)
        print(f"  Summary report saved to: {report_file}")
        
        print("="*50 + "\n")
        
        # Return for programmatic use
        return result
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up
        await data_feed.disconnect()

def generate_markdown_report(results: dict, output_file: Path):
    """Generate a markdown summary report"""
    
    config = results['backtest_config']
    metrics = results['performance_metrics']
    trades = results['trades']
    
    # Calculate additional metrics
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    
    report = f"""# Backtest Report

## Configuration
- **Strategy**: {config['strategy']}
- **Symbol**: {config['symbol']}
- **Period**: {config['start_date']} to {config['end_date']}
- **Initial Capital**: ${config['initial_cash']:,.2f}
- **Commission**: {config['commission']*100:.3f}%

## Parameters
```json
{json.dumps(config['parameters'], indent=2)}
```

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0):.2f}% |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |
| Max Drawdown | {metrics.get('max_drawdown', 0):.2f}% |
| Win Rate | {metrics.get('win_rate', 0):.2f}% |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} |
| Final Equity | ${metrics.get('final_equity', 0):,.2f} |

## Trade Statistics

- **Total Trades**: {len(trades)}
- **Winning Trades**: {len(winning_trades)}
- **Losing Trades**: {len(losing_trades)}
- **Average Win**: ${metrics.get('avg_winning_trade', 0):.2f}
- **Average Loss**: ${metrics.get('avg_losing_trade', 0):.2f}
- **Best Trade**: ${metrics.get('max_winning_trade', 0):.2f}
- **Worst Trade**: ${metrics.get('max_losing_trade', 0):.2f}

## Risk Analysis

- **Maximum Drawdown**: {metrics.get('max_drawdown', 0):.2f}%
- **Recovery Factor**: {metrics.get('recovery_factor', 0):.2f}
- **Calmar Ratio**: {metrics.get('calmar_ratio', 0):.2f}
- **Volatility**: {metrics.get('volatility', 0):.2f}%

## Execution Details

- **Execution Time**: {results['execution_time']:.2f} seconds
- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*Report generated by Trading Framework v1.0*
"""
    
    with open(output_file, 'w') as f:
        f.write(report)

def main():
    """Main entry point for the backtest runner"""
    
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    
    # Required arguments
    parser.add_argument('--strategy', required=True, 
                       choices=list(STRATEGY_CLASSES.keys()),
                       help='Strategy to run')
    parser.add_argument('--symbol', required=True, 
                       help='Trading symbol (e.g., EUR_USD)')
    parser.add_argument('--start', required=True, 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, 
                       help='End date (YYYY-MM-DD)')
    
    # Optional arguments
    parser.add_argument('--config', default='config/settings.yaml',
                       help='Configuration file path')
    parser.add_argument('--cash', type=float, default=100000,
                       help='Initial cash amount')
    parser.add_argument('--commission', type=float, default=0.001,
                       help='Commission rate (e.g., 0.001 for 0.1%%)')
    parser.add_argument('--output-dir', default='backtest_results',
                       help='Output directory for results')
    parser.add_argument('--strategy-params', type=str,
                       help='Strategy parameters as JSON string')
    
    # Advanced options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Run the backtest
    try:
        asyncio.run(run_backtest(args))
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()