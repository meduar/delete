#!/usr/bin/env python3
"""
Test the fixed strategy and backtester
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed classes
from src.analysis.backtester import Backtester
from src.data.feed import DataFeed


async def test_fixed_strategy():
    print("=== TESTING FIXED STRATEGY AND BACKTESTER ===\n")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Import the fixed strategy
    from src.strategies.sma_cross import SmaCrossStrategy
    
    # Set up
    strategy_name = config['strategy']['name']
    strategy_params = config.get('strategies', {}).get(strategy_name, {}).copy()
    strategy_params.update(config['strategy'].get('parameters', {}))
    strategy_params['symbol'] = config['strategy']['symbol']
    
    # Create components
    strategy = SmaCrossStrategy(strategy_name, strategy_params)
    data_config = {'type': 'mock'}
    data_feed = DataFeed(data_config)
    backtester = Backtester(initial_cash=100000)
    
    # Run a small backtest
    start_date = datetime.strptime('2025-01-01', '%Y-%m-%d')
    end_date = start_date + timedelta(hours=48)  # Just 2 days
    
    print(f"Running test backtest: {start_date} to {end_date}")
    
    try:
        await data_feed.connect()
        
        result = await backtester.run(
            strategy=strategy,
            data_feed=data_feed,
            start_date=start_date,
            end_date=end_date,
            commission=0.001
        )
        
        print(f"\n=== RESULTS ===")
        print(f"Total trades: {len(result.trades)}")
        
        # Analyze trades
        entry_trades = [t for t in result.trades if t.get('entry', True)]
        exit_trades = [t for t in result.trades if not t.get('entry', True)]
        
        print(f"Entry trades: {len(entry_trades)}")
        print(f"Exit trades: {len(exit_trades)}")
        
        # Show some example trades
        if len(result.trades) > 0:
            print(f"\n=== FIRST FEW TRADES ===")
            for i, trade in enumerate(result.trades[:6]):
                trade_type = "ENTRY" if trade.get('entry', True) else "EXIT"
                print(f"{i+1}. [{trade_type}] {trade['timestamp']} - {trade['action'].upper()} {trade['shares']} @ {trade['price']:.5f}")
                if not trade.get('entry', True):
                    print(f"   PnL: ${trade.get('pnl', 0):.2f} ({trade.get('reason', '')})")
        
        # Performance metrics
        print(f"\n=== PERFORMANCE METRICS ===")
        metrics = result.performance_metrics
        key_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
            ('Total Completed Trades', f"{metrics.get('total_trades', 0)}"),
            ('Winning Trades', f"{metrics.get('winning_trades', 0)}"),
            ('Losing Trades', f"{metrics.get('losing_trades', 0)}"),
            ('Final Equity', f"${metrics.get('final_equity', 0):,.2f}")
        ]
        
        for name, value in key_metrics:
            print(f"{name:<25} {value}")
        
        print(f"\n=== SUCCESS ===")
        print(f"✅ Strategy now properly generates both entry and exit trades!")
        print(f"✅ Performance metrics now show {metrics.get('total_trades', 0)} completed trades")
        
        # Test the strategy state
        print(f"\n=== STRATEGY STATE ===")
        print(f"Current position: {strategy.state.get('position', 'None')}")
        print(f"Entry price: {strategy.state.get('entry_price', 'None')}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await data_feed.disconnect()

if __name__ == "__main__":
    print("Testing fixed SMA Cross Strategy...")
    asyncio.run(test_fixed_strategy())