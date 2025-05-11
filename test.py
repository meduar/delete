#!/usr/bin/env python3
"""
Debug the metrics calculation issue
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis.backtester import Backtester
from src.data.feed import DataFeed
from src.strategies.sma_cross import SmaCrossStrategy

async def debug_metrics_issue():
    print("=== DEBUGGING METRICS CALCULATION ISSUE ===\n")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
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
    
    print(f"Running short backtest: {start_date} to {end_date}")
    
    try:
        await data_feed.connect()
        
        result = await backtester.run(
            strategy=strategy,
            data_feed=data_feed,
            start_date=start_date,
            end_date=end_date,
            commission=0.001
        )
        
        print(f"\n=== RAW RESULTS ===")
        print(f"result.trades length: {len(result.trades)}")
        print(f"result.performance_metrics: {result.performance_metrics}")
        
        # Let's look at the trade objects
        print(f"\n=== TRADE ANALYSIS ===")
        
        if len(result.trades) > 0:
            # Look at first trade
            first_trade = result.trades[0]
            print(f"First trade: {first_trade}")
            print(f"First trade keys: {first_trade.keys()}")
            print(f"Is 'entry' in first trade? {'entry' in first_trade}")
            print(f"Entry value: {first_trade.get('entry', 'NOT SET')}")
            
            # Check how many are entry vs exit trades
            entry_trades = [t for t in result.trades if t.get('entry', True)]
            exit_trades = [t for t in result.trades if not t.get('entry', True)]
            
            print(f"\nEntry trades: {len(entry_trades)}")
            print(f"Exit trades: {len(exit_trades)}")
            
            # Check trade structure expected by performance analyzer
            print(f"\n=== CHECKING PERFORMANCE ANALYZER EXPECTATIONS ===")
            
            # The performance analyzer might be looking for different trade structure
            # Let's check what calculate_trade_statistics expects
            
            from src.analysis.performance import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            
            # Check what calculate_trade_statistics is doing
            metrics = analyzer.calculate_metrics(
                equity_curve=result.equity_curve,
                trades=result.trades,
                initial_capital=100000
            )
            
            print(f"Metrics from analyzer: {metrics}")
            print(f"Total trades in metrics: {metrics.get('total_trades', 'NOT FOUND')}")
            
            # Let's manually filter trades like the analyzer does
            exit_trades_manual = [t for t in result.trades if not t.get('entry', True)]
            print(f"\nManual exit trades count: {len(exit_trades_manual)}")
            
            if len(exit_trades_manual) > 0:
                print(f"First exit trade: {exit_trades_manual[0]}")
            else:
                print("NO EXIT TRADES FOUND!")
                print("This is the issue - strategy only creates entry trades")
                
                # Check if we're closing positions
                print(f"\n=== CHECKING POSITION CLOSING ===")
                
                # Look for position closing logic in strategy
                print("Check if strategy._check_exit_conditions is being called")
                print("Check if strategy has position tracking")
        
        print(f"\n=== DIAGNOSIS ===")
        if len(result.trades) > len([t for t in result.trades if not t.get('entry', True)]):
            print("❌ ISSUE FOUND: Strategy only creates entry trades, no exit trades!")
            print("The performance analyzer counts 'completed trades' (entry + exit)")
            print("Since there are no exit trades, completed trades = 0")
            print("\nSOLUTION: Fix the strategy to properly close positions")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await data_feed.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_metrics_issue())