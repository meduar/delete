#!/usr/bin/env python3
"""
View backtest results
"""

import json
import os
import sys
from datetime import datetime

def view_latest_results():
    """View the most recent backtest results"""
    
    results_dir = "backtest_results"
    
    if not os.path.exists(results_dir):
        print(f"No results directory found at {results_dir}")
        return
    
    # Find the most recent JSON file
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No result files found")
        return
    
    # Sort by modification time
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_file = json_files[0]
    
    print(f"Loading results from: {latest_file}")
    
    # Load and display results
    with open(os.path.join(results_dir, latest_file), 'r') as f:
        data = json.load(f)
    
    config = data.get('config', {})
    results = data.get('results', {})
    metrics = results.get('metrics', {})
    trades = results.get('trades', [])
    
    print("\n=== BACKTEST CONFIGURATION ===")
    print(f"Strategy: {config.get('strategy', {}).get('name', 'N/A')}")
    print(f"Symbol: {config.get('strategy', {}).get('symbol', 'N/A')}")
    print(f"Period: {config.get('backtest', {}).get('start_date')} to {config.get('backtest', {}).get('end_date')}")
    print(f"Initial Cash: ${config.get('trading', {}).get('initial_cash', 0):,}")
    print(f"Commission: {config.get('trading', {}).get('commission', 0)*100:.3f}%")
    
    print("\n=== PERFORMANCE METRICS ===")
    key_metrics = [
        ('Total Return', f"{metrics.get('total_return', 0):.2f}%"),
        ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
        ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
        ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
        ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
        ('Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"),
        ('Total Trades', f"{metrics.get('total_trades', 0)}"),
        ('Winning Trades', f"{metrics.get('winning_trades', 0)}"),
        ('Losing Trades', f"{metrics.get('losing_trades', 0)}"),
        ('Avg Win', f"${metrics.get('avg_winning_trade', 0):.2f}"),
        ('Avg Loss', f"${metrics.get('avg_losing_trade', 0):.2f}"),
        ('Final Equity', f"${metrics.get('final_equity', 0):,.2f}")
    ]
    
    for name, value in key_metrics:
        print(f"{name:<15} {value}")
    
    print("\n=== TRADE SUMMARY ===")
    if trades:
        print(f"Total Trades: {len(trades)}")
        
        # Summarize by action
        buy_trades = [t for t in trades if t.get('action') == 'buy']
        sell_trades = [t for t in trades if t.get('action') == 'sell']
        
        print(f"Buy Signals: {len(buy_trades)}")
        print(f"Sell Signals: {len(sell_trades)}")
        
        # Show first few trades
        print("\nFirst 5 trades:")
        for i, trade in enumerate(trades[:5]):
            date = trade.get('date', '')
            action = trade.get('action', '').upper()
            price = trade.get('price', 0)
            shares = trade.get('shares', 0)
            pnl = trade.get('pnl', 0)
            print(f"  {i+1}. {date[:19]} - {action} {shares} @ {price:.5f} (PnL: ${pnl:.2f})")
    else:
        print("No trades executed")
    
    print("\n=== FILES GENERATED ===")
    chart_file = latest_file.replace('.json', '').replace('backtest_results/', 'backtest_results/equity_curve_') + '.html'
    if os.path.exists(chart_file):
        print(f"Equity Chart: {chart_file}")
    print(f"Results JSON: {os.path.join(results_dir, latest_file)}")

if __name__ == "__main__":
    view_latest_results()