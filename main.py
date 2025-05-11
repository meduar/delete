#!/usr/bin/env python3
"""
Main entry point for the trading framework
Reads config.json and executes accordingly
"""

import sys
import os
import json
import asyncio
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports
from src.core.engine import TradingEngine
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

def load_config(config_file='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Set environment variables for OANDA
        if config.get('oanda'):
            os.environ['OANDA_API_KEY'] = config['oanda']['api_key']
            os.environ['OANDA_ACCOUNT_ID'] = config['oanda']['account_id']
        
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found!")
        print("Please create config.json with your settings.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_file}")
        print(e)
        sys.exit(1)

async def run_backtest(config):
    """Run backtest mode"""
    print("=== BACKTEST MODE ===")
    
    # Get configuration
    backtest_config = config['backtest']
    strategy_config = config['strategy']
    trading_config = config['trading']
    
    # Parse dates
    start_date = datetime.strptime(backtest_config['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(backtest_config['end_date'], '%Y-%m-%d')
    
    print(f"Strategy: {strategy_config['name']}")
    print(f"Symbol: {strategy_config['symbol']}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Cash: ${trading_config['initial_cash']:,}")
    
    # Create strategy
    strategy_name = strategy_config['name']
    strategy_params = config.get('strategies', {}).get(strategy_name, {}).copy()
    strategy_params.update(strategy_config.get('parameters', {}))
    strategy_params['symbol'] = strategy_config['symbol']
    
    strategy = STRATEGY_CLASSES[strategy_name](strategy_name, strategy_params)
    
    # Create data feed
    data_config = config['data']['provider']
    if config.get('oanda'):
        data_config.update(config['oanda'])
    
    # Use mock data if specified
    if config['data'].get('use_mock', False):
        data_config['type'] = 'mock'
    
    data_feed = DataFeed(data_config)
    
    # Create backtester
    backtester = Backtester(initial_cash=trading_config['initial_cash'])
    
    try:
        # Connect to data feed
        await data_feed.connect()
        
        # Run backtest
        print("\nRunning backtest...")
        result = await backtester.run(
            strategy=strategy,
            data_feed=data_feed,
            start_date=start_date,
            end_date=end_date,
            commission=trading_config['commission']
        )
        
        # Display results
        print("\n=== BACKTEST RESULTS ===")
        metrics = result.performance_metrics
        
        key_metrics = [
            ('Total Return', f"{metrics.get('total_return', 0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
            ('Win Rate', f"{metrics.get('win_rate', 0):.2f}%"),
            ('Total Trades', f"{metrics.get('total_trades', 0)}"),
            ('Final Equity', f"${metrics.get('final_equity', 0):,.2f}")
        ]
        
        for name, value in key_metrics:
            print(f"{name:<15} {value}")
        
        # Save results
        if backtest_config.get('output_dir'):
            output_dir = backtest_config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON results
            results_file = f"{output_dir}/{strategy_name}_{strategy_config['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'config': config,
                    'results': {
                        'metrics': metrics,
                        'trades': [
                            {
                                'date': str(trade.get('timestamp', '')),
                                'action': trade.get('action', ''),
                                'price': trade.get('price', 0),
                                'shares': trade.get('shares', 0),
                                'pnl': trade.get('pnl', 0)
                            }
                            for trade in result.trades
                        ]
                    }
                }, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
            
            # Generate charts if requested
            if backtest_config.get('generate_charts', False):
                visualizer = TradingVisualizer()
                
                # Equity curve
                fig = visualizer.plot_equity_curve(result.equity_curve, result.trades)
                chart_file = f"{output_dir}/equity_curve_{strategy_name}_{strategy_config['symbol']}.html"
                fig.write_html(chart_file)
                print(f"Equity curve saved to: {chart_file}")
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await data_feed.disconnect()

async def run_live_trading(config):
    """Run live trading mode"""
    print("=== LIVE TRADING MODE ===")
    
    live_config = config['live_trading']
    
    if live_config.get('paper_trading', False):
        print("Paper trading enabled - no real orders will be placed")
    else:
        print("WARNING: Real trading enabled!")
        response = input("Are you sure you want to continue? (yes/no): ").lower()
        if response != 'yes':
            print("Cancelled.")
            return
    
    # Create trading engine
    engine = TradingEngine(config)
    
    try:
        # Initialize engine
        await engine.initialize()
        
        # Load strategy
        strategy_name = config['strategy']['name']
        strategy_params = config.get('strategies', {}).get(strategy_name, {}).copy()
        strategy_params.update(config['strategy'].get('parameters', {}))
        
        # Create strategy instance
        strategy = STRATEGY_CLASSES[strategy_name](strategy_name, strategy_params)
        engine.strategy = strategy
        
        # Run live trading
        symbols = live_config.get('symbols', [config['strategy']['symbol']])
        await engine.run(mode='live', symbols=symbols)
        
    except KeyboardInterrupt:
        print("\nStopping live trading...")
    except Exception as e:
        print(f"Error during live trading: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.shutdown()

async def run_test_mode(config):
    """Run test mode to verify setup"""
    print("=== TEST MODE ===")
    
    # Test data connection
    print("Testing data connection...")
    data_config = config['data']['provider']
    if config.get('oanda'):
        data_config.update(config['oanda'])
    
    data_feed = DataFeed(data_config)
    
    try:
        await data_feed.connect()
        
        # Test getting current price
        symbol = config['strategy']['symbol']
        price = await data_feed.get_current_price(symbol)
        print(f"✓ Connection successful! {symbol} price: {price}")
        
        # Test streaming for 5 seconds
        print(f"Testing live data stream for 5 seconds...")
        count = 0
        start_time = asyncio.get_event_loop().time()
        
        async for market_data in data_feed.stream_live_data([symbol]):
            count += 1
            if count % 10 == 0:
                print(f"Received {count} ticks...")
            
            if asyncio.get_event_loop().time() - start_time > 5:
                break
        
        print(f"✓ Received {count} ticks")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    finally:
        await data_feed.disconnect()

async def main():
    """Main entry point"""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Determine mode
    mode = config.get('mode', 'backtest')
    
    print(f"\nRunning in {mode.upper()} mode")
    print("="*50)
    
    # Execute based on mode
    if mode == 'backtest':
        await run_backtest(config)
    elif mode == 'live':
        await run_live_trading(config)
    elif mode == 'test':
        await run_test_mode(config)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: backtest, live, test")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Trading Framework ===")
    print("Configuration file: config.json")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)