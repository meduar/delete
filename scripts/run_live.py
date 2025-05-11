#!/usr/bin/env python3
import argparse
from src.core.engine import TradingEngine

def main():
    parser = argparse.ArgumentParser(description='Run live trading')
    parser.add_argument('--config', default='config/settings.yaml')
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--symbol', required=True)
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = TradingEngine(args.config)
    engine.initialize()
    
    # Load strategy
    engine.load_strategy(args.strategy)
    
    # Run live trading
    engine.run(mode='live')

if __name__ == "__main__":
    main()