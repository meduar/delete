#!/usr/bin/env python3
"""
Script to create all the necessary files for the trading framework
Run this script from the root directory of your project
"""

import os
from pathlib import Path

# Files and their content
FILES = {
    # Init files
    'src/__init__.py': '''"""
Modern Trading Framework
"""
__version__ = "1.0.0"

from src.core.engine import TradingEngine
from src.core.strategy import BaseStrategy, Signal
from src.core.portfolio import Portfolio, Position
from src.analysis.backtester import Backtester

__all__ = [
    'TradingEngine',
    'BaseStrategy',
    'Signal',
    'Portfolio',
    'Position',
    'Backtester'
]''',
    
    'src/core/__init__.py': '''from .engine import TradingEngine
from .strategy import BaseStrategy, Signal
from .portfolio import Portfolio, Position
from .risk_manager import RiskManager

__all__ = [
    'TradingEngine',
    'BaseStrategy', 
    'Signal',
    'Portfolio',
    'Position',
    'RiskManager'
]''',
    
    'src/data/__init__.py': '''from .feed import DataFeed
from .processor import DataProcessor

__all__ = [
    'DataFeed',
    'DataProcessor'
]''',
    
    'src/data/providers/__init__.py': '''from .oanda import OandaDataProvider

__all__ = [
    'OandaDataProvider'
]''',
    
    'src/execution/__init__.py': '''from .broker import Broker
from .order_manager import OrderManager
from .oanda_broker import OandaBroker

__all__ = [
    'Broker',
    'OrderManager',
    'OandaBroker'
]''',
    
    'src/strategies/__init__.py': '''from .sma_cross import SmaCrossStrategy
from .bollinger import BollingerStrategy
from .momentum import MomentumStrategy

__all__ = [
    'SmaCrossStrategy',
    'BollingerStrategy',
    'MomentumStrategy'
]''',
    
    'src/analysis/__init__.py': '''from .backtester import Backtester, BacktestResult
from .performance import PerformanceAnalyzer
from .visualizer import TradingVisualizer
from .indicators import SMA, EMA, ATR, BollingerBands, RSI, ROC, MACD, Stochastic, CrossDetector

__all__ = [
    'Backtester',
    'BacktestResult',
    'PerformanceAnalyzer', 
    'TradingVisualizer',
    'SMA',
    'EMA',
    'ATR',
    'BollingerBands',
    'RSI',
    'ROC',
    'MACD',
    'Stochastic',
    'CrossDetector'
]''',
    
    'src/monitoring/__init__.py': '''from .logger import TradingLogger
from .alerts import AlertManager
from .metrics import MetricsCollector

__all__ = [
    'TradingLogger',
    'AlertManager',
    'MetricsCollector'
]''',
    
    'src/utils/__init__.py': '''from .config import Config
from .helpers import retry_async, timing_decorator, RateLimiter

__all__ = [
    'Config',
    'retry_async',
    'timing_decorator',
    'RateLimiter'
]''',
    
    'tests/__init__.py': '# Tests package',
    'scripts/__init__.py': '# Scripts package',
    
    # Create empty files that will be filled with content
    'src/core/engine.py': '# Engine placeholder',
    'src/core/strategy.py': '# Strategy placeholder',
    'src/core/risk_manager.py': '# Risk manager placeholder',
    'src/data/feed.py': '# Data feed placeholder',
    'src/data/processor.py': '# Data processor placeholder',
    'src/data/providers/oanda.py': '# OANDA provider placeholder',
    'src/execution/broker.py': '# Broker placeholder',
    'src/execution/order_manager.py': '# Order manager placeholder',
    'src/execution/oanda_broker.py': '# OANDA broker placeholder',
    'src/strategies/sma_cross.py': '# SMA strategy placeholder',
    'src/strategies/bollinger.py': '# Bollinger strategy placeholder',
    'src/strategies/momentum.py': '# Momentum strategy placeholder',
    'src/analysis/backtester.py': '# Backtester placeholder',
    'src/analysis/performance.py': '# Performance placeholder',
    'src/analysis/visualizer.py': '# Visualizer placeholder',
    'src/analysis/indicators.py': '# Indicators placeholder',
    'src/monitoring/logger.py': '# Logger placeholder',
    'src/monitoring/alerts.py': '# Alerts placeholder',
    'src/monitoring/metrics.py': '# Metrics placeholder',
    'src/utils/config.py': '# Config placeholder',
    'src/utils/helpers.py': '# Helpers placeholder',
}

def create_directories():
    """Create all necessary directories"""
    directories = [
        'src/core',
        'src/data/providers',
        'src/execution',
        'src/strategies',
        'src/analysis',
        'src/monitoring',
        'src/utils',
        'tests',
        'scripts',
        'config',
        'logs',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def create_files():
    """Create all necessary files"""
    for file_path, content in FILES.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only create if file doesn't exist or is placeholder
        if not path.exists() or path.read_text().startswith('# '):
            path.write_text(content)
            print(f"Created file: {file_path}")

def create_config_files():
    """Create configuration files"""
    config_files = {
        'config/settings.yaml': '''trading:
  initial_cash: 100000
  max_position_size: 0.1
  max_daily_loss: -1000
  max_trades_per_day: 10
  max_portfolio_drawdown: 0.15
  symbols:
    - EUR_USD
    - GBP_USD

data:
  provider:
    type: oanda
    environment: practice
    api_key: ${OANDA_API_KEY}
    account_id: ${OANDA_ACCOUNT_ID}

execution:
  broker:
    type: oanda
    environment: practice
    api_key: ${OANDA_API_KEY}
    account_id: ${OANDA_ACCOUNT_ID}

strategies:
  sma_cross:
    type: sma_cross
    parameters:
      fast_period: 10
      slow_period: 30
      atr_period: 14
      stop_loss_atr: 1.5
      take_profit_atr: 3.0
  
  bollinger:
    type: bollinger
    parameters:
      bb_period: 20
      bb_std_dev: 2.0
      rsi_period: 14
      mode: 'reversion'
      rsi_oversold: 30
      rsi_overbought: 70
  
  momentum:
    type: momentum
    parameters:
      rsi_period: 14
      roc_period: 10
      ema_period: 50
      roc_threshold: 0.02

monitoring:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
  
  logging:
    level: INFO
    dir: logs
''',
        
        '.env.example': '''# OANDA Configuration
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Environment
ENV=development
LOG_LEVEL=INFO
''',
        
        'requirements.txt': '''# Core
python>=3.11
pyyaml>=6.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
aiohttp>=3.8.0

# OANDA
oandapyV20>=0.6.0

# Analysis
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Monitoring
python-telegram-bot>=20.0
psutil>=5.9.0

# Testing
pytest>=7.3.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.10.0

# Development
black>=23.0.0
isort>=5.12.0
mypy>=1.2.0
'''
    }
    
    for file_path, content in config_files.items():
        path = Path(file_path)
        if not path.exists():
            path.write_text(content)
            print(f"Created config file: {file_path}")

def main():
    print("🚀 Setting up trading framework structure...")
    
    # Create all directories
    create_directories()
    
    # Create all files
    create_files()
    
    # Create config files
    create_config_files()
    
    print("\n✅ Basic structure created!")
    print("\nNext steps:")
    print("1. Copy all the actual code from our conversation into the placeholder files")
    print("2. Run: python -m venv venv")
    print("3. Run: source venv/bin/activate  # Windows: venv\\Scripts\\activate")
    print("4. Run: pip install -r requirements.txt")
    print("5. Copy .env.example to .env and fill in your OANDA credentials")
    print("6. Run tests: python -m pytest tests/test_strategies.py -v")

if __name__ == "__main__":
    main()