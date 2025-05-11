# src/__init__.py
"""
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
]

