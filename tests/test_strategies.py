"""
Fixed test module for trading strategies
"""

import unittest
import asyncio
import pytest
from datetime import datetime
import pandas as pd
import numpy as np

# Import with fixed order to avoid circular dependencies
from src.data.feed import MarketData
from src.core.strategy import Signal
from src.strategies.sma_cross import SmaCrossStrategy
from src.strategies.bollinger import BollingerStrategy
from src.strategies.momentum import MomentumStrategy


class TestStrategies(unittest.TestCase):
    """Test trading strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sma_params = {
            'fast_period': 10,
            'slow_period': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'symbol': 'EUR_USD'
        }
        
        self.bollinger_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'mode': 'reversion',
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        self.momentum_params = {
            'rsi_period': 14,
            'roc_period': 10,
            'ema_period': 50,
            'roc_threshold': 0.02
        }
        
        # Create test market data
        self.test_market_data = MarketData(
            timestamp=datetime.now(),
            symbol='EUR_USD',
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000
        )
    
    def test_sma_strategy_initialization(self):
        """Test SMA strategy initialization"""
        strategy = SmaCrossStrategy('test_sma', self.sma_params)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, 'test_sma')
        self.assertEqual(strategy.parameters['fast_period'], 10)
        self.assertEqual(strategy.parameters['slow_period'], 30)
    
    def test_bollinger_strategy_initialization(self):
        """Test Bollinger strategy initialization"""
        strategy = BollingerStrategy('test_bollinger', self.bollinger_params)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, 'test_bollinger')
        self.assertEqual(strategy.params['bb_period'], 20)
        self.assertEqual(strategy.params['mode'], 'reversion')
    
    def test_momentum_strategy_initialization(self):
        """Test Momentum strategy initialization"""
        strategy = MomentumStrategy('test_momentum', self.momentum_params)
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, 'test_momentum')
        self.assertEqual(strategy.parameters['rsi_period'], 14)
    
    def test_signal_metadata(self):
        """Test that signals contain proper metadata"""
        # Test signal creation
        test_signal = Signal(
            action='buy',
            strength=0.8,
            price=1.1005,
            symbol='EUR_USD',
            timestamp=datetime.now(),
            metadata={
                'reason': 'Test signal',
                'stop_loss': 1.0980,
                'take_profit': 1.1050
            }
        )
        
        self.assertEqual(test_signal.action, 'buy')
        self.assertEqual(test_signal.strength, 0.8)
        self.assertIn('reason', test_signal.metadata)
        self.assertIn('stop_loss', test_signal.metadata)
        self.assertIn('take_profit', test_signal.metadata)


class TestAsyncStrategies:
    """Async tests for strategies using pytest"""
    
    @pytest.mark.asyncio
    async def test_sma_signal_generation(self):
        """Test SMA signal generation with mock data"""
        sma_params = {
            'fast_period': 10,
            'slow_period': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'symbol': 'EUR_USD'
        }
        
        strategy = SmaCrossStrategy('test_sma', sma_params)
        
        # Setup indicators with mock data feed
        class MockDataFeed:
            pass
        
        await strategy.initialize(MockDataFeed())
        
        # Create test market data
        test_market_data = MarketData(
            timestamp=datetime.now(),
            symbol='EUR_USD',
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000
        )
        
        # Test with insufficient data (should return hold)
        signal = await strategy._generate_signal(test_market_data)
        assert isinstance(signal, Signal)
        assert signal.action == 'hold'
        
        # Update strategy with more data points
        for i in range(40):  # Add enough data for indicators
            price = 1.1000 + i * 0.001
            market_data = MarketData(
                timestamp=datetime.now(),
                symbol='EUR_USD',
                open=price,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
                volume=1000
            )
            await strategy._update_indicators(market_data)
        
        # Test signal generation with sufficient data
        signal = await strategy._generate_signal(test_market_data)
        assert isinstance(signal, Signal)
        assert signal.action in ['buy', 'sell', 'hold']
    
    @pytest.mark.asyncio
    async def test_bollinger_strategy_modes(self):
        """Test Bollinger strategy in different modes"""
        bollinger_params = {
            'bb_period': 20,
            'bb_std_dev': 2.0,
            'rsi_period': 14,
            'mode': 'reversion',
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
        
        # Test reversion mode
        strategy_reversion = BollingerStrategy('test_bb_reversion', {
            **bollinger_params,
            'mode': 'reversion'
        })
        
        # Test breakout mode
        strategy_breakout = BollingerStrategy('test_bb_breakout', {
            **bollinger_params,
            'mode': 'breakout'
        })
        
        # Test squeeze mode
        strategy_squeeze = BollingerStrategy('test_bb_squeeze', {
            **bollinger_params,
            'mode': 'squeeze'
        })
        
        # Initialize all strategies
        mock_feed = type('MockFeed', (), {})()
        await strategy_reversion.initialize(mock_feed)
        await strategy_breakout.initialize(mock_feed)
        await strategy_squeeze.initialize(mock_feed)
        
        # Test that they all initialize without error
        assert strategy_reversion.params['mode'] == 'reversion'
        assert strategy_breakout.params['mode'] == 'breakout'
        assert strategy_squeeze.params['mode'] == 'squeeze'
    
    @pytest.mark.asyncio
    async def test_momentum_signal_strength(self):
        """Test momentum strategy signal strength calculation"""
        momentum_params = {
            'rsi_period': 14,
            'roc_period': 10,
            'ema_period': 50,
            'roc_threshold': 0.02
        }
        
        strategy = MomentumStrategy('test_momentum', momentum_params)
        
        # Initialize with mock data
        await strategy.initialize(type('MockFeed', (), {})())
        
        # Add data points to build up indicator values
        for i in range(20):
            price = 1.1000 + i * 0.002  # Trending upward
            market_data = MarketData(
                timestamp=datetime.now(),
                symbol='EUR_USD',
                open=price,
                high=price + 0.001,
                low=price - 0.001,
                close=price,
                volume=1000
            )
            await strategy._update_indicators(market_data)
        
        # Generate signal
        test_market_data = MarketData(
            timestamp=datetime.now(),
            symbol='EUR_USD',
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000
        )
        
        signal = await strategy._generate_signal(test_market_data)
        
        # Check signal properties
        assert isinstance(signal, Signal)
        if signal.action != 'hold':
            assert signal.strength >= 0.0
            assert signal.strength <= 1.0
    
    @pytest.mark.asyncio
    async def test_strategy_state_management(self):
        """Test strategy state management"""
        sma_params = {
            'fast_period': 10,
            'slow_period': 30,
            'atr_period': 14,
            'stop_loss_atr': 1.5,
            'take_profit_atr': 3.0,
            'symbol': 'EUR_USD'
        }
        
        strategy = SmaCrossStrategy('test_sma', sma_params)
        
        # Check initial state
        assert strategy.indicators is not None
        assert strategy.state is not None
        
        # Initialize and check state update
        await strategy.initialize(type('MockFeed', (), {})())
        assert 'last_signal' in strategy.state


# Functions to run tests
def run_unittests():
    """Run unittest tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStrategies)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_async_tests():
    """Run async tests with pytest"""
    return pytest.main([__file__, '-v', '-k', 'TestAsyncStrategies']) == 0


if __name__ == '__main__':
    # Run unittest tests first
    unittest_success = run_unittests()
    
    # Then run async tests
    async_success = run_async_tests()
    
    # Exit with appropriate code
    import sys
    if unittest_success and async_success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)