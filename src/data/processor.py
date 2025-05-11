import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataProcessor:
    """Process and prepare market data for analysis"""
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """Calculate price returns"""
        return df['close'].pct_change(periods=periods)
    
    @staticmethod
    def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate rolling volatility"""
        returns = DataProcessor.calculate_returns(df)
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def detect_gaps(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
        """Detect price gaps in the data"""
        df = df.copy()
        df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['is_gap'] = df['gap'] > threshold
        return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data to different timeframe"""
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Forward fill missing values
        resampled.fillna(method='ffill', inplace=True)
        return resampled
    
    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe"""
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        
        # Add session indicators (assuming futures market hours)
        df['is_asian_session'] = df['hour'].between(23, 8)
        df['is_london_session'] = df['hour'].between(8, 16)
        df['is_ny_session'] = df['hour'].between(13, 21)
        
        return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        df = df.copy()
        
        # Remove rows with invalid OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        df = df[~invalid_mask]
        
        # Remove extreme outliers
        for col in ['open', 'high', 'low', 'close']:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df