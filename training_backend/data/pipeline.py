"""Data pipeline for processing and preparing trading data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from datetime import datetime, timedelta
from loguru import logger

from .features import FeatureEngineer
from .storage import DataStorage


class DataPipeline:
    """
    End-to-end data pipeline for trading ML system.
    
    Handles:
    - Data loading from multiple sources
    - Feature engineering
    - Train/validation/test splitting
    - Sequence creation for time series
    - Normalization
    """
    
    def __init__(
        self,
        storage: Optional[DataStorage] = None,
        feature_engineer: Optional[FeatureEngineer] = None
    ):
        """
        Initialize data pipeline.
        
        Args:
            storage: DataStorage instance for data persistence
            feature_engineer: FeatureEngineer for creating features
        """
        self.storage = storage or DataStorage()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.normalization_params: Dict[str, Tuple[float, float]] = {}
    
    def load_and_process(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_source: str = "parquet",
        data_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load and process data for given symbols and date range.
        
        Args:
            symbols: List of symbols to load
            start_date: Start date
            end_date: End date
            data_source: Source type ('parquet', 'csv', 'database')
            data_path: Path to data files
            
        Returns:
            Processed DataFrame with features
        """
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Load raw data
        all_data = []
        for symbol in symbols:
            try:
                if data_source == "parquet":
                    file_path = data_path / f"{symbol}.parquet"
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        all_data.append(df)
                elif data_source == "csv":
                    file_path = data_path / f"{symbol}.csv"
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
                continue
        
        if not all_data:
            raise RuntimeError("No data loaded")
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter by date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        # Sort by symbol and timestamp
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
        return df
    
    def add_features(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Add all technical indicators and features.
        
        Args:
            df: Raw OHLCV DataFrame
            include_temporal: Include time-based features
            
        Returns:
            DataFrame with features
        """
        logger.info("Adding features...")
        
        # Process each symbol separately to avoid look-ahead bias
        processed = []
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = self.feature_engineer.add_all_features(
                symbol_df,
                include_temporal=include_temporal
            )
            processed.append(symbol_df)
        
        df_features = pd.concat(processed, ignore_index=True)
        logger.info(f"Added {len(df_features.columns) - len(df.columns)} features")
        
        return df_features
    
    def create_labels(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 4, 24],
        threshold: float = 0.005
    ) -> pd.DataFrame:
        """
        Create prediction labels for supervised learning.
        
        Args:
            df: DataFrame with features
            horizons: Forecast horizons in bars
            threshold: Threshold for direction classification (0.5% = 0.005)
            
        Returns:
            DataFrame with labels
        """
        df = df.copy()
        
        for h in horizons:
            # Future return (regression target)
            df[f'target_return_{h}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.shift(-h) / x - 1
            )
            
            # Direction classification (up/neutral/down)
            df[f'target_direction_{h}'] = 1  # neutral
            df.loc[df[f'target_return_{h}'] > threshold, f'target_direction_{h}'] = 2  # up
            df.loc[df[f'target_return_{h}'] < -threshold, f'target_direction_{h}'] = 0  # down
            
            # Future volatility
            df[f'target_volatility_{h}'] = df.groupby('symbol')['return_1'].transform(
                lambda x: x.shift(-h).rolling(h).std()
            )
        
        # Primary target: next bar direction and return
        df['target_return'] = df['target_return_1']
        df['target_direction'] = df['target_direction_1']
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        method: str = "time"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            method: Split method ('time' or 'random')
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if method == "time":
            # Time-based split (no look-ahead bias)
            df = df.sort_values('timestamp')
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
        else:
            # Random split (for testing only, causes look-ahead bias)
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=42)
            val_size = val_ratio / (1 - train_ratio)
            val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=42)
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series model input.
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to use as features
            target_cols: Columns to use as targets
            sequence_length: Lookback window size
            
        Returns:
            Tuple of (X, y) arrays
        """
        X_list = []
        y_list = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('timestamp')
            
            features = symbol_df[feature_cols].values
            targets = symbol_df[target_cols].values
            
            for i in range(sequence_length, len(features)):
                X_list.append(features[i-sequence_length:i])
                y_list.append(targets[i])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X)} sequences of shape {X.shape[1:]}")
        return X, y
    
    def normalize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.
        
        Args:
            df: DataFrame to normalize
            feature_cols: Columns to normalize
            fit: Whether to fit normalization params (True for training)
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        if fit:
            df, self.normalization_params = self.feature_engineer.normalize_features(
                df, feature_cols, method='zscore'
            )
        else:
            # Apply saved parameters
            for col, (mean, std) in self.normalization_params.items():
                if col in df.columns and std > 0:
                    df[col] = (df[col] - mean) / std
        
        return df
    
    def get_walk_forward_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 10,
        train_years: float = 2.0,
        test_months: int = 3
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward validation splits.
        
        Args:
            df: Full DataFrame
            n_splits: Number of walk-forward windows
            train_years: Training period in years
            test_months: Testing period in months
            
        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values('timestamp')
        
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        train_days = int(train_years * 365)
        test_days = int(test_months * 30)
        
        splits = []
        current_start = min_date
        
        for i in range(n_splits):
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            if test_end > max_date:
                break
            
            train_df = df[(df['timestamp'] >= current_start) & (df['timestamp'] < train_end)]
            test_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < test_end)]
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
                logger.info(f"Split {i+1}: Train {len(train_df)} bars, Test {len(test_df)} bars")
            
            current_start = train_end
        
        return splits
    
    def run_full_pipeline(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_path: Path,
        sequence_length: int = 60,
        horizons: List[int] = [1, 4, 24]
    ) -> Dict[str, np.ndarray]:
        """
        Run the complete data pipeline.
        
        Args:
            symbols: Symbols to process
            start_date: Start date
            end_date: End date
            data_path: Path to data files
            sequence_length: Lookback window
            horizons: Forecast horizons
            
        Returns:
            Dictionary with train/val/test data
        """
        # Load data
        df = self.load_and_process(symbols, start_date, end_date, data_path=data_path)
        
        # Add features
        df = self.add_features(df)
        
        # Create labels
        df = self.create_labels(df, horizons=horizons)
        
        # Drop rows with NaN targets
        df = df.dropna(subset=[f'target_return_{h}' for h in horizons])
        
        # Get feature columns (exclude targets and metadata)
        exclude_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        exclude_cols += [col for col in df.columns if col.startswith('target_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        target_cols = ['target_return', 'target_direction']
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Normalize features
        train_df = self.normalize(train_df, feature_cols, fit=True)
        val_df = self.normalize(val_df, feature_cols, fit=False)
        test_df = self.normalize(test_df, feature_cols, fit=False)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_df, feature_cols, target_cols, sequence_length)
        X_val, y_val = self.create_sequences(val_df, feature_cols, target_cols, sequence_length)
        X_test, y_test = self.create_sequences(test_df, feature_cols, target_cols, sequence_length)
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_cols': feature_cols,
            'normalization_params': self.normalization_params
        }
