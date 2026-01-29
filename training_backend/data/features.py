"""Feature engineering module with 50+ technical indicators using ta library."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger

try:
    from ta import add_all_ta_features
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ChaikinMoneyFlowIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta not installed. Run: pip install ta")


class FeatureEngineer:
    """
    Generate 50+ technical indicators and features for ML models.
    
    Uses the 'ta' library (Technical Analysis Library in Python).
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        if not TA_AVAILABLE:
            raise ImportError("ta library required. Install with: pip install ta")
    
    def add_all_features(
        self,
        df: pd.DataFrame,
        include_temporal: bool = True,
        include_returns: bool = True,
        include_trend: bool = True,
        include_momentum: bool = True,
        include_volatility: bool = True,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """Add all technical indicators and features to dataframe."""
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if include_returns:
            df = self._add_return_features(df)
        
        if include_trend:
            df = self._add_trend_indicators(df)
        
        if include_momentum:
            df = self._add_momentum_indicators(df)
        
        if include_volatility:
            df = self._add_volatility_indicators(df)
        
        if include_volume:
            df = self._add_volume_indicators(df)
        
        if include_temporal and 'timestamp' in df.columns:
            df = self._add_temporal_features(df)
        
        if 'symbol' in df.columns:
            df = self._add_relative_features(df)
        
        df = df.ffill().bfill()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        logger.info(f"Generated {len(df.columns)} features")
        return df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        for period in [1, 2, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_range'] = (df['close'] - df['open']) / df['open']
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        df['cum_return_5'] = df['return_1'].rolling(5).sum()
        df['cum_return_10'] = df['return_1'].rolling(10).sum()
        df['cum_return_20'] = df['return_1'].rolling(20).sum()
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            sma = SMAIndicator(df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [9, 12, 21, 26, 50]:
            ema = EMAIndicator(df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
            df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
        
        # MACD
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # ADX
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # Aroon
        aroon = AroonIndicator(df['high'], df['low'])
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
        
        # SMA crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['sma_cross_10_50'] = (df['sma_10'] > df['sma_50']).astype(int)
        df['sma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [7, 14, 21]:
            rsi = RSIIndicator(df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        
        # Stochastic
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        willr = WilliamsRIndicator(df['high'], df['low'], df['close'])
        df['willr'] = willr.williams_r()
        
        # ROC
        for period in [5, 10, 20]:
            roc = ROCIndicator(df['close'], window=period)
            df[f'roc_{period}'] = roc.roc()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        for period in [7, 14, 21]:
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
            df[f'atr_{period}'] = atr.average_true_range()
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct'] = bb.bollinger_pband()
        
        # Keltner Channel
        kc = KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        df['kc_middle'] = kc.keltner_channel_mband()
        
        # Donchian Channel
        dc = DonchianChannel(df['high'], df['low'], df['close'])
        df['dc_upper'] = dc.donchian_channel_hband()
        df['dc_lower'] = dc.donchian_channel_lband()
        df['dc_middle'] = dc.donchian_channel_mband()
        
        # Historical Volatility
        for period in [5, 10, 20, 60]:
            df[f'volatility_{period}'] = df['return_1'].rolling(period).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        # OBV
        obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # Volume SMA
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
        
        # VWAP
        if 'vwap' not in df.columns:
            try:
                vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
                df['vwap'] = vwap.volume_weighted_average_price()
            except:
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        # CMF
        cmf = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = cmf.chaikin_money_flow()
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with cyclical encoding."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        hour = df['timestamp'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        day = df['timestamp'].dt.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * day / 7)
        df['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        dom = df['timestamp'].dt.day
        df['dom_sin'] = np.sin(2 * np.pi * dom / 31)
        df['dom_cos'] = np.cos(2 * np.pi * dom / 31)
        
        month = df['timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        week = df['timestamp'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * week / 52)
        df['week_cos'] = np.cos(2 * np.pi * week / 52)
        
        df['is_monday'] = (day == 0).astype(int)
        df['is_friday'] = (day == 4).astype(int)
        df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        
        return df
    
    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative/cross-asset features."""
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            if len(symbols) > 1:
                market_returns = df.groupby('timestamp')['return_1'].mean()
                df = df.merge(
                    market_returns.rename('market_return'),
                    left_on='timestamp',
                    right_index=True,
                    how='left'
                )
                df['relative_strength'] = df['return_1'] - df['market_return']
                df['relative_strength_10'] = df.groupby('symbol')['relative_strength'].transform(
                    lambda x: x.rolling(10).sum()
                )
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated."""
        features = []
        
        for period in [1, 2, 5, 10, 20]:
            features.extend([f'return_{period}', f'log_return_{period}'])
        features.extend(['hl_range', 'oc_range', 'gap', 'cum_return_5', 'cum_return_10', 'cum_return_20'])
        
        for period in [5, 10, 20, 50, 200]:
            features.extend([f'sma_{period}', f'sma_ratio_{period}'])
        for period in [9, 12, 21, 26, 50]:
            features.extend([f'ema_{period}', f'ema_ratio_{period}'])
        features.extend([
            'macd', 'macd_signal', 'macd_hist', 'adx', 'di_plus', 'di_minus',
            'aroon_up', 'aroon_down', 'aroon_osc',
            'sma_cross_5_20', 'sma_cross_10_50', 'sma_cross_50_200'
        ])
        
        for period in [7, 14, 21]:
            features.append(f'rsi_{period}')
        features.extend(['stoch_k', 'stoch_d', 'willr', 'roc_5', 'roc_10', 'roc_20'])
        
        for period in [7, 14, 21]:
            features.extend([f'atr_{period}', f'atr_ratio_{period}'])
        features.extend([
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
            'kc_upper', 'kc_lower', 'kc_middle',
            'dc_upper', 'dc_lower', 'dc_middle',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_60'
        ])
        
        features.extend(['obv', 'obv_sma', 'vwap', 'vwap_ratio', 'cmf'])
        for period in [5, 10, 20]:
            features.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
        
        features.extend([
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'dom_sin', 'dom_cos', 'month_sin', 'month_cos',
            'week_sin', 'week_cos',
            'is_monday', 'is_friday', 'is_month_start', 'is_month_end'
        ])
        
        return features
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        method: str = 'zscore'
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """Normalize feature columns."""
        df = df.copy()
        params = {}
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            if method == 'zscore':
                mean = df[col].mean(skipna=True)
                std = df[col].std(skipna=True)
                if pd.notna(std) and std > 0:
                    df[col] = (df[col] - mean) / std
                params[col] = (mean if pd.notna(mean) else 0, std if pd.notna(std) else 1)
            elif method == 'minmax':
                min_val = df[col].min(skipna=True)
                max_val = df[col].max(skipna=True)
                if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                params[col] = (min_val if pd.notna(min_val) else 0, max_val if pd.notna(max_val) else 1)
        
        return df, params


if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'TEST',
        'open': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, n)
    })
    df['high'] = df['open'] + np.abs(np.random.randn(n))
    df['low'] = df['open'] - np.abs(np.random.randn(n))
    df['close'] = df['open'] + np.random.randn(n) * 0.5
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    
    fe = FeatureEngineer()
    df_features = fe.add_all_features(df)
    
    print(f"Original columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")
