"""Alpaca API data fetcher for US stocks."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
from loguru import logger

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not installed. Run: pip install alpaca-trade-api")


class AlpacaFetcher:
    """Fetch historical stock data from Alpaca Markets API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca data fetcher.
        
        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            paper: Use paper trading endpoint
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._client = None
        
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Install with: pip install alpaca-py"
            )
    
    @property
    def client(self) -> 'StockHistoricalDataClient':
        """Lazy initialization of Alpaca client."""
        if self._client is None:
            if self.api_key and self.secret_key:
                self._client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
            else:
                # Use without auth for free tier
                self._client = StockHistoricalDataClient()
        return self._client
    
    def fetch_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1Day",
        adjustment: str = "all"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            adjustment: Price adjustment (raw, split, dividend, all)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Parse timeframe
        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use one of {list(tf_map.keys())}")
        
        logger.info(f"Fetching {len(symbols)} symbols from {start_date} to {end_date}")
        
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf_map[timeframe],
            start=start_date,
            end=end_date,
        )
        
        try:
            bars = self.client.get_stock_bars(request_params)
            df = bars.df.reset_index()
            
            # Standardize column names
            df = df.rename(columns={
                'symbol': 'symbol',
                'timestamp': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vwap': 'vwap',
                'trade_count': 'trade_count'
            })
            
            logger.info(f"Fetched {len(df)} bars for {len(symbols)} symbols")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Alpaca data: {e}")
            raise
    
    def fetch_single_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """Fetch data for a single symbol."""
        return self.fetch_bars([symbol], start_date, end_date, timeframe)
    
    def fetch_sp500(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """Fetch S&P 500 stocks data."""
        # Top 50 S&P 500 stocks by market cap
        sp500_top = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
            "XOM", "V", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
            "PEP", "KO", "COST", "AVGO", "MCD", "WMT", "TMO", "CSCO", "ACN", "ABT",
            "DHR", "VZ", "ADBE", "CRM", "NEE", "CMCSA", "NKE", "TXN", "PM", "BMY",
            "UPS", "RTX", "HON", "QCOM", "T", "ORCL", "MS", "LOW", "INTC", "UNP"
        ]
        
        all_data = []
        batch_size = 10  # Fetch in batches to avoid rate limits
        
        for i in range(0, len(sp500_top), batch_size):
            batch = sp500_top[i:i + batch_size]
            try:
                df = self.fetch_bars(batch, start_date, end_date, timeframe)
                all_data.append(df)
                time.sleep(0.5)  # Rate limit protection
            except Exception as e:
                logger.warning(f"Failed to fetch batch {i//batch_size}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise RuntimeError("Failed to fetch any S&P 500 data")
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Path,
        partition_by: str = "symbol"
    ) -> None:
        """Save DataFrame to parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False, partition_cols=[partition_by] if partition_by else None)
        logger.info(f"Saved {len(df)} rows to {output_path}")
    
    def load_from_parquet(self, input_path: Path) -> pd.DataFrame:
        """Load DataFrame from parquet file."""
        return pd.read_parquet(input_path)


# Example usage
if __name__ == "__main__":
    # Test the fetcher
    fetcher = AlpacaFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = fetcher.fetch_bars(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date=start_date,
        end_date=end_date,
        timeframe="1Day"
    )
    
    print(df.head())
    print(f"Fetched {len(df)} bars")
