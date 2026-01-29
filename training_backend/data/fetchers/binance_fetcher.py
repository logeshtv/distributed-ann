"""Binance API data fetcher for cryptocurrency markets."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import asyncio
from loguru import logger

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logger.warning("Binance SDK not installed. Run: pip install python-binance")


class BinanceFetcher:
    """Fetch historical cryptocurrency data from Binance API."""
    
    # Binance interval mapping
    INTERVALS = {
        '1m': Client.KLINE_INTERVAL_1MINUTE if BINANCE_AVAILABLE else '1m',
        '3m': Client.KLINE_INTERVAL_3MINUTE if BINANCE_AVAILABLE else '3m',
        '5m': Client.KLINE_INTERVAL_5MINUTE if BINANCE_AVAILABLE else '5m',
        '15m': Client.KLINE_INTERVAL_15MINUTE if BINANCE_AVAILABLE else '15m',
        '30m': Client.KLINE_INTERVAL_30MINUTE if BINANCE_AVAILABLE else '30m',
        '1h': Client.KLINE_INTERVAL_1HOUR if BINANCE_AVAILABLE else '1h',
        '4h': Client.KLINE_INTERVAL_4HOUR if BINANCE_AVAILABLE else '4h',
        '1d': Client.KLINE_INTERVAL_1DAY if BINANCE_AVAILABLE else '1d',
        '1w': Client.KLINE_INTERVAL_1WEEK if BINANCE_AVAILABLE else '1w',
    }
    
    # Top cryptocurrencies by market cap
    TOP_CRYPTO = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ETCUSDT",
        "XLMUSDT", "TRXUSDT", "NEARUSDT", "APTUSDT", "OPUSDT"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize Binance data fetcher.
        
        Args:
            api_key: Binance API key (optional for historical data)
            secret_key: Binance secret key (optional for historical data)
        """
        self.api_key = api_key or ""
        self.secret_key = secret_key or ""
        self._client = None
        
        if not BINANCE_AVAILABLE:
            raise ImportError(
                "Binance SDK not installed. Install with: pip install python-binance"
            )
    
    @property
    def client(self) -> 'Client':
        """Lazy initialization of Binance client."""
        if self._client is None:
            self._client = Client(self.api_key, self.secret_key)
        return self._client
    
    def fetch_klines(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines for a single symbol.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            start_date: Start date for data
            end_date: End date for data
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Use one of {list(self.INTERVALS.keys())}")
        
        start_str = start_date.strftime("%d %b, %Y")
        end_str = end_date.strftime("%d %b, %Y")
        
        logger.info(f"Fetching {symbol} from {start_str} to {end_str} ({interval})")
        
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=self.INTERVALS[interval],
                start_str=start_str,
                end_str=end_str
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trade_count',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trade_count'] = pd.to_numeric(df['trade_count'], errors='coerce').astype(int)
            df['symbol'] = symbol
            
            # Calculate VWAP
            df['vwap'] = df['quote_volume'] / df['volume']
            df['vwap'] = df['vwap'].replace([np.inf, -np.inf], np.nan).fillna(df['close'])
            
            # Keep only needed columns
            df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']]
            
            logger.info(f"Fetched {len(df)} klines for {symbol}")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        delay: float = 0.5
    ) -> pd.DataFrame:
        """
        Fetch klines for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            start_date: Start date for data
            end_date: End date for data
            interval: Kline interval
            delay: Delay between requests (rate limiting)
            
        Returns:
            Combined DataFrame with all symbols
        """
        all_data = []
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.fetch_klines(symbol, start_date, end_date, interval)
                all_data.append(df)
                
                if i < len(symbols) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            raise RuntimeError("Failed to fetch any crypto data")
    
    def fetch_top_crypto(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        n_symbols: int = 20
    ) -> pd.DataFrame:
        """Fetch top cryptocurrencies by market cap."""
        symbols = self.TOP_CRYPTO[:n_symbols]
        return self.fetch_multiple_symbols(symbols, start_date, end_date, interval)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def get_all_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        tickers = self.client.get_all_tickers()
        return {t['symbol']: float(t['price']) for t in tickers}
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_path: Path,
        partition_by: str = "symbol"
    ) -> None:
        """Save DataFrame to parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
    
    def load_from_parquet(self, input_path: Path) -> pd.DataFrame:
        """Load DataFrame from parquet file."""
        return pd.read_parquet(input_path)


# Example usage
if __name__ == "__main__":
    # Test the fetcher
    fetcher = BinanceFetcher()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = fetcher.fetch_klines(
        symbol="BTCUSDT",
        start_date=start_date,
        end_date=end_date,
        interval="1d"
    )
    
    print(df.head())
    print(f"Fetched {len(df)} klines")
