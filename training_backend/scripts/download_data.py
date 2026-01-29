#!/usr/bin/env python3
"""Download historical data from Alpaca and Binance."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetchers import AlpacaFetcher, BinanceFetcher
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger()


def download_alpaca_data(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "1Day",
    output_dir: Path = None
):
    """Download US stock data from Alpaca or fallback to yfinance."""
    output_dir = output_dir or settings.RAW_DATA_DIR / "stocks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(symbols)} symbols...")
    
    # Try to use Alpaca first if credentials available
    if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
        try:
            fetcher = AlpacaFetcher(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY
            )
            df = fetcher.fetch_bars(symbols, start_date, end_date, timeframe)
            output_path = output_dir / f"stocks_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
            fetcher.save_to_parquet(df, output_path)
            logger.info(f"Saved {len(df)} rows to {output_path}")
            return df
        except Exception as e:
            logger.warning(f"Alpaca failed: {e}. Falling back to yfinance...")
    
    # Fallback to yfinance (free, no API key needed)
    import yfinance as yf
    
    all_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Adjust start date for intraday yfinance limits
            fetch_start = start_date
            if timeframe in ["1m", "2m", "5m", "15m", "30m", "90m"]:
                limit_date = datetime.now() - timedelta(days=59)
                if start_date < limit_date:
                    logger.warning(f"yfinance 60-day limit for {timeframe}. Adjusting start date to {limit_date.date()}")
                    fetch_start = limit_date
            elif timeframe in ["1h", "60m"]:
                limit_date = datetime.now() - timedelta(days=729)
                if start_date < limit_date:
                    logger.warning(f"yfinance 730-day limit for {timeframe}. Adjusting start date to {limit_date.date()}")
                    fetch_start = limit_date

            df = ticker.history(start=fetch_start, end=end_date, interval=timeframe)
            if len(df) > 0:
                df = df.reset_index()
                df['symbol'] = symbol
                # Map columns (yfinance returns different casing)
                df = df.rename(columns={
                    'Date': 'timestamp',
                    'Datetime': 'timestamp',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                all_data.append(df)
                logger.info(f"Fetched {len(df)} rows for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        output_path = output_dir / f"stocks_1Day_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return df
    else:
        raise RuntimeError("Failed to fetch any stock data")


def download_binance_data(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    output_dir: Path = None
):
    """Download crypto data from Binance."""
    output_dir = output_dir or settings.RAW_DATA_DIR / "crypto"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {len(symbols)} symbols from Binance...")
    
    try:
        fetcher = BinanceFetcher(
            api_key=settings.BINANCE_API_KEY,
            secret_key=settings.BINANCE_SECRET_KEY
        )
        
        df = fetcher.fetch_multiple_symbols(symbols, start_date, end_date, interval)
        
        # Save to parquet
        output_path = output_dir / f"crypto_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        fetcher.save_to_parquet(df, output_path)
        
        logger.info(f"Saved {len(df)} rows to {output_path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to download Binance data: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download historical trading data")
    parser.add_argument("--source", choices=["alpaca", "binance", "all"], default="all")
    parser.add_argument("--start", type=str, default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Timeframe for Alpaca")
    parser.add_argument("--interval", type=str, default="1d", help="Interval for Binance")
    parser.add_argument("--universe", choices=["small", "medium", "large"], default="medium", help="Size of asset universe")
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    
    # Load symbols based on universe size
    from data.tickers import US_STOCKS, CRYPTO_PAIRS
    
    if args.universe == "small":
        stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"]
        cryptos = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    elif args.universe == "medium":
         stocks = US_STOCKS[:50]
         cryptos = CRYPTO_PAIRS[:10]
    else: # large/full
        stocks = US_STOCKS
        cryptos = CRYPTO_PAIRS
    
    if args.source in ["alpaca", "all"]:
        download_alpaca_data(stocks, start_date, end_date, args.timeframe)
    
    if args.source in ["binance", "all"]:
        download_binance_data(cryptos, start_date, end_date, args.interval)
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    main()
