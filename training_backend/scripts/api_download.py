#!/usr/bin/env python3
"""API wrapper for data download - called by Node.js server."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf

# Simple logging
def log_info(msg):
    print(msg, flush=True)

def log_error(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)


def download_stocks(symbols, start_date, end_date, output_dir):
    """Download stock data using yfinance."""
    output_dir = Path(output_dir) / 'raw' / 'stocks'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_info(f"Downloading {len(symbols)} stocks from {start_date} to {end_date}...")
    
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            log_info(f"Fetching {symbol} ({i+1}/{len(symbols)})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if len(df) > 0:
                df = df.reset_index()
                df['symbol'] = symbol
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
                log_info(f"✓ {symbol}: {len(df)} rows")
        except Exception as e:
            log_error(f"Failed to fetch {symbol}: {e}")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        filename = f"stocks_1Day_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)
        log_info(f"Saved {len(df)} rows to {output_path}")
        return df
    else:
        raise RuntimeError("Failed to fetch any stock data")


def download_crypto(symbols, start_date, end_date, output_dir):
    """Download crypto data using yfinance."""
    output_dir = Path(output_dir) / 'raw' / 'crypto'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_info(f"Downloading {len(symbols)} crypto pairs from {start_date} to {end_date}...")
    
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            log_info(f"Fetching {symbol} ({i+1}/{len(symbols)})...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if len(df) > 0:
                df = df.reset_index()
                df['symbol'] = symbol
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
                log_info(f"✓ {symbol}: {len(df)} rows")
        except Exception as e:
            log_error(f"Failed to fetch {symbol}: {e}")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        filename = f"crypto_1Day_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
        output_path = output_dir / filename
        df.to_parquet(output_path, index=False)
        log_info(f"Saved {len(df)} rows to {output_path}")
        return df
    else:
        raise RuntimeError("Failed to fetch any crypto data")


def main():
    parser = argparse.ArgumentParser(description='Download market data')
    parser.add_argument('--source', required=True, choices=['stocks', 'crypto', 'all'])
    parser.add_argument('--universe', required=True, choices=['small', 'medium', 'large'])
    parser.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--data-dir', default='/data', help='Data directory')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Define universe sizes
    universes = {
        'small': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM', 'V'],
        'medium': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM', 'V',
                   'WMT', 'UNH', 'JNJ', 'XOM', 'PG', 'HD', 'CVX', 'MA', 'ABBV', 'KO',
                   'BAC', 'PEP', 'COST', 'AVGO', 'MRK'],
        'large': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'JPM', 'V',
                  'WMT', 'UNH', 'JNJ', 'XOM', 'PG', 'HD', 'CVX', 'MA', 'ABBV', 'KO',
                  'BAC', 'PEP', 'COST', 'AVGO', 'MRK', 'CSCO', 'ACN', 'TMO', 'LLY', 'NKE',
                  'DIS', 'ADBE', 'CRM', 'MCD', 'NFLX', 'ABT', 'WFC', 'AMD', 'ORCL', 'TXN',
                  'DHR', 'PM', 'CMCSA', 'INTC', 'VZ', 'NEE', 'UPS', 'RTX', 'HON', 'QCOM']
    }
    
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD']
    
    symbols = universes[args.universe]
    results = []
    
    try:
        if args.source in ['stocks', 'all']:
            df = download_stocks(symbols, start_date, end_date, args.data_dir)
            results.append({
                'type': 'stocks',
                'symbols': len(symbols),
                'rows': len(df) if df is not None else 0
            })
        
        if args.source in ['crypto', 'all']:
            df = download_crypto(crypto_symbols, start_date, end_date, args.data_dir)
            results.append({
                'type': 'crypto',
                'symbols': len(crypto_symbols),
                'rows': len(df) if df is not None else 0
            })
        
        print(json.dumps({'success': True, 'results': results}))
        return 0
        
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
