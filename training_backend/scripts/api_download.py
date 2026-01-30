#!/usr/bin/env python3
"""API wrapper for data download - called by Node.js server."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.download_data import download_alpaca_data, download_binance_data
from data.tickers import UNIVERSE_SMALL, UNIVERSE_MEDIUM, UNIVERSE_LARGE


def main():
    parser = argparse.ArgumentParser(description='Download market data')
    parser.add_argument('--source', required=True, choices=['stocks', 'crypto', 'all'])
    parser.add_argument('--universe', required=True, choices=['small', 'medium', 'large'])
    parser.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Select universe
    universes = {
        'small': UNIVERSE_SMALL,
        'medium': UNIVERSE_MEDIUM,
        'large': UNIVERSE_LARGE
    }
    symbols = universes[args.universe]
    
    results = []
    
    try:
        if args.source in ['stocks', 'all']:
            print(f"Downloading {len(symbols)} stocks...")
            df = download_alpaca_data(symbols, start_date, end_date)
            results.append({
                'type': 'stocks',
                'symbols': len(symbols),
                'rows': len(df) if df is not None else 0
            })
        
        if args.source in ['crypto', 'all']:
            crypto_symbols = [s + 'USDT' for s in ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']]
            print(f"Downloading {len(crypto_symbols)} crypto pairs...")
            df = download_binance_data(crypto_symbols, start_date, end_date)
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
