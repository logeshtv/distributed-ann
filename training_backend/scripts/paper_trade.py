#!/usr/bin/env python3
"""Run paper trading with live market data."""

import argparse
from pathlib import Path
from datetime import datetime
import time
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from models.trading_nn import TradingNeuralNetwork
from execution.paper_trader import PaperTrader
from data.fetchers import AlpacaFetcher, BinanceFetcher
from utils.logger import setup_logger

logger = setup_logger()


class SimpleTradingBot:
    """Simple trading bot for paper trading."""
    
    def __init__(
        self,
        paper_trader: PaperTrader,
        symbols: list,
        data_source: str = "alpaca"
    ):
        self.paper_trader = paper_trader
        self.symbols = symbols
        self.data_source = data_source
        self.running = False
        
        # Initialize data fetcher
        if data_source == "alpaca":
            self.fetcher = AlpacaFetcher(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY
            )
        else:
            self.fetcher = BinanceFetcher()
    
    def get_current_prices(self) -> dict:
        """Fetch current prices for all symbols."""
        prices = {}
        for symbol in self.symbols:
            try:
                if self.data_source == "binance":
                    price = self.fetcher.get_current_price(symbol)
                else:
                    # For Alpaca, would need to use streaming or latest bar
                    price = 100.0  # Placeholder
                prices[symbol] = price
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        return prices
    
    def generate_signal(self, symbol: str) -> int:
        """Generate trading signal (placeholder - replace with model)."""
        # This would use your trained model
        import random
        return random.choice([-1, 0, 0, 0, 1])  # Mostly hold
    
    def run(self, interval: int = 60):
        """Run the trading bot."""
        self.running = True
        self.paper_trader.start()
        
        logger.info(f"Starting paper trading with {len(self.symbols)} symbols")
        logger.info(f"Initial capital: ${self.paper_trader.position_tracker.initial_capital:,.2f}")
        
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")
                
                # Update prices
                prices = self.get_current_prices()
                self.paper_trader.update_prices(prices)
                
                # Generate signals and trade
                for symbol in self.symbols:
                    if symbol not in prices:
                        continue
                    
                    signal = self.generate_signal(symbol)
                    position = self.paper_trader.position_tracker.get_position(symbol)
                    
                    if signal == 1 and position is None:
                        # Buy
                        quantity = (self.paper_trader.position_tracker.cash * 0.1) / prices[symbol]
                        self.paper_trader.submit_order(symbol, 'buy', quantity)
                    
                    elif signal == -1 and position is not None:
                        # Sell
                        self.paper_trader.submit_order(symbol, 'sell', position.quantity)
                
                # Print status
                perf = self.paper_trader.get_performance()
                logger.info(f"Portfolio: ${perf['current_value']:,.2f} | Return: {perf['total_return']:.2%}")
                
                # Wait for next iteration
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(interval)
        
        self.stop()
    
    def stop(self):
        """Stop the bot."""
        self.running = False
        self.paper_trader.stop()
        
        # Print final summary
        perf = self.paper_trader.get_performance()
        print("\n" + "="*50)
        print("PAPER TRADING SUMMARY")
        print("="*50)
        print(f"Initial Capital:   ${perf['initial_capital']:>12,.2f}")
        print(f"Final Value:       ${perf['current_value']:>12,.2f}")
        print(f"Total Return:      {perf['total_return']:>12.2%}")
        print(f"Realized P&L:      ${perf['realized_pnl']:>12,.2f}")
        print(f"Unrealized P&L:    ${perf['unrealized_pnl']:>12,.2f}")
        print(f"Total Trades:      {perf['trades']:>12}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--source", choices=["alpaca", "binance"], default="binance")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--slippage", type=float, default=10, help="Slippage in basis points")
    args = parser.parse_args()
    
    # Default symbols
    if args.source == "alpaca":
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    else:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    
    # Create paper trader
    paper_trader = PaperTrader(
        initial_capital=args.capital,
        slippage_bps=args.slippage,
        data_source=args.source
    )
    
    # Create and run bot
    bot = SimpleTradingBot(paper_trader, symbols, args.source)
    bot.run(interval=args.interval)


if __name__ == "__main__":
    main()
