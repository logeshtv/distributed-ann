#!/usr/bin/env python3
"""Run backtesting on historical data."""

import argparse
from pathlib import Path
from datetime import datetime
import sys
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from data.features import FeatureEngineer
from models.trading_nn import TradingNeuralNetwork
from execution.backtester import Backtester
from execution.risk_manager import RiskManager, RiskLimits
from utils.metrics import TradingMetrics
from utils.logger import setup_logger

logger = setup_logger()


def load_model(model_path: Path, input_dim: int, device: str):
    """Load trained model."""
    model = TradingNeuralNetwork(
        input_dim=input_dim,
        xlstm_hidden=256,
        xlstm_layers=2,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        use_position_state=False,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to data per symbol to match training."""
    fe = FeatureEngineer()
    
    processed = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
        symbol_df = fe.add_all_features(symbol_df)
        processed.append(symbol_df)
    
    return pd.concat(processed, ignore_index=True)


def create_signal_function(model, feature_cols, seq_length, device):
    """Create signal function for backtester."""
    
    def get_signal(data: pd.DataFrame, idx: int) -> int:
        if idx < seq_length:
            return 0  # Hold
        
        # Get sequence
        seq_data = data.iloc[idx-seq_length:idx][feature_cols].values
        seq_tensor = torch.FloatTensor(seq_data.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(seq_tensor)
            direction_probs = torch.softmax(outputs['direction_logits'], dim=-1)
            direction = direction_probs.argmax(dim=-1).item()
        
        # 0=down, 1=neutral, 2=up
        if direction == 2:
            return 1  # Buy
        elif direction == 0:
            return -1  # Sell
        else:
            return 0  # Hold
    
    return get_signal


def run_backtest(args):
    """Run backtest."""
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    
    # Add features
    logger.info("Generating features...")
    df = prepare_features(df)
    
    # Filter by date
    if args.start:
        start_date = pd.to_datetime(args.start)
        if df['timestamp'].dt.tz is not None and start_date.tz is None:
            start_date = start_date.tz_localize(df['timestamp'].dt.tz)
        df = df[df['timestamp'] >= start_date]
        
    if args.end:
        end_date = pd.to_datetime(args.end)
        if df['timestamp'].dt.tz is not None and end_date.tz is None:
            end_date = end_date.tz_localize(df['timestamp'].dt.tz)
        df = df[df['timestamp'] <= end_date]
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    logger.info(f"Backtesting on {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Get feature columns
    exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
    exclude += [c for c in df.columns if c.startswith('target_')]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Ensure only numeric columns
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    # Normalize features
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(Path(args.model_path), len(feature_cols), device)
        signal_fn = create_signal_function(model, feature_cols, args.seq_length, device)
    else:
        # Simple moving average crossover strategy for testing
        logger.info("Using simple SMA crossover strategy (no model)")
        def signal_fn(data, idx):
            if idx < 50:
                return 0
            sma_10 = data['close'].iloc[idx-10:idx].mean()
            sma_50 = data['close'].iloc[idx-50:idx].mean()
            if sma_10 > sma_50 * 1.01:
                return 1
            elif sma_10 < sma_50 * 0.99:
                return -1
            return 0
    
    # Run backtest
    logger.info("Running backtest...")
    risk_limits = RiskLimits(
        daily_loss_limit=-0.03,
        weekly_loss_limit=-0.10,
        max_drawdown=-0.25
    )
    
    backtester = Backtester(
        initial_capital=args.capital,
        slippage_bps=args.slippage,
        risk_manager=RiskManager(risk_limits)
    )
    
    result = backtester.run(df, signal_fn, position_size=args.position_size)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return:      {result.total_return:>10.2%}")
    print(f"Annual Return:     {result.annual_return:>10.2%}")
    print(f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"Win Rate:          {result.win_rate:>10.2%}")
    print(f"Profit Factor:     {result.profit_factor:>10.2f}")
    print(f"Total Trades:      {result.total_trades:>10}")
    print("="*60)
    
    # Save results
    output_dir = Path(args.output or "backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result.equity_curve.to_csv(output_dir / f"equity_{timestamp}.csv", index=False)
    pd.DataFrame(result.trades).to_csv(output_dir / f"trades_{timestamp}.csv", index=False)
    
    logger.info(f"Results saved to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run backtest on historical data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data parquet file")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--position-size", type=float, default=0.1, help="Position size (0-1)")
    parser.add_argument("--slippage", type=float, default=10, help="Slippage in basis points")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length for model")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    run_backtest(args)


if __name__ == "__main__":
    main()
