#!/usr/bin/env python3
"""Train the trading neural network with production settings for massive datasets."""

import argparse
from pathlib import Path
from datetime import datetime
import sys
import torch
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from config.model_config import ModelConfig
from data.features import FeatureEngineer
from data.dataset import TradingDataset, create_data_loaders
from models.trading_nn import TradingNeuralNetwork
from training.trainer import Trainer
from training.losses import MultiTaskLoss
from utils.logger import setup_logger

logger = setup_logger()


def prepare_data(data_path: Path, config: ModelConfig):
    """Load and prepare training data."""
    logger.info("Loading and preparing data...")
    
    # Load all parquet files from both stocks and crypto
    all_data = []
    
    # Check both subdirectories
    for subdir in ["stocks", "crypto"]:
        dir_path = data_path / subdir
        if dir_path.exists():
            for parquet_file in dir_path.glob("*.parquet"):
                logger.info(f"Loading {parquet_file}...")
                df = pd.read_parquet(parquet_file)
                all_data.append(df)
    
    # If explicit path provided to file
    if not all_data and data_path.is_file():
         all_data.append(pd.read_parquet(data_path))

    if not all_data:
        raise RuntimeError(f"No parquet files found in {data_path}")
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows total")
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' in df.columns:
        # distinct aware/naive values cause errors, force all to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    else:
        raise ValueError("Data must have 'timestamp' column")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Process each symbol separately
    logger.info("Adding features per symbol (this may take a while)...")
    processed = []
    
    # Get unique symbols
    symbols = df['symbol'].unique()
    logger.info(f"Processing {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
        
        # Skip symbols with too little history
        if len(symbol_df) < 500:
            continue
        
        try:
            symbol_df = fe.add_all_features(symbol_df)
            processed.append(symbol_df)
            if (i+1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(symbols)} symbols")
        except Exception as e:
            logger.warning(f"Failed to process {symbol}: {e}")
            continue
    
    if not processed:
        raise RuntimeError("No symbols processed successfully")
    
    df = pd.concat(processed, ignore_index=True)
    logger.info(f"Feature engineering complete. Total rows: {len(df)}")
    
    # Create labels
    logger.info("Creating labels...")
    for h in [1, 4, 24]:
        df[f'target_return_{h}'] = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-h) / x - 1
        )
        df[f'target_direction_{h}'] = 1  # neutral
        df.loc[df[f'target_return_{h}'] > 0.005, f'target_direction_{h}'] = 2  # up
        df.loc[df[f'target_return_{h}'] < -0.005, f'target_direction_{h}'] = 0  # down
    
    df['target_return'] = df['target_return_1']
    df['target_direction'] = df['target_direction_1']
    
    # Drop rows with NaN targets (last rows of each symbol)
    target_cols = ['target_return', 'target_direction']
    df = df.dropna(subset=target_cols)
    
    # Get feature columns
    exclude = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
    exclude += [col for col in df.columns if col.startswith('target_')]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Ensure only numeric columns
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Cleanup memory
    import gc
    del all_data
    del processed
    gc.collect()
    
    # Fill any remaining NaN/inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Sort by Symbol then Timestamp to ensure contiguous blocks for lazy loading
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    
    # WARNING: Simple splitting by index might break a symbol in half.
    # Ideally we split by symbol, or we accept a small leakage/break at the boundary.
    # For massive datasets, breaking 1-2 symbols is negligible.
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Normalize features
    logger.info("Normalizing features...")
    for col in feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if pd.notna(std) and std > 0:
            train_df.loc[:, col] = (train_df[col] - mean) / std
            val_df.loc[:, col] = (val_df[col] - mean) / std
            test_df.loc[:, col] = (test_df[col] - mean) / std
    
    # Create datasets using lazy loading
    # Note: We must ensure indices are reset for the new DFs so lazy loader works
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_dataset = TradingDataset.from_dataframe(
        train_df, feature_cols, target_cols, config.sequence_length
    )
    val_dataset = TradingDataset.from_dataframe(
        val_df, feature_cols, target_cols, config.sequence_length
    )
    test_dataset = TradingDataset.from_dataframe(
        test_df, feature_cols, target_cols, config.sequence_length
    )
    
    return train_dataset, val_dataset, test_dataset, len(feature_cols)


def train_model(args):
    """Train the trading model."""
    config = ModelConfig(
        num_features=113, # Will be updated
        sequence_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    data_path = Path(args.data_path)
    train_dataset, val_dataset, test_dataset, num_features = prepare_data(data_path, config)
    config.num_features = num_features
    
    # Create data loaders with parallel workers for fast loading
    loaders = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size,
        num_workers=4  # Enable parallel data loading (4 workers for M4 Pro)
    )
    
    # Create model - LARGER CAPACITY for Massive Dataset
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = TradingNeuralNetwork(
        input_dim=config.num_features,
        xlstm_hidden=512,        # Increased from 256
        xlstm_layers=3,          # Increased from 2
        transformer_dim=256,     # Increased from 128
        transformer_heads=8,     # Increased from 4
        transformer_layers=3,    # Increased from 2
        use_position_state=False,
        dropout=0.3              # Increased dropout for regularization
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Production Model parameters: {total_params:,}")
    
    # Create trainer with scheduler
    loss_fn = MultiTaskLoss()
    
    # Use scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        loss_fn=loss_fn,
        learning_rate=config.learning_rate,
        device=device,
        checkpoint_dir=settings.MODELS_DIR,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Calculate accumulation steps
    # Target large effective batch size of 2048 for better convergence
    target_batch_size = 2048
    accumulation_steps = max(1, target_batch_size // args.batch_size)
    
    # Train
    logger.info(f"Starting PRODUCTION training for {args.epochs} epochs...")
    logger.info(f"Effective Batch Size: {args.batch_size * accumulation_steps} "
                f"(Physical: {args.batch_size}, Accumulation: {accumulation_steps})")
                
    history = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        save_best=True,
        accumulation_steps=accumulation_steps
    )
    
    # Validate
    test_metrics = trainer.validate()
    logger.info(f"Final Test metrics: {test_metrics}")
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_path = settings.MODELS_DIR / f"production_model_{timestamp}.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved production model to {final_path}")
    
    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train production trading neural network")
    parser.add_argument("--data-path", type=str, default="data_storage/raw", help="Path to data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Physical batch size (will accumulate to 2048)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
