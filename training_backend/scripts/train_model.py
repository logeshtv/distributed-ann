#!/usr/bin/env python3
"""Train the trading neural network."""

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
    
    # Load all parquet files
    all_data = []
    for parquet_file in data_path.glob("*.parquet"):
        df = pd.read_parquet(parquet_file)
        all_data.append(df)
    
    if not all_data:
        # Try loading from subdirectories
        for parquet_file in data_path.rglob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            all_data.append(df)
    
    if not all_data:
        raise RuntimeError(f"No parquet files found in {data_path}")
    
    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows from {len(all_data)} files")
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("Data must have 'timestamp' column")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Process each symbol separately
    logger.info("Adding features per symbol...")
    processed = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)
        
        if len(symbol_df) < 250:
            logger.warning(f"Skipping {symbol}: only {len(symbol_df)} rows")
            continue
        
        try:
            symbol_df = fe.add_all_features(symbol_df)
            processed.append(symbol_df)
        except Exception as e:
            logger.warning(f"Failed to process {symbol}: {e}")
            continue
    
    if not processed:
        raise RuntimeError("No symbols processed successfully")
    
    df = pd.concat(processed, ignore_index=True)
    logger.info(f"Processed {len(df)} rows for {len(processed)} symbols")
    
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
    exclude += [c for c in df.columns if c.startswith('target_')]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    logger.info(f"Using {len(feature_cols)} features, {len(df)} total rows")
    
    # Fill any remaining NaN/inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Time-based split
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Normalize features
    logger.info("Normalizing features...")
    norm_params = {}
    for col in feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if pd.notna(std) and std > 0:
            train_df.loc[:, col] = (train_df[col] - mean) / std
            val_df.loc[:, col] = (val_df[col] - mean) / std
            test_df.loc[:, col] = (test_df[col] - mean) / std
            norm_params[col] = (mean, std)
    
    # Create datasets
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
        num_features=60,
        sequence_length=args.seq_length,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    data_path = Path(args.data_path)
    train_dataset, val_dataset, test_dataset, num_features = prepare_data(data_path, config)
    config.num_features = num_features
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.batch_size
    )
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model = TradingNeuralNetwork(
        input_dim=config.num_features,
        xlstm_hidden=256,  # Reduced for faster training
        xlstm_layers=2,
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        use_position_state=False,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create trainer
    loss_fn = MultiTaskLoss()
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        loss_fn=loss_fn,
        learning_rate=config.learning_rate,
        device=device,
        checkpoint_dir=settings.MODELS_DIR
    )
    
    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        epochs=args.epochs,
        patience=args.patience,
        save_best=True
    )
    
    # Final evaluation
    test_metrics = trainer.validate()
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final model
    final_path = settings.MODELS_DIR / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    
    return history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train trading neural network")
    parser.add_argument("--data-path", type=str, default="data_storage/raw", help="Path to data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
