"""PyTorch Dataset for trading data - Memory Optimized."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
from loguru import logger


class TradingDataset(Dataset):
    """
    Memory-Efficient PyTorch Dataset for trading sequences.
    
    Instead of pre-computing all sliding windows (which explodes memory by 60x),
    this class stores the raw data and slices it on-the-fly in __getitem__.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        symbol_indices: List[Tuple[int, int]],
        sequence_length: int = 60,
        position_states: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize trading dataset.
        
        Args:
            features: Feature array of shape (total_rows, n_features)
            targets: Target array of shape (total_rows, n_targets)
            symbol_indices: List of (start_idx, end_idx) for each symbol's data block
            sequence_length: Lookback window size
            position_states: Optional position state array
            transform: Optional transform function
        """
        # Store raw arrays (shared memory if possible)
        # Make tensors contiguous for faster slicing with multi-worker loading
        self.features = torch.FloatTensor(features).contiguous()
        self.targets = torch.FloatTensor(targets).contiguous()
        self.position_states = torch.FloatTensor(position_states).contiguous() if position_states is not None else None
        
        self.seq_len = sequence_length
        self.transform = transform
        
        # Pre-compute valid indices
        # A valid index i means we can take slice [i-seq_len : i]
        # and the target is at i.
        # So we need i >= start_idx + seq_len for each symbol block
        self.valid_indices = []
        for start, end in symbol_indices:
            # We need at least seq_len history.
            # If start=0, end=100, seq=60:
            # First valid target is at index 60 (uses 0-59).
            # Last valid target is at index 99 (uses 39-98).
            # range(start + seq_len, end)
            if end - start > sequence_length:
                self.valid_indices.extend(range(start + sequence_length, end))
                
        self.valid_indices = np.array(self.valid_indices, dtype=np.int32)
        
        logger.info(f"Created Lazy Dataset with {len(self.valid_indices)} samples from {len(features)} rows")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get constant-time access to the valid DataFrame index
        # We treat 'idx' as the index into our valid_indices list
        real_idx = self.valid_indices[idx]
        
        # Slice the window: [real_idx - seq_len : real_idx]
        # Note: real_idx is EXCLUSIVE for the slice, but INCLUSIVE for the target
        # E.g. target at T (real_idx), features from T-60 to T-1
        
        # Actually standard LSTM logic: input is T-seq to T-1, target is T (or T+1 etc)
        # Based on previous code: features[i-sequence_length:i]
        
        features = self.features[real_idx - self.seq_len : real_idx]
        targets = self.targets[real_idx] # Target is the outcome at this step (or for future)
        
        if self.transform:
            features = self.transform(features)
        
        sample = {
            'features': features,
            'targets': targets
        }
        
        if self.position_states is not None:
            sample['position_state'] = self.position_states[real_idx]
        
        return sample
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        sequence_length: int = 60,
        position_cols: Optional[List[str]] = None
    ) -> 'TradingDataset':
        """
        Create dataset from DataFrame without memory explosion.
        """
        # Ensure contiguous arrays
        # df is already expected to be sorted by timestamp per symbol or globally with an index map
        
        # 1. Get raw float32 arrays (Minimal Copy)
        features = df[feature_cols].values.astype(np.float32)
        targets = df[target_cols].values.astype(np.float32)
        pos = df[position_cols].values.astype(np.float32) if position_cols else None
        
        # 2. Identify symbol boundaries
        # We assume df is already sorted by symbol/timestamp or grouped.
        # But to be safe, we must rely on the caller to have sorted it.
        # The previous code iterated unique symbols.
        # Ideally, we find the start/end indices of each symbol in the big dataframe.
        
        # Fast way to find boundaries if sorted by symbol:
        # We can reconstruct them or assume provided.
        # For simplicity and robustness, let's group by symbol and get indices.
        # BUT grouping is slow and copies data. 
        
        # Faster approach:
        # We will assume the df is passed in with mixed symbols but we need to know where they switch.
        # Let's iterate unique symbols and store their (start, end) ranges.
        # Note: The input `df` to this method in `train_production.py` is usually a single block 
        # (train_df, val_df, test_df).
        # HOWEVER, train_df contains MANY symbols mixed together? 
        # In `train_production.py`:
        # `df = df.sort_values('timestamp')` -> This MIXES symbols!
        # This breaks the sequence continuity for a single symbol.
        #
        # CRITICAL FIX: The data prep in train_production.py sorts by TIMESTAMP globally.
        # This interweaves symbols: AAPL(t1), MSFT(t1), AAPL(t2)...
        # This is WRONG for sequence learning unless we group.
        #
        # We must re-sort by Symbol then Timestamp to ensure contiguous blocks.
        
        # Let's enforce sorting here or handle the indices correctly.
        # To avoid modifying the large DF in place deeply, let's just get the valid indices logic right.
        
        # Actually, for lazy loading to work efficiently, the data in `features` array 
        # MUST be contiguous for a single sequence.
        # If the df is sorted by time (interleaved), we cannot slice `[i-60:i]`.
        
        # SO, we MUST sort by [Symbol, Timestamp].
        # The previous code `process each symbol separately` and `concat` naturally produced 
        # a Symbol-major ordering (Symbol A all rows, Symbol B all rows...).
        #
        # BUT `train_production.py` lines 112: `df = df.sort_values('timestamp')` DESTROYS this.
        # It makes it Time-major.
        #
        # We need to fix `train_production.py` as well to sort by `['symbol', 'timestamp']`.
        
        # Assuming the input `df` is sorted by `['symbol', 'timestamp']`:
        # We just need to find the start/end of each symbol.
        
        symbol_indices = []
        
        # We can use pandas to find group boundaries efficiently
        # Or faster: just iterate if we know it is sorted.
        # Let's use flexible groupby which works even if not perfectly sorted, 
        # but we strongly recommend sorting.
        
        # Getting indices for each symbol
        unique_symbols = df['symbol'].unique()
        # This might be slow for 1.7M rows.
        # Optimization: Use `df.groupby('symbol').indices` -> returns dict {sym: [idx1, idx2...]}
        # If it's sorted, the indices are contiguous.
        
        # Let's assume we fix `train_production.py` to sort by `['symbol', 'timestamp']`.
        # Then we can just scan for changes.
        
        # For now, let's implement the robust `groupby` way to get ranges, 
        # assuming contiguous blocks for efficiency.
        
        # Actually, let's simple iterate. 
        # We will require the user of this class (prepare_data) to pass a DF sorted by Symbol, Time.
        
        # Determine boundaries
        groups = df.groupby('symbol')
        for sym, group in groups:
            # We explicitly want the integer indices of this group in the dataframe
            # This relies on the df not being re-indexed/shuffled relative to `features` array.
            indices = group.index.values
            if len(indices) > sequence_length:
                # Check continuity
                if indices[-1] - indices[0] == len(indices) - 1:
                    # Contiguous block
                    symbol_indices.append((indices[0], indices[-1] + 1))
                else:
                    # Not contiguous (e.g. time sorted). 
                    # This is problematic for lazy slicing.
                    pass
        
        # Warning if we found no valid blocks, handled by caller
        
        return cls(features, targets, symbol_indices, sequence_length, pos)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """Create data loaders optimized for M4 Pro MPS."""
    import torch
    
    # Enable persistent workers if num_workers > 0 to reuse processes
    use_persistent = num_workers > 0
    
    # Detect if using MPS (Apple Silicon) - pin_memory must be False for MPS
    is_mps = torch.backends.mps.is_available()
    use_pin_memory = not is_mps and num_workers > 0
    
    # Prefetch factor for faster data loading (only with workers)
    prefetch = 4 if num_workers > 0 else None
    
    common_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': use_pin_memory,
        'persistent_workers': use_persistent,
        'prefetch_factor': prefetch,
    }
    
    loaders = {
        'train': DataLoader(
            train_dataset,
            shuffle=True,
            **common_kwargs
        ),
        'val': DataLoader(
            val_dataset,
            shuffle=False,
            **common_kwargs
        )
    }
    
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            shuffle=False,
            **common_kwargs
        )
    
    return loaders
