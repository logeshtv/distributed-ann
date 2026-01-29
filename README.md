# Aggressive ML Trading System

> **Target**: 100%+ annual returns with xLSTM-Transformer hybrid architecture  
> **Markets**: US Equities, Cryptocurrency  
> **Risk Level**: Extreme

⚠️ **WARNING**: This system is for educational purposes. Trading involves significant risk of loss. Only use capital you can afford to lose.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### 2. Download Historical Data

```bash
# Download 5 years of data
python scripts/download_data.py --source all --start 2019-01-01

# Download specific sources
python scripts/download_data.py --source alpaca --timeframe 1Day
python scripts/download_data.py --source binance --interval 1d
```

### 3. Train the Model

```bash
python scripts/train_model.py \
    --data-path data_storage/raw \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001
```

### 4. Run Backtest

```bash
python scripts/backtest.py \
    --data-path data_storage/raw/stocks_1Day_20190101_20240101.parquet \
    --model-path data_storage/models/best_model.pt \
    --start 2023-01-01 \
    --capital 100000
```

### 5. Paper Trading

```bash
# Crypto paper trading
python scripts/paper_trade.py --source binance --capital 100000

# US stocks paper trading (requires Alpaca API)
python scripts/paper_trade.py --source alpaca --capital 100000
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│              (60 features, 60 timesteps)                 │
└─────────────────────────┬────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────┐
│    xLSTM ENCODER    │       │   TRANSFORMER ENCODER   │
│  (512 hidden, 2L)   │       │   (256 dim, 8 heads)    │
│  - Exponential gates│       │   - 3 encoder layers    │
│  - Scalar memory    │       │   - Positional encoding │
└─────────┬───────────┘       └───────────┬─────────────┘
          │                               │
          └───────────┬───────────────────┘
                      ▼
          ┌─────────────────────┐
          │    FUSION LAYER     │
          │  (Attention-based)  │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │  TEMPORAL FUSION    │
          │  (Sequence → Vector)│
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │  MULTI-TASK HEADS   │
          ├─────────────────────┤
          │ • Price Prediction  │
          │ • Direction (3-way) │
          │ • Position Signal   │
          │ • Volatility        │
          │ • Risk Signal       │
          └─────────────────────┘
```

## Project Structure

```
trading_ml_system/
├── config/                 # Configuration
│   ├── settings.py        # Global settings
│   ├── model_config.py    # Model hyperparameters
│   └── trading_config.py  # Trading parameters
│
├── data/                   # Data handling
│   ├── fetchers/          # API connectors
│   │   ├── alpaca_fetcher.py
│   │   └── binance_fetcher.py
│   ├── features.py        # 50+ technical indicators
│   ├── pipeline.py        # Data processing
│   ├── dataset.py         # PyTorch datasets
│   └── storage.py         # Database operations
│
├── models/                 # Neural networks
│   ├── xlstm.py           # xLSTM implementation
│   ├── transformer.py     # Transformer encoder
│   ├── fusion.py          # Fusion layer
│   ├── position_encoder.py # Position state encoder
│   ├── heads.py           # Multi-task output heads
│   └── trading_nn.py      # Main model
│
├── training/               # Training infrastructure
│   ├── losses.py          # Multi-task losses
│   ├── trainer.py         # Training loop
│   └── validation.py      # Walk-forward validation
│
├── execution/              # Trading execution
│   ├── backtester.py      # Historical backtesting
│   ├── position_tracker.py # Position management
│   ├── risk_manager.py    # Risk limits
│   └── paper_trader.py    # Live simulation
│
├── scripts/                # Runnable scripts
│   ├── download_data.py   # Data collection
│   ├── train_model.py     # Model training
│   ├── backtest.py        # Backtesting
│   └── paper_trade.py     # Paper trading
│
└── utils/                  # Utilities
    ├── metrics.py         # Performance metrics
    └── logger.py          # Logging
```

## Features

### Technical Indicators (50+)
- **Trend**: SMA, EMA, MACD, ADX, Parabolic SAR, Aroon
- **Momentum**: RSI, Stochastic, Williams %R, CCI, ROC, MFI
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, VWAP, CMF, Force Index

### Risk Management
- Daily/weekly/monthly loss limits
- Position size limits (Kelly Criterion)
- Maximum drawdown protection
- Automatic stop-loss/take-profit

### Performance Metrics
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown
- Win Rate, Profit Factor
- Calmar Ratio

## Data Format

The system expects OHLCV data in Parquet format:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | UTC timestamp |
| symbol | string | Ticker symbol |
| open | float | Opening price |
| high | float | High price |
| low | float | Low price |
| close | float | Closing price |
| volume | float | Trading volume |
| vwap | float | VWAP (optional) |
| trade_count | int | Trade count (optional) |

## Expected Performance

| Metric | Conservative | Moderate | Aggressive |
|--------|-------------|----------|------------|
| Annual Return | 25-35% | 50-70% | 100%+ |
| Sharpe Ratio | 1.0-1.5 | 1.5-2.0 | 2.0+ |
| Max Drawdown | -10% to -15% | -15% to -20% | -20% to -30% |
| Win Rate | 52-55% | 55-58% | 55-60% |

## License

MIT License - Use at your own risk.

## Disclaimer

This software is for educational purposes only. Trading financial instruments involves significant risk of loss. Past performance does not guarantee future results. The developers are not responsible for any financial losses incurred from using this system.
