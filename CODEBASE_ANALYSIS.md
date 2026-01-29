# Trading ML System - Complete Codebase Analysis

## 📋 Overview
This is an **aggressive ML trading system** targeting 100%+ annual returns using a hybrid **xLSTM-Transformer** architecture trained on US equities and cryptocurrency data.

---

## 🔄 DATA DOWNLOAD FLOW

### Entry Point: `scripts/download_data.py`

#### 1. **Data Sources**
- **US Stocks**: Alpaca API (primary) → fallback to yfinance (free)
- **Crypto**: Binance API only
- **Historical Range**: 2000-present (customizable)

#### 2. **Download Process**
```
download_data.py
├── parse arguments (source, start_date, end_date, universe_size)
├── load symbol lists from data/tickers.py
│   └── Small: 7 stocks + 3 crypto
│   └── Medium: 50 stocks + 10 crypto  
│   └── Large: 300+ stocks + 50+ crypto
├── Call download_alpaca_data() or download_binance_data()
└── Save to parquet files
    ├── data_storage/raw/stocks/
    ├── data_storage/raw/crypto/
```

#### 3. **Data Format** (after download)
- OHLCV data (Open, High, Low, Close, Volume)
- Columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `vwap`
- Stored as `.parquet` files (efficient columnar format)

#### 4. **Command Examples**
```bash
# Download all data (small universe, last 5 years)
python scripts/download_data.py --source all --start 2019-01-01

# Download only stocks
python scripts/download_data.py --source alpaca --timeframe 1Day

# Download crypto only
python scripts/download_data.py --source binance --interval 1d
```

---

## 🧠 MODEL TRAINING FLOW

### Entry Point: `scripts/train_model.py`

#### 1. **Data Preparation Pipeline** (`data/pipeline.py` + `data/dataset.py`)

```
Raw Parquet Data
    ↓
DataPipeline.load_and_process()
    ├── Load all symbols from data_storage/raw/
    ├── Filter by date range
    └── Sort by symbol + timestamp
    ↓
FeatureEngineer.add_all_features()
    ├── Return features (5, 10, 20 period returns)
    ├── Trend indicators (SMA, EMA, MACD, ADX, Aroon)
    ├── Momentum (RSI, Stochastic, Williams%R, ROC)
    ├── Volatility (ATR, Bollinger Bands, Keltner)
    ├── Volume (OBV, VWAP, CMF, Force Index)
    └── Temporal (day of week, month - cyclical encoding)
    ↓
Total Features Generated: 60 features per timestamp
    ↓
Label Creation
    ├── target_return_1d (1-day % return)
    ├── target_return_4d (4-day % return)
    ├── target_return_24d (24-day % return)
    ├── target_direction (0=down, 1=neutral, 2=up)
    └── Threshold: ±0.5% for classification
    ↓
Train/Val/Test Split
    ├── 70% training
    ├── 15% validation
    └── 15% testing
```

#### 2. **TradingDataset** (`data/dataset.py`)
- Creates sequences: **60 timesteps × 60 features = (batch_size, 60, 60)**
- Sliding window approach with no lookahead bias
- Returns: `{'features': Tensor, 'targets': Tensor}`

#### 3. **Model Architecture** (`models/trading_nn.py`)

```
INPUT LAYER (60 features, 60 timesteps)
    ↓
┌─────────────────────────────────────────────┐
│         DUAL-PATH ENCODING                  │
├─────────────────────────────────────────────┤
│
├─→ PATH A: xLSTM ENCODER
│   ├── 3 layers
│   ├── 512 hidden units
│   ├── Exponential gating (prevents gradient explosion)
│   └── Output: 512 dimensions
│
├─→ PATH B: TRANSFORMER ENCODER
│   ├── 3 layers
│   ├── 8 attention heads
│   ├── 256 embedding dimension
│   ├── Self-attention (pattern recognition)
│   └── Output: 256 dimensions
│
└─────────────────────────────────────────────┘
    ↓
FUSION LAYER
├── Concatenate xLSTM (512) + Transformer (256)
├── Attention mechanism to weight contributions
└── Output: 256 dimensions
    ↓
POSITION STATE ENCODER (optional)
├── Embeds current portfolio state
├── Embedding dimension: 320
├── Integrates with market features
└── Output: 256 dimensions (after integration)
    ↓
TEMPORAL FUSION
├── Aggregates sequence to vector
└── Output: 256 dimensions
    ↓
MULTI-TASK OUTPUT HEADS
├── 1. Price Prediction (3 outputs: 1d, 4d, 24d returns)
├── 2. Direction Classification (3 classes: up/neutral/down)
├── 3. Position Sizing (3 classes: buy/hold/sell)
├── 4. Volatility Forecast (regression)
├── 5. Confidence Score (0-1)
└── 6. Risk Signal (0-1)
```

**Model Parameters**: ~27.1 million

#### 4. **Training Loop** (`training/trainer.py`)

```
Trainer() class
├── Optimizer: AdamW (learning_rate=5e-4, weight_decay=1e-5)
├── Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
├── Loss Function: MultiTaskLoss (weighted)
│   ├── Price loss: 0.30 weight
│   ├── Direction loss: 0.20 weight
│   ├── Volatility loss: 0.15 weight
│   ├── Position loss: 0.20 weight
│   ├── Risk loss: 0.10 weight
│   └── Confidence loss: 0.05 weight
│
├── Gradient Accumulation: supports for large batch sizes
├── Gradient Clipping: max_norm=1.0
├── Early Stopping: patience=15 epochs
└── Checkpointing: best_model.pt saved when val_loss improves
```

#### 5. **Training Command**
```bash
python scripts/train_model.py \
    --data-path data_storage/raw \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --device cuda
```

#### 6. **Output**
- Trained model saved to: `data_storage/models/best_model.pt`
- Training logs to: `logs/`
- Validation metrics tracked per epoch

---

## 🎯 EXECUTION & BACKTESTING

### Backtesting: `scripts/backtest.py` → `execution/backtester.py`

```
Historical Data (OHLCV)
    ↓
Backtester.run()
    ├── Load trained model
    ├── For each timestamp:
    │   ├── Get latest 60 bars
    │   ├── Extract features
    │   ├── Pass through model → get predictions
    │   ├── Generate trading signal (direction output)
    │   ├── Check risk limits (RiskManager)
    │   ├── Execute position (PositionTracker)
    │   ├── Apply slippage (10 bps default)
    │   ├── Apply commission (0.1% default)
    │   └── Record equity curve
    │
    └── Generate BacktestResult
        ├── Total return
        ├── Sharpe ratio
        ├── Max drawdown
        ├── Win rate
        ├── Profit factor
        └── Trades list
```

### Risk Management: `execution/risk_manager.py`
- Max position size: configurable
- Max daily loss: stop-loss trigger
- Correlation checks: prevent over-concentration
- Leverage limits: drawdown protection

### Paper Trading: `scripts/paper_trade.py`
- Real-time trading without capital risk
- Connects to Alpaca/Binance APIs
- Records signals for later analysis

---

## 🌐 FRONTEND (CURRENT)

### Web Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **WebSocket**: Real-time training updates
- **Port**: 8000

### Files
- `web/app.py` - FastAPI endpoints
- `web/static/index.html` - UI (HTML)
- `web/static/app.js` - UI logic (JavaScript)
- `web/static/styles.css` - Styling
- `web/training_service.py` - Background training tasks

### Current Frontend Features
1. **Data Download Manager**
   - Source selection (stocks/crypto/all)
   - Universe size (small/medium/large)
   - Date range picker
   - Progress tracking

2. **Training Configuration**
   - Epochs, batch size, learning rate
   - Sequence length, patience
   - Data path specification

3. **Training Monitor**
   - Real-time loss curves
   - Progress bar
   - Logs streaming

4. **Model Management**
   - List trained models
   - Download/delete models
   - View model info

### API Endpoints
- `POST /api/download-data` - Start data download
- `GET /api/download-status` - Check download progress
- `POST /api/train` - Start training
- `GET /api/training-status` - Training progress
- `WS /ws` - WebSocket for real-time updates
- `GET /api/models` - List models
- `POST /api/backtest` - Run backtest

---

## 🏗️ ARCHITECTURE INSIGHTS

### Strengths ✅
1. **Dual-path architecture** captures both temporal dependencies (xLSTM) and patterns (Transformer)
2. **Feature-rich**: 60+ technical indicators provide comprehensive market context
3. **Multi-task learning**: Predicts price, direction, volatility, confidence simultaneously
4. **Position state integration**: Model aware of current holdings
5. **Robust backtesting**: Realistic slippage, commission, risk management
6. **Web interface**: Easy to configure & monitor training

### Limitations/Concerns ⚠️
1. **60 features → may be redundant** (correlation between indicators)
2. **3-layer xLSTM + 3-layer Transformer** = Deep model, risk of overfitting
3. **27.1M parameters** on potentially limited data → needs regularization
4. **Frontend is basic** - no portfolio monitoring, no live trading view
5. **Index data not supported** - only individual symbols (SPY, QQQ, individual stocks)
6. **No support for index futures** or spread trading

---

## 📊 PROPOSED ARCHITECTURE IMPROVEMENTS

### 1. **Feature Reduction**
- Current: 60 features
- Proposed: 25-30 key features (PCA or correlation filtering)
- Benefits: Faster training, less overfitting, interpretability

### 2. **Model Simplification Option A**
```
Input → xLSTM (2 layers) → Temporal Pooling → Output Heads
(Simpler, faster training)
```

### 3. **Model Simplification Option B** (Ensemble approach)
```
Input → Separate models per task:
  ├─ Direction predictor (2-layer xLSTM)
  ├─ Price predictor (Transformer only)
  └─ Risk predictor (Simpler network)
(More interpretable, easier to debug)
```

### 4. **Index Data Support**
- Add index calculation module
- Support OHLCV reconstructed from constituents
- Track index momentum separately

### 5. **New Frontend (React-based)**
- Real-time portfolio dashboard
- Live P&L tracking
- Trade notifications
- Model performance metrics
- Data quality indicators

### 6. **Advanced Training Features**
- Meta-learning for quick adaptation to new markets
- Online learning for live retraining
- Attention visualization for interpretability

---

## 📁 KEY FILES SUMMARY

| File | Purpose |
|------|---------|
| `scripts/download_data.py` | Download historical data |
| `scripts/train_model.py` | Train the model |
| `scripts/backtest.py` | Backtest trained model |
| `scripts/paper_trade.py` | Paper trading |
| `data/features.py` | Feature engineering (60+ indicators) |
| `data/pipeline.py` | Data processing pipeline |
| `models/trading_nn.py` | Main neural network |
| `models/xlstm.py` | xLSTM implementation |
| `models/transformer.py` | Transformer encoder |
| `training/trainer.py` | Training loop |
| `execution/backtester.py` | Backtesting engine |
| `execution/risk_manager.py` | Risk management |
| `web/app.py` | FastAPI application |
| `config/model_config.py` | Model hyperparameters |
| `config/settings.py` | Global settings |

---

## 🚀 NEXT STEPS FOR REDESIGN

1. **Phase 1**: Understand current results (backtest performance)
2. **Phase 2**: Reduce features to 25-30 key indicators
3. **Phase 3**: Add index data support (SPY, QQQ, sector ETFs)
4. **Phase 4**: Redesign frontend (React-based or Vue.js)
5. **Phase 5**: Implement new frontend backend API
6. **Phase 6**: Support index futures and spread trading
7. **Phase 7**: Add ensemble methods or model selection logic

