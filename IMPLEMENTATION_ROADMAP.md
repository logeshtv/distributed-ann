# 🗺️ DETAILED ACTION PLAN: Data, Training & Frontend Rebuild

## Part 1: DATA PIPELINE - Comprehensive View

### Current Data Flow Diagram
```
┌────────────────────────────────────────────────────────────────┐
│                  DATA SOURCES                                  │
└────────────────────────────────────────────────────────────────┘
          ↓                      ↓                      ↓
    ┌──────────┐          ┌──────────┐          ┌──────────┐
    │  ALPACA  │          │ BINANCE  │          │ yfinance │
    │(Stocks)  │          │ (Crypto) │          │(Fallback)│
    └──────────┘          └──────────┘          └──────────┘
          ↓                      ↓                      ↓
    ┌──────────────────────────────────────────────────┐
    │        download_data.py (Entry Point)            │
    │  - Parse args (source, dates, universe_size)    │
    │  - Load symbol lists from data/tickers.py       │
    │  - Call fetch functions                         │
    │  - Handle errors, retry logic                   │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  DATA FORMAT: Raw OHLCV Parquet Files            │
    │  Location: data_storage/raw/{stocks,crypto}/     │
    │  Columns: timestamp, symbol, open, high,        │
    │           low, close, volume, vwap              │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  IMPROVEMENT #1: Add Index Data                  │
    │  ├─ S&P 500 (SPY)                               │
    │  ├─ NASDAQ 100 (QQQ)                            │
    │  ├─ Russell 2000 (IWM)                          │
    │  ├─ Sector ETFs (XLK, XLF, XLV, etc.)          │
    │  └─ Volatility Index (VIX)                      │
    │  Location: data_storage/raw/indices/            │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  train_model.py → DataPipeline                   │
    │  - Load all parquet files                       │
    │  - Sort by symbol + timestamp                   │
    │  - Filter by date range                         │
    │  - Handle missing data                          │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  IMPROVEMENT #2: Feature Selection               │
    │  Current: 60 features → Target: 35-40 features  │
    │  Methods:                                        │
    │  ├─ Correlation filtering (|r| > 0.8)           │
    │  ├─ Feature importance (XGBoost ranking)        │
    │  ├─ PCA (25-30 components)                      │
    │  └─ Domain knowledge (keep proven indicators)   │
    │  Output: features_selected_df.parquet           │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  FeatureEngineer.add_all_features()              │
    │  ├─ Return features (5,10,20 period)            │
    │  ├─ Trend: SMA, EMA, MACD, ADX, Aroon          │
    │  ├─ Momentum: RSI, Stochastic, Williams%R      │
    │  ├─ Volatility: ATR, Bollinger Bands           │
    │  ├─ Volume: OBV, VWAP, CMF                     │
    │  ├─ Temporal: day_of_week, month               │
    │  └─ Index relatives (NEW)                       │
    │     ├─ Correlation to index                     │
    │     ├─ Beta calculation                         │
    │     └─ Relative strength vs index               │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  Label Creation (per symbol)                     │
    │  For each timestamp:                             │
    │  ├─ target_return_1d = (close[t+1]/close[t])-1  │
    │  ├─ target_return_4d = (close[t+4]/close[t])-1  │
    │  ├─ target_return_24d = (close[t+24]/close[t])-1│
    │  ├─ target_direction = classify by ±0.5%        │
    │  ├─ target_volatility = future ATR              │
    │  └─ target_confidence = model certainty (0-1)   │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  Train/Val/Test Split                            │
    │  ├─ 70% Training (2000-2022)                    │
    │  ├─ 15% Validation (2022-2023)                  │
    │  └─ 15% Testing (2023-2024)                     │
    │  Note: Temporal split to avoid look-ahead bias  │
    └──────────────────────────────────────────────────┘
          ↓
    ┌──────────────────────────────────────────────────┐
    │  TradingDataset (PyTorch)                        │
    │  - Sequences: 60 timesteps × 40 features        │
    │  - Batch: DataLoader(batch_size=512)            │
    │  - Returns: {features, targets}                  │
    └──────────────────────────────────────────────────┘
```

### Implementation Tasks

#### Task 1.1: Add Index Data Source
```python
# data/fetchers/index_fetcher.py (NEW)

class IndexFetcher:
    """Fetch major indices and sector ETFs."""
    
    INDICES = {
        'SPY': 'S&P 500',
        'QQQ': 'NASDAQ 100',
        'IWM': 'Russell 2000',
        'VIX': 'Volatility Index'
    }
    
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLRE': 'Real Estate',
        'XLE': 'Energy',
        'XLU': 'Utilities',
        'XLRE': 'Materials'
    }
    
    def fetch_indices(self, start_date, end_date):
        """Fetch all indices."""
        # Use yfinance for all (free, no API key)
        pass
```

#### Task 1.2: Feature Selection Module
```python
# data/feature_selector.py (NEW)

class FeatureSelector:
    """Select optimal features from 60 available."""
    
    def rank_by_correlation(self, df):
        """Remove highly correlated features."""
        # Keep 1 from each correlation group
        # Result: 60 → 40 features
        pass
    
    def rank_by_importance(self, df, labels):
        """Train quick XGBoost, get importances."""
        # Quick model for feature ranking
        # Keep top 35 features
        pass
    
    def rank_by_pca(self, df, n_components=30):
        """PCA transformation."""
        # Reduce to 30 principal components
        pass
    
    def get_selected_features(self, method='correlation'):
        """Return list of selected feature columns."""
        pass
```

---

## Part 2: MODEL TRAINING - Architecture Comparison

### Training Data Flow with Three Model Variants

```
                     PREPARED DATA (70% train set)
                              ↓
                ┌─────────────┴─────────────┐
                ↓                           ↓
        ┌──────────────┐            ┌──────────────┐
        │ CONFIG MODEL │            │  INITIALIZE  │
        │ ARCHITECTURE │            │   TRAINER    │
        └──────────────┘            └──────────────┘
             ↓                              ↓
    ┌────────────────────────────────────────┐
    │     THREE MODEL ARCHITECTURES          │
    └────────────────────────────────────────┘
         ↓              ↓              ↓
    
OPTION A          OPTION B          OPTION C
SimplexLSTM       EfficientHybrid    TaskEnsemble
────────          ───────────────    ────────────
Input(40)         Input(40)          Input(40)
    ↓                 ↓                 ├─→ DirectionNet
xLSTM             xLSTM(256,2L)      │   (xLSTM 2L)
(256,2L)          + Transformer      ├─→ PriceNet
    ↓             (128,2L)           │   (Transformer 2L)
Pool              ↓                  └─→ RiskNet
    ↓             Fusion             (MLP 2L)
Head            ↓                  ↓
                 Head            Ensemble
~3M params       ~6M params      ~8M params
Fast ⚡         Balanced        Interpretable

TRAINING (All Options):
├─ Optimizer: AdamW (lr=5e-4)
├─ Scheduler: CosineAnnealingWarmRestarts
├─ Loss: MultiTaskLoss (weighted)
├─ Batch size: 512
├─ Epochs: 100
├─ Early stopping: patience=15
└─ Device: GPU (cuda)

PER EPOCH:
├─ Forward pass on all batches
├─ Backward pass (grad accumulation)
├─ Update weights
├─ Validation on val set (15%)
├─ Log metrics
└─ Save checkpoint if val_loss improves

OUTPUTS:
├─ Best model saved: data_storage/models/
├─ Training curves: loss vs epoch
├─ Validation metrics: MAE, accuracy, AUC
└─ Training logs: logs/
```

### Implementation: Model Architecture Variants

#### Task 2.1: SimplexLSTM (Fast Option)
```python
# models/variants/simplex_lstm.py

import torch.nn as nn

class SimplexLSTM(nn.Module):
    """Single-path xLSTM architecture."""
    
    def __init__(self, 
                 input_dim: int = 40,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 output_dim: int = 256):
        super().__init__()
        self.xlstm = xLSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.3
        )
        self.temporal_pool = TemporalFusion(hidden_dim, output_dim)
        self.heads = MultiTaskHead(output_dim)
    
    def forward(self, x):
        x = self.xlstm(x)  # (batch, seq, hidden)
        x = self.temporal_pool(x)  # (batch, output_dim)
        outputs = self.heads(x)
        return outputs

# Parameters: ~3M
# Training time: ~30-45 min on GPU
# Inference: ~15ms per sample
```

#### Task 2.2: EfficientHybrid (Balanced)
```python
# models/variants/efficient_hybrid.py

class EfficientHybrid(nn.Module):
    """Reduced-complexity hybrid architecture."""
    
    def __init__(self,
                 input_dim: int = 40,
                 xlstm_hidden: int = 256,
                 transformer_dim: int = 128,
                 output_dim: int = 256):
        super().__init__()
        self.xlstm = xLSTM(input_dim, xlstm_hidden, num_layers=2)
        self.transformer = TransformerEncoder(
            input_dim, 
            d_model=transformer_dim,
            nhead=4,  # Reduced from 8
            num_layers=2  # Reduced from 3
        )
        self.fusion = FusionLayer(
            xlstm_dim=xlstm_hidden,
            transformer_dim=transformer_dim,
            output_dim=output_dim
        )
        self.heads = MultiTaskHead(output_dim)
    
    def forward(self, x):
        xlstm_out = self.xlstm(x)
        transformer_out = self.transformer(x)
        fused = self.fusion(xlstm_out, transformer_out)
        outputs = self.heads(fused)
        return outputs

# Parameters: ~6M
# Training time: ~45-60 min
# Inference: ~20ms per sample
```

#### Task 2.3: TaskEnsemble (Interpretable)
```python
# models/variants/task_ensemble.py

class DirectionNet(nn.Module):
    """Predict direction only."""
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            xLSTM(input_dim, 256, 2),
            TemporalFusion(256, 256),
            nn.Linear(256, 3)  # 3 classes: down/neutral/up
        )
    
    def forward(self, x):
        return self.net(x)

class PriceNet(nn.Module):
    """Predict future returns."""
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            TransformerEncoder(input_dim, 128, 4, 2),
            TemporalFusion(128, 256),
            nn.Linear(256, 3)  # 1d, 4d, 24d returns
        )
    
    def forward(self, x):
        return self.net(x)

class RiskNet(nn.Module):
    """Predict volatility & confidence."""
    def __init__(self, input_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*60, 128),  # Flatten
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # volatility, confidence
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class TaskEnsemble(nn.Module):
    """Ensemble of task-specific models."""
    
    def __init__(self, input_dim=40):
        super().__init__()
        self.direction_net = DirectionNet(input_dim)
        self.price_net = PriceNet(input_dim)
        self.risk_net = RiskNet(input_dim)
    
    def forward(self, x):
        direction = self.direction_net(x)
        price = self.price_net(x)
        risk = self.risk_net(x)
        
        return {
            'direction': direction,
            'price': price,
            'risk': risk
        }

# Parameters: ~8M total
# Training time: ~50-70 min
# Inference: ~25ms per sample
```

### Training Script Comparison
```bash
# Compare all three architectures
python scripts/train_model.py \
    --data-path data_storage/raw \
    --model simplex_lstm \
    --epochs 100 \
    --output-dir results/model_comparison/simplex_lstm

python scripts/train_model.py \
    --data-path data_storage/raw \
    --model efficient_hybrid \
    --epochs 100 \
    --output-dir results/model_comparison/efficient_hybrid

python scripts/train_model.py \
    --data-path data_storage/raw \
    --model task_ensemble \
    --epochs 100 \
    --output-dir results/model_comparison/task_ensemble

# Compare results
python scripts/compare_models.py --results-dir results/model_comparison/
```

---

## Part 3: FRONTEND REDESIGN - Complete New Stack

### Current Frontend Issues
```
❌ Single HTML file (260 lines)
❌ No state management
❌ No component reusability
❌ Limited visualization
❌ No portfolio tracking
❌ No real-time updates
❌ Hard to extend
```

### New Frontend Architecture

```
┌─────────────────────────────────────────────────────┐
│              REACT APPLICATION (Frontend)           │
├─────────────────────────────────────────────────────┤
│
├─ src/
│  ├─ pages/
│  │  ├─ Dashboard.tsx        (Main portfolio view)
│  │  ├─ DataManager.tsx       (Download & preprocess)
│  │  ├─ Training.tsx          (Configuration & monitor)
│  │  ├─ Backtest.tsx          (Simulation & results)
│  │  ├─ LiveTrading.tsx       (Real-time signals)
│  │  └─ Settings.tsx          (Configuration)
│  │
│  ├─ components/
│  │  ├─ PortfolioCard.tsx     (Equity, drawdown, Sharpe)
│  │  ├─ SignalsList.tsx       (Recent signals)
│  │  ├─ TrainingMonitor.tsx   (Loss curves)
│  │  ├─ DataQualityCheck.tsx  (Data health)
│  │  ├─ RiskDashboard.tsx     (Risk metrics)
│  │  └─ Charts/
│  │     ├─ EquityCurve.tsx
│  │     ├─ LossCurve.tsx
│  │     ├─ DrawdownChart.tsx
│  │     └─ SignalHeatmap.tsx
│  │
│  ├─ hooks/
│  │  ├─ useWebSocket.ts      (Real-time connection)
│  │  ├─ usePortfolio.ts      (Portfolio state)
│  │  ├─ useTraining.ts       (Training state)
│  │  └─ useBacktest.ts       (Backtest results)
│  │
│  ├─ services/
│  │  ├─ api.ts              (API calls)
│  │  ├─ websocket.ts        (WebSocket manager)
│  │  └─ storage.ts          (Local storage)
│  │
│  ├─ types/
│  │  └─ index.ts            (TypeScript types)
│  │
│  └─ App.tsx               (Root component)
│
├─ public/
│  ├─ index.html
│  ├─ favicon.ico
│  └─ config.json
│
├─ package.json
├─ tsconfig.json
└─ .env.example
```

### New Frontend Stack

```
Frontend:
├─ React 18 (UI framework)
├─ TypeScript (type safety)
├─ Recharts (data visualization)
├─ Zustand (state management)
├─ React Query (server state)
├─ Tailwind CSS (styling)
├─ Axios (HTTP client)
└─ date-fns (date handling)

Backend API (FastAPI):
├─ /api/v1/portfolio
│  ├─ GET /current        → Current positions & P&L
│  ├─ GET /history        → Historical trades
│  └─ GET /performance    → Sharpe, max DD, returns
│
├─ /api/v1/signals
│  ├─ GET /recent         → Latest 20 signals
│  ├─ GET /by-symbol/{s}  → Signals for symbol
│  └─ GET /confidence     → High confidence signals
│
├─ /api/v1/training
│  ├─ GET /status         → Current training status
│  ├─ GET /metrics        → Loss, accuracy curves
│  ├─ POST /start         → Start training
│  └─ POST /stop          → Stop training
│
├─ /api/v1/backtest
│  ├─ POST /run           → Start backtest
│  ├─ GET /results/{id}   → Backtest results
│  └─ GET /trades/{id}    → Trades list
│
├─ /api/v1/data
│  ├─ POST /download      → Start download
│  ├─ GET /status         → Download progress
│  └─ GET /quality        → Data quality check
│
└─ /api/v1/system
   ├─ GET /health         → System status
   ├─ GET /config         → Current configuration
   └─ GET /logs           → Recent logs
```

### Frontend Component Examples

#### Dashboard Page
```typescript
// src/pages/Dashboard.tsx

import { usePortfolio } from '../hooks/usePortfolio';
import { useWebSocket } from '../hooks/useWebSocket';
import PortfolioCard from '../components/PortfolioCard';
import SignalsList from '../components/SignalsList';
import EquityCurve from '../components/Charts/EquityCurve';
import RiskDashboard from '../components/RiskDashboard';

export default function Dashboard() {
  const portfolio = usePortfolio();
  const signals = useWebSocket('/ws/signals');
  
  return (
    <div className="grid grid-cols-4 gap-4">
      <PortfolioCard
        equity={portfolio.equity}
        return={portfolio.totalReturn}
        sharpe={portfolio.sharpeRatio}
        maxDD={portfolio.maxDrawdown}
      />
      
      <EquityCurve data={portfolio.equityCurve} />
      
      <RiskDashboard
        positions={portfolio.positions}
        dailyLoss={portfolio.dailyLoss}
        var95={portfolio.var95}
      />
      
      <SignalsList signals={signals} />
    </div>
  );
}
```

#### Training Monitor Component
```typescript
// src/components/TrainingMonitor.tsx

import { useTraining } from '../hooks/useTraining';
import { LineChart, Line, XAxis, YAxis } from 'recharts';

export default function TrainingMonitor() {
  const training = useTraining();
  
  return (
    <div className="card">
      <h2>Training Progress</h2>
      
      <div className="grid grid-cols-2">
        <div>
          <p>Epoch: {training.epoch}/100</p>
          <p>Loss: {training.loss.toFixed(4)}</p>
          <p>Val Loss: {training.valLoss.toFixed(4)}</p>
        </div>
        
        <LineChart width={400} height={300} data={training.losses}>
          <XAxis dataKey="epoch" />
          <YAxis />
          <Line type="monotone" dataKey="loss" stroke="#8884d8" />
          <Line type="monotone" dataKey="valLoss" stroke="#82ca9d" />
        </LineChart>
      </div>
    </div>
  );
}
```

### Backend API Additions

```python
# web/app.py (Extended)

from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime

app = FastAPI()

# Portfolio endpoints
@app.get("/api/v1/portfolio/current")
async def get_portfolio():
    """Get current portfolio state."""
    return {
        "equity": position_tracker.equity,
        "cash": position_tracker.cash,
        "positions": position_tracker.positions,
        "totalReturn": (position_tracker.equity - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/portfolio/performance")
async def get_performance():
    """Get performance metrics."""
    return {
        "sharpeRatio": calculate_sharpe(equity_curve),
        "sortinoRatio": calculate_sortino(equity_curve),
        "maxDrawdown": calculate_max_dd(equity_curve),
        "winRate": calculate_win_rate(trades),
        "profitFactor": calculate_profit_factor(trades)
    }

# Signal endpoints
@app.get("/api/v1/signals/recent")
async def get_recent_signals(limit: int = 20):
    """Get recent trading signals."""
    return signal_buffer[-limit:]

@app.get("/api/v1/signals/confidence")
async def get_high_confidence_signals(threshold: float = 0.7):
    """Get signals with confidence > threshold."""
    return [s for s in signal_buffer if s['confidence'] > threshold]

# Training endpoints
@app.post("/api/v1/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start model training."""
    background_tasks.add_task(training_service.train, config)
    return {"status": "training_started"}

@app.get("/api/v1/training/status")
async def get_training_status():
    """Get current training status."""
    return training_service.get_status()

# WebSocket for real-time updates
@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signals."""
    await websocket.accept()
    try:
        while True:
            signal = signal_queue.get()  # Non-blocking queue
            await websocket.send_json(signal)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
```

---

## Part 4: EXECUTION & IMPLEMENTATION TIMELINE

### Week 1-2: Foundation
```
Tasks:
└─ [ ] Document current performance (backtest results)
└─ [ ] Create baseline metrics file
└─ [ ] Add index data fetching (SPY, QQQ, VIX)
└─ [ ] Implement feature selection module
└─ [ ] Run correlation analysis (60 → 35 features)
```

### Week 3-4: Model Optimization
```
Tasks:
└─ [ ] Implement SimplexLSTM variant
└─ [ ] Implement EfficientHybrid variant
└─ [ ] Implement TaskEnsemble variant
└─ [ ] Compare all 3 architectures
└─ [ ] Benchmark: speed, accuracy, generalization
└─ [ ] Select best architecture
```

### Week 5-7: Frontend V1
```
Tasks:
└─ [ ] Setup React project structure
└─ [ ] Implement Dashboard page
└─ [ ] Implement DataManager page
└─ [ ] Implement Training page
└─ [ ] Connect to backend APIs
└─ [ ] Test all components
└─ [ ] Deploy to localhost:3000
```

### Week 8-10: Advanced Features
```
Tasks:
└─ [ ] Implement curriculum learning
└─ [ ] Implement online learning
└─ [ ] Add attention visualization
└─ [ ] Add model ensembling
└─ [ ] Implement walk-forward validation
└─ [ ] Stress testing (market crashes)
```

### Week 11+: Production
```
Tasks:
└─ [ ] Live trading pipeline
└─ [ ] Monitoring & alerting
└─ [ ] Cloud deployment (Railway/AWS)
└─ [ ] Documentation
└─ [ ] Performance monitoring
```

---

## 📝 QUICK START COMMANDS

### Phase 1: Add Index Data
```bash
# Create new fetcher
touch data/fetchers/index_fetcher.py

# Update tickers.py
# Add: INDEX_SYMBOLS = ['SPY', 'QQQ', 'IWM', 'VIX']

# Update download_data.py
# Add: --include-indices flag

# Test
python scripts/download_data.py --source all --include-indices
```

### Phase 2: Feature Selection
```bash
# Create feature selector
touch data/feature_selector.py

# Run feature selection
python -c "
from data.feature_selector import FeatureSelector
selector = FeatureSelector()
selected = selector.select_by_correlation(df, threshold=0.8)
print(f'Selected {len(selected)} features')
"
```

### Phase 3: Model Variants
```bash
# Create variant models
mkdir -p models/variants
touch models/variants/simplex_lstm.py
touch models/variants/efficient_hybrid.py
touch models/variants/task_ensemble.py

# Train and compare
for model in simplex_lstm efficient_hybrid task_ensemble; do
    python scripts/train_model.py \
        --model $model \
        --output results/comparison_$model
done
```

### Phase 4: Frontend Setup
```bash
# Create React app
npx create-react-app frontend --template typescript
cd frontend

# Install dependencies
npm install react-router-dom recharts zustand react-query axios date-fns tailwindcss

# Start development server
npm start  # Runs on http://localhost:3000
```

---

## ✅ SUCCESS CRITERIA

✓ Phase 1 Complete:
- Index data integrated
- Feature selection reduces 60→35 features
- Data pipeline supports both individual symbols and indices

✓ Phase 2 Complete:
- All 3 model variants trained
- Performance comparison documented
- Best model selected based on: training time, accuracy, generalization

✓ Phase 3 Complete:
- React frontend running locally
- All pages (Dashboard, DataManager, Training, Backtest) functional
- Connected to backend APIs

✓ Phase 4 Complete:
- Advanced features implemented (curriculum, online learning, attention viz)
- Stress tests passing
- Ready for production deployment

