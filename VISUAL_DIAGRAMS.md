# 📊 VISUAL SYSTEM DIAGRAMS & FLOWS

## 1. DATA PIPELINE SEQUENCE DIAGRAM

```
Timeline: Data → Features → Sequences → Training → Model

DATA DOWNLOAD PHASE
═══════════════════
                    ┌─────────────────────────────────┐
                    │  Alpaca / Binance / yfinance   │
                    │   (Historical OHLCV data)      │
                    └──────────────┬──────────────────┘
                                   │ (Raw data)
                                   ▼
                    ┌─────────────────────────────────┐
                    │   data_storage/raw/             │
                    │   ├─ stocks/*.parquet           │
                    │   ├─ crypto/*.parquet           │
                    │   └─ indices/*.parquet (NEW)    │
                    └──────────────┬──────────────────┘
                                   │ (Parquet files)

FEATURE ENGINEERING PHASE
══════════════════════════
                                   ▼
        ┌──────────────────────────────────────────────┐
        │  FeatureEngineer.add_all_features()          │
        │  (Process per symbol to avoid look-ahead)    │
        ├──────────────────────────────────────────────┤
        │ Returns:                                     │
        │ ├─ return_1, return_2, ..., return_20 (7)   │
        │ ├─ sma_5, sma_10, ..., ema_50 (10)          │
        │ ├─ macd, macd_signal, macd_hist (3)         │
        │ ├─ rsi_7, rsi_14, rsi_21 (3)                │
        │ ├─ atr, bollinger_upper, bollinger_lower (3)│
        │ ├─ obv, vwap, cmf (3)                       │
        │ ├─ day_of_week, month_sin, month_cos (3)    │
        │ └─ + correlation_to_index, beta (NEW)       │
        │                                              │
        │ TOTAL: 35-40 features (down from 60)        │
        └──────────────┬───────────────────────────────┘
                       │ (Features added per row)

LABEL CREATION PHASE
════════════════════
                       ▼
        ┌──────────────────────────────────────────────┐
        │  Create targets for supervised learning      │
        │  For each row [t]:                           │
        │  ├─ target_return_1d = return[t+1]           │
        │  ├─ target_return_4d = return[t+4]           │
        │  ├─ target_return_24d = return[t+24]         │
        │  ├─ target_direction:                        │
        │  │  └─ 0 if return < -0.5%                  │
        │  │  └─ 1 if -0.5% < return < +0.5%          │
        │  │  └─ 2 if return > +0.5%                  │
        │  ├─ target_volatility = future ATR           │
        │  └─ target_confidence = (set to 0.5 initial) │
        └──────────────┬───────────────────────────────┘
                       │ (Labels added)

SEQUENCE CREATION PHASE
═══════════════════════
                       ▼
        ┌──────────────────────────────────────────────┐
        │  TradingDataset (PyTorch)                    │
        │  Sliding window approach:                    │
        │                                              │
        │  Window size: 60 timesteps (60 bars)        │
        │  Features per timestep: 40 (after selection)│
        │  Shape: (60, 40)                            │
        │                                              │
        │  No overlapping sequences (1-bar shift)      │
        │  No look-ahead bias                          │
        │                                              │
        │  Per symbol:                                │
        │  ├─ 1000 bars → ~940 sequences             │
        │  ├─ 100 symbols → ~94,000 sequences         │
        │  └─ Total samples: ~100,000 sequences       │
        └──────────────┬───────────────────────────────┘
                       │ (Sequences created)

TRAIN/VAL/TEST SPLIT
════════════════════
                       ▼
        ┌──────────────────────────────────────────────┐
        │  Temporal Split (NO random shuffling)        │
        │                                              │
        │  Train: 70% (2000-2022) ~ 70,000 samples    │
        │  Val:   15% (2022-2023) ~ 15,000 samples    │
        │  Test:  15% (2023-2024) ~ 15,000 samples    │
        │                                              │
        │  DataLoader:                                 │
        │  ├─ batch_size = 512                        │
        │  ├─ shuffle = True (only within train set)  │
        │  └─ num_workers = 4                         │
        └──────────────┬───────────────────────────────┘
                       │ (Data ready for training)

TRAINING PHASE
══════════════
                       ▼
        ┌──────────────────────────────────────────────┐
        │  For each epoch (1-100):                     │
        │  ├─ For each batch (136 batches):           │
        │  │  ├─ Forward pass (batch through model)   │
        │  │  ├─ Compute loss (MultiTaskLoss)         │
        │  │  ├─ Backward pass (gradients)            │
        │  │  └─ Update weights (optimizer step)      │
        │  │                                           │
        │  ├─ Validation (every epoch)                │
        │  │  └─ Evaluate on val set                  │
        │  │                                           │
        │  └─ Early stopping (if val loss plateaus)   │
        │                                              │
        │  Result: data_storage/models/best_model.pt  │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │  TRAINED MODEL (5-8M parameters)            │
        │  Ready for backtesting & paper trading      │
        └──────────────────────────────────────────────┘
```

---

## 2. MODEL ARCHITECTURE DETAILED

### Current Architecture (27.1M params)
```
┌─────────────────────────────────────┐
│  INPUT: (batch, 60, 60)             │  ← batch_size, timesteps, features
│  ├─ 60 timesteps (60 trading days)  │
│  └─ 60 features (technical indicators)
└──────────────┬──────────────────────┘
               │
        ┌──────┴────────┐
        ▼               ▼
  ┌───────────┐    ┌──────────────┐
  │   xLSTM   │    │ TRANSFORMER  │
  ├───────────┤    ├──────────────┤
  │ Layers: 2 │    │  Layers: 3   │
  │ Hidden:512│    │  Heads: 8    │
  │ Output:512│    │  Dim: 256    │
  │           │    │  Output: 256 │
  │ Special:  │    │              │
  │ Exp gates │    │ Self-Attn:   │
  │ Prevents  │    │ Q,K,V        │
  │ grad exp  │    │              │
  └─────┬─────┘    └──────┬───────┘
        │                 │
        │ (512 dims)      │ (256 dims)
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  FUSION LAYER   │
        ├─────────────────┤
        │ Concatenate:    │
        │ 512 + 256 = 768 │
        │                 │
        │ Attention:      │
        │ Weight each     │
        │ path            │
        │                 │
        │ Output: 256 dim │
        └────────┬────────┘
                 │
        ┌────────▼──────────────┐
        │ POSITION STATE        │
        │ ENCODER (optional)    │
        ├───────────────────────┤
        │ Input: portfolio info │
        │ ├─ positions held     │
        │ ├─ entry price        │
        │ ├─ current P&L        │
        │ └─ risk metrics       │
        │                       │
        │ Embedding: 320 dims   │
        │ Integration with      │
        │ market features       │
        │                       │
        │ Output: 256 dims      │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────┐
        │ TEMPORAL FUSION       │
        ├───────────────────────┤
        │ Aggregate sequence    │
        │ to single vector      │
        │                       │
        │ Method: Attention    │
        │ weights over time    │
        │                       │
        │ Output: 256 dims      │
        └────────┬──────────────┘
                 │
        ┌────────▼──────────────────────┐
        │    MULTI-TASK OUTPUT HEADS    │
        ├───────────────────────────────┤
        │                               │
        │ 1. PRICE HEAD                │
        │    ├─ MLP: 256 → 128 → 64    │
        │    └─ Output: 3 values       │
        │       ├─ return_1d_pred      │
        │       ├─ return_4d_pred      │
        │       └─ return_24d_pred     │
        │                               │
        │ 2. DIRECTION HEAD             │
        │    ├─ MLP: 256 → 128 → 64    │
        │    └─ Output: 3 logits       │
        │       ├─ P(down)             │
        │       ├─ P(neutral)          │
        │       └─ P(up)               │
        │                               │
        │ 3. POSITION HEAD              │
        │    ├─ MLP: 256 → 128 → 64    │
        │    └─ Output: 3 logits       │
        │       ├─ P(sell)             │
        │       ├─ P(hold)             │
        │       └─ P(buy)              │
        │                               │
        │ 4. VOLATILITY HEAD            │
        │    ├─ MLP: 256 → 64          │
        │    └─ Output: 1 value        │
        │       └─ predicted_vol        │
        │                               │
        │ 5. CONFIDENCE HEAD            │
        │    ├─ MLP: 256 → 64          │
        │    └─ Output: 1 value        │
        │       └─ confidence (0-1)     │
        │                               │
        │ 6. RISK HEAD                  │
        │    ├─ MLP: 256 → 64          │
        │    └─ Output: 1 value        │
        │       └─ risk_signal (0-1)    │
        │                               │
        └───────────────────────────────┘
                 │
        ┌────────▼──────────────────────┐
        │   OUTPUTS (per sample)        │
        ├───────────────────────────────┤
        │ ├─ price_1d, 4d, 24d (float) │
        │ ├─ direction (logits)         │
        │ ├─ position (logits)          │
        │ ├─ volatility (float)         │
        │ ├─ confidence (0-1)           │
        │ └─ risk_signal (0-1)          │
        └───────────────────────────────┘

PARAMETER BREAKDOWN:
═══════════════════
xLSTM:              ~2.0M params
  - Embeddings & gates for 60→512→512 path
  
Transformer:        ~1.8M params
  - Attention heads, FFN for 60→256→256 path
  
Fusion:             ~0.3M params
  - Concatenation & attention weighting

Position Encoder:   ~1.5M params
  - Embeddings for portfolio state

Temporal Fusion:    ~0.1M params
  - Attention weights aggregation

Output Heads:       ~21.3M params  ← LARGEST PART!
  - 6 separate MLPs with 256 input dim

TOTAL:              27.1M params
```

### Simplified Architecture (5M params)
```
INPUT: (batch, 60, 40)
    ▼
xLSTM (256 hidden, 2L)
    ▼
TemporalFusion
    ▼
MultiTaskHeads
    │
    ├─ Price (256→64→3)
    ├─ Direction (256→64→3)
    ├─ Position (256→64→3)
    └─ Volatility (256→64→1)
    
Parameters: ~5M
Training: 40-50 min
Inference: 20ms
```

---

## 3. TRAINING LOOP FLOW

```
┌─────────────────────────────┐
│  START TRAINING             │
│  - Load config              │
│  - Initialize model         │
│  - Setup optimizer          │
│  - Setup loss function      │
└──────────────┬──────────────┘
               │
        FOR EPOCH = 1 to 100:
        ┌──────────────┐
        │ EPOCH LOOP   │
        └──────┬───────┘
               │
        ┌──────▼──────────────────┐
        │ FOR BATCH in train_set: │
        │ (136 batches/epoch)     │
        └──────┬───────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 1. GET BATCH                   │
        │ batch = {                      │
        │   'features': Tensor(512, 60, 40),
        │   'targets': Tensor(512, 2)    │
        │ }                              │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 2. FORWARD PASS                │
        │ output = model(features)       │
        │ output = {                     │
        │   'price': (512, 3),           │
        │   'direction': (512, 3),       │
        │   'volatility': (512, 1),      │
        │   ...                          │
        │ }                              │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 3. COMPUTE LOSS                │
        │ loss = MultiTaskLoss(output,   │
        │   targets, weights={           │
        │   'price': 0.30,               │
        │   'direction': 0.20,           │
        │   'volatility': 0.15,          │
        │   ...                          │
        │ })                             │
        │                                │
        │ Returns:                       │
        │ loss = {                       │
        │   'price_loss': 0.25,          │
        │   'dir_loss': 0.18,            │
        │   'total_loss': 0.65           │
        │ }                              │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 4. BACKWARD PASS               │
        │ loss['total_loss'].backward()  │
        │ (Compute gradients)            │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 5. GRADIENT CLIPPING           │
        │ torch.nn.utils.clip_grad_norm_(│
        │   model.parameters(),          │
        │   max_norm=1.0)                │
        │ (Prevent exploding gradients)  │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 6. OPTIMIZER STEP              │
        │ optimizer.step()               │
        │ (Update weights)               │
        │                                │
        │ optimizer.zero_grad()          │
        │ (Reset gradients)              │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 7. LOG METRICS                 │
        │ track_loss.append(loss)        │
        │ pbar.update(1)                 │
        └──────┬─────────────────────────┘
               │
        (end FOR BATCH loop)
        │
        ┌──────▼─────────────────────────┐
        │ 8. VALIDATION                  │
        │ FOR val_batch in val_set:      │
        │   (validation loop, no grads)  │
        │   val_loss = evaluate()        │
        │                                │
        │ history['val_loss'].append()   │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 9. CHECK EARLY STOPPING        │
        │ IF val_loss < best_val_loss:   │
        │   ├─ best_val_loss = val_loss  │
        │   ├─ patience_counter = 0      │
        │   └─ Save checkpoint           │
        │ ELSE:                          │
        │   ├─ patience_counter += 1     │
        │   └─ IF patience_counter >= 15:│
        │       └─ STOP TRAINING        │
        └──────┬─────────────────────────┘
               │
        ┌──────▼─────────────────────────┐
        │ 10. SCHEDULER STEP             │
        │ scheduler.step()               │
        │ (Adjust learning rate)         │
        │                                │
        │ CosineAnnealingWarmRestarts:   │
        │ ├─ T_0 = 10 (initial period)   │
        │ └─ T_mult = 2 (period doubles) │
        │                                │
        │ LR schedule:                   │
        │ Epoch 1-10:   High → Low       │
        │ Epoch 10-30:  Low → High       │
        │ (Warm restart)                 │
        │ Epoch 30-90:  High → Low (×4)  │
        └──────┬─────────────────────────┘
               │
        (end FOR EPOCH loop)
        │
        ┌──────▼──────────────────────┐
        │ TRAINING COMPLETE            │
        │                              │
        │ Outputs:                     │
        │ ├─ best_model.pt (saved)    │
        │ ├─ training_history.json    │
        │ ├─ loss_curves.png          │
        │ └─ metrics.txt              │
        └──────────────────────────────┘
```

---

## 4. BACKTESTING FLOW

```
┌────────────────────────────────────────┐
│ HISTORICAL DATA (test set)             │
│ ├─ 2023-2024 (unseen by model)        │
│ ├─ 1000+ trading days                 │
│ └─ 50-100 symbols                     │
└──────────────┬─────────────────────────┘
               │
        FOR EACH DAY in test_period:
        ┌──────────────┐
        │ BACKTEST     │
        │ LOOP         │
        └──────┬───────┘
               │
        ┌──────▼──────────────────────────┐
        │ 1. GET MARKET DATA               │
        │ ├─ OHLCV data for all symbols  │
        │ ├─ Calculate indicators        │
        │ └─ Build feature vectors       │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 2. MODEL PREDICTION             │
        │ FOR EACH symbol:                │
        │   ├─ Feed 60-day sequence       │
        │   ├─ Get predictions:           │
        │   │  ├─ direction (0/1/2)      │
        │   │  ├─ price_return            │
        │   │  └─ confidence              │
        │   └─ Filter (confidence > 0.7) │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 3. RISK CHECKS                  │
        │ ├─ Position size limit          │
        │ ├─ Daily loss stop              │
        │ ├─ Correlation check            │
        │ └─ Leverage limits              │
        │                                  │
        │ IF NOT allowed:                 │
        │   └─ Skip signal                │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 4. EXECUTE SIGNAL               │
        │ IF direction = 2 (UP):          │
        │   ├─ Buy position               │
        │   ├─ Size = min(capital × 0.05) │
        │   └─ Entry price = close[today] │
        │                                  │
        │ IF direction = 0 (DOWN):        │
        │   ├─ Close/short position       │
        │   └─ Exit price = close[today]  │
        │                                  │
        │ Apply:                          │
        │ ├─ Slippage (10 bps)           │
        │ ├─ Commission (0.1%)            │
        │ └─ Tax (simplified)             │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 5. UPDATE PORTFOLIO             │
        │ ├─ Update positions             │
        │ ├─ Update cash                  │
        │ ├─ Calculate P&L                │
        │ └─ Record trade                 │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 6. LOG METRICS                  │
        │ ├─ Daily equity                 │
        │ ├─ Daily returns                │
        │ ├─ Positions held               │
        │ └─ Trades executed              │
        └──────┬──────────────────────────┘
               │
        (end FOR DAY loop)
        │
        ┌──────▼──────────────────────────┐
        │ 7. CALCULATE PERFORMANCE        │
        │                                  │
        │ Equity Curve Analysis:          │
        │ ├─ Total Return                 │
        │ │  └─ (Final - Initial) / Initl│
        │ ├─ Annual Return                │
        │ │  └─ Total^(252/days)         │
        │ ├─ Daily Returns                │
        │ │  └─ (Eq[t] - Eq[t-1])/Eq[t-1]│
        │ └─ Cumulative Returns           │
        │    └─ Expanding product         │
        │                                  │
        │ Risk Metrics:                   │
        │ ├─ Volatility (std of returns)  │
        │ ├─ Max Drawdown                 │
        │ │  └─ max(Eq) - Eq[t] / max(Eq)│
        │ └─ Drawdown Duration            │
        │    └─ Days to recover from peak │
        │                                  │
        │ Return Metrics:                 │
        │ ├─ Sharpe Ratio                 │
        │ │  └─ (μ - rf) / σ              │
        │ ├─ Sortino Ratio                │
        │ │  └─ (μ - rf) / σ_down         │
        │ └─ Information Ratio            │
        │    └─ (Return - benchmark) / TE │
        │                                  │
        │ Trade Metrics:                  │
        │ ├─ Total Trades                 │
        │ ├─ Win Rate                     │
        │ │  └─ winning_trades / total    │
        │ ├─ Profit Factor                │
        │ │  └─ gross_profit / gross_loss │
        │ ├─ Avg Win / Avg Loss           │
        │ └─ Largest Win / Loss           │
        │                                  │
        └──────┬──────────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ 8. OUTPUT RESULTS               │
        │                                  │
        │ BacktestResult {                │
        │   total_return: 0.45,           │
        │   annual_return: 0.32,          │
        │   sharpe_ratio: 1.8,            │
        │   sortino_ratio: 2.1,           │
        │   max_drawdown: -0.25,          │
        │   win_rate: 0.55,               │
        │   profit_factor: 1.8,           │
        │   total_trades: 250,            │
        │   trades: [...],                │
        │   equity_curve: [...]           │
        │ }                                │
        │                                  │
        │ Saved to:                       │
        │ ├─ results/backtest.json        │
        │ ├─ results/equity_curve.csv     │
        │ └─ results/equity_curve.png     │
        └──────────────────────────────────┘
```

---

## 5. Frontend Architecture Transition

### CURRENT Frontend
```
Single HTML File (260 lines)
    │
    ├─ Header
    ├─ Download Form
    ├─ Training Form
    ├─ Progress Bar
    ├─ Model List
    └─ Basic Charts

Limitations:
├─ No state management
├─ No component reuse
├─ Hard to extend
├─ No real-time updates
└─ No portfolio tracking
```

### NEW Frontend (React)
```
React App
├─ Layout
│  ├─ Header (logo, status indicator)
│  ├─ Sidebar (navigation)
│  └─ Main Content (pages)
│
├─ Pages
│  ├─ Dashboard
│  │  ├─ PortfolioCard
│  │  │  ├─ Equity: $125,400
│  │  │  ├─ Return: +25.4%
│  │  │  ├─ Sharpe: 1.8
│  │  │  └─ Max DD: -12.3%
│  │  ├─ EquityCurve (chart)
│  │  ├─ SignalsList
│  │  │  ├─ Date | Symbol | Direction | Confidence
│  │  │  └─ 2024-01-25 | AAPL | BUY | 0.85
│  │  └─ RiskDashboard
│  │     ├─ Positions: 5 open
│  │     ├─ Daily Loss: -$500 / -$5000 limit
│  │     └─ VaR 95%: -$8,500
│  │
│  ├─ DataManager
│  │  ├─ DownloadForm
│  │  │  ├─ Source: [All / Stocks / Crypto]
│  │  │  ├─ Universe: [Small / Medium / Large]
│  │  │  ├─ Start Date: 2000-01-01
│  │  │  └─ End Date: 2024-01-25
│  │  ├─ ProgressBar
│  │  │  ├─ Progress: 35/500 symbols
│  │  │  └─ ETA: 2h 15m
│  │  └─ DataQualityCheck
│  │     ├─ Total Records: 5.2M
│  │     ├─ Missing Data: 0.2%
│  │     └─ Outliers: 15
│  │
│  ├─ Training
│  │  ├─ ConfigForm
│  │  │  ├─ Epochs: 100
│  │  │  ├─ Batch Size: 512
│  │  │  ├─ Learning Rate: 0.0005
│  │  │  └─ Start Training
│  │  ├─ TrainingMonitor
│  │  │  ├─ Epoch: 45/100
│  │  │  ├─ Loss: 0.2345
│  │  │  ├─ Val Loss: 0.2567
│  │  │  └─ LossCurve (chart)
│  │  └─ MetricsTable
│  │     ├─ Accuracy: 54.2%
│  │     ├─ Precision: 52.1%
│  │     └─ F1-Score: 0.521
│  │
│  ├─ Backtest
│  │  ├─ BacktestForm
│  │  │  ├─ Model: best_model_2024_01_25
│  │  │  ├─ Start Date: 2023-01-01
│  │  │  ├─ End Date: 2024-01-25
│  │  │  └─ Capital: $100,000
│  │  ├─ ResultsTable
│  │  │  ├─ Total Return: 45.2%
│  │  │  ├─ Sharpe: 1.82
│  │  │  └─ Win Rate: 55%
│  │  ├─ EquityCurve (chart)
│  │  ├─ DrawdownChart (chart)
│  │  └─ TradesList
│  │     ├─ Date | Symbol | Type | Entry | Exit | P&L
│  │     └─ 2023-02-15 | AAPL | BUY | 150.25 | 152.35 | +$210
│  │
│  └─ Settings
│     ├─ ModelConfig
│     │  ├─ Architecture: EfficientHybrid
│     │  ├─ Features: 40
│     │  ├─ xLSTM Layers: 2
│     │  └─ Save Settings
│     └─ SystemConfig
│        ├─ API Key: ••••••
│        ├─ Database URL: ••••••
│        └─ Save Config
│
├─ Services
│  ├─ API (axios)
│  │  ├─ GET /api/v1/portfolio/current
│  │  ├─ POST /api/v1/training/start
│  │  ├─ GET /api/v1/backtest/results
│  │  └─ GET /api/v1/data/status
│  │
│  ├─ WebSocket
│  │  ├─ Connect to /ws/signals
│  │  ├─ Receive signal updates
│  │  └─ Update SignalsList in real-time
│  │
│  └─ Storage (localStorage)
│     ├─ Save user preferences
│     ├─ Cache API responses
│     └─ Persist form data
│
└─ Components (Reusable)
   ├─ Card
   ├─ Button
   ├─ Input
   ├─ Select
   ├─ Table
   ├─ Chart (LineChart, AreaChart)
   ├─ ProgressBar
   ├─ Alert
   └─ Modal
```

**Benefits**:
- ✅ Component reusability
- ✅ State management (Zustand)
- ✅ Real-time updates (WebSocket)
- ✅ Better UX
- ✅ Easy to extend

---

## 6. Integration Points

```
SYSTEM ARCHITECTURE OVERVIEW
═════════════════════════════

┌──────────────────────────────────────────────────────────┐
│                    DATA SOURCES                          │
│  Alpaca │ Binance │ yfinance │ Index APIs               │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              DATA DOWNLOAD & STORAGE                     │
│  download_data.py → data_storage/raw/*.parquet         │
└──────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ BACKTESTING  │  │   TRAINING   │  │ WEB FRONTEND │
│              │  │              │  │              │
│ backtest.py  │  │ train_model. │  │ React App    │
│              │  │ py           │  │              │
│ ├─ Load data │  │              │  │ ├─ Dashboard │
│ ├─ Features  │  │ ├─ Load data │  │ ├─ DataMgr  │
│ ├─ Model     │  │ ├─ Features  │  │ ├─ Training │
│ ├─ Execute   │  │ ├─ Labels    │  │ ├─ Backtest │
│ ├─ Metrics   │  │ ├─ Training  │  │ └─ Settings │
│ └─ Report    │  │ ├─ Validation│  │              │
│              │  │ ├─ Testing   │  │ FastAPI API  │
│ Output:      │  │ └─ Checkpoint│  │ ├─ Portfolio │
│ ├─ Sharpe    │  │              │  │ ├─ Signals   │
│ ├─ MaxDD     │  │ Output:      │  │ ├─ Training  │
│ ├─ WinRate   │  │ best_model.pt   │ ├─ Backtest  │
│ └─ Trades    │  │              │  │ ├─ Data      │
└──────────────┘  └──────────────┘  │ └─ System    │
        │                │           │ WebSocket:  │
        │                │           │ /ws/signals │
        │                │           │             │
        └────────┬────────┴───────────┼─────────────┘
                 │                   │
        ┌────────▼───────────────────▼─────┐
        │  MONITORING & ALERTING            │
        │                                   │
        │ ├─ Training progress              │
        │ ├─ Model performance              │
        │ ├─ Portfolio health               │
        │ ├─ Risk alerts                    │
        │ └─ Error notifications            │
        └───────────────────────────────────┘
```

---

## 7. Development Workflow Diagram

```
START HERE
    │
    ▼
UNDERSTAND CURRENT PERFORMANCE
    ├─ Run backtest
    ├─ Get baseline metrics
    └─ Document results
    │
    ▼
PHASE 1: DATA IMPROVEMENTS (Week 1-2)
    ├─ Add index data fetching
    │  ├─ Create data/fetchers/index_fetcher.py
    │  ├─ Fetch SPY, QQQ, VIX, XLK, XLF, XLV
    │  └─ Test end-to-end
    │
    ├─ Implement feature selection
    │  ├─ Create data/feature_selector.py
    │  ├─ Run correlation analysis
    │  └─ Reduce 60 → 35-40 features
    │
    └─ Validate data pipeline
       ├─ No look-ahead bias
       ├─ No data leakage
       └─ Quality checks
    │
    ▼
PHASE 2: MODEL OPTIMIZATION (Week 3-4)
    ├─ Implement 3 model variants
    │  ├─ SimplexLSTM (3M params)
    │  ├─ EfficientHybrid (6M params)
    │  └─ TaskEnsemble (8M params)
    │
    ├─ Train all 3 architectures
    │  ├─ 100 epochs each
    │  ├─ Same data
    │  └─ Track metrics
    │
    ├─ Compare & benchmark
    │  ├─ Training speed
    │  ├─ Final accuracy
    │  ├─ Inference speed
    │  └─ Generalization
    │
    └─ Select best architecture
       └─ Update main training script
    │
    ▼
PHASE 3: FRONTEND REDESIGN (Week 5-7)
    ├─ Setup React project
    │  ├─ npx create-react-app
    │  ├─ Install deps
    │  └─ Configure TS
    │
    ├─ Build frontend components
    │  ├─ Dashboard page
    │  ├─ DataManager page
    │  ├─ Training page
    │  ├─ Backtest page
    │  └─ Settings page
    │
    ├─ Implement backend APIs
    │  ├─ Portfolio endpoints
    │  ├─ Training endpoints
    │  ├─ Backtest endpoints
    │  └─ WebSocket streaming
    │
    └─ Integration testing
       ├─ Frontend ↔ Backend
       ├─ Real-time updates
       └─ Error handling
    │
    ▼
PHASE 4: ADVANCED FEATURES (Week 8-10)
    ├─ Curriculum learning
    ├─ Online learning
    ├─ Attention visualization
    └─ Model ensembling
    │
    ▼
PHASE 5: PRODUCTION (Week 11+)
    ├─ Stress testing
    ├─ Cloud deployment
    ├─ Monitoring setup
    └─ Documentation
    │
    ▼
DONE ✅

```

