# 🧠 Neural Network Logic: The "brain" of the Trading System

This system uses a state-of-the-art **Hybrid xLSTM-Transformer** architecture designed to capture both short-term market volatility and long-term trend dependencies.

---

## 🏗️ 1. Input Layer (The Senses)
The model treats the market like a video game, observing **102 distinct features** for every single timestamp (day).

### Inputs (102 Features per Step):
*   **Price Action (OHLCV)**: Open, High, Low, Close, Volume.
*   **Trend Indicators**: EMA, MACD, ADX, Parabolic SAR, Aroon.
*   **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI, ROC.
*   **Volatility Indicators**: ATR, Bollinger Bands, Keltner Channels.
*   **Volume Indicators**: OBV, VWAP, CMF, Force Index.
*   **Time Embeddings**: Day of week, Month of year (Cyclical encoding).

**Shape**: `[Batch_Size, Sequence_Length (60 days), Features (102)]`
*   *Analogy*: It looks at the last 60 days of data at once, analyzing 102 variables for each of those days.

---

## 🧠 2. Hidden Layers (The Brain)
The input flows into two parallel "thinking" systems that fuse together.

### Path A: xLSTM (Extended Long Short-Term Memory)
*   **Role**: **Short-Term Memory & Trend Stability**
*   **Specs**: 
    *   **3 Layers**
    *   **512 Hidden Units**
*   **Why xLSTM?**: Unlike standard LSTMs, xLSTM uses **Exponential Gating**. This allows it to:
    1.  Remember crucial events from a long time ago (perfect memory).
    2.  Forget noise almost instantly.
    3.  Prevent "Gradient Explosion" (math errors) during training.

### Path B: Transformer Encoder
*   **Role**: **Pattern Recognition & Global Context**
*   **Specs**:
    *   **3 Layers**
    *   **8 Attention Heads**
    *   **256 Embedding Dimension**
*   **Why Transformer?**: It uses **Self-Attention** to ask questions like: *"How similar is today's price action to the crash we saw 45 days ago?"* It spots repeating patterns regardless of when they happened.

### Fusion Layer
*   The output of the xLSTM (Time-aware) and Transformer (Pattern-aware) are concatenated.
*   **Fusion Dimension**: 256 Units.
*   **Mechanism**: An attention mechanism decides which path (LSTM vs Transformer) is more important for the current specific market situation.

**Total Parameters**: **~27.1 Million** (A very large, capacity-rich model).

---

## 🎯 3. Output Layer (The Predictions)
The model doesn't just guess "Up" or "Down". It performs **Multi-Task Learning** (6 simultaneous predictions).

### 1. Price Prediction Head (Regression)
*   Predicts the exact **% Return** for:
    *   Next 1 Day
    *   Next 4 Days
    *   Next 24 Days

### 2. Direction Classification Head (Classifier)
*   Predicts the **Trajectory**:
    *   `0`: Down
    *   `1`: Neutral / Sideways
    *   `2`: Up

### 3. Position Sizing Head (Risk Engine)
*   Recommends accurate trade allocation:
    *   `0`: Sell / Short
    *   `1`: Hold / Cash
    *   `2`: Buy / Long

### 4. Volatility Forecast (Risk Management)
*   Predicts future market volatility (ATR). If volatility is predicted to be high, the system automatically reduces trade size.

### 5. Confidence Score
*   The model outputs a score (0-1) indicating how "sure" it is. We filter out any trade with confidence < 0.7.

---

## 🚀 Summary
*   **Input**: 60 days x 102 indicators.
*   **Brain**: xLSTM (Time) + Transformer (Pattern) Hybrid.
*   **Output**: Future Return, Direction, Volatility, and Trade Confidence.
*   **Goal**: To be right more often than wrong, and win big when right.
