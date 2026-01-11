<div align="center">

# 🛡️ Sentinel: Hybrid Quantitative Trading Infrastructure!

### Regime-Adaptive Institutional Execution Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Autoencoders-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-EB4034?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge)

<p align="center">
  <em>An institutional-grade algorithmic ecosystem engineered to solve the "Trilogy of Failure": Look-Ahead Bias, Overfitting, and Regime Shifts.</em>
</p>

</div>

---

## 📖 Executive Summary

**Sentinel** is a comprehensive Quantitative Research & Execution Pipeline bridging the gap between backtest theory and live market execution. Unlike standard retail bots, Sentinel employs a **Hybrid AI Architecture** that dynamically classifies market regimes and deploys specialized "Agent" models.

It integrates a **Forensic Data Pipeline** ensuring strict causal integrity ($T-1$), making all backtest signals theoretically executable live via the MetaTrader 5 bridge.

---

## 🧠 Core Architecture  :  The "Hybrid Brain"

The decision engine uses a **"Committee of Experts"** structure:

```mermaid
graph TD
    subgraph Ingestion
        A[Live Market Data] -->|Fetch & Clean| B(Feature Engineering)
    end

    subgraph The Hybrid Brain
        B --> C{Regime Filter}
        C -->|Low Volatility| D[Scalper Agent]
        C -->|High Volatility| E[Breaker Agent]
        D & E --> F[XGBoost Directional Alpha]
        B --> G[Autoencoder Anomaly Detector]
    end

    subgraph Execution
        F --> H{Consensus Logic}
        G -->|High Reconstruction Error| I[❌ HARD LOCK: Black Swan]
        H -->|Prediction > Threshold| J[✅ Dynamic Sizing & Execution]
    end
```

---

### 1. Ensemble Logic

* **Regime Detection:** Causal volatility filter (Short-term vs. Long-term ATR) splits market into Scalper vs. Breaker regimes.
* **Directional Alpha (XGBoost):** Classifiers trained on Triple Barrier Labels (TP, SL, Time-out).
* **Anomaly Detection (Autoencoder):** Neural network reconstructs "normal" market patterns; high error triggers a safety lock.

### 2. Forensic Data Engineering ("Honest AI")

* **Anti-Leakage Protocols:** All features strictly time-shifted (shift(1)) prior to inference.
* **Advanced Features:** Statistical Z-Scores, Volume Regimes, Price Action derivatives.
* **Robust Normalization:** Adaptive scaling fitted only on training folds.

### 3. Live Execution Engine (MT5 Bridge)

* **Real-Time Sync:** Threaded loop syncing candle closures.
* **Dynamic Risk:** Position sizing based on Account Equity % and Volatility-Adjusted Stop Loss.
* **Self-Healing:** Handles API timeouts and connection drops.

---

## 🛠️ Technical Stack

| Component    | Technology       | Description                                      |
| ------------ | ---------------- | ------------------------------------------------ |
| Core Logic   | Python 3.10+     | Async event loop & orchestration                 |
| Alpha Model  | XGBoost          | Gradient Boosting for directional classification |
| Safety Model | TensorFlow/Keras | Autoencoder for OOD detection                    |
| Data Ops     | Pandas / NumPy   | High-performance vectorization                   |
| Broker API   | MetaTrader 5     | Direct DMA/STP execution bridge                  |
| Optimization | Numba            | JIT compilation for backtesting loops            |

---

## 🧬 Forensic Data Pipeline (ETL)

* **Gap Filling:** Forward-fill missing M1 candles.
* **Resampling Engine:** Converts M1 to M3/M10 timeframes with precise aggregation.
* **Latency Handling:** Aligns candle closures with broker server time to prevent Look-Ahead Bias.

---

## 📂 Project Structure

```bash
Algorithmic-Trading-Infrastructure/
├── data_engineering.py     # ETL & forensic cleaning
├── trainer.py              # Model training (XGBoost + Autoencoder)
├── backtester.py           # Event-driven validation engine
├── master.py               # Live execution bridge (MT5)
└── README.md               # Documentation
```

---

## 🚀 Workflow: Research to Production

1. **Forensic Ingestion:** Clean & resample raw tick data.
2. **Trainer:**

   * Generates "Honest" Triple Barrier Labels.
   * Trains Autoencoder on safe market manifolds.
   * Trains XGBoost Agents on latent features.
3. **Backtester:** Event-driven simulations on out-of-sample data, including spread/slippage.
4. **Master:** Deploys ensemble to live trading.

---

## ⚠️ Reality Check Philosophy

99% of trading bots fail due to data leakage. Sentinel is designed for robustness:

* Autoencoder rejects anomalous data.
* Strict causal feature engineering prevents look-ahead bias.

<div align="center">
  **Disclaimer:** This software is for educational & research purposes only. Algorithmic trading involves significant financial risk.
</div>

Engineered by Redouane Boundra.
