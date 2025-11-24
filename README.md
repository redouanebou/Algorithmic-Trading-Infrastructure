🤖 Hybrid Quantitative Trading Infrastructure: XGBoost & Autoencoder

An institutional-grade algorithmic trading system leveraging ensemble machine learning for regime-based market prediction. Engineered with a "Safety-First" architecture, combining XGBoost for trend classification and Deep Learning (Autoencoders) for anomaly detection.

📖 Overview
This project represents a complete end-to-end quantitative trading pipeline, built from scratch to solve the hardest problems in algorithmic trading: Overfitting, Look-ahead Bias, and Market Regime Shifts.

Unlike traditional bots that rely on static indicators, this system utilizes a Hybrid Regime Architecture. It dynamically classifies market conditions into "High Volatility" (Breaker) and "Low Volatility" (Scalper) regimes, deploying specialized XGBoost models optimized for each state. To ensure robustness, a Keras Autoencoder acts as a gatekeeper, filtering out anomalous market data (statistically distinct from training distributions) to prevent trading during unpredictable "Black Swan" events or data corruptions.

The system features a Live Bridge to MetaTrader 5 (MT5) for real-time low-latency execution and a rigorous Event-Driven Backtester designed to expose realistic performance by accounting for variable spreads, slippage, and commissions.

🧠 Core Architecture
1. The "Hybrid Brain" (Ensemble Learning)
The system does not rely on a single model. It employs a "Specialist" approach:

Regime Filter: A causal, leak-proof volatility filter (comparing current ATR to long-term moving averages) splits the market into two states.

Models:

XGBoost Regressors/Classifiers: Trained specifically on Triple Barrier Labels (Take Profit, Stop Loss, Time-out) to predict directional probability.

Autoencoder (TensorFlow/Keras): A deep neural network trained to reconstruct "normal" market features. High Reconstruction Error signals an anomaly, triggering a "No-Trade" safety lock.

2. Forensic Data Engineering
The pipeline enforces strict Causal Integrity:

Anti-Lookahead Protocols: All features are shifted (t-1) prior to inference. The regime filter uses a proven causal logic (comparing ATR[t-1] vs Mean(ATR[t-480...t-1])) to ensure zero leakage of future data.

Feature Engineering: Includes statistical z-scores, volume regime analysis, and custom price action derivatives (e.g., F-35 Pullback logic).

3. Live Execution Engine (MT5 Bridge)
Real-Time Synchronization: Connects to MetaTrader 5 via API to fetch OHLCV data, sync positions, and execute orders with sub-second latency.

Dynamic Risk Management: Automatically calculates position size based on account equity and volatility-adjusted Stop Loss distances (ATR Multiples).

Self-Healing: Handles connection drops, API timeouts, and data gaps autonomously.

🛠️ Technical Stack
Core Language: Python 3.10+

Machine Learning:

XGBoost (Gradient Boosting for Directional Prediction)

TensorFlow / Keras (Autoencoders for Anomaly Detection)

Scikit-Learn (Data Scaling & Pipelines)

Data Engineering:

Pandas & NumPy (High-performance Vectorization)

Pandas-TA & TA-Lib (Technical Indicators)

Numba (JIT Compilation for Backtesting loops)

Execution:

MetaTrader5 (Python API Integration)

🚀 Workflow
Ingestion: Raw M1/Tick data is ingested, cleaned, and resampled (e.g., to M3 or M15) with gap-filling logic.

Training (The "Trainer"):

Generates Triple Barrier Labels to define "True" buy/sell opportunities.

Splits data into Train/Validation sets (strictly time-series split).

Trains the Autoencoder to learn the "manifold" of normal market data.

Trains XGBoost models on the latent features + technical indicators.

Validation (The "Backtester"):

Simulates trading on unseen data using an event-driven loop.

Applies realistic costs (Spread + Commission).

Logs every decision for "Forensic" review.

Live Trading (The "Master"):

Waits for candle closure -> Fetches Data -> Generates Features -> Queries Models -> Executes Trade.

⚠️ Methodology: "Honest" AI
This project was born out of a frustration with "fake" backtests. Many trading bots fail because they accidentally peek at future data (e.g., using the High of the current candle to decide entry).This system is built to be paranoid. It assumes the market is trying to trick the model. By enforcing strict causal feature engineering and using an Autoencoder to reject "weird" data, we prioritize Capital Preservation over hypothetical gains.

Disclaimer: This software is for educational and research purposes only. Algorithmic trading involves significant risk of financial loss.
