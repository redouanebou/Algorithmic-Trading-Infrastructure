🛡️ Sentinel: Hybrid Quantitative Trading Infrastructure
An institutional-grade algorithmic trading ecosystem designed for regime-based market prediction. Features a "Safety-First" architecture combining Gradient Boosting (XGBoost) for directional alpha and Deep Learning Autoencoders for anomaly detection.

📖 Executive Summary
Sentinel is not just a trading bot; it is a comprehensive Quantitative Research & Execution Pipeline engineered to solve the "Trilogy of Failure" in algorithmic trading: Look-ahead Bias, Overfitting, and Market Regime Shifts.

While most retail bots rely on lagging indicators, Sentinel employs a Hybrid AI Architecture. It dynamically classifies market states (e.g., Low Volatility vs. High Volatility) and deploys specialized "Agent" models. Crucially, it integrates a Forensic Data Pipeline that enforces strict causal integrity, ensuring that every backtest signal is theoretically executable in a live environment.

The system features a robust Live Bridge to MetaTrader 5 (MT5), handling real-time data ingestion, signal inference, and sub-second order execution with dynamic risk sizing.

🧠 Core Architecture
1. The "Hybrid Brain" (Ensemble Logic)
The decision engine does not rely on a single monolithic model. Instead, it uses a "Committee of Experts" approach:

Regime Detection: A causal volatility filter (comparing short-term vs. long-term ATR) splits the market into distinct regimes (e.g., Scalper vs. Breaker).

Directional Alpha (XGBoost): Specialized classifiers trained on Triple Barrier Labels (Fixed TP, SL, and Time-out horizons) to predict price direction with high probability.

Anomaly Detection (Autoencoder): A deep neural network (Keras) trained to reconstruct "normal" market features. High Reconstruction Error signals a statistical anomaly or "Black Swan" event, triggering a hard "No-Trade" lock to preserve capital.

2. Forensic Data Engineering
The pipeline is built on a philosophy of "Honest AI":

Anti-Leakage Protocols: All feature engineering is strictly time-shifted (t-1) prior to inference.

Advanced Feature Sets: Includes statistical z-scores, volume regime analysis, and custom price action derivatives (e.g., F-35 Pullback logic for trend continuation).

Robust Normalization: Uses adaptive scaling to ensure models remain stable across different price levels and eras.

3. Live Execution Engine (MT5 Bridge)
Real-Time Synchronization: A threaded loop connects to the MetaTrader 5 API, syncing candle closures to within milliseconds.

Dynamic Risk Management: Position sizing is calculated dynamically based on account equity and volatility-adjusted Stop Loss distances (ATR Multiples).

Resilience: Self-healing logic handles API timeouts, connection drops, and data gaps without crashing.

🛠️ Technical Stack
Core: Python 3.10+

Machine Learning:

XGBoost (Gradient Boosting for Classification/Regression)

TensorFlow / Keras (Autoencoders for Unsupervised Learning)

Scikit-Learn (Pipelines & Scaling)

Data Science:

Pandas & NumPy (High-performance Vectorization)

Pandas-TA (Technical Analysis Library)

Numba (JIT Compilation for high-speed backtesting loops)

Integration:

MetaTrader5 (Python API for Broker connectivity)

🚀 Workflow: From Research to Production
Forensic Ingestion: Raw tick/M1 data is ingested, cleaned, and resampled with gap-filling logic to create a pristine dataset.

Model Training ("The Trainer"):

Generates "Honest" labels based on future outcomes (Triple Barrier Method).

Trains the Autoencoder to learn the manifold of "safe" market conditions.

Trains XGBoost agents on the latent features + technical indicators.

Validation ("The Backtester"):

Runs an event-driven simulation on Unseen Out-of-Sample Data.

Accounts for Spread, Slippage, and Commissions to simulate realistic P&L.

Live Trading ("The Master"):

Monitors market state -> Fetches Real-time Data -> Generates Features -> Queries the "Brain" -> Executes Orders.

⚠️ The "Reality Check" Philosophy
This project was born from the realization that 99% of trading bots fail due to Data Leakage (peeking at future data). Sentinel is built to be paranoid. It assumes the market is adversarial. By using an Autoencoder to reject "weird" data and enforcing strict causal feature engineering, the system prioritizes Robustness over hypothetical hyper-optimized returns.

Disclaimer: This software is for educational and research purposes only. Algorithmic trading involves significant risk of financial loss. of financial loss.
