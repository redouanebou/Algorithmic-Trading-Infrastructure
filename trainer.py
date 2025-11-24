import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import MetaTrader5 as mt5  
from datetime import datetime, timezone  
from numba import njit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
import sys
import warnings
from typing import Tuple, List, Dict
import json
import os
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

MODEL_DIR = "test"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
SYMBOL = "GBPUSD"

PIP_SIZE = 0.0001
SL_PIPS = 5.0
TP_PIPS = 5.0
RR_STRATEGY = TP_PIPS / SL_PIPS
SL_TARGET = SL_PIPS * PIP_SIZE
TP_TARGET = TP_PIPS * PIP_SIZE
LOOKAHEAD_PERIOD = 400

TRAIN_START = '2010-01-01'
TRAIN_END = '2021-12-31'
VAL_START = '2022-01-01'
VAL_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2025-08-13' 
def fetch_mt5_data(symbol: str, timeframe_mt5, start_dt_utc, end_dt_utc) -> pd.DataFrame:
    """
    Connects to MT5, fetches data, and returns a clean DataFrame
    matching the CSV format.
    """
    print(f"Connecting to MT5 to fetch {symbol} {timeframe_mt5} data...")
    
    if not mt5.initialize():
        print(f"FATAL ERROR: initialize() failed. Error code: {mt5.last_error()}", file=sys.stderr)
        print("KHASSEK T7EL L-TERMINAL DYAL MT5 O T'LOGGI", file=sys.stderr)
        mt5.shutdown()
        sys.exit(1)
    
    print(f"MT5 Connection Initialized. Fetching data from {start_dt_utc} to {end_dt_utc}...")
    
    try:
        rates = mt5.copy_rates_range(symbol, timeframe_mt5, start_dt_utc, end_dt_utc)
    except Exception as e:
        print(f"FATAL ERROR: Failed to fetch rates. Error: {e}", file=sys.stderr)
        mt5.shutdown()
        sys.exit(1)
        
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print(f"FATAL ERROR: No data returned from MT5 for {symbol}. Check symbol and date range.", file=sys.stderr)
        sys.exit(1)
        
    df = pd.DataFrame(rates)
    
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    
    df = df.set_index('datetime')
    
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Tick volume' 
    })
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Tick volume']
    df = df[required_cols]
    
    df = df[df.index.dayofweek < 5]
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Successfully fetched {len(df)} candles from MT5.")
    return df

@njit
def create_triple_barrier_labels(
    open_prices: np.ndarray, 
    high_prices: np.ndarray, 
    low_prices: np.ndarray, 
    n: int, 
    lookahead: int, 
    tp_points: float, 
    sl_points: float
) -> Tuple[np.ndarray, np.ndarray]:
    
    buy_labels = np.full(n, 2, dtype=np.int8)
    sell_labels = np.full(n, 2, dtype=np.int8) 

    for i in range(n - lookahead):
        entry_price_buy = open_prices[i]
        tp_level_buy = entry_price_buy + tp_points
        sl_level_buy = entry_price_buy - sl_points
        entry_price_sell = open_prices[i]
        tp_level_sell = entry_price_sell - tp_points
        sl_level_sell = entry_price_sell + sl_points

        for k in range(1, lookahead + 1):
            high_k = high_prices[i + k]
            low_k = low_prices[i + k]
            if buy_labels[i] == 2: 
                if low_k <= sl_level_buy: buy_labels[i] = 0 
                elif high_k >= tp_level_buy: buy_labels[i] = 1
            if sell_labels[i] == 2: 
                if high_k >= sl_level_sell: sell_labels[i] = 0 
                elif low_k <= tp_level_sell: sell_labels[i] = 1 
            if buy_labels[i] != 2 and sell_labels[i] != 2:
                break
    return buy_labels, sell_labels

def create_volume_features(df: pd.DataFrame, prefix: str) -> List[pd.Series]:
    """
    [MONGO] Creates our new, god-tier VOLUME features.
    """
    print(f"   Calculating V26 'Volume' features for {prefix}...")
    features_list = []
    close = df['Close']
    volume = df['Tick volume']
    
    vol_mean = volume.rolling(window=200).mean() 
    vol_std = volume.rolling(window=200).std()
    vol_zscore = (volume - vol_mean) / (vol_std + 1e-9)
    features_list.append(vol_zscore.rename(f'{prefix}_vol_zscore_200'))
    
    vol_ema_fast = ta.ema(volume, 5)
    vol_ema_slow = ta.ema(volume, 50)
    vol_regime = (vol_ema_fast - vol_ema_slow).rename(f'{prefix}_vol_regime_5_50')
    features_list.append(vol_regime)
    
    price_mom = close.diff(5)
    price_x_vol = (price_mom * volume).rename(f'{prefix}_price_x_vol')
    features_list.append(price_x_vol)
    
    return features_list


def create_breaker_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    [MONGO] V26: "BREAKER" FEATURES (HIGH VOL)
    V21 (Momentum) + V26 (Volume)
    """
    print("Calculating V26 'Breaker' (High-Vol) features...")
    features_list = []
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    atr_14 = ta.atr(high, low, close, 14).rename('atr_14')
    features_list.append(atr_14)
    adx_df = ta.adx(high, low, close, 14)
    if isinstance(adx_df, pd.DataFrame) and 'ADX_14' in adx_df.columns:
        features_list.append(adx_df['ADX_14'].rename('adx_14'))
    else: features_list.append(pd.Series(np.nan, index=df.index, name='adx_14'))
    features_list.append(ta.rsi(close, 14).rename('rsi_14'))
    features_list.append(close.diff(5).rename('mom_5'))
    features_list.append(close.diff(20).rename('mom_20'))
    features_list.append((close.diff(10) / (atr_14 + 1e-9)).rename('norm_mom_10'))
    features_list.append(pd.Series(np.sin(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_sin'))
    features_list.append(pd.Series(np.cos(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_cos'))
    
    features_list.extend(create_volume_features(df, prefix='breaker'))
    
    final_features_df = pd.concat(features_list, axis=1).add_prefix('breaker_')
    feature_names = final_features_df.columns.tolist()
    print(f"Created {len(feature_names)} V26 'Breaker' features.")
    return final_features_df, feature_names

def create_scalper_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    [MONGO] V26: "SCALPER" FEATURES (LOW VOL)
    V21 (Mean Reversion) + V26 (Volume)
    """
    print("Calculating V26 'Scalper' (Low-Vol) features...")
    features_list = []
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    atr_14 = ta.atr(high, low, close, 14).rename('atr_14')
    features_list.append(atr_14)
    for w in [20, 60]:
        roll_mean, roll_std = close.rolling(window=w).mean(), close.rolling(window=w).std()
        features_list.append(((close - roll_mean) / (roll_std + 1e-9)).rename(f'zscore_{w}'))
    for w in [60, 120]:
        roll_high, roll_low = high.rolling(window=w).max(), low.rolling(window=w).min()
        roll_range = roll_high - roll_low
        features_list.append(((close - roll_low) / (roll_range + 1e-9)).rename(f'pos_in_range_{w}'))
    features_list.append(ta.rsi(atr_14, 14).rename('atr_rsi_14'))
    features_list.append(pd.Series(np.sin(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_sin'))
    features_list.append(pd.Series(np.cos(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_cos'))

    features_list.extend(create_volume_features(df, prefix='scalper'))
    
    final_features_df = pd.concat(features_list, axis=1).add_prefix('scalper_')
    feature_names = final_features_df.columns.tolist()
    print(f"Created {len(feature_names)} V26 'Scalper' features.")
    return final_features_df, feature_names

def train_specialist_models(
    X_train: pd.DataFrame, 
    y_train_buy: pd.Series,
    y_train_sell: pd.Series,
    X_val: pd.DataFrame, 
    y_val_buy: pd.Series,
    y_val_sell: pd.Series,
    model_suffix: str 
):
    
    print(f"\n--- {model_suffix.upper()} SPECIALIST: Training Anomaly Filter ---")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) 
    n_features = X_train_scaled.shape[1]
    
    input_layer = Input(shape=(n_features,))
    encoded = Dense(n_features // 2, activation='relu')(input_layer)
    encoded = Dense(4, activation='relu')(encoded) 
    decoded = Dense(n_features // 2, activation='relu')(encoded)
    decoded = Dense(n_features, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print(f"Training {model_suffix} Autoencoder on {len(X_train_scaled)} TRAIN samples...")
    autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=20, batch_size=256, shuffle=True,
        validation_data=(X_val_scaled, X_val_scaled), 
        verbose=0 
    )
    print("Autoencoder trained.")
    
    print("Calculating reconstruction error...")
    pred_train = autoencoder.predict(X_train_scaled, verbose=0)
    pred_val = autoencoder.predict(X_val_scaled, verbose=0) 
    error_train = np.mean(np.power(X_train_scaled - pred_train, 2), axis=1)
    error_val = np.mean(np.power(X_val_scaled - pred_val, 2), axis=1) 
    
    X_train_final = X_train.join(pd.Series(error_train, index=X_train.index, name='recon_error'))
    X_val_final = X_val.join(pd.Series(error_val, index=X_val.index, name='recon_error'))
    final_feature_names = X_train_final.columns.tolist()

    print(f"\n--- {model_suffix.upper()} SPECIALIST: Training XGBoost ---")
    
    print(f"\n--- Training {model_suffix.upper()} BUY Model ---")
    model_buy = xgb.XGBClassifier(
        objective='multi:softmax', num_class=3, n_jobs=-1, random_state=42, 
        max_depth=6, n_estimators=250, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8
    )
    model_buy.fit(X_train_final, y_train_buy) 
    
    print(f"\n--- Training {model_suffix.upper()} SELL Model ---")
    model_sell = xgb.XGBClassifier(
        objective='multi:softmax', num_class=3, n_jobs=-1, random_state=42, 
        max_depth=6, n_estimators=250, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8
    )
    model_sell.fit(X_train_final, y_train_sell)
    
    print(f"\n--- V26 (RR={RR_STRATEGY}) {model_suffix.upper()} THRESHOLD-FINDING REPORT ---")
    
    buy_preds_val = model_buy.predict(X_val_final)
    sell_preds_val = model_sell.predict(X_val_final)
    buy_probas_val = model_buy.predict_proba(X_val_final)
    sell_probas_val = model_sell.predict_proba(X_val_final)

    print(f"\n--- {model_suffix.upper()} BUY MODEL CONFUSION MATRIX ({VAL_START}-{VAL_END}) ---")
    print(confusion_matrix(y_val_buy, buy_preds_val, labels=[0, 1, 2]))
    print("Labels: 0=Loss, 1=Win, 2=Timeout")

    print(f"\n--- {model_suffix.upper()} BUY MODEL 'SNIPER' TEST (Predicts 1) ---")
    print("Threshold | Win Rate | Trades Taken")
    print("--------------------------------------")
    
    best_buy_thresh = 0.5
    best_buy_wr = 0.0
    proba_of_win = buy_probas_val[:, 1] 
    
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]:
        trades_taken_mask = (proba_of_win > thresh)
        trades_taken = trades_taken_mask.sum()
        if trades_taken == 0:
            print(f"   {thresh:.2f}    |   ---    | 0")
            continue
        actual_outcomes = y_val_buy[trades_taken_mask] 
        trades_won = (actual_outcomes == 1).sum()
        win_rate = (trades_won / trades_taken) if trades_taken > 0 else 0.0
        print(f"   {thresh:.2f}    |   {win_rate * 100:.2f}%  | {trades_taken}")
        if win_rate > best_buy_wr:
            best_buy_wr = win_rate
            best_buy_thresh = thresh

    print(f"\n--- {model_suffix.upper()} SELL MODEL CONFUSION MATRIX ({VAL_START}-{VAL_END}) ---")
    print(confusion_matrix(y_val_sell, sell_preds_val, labels=[0, 1, 2]))
    print("Labels: 0=Loss, 1=Win, 2=Timeout")

    print(f"\n--- {model_suffix.upper()} SELL MODEL 'SNIPER' TEST (Predicts 1) ---")
    print("Threshold | Win Rate | Trades Taken")
    print("--------------------------------------")
    
    best_sell_thresh = 0.5
    best_sell_wr = 0.0
    proba_of_win = sell_probas_val[:, 1] 
    
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]:
        trades_taken_mask = (proba_of_win > thresh)
        trades_taken = trades_taken_mask.sum()
        if trades_taken == 0:
            print(f"   {thresh:.2f}    |   ---    | 0")
            continue
        actual_outcomes = y_val_sell[trades_taken_mask] 
        trades_won = (actual_outcomes == 1).sum()
        win_rate = (trades_won / trades_taken) if trades_taken > 0 else 0.0
        print(f"   {thresh:.2f}    |   {win_rate * 100:.2f}%  | {trades_taken}")
        if win_rate > best_sell_wr:
            best_sell_wr = win_rate
            best_sell_thresh = thresh
            
    print(f"\nSaving V26 {model_suffix.upper()} models and assets...")
    
    buy_model_path = os.path.join(MODEL_DIR, f"v26_buy_model_{model_suffix}.json")
    sell_model_path = os.path.join(MODEL_DIR, f"v26_sell_model_{model_suffix}.json")
    model_buy.save_model(buy_model_path)
    model_sell.save_model(sell_model_path)
    
    autoencoder_path = os.path.join(MODEL_DIR, f"v26_autoencoder_{model_suffix}.keras")
    autoencoder.save(autoencoder_path)
    
    scaler_path = os.path.join(MODEL_DIR, f"v26_scaler_{model_suffix}.pkl")
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
        
    features_path = os.path.join(MODEL_DIR, f"v26_features_{model_suffix}.json")
    with open(features_path, "w") as f: json.dump(final_feature_names, f, indent=2)
        
    thresholds = {'buy_threshold': best_buy_thresh, 'sell_threshold': best_sell_thresh}
    thresh_path = os.path.join(MODEL_DIR, f"v26_thresholds_{model_suffix}.json")
    with open(thresh_path, "w") as f: json.dump(thresholds, f, indent=2)

    print(f"\n✅ --- V26 {model_suffix.upper()} SPECIALIST TRAINER COMPLETE --- ✅")

def main():
    print(f"---  V26 VOLUME HUNTER TRAINER (RR={RR_STRATEGY}) [MT5 DATA] ---")
    
    print(f"Connecting to MT5 to fetch {SYMBOL} data...")
    TIMEZONE = timezone.utc
    
    FETCH_START_DT = datetime.strptime(TRAIN_START, '%Y-%m-%d').replace(tzinfo=TIMEZONE)
    FETCH_END_DT = datetime.strptime(TEST_END, '%Y-%m-%d').replace(tzinfo=TIMEZONE)

    try:
        df_m3 = fetch_mt5_data(
            SYMBOL, 
            mt5.TIMEFRAME_M3, 
            FETCH_START_DT, 
            FETCH_END_DT
        )
        df_m3 = df_m3.sort_index().dropna()
    except Exception as e:
        print(f"FATAL ERROR: Failed to fetch data from MT5. Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded {len(df_m3)} M3 candles from MT5.")

    print(f"Generating V19 (RR={RR_STRATEGY}) TRIPLE BARRIER labels (TP={TP_PIPS}p, SL={SL_PIPS}p, Timeout={LOOKAHEAD_PERIOD} candles)...")
    buy_labels, sell_labels = create_triple_barrier_labels(
        open_prices=df_m3['Open'].to_numpy(dtype=np.float64),
        high_prices=df_m3['High'].to_numpy(dtype=np.float64),
        low_prices=df_m3['Low'].to_numpy(dtype=np.float64),
        n=len(df_m3),
        lookahead=LOOKAHEAD_PERIOD, 
        tp_points=TP_TARGET,
        sl_points=SL_TARGET
    )
    df_m3['Label_Buy'] = buy_labels
    df_m3['Label_Sell'] = sell_labels
    
    print("Buy Label Counts (0=SL, 1=TP, 2=Timeout):")
    print(pd.Series(buy_labels).value_counts())
    print("Sell Label Counts (0=SL, 1=TP, 2=Timeout):")
    print(pd.Series(sell_labels).value_counts())

    breaker_features, breaker_feature_names = create_breaker_features(df_m3)
    scalper_features, scalper_feature_names = create_scalper_features(df_m3)

    print("Applying Anti-Look-Ahead-Bias Protocol (shift(1)) to ALL features...")
    breaker_features_shifted = breaker_features.shift(1)
    scalper_features_shifted = scalper_features.shift(1)
    
    model_data = df_m3[['Label_Buy', 'Label_Sell']]
    model_data = model_data.join(breaker_features_shifted)
    model_data = model_data.join(scalper_features_shifted)
    
    print("Calculating V22 *LEAK-FREE* 'Volatility Regime' filter (DAN V-Final Fix)...")
    
    atr_14_t_minus_1 = breaker_features_shifted['breaker_atr_14']
    
    long_term_atr = atr_14_t_minus_1.rolling(window=480, min_periods=480).mean() 
    
    model_data['regime'] = (atr_14_t_minus_1 > long_term_atr).astype(int) 
 
    
    print("Volatility Regime Counts:")
    print(model_data['regime'].value_counts())
    
    model_data = model_data.dropna()
    if LOOKAHEAD_PERIOD > 0:
        model_data = model_data.iloc[:-LOOKAHEAD_PERIOD]
    print(f"Final clean V26 dataset size: {len(model_data)} rows")

    print(f"Splitting V22 data into Train ({TRAIN_START}-{TRAIN_END}) and Validation ({VAL_START}-{VAL_END}) sets...")
    train_data = model_data.loc[TRAIN_START:TRAIN_END]
    val_data = model_data.loc[VAL_START:VAL_END] 
    

    if train_data.empty or val_data.empty:
        print("FATAL ERROR: Train or Validation data is empty. Check dates.", file=sys.stderr)
        sys.exit(1)
        
    print("Splitting Train/Val/Test data into High/Low Volatility regimes...")
    
    train_high_vol = train_data[train_data['regime'] == 1]
    train_low_vol = train_data[train_data['regime'] == 0]
    val_high_vol = val_data[val_data['regime'] == 1] 
    val_low_vol = val_data[val_data['regime'] == 0] 

    if not train_high_vol.empty and not val_high_vol.empty:
        train_specialist_models(
            train_high_vol[breaker_feature_names], 
            train_high_vol['Label_Buy'],
            train_high_vol['Label_Sell'],
            val_high_vol[breaker_feature_names],
            val_high_vol['Label_Buy'],
            val_high_vol['Label_Sell'],
            "high_vol_breaker"
        )
    else:
        print("--- SKIPPING HIGH VOL (BREAKER) MODELS (No data) ---")

    if not train_low_vol.empty and not val_low_vol.empty:
        train_specialist_models(
            train_low_vol[scalper_feature_names], 
            train_low_vol['Label_Buy'],
            train_low_vol['Label_Sell'],
            val_low_vol[scalper_feature_names], 
            val_low_vol['Label_Buy'],
            val_low_vol['Label_Sell'],
            "low_vol_scalper"
        )
    else:
        print("--- SKIPPING LOW VOL (SCALPER) MODELS (No data) ---")

    print(f"\n✅---Redouane Boundra VOLUME HUNTER TRAINER [MT5 DATA] COMPLETE --- ✅")
    print(f"[MONGO] This trainer did NOT touch the Test Set ({TEST_START}-{TEST_END}).")

if __name__ == "__main__":
    main()