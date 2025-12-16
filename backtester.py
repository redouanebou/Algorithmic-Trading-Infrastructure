import pandas as pd
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import sys
import warnings
from typing import Tuple, List, Dict
import json
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import MetaTrader5 as mt5  
from datetime import datetime, timezone

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

MODEL_DIR = "AUD_MT5"

TEST_START = '2024-01-01'
TEST_END = '2025-08-13'

FETCH_START_DATE = '2010-01-01'

INITIAL_BALANCE = 100000.0
RISK_PER_TRADE_PCT = 0.005
COMMISSION_PER_LOT = 7.0
CONTRACT_SIZE = 100000.0
MAX_LOT_SIZE = 100.0
MIN_LOT_SIZE = 0.01
PIP_SIZE = 0.0001
SL_PIPS = 5.0
TP_PIPS = 5.0
RR_STRATEGY = TP_PIPS / SL_PIPS
SL_TARGET = SL_PIPS * PIP_SIZE
TP_TARGET = TP_PIPS * PIP_SIZE
LOOKAHEAD_PERIOD = 400 

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

def create_volume_features(df: pd.DataFrame, prefix: str) -> List[pd.Series]:
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
    return final_features_df, feature_names

def create_scalper_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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
    return final_features_df, feature_names

def calculate_lot_size(equity: float, sl_pips: float) -> float:
    try:
        risk_amount = equity * RISK_PER_TRADE_PCT
        sl_points_cost = sl_pips * PIP_SIZE * CONTRACT_SIZE
        total_cost_per_lot = sl_points_cost + COMMISSION_PER_LOT
        if total_cost_per_lot <= 0: return 0.0
        lot_size = risk_amount / total_cost_per_lot
        lot_size = max(MIN_LOT_SIZE, lot_size)
        lot_size = min(MAX_LOT_SIZE, lot_size)
        return np.floor(lot_size * 100) / 100
    except Exception:
        return 0.0

def calculate_profit(direction: str, open_price: float, close_price: float, lot_size: float) -> float:
    if direction == 'BUY': points_gained = close_price - open_price
    else: points_gained = open_price - close_price
    gross_profit = points_gained * CONTRACT_SIZE * lot_size
    net_profit = gross_profit - (COMMISSION_PER_LOT * lot_size)
    return net_profit

def run_validation_backtest(
    df: pd.DataFrame, 
    models: Dict,
    scalers: Dict,
    feature_names: Dict,
    thresholds: Dict
):
    print("\n--- RUNNING V27 (LOW VOL VOLUME) VALIDATION BACKTEST ---")
    print(f"Period: {df.index.min().date()} to {df.index.max().date()}")
    
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    peak_equity = INITIAL_BALANCE
    max_drawdown = 0.0
    trade_open = False
    current_trade = {}
    trade_log = [] 
    
    print("Pre-calculating all features (Corrected AE Alignment)...")
    f_names_low_base = [f for f in feature_names['low_vol'] if f != 'recon_error']
    
    if 'low_vol' not in feature_names or not all(f in df.columns for f in f_names_low_base):
        print(f"FATAL ERROR: Missing required 'low_vol' features in DataFrame.", file=sys.stderr)
        print(f"Available features: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
        
    features_low = df[f_names_low_base]

    print("Shifting features *before* AE prediction (ANTI BIAS PROTOCOL)...")
    features_low_shifted = features_low.shift(1)
    
    features_low_shifted_nonan = features_low_shifted.dropna()
    
    print("Pre-calculating V26 Anomaly Filter (Low-Vol Only) on SHIFTED data...")
    features_low_scaled = scalers['low_vol'].transform(features_low_shifted_nonan)
    features_low_pred = models['ae_low_vol'].predict(features_low_scaled, verbose=0)
    recon_error_arr = np.mean(np.power(features_low_scaled - features_low_pred, 2), axis=1)
    
    recon_error_low = pd.Series(recon_error_arr, index=features_low_shifted_nonan.index, name='recon_error')
    
    final_features_low = features_low_shifted_nonan.join(recon_error_low)
    
    backtest_base = df[['Open','High','Low','Close','regime']].loc[final_features_low.index]
    backtest_data = backtest_base.join(final_features_low)
    
    print(f"Starting loop with {len(backtest_data)} candles...")
    
    f_names_low_final = feature_names['low_vol'] 
    
    for candle in backtest_data.itertuples():
        
        if trade_open:
            close_price = 0.0
            close_reason = "NONE"
            
            if current_trade['direction'] == 'BUY':
                if candle.Low <= current_trade['sl']:
                    close_price = current_trade['sl']
                    close_reason = "SL_HIT"
                elif candle.High >= current_trade['tp']:
                    close_price = current_trade['tp']
                    close_reason = "TP_HIT"
            
            elif current_trade['direction'] == 'SELL':
                if candle.High >= current_trade['sl']:
                    close_price = current_trade['sl']
                    close_reason = "SL_HIT"
                elif candle.Low <= current_trade['tp']:
                    close_price = current_trade['tp']
                    close_reason = "TP_HIT"
            
            if close_reason != "NONE":
                profit = calculate_profit(
                    direction=current_trade['direction'],
                    open_price=current_trade['entry_price'],
                    close_price=close_price,
                    lot_size=current_trade['lot_size']
                )
                balance += profit
                equity = balance
                if equity > peak_equity: peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                if drawdown > max_drawdown: max_drawdown = drawdown
                trade_log.append({
                    "open_time": current_trade['open_time'],
                    "close_time": candle.Index,
                    "direction": current_trade['direction'],
                    "entry_price": current_trade['entry_price'],
                    "tp": current_trade['tp'],
                    "sl": current_trade['sl'],
                    "close_price": close_price,
                    "hit": close_reason,
                    "lot_size": current_trade['lot_size'],
                    "profit": profit,
                    "equity": equity,
                    "regime": "LOW_VOL"
                })
                trade_open = False
                current_trade = {}

        if not trade_open and candle.regime == 0:
            
            candle_data = pd.Series(candle._asdict())
            current_features = [candle_data[f_names_low_final].values]
            
            buy_proba = models['buy_low_vol'].predict_proba(current_features)[0, 1]
            sell_proba = models['sell_low_vol'].predict_proba(current_features)[0, 1]
            
            buy_thresh = thresholds['low_vol']['buy_threshold']
            sell_thresh = thresholds['low_vol']['sell_threshold']
            
            buy_signal = (buy_proba > buy_thresh)
            sell_signal = (sell_proba > sell_thresh)
            
            direction = "NONE"
            if buy_signal and not sell_signal: direction = "BUY"
            elif sell_signal and not buy_signal: direction = "SELL"
                    
            if direction != "NONE":
                lot_size = calculate_lot_size(equity, SL_PIPS) 
                if lot_size < MIN_LOT_SIZE:
                    continue 
                entry_price = candle.Open 
                if direction == "BUY":
                    sl_price = entry_price - SL_TARGET
                    tp_price = entry_price + TP_TARGET
                else: 
                    sl_price = entry_price + SL_TARGET
                    tp_price = entry_price - TP_TARGET
                trade_open = True
                current_trade = {
                    "open_time": candle.Index,
                    "direction": direction,
                    "entry_price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "lot_size": lot_size,
                    "regime": "LOW_VOL"
                }
        
    if trade_open:
        last_candle_close = backtest_data.iloc[-1].Close
        profit = calculate_profit(
            direction=current_trade['direction'],
            open_price=current_trade['entry_price'],
            close_price=last_candle_close,
            lot_size=current_trade['lot_size']
        )
        balance += profit
        equity = balance
        trade_log.append({
            "open_time": current_trade['open_time'],
            "close_time": backtest_data.index[-1],
            "direction": current_trade['direction'],
            "entry_price": current_trade['entry_price'],
            "tp": current_trade['tp'],
            "sl": current_trade['sl'],
            "close_price": last_candle_close,
            "hit": "SIM_END",
            "lot_size": current_trade['lot_size'],
            "profit": profit,
            "equity": equity,
            "regime": current_trade['regime']
        })

    print("\n--- V27 (LOW VOL ONLY) VALIDATION COMPLETE ---")
    
    if not trade_log:
        print("No trades were taken during the validation period.")
        return

    log_df = pd.DataFrame(trade_log)
    log_df = log_df.set_index(pd.to_datetime(log_df['close_time'])) 
    
    print("\n--- FUCKING MONTHLY PERFORMANCE REPORT ---")
    if not log_df.empty:
        monthly_equity = log_df['equity'].resample('ME').last()
        monthly_equity = monthly_equity.ffill()
        monthly_report = pd.DataFrame(monthly_equity)
        monthly_report['prev_equity'] = monthly_report['equity'].shift(1)
        monthly_report.iloc[0, monthly_report.columns.get_loc('prev_equity')] = INITIAL_BALANCE
        monthly_report['Monthly Return %'] = (monthly_report['equity'] - monthly_report['prev_equity']) / monthly_report['prev_equity'] * 100
        monthly_report.index = monthly_report.index.strftime('%Y-%m')
        monthly_report.index.name = 'Month'
        
        print(monthly_report.to_string(
            columns=['equity', 'Monthly Return %'],
            header=['End of Month Equity', 'Monthly Return'],
            formatters={
                'equity': lambda x: f"${x:,.2f}",
                'Monthly Return %': lambda x: f"{x:+.2f}%"
            }
        ))
    else:
        print("No trades taken, so no monthly report. What a fucking waste.")
    
    end_balance = equity
    profit_made = end_balance - INITIAL_BALANCE
    total_trades = len(log_df)
    
    gross_profit = log_df[log_df['profit'] > 0]['profit'].sum()
    gross_loss = abs(log_df[log_df['profit'] <= 0]['profit'].sum())
    
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = (log_df['profit'] > 0).mean()
    
    print(f"\n---@Redouane.Boundra FINAL PERFORMANCE REPORT ({log_df.index.min().date()} - {log_df.index.max().date()}) ---")
    print("=" * 40)
    print(f"Start Balance:       ${INITIAL_BALANCE:,.2f}")
    print(f"End Balance:         ${end_balance:,.2f}")
    print(f"Net Profit:          ${profit_made:,.2f}")
    print(f"Total Return:        {profit_made / INITIAL_BALANCE:.2%}")
    print("-" * 40)
    print(f"Total Trades:        {total_trades}")
    print(f"Win Rate:            {win_rate:.2%}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Max Drawdown:        {max_drawdown:.2%}")
    print(f"Gross Profit:        ${gross_profit:,.2f}")
    print(f"Gross Loss:          ${gross_loss:,.2f}")
    print("=" * 40)

def main():
    print("--- V27 (MT5 MIRROR) VALIDATOR (LOW VOL ONLY) ---")
    
    print(f"Loading models from {MODEL_DIR}...")
    models = {}
    scalers = {}
    feature_names = {}
    thresholds = {}
    
    regimes = ["low_vol_scalper"]
    
    try:
        for regime in regimes:
            print(f"   Loading {regime} assets...")
            suffix = regime.replace("_breaker", "").replace("_scalper", "") 
            
            models[f'buy_{suffix}'] = xgb.XGBClassifier()
            models[f'buy_{suffix}'].load_model(os.path.join(MODEL_DIR, f"v26_buy_model_{regime}.json"))
            
            models[f'sell_{suffix}'] = xgb.XGBClassifier()
            models[f'sell_{suffix}'].load_model(os.path.join(MODEL_DIR, f"v26_sell_model_{regime}.json"))
            
            models[f'ae_{suffix}'] = keras.models.load_model(os.path.join(MODEL_DIR, f"v26_autoencoder_{regime}.keras"))
            
            with open(os.path.join(MODEL_DIR, f"v26_scaler_{regime}.pkl"), 'rb') as f:
                scalers[suffix] = pickle.load(f)
            
            with open(os.path.join(MODEL_DIR, f"v26_features_{regime}.json"), 'r') as f:
                feature_names[suffix] = json.load(f) 
            
            with open(os.path.join(MODEL_DIR, f"v26_thresholds_{regime}.json"), 'r') as f:
                thresholds[suffix] = json.load(f)
            
    except Exception as e:
        print(f"FATAL ERROR: Could not load V26 (LOW VOL) models or asset file.", file=sys.stderr)
        print(f"Make sure '2_train_models.py' ran successfully.", file=sys.stderr)
        print(f"Make sure MODEL_DIR is '{MODEL_DIR}'.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Loaded all V26 LOW-VOL specialist models and assets.")

    SYMBOL = "AUDUSD" 
    TIMEZONE = timezone.utc
    
    FETCH_START_DT = datetime.strptime(FETCH_START_DATE, '%Y-%m-%d').replace(tzinfo=TIMEZONE)
    FETCH_END_DT = datetime.strptime(TEST_END, '%Y-%m-%d').replace(tzinfo=TIMEZONE)

    print(f"Fetching ALL M3 data for {SYMBOL} from MT5...")
    try:
        df = fetch_mt5_data(
            SYMBOL, 
            mt5.TIMEFRAME_M3, 
            FETCH_START_DT, 
            FETCH_END_DT
        )
        df = df.sort_index()
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    breaker_features, breaker_feature_names = create_breaker_features(df)
    scalper_features, scalper_feature_names = create_scalper_features(df)
    
    full_data = df.join(breaker_features).join(scalper_features)
    
    print("Calculating V22 *LEAK-FREE* 'Volatility Regime' filter...")
    
    atr_14_shifted = full_data['breaker_atr_14'].shift(1) 
    long_term_atr = atr_14_shifted.rolling(window=480, min_periods=480).mean() 
    full_data['regime'] = (atr_14_shifted > long_term_atr).astype(int)
    
    test_df = full_data.loc[TEST_START:TEST_END]

    if test_df.empty:
        print("FATAL ERROR: Test data is empty. Check dates.", file=sys.stderr)
        sys.exit(1)
        
    run_validation_backtest(
        test_df, 
        models, 
        scalers,
        feature_names,
        thresholds
    )

if __name__ == "__main__":

    main()
