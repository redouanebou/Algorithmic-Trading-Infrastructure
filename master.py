import sys
import os
import json
import pickle
import time
import warnings
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import pytz
from typing import Tuple, List, Dict, Optional
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import xgboost as xgb
import pandas_ta as ta
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

pd.options.mode.chained_assignment = None


PAIR_CONFIG = {
    "GBPUSD": {"model_dir": "GBPUSD1", "magic": 10001, "sl_pips": 5.0, "tp_pips": 5.0},
    "EURUSD": {"model_dir": "EURUSD1", "magic": 10002, "sl_pips": 5.0, "tp_pips": 5.0},
    "EURAUD": {"model_dir": "EURAUD1", "magic": 10003, "sl_pips": 5.0, "tp_pips": 5.0},
    "EURCAD": {"model_dir": "EURCAD1",  "magic": 10004, "sl_pips": 5.0, "tp_pips": 5.0},
}
ACCOUNT_CURRENCY = "USD"
TIMEFRAME_M3 = mt5.TIMEFRAME_M3
RISK_PER_TRADE_PCT = 0.005
CANDLES_TO_FETCH = 500 
INITIAL_BALANCE = 10000.0
FETCH_START_DATE = '2010-01-01' 
COMMISSION_PER_LOT_BACKTEST = 7.0
COMMISSION_PER_LOT_LIVE = 0.0
BROKER_TIMEZONE_STR = "Europe/Helsinki" 
BROKER_TZ = pytz.timezone(BROKER_TIMEZONE_STR)
UTC_TZ = pytz.timezone("UTC")


class TradingSystem:
    def __init__(self, symbol: str, mode: str):
        self.symbol = symbol
        self.mode = mode 
        
        if symbol not in PAIR_CONFIG:
            raise ValueError(f"FATAL: Symbol {symbol} not in PAIR_CONFIG.")
            
        self.config = PAIR_CONFIG[symbol]
        self.model_dir = self.config["model_dir"]
        self.magic_number = self.config["magic"]
        
        self.sl_pips = self.config["sl_pips"]
        self.tp_pips = self.config["tp_pips"]
        
        self.models = {}
        self.commission_per_lot = COMMISSION_PER_LOT_BACKTEST if mode == "backtest" else COMMISSION_PER_LOT_LIVE
        
        self.quote_currency = symbol[3:]
        self.needs_conversion = self.quote_currency != ACCOUNT_CURRENCY
        self.conv_symbol = ""
        self.conv_invert = False
        
        self.broker_config = {
            "CONTRACT_SIZE": None,
            "MIN_LOT_SIZE": None,
            "MAX_LOT_SIZE": None,
            "PIP_SIZE": None,
            "POINT": None,
            "VOLUME_STEP": None 
        }
        
    def initialize_mt5_specs(self) -> bool:
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"FATAL ERROR: Symbol {self.symbol} not found.", file=sys.stderr)
            return False
            
        self.broker_config["CONTRACT_SIZE"] = symbol_info.trade_contract_size
        self.broker_config["MIN_LOT_SIZE"] = symbol_info.volume_min
        self.broker_config["MAX_LOT_SIZE"] = symbol_info.volume_max
        self.broker_config["POINT"] = symbol_info.point
        self.broker_config["VOLUME_STEP"] = symbol_info.volume_step  
        
        digits = symbol_info.digits
        if digits % 2 == 1: 
            self.broker_config["PIP_SIZE"] = symbol_info.point * 10
        else:  
            self.broker_config["PIP_SIZE"] = symbol_info.point
        
        if self.needs_conversion:
            direct_symbol = ACCOUNT_CURRENCY + self.quote_currency  
            invert_symbol = self.quote_currency + ACCOUNT_CURRENCY  
            
            if mt5.symbol_info(invert_symbol): 
                self.conv_symbol = invert_symbol
                self.conv_invert = False 
            elif mt5.symbol_info(direct_symbol): 
                self.conv_symbol = direct_symbol
                self.conv_invert = True 
            else:
                print(f"FATAL: No conversion pair found for {self.quote_currency} -> {ACCOUNT_CURRENCY}", file=sys.stderr)
                return False
            print(f"   Conversion enabled for {self.symbol}: Using {self.conv_symbol} (Invert: {self.conv_invert})")
            
        return True
    
    def load_models(self) -> bool:
        try:
            regimes = ["low_vol_scalper"] 
            for regime in regimes:
                suffix = regime.replace("_breaker", "").replace("_scalper", "")
                self.models[f'buy_{suffix}'] = xgb.XGBClassifier()
                self.models[f'buy_{suffix}'].load_model(os.path.join(self.model_dir, f"v26_buy_model_{regime}.json"))
                
                self.models[f'sell_{suffix}'] = xgb.XGBClassifier()
                self.models[f'sell_{suffix}'].load_model(os.path.join(self.model_dir, f"v26_sell_model_{regime}.json"))
                
                self.models[f'ae_{suffix}'] = keras.models.load_model(os.path.join(self.model_dir, f"v26_autoencoder_{regime}.keras"))
                
                with open(os.path.join(self.model_dir, f"v26_scaler_{regime}.pkl"), 'rb') as f:
                    self.models[f'scaler_{suffix}'] = pickle.load(f)
                
                with open(os.path.join(self.model_dir, f"v26_features_{regime}.json"), 'r') as f:
                    self.models[f'features_{suffix}'] = json.load(f)
                
                with open(os.path.join(self.model_dir, f"v26_thresholds_{regime}.json"), 'r') as f:
                    self.models[f'thresholds_{suffix}'] = json.load(f)
            return True
        except Exception as e:
            print(f"FATAL ERROR: Could not load models from {self.model_dir}. Error: {e}", file=sys.stderr)
            return False
    
    def fetch_mt5_data(self, symbol_to_fetch: str, start_dt_utc, end_dt_utc) -> pd.DataFrame:
        try:
            rates = mt5.copy_rates_range(symbol_to_fetch, TIMEFRAME_M3, start_dt_utc, end_dt_utc)
        except Exception as e:
            print(f"FATAL ERROR: Failed to fetch rates for {symbol_to_fetch}. Error: {e}", file=sys.stderr)
            return pd.DataFrame()
            
        if rates is None or len(rates) == 0:
            print(f"FATAL ERROR: No data returned from MT5 for {symbol_to_fetch}.", file=sys.stderr)
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('datetime')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Tick volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Tick volume']]
        df = df[df.index.dayofweek < 5]
        df = df[~df.index.duplicated(keep='first')]
        return df
    
    def get_latest_data_utc(self, candles_to_fetch: int) -> pd.DataFrame:
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME_M3, 0, candles_to_fetch)
        except Exception as e:
            print(f"Error fetching rates: {e}", file=sys.stderr)
            return pd.DataFrame()
            
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df = df.set_index('datetime')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Tick volume'
        })
        df = df[['Open', 'High', 'Low', 'Close', 'Tick volume']]
        df = df[~df.index.duplicated(keep='first')]
        return df
    
    def create_volume_features(self, df: pd.DataFrame, prefix: str) -> List[pd.Series]:
        features_list = []
        close = df['Close']
        volume = df['Tick volume']
        vol_mean = volume.rolling(window=200).mean()
        vol_std = volume.rolling(window=200).std()
        features_list.append(((volume - vol_mean) / (vol_std + 1e-9)).rename(f'{prefix}_vol_zscore_200'))
        vol_ema_fast = ta.ema(volume, 5)
        vol_ema_slow = ta.ema(volume, 50)
        features_list.append((vol_ema_fast - vol_ema_slow).rename(f'{prefix}_vol_regime_5_50'))
        price_mom = close.diff(5)
        features_list.append((price_mom * volume).rename(f'{prefix}_price_x_vol'))
        return features_list
    
    def create_breaker_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        features_list = []
        close = df['Close']
        high = df['High']
        low = df['Low']
        atr_14 = ta.atr(high, low, close, 14).rename('atr_14')
        features_list.append(atr_14)
        adx_df = ta.adx(high, low, close, 14)
        if isinstance(adx_df, pd.DataFrame) and 'ADX_14' in adx_df.columns:
            features_list.append(adx_df['ADX_14'].rename('adx_14'))
        else:
            features_list.append(pd.Series(np.nan, index=df.index, name='adx_14'))
        features_list.append(ta.rsi(close, 14).rename('rsi_14'))
        features_list.append(close.diff(5).rename('mom_5'))
        features_list.append(close.diff(20).rename('mom_20'))
        features_list.append((close.diff(10) / (atr_14 + 1e-9)).rename('norm_mom_10'))
        features_list.append(pd.Series(np.sin(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_sin'))
        features_list.append(pd.Series(np.cos(2 * np.pi * df.index.hour / 24), index=df.index, name='hour_cos'))
        features_list.extend(self.create_volume_features(df, prefix='breaker'))
        final_features_df = pd.concat(features_list, axis=1).add_prefix('breaker_')
        return final_features_df, final_features_df.columns.tolist()
    
    def create_scalper_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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
        
        features_list.extend(self.create_volume_features(df, prefix='scalper')) 
        
        final_features_df = pd.concat(features_list, axis=1).add_prefix('scalper_')
        
        return final_features_df, final_features_df.columns.tolist()
    
    def calculate_lot_size(self, equity: float, sl_pips: float, conv_rate: float) -> float:
        try:
            if any(v is None for v in self.broker_config.values()):
                print("FATAL: Broker config is not loaded. Cannot calculate lot size.", file=sys.stderr)
                return 0.0
                
            risk_amount_usd = equity * RISK_PER_TRADE_PCT
            
            sl_points = sl_pips * self.broker_config["PIP_SIZE"]
            sl_cost_quote = sl_points * self.broker_config["CONTRACT_SIZE"]
            
            sl_cost_usd = sl_cost_quote * conv_rate
            
            total_cost_per_lot_usd = sl_cost_usd + self.commission_per_lot
            
            if total_cost_per_lot_usd <= 0:
                return 0.0
                
            lot_size = risk_amount_usd / total_cost_per_lot_usd
            
            lot_size = max(self.broker_config["MIN_LOT_SIZE"], lot_size)
            lot_size = min(self.broker_config["MAX_LOT_SIZE"], lot_size)
            
            volume_step = self.broker_config["VOLUME_STEP"]
            return np.floor(lot_size / volume_step) * volume_step
        except Exception:
            return 0.0
    
    def calculate_profit(self, direction: str, open_price: float, close_price: float, lot_size: float, conv_rate: float) -> float:
        if self.broker_config["CONTRACT_SIZE"] is None:
             print("FATAL: Broker config is not loaded. Cannot calculate profit.", file=sys.stderr)
             return 0.0
             
        if direction == 'BUY':
            points_gained = close_price - open_price
        else:
            points_gained = open_price - close_price
            
        gross_profit_quote = points_gained * self.broker_config["CONTRACT_SIZE"] * lot_size
        
        gross_profit_usd = gross_profit_quote * conv_rate
        
        net_profit_usd = gross_profit_usd - (self.commission_per_lot * lot_size)
        return net_profit_usd
    
    def get_live_conv_rate(self) -> float:
        if not self.needs_conversion:
            return 1.0
            
        try:
            tick = mt5.symbol_info_tick(self.conv_symbol)
            if tick is None:
                print(f"FATAL: Could not get live tick for conversion pair {self.conv_symbol}", file=sys.stderr)
                return 1.0 
            
            rate = tick.bid
            if rate == 0: rate = tick.ask 
            
            if self.conv_invert:
                return 1.0 / rate
            else:
                return rate
        except Exception as e:
            print(f"FATAL: Error getting live conversion rate: {e}", file=sys.stderr)
            return 1.0
            
    def open_trade(self, direction: str, lot_size: float):
        print("\n" + "="*30)
        print(f"🔥 SIGNAL: {direction} on {self.symbol} 🔥")
        print(f"   Lot Size: {lot_size}")
        print("="*30)
        
        if self.broker_config["PIP_SIZE"] is None:
            print("   ❌ ORDER FAILED: Broker specs not loaded.", file=sys.stderr)
            return
            
        sl_target_price = self.sl_pips * self.broker_config["PIP_SIZE"]
        tp_target_price = self.tp_pips * self.broker_config["PIP_SIZE"]
        
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
            sl = price - sl_target_price
            tp = price + tp_target_price
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
            sl = price + sl_target_price
            tp = price - tp_target_price
        else:
            return
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"-V27-{self.symbol}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        try:
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"   ❌ ORDER FAILED: retcode={result.retcode}, comment={result.comment}")
            else:
                print(f"   ✅ ORDER SENT: {direction} {lot_size} lot @ {price} | SL: {sl} TP: {tp}")
                print(f"   Ticket: {result.order}")
        except Exception as e:
            print(f"   ❌ ORDER EXCEPTION: {e}", file=sys.stderr)
        print("="*30 + "\n")
    
    def get_trade_signal(self) -> Tuple[str, float, float]:
        df = self.get_latest_data_utc(CANDLES_TO_FETCH)
        if df.empty or len(df) < CANDLES_TO_FETCH:
            return "NONE", 0.0, 0.0
        
        breaker_features, _ = self.create_breaker_features(df)
        scalper_features, _ = self.create_scalper_features(df)
        full_data = df.join(breaker_features).join(scalper_features)
        
        atr_14_shifted = full_data['breaker_atr_14'].shift(1)
        long_term_atr = atr_14_shifted.rolling(window=480, min_periods=480).mean()
        full_data['regime'] = (atr_14_shifted > long_term_atr).astype(int)
        
        full_data = full_data.dropna()
        if full_data.empty or len(full_data) < 2:
            return "NONE", 0.0, 0.0
            
        last_candle = full_data.iloc[-2] 
        if last_candle['regime'] != 0:
            return "NONE", 0.0, 0.0
        
        current_hour_utc = last_candle.name.hour
        if current_hour_utc == 22 or current_hour_utc == 23:
            print(f"   Time filter active ({current_hour_utc}:00 UTC). No trades.")
            return "NONE", 0.0, 0.0
            
        try:
            suffix = 'low_vol'
            scaler = self.models[f'scaler_{suffix}']
            ae = self.models[f'ae_{suffix}']
            model_buy = self.models[f'buy_{suffix}']
            model_sell = self.models[f'sell_{suffix}']
            f_names = self.models[f'features_{suffix}']
            thresholds = self.models[f'thresholds_{suffix}']
            
            f_names_base = [f for f in f_names if f != 'recon_error']
            
            if not all(f in last_candle.index for f in f_names_base):
                print(f"   FATAL: Feature mismatch. Model needs: {f_names_base}", file=sys.stderr)
                print(f"   Script generated: {last_candle.index.tolist()}", file=sys.stderr)
                return "NONE", 0.0, 0.0
            
            current_features_base = last_candle[f_names_base].values.reshape(1, -1)
            current_features_scaled = scaler.transform(current_features_base)
            pred = ae.predict(current_features_scaled, verbose=0)
            recon_error = np.mean(np.power(current_features_scaled - pred, 2), axis=1)[0]
            current_features_final = np.append(current_features_base, recon_error).reshape(1, -1)
            
            buy_proba = model_buy.predict_proba(current_features_final)[0, 1]
            sell_proba = model_sell.predict_proba(current_features_final)[0, 1]
            
            buy_thresh = thresholds['buy_threshold']
            sell_thresh = thresholds['sell_threshold']
            
            buy_signal = (buy_proba > buy_thresh)
            sell_signal = (sell_proba > sell_thresh)
            
            if buy_signal and not sell_signal:
                return "BUY", buy_proba, sell_proba
            elif sell_signal and not buy_signal:
                return "SELL", buy_proba, sell_proba
            else:
                return "NONE", buy_proba, sell_proba
        except Exception as e:
            print(f"ERROR in prediction: {e}", file=sys.stderr)
            return "NONE", 0.0, 0.0
    
    def run_backtest(self, start_date: str, end_date: str):
        print(f"\n--- BACKTEST: {self.symbol} ---")
        print(f"Period: {start_date} to {end_date}")
        
        fetch_start_dt = datetime.strptime(FETCH_START_DATE, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        fetch_end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) + timedelta(days=1)
        
        df = self.fetch_mt5_data(self.symbol, fetch_start_dt, fetch_end_dt)
        if df.empty:
            print("FATAL ERROR: No data fetched for main symbol.", file=sys.stderr)
            return
        
        conv_df = None
        if self.needs_conversion:
            print(f"   Fetching conversion data for {self.conv_symbol}...")
         
            
            conv_df = self.fetch_mt5_data(self.conv_symbol, fetch_start_dt, fetch_end_dt)
            
            if conv_df.empty:
                print(f"FATAL: Could not get conversion data for {self.conv_symbol}", file=sys.stderr)
                return
            if self.conv_invert:
                print(f"   Inverting conversion rate for {self.conv_symbol}")
                conv_df['Close'] = 1.0 / conv_df['Close'].replace(0, np.nan)

        print("   Calculating all features...")
        breaker_features, _ = self.create_breaker_features(df)
        scalper_features, _ = self.create_scalper_features(df)
        full_data = df.join(breaker_features).join(scalper_features)
        
        atr_14_shifted = full_data['breaker_atr_14'].shift(1)
        long_term_atr = atr_14_shifted.rolling(window=480, min_periods=480).mean()
        full_data['regime'] = (atr_14_shifted > long_term_atr).astype(int)
        
        test_df = full_data.loc[start_date:end_date]
        if test_df.empty:
            print("FATAL ERROR: Test data is empty for this date range.", file=sys.stderr)
            return
        
        print("   Preparing backtest data (AE + Shifting)...")
        suffix = 'low_vol'
        f_names = self.models[f'features_{suffix}']
        f_names_base = [f for f in f_names if f != 'recon_error']
        
        features_low = test_df[f_names_base]
        
        features_low_shifted = features_low.shift(1)
        features_low_shifted_nonan = features_low_shifted.dropna()
        
        features_low_scaled = self.models[f'scaler_{suffix}'].transform(features_low_shifted_nonan)
        features_low_pred = self.models[f'ae_{suffix}'].predict(features_low_scaled, verbose=0)
        recon_error_arr = np.mean(np.power(features_low_scaled - features_low_pred, 2), axis=1)
        recon_error_low = pd.Series(recon_error_arr, index=features_low_shifted_nonan.index, name='recon_error')
        final_features_low = features_low_shifted_nonan.join(recon_error_low)
        
        backtest_base = test_df[['Open','High','Low','Close','regime']].loc[final_features_low.index]
        backtest_data = backtest_base.join(final_features_low)
        
        if self.needs_conversion and conv_df is not None:
            backtest_data['conv_rate'] = conv_df['Close'].reindex(backtest_data.index, method='ffill')
            backtest_data = backtest_data.dropna(subset=['conv_rate']) 
        else:
            backtest_data['conv_rate'] = 1.0 
        
        print(f"   Starting backtest loop with {len(backtest_data)} candles...")
        
        balance = INITIAL_BALANCE
        equity = INITIAL_BALANCE
        peak_equity = INITIAL_BALANCE
        max_drawdown = 0.0
        trade_open = False
        current_trade = {}
        trade_log = []
        
        f_names_final = f_names
        
        for candle in backtest_data.itertuples():
            conv_rate = candle.conv_rate
            
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
                    profit = self.calculate_profit(
                        direction=current_trade['direction'],
                        open_price=current_trade['entry_price'],
                        close_price=close_price,
                        lot_size=current_trade['lot_size'],
                        conv_rate=conv_rate
                    )
                    balance += profit
                    equity = balance
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
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
                        "equity": equity
                    })
                    trade_open = False
                    current_trade = {}
            
            if not trade_open:
                current_hour = candle.Index.hour
                if current_hour == 22 or current_hour == 23:
                    continue 
                    
                if candle.regime == 0:
                    candle_data = pd.Series(candle._asdict())
                    current_features = [candle_data[f_names_final].values]
                    
                    buy_proba = self.models[f'buy_{suffix}'].predict_proba(current_features)[0, 1]
                    sell_proba = self.models[f'sell_{suffix}'].predict_proba(current_features)[0, 1]
                    
                    buy_thresh = self.models[f'thresholds_{suffix}']['buy_threshold']
                    sell_thresh = self.models[f'thresholds_{suffix}']['sell_threshold']
                    
                    buy_signal = (buy_proba > buy_thresh)
                    sell_signal = (sell_proba > sell_thresh)
                    
                    direction = "NONE"
                    if buy_signal and not sell_signal:
                        direction = "BUY"
                    elif sell_signal and not buy_signal:
                        direction = "SELL"
                    
                    if direction != "NONE":
                        lot_size = self.calculate_lot_size(equity, self.sl_pips, conv_rate)
                        if lot_size < self.broker_config["MIN_LOT_SIZE"]:
                            continue
                        entry_price = candle.Open
                        sl_target_distance = self.sl_pips * self.broker_config["PIP_SIZE"]
                        tp_target_distance = self.tp_pips * self.broker_config["PIP_SIZE"]
                        if direction == "BUY":
                            sl_price = entry_price - sl_target_distance
                            tp_price = entry_price + tp_target_distance
                        else:
                            sl_price = entry_price + sl_target_distance
                            tp_price = entry_price - tp_target_distance
                        trade_open = True
                        current_trade = {
                            "open_time": candle.Index,
                            "direction": direction,
                            "entry_price": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "lot_size": lot_size
                        }
        
        if trade_open:
            last_candle_close = backtest_data.iloc[-1].Close
            final_conv_rate = backtest_data.iloc[-1].conv_rate
            profit = self.calculate_profit(
                direction=current_trade['direction'],
                open_price=current_trade['entry_price'],
                close_price=last_candle_close,
                lot_size=current_trade['lot_size'],
                conv_rate=final_conv_rate
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
                "equity": equity
            })
        
        print("\n--- BACKTEST COMPLETE ---")
        
        if not trade_log:
            print("No trades were taken during the backtest period.")
            return
        
        log_df = pd.DataFrame(trade_log)
        log_df = log_df.set_index(pd.to_datetime(log_df['close_time']))
        
        print("\n--- MONTHLY PERFORMANCE ---")
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
        
        end_balance = equity
        profit_made = end_balance - INITIAL_BALANCE
        total_trades = len(log_df)
        gross_profit = log_df[log_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(log_df[log_df['profit'] <= 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        win_rate = (log_df['profit'] > 0).mean()
        
        print(f"\n--- FINAL PERFORMANCE: {self.symbol} ---")
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
    
    def live_trading_loop(self):
        print(f"\n--- LIVE TRADING: {self.symbol} ---")
        print("Bot is running. Waiting for signals...")
        
        while True:
            now_broker = datetime.now(BROKER_TZ)
            now_utc = now_broker.astimezone(timezone.utc)
            current_minute_of_hour = now_broker.minute
            current_second = now_broker.second
            minutes_to_next_candle = 3 - (current_minute_of_hour % 3)
            seconds_to_next_candle = (minutes_to_next_candle * 60) - current_second
            
            if seconds_to_next_candle > 2:
                print(f"   Waiting... {seconds_to_next_candle:<3}s | {self.symbol} | {now_broker.strftime('%H:%M:%S')}", end="\r")
                time.sleep(1)
                continue
            
            if current_second < 2:
                time.sleep(2 - current_second)
            
            print("\n" + "="*50)
            print(f"NEW CANDLE [{self.symbol}] @ {datetime.now(BROKER_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*50)
            
            positions = mt5.positions_get(symbol=self.symbol, magic=self.magic_number)
            if positions is None or len(positions) == 0:
                print(f"   No open positions for {self.symbol}. Running trade logic...")
                signal, buy_proba, sell_proba = self.get_trade_signal()
                
                print(f"   {self.symbol}: {signal if signal != 'NONE' else 'NO SIGNAL'} | BUY: {buy_proba*100:.1f}% | SELL: {sell_proba*100:.1f}%")
                
                if signal != "NONE":
                    account_info = mt5.account_info()
                    if account_info is None:
                        print("   Could not get account info.", file=sys.stderr)
                        continue
                    
                    live_conv_rate = self.get_live_conv_rate()
                    if live_conv_rate == 1.0 and self.needs_conversion:
                        print("   FATAL: Could not get live conversion rate. Skipping trade.", file=sys.stderr)
                        continue
                        
                    lot_size = self.calculate_lot_size(account_info.equity, self.sl_pips, live_conv_rate)
                    
                    if lot_size >= self.broker_config["MIN_LOT_SIZE"]:
                        self.open_trade(signal, lot_size)
                    else:
                        print(f"   Lot size {lot_size} too small. No trade.")
            else:
                print(f"   Position already open for {self.symbol} ({len(positions)}). Waiting for SL/TP.")
            
            print("\n   Check complete. Sleeping 60s...")
            time.sleep(60)


def display_menu():
    print("\n" + "="*60)
    print("           V27 UNIFIED TRADING SYSTEM")
    print("="*60)
    print("\n[1] Live Trading")
    print("[2] Backtesting")
    print("[0] Exit")
    print("\n" + "="*60)


def display_pair_menu(all_pairs: bool = False):
    print("\n--- SELECT CURRENCY PAIR ---")
    print("[1] GBPUSD")
    print("[2] EURUSD")
    print("[3] EURAUD")
    print("[4] EURCAD")
    if all_pairs:
        print("[5] ALL PAIRS (Live Only)")
    print("[0] Back to Main Menu")
    return input("\nYour choice: ").strip()


def display_backtest_period_menu():
    print("\n--- SELECT BACKTEST PERIOD ---")
    print("[1] Last Month")
    print("[2] Last 3 Months")
    print("[3] Last 6 Months")
    print("[4] Last Year")
    print("[5] Custom Date Range")
    print("[0] Back")
    return input("\nYour choice: ").strip()


def get_date_range(choice: str) -> Tuple[str, str]:
    today = datetime.now(timezone.utc)
    
    if choice == "1":
        start = today - relativedelta(months=1)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif choice == "2":
        start = today - relativedelta(months=3)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif choice == "3":
        start = today - relativedelta(months=6)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif choice == "4":
        start = today - relativedelta(years=1)
        return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    elif choice == "5":
        print("\nEnter start date (YYYY-MM-DD):")
        start_input = input("> ").strip()
        print("Enter end date (YYYY-MM-DD):")
        end_input = input("> ").strip()
        try:
            datetime.strptime(start_input, '%Y-%m-%d')
            datetime.strptime(end_input, '%Y-%m-%d')
            return start_input, end_input
        except ValueError:
            print("Invalid date format. Using last month as default.")
            start = today - relativedelta(months=1)
            return start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    else:
        return "", ""


def get_symbol_from_choice(choice: str) -> Optional[str]:
    symbol_map = {
        "1": "GBPUSD", "2": "EURUSD", "3": "EURAUD",
        "4": "EURCAD"
    }
    return symbol_map.get(choice)


def run_live_trading_multi():
    print("\n--- INITIALIZING LIVE TRADING SYSTEM ---")
    
    selected_pairs = []
    while True:
        pair_choice = display_pair_menu(all_pairs=True)
        if pair_choice == "0":
            break
        
        if pair_choice == "5":
            print("✅ ALL pairs selected.")
            selected_pairs = list(PAIR_CONFIG.keys())
            break
            
        symbol = get_symbol_from_choice(pair_choice)
        if symbol and symbol not in selected_pairs:
            selected_pairs.append(symbol)
            print(f"✅ {symbol} added to live trading list.")
        elif symbol in selected_pairs:
            print(f"⚠️  {symbol} already selected.")
        else:
            print("Invalid choice.")
        
        continue_choice = input("\nAdd another pair? (y/n): ").strip().lower()
        if continue_choice != 'y':
            break
    
    if not selected_pairs:
        print("No pairs selected. Returning to main menu.")
        return
    
    print(f"\n--- LOADING SYSTEMS FOR: {', '.join(selected_pairs)} ---")
    
    systems = []
    if not mt5.initialize():
        print("FATAL: Could not initialize MT5. Exiting.", file=sys.stderr)
        return

    for symbol in selected_pairs:
        system = TradingSystem(symbol, mode="live")
        if not system.initialize_mt5_specs(): 
             print(f"Failed to initialize specs for {symbol}. Skipping.")
             continue
             
        if not system.load_models():
            print(f"Failed to load models for {symbol}. Skipping.")
            continue
        systems.append(system)
        print(f"✅ {symbol} system ready. Magic Number: {system.magic_number}")
    
    if not systems:
        print("No systems initialized. Exiting.")
        mt5.shutdown()
        return
    
    print(f"\n--- STARTING LIVE TRADING FOR {len(systems)} PAIR(S) ---")
    
    try:
        while True:
            now_utc = datetime.now(UTC_TZ)
            current_minute = now_utc.minute
            current_second = now_utc.second
            
            is_candle_start = (current_minute % 3 == 0)
            
            if is_candle_start and current_second >= 2 and current_second < 5:
                print("\n" + "="*60)
                print(f"NEW CANDLE @ {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print("="*60)
                
                for system in systems:
                    print(f"\n--- Checking {system.symbol} ---")
                    positions = mt5.positions_get(symbol=system.symbol, magic=system.magic_number)
                    
                    if positions is None or len(positions) == 0:
                        print(f"   No open positions. Running trade logic...")
                        signal, buy_proba, sell_proba = system.get_trade_signal()
                        
                        print(f"   {system.symbol}: {signal if signal != 'NONE' else 'NO SIGNAL'} | BUY: {buy_proba*100:.1f}% | SELL: {sell_proba*100:.1f}%")
                        
                        if signal != "NONE":
                            account_info = mt5.account_info()
                            if account_info is None:
                                print("   Could not get account info.", file=sys.stderr)
                                continue
                            
                            live_conv_rate = system.get_live_conv_rate()
                            if live_conv_rate == 1.0 and system.needs_conversion:
                                print("   FATAL: Could not get live conversion rate. Skipping trade.", file=sys.stderr)
                                continue
                                
                            lot_size = system.calculate_lot_size(account_info.equity, system.sl_pips, live_conv_rate)
                            
                            if lot_size >= system.broker_config["MIN_LOT_SIZE"]:
                                system.open_trade(signal, lot_size)
                            else:
                                print(f"   Lot size {lot_size} too small. No trade.")
                    else:
                        print(f"   Position already open ({len(positions)}). Waiting for SL/TP.")
                
                print(f"\n   All pairs checked. Waiting for next candle...")
                time.sleep(3) 
            else:
                minutes_until_next = (3 - (current_minute % 3)) % 3
                seconds_until_next = (minutes_until_next * 60) - current_second + 2 
                if seconds_until_next <= 0:
                    seconds_until_next += 180
                print(f"   Waiting... {seconds_until_next:<3}s | {now_utc.strftime('%H:%M:%S')} UTC", end="\r")
                time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nLive trading stopped by user.")
        mt5.shutdown()


def run_backtesting():
    print("\n--- BACKTESTING MODE ---")
    
    pair_choice = display_pair_menu(all_pairs=False)
    symbol = get_symbol_from_choice(pair_choice)
    
    if not symbol:
        print("Invalid pair selection.")
        return
    
    period_choice = display_backtest_period_menu()
    start_date, end_date = get_date_range(period_choice)
    
    if not start_date or not end_date:
        print("Invalid period selection.")
        return
    
    if not mt5.initialize():
        print(f"Failed to initialize MT5 for {symbol}.")
        mt5.shutdown()
        return
    
    system = TradingSystem(symbol, mode="backtest")
    
    if not system.initialize_mt5_specs():
        print(f"Failed to initialize specs for {symbol}.")
        mt5.shutdown()
        return
    
    mt5.shutdown() 
    
    if not system.load_models():
        print(f"Failed to load models for {symbol}.")
        return
    
    if not mt5.initialize():
        print("FATAL: MT5 connection failed for backtest data.", file=sys.stderr)
        return
        
    system.run_backtest(start_date, end_date)
    mt5.shutdown() 


def main():
    while True:
        display_menu()
        choice = input("\nYour choice: ").strip()
        
        if choice == "1":
            run_live_trading_multi()
        elif choice == "2":
            run_backtesting()
        elif choice == "0":
            print("\nRedouane Boundra Exiting  V27. Good luck!")
            try:
                mt5.shutdown()
            except:
                pass
            sys.exit(0)
        else:
            print("\n⚠️  Invalid choice. Please try again.")



if __name__ == "__main__":
    main()