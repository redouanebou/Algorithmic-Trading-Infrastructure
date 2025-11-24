import pandas as pd
import sys

def load_and_clean_m1(m1_path: str) -> pd.DataFrame:
    """
    Loads M1 CSV from Dukascopy (Tick Data Suite format),
    handles weird date formats, removes duplicates,
    fills missing minutes (Gap Filling), and filters out shitty Sunday spreads.
    """
    print(f"Loading M1 data from: {m1_path} ...")

    COLS_OHLCV = ['Open', 'High', 'Low', 'Close', 'Tick volume']

    try:
        df = pd.read_csv(m1_path, engine='c')
        
        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['datetime'] = pd.to_datetime(datetime_str, utc=True, errors='coerce')

    except Exception as e:
        print(f"FATAL ERROR a l'hmri: {e}", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    df = df[~df.index.duplicated(keep='first')]

    for col in COLS_OHLCV:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=COLS_OHLCV)

    print("Filling missing minutes (Forward Fill)...")
    
    df = df.asfreq('1T')
    
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    
    df['Tick volume'] = df['Tick volume'].fillna(0)
    
    df = df.dropna(subset=COLS_OHLCV)

    print("Filtering out Saturday and garbage Sunday hours (< 22:00 UTC)...")
    
    df = df[df.index.dayofweek != 5]
    
    mask_bad_sunday = (df.index.dayofweek == 6) & (df.index.hour < 22)
    df = df[~mask_bad_sunday]

    print(f"M1 data loaded & cleaned: {len(df)} rows.")
    return df[COLS_OHLCV]


def resample_to_m3(df_m1: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples M1 to M3. 
    """
    print("Resampling M1 -> M3 ...")

    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Tick volume': 'sum'
    }

    df_m3 = df_m1.resample('3T').agg(agg_rules)

    df_m3 = df_m3.dropna(subset=['Open', 'Close'])
    
    return df_m3

def main():
    M1_FILE_PATH = r"D:\duka\GBPUSD.csv"
    M3_OUTPUT_PATH = r"D:\duka\GBPUSD_M3_clean.csv"

    df_m1 = load_and_clean_m1(M1_FILE_PATH)

    if df_m1.empty:
        print("FATAL ERROR: Data is empty after cleaning.", file=sys.stderr)
        sys.exit(1)

    df_m3 = resample_to_m3(df_m1)

    try:
        df_m3.to_csv(M3_OUTPUT_PATH, index_label='datetime')
        print(f"\n✅ SUCCESS: Saved clean M3 data to {M3_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
