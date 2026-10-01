"""Feature Engineering Module for 24-Hour-Ahead Soil Moisture Forecasting.

Constructs:
1. Ground Truth Target: Soil/Earth Moisture 24 Hours into the future (t + 24h)
2. Temporal & Cyclical Features: Hour sin/cos, Month sin/cos, Day of week, Weekend flag
3. Lag Features: Historical readings at t-1, t-2, t-3, t-6, t-12, t-24 hours
4. Rolling Window Statistics: Rolling mean, std, min, max over 6h, 12h, 24h windows
5. Signal Processing Features: Reconstructed signal, selected IMFs, residue
6. Precipitation Accumulation: Cumulative rainfall over 6h, 12h, 24h
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def create_24h_ahead_target(
    df: pd.DataFrame,
    target_col: str = "earth_humidity",
    horizon_hours: int = 24
) -> pd.DataFrame:
    """Create timestamp-aware 24-hour ahead forecasting target.
    
    Args:
        df: Input DataFrame sorted chronologically with unified 'timestamp'.
        target_col: Source target column.
        horizon_hours: Forecasting lead time in hours (default 24).
        
    Returns:
        DataFrame with new column 'target_24h_ahead'.
    """
    df = df.copy()
    
    # Verify timestamp ordering
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        
    # Since dataset has regular 1-hour cadence, shift(-horizon_hours) corresponds to t + 24h
    target_name = f"target_{horizon_hours}h_ahead"
    df[target_name] = df[target_col].shift(-horizon_hours)
    
    return df


def generate_features(
    df: pd.DataFrame,
    target_col: str = "earth_humidity",
    horizon_hours: int = 24
) -> Tuple[pd.DataFrame, List[str], str]:
    """Generate comprehensive feature set for LightGBM model.
    
    Args:
        df: Input DataFrame with cleaned and reconstructed signals.
        target_col: Target column name.
        horizon_hours: Lead time horizon in hours.
        
    Returns:
        Tuple of (engineered_df, feature_columns_list, target_column_name).
    """
    print(f"\n========================================================")
    print(f"[*] Engineering Features for {horizon_hours}-Hour-Ahead Forecasting")
    print(f"========================================================")
    
    # 1. Create Target
    df_feat = create_24h_ahead_target(df, target_col=target_col, horizon_hours=horizon_hours)
    target_name = f"target_{horizon_hours}h_ahead"
    
    # 2. Temporal & Cyclical Encodings
    timestamps = pd.to_datetime(df_feat["timestamp"])
    hour = timestamps.dt.hour
    month = timestamps.dt.month
    day_of_week = timestamps.dt.dayofweek
    day_of_year = timestamps.dt.dayofyear
    
    # Continuous cyclical sine/cosine representations
    df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df_feat["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    df_feat["day_of_week"] = day_of_week
    df_feat["is_weekend"] = (day_of_week >= 5).astype(int)
    
    # 3. Base Environmental Variables
    base_env_cols = [
        "air_temperature",
        "air_humidity",
        "air_pressure",
        "dew_point",
        "precipitation",
        "earth_temperature",
        "battery_voltage",
        "solar_panel_voltage",
    ]
    present_env = [c for c in base_env_cols if c in df_feat.columns]
    
    # 4. Lag Features for Primary Variables
    lag_steps = [1, 2, 3, 6, 12, 24]
    
    # Earth Humidity Lags
    for lag in lag_steps:
        df_feat[f"earth_humidity_lag_{lag}h"] = df_feat[target_col].shift(lag)
        
    # Earth Temperature Lags
    if "earth_temperature" in df_feat.columns:
        for lag in [1, 6, 12, 24]:
            df_feat[f"earth_temp_lag_{lag}h"] = df_feat["earth_temperature"].shift(lag)
            
    # Air Temperature & Air Humidity Lags
    for col_name in ["air_temperature", "air_humidity"]:
        if col_name in df_feat.columns:
            for lag in [1, 6, 24]:
                df_feat[f"{col_name}_lag_{lag}h"] = df_feat[col_name].shift(lag)
                
    # 5. Rolling Window Statistical Aggregations (Past History only -> Leakage Safe)
    rolling_windows = [6, 12, 24]
    for w in rolling_windows:
        # Earth Moisture Rolling Stats
        df_feat[f"earth_humidity_roll_mean_{w}h"] = df_feat[target_col].rolling(window=w, min_periods=w).mean()
        df_feat[f"earth_humidity_roll_std_{w}h"] = df_feat[target_col].rolling(window=w, min_periods=w).std()
        
    # 24-hour min, max, and range
    df_feat["earth_humidity_roll_min_24h"] = df_feat[target_col].rolling(window=24, min_periods=24).min()
    df_feat["earth_humidity_roll_max_24h"] = df_feat[target_col].rolling(window=24, min_periods=24).max()
    df_feat["earth_humidity_daily_range"] = df_feat["earth_humidity_roll_max_24h"] - df_feat["earth_humidity_roll_min_24h"]
    
    # Rolling Cumulative Precipitation
    if "precipitation" in df_feat.columns:
        df_feat["precip_accum_6h"] = df_feat["precipitation"].rolling(window=6, min_periods=1).sum()
        df_feat["precip_accum_24h"] = df_feat["precipitation"].rolling(window=24, min_periods=1).sum()
        
    # 6. Signal Processing & Reconstruction Features
    signal_cols = []
    if f"{target_col}_reconstructed" in df_feat.columns:
        signal_cols.append(f"{target_col}_reconstructed")
        # Lag of reconstructed signal
        df_feat[f"{target_col}_reconstructed_lag_1h"] = df_feat[f"{target_col}_reconstructed"].shift(1)
        df_feat[f"{target_col}_reconstructed_lag_24h"] = df_feat[f"{target_col}_reconstructed"].shift(24)
        signal_cols.extend([f"{target_col}_reconstructed_lag_1h", f"{target_col}_reconstructed_lag_24h"])
        
    # Add individual IMF and residue columns if present
    imf_cols = [c for c in df_feat.columns if c.startswith("imf_")]
    signal_cols.extend(imf_cols)
    
    # 7. Collect All Usable Feature Column Names (Strictly Numeric)
    exclude_cols = [
        "station_id", "date", "time", "communicate_at", "source_sheet", "source_year",
        "timestamp", target_name, f"{target_col}_raw", f"{target_col}_is_outlier",
        f"{target_col}_hampel_cleaned", "anomaly_type", "overall_status", "status"
    ]
    
    numeric_cols = df_feat.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and c != target_name]
    
    # 8. Drop NaNs introduced by lagging (first 24 rows) and target lead shift (last 24 rows)
    initial_len = len(df_feat)
    df_clean = df_feat.dropna(subset=feature_cols + [target_name]).reset_index(drop=True)
    dropped_count = initial_len - len(df_clean)
    
    print(f"  • Total Engineered Features : {len(feature_cols)}")
    print(f"  • Feature Names Overview    : {feature_cols[:8]} ... (+{len(feature_cols)-8} more)")
    print(f"  • Valid Sample Count        : {len(df_clean)} rows ({dropped_count} boundary lag/lead rows removed)")
    print(f"  • Target Column             : {target_name}")
    
    return df_clean, feature_cols, target_name


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.preprocessing import combine_and_clean_datasets
    from src.hampel_filter import run_hampel_pipeline
    from src.ceemdan import run_ceemdan_pipeline
    from src.reconstruction import run_reconstruction_pipeline
    
    raw_24, raw_25 = load_all_raw_data()
    _, df_clean = combine_and_clean_datasets(raw_24, raw_25)
    df_hampel = run_hampel_pipeline(df_clean)
    df_imfs, imfs = run_ceemdan_pipeline(df_hampel)
    df_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)
    df_ready, features, target = generate_features(df_recon)
