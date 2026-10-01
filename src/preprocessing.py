"""Data Preprocessing Module for Adaptive Irrigation Management System.

Handles:
- Datetime unification (date + time vs communicate_at)
- Column standardization & type coercion
- Chronological sorting & deduplication
- Time cadence/frequency diagnostics
- Missing value profiling & imputation without silent data loss
- Exporting combined and cleaned datasets to data/processed/
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def unify_timestamp(df: pd.DataFrame) -> pd.Series:
    """Unify different timestamp formats into a single pd.Timestamp Series."""
    # Case 1: communicate_at column exists
    if "communicate_at" in df.columns:
        ts = pd.to_datetime(df["communicate_at"], errors="coerce")
        return ts

    # Case 2: date and time columns exist separately
    if "date" in df.columns and "time" in df.columns:
        date_str = df["date"].astype(str).str.strip()
        time_str = df["time"].astype(str).str.strip()
        # Clean date string if it contains timestamp like '2024-01-01 00:00:00'
        date_clean = date_str.apply(lambda x: x.split(" ")[0] if " " in x else x)
        combined_str = date_clean + " " + time_str
        ts = pd.to_datetime(combined_str, errors="coerce")
        return ts

    # Case 3: single date or timestamp column
    for col in ["timestamp", "datetime", "date"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            return ts

    raise ValueError("Could not identify any timestamp columns (communicate_at, date, time) in DataFrame.")


def standardize_dataframe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Standardize data types and ensure unified timestamp column."""
    df = df.copy()
    print(f"\n[*] Standardizing {label} dataset schema...")
    
    # 1. Build unified timestamp
    df["timestamp"] = unify_timestamp(df)
    
    # Check timestamp parsing success
    nat_count = df["timestamp"].isna().sum()
    if nat_count > 0:
        print(f"  [!] Warning: {nat_count} invalid timestamps encountered and set to NaT.")
        df = df.dropna(subset=["timestamp"])
    
    # 2. Convert all environmental columns to float
    env_cols = [
        "air_temperature",
        "air_humidity",
        "air_pressure",
        "dew_point",
        "precipitation",
        "earth_humidity",
        "earth_temperature",
        "battery_voltage",
        "solar_panel_voltage",
    ]
    
    for col in env_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    print(f"  [+] {label} standardized shape: {df.shape}")
    return df


def print_missing_report(df: pd.DataFrame, stage_name: str) -> None:
    """Print a clean breakdown of missing values per column."""
    print(f"\n--- Missing Values Report: {stage_name} (Total Rows: {len(df)}) ---")
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100
    
    report_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": pct.round(2)})
    active_missing = report_df[report_df["Missing Count"] > 0]
    
    if active_missing.empty:
        print("  [+] No missing values found in any column!")
    else:
        for col, row in active_missing.iterrows():
            print(f"  - {col:25s}: {int(row['Missing Count']):5d} missing ({row['Percentage (%)']:.2f}%)")


def analyze_timestamp_frequency(df: pd.DataFrame) -> pd.Timedelta:
    """Inspect timestamp spacing, report intervals and gaps."""
    print("\n[*] Analyzing Time-Series Frequency & Sampling Cadence...")
    diffs = df["timestamp"].diff().dropna()
    
    if diffs.empty:
        print("  [!] Dataset too small to compute frequency.")
        return pd.Timedelta(hours=1)
        
    median_interval = diffs.median()
    min_interval = diffs.min()
    max_interval = diffs.max()
    
    print(f"  -> Median sampling interval : {median_interval}")
    print(f"  -> Minimum sampling interval: {min_interval}")
    print(f"  -> Maximum sampling interval: {max_interval}")
    
    # Count gaps greater than 2 * median
    large_gaps = (diffs > 2 * median_interval).sum()
    if large_gaps > 0:
        print(f"  [!] Found {large_gaps} time gap(s) exceeding 2x normal sampling interval.")
    else:
        print("  [+] Sampling cadence is highly regular.")
        
    return median_interval


# Physical reasonable bounds for environmental sensors to catch corrupted sentinels (-999, 999, etc.)
PHYSICAL_BOUNDS = {
    "air_temperature": (-40.0, 70.0),
    "earth_temperature": (-40.0, 70.0),
    "air_humidity": (0.0, 100.0),
    "earth_humidity": (0.0, 100.0),
    "dew_point": (-50.0, 60.0),
    "air_pressure": (700.0, 1150.0),
    "precipitation": (0.0, 500.0),
    "battery_voltage": (0.0, 30.0),
    "solar_panel_voltage": (0.0, 50.0),
}


def sanitize_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter impossible sentinel values (-999, 999) and unphysical bounds before signal processing."""
    df = df.copy()
    sanitized_count = 0
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            sentinel_mask = df[col].isin([-999, -999.0, 999, 999.0, -99, 9999, -9999])
            oob_mask = (df[col] < low) | (df[col] > high)
            invalid_mask = sentinel_mask | oob_mask
            count = int(invalid_mask.sum())
            if count > 0:
                sanitized_count += count
                print(f"  [!] Sanitize Bounds: {count} unphysical / sentinel value(s) in '{col}' coerced to NaN")
                df.loc[invalid_mask, col] = np.nan
    return df


def clean_and_impute_dataframe(df: pd.DataFrame, target_col: str = "earth_humidity") -> pd.DataFrame:
    """Sort, deduplicate, sanitize physical bounds, and interpolate continuous sensor readings."""
    cleaned = df.sort_values(by="timestamp").reset_index(drop=True)
    print(f"  -> Timestamp Range: {cleaned['timestamp'].min()} to {cleaned['timestamp'].max()}")
    
    # 1. Deduplicate
    dup_count = cleaned.duplicated(subset=["timestamp"]).sum()
    if dup_count > 0:
        print(f"  [!] Found {dup_count} duplicate timestamp records. Averaging duplicate observations...")
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in cleaned.columns if c not in numeric_cols and c != "timestamp"]
        
        agg_dict = {col: "mean" for col in numeric_cols}
        for col in non_numeric_cols:
            agg_dict[col] = "first"
            
        cleaned = cleaned.groupby("timestamp", as_index=False).agg(agg_dict)
        cleaned = cleaned.sort_values(by="timestamp").reset_index(drop=True)
        print(f"  [+] Deduplicated dataset shape: {cleaned.shape}")
    else:
        print("  [+] No duplicate timestamps detected.")
        
    # 2. Analyze frequency
    analyze_timestamp_frequency(cleaned)
    
    # 3. Sanitize physical bounds (convert -999 / out of bounds to NaN)
    cleaned = sanitize_physical_bounds(cleaned)
    
    # 4. Handle Missing Values & Out-of-Bounds Gaps
    print("\n[*] Handling Missing Values & Outlier Gaps in Environmental Features...")
    if target_col not in cleaned.columns:
        raise KeyError(f"Target column '{target_col}' not found in preprocessed data.")
        
    feature_cols = [
        "air_temperature",
        "air_humidity",
        "air_pressure",
        "dew_point",
        "precipitation",
        "earth_humidity",
        "earth_temperature",
        "battery_voltage",
        "solar_panel_voltage",
    ]
    present_features = [c for c in feature_cols if c in cleaned.columns]
    
    # Interpolate across gap regions (bounded window)
    cleaned[present_features] = cleaned[present_features].interpolate(
        method="linear", limit=12, limit_direction="both"
    )
    
    # Fill precipitation NaNs with 0.0
    if "precipitation" in cleaned.columns:
        cleaned["precipitation"] = cleaned["precipitation"].fillna(0.0)
        
    # Backfill and forward fill boundary edges
    cleaned[present_features] = cleaned[present_features].bfill().ffill()
    
    # Drop remaining NaNs in target if any
    if cleaned[target_col].isna().sum() > 0:
        dropped_nans = cleaned[target_col].isna().sum()
        print(f"  [!] Dropping {dropped_nans} remaining NaN rows in target '{target_col}'")
        cleaned = cleaned.dropna(subset=[target_col]).reset_index(drop=True)
        
    return cleaned


def preprocess_single_dataset(
    df: pd.DataFrame,
    label: str = "Dataset",
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Phase 1 & 2 preprocessing on a single input dataset."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    std_df = standardize_dataframe(df, label)
    
    raw_csv_path = output_dir / "combined_data.csv"
    std_df.to_csv(raw_csv_path, index=False)
    print(f"[+] Saved raw snapshot to: {raw_csv_path}")
    
    print_missing_report(std_df, f"BEFORE Preprocessing ({label} Raw)")
    
    cleaned = clean_and_impute_dataframe(std_df)
    
    print_missing_report(cleaned, f"AFTER Preprocessing ({label} Cleaned)")
    
    clean_csv_path = output_dir / "cleaned_data.csv"
    cleaned.to_csv(clean_csv_path, index=False)
    print(f"\n[+] Successfully saved cleaned dataset to: {clean_csv_path}")
    print(f"[+] Final preprocessed dataset shape: {cleaned.shape} ({len(cleaned)} rows)")
    
    return std_df, cleaned


def combine_and_clean_datasets(
    df_2024: pd.DataFrame,
    df_2025: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Phase 1 & 2 preprocessing pipeline on 2024 and 2025 datasets.
    
    Args:
        df_2024: Raw 2024 DataFrame.
        df_2025: Raw 2025 DataFrame.
        output_dir: Path to directory where processed CSVs should be stored.
        
    Returns:
        Tuple of (combined_raw_df, cleaned_df).
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standardize individual datasets
    std_2024 = standardize_dataframe(df_2024, "2024")
    std_2025 = standardize_dataframe(df_2025, "2025")
    
    # Combine datasets
    print("\n[*] Merging 2024 and 2025 datasets...")
    combined_raw = pd.concat([std_2024, std_2025], ignore_index=True)
    print(f"[+] Merged raw records count: {len(combined_raw)} rows, {len(combined_raw.columns)} columns")
    
    # Save combined raw snapshot
    raw_csv_path = output_dir / "combined_data.csv"
    combined_raw.to_csv(raw_csv_path, index=False)
    print(f"[+] Saved raw combined snapshot to: {raw_csv_path}")
    
    # Report initial missing values
    print_missing_report(combined_raw, "BEFORE Preprocessing (Raw Combined)")
    
    cleaned = clean_and_impute_dataframe(combined_raw)
    
    print_missing_report(cleaned, "AFTER Preprocessing (Cleaned Data)")
    
    # 5. Export cleaned dataset
    clean_csv_path = output_dir / "cleaned_data.csv"
    cleaned.to_csv(clean_csv_path, index=False)
    print(f"\n[+] Successfully saved cleaned dataset to: {clean_csv_path}")
    print(f"[+] Final preprocessed dataset shape: {cleaned.shape} ({len(cleaned)} rows)")
    
    return combined_raw, cleaned


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    print("[*] Running preprocessing.py standalone test...")
    raw_24, raw_25 = load_all_raw_data()
    raw_comb, df_clean = combine_and_clean_datasets(raw_24, raw_25)
