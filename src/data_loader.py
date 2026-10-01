"""Data Loader Module for Adaptive Irrigation Management System.

Handles programmatic discovery and ingestion of 2024 and 2025 agricultural/weather-station Excel files.
No sheet names or row counts are hardcoded.
"""

from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re


def find_dataset_files(base_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Automatically discover 2024 and 2025 Excel dataset files.
    
    Searches both `data/raw/` and the project root directory.
    If found in the root directory, copies them to `data/raw/` for clean organization.
    
    Args:
        base_dir: Base directory to start searching from. Defaults to project root.
        
    Returns:
        Tuple of (path_2024, path_2025).
    """
    if base_dir is None:
        # Base directory is the parent of src/ if run from src, or current working directory
        base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    search_locations = [raw_dir, base_dir]
    
    file_2024: Optional[Path] = None
    file_2025: Optional[Path] = None

    for loc in search_locations:
        for file_path in loc.glob("*.xlsx"):
            fname = file_path.name.lower()
            if "2024" in fname and file_2024 is None:
                file_2024 = file_path
            elif "2025" in fname and file_2025 is None:
                file_2025 = file_path

    # If found in root but not in raw_dir, copy to raw_dir
    if file_2024 and file_2024.parent != raw_dir:
        dest_2024 = raw_dir / file_2024.name
        if not dest_2024.exists():
            print(f"[*] Organizing raw file: Copying {file_2024.name} to data/raw/")
            shutil.copy2(file_2024, dest_2024)
        file_2024 = dest_2024

    if file_2025 and file_2025.parent != raw_dir:
        dest_2025 = raw_dir / file_2025.name
        if not dest_2025.exists():
            print(f"[*] Organizing raw file: Copying {file_2025.name} to data/raw/")
            shutil.copy2(file_2025, dest_2025)
        file_2025 = dest_2025

    return file_2024, file_2025


def normalize_column_name(col: str) -> str:
    """Standardize column names into snake_case.
    
    Converts camelCase, PascalCase, or mixed-case text to lowercase snake_case.
    """
    s = str(col).strip()
    # Replace camelCase like airTemperature -> air_Temperature
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Replace spaces, hyphens, dots with underscores
    s = re.sub(r'[\s\.\-]+', '_', s)
    # Convert to lowercase
    s = s.lower()
    # Clean up redundant underscores
    s = re.sub(r'_+', '_', s).strip('_')
    return s


# Canonical column mappings to unified schema
COLUMN_SYNONYMS = {
    "air_temperature": [
        "air_temperature", "airtemperature", "air_temp", "temp_air", "air_temperature1",
        "air_temperature_1", "temperature", "temp"
    ],
    "air_humidity": [
        "air_humidity", "airhumidity", "air_humid", "humidity_air", "air_humidity1",
        "air_humidity_1"
    ],
    "air_pressure": ["air_pressure", "airpressure", "pressure", "barometric_pressure"],
    "dew_point": ["dew_point", "dewpoint", "dew_pt"],
    "precipitation": ["precipitation", "rainfall", "rain", "precip"],
    "earth_humidity": [
        "earth_humidity_1", "earth_humidity1", "earthhumidity1", "earth_humidity",
        "earthhumidity", "soil_moisture", "soil_humidity", "humidity", "moisture"
    ],
    "earth_temperature": [
        "earth_temperature_1", "earth_temperature1", "earthtemperature1",
        "earth_temperature", "earthtemperature", "soil_temperature"
    ],
    "battery_voltage": ["battery_voltage", "batteryvoltage", "batt_volt", "v_batt"],
    "solar_panel_voltage": ["solar_panel_voltage", "solarpanelvoltage", "solar_voltage", "solar_volt", "v_solar"],
    "station_id": ["station_id", "stationid", "id", "device_id", "node_id"],
    "communicate_at": ["communicate_at", "communicateat", "datetime", "timestamp", "date_time"],
    "date": ["date", "record_date"],
    "time": ["time", "record_time"],
}


def map_columns_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column names to standard canonical names."""
    normalized_cols = {col: normalize_column_name(col) for col in df.columns}
    renamed_df = df.rename(columns=normalized_cols)
    
    mapping = {}
    for col in renamed_df.columns:
        for canonical, synonyms in COLUMN_SYNONYMS.items():
            if col in synonyms:
                mapping[col] = canonical
                break
                
    renamed_df = renamed_df.rename(columns=mapping)
    return renamed_df


def is_environmental_sheet(df: pd.DataFrame) -> bool:
    """Heuristic check whether a DataFrame sheet contains environmental time-series data."""
    if df.empty or len(df.columns) < 3:
        return False
    
    normalized = [normalize_column_name(c) for c in df.columns]
    
    # Check for environmental indicators
    has_temp = any("temp" in c for c in normalized)
    has_humid = any("humid" in c for c in normalized)
    has_earth = any("earth" in c or "soil" in c for c in normalized)
    has_time = any(c in ["date", "time", "communicate_at", "communicateat", "timestamp", "datetime"] or "time" in c for c in normalized)
    
    return (has_temp or has_humid or has_earth) and has_time


def load_excel_workbook(file_path: Path, label: str) -> pd.DataFrame:
    """Load and combine all valid data sheets from an Excel workbook.
    
    Args:
        file_path: Path to the Excel workbook.
        label: Dataset label (e.g. '2024' or '2025').
        
    Returns:
        Concatenated DataFrame containing all records from valid sheets.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found at: {file_path}")
        
    print(f"\n========================================================")
    print(f"[*] Inspecting {label} Excel Workbook: {file_path.name}")
    print(f"========================================================")
    
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names
    print(f"[*] Found {len(sheet_names)} sheet(s): {sheet_names}")
    
    valid_dfs = []
    
    for sheet in sheet_names:
        try:
            df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
            print(f"  -> Sheet '{sheet}': {df_sheet.shape[0]} rows, {df_sheet.shape[1]} columns")
            
            if is_environmental_sheet(df_sheet):
                print(f"     [+] Detected valid environmental data in sheet: '{sheet}'")
                df_canonical = map_columns_to_canonical(df_sheet)
                df_canonical["source_sheet"] = sheet
                df_canonical["source_year"] = label
                valid_dfs.append(df_canonical)
            else:
                print(f"     [-] Skipping non-environmental or metadata sheet: '{sheet}'")
        except Exception as e:
            print(f"     [!] Warning: Failed to parse sheet '{sheet}': {e}")
            
    if not valid_dfs:
        raise ValueError(f"No valid environmental time-series data sheets identified in {file_path.name}")
        
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"[+] Total raw records loaded for {label}: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    print(f"    Available canonical columns: {list(combined_df.columns)}")
    
    return combined_df


def find_abnormal_dataset_file(base_dir: Optional[Path] = None) -> Optional[Path]:
    """Find any abnormal sensor dataset file in data/raw or project root."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    search_locations = [base_dir / "data" / "raw", base_dir]
    for loc in search_locations:
        if loc.exists():
            for f in loc.glob("*abnormal*.*"):
                if f.suffix.lower() in [".csv", ".xlsx", ".xls"]:
                    return f
    return None


def load_abnormal_dataset(file_path: Optional[Path] = None, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and standardize an abnormal sensor test dataset (CSV or Excel).
    
    Args:
        file_path: Optional direct path to abnormal dataset.
        base_dir: Optional base directory.
        
    Returns:
        Canonicalized DataFrame with unified columns.
    """
    if file_path is None:
        file_path = find_abnormal_dataset_file(base_dir)
        
    if file_path is None or not file_path.exists():
        raise FileNotFoundError("Could not find abnormal sensor dataset file (*abnormal*.*) in data/raw/ or project root.")
        
    print(f"\n========================================================")
    print(f"[*] Ingesting Abnormal Sensor Test Dataset: {file_path.name}")
    print(f"========================================================")
    
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    print(f"[+] Raw abnormal dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    df_canonical = map_columns_to_canonical(df)
    df_canonical["source_sheet"] = file_path.stem
    df_canonical["source_year"] = "Abnormal_Test"
    
    # If earth_humidity is missing but air_humidity is present, use air_humidity
    if "earth_humidity" not in df_canonical.columns and "air_humidity" in df_canonical.columns:
        df_canonical["earth_humidity"] = df_canonical["air_humidity"]
        
    # If earth_temperature is missing but air_temperature is present, mirror it
    if "earth_temperature" not in df_canonical.columns and "air_temperature" in df_canonical.columns:
        df_canonical["earth_temperature"] = df_canonical["air_temperature"]
        
    print(f"[+] Canonicalized abnormal columns: {list(df_canonical.columns)}")
    return df_canonical


def load_all_raw_data(base_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both 2024 and 2025 datasets dynamically.
    
    Returns:
        Tuple of (df_2024, df_2025).
    """
    path_2024, path_2025 = find_dataset_files(base_dir)
    
    if path_2024 is None:
        raise FileNotFoundError("Could not find 2024 dataset (.xlsx file containing '2024') in data/raw or project root.")
    if path_2025 is None:
        raise FileNotFoundError("Could not find 2025 dataset (.xlsx file containing '2025') in data/raw or project root.")
        
    df_2024 = load_excel_workbook(path_2024, label="2024")
    df_2025 = load_excel_workbook(path_2025, label="2025")
    
    return df_2024, df_2025


if __name__ == "__main__":
    print("[*] Running data_loader.py standalone test...")
    abnormal_path = find_abnormal_dataset_file()
    if abnormal_path:
        df_ab = load_abnormal_dataset(abnormal_path)
        print("\n--- Abnormal Sample ---")
        print(df_ab.head(3))
    df24, df25 = load_all_raw_data()
    print("\n--- 2024 Sample ---")
    print(df24.head(3))

