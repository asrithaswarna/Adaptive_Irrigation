"""LightGBM Machine Learning Model Module for 24-Hour-Ahead Soil Moisture Prediction.

Features:
- Chronological, non-shuffled Train / Validation / Test temporal splitting
- Leakage-free training of LightGBMRegressor
- Validation-monitored early stopping
- Model persistence via Joblib
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd


def chronological_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Split time-series data strictly chronologically without shuffling.
    
    Args:
        df: Input DataFrame with features and target.
        feature_cols: List of input feature names.
        target_col: Target variable column name.
        train_ratio: Proportion of earliest data for training (default 0.70).
        val_ratio: Proportion of intermediate data for validation (default 0.15).
        
    Returns:
        Dictionary containing (X, y) datasets for train, val, and test splits.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    splits = {
        "train": (train_df[feature_cols], train_df[target_col], train_df),
        "val": (val_df[feature_cols], val_df[target_col], val_df),
        "test": (test_df[feature_cols], test_df[target_col], test_df),
    }
    
    print(f"\n========================================================")
    print(f"[*] Chronological Train / Validation / Test Splitting")
    print(f"========================================================")
    print(f"  • Total Observations : {n}")
    print(f"  • Training Set       : {len(train_df)} rows ({train_ratio*100:.1f}%) | {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
    print(f"  • Validation Set     : {len(val_df)} rows ({val_ratio*100:.1f}%) | {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}")
    print(f"  • Testing Set        : {len(test_df)} rows ({(1-train_ratio-val_ratio)*100:.1f}%) | {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")
    
    return splits


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    model_dir: Optional[Path] = None
) -> lgb.LGBMRegressor:
    """Train LightGBM Regressor with early stopping on validation set.
    
    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_val: Validation feature matrix.
        y_val: Validation target vector.
        params: Optional hyperparameter dictionary.
        model_dir: Directory to save serialized model.
        
    Returns:
        Fitted LGBMRegressor instance.
    """
    if model_dir is None:
        model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    default_params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 20,
        "random_state": 42,
        "objective": "regression",
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
        
    print(f"\n========================================================")
    print(f"[*] Training LightGBM 24h Soil Moisture Regressor")
    print(f"========================================================")
    print(f"  • Estimators (Max)    : {default_params['n_estimators']}")
    print(f"  • Learning Rate       : {default_params['learning_rate']}")
    print(f"  • Max Tree Depth      : {default_params['max_depth']}")
    
    model = lgb.LGBMRegressor(**default_params)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0) # Suppress verbose per-iteration printouts
    ]
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric="rmse",
        callbacks=callbacks
    )
    
    best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else default_params["n_estimators"]
    print(f"  [+] Optimal Iteration reached: {best_iter}")
    
    # Save trained model artifact
    save_path = model_dir / "lightgbm_earth_moisture_model.joblib"
    joblib.dump({"model": model, "feature_names": list(X_train.columns)}, save_path)
    print(f"  [+] Serialized model saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.preprocessing import combine_and_clean_datasets
    from src.hampel_filter import run_hampel_pipeline
    from src.ceemdan import run_ceemdan_pipeline
    from src.reconstruction import run_reconstruction_pipeline
    from src.feature_engineering import generate_features
    
    raw_24, raw_25 = load_all_raw_data()
    _, df_clean = combine_and_clean_datasets(raw_24, raw_25)
    df_hampel = run_hampel_pipeline(df_clean)
    df_imfs, imfs = run_ceemdan_pipeline(df_hampel)
    df_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)
    df_ready, features, target = generate_features(df_recon)
    
    splits = chronological_train_val_test_split(df_ready, features, target)
    X_tr, y_tr, _ = splits["train"]
    X_v, y_v, _ = splits["val"]
    model = train_lightgbm_model(X_tr, y_tr, X_v, y_v)
