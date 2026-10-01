"""Evaluation Module for 24-Hour-Ahead Soil Moisture Forecasting.

Calculates:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

Generates:
- Actual vs. Predicted Time-Series Curves (outputs/plots/04_actual_vs_predicted.png)
- Residual Distribution & Error Diagnostics (outputs/plots/05_residuals_analysis.png)
- Metric JSON Report (outputs/reports/evaluation_summary.json)
"""

import json
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "Test"
) -> Dict[str, float]:
    """Compute standard regression metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    # Safe MAPE calculation avoiding division by zero
    nonzero_mask = np.abs(y_true) > 1e-4
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        mape = 0.0
        
    metrics = {
        "Split": split_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
        "MAPE(%)": round(mape, 2),
    }
    return metrics


def run_evaluation_pipeline(
    model,
    splits: Dict,
    output_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None
) -> Dict:
    """Execute complete model evaluation, reporting, and visualization.
    
    Args:
        model: Fitted model object.
        splits: Dictionary containing train, val, and test splits.
        output_dir: Path for saving JSON metrics report.
        plot_dir: Path for saving plots.
        
    Returns:
        Dictionary containing metric summaries across all splits.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "reports"
    if plot_dir is None:
        plot_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n========================================================")
    print(f"[*] Evaluating LightGBM Model Performance")
    print(f"========================================================")
    
    all_metrics = {}
    
    # Evaluate each partition
    for split_key in ["train", "val", "test"]:
        X, y, df_split = splits[split_key]
        preds = model.predict(X)
        metrics = evaluate_predictions(y.to_numpy(), preds, split_name=split_key.capitalize())
        all_metrics[split_key] = metrics
        
        print(f"  [{split_key.upper():5s} SET] MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R^2: {metrics['R2_Score']:.4f} | MAPE: {metrics['MAPE(%)']:.2f}%")
        if split_key == "test":
            print(f"  -> Plain English Meaning: AI predicts soil moisture 24h ahead with {metrics['R2_Score']*100:.1f}% pattern correlation (Avg error: +/- {metrics['MAE']:.1f}%).")
        
    # Generate Plots on the holdout Test Set
    X_test, y_test, df_test = splits["test"]
    test_preds = model.predict(X_test)
    test_timestamps = df_test["timestamp"]
    
    # 1. Actual vs Predicted Curve
    plot_path_pred = plot_dir / "04_actual_vs_predicted.png"
    generate_actual_vs_pred_plot(test_timestamps, y_test.to_numpy(), test_preds, all_metrics["test"], plot_path_pred)
    
    # 2. Residuals and Error Diagnostics
    plot_path_resid = plot_dir / "05_residuals_analysis.png"
    generate_residuals_plot(y_test.to_numpy(), test_preds, plot_path_resid)
    
    # Save JSON report
    json_path = output_dir / "evaluation_summary.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n[+] Saved evaluation metrics report to: {json_path}")
    
    return all_metrics


def generate_actual_vs_pred_plot(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict,
    save_path: Path
) -> None:
    """Generate high-resolution Actual vs Predicted comparison plot."""
    plt.figure(figsize=(15, 6))
    
    plt.plot(timestamps, y_true, label="Actual Ground Truth (24h Ahead)", color="#0f172a", linewidth=1.5, alpha=0.85)
    plt.plot(timestamps, y_pred, label="LightGBM Predicted (24h Ahead)", color="#0284c7", linewidth=1.5, linestyle="--")
    
    metric_text = f"Test MAE: {metrics['MAE']:.3f} | Test RMSE: {metrics['RMSE']:.3f} | R²: {metrics['R2_Score']:.3f}"
    plt.title(f"24-Hour-Ahead Soil/Earth Moisture Forecasting Performance\n({metric_text})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date / Time (Chronological Test Partition)", fontsize=11, fontweight="semibold")
    plt.ylabel("Soil/Earth Moisture (%)", fontsize=11, fontweight="semibold")
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Actual vs Predicted plot to: {save_path}")


def generate_residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path
) -> None:
    """Generate residual error diagnostics plot (distribution + scatter)."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, color="#0284c7", alpha=0.4, edgecolors="none", s=25)
    ax1.axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs. Predicted Values", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Moisture (%)", fontsize=11)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Residual Distribution Histogram
    ax2.hist(residuals, bins=40, color="#10b981", edgecolor="black", alpha=0.75, density=True)
    ax2.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)
    ax2.set_title(f"Residual Error Distribution (Mean={np.mean(residuals):.3f}, Std={np.std(residuals):.3f})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Prediction Error", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Residuals analysis plot to: {save_path}")


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.preprocessing import combine_and_clean_datasets
    from src.hampel_filter import run_hampel_pipeline
    from src.ceemdan import run_ceemdan_pipeline
    from src.reconstruction import run_reconstruction_pipeline
    from src.feature_engineering import generate_features
    from src.model import chronological_train_val_test_split, train_lightgbm_model
    
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
    
    metrics = run_evaluation_pipeline(model, splits)
