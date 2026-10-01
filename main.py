"""End-to-End Pipeline Execution Driver for Adaptive Irrigation Management System.

Project Title:
"Adaptive Irrigation Management Through Environmental Condition-Based Water Requirement Estimation"

Subtitle:
"Signal Processing and Machine Learning Based Soil Moisture Prediction"

Executes all 12 Phases:
1. Dynamic Data Ingestion
2. Chronological Preprocessing & Cadence Verification
3. Hampel Outlier Filtering
4. CEEMDAN Signal Decomposition
5. IMF Analysis & Selective Reconstruction
6. 24-Hour-Ahead Feature Engineering
7. Chronological Train/Val/Test Split
8. LightGBM Regressor Training
9. Performance Evaluation (MAE, RMSE, R², MAPE)
10. SHAP Global & Local Explainability
11. Transparent Irrigation Recommendation
12. Final Demonstration Summary & Artifacts Manifest
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data, load_abnormal_dataset, find_abnormal_dataset_file
from src.preprocessing import combine_and_clean_datasets, preprocess_single_dataset
from src.hampel_filter import run_hampel_pipeline
from src.ceemdan import run_ceemdan_pipeline
from src.reconstruction import run_reconstruction_pipeline
from src.feature_engineering import generate_features
from src.model import chronological_train_val_test_split, train_lightgbm_model
from src.evaluation import run_evaluation_pipeline
from src.explainability import run_shap_analysis
from src.recommendation import generate_irrigation_advisories


def run_full_pipeline(dataset_mode: str = "auto"):
    start_time = time.time()
    
    print("=" * 80)
    print("  ADAPTIVE IRRIGATION MANAGEMENT THROUGH ENVIRONMENTAL ESTIMATION")
    print("  Signal Processing & Machine Learning Based Soil Moisture Forecasting")
    print("  End-to-End Execution Pipeline (Phases 1 to 12)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # PHASE 1: Data Discovery & Loading
    # -------------------------------------------------------------
    print("\n[PHASE 1] Dynamic Data Discovery & Loading...")
    print("-" * 80)
    
    # Check if abnormal mode is requested or auto-detected
    abnormal_file = find_abnormal_dataset_file(PROJECT_ROOT)
    use_abnormal = False
    
    if dataset_mode.lower() in ["abnormal", "--abnormal", "-a"]:
        use_abnormal = True
    elif dataset_mode.lower() == "auto" and abnormal_file is not None and any("abnormal" in arg.lower() for arg in sys.argv):
        use_abnormal = True
        
    if use_abnormal:
        print(f"[*] Mode Selected: Abnormal Sensor Test Dataset Mode ({abnormal_file.name})")
        df_raw = load_abnormal_dataset(abnormal_file, base_dir=PROJECT_ROOT)
        
        print("\n[PHASE 2] Schema Standardization, Physical Bounds Sanity & Preprocessing...")
        print("-" * 80)
        combined_raw, df_cleaned = preprocess_single_dataset(
            df=df_raw,
            label="Abnormal_Sensor_Data",
            output_dir=PROJECT_ROOT / "data" / "processed"
        )
    else:
        print("[*] Mode Selected: Standard 2024 & 2025 Multi-Sheet Excel Datasets")
        df_2024, df_2025 = load_all_raw_data(base_dir=PROJECT_ROOT)
        
        print("\n[PHASE 2] Schema Standardization, Chronological Alignment & Missing Value Handling...")
        print("-" * 80)
        combined_raw, df_cleaned = combine_and_clean_datasets(
            df_2024=df_2024,
            df_2025=df_2025,
            output_dir=PROJECT_ROOT / "data" / "processed"
        )
    
    # -------------------------------------------------------------
    # PHASE 3 & 4: Hampel Outlier Filtering
    # -------------------------------------------------------------
    print("\n[PHASE 3 & 4] Hampel Outlier Detection & Signal Cleaning...")
    print("-" * 80)
    df_hampel = run_hampel_pipeline(
        df=df_cleaned,
        target_col="earth_humidity",
        window_size=12,
        n_sigma=3.0,
        output_dir=PROJECT_ROOT / "data" / "processed",
        plot_dir=PROJECT_ROOT / "outputs" / "plots"
    )
    
    # -------------------------------------------------------------
    # PHASE 5: CEEMDAN Signal Decomposition
    # -------------------------------------------------------------
    print("\n[PHASE 5] CEEMDAN Signal Decomposition into IMFs...")
    print("-" * 80)
    df_imfs, imfs = run_ceemdan_pipeline(
        df=df_hampel,
        target_col="earth_humidity",
        max_imfs=7,
        trials=30,
        plot_dir=PROJECT_ROOT / "outputs" / "plots"
    )
    
    # -------------------------------------------------------------
    # PHASE 6: IMF Analysis & Selective Signal Reconstruction
    # -------------------------------------------------------------
    print("\n[PHASE 6] Quantitative IMF Analysis & Selective Signal Reconstruction...")
    print("-" * 80)
    df_recon, reconstructed_signal, imf_summary = run_reconstruction_pipeline(
        df=df_imfs,
        imfs=imfs,
        target_col="earth_humidity",
        correlation_threshold=0.10,
        energy_threshold_pct=1.0,
        output_dir=PROJECT_ROOT / "data" / "processed",
        plot_dir=PROJECT_ROOT / "outputs" / "plots",
        report_dir=PROJECT_ROOT / "outputs" / "reports"
    )
    
    # -------------------------------------------------------------
    # PHASE 7: 24-Hour-Ahead Feature Engineering
    # -------------------------------------------------------------
    print("\n[PHASE 7] 24-Hour-Ahead Target Construction & Feature Engineering...")
    print("-" * 80)
    df_ready, feature_cols, target_name = generate_features(
        df=df_recon,
        target_col="earth_humidity",
        horizon_hours=24
    )
    
    # -------------------------------------------------------------
    # PHASE 8: Chronological Splitting & LightGBM Model Training
    # -------------------------------------------------------------
    print("\n[PHASE 8] Chronological Partitioning & LightGBM Regressor Training...")
    print("-" * 80)
    splits = chronological_train_val_test_split(
        df=df_ready,
        feature_cols=feature_cols,
        target_col=target_name,
        train_ratio=0.70,
        val_ratio=0.15
    )
    
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    
    model = train_lightgbm_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=PROJECT_ROOT / "models"
    )
    
    # -------------------------------------------------------------
    # PHASE 9: Model Evaluation & Residual Diagnostics
    # -------------------------------------------------------------
    print("\n[PHASE 9] Quantitative Performance Evaluation (MAE, RMSE, R², MAPE)...")
    print("-" * 80)
    metrics = run_evaluation_pipeline(
        model=model,
        splits=splits,
        output_dir=PROJECT_ROOT / "outputs" / "reports",
        plot_dir=PROJECT_ROOT / "outputs" / "plots"
    )
    
    # -------------------------------------------------------------
    # PHASE 10: SHAP Feature Explainability
    # -------------------------------------------------------------
    print("\n[PHASE 10] TreeSHAP Interpretability & Feature Attribution Analysis...")
    print("-" * 80)
    X_test, y_test, df_test = splits["test"]
    shap_values = run_shap_analysis(
        model=model,
        X_sample=X_test,
        max_display=15,
        plot_dir=PROJECT_ROOT / "outputs" / "plots"
    )
    
    # -------------------------------------------------------------
    # PHASE 11: Transparent Irrigation Recommendation
    # -------------------------------------------------------------
    print("\n[PHASE 11] Decision-Support Irrigation Recommendation Generation...")
    print("-" * 80)
    test_preds = model.predict(X_test)
    advisories_df = generate_irrigation_advisories(
        df_test=df_test,
        predictions=test_preds,
        report_dir=PROJECT_ROOT / "outputs" / "reports",
        num_recent_samples=10
    )
    
    # -------------------------------------------------------------
    # PHASE 12: Pipeline Completion & Layman-Friendly Summary
    # -------------------------------------------------------------
    elapsed = time.time() - start_time
    test_metrics = metrics["test"]
    r2_pct = test_metrics["R2_Score"] * 100.0
    years = sorted(df_cleaned["source_year"].dropna().astype(str).unique())
    year_label = " & ".join(years) if years else "available"
    coverage_start = df_cleaned["timestamp"].min()
    coverage_end = df_cleaned["timestamp"].max()
    coverage_days = (coverage_end - coverage_start).total_seconds() / (24 * 60 * 60)
    train_start = splits["train"][2]["timestamp"].min()
    train_end = splits["train"][2]["timestamp"].max()
    test_start = splits["test"][2]["timestamp"].min()
    test_end = splits["test"][2]["timestamp"].max()
    outlier_count = int(df_hampel["earth_humidity_is_outlier"].sum())
    outlier_total = len(df_hampel)
    plot_count = len(list((PROJECT_ROOT / "outputs" / "plots").glob("*.png")))

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_features = shap_values.feature_names or list(X_test.columns)
    top_features = [
        str(feature)
        for feature, _ in sorted(
            zip(shap_features, shap_importance),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
    ]

    advisory_counts = advisories_df["Status"].value_counts()
    advisory_total = len(advisories_df)

    def advisory_label(status: str) -> str:
        labels = {
            "CRITICAL DEFICIT": "Critically Dry",
            "MOISTURE DEFICIT": "Moderate Deficit",
            "DEFICIT (RAIN EXPECTED)": "Deficit (Rain Expected)",
            "OPTIMAL MOISTURE": "Optimal",
            "SATURATION / EXCESS": "Saturated",
        }
        return labels.get(status, status.title())

    advisory_actions = {
        "CRITICAL DEFICIT": "Immediate watering required - saves crops",
        "MOISTURE DEFICIT": "Scheduled watering recommended",
        "DEFICIT (RAIN EXPECTED)": "Hold irrigation and monitor rain",
        "OPTIMAL MOISTURE": "No irrigation needed - saves water",
        "SATURATION / EXCESS": "Do not water - prevents root damage",
    }
    
    print("\n" + "=" * 80)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("  [TECHNICAL METRICS SUMMARY]")
    print(f"  • Total Pipeline Runtime       : {elapsed:.2f} seconds")
    print(f"  • Holdout Test Set MAE         : {test_metrics['MAE']:.4f} % moisture")
    print(f"  • Holdout Test Set RMSE        : {test_metrics['RMSE']:.4f} % moisture")
    print(f"  • Holdout Test Set R^2 Score   : {test_metrics['R2_Score']:.4f} ({r2_pct:.1f}% variance explained)")
    print(f"  • Serialized Model Saved       : models/lightgbm_earth_moisture_model.joblib")
    print(f"  • Visual Charts ({plot_count} Plots)      : outputs/plots/")
    print(f"  • Reports & CSV Logs           : outputs/reports/ & data/processed/")
    
    print("\n" + "-" * 80)
    print("  [LAYMAN / SIMPLE TERMS SUMMARY OF WHAT HAPPENED]")
    print("-" * 80)
    print("  1. DATA LOADED:")
    print(f"     Combined {year_label} sensor records into {len(df_cleaned):,} clean hourly time-steps.")
    print(f"     Covers {coverage_days / 365.25:.2f} continuous years ({coverage_start:%Y-%m-%d} to {coverage_end:%Y-%m-%d}).")
    print()
    print("  2. SENSOR CLEANING & SIGNAL PROCESSING:")
    print(f"     • Fixed {outlier_count:,} of {outlier_total:,} suspected sensor glitches ({outlier_count / outlier_total * 100:.1f}%) using the Hampel Filter.")
    print("     • Separated seasonal climate trends from short-term noise (CEEMDAN Decomposition).")
    print()
    print("  3. 24-HOUR-AHEAD AI PREDICTION (LightGBM):")
    print(f"     • The AI was trained on past history ({train_start:%Y-%m-%d} to {train_end:%Y-%m-%d}) and tested on unseen future data ({test_start:%Y-%m-%d} to {test_end:%Y-%m-%d}).")
    print(f"     • 24-Hour Prediction Accuracy: ~{r2_pct:.1f}% reliable in tracking real soil moisture.")
    print(f"     • Typical prediction margin: +/- {test_metrics['MAE']:.1f}% volumetric moisture.")
    print()
    print("  4. TOP FACTORS INFLUENCING SOIL MOISTURE (SHAP):")
    print(f"     • Strongest measured drivers were: {', '.join(top_features)}.")
    print("     • Driver ranking is calculated from mean absolute SHAP impact on the test set.")
    print()
    print("  5. ACTIONABLE IRRIGATION ADVISORY FOR FARMERS:")
    for status, count in advisory_counts.items():
        percentage = count / advisory_total * 100 if advisory_total else 0.0
        action = advisory_actions.get(status, "Review the generated advisory details")
        print(f"     • {percentage:.1f}% {advisory_label(status)} ({action}).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_full_pipeline(dataset_mode=mode)
