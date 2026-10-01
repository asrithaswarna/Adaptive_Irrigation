"""SHAP Explainability Module for 24-Hour Soil Moisture Predictions.

Generates:
1. TreeSHAP Global Feature Importance Bar Plot (outputs/plots/06_shap_importance.png)
2. SHAP Summary Beeswarm Plot showing positive/negative contribution dynamics (outputs/plots/07_shap_summary.png)
3. Local Waterfall Explanation for individual forecast instances (outputs/plots/08_shap_waterfall.png)
4. Terminal interpretation of dominant environmental and signal drivers.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def run_shap_analysis(
    model,
    X_sample: pd.DataFrame,
    max_display: int = 15,
    plot_dir: Optional[Path] = None
) -> shap.Explanation:
    """Compute TreeSHAP values and generate publication-ready explainability plots.
    
    Args:
        model: Trained LightGBM model.
        X_sample: DataFrame of features (typically holdout test samples).
        max_display: Maximum number of top features to display on plots.
        plot_dir: Directory where SHAP visual plots will be saved.
        
    Returns:
        SHAP Explanation object.
    """
    if plot_dir is None:
        plot_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n========================================================")
    print(f"[*] Computing SHAP (SHapley Additive exPlanations)")
    print(f"========================================================")
    
    # Subsample if dataset is large to maintain speed while preserving fidelity
    if len(X_sample) > 500:
        sample_subset = X_sample.sample(n=500, random_state=42)
    else:
        sample_subset = X_sample
        
    print(f"  • Calculating TreeSHAP values over {len(sample_subset)} test instances...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample_subset)
    
    # 1. Feature Importance Bar Plot
    plot_path_imp = plot_dir / "06_shap_importance.png"
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(plot_path_imp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved SHAP Feature Importance plot to: {plot_path_imp}")
    
    # 2. SHAP Beeswarm Summary Plot
    plot_path_sum = plot_dir / "07_shap_summary.png"
    plt.figure(figsize=(11, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title("SHAP Summary Plot: Feature Impact on 24h Soil Moisture Prediction", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(plot_path_sum, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved SHAP Beeswarm Summary plot to: {plot_path_sum}")
    
    # 3. Individual Instance Waterfall Plot (Latest / Representative Sample)
    plot_path_water = plot_dir / "08_shap_waterfall.png"
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("Local Prediction Explanation: Individual Test Instance Waterfall", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(plot_path_water, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved SHAP Local Waterfall plot to: {plot_path_water}")
    
    # Compute and display top 5 features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": sample_subset.columns,
        "Mean_|SHAP|": mean_abs_shap
    }).sort_values(by="Mean_|SHAP|", ascending=False).reset_index(drop=True)
    
    print("\n  Top 5 Predictive Features identified by SHAP:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"    {idx+1}. {row['Feature']:35s} (Impact magnitude: {row['Mean_|SHAP|']:.4f})")
        
    return shap_values


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
    
    X_test, _, _ = splits["test"]
    run_shap_analysis(model, X_test)
