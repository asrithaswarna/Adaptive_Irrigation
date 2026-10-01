"""Hampel Filter Module for Outlier Detection and Signal Cleaning.

Implements Hampel identifier:
- Uses a rolling window of size 2*window_size + 1
- Computes rolling median and Median Absolute Deviation (MAD)
- Flags observations exceeding n_sigma * 1.4826 * MAD as outliers
- Replaces detected outliers with the local rolling median
- Preserves original raw signal alongside cleaned signal and outlier boolean flags
- Generates high-resolution visualization for project reports and demonstrations
"""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def apply_hampel_filter(
    series: pd.Series,
    window_size: int = 12,
    n_sigma: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Apply Hampel filter to detect and replace outliers in a 1D time series.
    
    Args:
        series: Pandas Series representing the target signal (e.g., earth_humidity).
        window_size: Half-window size (total window = 2 * window_size + 1). Default 12 (approx 24-hr window for hourly data).
        n_sigma: Number of standard deviations threshold (default 3.0).
        
    Returns:
        Tuple of (cleaned_series, outlier_flag_series).
    """
    signal = series.copy()
    cleaned = signal.copy()
    
    # 1. Rolling Median
    rolling_median = signal.rolling(window=2 * window_size + 1, center=True, min_periods=1).median()
    
    # 2. Rolling Median Absolute Deviation (MAD)
    # MAD = median(|x_i - median(x)|)
    diff = (signal - rolling_median).abs()
    rolling_mad = diff.rolling(window=2 * window_size + 1, center=True, min_periods=1).median()
    
    # 3. Scale factor for normal distribution consistency (1.4826)
    threshold = n_sigma * 1.4826 * rolling_mad
    
    # Avoid zero threshold in flat regions
    threshold = threshold.replace(0, 1e-6)
    
    # 4. Outlier detection
    difference_from_median = (signal - rolling_median).abs()
    outlier_mask = difference_from_median > threshold
    
    # 5. Outlier replacement with local rolling median
    cleaned[outlier_mask] = rolling_median[outlier_mask]
    
    return cleaned, outlier_mask


def run_hampel_pipeline(
    df: pd.DataFrame,
    target_col: str = "earth_humidity",
    window_size: int = 12,
    n_sigma: float = 3.0,
    output_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Run Hampel filter on the target column of a DataFrame and save results.
    
    Args:
        df: Input DataFrame containing timestamp and target_col.
        target_col: Target column name (default 'earth_humidity').
        window_size: Half window size for Hampel filtering.
        n_sigma: Sigma multiplier threshold.
        output_dir: Path to directory for saving processed CSV.
        plot_dir: Path to directory for saving visualization plot.
        
    Returns:
        DataFrame with raw target preserved, cleaned target, and outlier flag.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    if plot_dir is None:
        plot_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n========================================================")
    print(f"[*] Executing Hampel Outlier Detection Pipeline")
    print(f"========================================================")
    print(f"  • Target Column       : {target_col}")
    print(f"  • Half-Window Size    : {window_size} (Effective window: {2 * window_size + 1} steps)")
    print(f"  • Sigma Threshold     : {n_sigma} (MAD scale: {n_sigma * 1.4826:.4f})")
    
    result_df = df.copy()
    raw_col_name = f"{target_col}_raw"
    cleaned_col_name = f"{target_col}_hampel_cleaned"
    flag_col_name = f"{target_col}_is_outlier"
    
    # Preserve raw signal
    result_df[raw_col_name] = result_df[target_col].copy()
    
    # Apply filter
    cleaned_signal, outlier_mask = apply_hampel_filter(
        result_df[target_col],
        window_size=window_size,
        n_sigma=n_sigma
    )
    
    result_df[cleaned_col_name] = cleaned_signal
    result_df[flag_col_name] = outlier_mask
    
    # Overwrite target_col with cleaned version for downstream steps while raw is preserved in raw_col_name
    result_df[target_col] = cleaned_signal
    
    outlier_count = int(outlier_mask.sum())
    outlier_pct = (outlier_count / len(result_df)) * 100
    print(f"  [+] Outliers Detected : {outlier_count} / {len(result_df)} points ({outlier_pct:.2f}%)")
    
    # Generate visualization
    plot_path = plot_dir / "01_hampel_filter_comparison.png"
    generate_hampel_plot(result_df, target_col, raw_col_name, flag_col_name, plot_path)
    
    # Save processed CSV
    csv_path = output_dir / "hampel_data.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  [+] Saved Hampel processed dataset to: {csv_path}")
    
    return result_df


def generate_hampel_plot(
    df: pd.DataFrame,
    target_col: str,
    raw_col: str,
    flag_col: str,
    save_path: Path
) -> None:
    """Generate and save publication-grade Hampel filtering visual comparison."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # Top Plot: Signal vs Cleaned Signal & Outliers
    ax1.plot(df["timestamp"], df[raw_col], color="#94a3b8", alpha=0.7, label="Original Raw Signal", linewidth=1.2)
    ax1.plot(df["timestamp"], df[target_col], color="#0284c7", label="Hampel Cleaned Signal", linewidth=1.5)
    
    outliers = df[df[flag_col]]
    if not outliers.empty:
        ax1.scatter(
            outliers["timestamp"],
            outliers[raw_col],
            color="#ef4444",
            s=40,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
            label=f"Detected Outliers (n={len(outliers)})"
        )
        
    ax1.set_title("Hampel Outlier Filtering: Soil/Earth Moisture Time Series", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Earth Moisture (%)", fontsize=11, fontweight="semibold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Bottom Plot: Residual Outlier Magnitude
    residuals = (df[raw_col] - df[target_col]).abs()
    ax2.plot(df["timestamp"], residuals, color="#e11d48", linewidth=1.0, label="|Raw - Cleaned| (Noise Deviation)")
    ax2.set_title("Absolute Outlier Correction Magnitude", fontsize=12, fontweight="semibold")
    ax2.set_xlabel("Date / Time", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Deviation", fontsize=11, fontweight="semibold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved Hampel comparison plot to: {save_path}")


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.preprocessing import combine_and_clean_datasets
    
    raw_24, raw_25 = load_all_raw_data()
    _, df_clean = combine_and_clean_datasets(raw_24, raw_25)
    df_hampel = run_hampel_pipeline(df_clean)
