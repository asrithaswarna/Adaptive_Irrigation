"""IMF Analysis and Selective Signal Reconstruction Module.

Performs:
1. Quantitative IMF characterization:
   - Variance, Standard Deviation, Signal Energy, Energy Percentage (%)
   - Pearson Correlation with the cleaned target signal
2. Measurable, rule-based IMF selection to filter high-frequency stochastic noise
3. Selective signal reconstruction
4. High-resolution visualization comparing cleaned signal, selected IMFs, and reconstructed signal
5. Exporting reconstructed dataset and IMF summary reports
"""

from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def analyze_imf_characteristics(
    original_signal: np.ndarray,
    imfs: np.ndarray,
    report_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Compute mathematical characteristics for each IMF and residue.
    
    Args:
        original_signal: 1D array of cleaned target signal.
        imfs: 2D array of extracted IMFs (shape: [num_components, n_samples]).
        report_dir: Path to directory for saving CSV report.
        
    Returns:
        DataFrame summarizing characteristics of all components.
    """
    if report_dir is None:
        report_dir = Path(__file__).resolve().parent.parent / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    num_components = imfs.shape[0]
    total_energy = np.sum(imfs ** 2)
    
    records = []
    for i in range(num_components):
        comp = imfs[i]
        name = f"IMF {i+1}" if i < num_components - 1 else "Residue"
        
        var = float(np.var(comp))
        std = float(np.std(comp))
        energy = float(np.sum(comp ** 2))
        energy_pct = float((energy / total_energy) * 100.0) if total_energy > 0 else 0.0
        
        # Pearson correlation with original target signal
        corr, p_val = pearsonr(comp, original_signal)
        
        records.append({
            "Component": name,
            "Variance": round(var, 4),
            "Std_Dev": round(std, 4),
            "Energy": round(energy, 2),
            "Energy_Percent(%)": round(energy_pct, 2),
            "Pearson_Correlation": round(float(corr), 4),
            "P_Value": round(float(p_val), 6)
        })
        
    summary_df = pd.DataFrame(records)
    
    print(f"\n========================================================")
    print(f"[*] Quantitative IMF Analysis Summary Table")
    print(f"========================================================")
    print(summary_df.to_string(index=False))
    
    # Save report
    report_path = report_dir / "imf_characteristics.csv"
    summary_df.to_csv(report_path, index=False)
    print(f"  [+] Saved IMF characteristics report to: {report_path}")
    
    return summary_df


def select_imfs_rule_based(
    summary_df: pd.DataFrame,
    correlation_threshold: float = 0.10,
    energy_threshold_pct: float = 1.0
) -> Tuple[List[int], List[int]]:
    """Select informative IMFs using a measurable, transparent rule.
    
    Selection Criteria:
    - Include Residue (carries long-term environmental baseline trend).
    - Include IMFs where |Pearson Correlation| >= correlation_threshold OR Energy% >= energy_threshold_pct.
    - Discard high-frequency noise modes (typically IMF 1 if it has weak correlation and low energy).
    
    Args:
        summary_df: Output from analyze_imf_characteristics.
        correlation_threshold: Minimum absolute correlation threshold (default 0.10).
        energy_threshold_pct: Minimum energy percentage threshold (default 1.0%).
        
    Returns:
        Tuple of (selected_indices, discarded_indices).
    """
    selected_indices = []
    discarded_indices = []
    num_components = len(summary_df)
    
    print(f"\n[*] Evaluating IMF Selection Rule:")
    print(f"  • Criterion: (|Correlation| >= {correlation_threshold}) OR (Energy% >= {energy_threshold_pct}%) OR (Residue)")
    
    for idx, row in summary_df.iterrows():
        comp_name = row["Component"]
        corr_abs = abs(row["Pearson_Correlation"])
        energy_pct = row["Energy_Percent(%)"]
        is_residue = (idx == num_components - 1)
        
        if is_residue or corr_abs >= correlation_threshold or energy_pct >= energy_threshold_pct:
            selected_indices.append(idx)
            reason = "Residue Trend" if is_residue else f"|r|={corr_abs:.3f} >= {correlation_threshold} or E%={energy_pct:.1f}%"
            print(f"  [+] Selected  : {comp_name:10s} (Reason: {reason})")
        else:
            discarded_indices.append(idx)
            print(f"  [-] Discarded : {comp_name:10s} (High-frequency stochastic noise, |r|={corr_abs:.3f}, E%={energy_pct:.1f}%)")
            
    # Guarantee at least one component selected
    if not selected_indices:
        selected_indices = [num_components - 1]
        
    return selected_indices, discarded_indices


def reconstruct_signal(
    imfs: np.ndarray,
    selected_indices: List[int]
) -> np.ndarray:
    """Sum selected IMFs to form the reconstructed, denoised signal."""
    selected_imfs = imfs[selected_indices, :]
    reconstructed = np.sum(selected_imfs, axis=0)
    return reconstructed


def run_reconstruction_pipeline(
    df: pd.DataFrame,
    imfs: np.ndarray,
    target_col: str = "earth_humidity",
    correlation_threshold: float = 0.10,
    energy_threshold_pct: float = 1.0,
    output_dir: Optional[Path] = None,
    plot_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Execute complete IMF analysis and selective signal reconstruction pipeline.
    
    Args:
        df: Input DataFrame with cleaned target series.
        imfs: 2D array of extracted IMFs.
        target_col: Target column name.
        correlation_threshold: Correlation cutoff for IMF selection.
        energy_threshold_pct: Energy % cutoff for IMF selection.
        output_dir: Path for saving reconstructed CSV.
        plot_dir: Path for saving plot.
        report_dir: Path for saving characteristic table.
        
    Returns:
        Tuple of (df_reconstructed, reconstructed_signal, summary_df).
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    if plot_dir is None:
        plot_dir = Path(__file__).resolve().parent.parent / "outputs" / "plots"
    if report_dir is None:
        report_dir = Path(__file__).resolve().parent.parent / "outputs" / "reports"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    signal = df[target_col].to_numpy(dtype=float)
    
    # 1. IMF Characterization
    summary_df = analyze_imf_characteristics(signal, imfs, report_dir)
    
    # 2. Rule-Based Selection
    selected_idx, discarded_idx = select_imfs_rule_based(
        summary_df,
        correlation_threshold=correlation_threshold,
        energy_threshold_pct=energy_threshold_pct
    )
    
    # 3. Signal Reconstruction
    reconstructed_signal = reconstruct_signal(imfs, selected_idx)
    noise_component = signal - reconstructed_signal
    
    # Add to DataFrame
    df_recon = df.copy()
    df_recon[f"{target_col}_reconstructed"] = reconstructed_signal
    df_recon[f"{target_col}_extracted_noise"] = noise_component
    
    # 4. Generate Plot
    plot_path = plot_dir / "03_signal_reconstruction.png"
    generate_reconstruction_plot(
        df["timestamp"],
        signal,
        imfs,
        selected_idx,
        reconstructed_signal,
        noise_component,
        plot_path
    )
    
    # 5. Save dataset
    csv_path = output_dir / "reconstructed_data.csv"
    df_recon.to_csv(csv_path, index=False)
    print(f"\n[+] Saved reconstructed dataset to: {csv_path}")
    
    return df_recon, reconstructed_signal, summary_df


def generate_reconstruction_plot(
    timestamps: pd.Series,
    original_signal: np.ndarray,
    imfs: np.ndarray,
    selected_indices: List[int],
    reconstructed_signal: np.ndarray,
    noise_component: np.ndarray,
    save_path: Path
) -> None:
    """Generate visual comparison of Cleaned vs Reconstructed Signal."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    
    # Panel 1: Original Cleaned vs Reconstructed
    ax1.plot(timestamps, original_signal, color="#94a3b8", alpha=0.8, linewidth=1.2, label="Hampel Cleaned Signal")
    ax1.plot(timestamps, reconstructed_signal, color="#059669", linewidth=1.6, label=f"Reconstructed Signal (Sum of Selected IMFs: {selected_indices})")
    ax1.set_title("Selective Signal Reconstruction via Informative IMFs", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Earth Moisture (%)", fontsize=11, fontweight="semibold")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Selected IMFs Overlay
    palette = ["#0284c7", "#0d9488", "#16a34a", "#ca8a04", "#ea580c", "#dc2626", "#7c3aed"]
    num_comp = imfs.shape[0]
    for idx in selected_indices:
        label = f"IMF {idx+1}" if idx < num_comp - 1 else "Residue"
        color = palette[idx % len(palette)]
        ax2.plot(timestamps, imfs[idx], label=label, color=color, linewidth=1.1, alpha=0.85)
    ax2.set_title("Selected Informative Component Modes (IMFs + Residue)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Amplitude", fontsize=11, fontweight="semibold")
    ax2.legend(loc="upper right", ncol=4, frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # Panel 3: Filtered Noise Deviation
    ax3.plot(timestamps, noise_component, color="#dc2626", linewidth=0.9, label="Filtered High-Frequency Noise Residual")
    ax3.set_title("Filtered Noise Residual (Raw - Reconstructed)", fontsize=11, fontweight="semibold")
    ax3.set_xlabel("Date / Time", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Noise", fontsize=11, fontweight="semibold")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [+] Saved selective reconstruction plot to: {save_path}")


if __name__ == "__main__":
    from src.data_loader import load_all_raw_data
    from src.preprocessing import combine_and_clean_datasets
    from src.hampel_filter import run_hampel_pipeline
    from src.ceemdan import run_ceemdan_pipeline
    
    raw_24, raw_25 = load_all_raw_data()
    _, df_clean = combine_and_clean_datasets(raw_24, raw_25)
    df_hampel = run_hampel_pipeline(df_clean)
    df_imfs, imfs = run_ceemdan_pipeline(df_hampel)
    df_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)
