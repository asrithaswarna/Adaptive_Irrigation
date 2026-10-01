# Adaptive Irrigation Management Through Environmental Condition-Based Water Requirement Estimation

**Project Subtitle:** Signal Processing and Machine Learning Based Soil Moisture Prediction  
**Project Category:** Software-Only Data Science, Signal Processing & Machine Learning Pipeline  
**Target Variable:** 24-Hour-Ahead Soil/Earth Moisture (`earth_humidity`)  
**Deployment Environment:** Python 3.11+ / 3.12 compatible (VS Code / Terminal / Jupyter)

---

## 📌 1. Project Overview & Objective

Agricultural crop health and water efficiency depend heavily on anticipating soil moisture changes before severe moisture deficits or waterlogging occur. 

This is a **100% software-only project** that ingests real-world multi-year agricultural/weather-station sensor time series (2024 & 2025 datasets), filters sensor noise and physical outliers using advanced signal processing (**Hampel Filter + CEEMDAN**), extracts multi-scale temporal dynamics, trains a high-precision **LightGBM Regressor** to forecast soil moisture **24 hours into the future**, explains feature contributions using **SHAP**, and delivers **transparent rule-based irrigation recommendations** for farm management.

> [!NOTE]
> **No physical hardware** (ESP32, Arduino, soil probes, relays, LoRa, or mechanical pumps) is required. The system is designed as an intelligent **decision-support software layer**.

---

## 🔄 2. End-to-End Pipeline Architecture

The workflow follows a rigorous 12-stage sequential pipeline designed to prevent data leakage and ensure scientific reproducibility:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & DISCOVERY                      │
│   • 2024 & 2025 Multi-Sheet Excel Datasets (or Abnormal Sensor CSV)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING & CADENCE                         │
│   • Datetime Unification, Chronological Sorting & Cadence (1h Hourly)  │
│   • Physical Bounds Sanitization (Filter -999, Dropouts, Sentinels)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: HAMPEL OUTLIER FILTER                     │
│   • Rolling Median & MAD Window to repair sensor spikes & glitches     │
│   • Preserves raw signal & logs outlier boolean flags                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 2: CEEMDAN SIGNAL DECOMPOSITION                  │
│   • Complete Ensemble Empirical Mode Decomposition + Adaptive Noise    │
│   • Decomposes non-linear moisture signal into 8 IMFs + 1 Residue      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            STAGE 3: IMF ANALYSIS & SELECTIVE RECONSTRUCTION            │
│   • Computes Variance, Energy % & Pearson Correlation per IMF          │
│   • Discards stochastic high-frequency noise modes (IMFs 1-6)          │
│   • Sums informative modes (IMFs 7, 8, Residue) into Denoised Signal   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     STAGE 4: FEATURE ENGINEERING                       │
│   • 24-Hour Target Lead Shift (t + 24h)                                │
│   • Lags (t-1 to t-24), Rolling Stats (Mean, Std, Min, Max), Rain Sum │
│   • Cyclical Time Encodings (Hour Sin/Cos, Month Sin/Cos) [55 total]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STAGE 5: CHRONOLOGICAL LIGHTGBM FORECASTING                │
│   • Time-Series Split: 70% Train (Past) / 15% Val / 15% Test (Future)  │
│   • Leak-Free LightGBMRegressor with Early Stopping                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 6: EVALUATION & EXPLAINABILITY                   │
│   • Quantitative Metrics: MAE, RMSE, R² (78.8%), MAPE                  │
│   • TreeSHAP Interpretability: Global Beeswarm, Importance & Waterfall │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            STAGE 7: DECISION-SUPPORT IRRIGATION ADVISORY               │
│   • Configurable Agronomic Rules: Critical / Moderate / Optimal / Sat  │
│   • Produces actionable hourly recommendations for farm operations     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 3. File Structure & Role of Each Component

```
Demonstration/
│
├── data/
│   ├── raw/                                # Raw unprocessed sensor datasets
│   │   ├── P18_BIORO_WeatherStationData_AgriDataValue_2024.xlsx # 2024 Multi-sheet data
│   │   ├── P18_BIORO_WeatherStationData_AgriDataValue_2025.xlsx # 2025 Multi-sheet data
│   │   └── abnormal_sensor_dataset.csv     # 15,000-row stress-test anomaly dataset
│   │
│   └── processed/                          # Pipeline-generated clean datasets
│       ├── combined_data.csv               # Merged raw dataset with unified timestamps
│       ├── cleaned_data.csv                # Chronologically sorted, deduplicated & imputed
│       ├── hampel_data.csv                 # Hampel-filtered signal with outlier flags
│       └── reconstructed_data.csv          # Extracted IMFs & selectively reconstructed signal
│
├── notebooks/                              # 8 Interactive Step-by-Step Educational Notebooks
│   ├── 01_data_exploration.ipynb           # Dynamic workbook inspection & sensor stats
│   ├── 02_data_preprocessing.ipynb         # Cadence checking & missing value management
│   ├── 03_hampel_filter.ipynb              # Rolling median/MAD outlier cleaning demonstration
│   ├── 04_ceemdan_analysis.ipynb           # Multi-scale mode decomposition into IMFs
│   ├── 05_signal_reconstruction.ipynb      # IMF variance, energy & correlation selection
│   ├── 06_feature_engineering.ipynb        # 24h target shift, lag creation & cyclical time
│   ├── 07_lightgbm_prediction.ipynb        # Chronological Train/Val/Test split & LightGBM
│   └── 08_shap_analysis.ipynb              # SHAP feature attribution & irrigation recommendation
│
├── src/                                    # Modular Clean Source Code
│   ├── __init__.py                         # Python package initializer
│   ├── data_loader.py                      # Dynamic Excel/CSV loader & column normalizer
│   ├── preprocessing.py                    # Datetime parser, physical bounds sanity & gap handler
│   ├── hampel_filter.py                    # Rolling median & MAD outlier detection module
│   ├── ceemdan.py                          # CEEMDAN mode decomposition into IMFs + residue
│   ├── reconstruction.py                   # Quantitative IMF analysis & selective reconstruction
│   ├── feature_engineering.py              # 24h target construction & 55 predictive features
│   ├── model.py                            # Chronological temporal splitting & LightGBM training
│   ├── evaluation.py                       # MAE, RMSE, R², MAPE metrics & residual plots
│   ├── explainability.py                   # TreeSHAP beeswarm, feature bar & waterfall plots
│   ├── recommendation.py                   # Configurable rule-based irrigation decision support
│   └── generate_notebooks.py               # Automated Jupyter notebook generation script
│
├── models/
│   └── lightgbm_earth_moisture_model.joblib # Serialized fitted LightGBM model artifact
│
├── outputs/
│   ├── plots/                              # 8 Publication-Grade Visualizations (300 DPI)
│   │   ├── 01_hampel_filter_comparison.png # Raw vs. Cleaned signal & detected outlier flags
│   │   ├── 02_ceemdan_imfs.png             # Multi-panel decomposition plot of all IMFs
│   │   ├── 03_signal_reconstruction.png    # Selected IMFs vs. Reconstructed signal vs. Noise
│   │   ├── 04_actual_vs_predicted.png      # Holdout test set actual vs. 24h predicted curves
│   │   ├── 05_residuals_analysis.png       # Residual error distribution & scatter diagnostics
│   │   ├── 06_shap_importance.png          # Global mean |SHAP| feature importance bar chart
│   │   ├── 07_shap_summary.png             # SHAP beeswarm plot (positive/negative impacts)
│   │   └── 08_shap_waterfall.png           # Local individual sample waterfall explanation
│   │
│   └── reports/                            # Generated Metrics & CSV Advisory Logs
│       ├── evaluation_summary.json         # Train, Val, and Test numeric evaluation report
│       ├── imf_characteristics.csv         # Variance, energy %, and Pearson correlation table
│       └── irrigation_recommendations.csv  # 24h-ahead hourly irrigation advisory logs
│
├── main.py                                 # Master Pipeline Driver (Dual-Mode: Normal & Abnormal)
├── requirements.txt                        # Python package dependencies
└── README.md                               # Project execution & reproduction guide
```

---

## 🚀 4. How the Files Work Together

1. **`main.py`** is the master orchestrator. When executed, it calls each module in `src/` sequentially.
2. **`src/data_loader.py`** inspects the `data/raw/` directory, discovers the sheets dynamically, standardizes column synonyms (e.g. `airTemperature` $\rightarrow$ `air_temperature`, `earthHumidity1` $\rightarrow$ `earth_humidity`), and loads the raw data.
3. **`src/preprocessing.py`** unifies timestamps, verifies regular hourly cadence ($1\text{ hour}$), enforces physical sanity bounds (filtering unphysical sentinels like $-999$), and outputs `data/processed/cleaned_data.csv`.
4. **`src/hampel_filter.py`** scans the moisture series with a rolling window of 25 steps ($2 \times 12 + 1$), replaces local spikes/glitches with the rolling median, and saves `outputs/plots/01_hampel_filter_comparison.png`.
5. **`src/ceemdan.py`** decomposes the cleaned series into Intrinsic Mode Functions (IMFs) and saves `outputs/plots/02_ceemdan_imfs.png`.
6. **`src/reconstruction.py`** computes variance, energy percentage, and Pearson correlation for each IMF, rejects high-frequency stochastic noise (IMFs 1 to 6), and reconstructs the denoised signal from significant modes (IMFs 7, 8, Residue), saving `data/processed/reconstructed_data.csv`.
7. **`src/feature_engineering.py`** shifts the target series by $-24\text{ hours}$ to create the 24-hour-ahead target, adds multi-scale historical lags, rolling statistics (mean, std, min, max), precipitation totals, and cyclical sine/cosine time representations (55 features total).
8. **`src/model.py`** splits the dataset strictly chronologically (70% Train / 15% Validation / 15% Test) without shuffling to avoid data leakage, trains the `LightGBMRegressor` with early stopping, and serializes the model to `models/lightgbm_earth_moisture_model.joblib`.
9. **`src/evaluation.py`** computes MAE, RMSE, $R^2$, and MAPE metrics on the test partition, and saves actual vs. predicted curves (`outputs/plots/04_actual_vs_predicted.png`) and residual plots (`05_residuals_analysis.png`).
10. **`src/explainability.py`** runs `shap.TreeExplainer` over test instances, identifying the exact positive/negative contributions of features and saving SHAP plots (`06`, `07`, `08`).
11. **`src/recommendation.py`** translates the predicted moisture levels into clear, actionable agronomic advice (Critical Deficit, Moderate Deficit, Optimal Moisture, Saturation Risk) and exports `outputs/reports/irrigation_recommendations.csv`.

---

## 💻 5. Step-by-Step Execution Guide

### Step 1: Environment Setup
Ensure Python 3.11 or 3.12 is installed. Clone or navigate to the project directory:
```powershell
cd "c:\2nd Year\Odd Sem\Signal Processing\Project_SP\Demonstration"
```

### Step 2: Install Dependencies
Install all required libraries using `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Full Pipeline

#### Mode A: Standard Real-World Agricultural Datasets (2024 & 2025)
To execute the complete 12-phase pipeline on the standard 2024–2025 multi-sheet weather station datasets:
```powershell
python main.py
```
*(or explicitly: `python main.py standard`)*

#### Mode B: Stress-Test on Abnormal Sensor Dataset
To test the system against extreme anomalies (containing $-999$ sensor dropouts, abrupt jumps, frozen sensors, and drift):
```powershell
python main.py abnormal
```

### Step 4: Run Interactive Jupyter Notebooks
If you prefer an interactive walkthrough of each phase in VS Code or Jupyter Lab:
1. Open VS Code and navigate to the `notebooks/` folder.
2. Select your Python kernel.
3. Run notebooks sequentially from `01_data_exploration.ipynb` through `08_shap_analysis.ipynb`.

---

## 📊 6. Empirical Results & Verification Summary

### Quantitative Results on 2024–2025 Agricultural Datasets

| Metric / Parameter | Value | Plain English Interpretation |
| :--- | :--- | :--- |
| **Total Ingested Data** | 22,112 raw rows $\rightarrow$ 13,567 clean hourly steps | Over **1.6 continuous years** of real weather & soil sensor history. |
| **Hampel Filter Glitches Fixed** | 1,496 points (11.0%) | Sensor noise spikes removed without altering true ground moisture. |
| **CEEMDAN Decomposition** | 8 IMFs + 1 Residue | Noise modes (IMFs 1–6) discarded; significant modes (IMFs 7, 8, Residue) retained. |
| **Chronological Split** | 70% Train (9,463 rows) \| 15% Val \| 15% Test (2,028 rows) | Strictly time-ordered without shuffling to ensure zero data leakage. |
| **Holdout Test Set $R^2$ Score** | **0.7880 (78.8%)** | Strong predictive power in forecasting soil moisture 24 hours ahead. |
| **Holdout Test Set MAE** | **9.5655% moisture** | Average prediction deviation is within $\pm 9.6\%$ volumetric moisture. |
| **Holdout Test Set RMSE** | **13.8030% moisture** | Low root-mean-squared penalty across sharp seasonal transitions. |

### Top 5 Predictive Drivers Identified by SHAP:
1. **`earth_humidity`** (Current soil moisture baseline) — Impact magnitude: `24.58`
2. **`earth_humidity_reconstructed`** (Denoised CEEMDAN signal component) — Impact magnitude: `8.81`
3. **`earth_humidity_lag_1h`** (Moisture 1 hour prior) — Impact magnitude: `6.70`
4. **`imf_residue`** (Long-term climate seasonal trend) — Impact magnitude: `4.86`
5. **`earth_humidity_reconstructed_lag_1h`** (Lagged reconstructed signal) — Impact magnitude: `1.74`

### Irrigation Advisory Distribution on Test Data:
* 🟢 **39.5% Optimal Moisture (35%–65%)**: No irrigation required (conserves water).
* 🔴 **26.9% Critical Deficit (< 20%)**: Immediate irrigation required (prevents crop stress).
* 🔵 **19.2% Saturation Risk (> 65%)**: Irrigation prohibited (prevents waterlogging and root rot).
* 🟡 **14.3% Moderate Deficit (20%–35%)**: Scheduled irrigation recommended.

---

## 🛡️ 7. Handling of Abnormal Sensor Values & Extreme Weather

The pipeline features a two-tiered defensive design for handling real-world data imperfections:

1. **Machine Sensor Faults (Handled Automatically)**:
   - **Out-of-Range Sentinels (`-999`, `999`, etc.)**: Caught by `sanitize_physical_bounds()` in `preprocessing.py` and coerced to NaN before safe interpolation.
   - **Short-Term Spikes & Blips**: Cleaned by `apply_hampel_filter()` using local Median Absolute Deviation (MAD).
   - **High-Frequency Electrical Jitter**: Removed by CEEMDAN selective reconstruction.

2. **Real-World Extreme Weather (Heatwaves, Storms, Freezes)**:
   - **Sudden Heatwaves**: Captured through temperature lags ($t-1, t-6, t-24$) and rolling standard deviations, causing LightGBM to predict accelerated drying and trigger **Critical Irrigation** alerts.
   - **Heavy Rainfall**: Captured through $6\text{h}$ and $24\text{h}$ cumulative precipitation features, causing LightGBM to adjust moisture upward and trigger **Hold Irrigation** alerts.

---

## 📦 8. Deliverables & Reproducibility Checklist

- [x] Complete modular source code in `src/` (all 9 modules).
- [x] Master execution driver `main.py` supporting both standard and abnormal datasets.
- [x] 8 step-by-step Jupyter Notebooks in `notebooks/`.
- [x] 8 publication-ready high-resolution plots saved in `outputs/plots/`.
- [x] Numeric metric reports and advisory CSVs saved in `outputs/reports/`.
- [x] Serialized LightGBM model saved in `models/lightgbm_earth_moisture_model.joblib`.
- [x] Dependencies file `requirements.txt`.
- [x] Comprehensive documentation and execution guide `README.md`.
