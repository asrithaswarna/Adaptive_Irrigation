
================================================================================
                    PROJECT DEMONSTRATION & REPRODUCTION GUIDE
================================================================================
PROJECT TITLE:
"Adaptive Irrigation Management Through Environmental Condition-Based
 Water Requirement Estimation"
PROJECT TYPE:
Software-Only Signal Processing & Machine Learning Pipeline
(No physical hardware, ESP32, Arduino, LoRa, or mechanical pumps required)
COURSE / DEPARTMENT:
Signal Processing / Electronics and Communication Engineering
TEAM MEMBERS / STUDENT DETAILS:
--------------------------------------------------------------------------------
1. Name: Monica Raghini Chelle        Roll No : 2520040093
2. Name: Swarna Asritha               Roll No : 2520040114
3. Name: Dhanush Karthikeya           Roll No : 2520040032
Department:  Electronics & Communication
Institution: KLH
Academic Year: 2026 - 2027
--------------------------------------------------------------------------------
================================================================================
1. SOFTWARE REQUIREMENTS, LIBRARIES & INSTALLATION INSTRUCTIONS
================================================================================
A. ENVIRONMENT & PREREQUISITES:
   - Python Version: Python 3.11 or Python 3.12 (64-bit recommended)
   - Operating System: Windows 10/11, Linux, or macOS
   - Editor/IDE: Visual Studio Code, PyCharm, or JupyterLab
B. DATASET SOURCES & CITATIONS:
   - 2024 Agricultural Weather Station Dataset:
     File: P18_BIORO_WeatherStationData_AgriDataValue_2024.xlsx
     Source: Public Open Agricultural Weather Station Repository / Sensor Time Series
     Contents: 11,548 hourly records (Air Temp, Humidity, Pressure, Dew Point, Rain, Earth Humidity/Temp, Solar).
   - 2025 Agricultural Weather Station Dataset:
     File: P18_BIORO_WeatherStationData_AgriDataValue_2025.xlsx
     Source: Public Open Agricultural Weather Station Repository / Sensor Time Series
     Contents: 10,564 hourly records spanning up to October 2025.
   - Abnormal Sensor Stress-Test Dataset:
     File: abnormal_sensor_dataset.csv
     Contents: 15,000 hourly records with synthetic & real sensor anomalies (-999 dropouts, spikes, stuck, drift).
   - Local Package Location:
     All datasets are pre-packaged in the data/raw/ directory and project root. No external download is needed.
C. REQUIRED PYTHON LIBRARIES:
   - pandas (>=2.0.0)       : Time-series handling, datetime parsing, indexing
   - numpy (>=1.24.0, <2.0) : Numerical arrays and mathematical operations
   - scipy (>=1.10.0, <1.14): Signal filtering, peak detection, correlations
   - openpyxl (>=3.1.0)     : Programmatic Excel (.xlsx) workbook ingestion
   - matplotlib (>=3.7.0)   : Generation of publication-grade 300 DPI plots
   - scikit-learn (>=1.3.0) : Dataset splitting, MAE/RMSE/R2 regression metrics
   - lightgbm (>=4.0.0)     : Gradient boosted decision tree regression model
   - shap (>=0.42.0, <0.46) : TreeSHAP game-theoretic explainability
   - EMD-signal (>=1.4.0)   : CEEMDAN empirical mode decomposition
   - joblib (>=1.3.0)       : Serialized model persistence and loading
   - jupyter (>=1.0.0)      : Interactive notebook execution support
D. STEP-BY-STEP INSTALLATION:
   1. Open PowerShell, Command Prompt, or VS Code Terminal in project folder:
      cd "c:\2nd Year\Odd Sem\Signal Processing\Project_SP\Demonstration"
   2. Install all dependencies with a single command:
      pip install -r requirements.txt
================================================================================
2. PURPOSE OF EACH FILE (INPUTS AND OUTPUTS)
================================================================================
--------------------------------------------------------------------------------
ROOT FILES:
--------------------------------------------------------------------------------
• main.py
  - Purpose : Master execution driver that runs all 12 pipeline stages end-to-end.
  - Inputs  : Raw Excel files in data/raw/ or abnormal sensor CSV.
  - Outputs : Executes pipeline, prints technical & layman summaries to console,
              and generates all artifacts in data/processed/, models/, outputs/.
• requirements.txt
  - Purpose : Lists exact Python library versions required for full reproducibility.
  - Inputs  : Used by pip package installer.
  - Outputs : Configures local Python virtual/system environment.
• README.txt / README.md
  - Purpose : Complete documentation guide and academic submission manifest.
--------------------------------------------------------------------------------
SOURCE FILES (src/ directory):
--------------------------------------------------------------------------------
• src/__init__.py
  - Purpose : Marks src as a modular Python package and defines version metadata.
• src/data_loader.py
  - Purpose : Programmatically discovers Excel sheets and CSVs without hardcoded
              sheet names; standardizes column synonyms into snake_case.
  - Inputs  : data/raw/*.xlsx or data/raw/*abnormal*.csv.
  - Outputs : Standardized pandas DataFrame with canonical columns.
• src/preprocessing.py
  - Purpose : Unifies date/time into single timestamp, checks 1-hour cadence,
              sanitizes physical bounds (removes -999 sentinels), deduplicates,
              and performs non-destructive interpolation.
  - Inputs  : Raw DataFrames from data_loader.py.
  - Outputs : data/processed/combined_data.csv and data/processed/cleaned_data.csv.
• src/hampel_filter.py
  - Purpose : Implements rolling median and Median Absolute Deviation (MAD) to
              detect and repair sensor glitches/spikes while preserving raw data.
  - Inputs  : data/processed/cleaned_data.csv.
  - Outputs : data/processed/hampel_data.csv,
              outputs/plots/01_hampel_filter_comparison.png.
• src/ceemdan.py
  - Purpose : Complete Ensemble Empirical Mode Decomposition with Adaptive Noise.
              Decomposes non-linear moisture signal into 8 IMFs + 1 Residue.
  - Inputs  : Hampel-cleaned earth_humidity signal.
  - Outputs : 2D IMF decomposition matrix,
              outputs/plots/02_ceemdan_imfs.png.
• src/reconstruction.py
  - Purpose : Quantitative analysis of each IMF (variance, energy %, correlation);
              filters high-frequency noise modes (IMFs 1-6) and sums significant
              modes (IMFs 7, 8, Residue) into a clean reconstructed signal.
  - Inputs  : IMFs matrix and cleaned signal.
  - Outputs : data/processed/reconstructed_data.csv,
              outputs/reports/imf_characteristics.csv,
              outputs/plots/03_signal_reconstruction.png.
• src/feature_engineering.py
  - Purpose : Shifts target by -24h (t + 24h ahead), constructs lag features
              (t-1 to t-24), rolling statistics (mean, std, min, max over 6h, 12h,
              24h), rainfall accumulation, and cyclical time encodings (55 total).
  - Inputs  : data/processed/reconstructed_data.csv.
  - Outputs : Feature matrix X (14,952 rows, 55 features) and target vector y.
• src/model.py
  - Purpose : Strictly chronological train/val/test splitting (70% / 15% / 15%)
              without random shuffling (leak-free); trains LightGBMRegressor with
              validation early stopping.
  - Inputs  : Engineered feature matrix and target.
  - Outputs : models/lightgbm_earth_moisture_model.joblib.
• src/evaluation.py
  - Purpose : Evaluates regression metrics (MAE, RMSE, R2, MAPE) across Train,
              Val, and Test sets; generates error distribution plots.
  - Inputs  : Trained model and split partitions.
  - Outputs : outputs/reports/evaluation_summary.json,
              outputs/plots/04_actual_vs_predicted.png,
              outputs/plots/05_residuals_analysis.png.
• src/explainability.py
  - Purpose : TreeSHAP interpretability computing feature contribution impact.
  - Inputs  : Trained LightGBM model and holdout test feature instances.
  - Outputs : outputs/plots/06_shap_importance.png,
              outputs/plots/07_shap_summary.png,
              outputs/plots/08_shap_waterfall.png.
• src/recommendation.py
  - Purpose : Rule-based decision-support system classifying 24h forecasted
              moisture into agronomic advisories (Critical, Moderate, Optimal, Saturation).
  - Inputs  : Model predictions and test timestamps.
  - Outputs : outputs/reports/irrigation_recommendations.csv.
• src/generate_notebooks.py
  - Purpose : Automatically constructs the 8 clean Jupyter Notebooks in notebooks/.
--------------------------------------------------------------------------------
NOTEBOOKS (notebooks/ directory):
--------------------------------------------------------------------------------
• 01_data_exploration.ipynb      : Workbook inspection and sensor distribution exploration
• 02_data_preprocessing.ipynb    : Datetime unification, cadence check & missing value handling
• 03_hampel_filter.ipynb         : Rolling median and MAD outlier detection walkthrough
• 04_ceemdan_analysis.ipynb      : CEEMDAN mode decomposition and multi-panel plotting
• 05_signal_reconstruction.ipynb : IMF variance/energy characterization & selective filtering
• 06_feature_engineering.ipynb   : 24h target construction, lags & cyclical time encodings
• 07_lightgbm_prediction.ipynb   : Chronological model training & evaluation metrics
• 08_shap_analysis.ipynb         : TreeSHAP explainability & decision-support advisories
================================================================================
3. MAIN FILE TO RUN & CORRECT EXECUTION SEQUENCE
================================================================================
A. PRIMARY EXECUTION (Command Line / Terminal):
   Option 1: Standard Mode (2024 & 2025 Farm Datasets)
   ---------------------------------------------------
   Command:
      python main.py
   (or explicitly: python main.py standard)
   What happens:
   - Ingests 2024 & 2025 Excel workbooks (13,567 clean hourly hours).
   - Cleans 1,496 sensor outliers using Hampel filter.
   - Decomposes signals using CEEMDAN and reconstructs denoised signal.
   - Trains LightGBM model on past data (2024 to early 2025).
   - Evaluates on future test set (R2 = 0.788, MAE = 9.56%).
   - Generates 8 high-res plots, SHAP explainability, and irrigation advice.
   Option 2: Abnormal Stress-Test Mode (Abnormal Sensor Dataset)
   ------------------------------------------------------------
   Command:
      python main.py abnormal
   What happens:
   - Ingests data/raw/abnormal_sensor_dataset.csv (15,000 rows).
   - Filters out -999 sentinels, dropouts, and extreme physical anomalies.
   - Repairs sensor spikes using Hampel filter.
   - Trains and evaluates LightGBM (R2 = 0.769, MAE = 7.52%).
   - Generates full diagnostic plots and recommendations without crashing.
B. INTERACTIVE EXECUTION (Jupyter Notebooks):
   Open VS Code, navigate to notebooks/, and run cells in sequence:
   01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08
================================================================================
4. SETTINGS & CONFIGURABLE PARAMETERS
================================================================================
All paths are dynamically resolved using Python's pathlib. No hardcoded absolute
system paths exist in the code.
Configurable parameters inside source files:
1. Soil Moisture Target Lead Horizon (src/feature_engineering.py):
   - horizon_hours = 24  (Forecasts 24 hours into the future)
2. Hampel Filter Tuning (src/hampel_filter.py):
   - window_size = 12    (Half window -> effective window is 2*12 + 1 = 25 hours)
   - n_sigma = 3.0       (MAD multiplier threshold = 3.0 * 1.4826)
3. CEEMDAN Decomposition (src/ceemdan.py):
   - trials = 30         (Ensemble noise iterations)
   - max_imfs = 7        (Maximum number of intrinsic mode functions)
4. Train / Val / Test Split Ratios (src/model.py):
   - train_ratio = 0.70  (First 70% of chronological timeline)
   - val_ratio = 0.15    (Next 15% for validation & early stopping)
   - test_ratio = 0.15   (Final 15% for holdout test evaluation)
5. Agronomic Irrigation Thresholds (src/recommendation.py):
   - critical_dry = 20.0%       (Severe moisture deficit -> Urgent watering)
   - irrigation_needed = 35.0%  (Sub-optimal moisture -> Scheduled watering)
   - adequate_max = 65.0%       (Optimal field capacity -> No watering)
   - saturated = 85.0%          (Excess moisture -> Irrigation prohibited)
================================================================================
5. EXPECTED OUTPUTS & WHERE RESULTS ARE SAVED
================================================================================
A. TERMINAL CONSOLE OUTPUT:
   Prints phase-by-phase execution progress followed by:
   - [TECHNICAL METRICS SUMMARY] : Runtime, MAE, RMSE, R2 Score, MAPE
   - [LAYMAN / SIMPLE TERMS SUMMARY] : Plain English recap of all 5 stages
B. SAVED PLOTS (outputs/plots/ - 300 DPI High Resolution):
   • 01_hampel_filter_comparison.png : Raw vs Cleaned signal & flagged outliers
   • 02_ceemdan_imfs.png             : Multi-panel plot of all 8 IMFs + Residue
   • 03_signal_reconstruction.png    : Selected IMFs vs Reconstructed signal vs Noise
   • 04_actual_vs_predicted.png      : Test set actual vs 24h predicted curves
   • 05_residuals_analysis.png       : Residual error distribution & scatter diagnostics
   • 06_shap_importance.png          : Mean |SHAP| global feature importance bar chart
   • 07_shap_summary.png             : SHAP beeswarm plot showing impact directions
   • 08_shap_waterfall.png           : Local single-sample waterfall explanation
C. SAVED REPORTS (outputs/reports/):
   • evaluation_summary.json         : JSON file with exact Train/Val/Test metrics
   • imf_characteristics.csv         : IMF variance, energy %, and correlations
   • irrigation_recommendations.csv  : Hourly 24h-ahead farm advisories log
D. SAVED PROCESSED DATA (data/processed/):
   • combined_data.csv, cleaned_data.csv, hampel_data.csv, reconstructed_data.csv
E. SERIALIZED MODEL (models/):
   • lightgbm_earth_moisture_model.joblib : Fitted LightGBM model artifact
================================================================================
6. KNOWN LIMITATIONS & EXECUTION NOTES
================================================================================
1. CEEMDAN Computation Time:
   - Decomposing 13,500+ hourly samples with 30 ensemble trials is CPU-intensive
     and typically takes 60 to 90 seconds. This is normal mathematical behavior.
2. Pure Software Decision-Support System:
   - This project produces predictive recommendations. It does not send direct
     electrical control signals to physical irrigation valves/pumps.
3. Character Encoding on Windows Terminals:
   - All console outputs use clean ASCII markers ([+], [-], *) to ensure 100%
     compatibility with Windows cp1252 / UTF-8 terminals without encoding errors.
4. Missing Sensor Channel Fallback:
   - If a standalone dataset lacks secondary channels (e.g. dew_point, air_pressure),
     the pipeline automatically imputes or adapts without failing.
================================================================================
                          END OF README.txt
================================================================================
