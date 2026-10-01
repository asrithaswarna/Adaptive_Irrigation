# Adaptive Irrigation Management Through Environmental Condition-Based Water Requirement Estimation

An advanced, software-only signal processing and machine learning pipeline designed to clean complex environmental sensor telemetry, decompose non-linear trends, and accurately forecast 24-hour ahead earth moisture levels to generate agronomic watering advisories.

---

##  Course & Student Information
* **Course / Department:** Signal Processing / Electronics and Communication Engineering
* **Institution:** KLH
* **Academic Year:** 2026 - 2027

###  Team Members

| Name | Roll Number | Department |
| :--- | :--- | :--- |
| **Monica Raghini Chelle** | `2520040093` | Electronics & Communication |
| **Swarna Asritha** | `2520040114` | Electronics & Communication |
| **Dhanush Karthikeya** | `2520040032` | Electronics & Communication |

>  **Project Nature:** This is a **software-only** research and implementation pipeline. No physical microcontrollers (ESP32, Arduino), wireless modules (LoRa), or mechanical water pumps are required to run this code.

---

##  1. Software Requirements & Installation

### Environment & Prerequisites
* **Python Version:** 3.11 or 3.12 (64-bit recommended)
* **Operating System:** Windows 10/11, Linux, or macOS
* **IDE/Editor:** Visual Studio Code, PyCharm, or JupyterLab

### Required Python Libraries
```text
pandas>=2.0.0       # Time-series handling, datetime parsing, indexing
numpy>=1.24.0,<2.0  # Numerical arrays and mathematical operations
scipy>=1.10.0,<1.14 # Signal filtering, peak detection, correlations
openpyxl>=3.1.0     # Programmatic Excel (.xlsx) workbook ingestion
matplotlib>=3.7.0   # Generation of publication-grade 300 DPI plots
scikit-learn>=1.3.0 # Dataset splitting, MAE/RMSE/R2 regression metrics
lightgbm>=4.0.0     # Gradient boosted decision tree regression model
shap>=0.42.0,<0.46  # TreeSHAP game-theoretic explainability
EMD-signal>=1.4.0   # CEEMDAN empirical mode decomposition
joblib>=1.3.0       # Serialized model persistence and loading
jupyter>=1.0.0      # Interactive notebook execution support
```

### Step-by-Step Installation
1. Open your terminal (PowerShell, Command Prompt, or VS Code Terminal) and navigate to the project directory:
   ```bash
   cd "c:\2nd Year\Odd Sem\Signal Processing\Project_SP\Demonstration"
   ```
2. Install all required software dependencies cleanly via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

##  2. Dataset Ingestion & Sources
All datasets are pre-packaged within the `data/raw/` directory and project root. **No external network downloads are needed during execution:**

* **2024 Agricultural Weather Station Dataset (`P18_BIORO_WeatherStationData_AgriDataValue_2024.xlsx`)**
  * *Source:* Public Open Agricultural Weather Station Repository / Sensor Time Series.
  * *Contents:* 11,548 hourly records tracking Air Temperature, Humidity, Pressure, Dew Point, Rain, Earth Humidity/Temperature, and Solar Radiation.
* **2025 Agricultural Weather Station Dataset (`P18_BIORO_WeatherStationData_AgriDataValue_2025.xlsx`)**
  * *Contents:* 10,564 hourly records spanning up to October 2025.
* **Abnormal Sensor Stress-Test Dataset (`abnormal_sensor_dataset.csv`)**
  * *Contents:* 15,000 hourly entries packed with artificial and real-world sensor anomalies (`-999` dropouts, spikes, stuck values, and drift) to prove pipeline resilience.

---

##  3. Directory Architecture & File Manifesto

```text
├── main.py                          # Master driver script executing the 12-stage pipeline
├── requirements.txt                 # Exact library versions manifest for dependency setup
├── README.md                        # Project documentation and submission manifest
│
├── src/                             # Modular Python package source directory
│   ├── __init__.py                  # Declares src folder as a python package
│   ├── data_loader.py               # Dynamically parses Excel/CSV files and maps column synonyms
│   ├── preprocessing.py             # Unifies datetime, ensures 1-hour cadence, interpolates gaps
│   ├── hampel_filter.py             # Implements rolling median & MAD tracking to fix signal spikes
│   ├── ceemdan.py                   # Decomposes earth moisture signal into 8 IMFs + 1 Residue
│   ├── reconstruction.py            # Analyzes IMF energy and sums significant low-frequency modes
│   ├── feature_engineering.py       # Builds 24h targets, lags, rolling metrics, and cyclical encodings
│   ├── model.py                     # Executes Chronological train/val/test split and trains LightGBM
│   ├── evaluation.py                # Calculates MAE, RMSE, R2, MAPE and draws diagnostic plots
│   ├── explainability.py            # Runs TreeSHAP values for clear feature importance transparency
│   ├── recommendation.py            # Rule-based decision-support engine mapping forecasts to advice
│   └── generate_notebooks.py        # Compiles standalone Jupyter walkthroughs from pipeline scripts
│
├── notebooks/                       # Step-by-step interactive educational walkthroughs
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_hampel_filter.ipynb
│   ├── 04_ceemdan_analysis.ipynb
│   ├── 05_signal_reconstruction.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_lightgbm_prediction.ipynb
│   └── 08_shap_analysis.ipynb
│
├── data/                            # Dynamic pipeline data storage
│   ├── raw/                         # Pristine source spreadsheets and test sets
│   └── processed/                   # Intermediary steps (combined, cleaned, hampel, reconstructed)
│
├── models/                          # Serialized trained models
│   └── lightgbm_earth_moisture_model.joblib
│
└── outputs/                         # Generated project outcomes and analytics
    ├── plots/                       # Publication-grade 300 DPI visualization artifacts (01 to 08)
    └── reports/                     # Statistical evaluations and irrigation logs (JSON/CSV)
```

---

##  4. Execution Instructions

### A. Terminal Execution (Command Line)

#### Run Standard Pipeline
Processes real farm telemetry data across the 2024 and 2025 timelines.
```bash
python main.py
# or explicitly
python main.py standard
```
* **Pipeline Output:** Filters out 1,496 sensor outliers via rolling Hampel checks, filters out high-frequency noise modes using CEEMDAN mathematical decomposition, trains the LightGBM Regressor chronologically, and yields a test performance of $R^2 = 0.788$ (MAE = 9.56%). Generates 8 high-resolution analytical plots and full agronomic advice records.

#### Run Abnormal Sensor Stress-Test
Evaluates pipeline robustness against corrupted datasets.
```bash
python main.py abnormal
```
* **Pipeline Output:** Ingests the 15,000-row stress dataset. The code handles the `-999` invalid dropouts, flags severe drift patterns, finishes training successfully with an $R^2$ score of $0.769$ (MAE = 7.52%), and exits smoothly without crashing.

### B. Interactive Jupyter Notebooks
For research reviews or classroom presentations, boot your Jupyter environment in the project directory, enter the `notebooks/` directory, and run the sheets sequentially from `01_data_exploration.ipynb` through `08_shap_analysis.ipynb`.

---

##  5. Configurable Pipeline Parameters
To tune or reconfigure the signal processing and machine learning stages, modify the following variables inside their corresponding package modules under `src/`:

1. **Target Lead Horizon (`src/feature_engineering.py`):**
   * `horizon_hours = 24` — Configures the system to forecast soil moisture exactly 24 hours into the future.
2. **Hampel Outlier Filter (`src/hampel_filter.py`):**
   * `window_size = 12` — Sets a half-window size (total rolling window calculation span is $2 \times 12 + 1 = 25$ hours).
   * `n_sigma = 3.0` — Threshold multiplier for Median Absolute Deviation (MAD).
3. **CEEMDAN Signal Decomposition (`src/ceemdan.py`):**
   * `trials = 30` — Number of white noise ensemble iterations for decomposition processing.
   * `max_imfs = 7` — Upper bound limit on extracted Intrinsic Mode Functions.
4. **Chronological Machine Learning Data Split (`src/model.py`):**
   * `train_ratio = 0.70` | `val_ratio = 0.15` | `test_ratio = 0.15` — Splits timelines strictly chronologically without random shuffling to prevent data leakage.

---

##  6. Agronomic Decision-Support Rules
The forecasting engine pipes output values directly into `src/recommendation.py` to classify earth moisture projections into standard field-capacity categories:

| Projected Soil Moisture Threshold | Advisory Classification | Tactical System Recommendation |
| :--- | :--- | :--- |
| **$< 20.0\%$** | Critical Dry | Severe field moisture deficit detected. **Urgent watering required.** |
| **$20.0\% \text{ to } 35.0\%$** | Irrigation Needed | Moisture falling below sub-optimal baseline. **Schedule standard watering.** |
| **$35.0\% \text{ to } 65.0\%$** | Adequate Max | Field is sitting at optimal crop moisture capacity. **No watering needed.** |
| **$> 85.0\%$** | Saturated | Field is fully waterlogged. **Irrigation strictly prohibited.** |

---

##  7. Expected Outputs & Saved Artifacts
Upon completion, files are exported to the following default locations:

* **Console Summaries:** Prints interactive terminal status notifications, an engineering metric summary table, and a clear layman overview text block.
* **Outputs Folder (`outputs/plots/` & `outputs/reports/`):**
  * `01_hampel_filter_comparison.png` to `03_signal_reconstruction.png`: Visual breakdowns of noise reduction.
