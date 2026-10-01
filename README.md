# PROJECT DEMONSTRATION & REPRODUCTION GUIDE

**PROJECT TITLE:**  
"Adaptive Irrigation Management Through Environmental Condition-Based Water Requirement Estimation"

**PROJECT TYPE:**  
Software-Only Signal Processing & Machine Learning Pipeline  
*(No physical hardware, ESP32, Arduino, LoRa, or mechanical pumps required)*

**COURSE / DEPARTMENT:**  
Signal Processing / Electronics and Communication Engineering

---

## TEAM MEMBERS / STUDENT DETAILS:
1. **Monica Raghini Chelle** — Roll No: `2520040093`
2. **Swarna Asritha** — Roll No: `2520040114`
3. **Dhanush Karthikeya** — Roll No: `2520040032`

* **Department:** Electronics & Communication  
* **Institution:** KLH  
* **Academic Year:** 2026 - 2027  

---

## 1. SOFTWARE REQUIREMENTS, LIBRARIES & INSTALLATION INSTRUCTIONS

### A. Environment & Prerequisites
* **Python Version:** Python 3.11 or Python 3.12 (64-bit recommended)
* **Operating System:** Windows 10/11, Linux, or macOS
* **Editor/IDE:** Visual Studio Code, PyCharm, or JupyterLab

### B. Dataset Sources & Citations
* **2024 Agricultural Weather Station Dataset:**
  * **File:** `P18_BIORO_WeatherStationData_AgriDataValue_2024.xlsx`
  * **Source:** Public Open Agricultural Weather Station Repository / Sensor Time Series
  * **Contents:** 11,548 hourly records (Air Temp, Humidity, Pressure, Dew Point, Rain, Earth Humidity/Temp, Solar).
* **2025 Agricultural Weather Station Dataset:**
  * **File:** `P18_BIORO_WeatherStationData_AgriDataValue_2025.xlsx`
  * **Source:** Public Open Agricultural Weather Station Repository / Sensor Time Series
  * **Contents:** 10,564 hourly records spanning up to October 2025.
* **Abnormal Sensor Stress-Test Dataset:**
  * **File:** `abnormal_sensor_dataset.csv`
  * **Contents:** 15,000 hourly records with synthetic & real sensor anomalies (-999 dropouts, spikes, stuck, drift).
* **Local Package Location:**  
  All datasets are pre-packaged in the `data/raw/` directory and project root. No external download is needed.

### C. Required Python Libraries
* `pandas` (>=2.0.0) — Time-series handling, datetime parsing, indexing
* `numpy` (>=1.24.0, <2.0) — Numerical arrays and mathematical operations
* `scipy` (>=1.10.0, <1.14) — Signal filtering, peak detection, correlations
* `openpyxl` (>=3.1.0) — Programmatic Excel (`.xlsx`) workbook ingestion
* `matplotlib` (>=3.7.0) — Generation of publication-grade 300 DPI plots
* `scikit-learn` (>=1.3.0) — Dataset splitting, MAE/RMSE/R2 regression metrics
* `lightgbm` (>=4.0.0) — Gradient boosted decision tree regression model
* `shap` (>=0.42.0, <0.46) — TreeSHAP game-theoretic explainability
* `EMD-signal` (>=1.4.0) — CEEMDAN empirical mode decomposition
* `joblib` (>=1.3.0) — Serialized model persistence and loading
* `jupyter` (>=1.0.0) — Interactive notebook execution support

### D. Step-by-Step Installation
1. Open PowerShell, Command Prompt, or VS Code Terminal in the project folder:
   ```bash
   cd "c:\2nd Year\Odd Sem\Signal Processing\Project_SP\Demonstration"
