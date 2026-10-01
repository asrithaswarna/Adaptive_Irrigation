"""Helper script to create all 8 educational and demonstration Jupyter Notebooks."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(title: str, cells_data: list, filepath: Path):
    cells = []
    for cell_type, source in cells_data:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        
    nb_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)
    print(f"[+] Created Notebook: {filepath.name}")


def generate_all_notebooks():
    # 01 Data Exploration
    nb1 = [
        ("markdown", "# 01 - Agricultural Weather Station Data Exploration\n\n**Adaptive Irrigation Management System**\nThis notebook inspects the raw 2024 and 2025 Excel workbooks, dynamically detects data sheets, and explores sensor distributions."),
        ("code", "import sys\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Add parent directory to path\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\n\ndf_2024, df_2025 = load_all_raw_data()"),
        ("code", "print('2024 Dataset Shape:', df_2024.shape)\nprint('2025 Dataset Shape:', df_2025.shape)\ndf_2024.head()")
    ]
    create_notebook("Data Exploration", nb1, NOTEBOOKS_DIR / "01_data_exploration.ipynb")

    # 02 Preprocessing
    nb2 = [
        ("markdown", "# 02 - Data Standardization & Preprocessing\n\nStandardizes schemas, aligns datetimes, verifies 1-hour sampling cadence, and manages missing values."),
        ("code", "import sys\nfrom pathlib import Path\nimport pandas as pd\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\n\nraw_24, raw_25 = load_all_raw_data()\ncombined_raw, df_clean = combine_and_clean_datasets(raw_24, raw_25)"),
        ("code", "df_clean.info()")
    ]
    create_notebook("Data Preprocessing", nb2, NOTEBOOKS_DIR / "02_data_preprocessing.ipynb")

    # 03 Hampel Filter
    nb3 = [
        ("markdown", "# 03 - Hampel Outlier Detection & Signal Cleaning\n\nApplies rolling median and MAD to identify and clean abnormal sensor spikes while preserving the raw time series."),
        ("code", "import sys\nfrom pathlib import Path\nimport pandas as pd\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)"),
        ("code", "print('Outliers flagged:', df_hampel['earth_humidity_is_outlier'].sum())\ndf_hampel[['timestamp', 'earth_humidity_raw', 'earth_humidity', 'earth_humidity_is_outlier']].head(10)")
    ]
    create_notebook("Hampel Filter", nb3, NOTEBOOKS_DIR / "03_hampel_filter.ipynb")

    # 04 CEEMDAN Analysis
    nb4 = [
        ("markdown", "# 04 - CEEMDAN Signal Decomposition\n\nDecomposes non-linear, non-stationary soil moisture signals into Intrinsic Mode Functions (IMFs) with adaptive noise injection."),
        ("code", "import sys\nfrom pathlib import Path\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\nfrom src.ceemdan import run_ceemdan_pipeline\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)\ndf_imfs, imfs = run_ceemdan_pipeline(df_hampel)"),
        ("code", "print('Extracted IMF components count:', imfs.shape[0])")
    ]
    create_notebook("CEEMDAN Analysis", nb4, NOTEBOOKS_DIR / "04_ceemdan_analysis.ipynb")

    # 05 Signal Reconstruction
    nb5 = [
        ("markdown", "# 05 - IMF Analysis & Selective Signal Reconstruction\n\nCalculates variance, energy percentage, and Pearson correlation for each IMF to filter high-frequency noise."),
        ("code", "import sys\nfrom pathlib import Path\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\nfrom src.ceemdan import run_ceemdan_pipeline\nfrom src.reconstruction import run_reconstruction_pipeline\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)\ndf_imfs, imfs = run_ceemdan_pipeline(df_hampel)\ndf_recon, recon_sig, summary_df = run_reconstruction_pipeline(df_imfs, imfs)"),
        ("code", "summary_df")
    ]
    create_notebook("Signal Reconstruction", nb5, NOTEBOOKS_DIR / "05_signal_reconstruction.ipynb")

    # 06 Feature Engineering
    nb6 = [
        ("markdown", "# 06 - 24-Hour-Ahead Target & Feature Engineering\n\nBuilds cyclical time encodings, environmental lags, rolling statistics, and signal features."),
        ("code", "import sys\nfrom pathlib import Path\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\nfrom src.ceemdan import run_ceemdan_pipeline\nfrom src.reconstruction import run_reconstruction_pipeline\nfrom src.feature_engineering import generate_features\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)\ndf_imfs, imfs = run_ceemdan_pipeline(df_hampel)\ndf_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)\ndf_ready, feature_cols, target = generate_features(df_recon)"),
        ("code", "print('Total Features:', len(feature_cols))\ndf_ready[feature_cols].head()")
    ]
    create_notebook("Feature Engineering", nb6, NOTEBOOKS_DIR / "06_feature_engineering.ipynb")

    # 07 LightGBM Prediction
    nb7 = [
        ("markdown", "# 07 - Chronological LightGBM Prediction & Evaluation\n\nPerforms leak-free chronological splitting (Train/Val/Test), trains LightGBM, and computes MAE, RMSE, and R²."),
        ("code", "import sys\nfrom pathlib import Path\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\nfrom src.ceemdan import run_ceemdan_pipeline\nfrom src.reconstruction import run_reconstruction_pipeline\nfrom src.feature_engineering import generate_features\nfrom src.model import chronological_train_val_test_split, train_lightgbm_model\nfrom src.evaluation import run_evaluation_pipeline\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)\ndf_imfs, imfs = run_ceemdan_pipeline(df_hampel)\ndf_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)\ndf_ready, feature_cols, target = generate_features(df_recon)\n\nsplits = chronological_train_val_test_split(df_ready, feature_cols, target)\nmodel = train_lightgbm_model(splits['train'][0], splits['train'][1], splits['val'][0], splits['val'][1])\nmetrics = run_evaluation_pipeline(model, splits)"),
        ("code", "print('Test Metrics:', metrics['test'])")
    ]
    create_notebook("LightGBM Prediction", nb7, NOTEBOOKS_DIR / "07_lightgbm_prediction.ipynb")

    # 08 SHAP Analysis
    nb8 = [
        ("markdown", "# 08 - SHAP Explainability & Irrigation Recommendation\n\nComputes TreeSHAP feature contributions and generates transparent decision-support irrigation recommendations."),
        ("code", "import sys\nfrom pathlib import Path\n\nsys.path.append(str(Path.cwd().parent))\nfrom src.data_loader import load_all_raw_data\nfrom src.preprocessing import combine_and_clean_datasets\nfrom src.hampel_filter import run_hampel_pipeline\nfrom src.ceemdan import run_ceemdan_pipeline\nfrom src.reconstruction import run_reconstruction_pipeline\nfrom src.feature_engineering import generate_features\nfrom src.model import chronological_train_val_test_split, train_lightgbm_model\nfrom src.explainability import run_shap_analysis\nfrom src.recommendation import generate_irrigation_advisories\n\n_, df_clean = combine_and_clean_datasets(*load_all_raw_data())\ndf_hampel = run_hampel_pipeline(df_clean)\ndf_imfs, imfs = run_ceemdan_pipeline(df_hampel)\ndf_recon, _, _ = run_reconstruction_pipeline(df_imfs, imfs)\ndf_ready, feature_cols, target = generate_features(df_recon)\n\nsplits = chronological_train_val_test_split(df_ready, feature_cols, target)\nmodel = train_lightgbm_model(splits['train'][0], splits['train'][1], splits['val'][0], splits['val'][1])\n\n# SHAP Analysis\nrun_shap_analysis(model, splits['test'][0])\n\n# Irrigation Recommendation\ntest_preds = model.predict(splits['test'][0])\nadvisories = generate_irrigation_advisories(splits['test'][2], test_preds)"),
        ("code", "advisories.head(10)")
    ]
    create_notebook("SHAP Analysis", nb8, NOTEBOOKS_DIR / "08_shap_analysis.ipynb")


if __name__ == "__main__":
    generate_all_notebooks()
