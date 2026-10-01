"""Decision Support Irrigation Recommendation Module.

Implements transparent, configurable rule-based decision support:
- Evaluates forecasted 24-hour-ahead earth/soil moisture levels
- Classifies soil hydration status into actionable irrigation advisories
- Provides clear agronomic reasoning and water requirement estimation
- Pure software decision support system for farm managers and agronomists
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


# Default Configurable Agronomic Thresholds (in % Volumetric Moisture)
DEFAULT_THRESHOLDS = {
    "critical_dry": 20.0,      # Permanent wilting risk / extreme dry
    "irrigation_needed": 35.0,  # Readily available water depleted -> Irrigation recommended
    "adequate_max": 65.0,       # Field capacity range
    "saturated": 85.0          # Risk of waterlogging / anaerobic stress
}


def evaluate_irrigation_need(
    predicted_moisture: float,
    thresholds: Optional[Dict[str, float]] = None,
    precipitation_forecast: float = 0.0
) -> Dict[str, str]:
    """Evaluate a single predicted moisture value and return an advisory.
    
    Args:
        predicted_moisture: Forecasted 24-hour-ahead earth moisture (%).
        thresholds: Dictionary of configurable moisture thresholds.
        precipitation_forecast: Optional forecasted rainfall accumulation in next 24h (mm).
        
    Returns:
        Dictionary with status, recommendation, urgency, and agronomic rationale.
    """
    th = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)
        
    if predicted_moisture < th["critical_dry"]:
        status = "CRITICAL DEFICIT"
        recommendation = "Immediate Irrigation Urgently Required"
        urgency = "High"
        rationale = f"Forecasted moisture ({predicted_moisture:.1f}%) is below critical wilting threshold ({th['critical_dry']}%)."
    elif predicted_moisture < th["irrigation_needed"]:
        if precipitation_forecast >= 5.0:
            status = "DEFICIT (RAIN EXPECTED)"
            recommendation = "Hold Irrigation / Monitor Rain"
            urgency = "Low"
            rationale = f"Moisture is low ({predicted_moisture:.1f}%), but upcoming rainfall ({precipitation_forecast:.1f}mm) will replenish soil."
        else:
            status = "MOISTURE DEFICIT"
            recommendation = "Irrigation Recommended"
            urgency = "Medium"
            rationale = f"Forecasted moisture ({predicted_moisture:.1f}%) falls below optimal threshold ({th['irrigation_needed']}%)."
    elif predicted_moisture <= th["adequate_max"]:
        status = "OPTIMAL MOISTURE"
        recommendation = "No Irrigation Required"
        urgency = "None"
        rationale = f"Forecasted moisture ({predicted_moisture:.1f}%) is within optimal agronomic field capacity ({th['irrigation_needed']}% - {th['adequate_max']}%)."
    else:
        status = "SATURATION / EXCESS"
        recommendation = "Irrigation Prohibited (Risk of Waterlogging)"
        urgency = "None"
        rationale = f"Forecasted moisture ({predicted_moisture:.1f}%) indicates near-saturated conditions (> {th['adequate_max']}%)."
        
    return {
        "Predicted_Moisture(%)": round(predicted_moisture, 2),
        "Status": status,
        "Recommendation": recommendation,
        "Urgency": urgency,
        "Agronomic_Rationale": rationale
    }


def generate_irrigation_advisories(
    df_test: pd.DataFrame,
    predictions: pd.Series,
    thresholds: Optional[Dict[str, float]] = None,
    report_dir: Optional[Path] = None,
    num_recent_samples: int = 10
) -> pd.DataFrame:
    """Generate irrigation recommendations across holdout test observations.
    
    Args:
        df_test: Test partition DataFrame containing timestamps and environmental readings.
        predictions: Model predictions corresponding to df_test.
        thresholds: Configurable thresholds dictionary.
        report_dir: Directory to save recommendation CSV report.
        num_recent_samples: Number of latest forecast steps to print in console.
        
    Returns:
        DataFrame containing advisory logs.
    """
    if report_dir is None:
        report_dir = Path(__file__).resolve().parent.parent / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n========================================================")
    print(f"[*] Generating Decision-Support Irrigation Recommendations")
    print(f"========================================================")
    
    th = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)
    print(f"  • Configured Thresholds: Dry < {th['critical_dry']}% | Recommended < {th['irrigation_needed']}% | Optimal <= {th['adequate_max']}%")
    
    records = []
    timestamps = df_test["timestamp"].reset_index(drop=True)
    preds = pd.Series(predictions).reset_index(drop=True)
    
    precip_series = df_test["precipitation"].reset_index(drop=True) if "precipitation" in df_test.columns else pd.Series(0, index=range(len(df_test)))
    
    for i in range(len(preds)):
        t_current = timestamps[i]
        t_forecast = t_current + pd.Timedelta(hours=24)
        p_val = preds[i]
        precip = precip_series[i]
        
        advisory = evaluate_irrigation_need(p_val, thresholds=th, precipitation_forecast=precip)
        advisory["Current_Timestamp"] = t_current
        advisory["Forecast_Target_Time"] = t_forecast
        records.append(advisory)
        
    advisories_df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = report_dir / "irrigation_recommendations.csv"
    advisories_df.to_csv(csv_path, index=False)
    print(f"  [+] Saved full recommendations log to: {csv_path}")
    
    # Print latest window
    print(f"\n  Latest {num_recent_samples} Forecast Horizon Advisories (Decision Support View):")
    recent = advisories_df.tail(num_recent_samples)[[
        "Forecast_Target_Time", "Predicted_Moisture(%)", "Status", "Recommendation", "Urgency"
    ]]
    print(recent.to_string(index=False))
    
    # Summary of recommendations in test set
    summary = advisories_df["Recommendation"].value_counts()
    print(f"\n  Test Set Recommendation Distribution:")
    for rec, count in summary.items():
        print(f"    - {rec:45s}: {count} hours ({count/len(advisories_df)*100:.1f}%)")
        
    return advisories_df


if __name__ == "__main__":
    test_preds = [15.2, 28.4, 45.0, 72.1]
    for p in test_preds:
        print(evaluate_irrigation_need(p))
