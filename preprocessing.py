"""
preprocessing.py — Weather Forecasting Using Markov
=====================================
Data loading, validation, cleaning, and weather-state feature engineering.

Design principle: Pure functions wherever possible. Each function has a single
responsibility and can be tested independently. No side effects on the input df.

Author: Weather Forecasting Using Markov
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

from config import (
    DATA_PATH,
    COLUMNS,
    HUMIDITY_THRESHOLDS,
    WEATHER_STATES,
)


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw weather CSV and normalize column names.

    Strips leading/trailing whitespace from column headers — a common issue
    with sensor-exported CSVs. Raises a clear error if the file is missing.

    Args:
        path: File path to the CSV file.

    Returns:
        Raw DataFrame with cleaned column names.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        ValueError: If required columns are missing.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please place weather.csv in the data/ directory."
        )

    # Normalize column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Validate required columns exist
    required = [
        COLUMNS["day"], COLUMNS["hour"],
        COLUMNS["temperature"], COLUMNS["humidity"],
        COLUMNS["wind_speed"], COLUMNS["wind_direction"],
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


# ─── Data Validation ──────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run data quality checks and return a validation report.

    This is the kind of observability that separates production ML pipelines
    from notebook experiments. Returns a dict so the UI can surface warnings.

    Args:
        df: Raw DataFrame.

    Returns:
        Dictionary with keys: 'is_valid', 'warnings', 'stats'
    """
    report = {"is_valid": True, "warnings": [], "stats": {}}

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        for col, cnt in missing[missing > 0].items():
            report["warnings"].append(f"Column '{col}' has {cnt} missing values.")

    # Check value ranges
    humidity_col = COLUMNS["humidity"]
    if humidity_col in df.columns:
        hum = df[humidity_col]
        if hum.min() < 0 or hum.max() > 100:
            report["warnings"].append(
                f"Humidity values outside [0, 100]: min={hum.min()}, max={hum.max()}"
            )
        report["stats"]["humidity_range"] = (float(hum.min()), float(hum.max()))

    temp_col = COLUMNS["temperature"]
    if temp_col in df.columns:
        temp = df[temp_col]
        report["stats"]["temp_range"] = (float(temp.min()), float(temp.max()))

    wind_col = COLUMNS["wind_speed"]
    if wind_col in df.columns:
        wind = df[wind_col]
        if wind.min() < 0:
            report["warnings"].append(f"Negative wind speed values detected.")
        report["stats"]["wind_range"] = (float(wind.min()), float(wind.max()))

    # Minimum row check
    if len(df) < 10:
        report["is_valid"] = False
        report["warnings"].append(
            f"Dataset too small ({len(df)} rows). Need at least 10 observations."
        )

    return report


# ─── Weather State Classification ─────────────────────────────────────────────

def classify_weather_state(humidity: float) -> str:
    """
    Map a single humidity reading to a weather state label.

    Thresholds are defined in config.py to keep classification rules
    in one place. This function is vectorizable via df.apply().

    Args:
        humidity: Relative humidity value (0–100).

    Returns:
        One of: 'Rainy', 'Cloudy', 'Sunny'
    """
    if humidity >= HUMIDITY_THRESHOLDS["rainy_min"]:
        return "Rainy"
    elif humidity >= HUMIDITY_THRESHOLDS["cloudy_min"]:
        return "Cloudy"
    else:
        return "Sunny"


def assign_weather_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'State' column to the DataFrame using humidity-based classification.

    Returns a copy — never mutates input. This is a critical habit: pure
    transformations make debugging and testing far easier.

    Args:
        df: DataFrame with a humidity column.

    Returns:
        New DataFrame with added 'State' column.
    """
    df = df.copy()
    humidity_col = COLUMNS["humidity"]
    df[COLUMNS["state"]] = df[humidity_col].apply(classify_weather_state)
    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features useful for visualization and analysis.

    Features added:
    - Rolling 3-hour mean for temperature, humidity, wind speed
    - Humidity category (for coloring)
    - Hour bucket (Morning / Afternoon / Evening / Night)

    Args:
        df: DataFrame with assigned weather states.

    Returns:
        Enriched DataFrame.
    """
    df = df.copy()

    temp_col = COLUMNS["temperature"]
    hum_col = COLUMNS["humidity"]
    wind_col = COLUMNS["wind_speed"]

    # Rolling means (window=3 hours, min_periods=1 to handle edges)
    df["Temp_Rolling3h"] = df[temp_col].rolling(window=3, min_periods=1).mean().round(2)
    df["Humidity_Rolling3h"] = df[hum_col].rolling(window=3, min_periods=1).mean().round(2)
    df["Wind_Rolling3h"] = df[wind_col].rolling(window=3, min_periods=1).mean().round(2)

    # Hour of day → time bucket
    hour_col = COLUMNS["hour"]
    df["TimeBucket"] = pd.cut(
        df[hour_col],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (0–5)", "Morning (6–11)", "Afternoon (12–17)", "Evening (18–23)"],
    )

    # Humidity delta (how fast humidity is changing)
    df["Humidity_Delta"] = df[hum_col].diff().fillna(0).round(2)

    return df


# ─── Daily Aggregation ────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly sensor data to daily summaries.

    Used by the dashboard to show day-level trends and stats.

    Args:
        df: Hourly DataFrame with weather states.

    Returns:
        Daily DataFrame with mean/min/max and dominant state per day.
    """
    day_col = COLUMNS["day"]
    temp_col = COLUMNS["temperature"]
    hum_col = COLUMNS["humidity"]
    wind_col = COLUMNS["wind_speed"]
    state_col = COLUMNS["state"]

    daily = df.groupby(day_col).agg(
        Temp_Mean=(temp_col, "mean"),
        Temp_Min=(temp_col, "min"),
        Temp_Max=(temp_col, "max"),
        Humidity_Mean=(hum_col, "mean"),
        Wind_Mean=(wind_col, "mean"),
        Wind_Max=(wind_col, "max"),
        Hours_Sunny=(state_col, lambda x: (x == "Sunny").sum()),
        Hours_Cloudy=(state_col, lambda x: (x == "Cloudy").sum()),
        Hours_Rainy=(state_col, lambda x: (x == "Rainy").sum()),
    ).reset_index()

    # Dominant state per day (modal state)
    dominant = (
        df.groupby(day_col)[state_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={state_col: "Dominant_State"})
    )
    daily = daily.merge(dominant, on=day_col)

    # Round floats
    float_cols = ["Temp_Mean", "Temp_Min", "Temp_Max", "Humidity_Mean", "Wind_Mean", "Wind_Max"]
    daily[float_cols] = daily[float_cols].round(2)

    return daily


# ─── Summary Statistics ───────────────────────────────────────────────────────

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return a structured summary of the dataset for the dashboard.

    Args:
        df: Fully processed DataFrame (with states).

    Returns:
        Dictionary of summary statistics.
    """
    temp_col = COLUMNS["temperature"]
    hum_col = COLUMNS["humidity"]
    wind_col = COLUMNS["wind_speed"]
    state_col = COLUMNS["state"]
    day_col = COLUMNS["day"]

    state_counts = df[state_col].value_counts().to_dict() if state_col in df.columns else {}

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "n_days": df[day_col].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "temp_mean": round(df[temp_col].mean(), 2),
        "temp_min": round(df[temp_col].min(), 2),
        "temp_max": round(df[temp_col].max(), 2),
        "humidity_mean": round(df[hum_col].mean(), 2),
        "humidity_min": round(df[hum_col].min(), 2),
        "humidity_max": round(df[hum_col].max(), 2),
        "wind_mean": round(df[wind_col].mean(), 2),
        "wind_max": round(df[wind_col].max(), 2),
        "state_counts": state_counts,
        "dominant_state": max(state_counts, key=state_counts.get) if state_counts else "N/A",
    }


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete preprocessing pipeline.

    This is the single entry point called by app.py and cached by Streamlit.
    Returns all artifacts needed by the rest of the application.

    Args:
        path: Path to the raw CSV.

    Returns:
        Tuple of (processed_hourly_df, daily_df, summary_dict)
    """
    raw_df = load_data(path)
    validation = validate_data(raw_df)

    # Assign states, engineer features
    df = assign_weather_states(raw_df)
    df = engineer_features(df)

    # Aggregate daily
    daily_df = aggregate_daily(df)

    # Summary stats
    summary = get_dataset_summary(df)
    summary["validation"] = validation

    return df, daily_df, summary
