"""Load daily aggregated metrics from Garmin UDS (User Daily Summary) files."""
import json
from pathlib import Path

import pandas as pd
import numpy as np


def load_daily_metrics(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load daily aggregated metrics from Garmin UDS files.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with columns: date, resting_heart_rate, total_steps,
        active_kilocalories, moderate_intensity_minutes, vigorous_intensity_minutes,
        highly_active_seconds, active_seconds, min_heart_rate, max_heart_rate,
        avg_stress_level, stress_duration, rest_duration, activity_duration,
        floors_ascended_meters, floors_descended_meters, etc.
    """
    data_dir = Path(data_dir)
    uds_dir = data_dir / "DI_CONNECT/DI-Connect-Aggregator"

    # Find all UDS files
    uds_files = list(uds_dir.glob("UDSFile_*.json"))
    if not uds_files:
        raise FileNotFoundError(f"No UDS files found in {uds_dir}")

    records = []
    for filepath in uds_files:
        with open(filepath) as f:
            data = json.load(f)

        for entry in data:
            # Extract basic date
            date_str = entry.get("calendarDate")
            if not date_str:
                continue

            # Extract stress metrics
            stress = entry.get("allDayStress", {})
            # aggregate list may contain entries of type TOTAL, ACTIVITY, REST, etc.
            avg_stress = None
            stress_duration = None
            rest_duration = None
            activity_duration = None
            if isinstance(stress, dict) and "aggregatorList" in stress:
                for agg in stress["aggregatorList"]:
                    agg_type = agg.get("type")
                    if agg_type == "TOTAL":
                        avg_stress = agg.get("averageStressLevel")
                        stress_duration = agg.get("stressDuration")
                    elif agg_type == "REST":
                        rest_duration = agg.get("restDuration")
                    elif agg_type == "ACTIVITY":
                        activity_duration = agg.get("activityDuration")

            record = {
                "date": pd.to_datetime(date_str),
                "resting_heart_rate": entry.get("restingHeartRate"),
                "current_day_resting_heart_rate": entry.get("currentDayRestingHeartRate"),
                "total_steps": entry.get("totalSteps"),
                "active_kilocalories": entry.get("activeKilocalories"),
                "bmr_kilocalories": entry.get("bmrKilocalories"),
                "total_kilocalories": entry.get("totalKilocalories"),
                "moderate_intensity_minutes": entry.get("moderateIntensityMinutes"),
                "vigorous_intensity_minutes": entry.get("vigorousIntensityMinutes"),
                "highly_active_seconds": entry.get("highlyActiveSeconds"),
                "active_seconds": entry.get("activeSeconds"),
                "min_heart_rate": entry.get("minHeartRate"),
                "max_heart_rate": entry.get("maxHeartRate"),
                "avg_stress_level": avg_stress,
                "stress_duration": stress_duration,
                "rest_duration": rest_duration,
                "activity_duration": activity_duration,
                "floors_ascended_meters": entry.get("floorsAscendedInMeters"),
                "floors_descended_meters": entry.get("floorsDescendedInMeters"),
                "daily_step_goal": entry.get("dailyStepGoal"),
                "net_calorie_goal": entry.get("netCalorieGoal"),
                "user_intensity_minutes_goal": entry.get("userIntensityMinutesGoal"),
                "includes_wellness_data": entry.get("includesWellnessData"),
                "includes_activity_data": entry.get("includesActivityData"),
            }
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates (keep first occurrence)
    df = df.drop_duplicates(subset="date", keep="first")

    return df


def prepare_daily_metrics_for_stan(
    df: pd.DataFrame,
    target_variable: str = "resting_heart_rate",
    include_date_index: bool = True,
) -> dict:
    """Prepare daily metrics for Stan modeling.

    Args:
        df: DataFrame from load_daily_metrics()
        target_variable: Which variable to extract as primary outcome.
        include_date_index: Whether to include date index for alignment.

    Returns:
        Dictionary with Stan data fields.
    """
    # Ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)

    # Create days since start
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    # Extract target variable
    if target_variable not in df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in DataFrame")

    y = df[target_variable].values
    # Remove missing values? For now, require complete data
    if np.any(pd.isna(y)):
        raise ValueError(f"Target variable '{target_variable}' contains missing values")

    # Scale target variable
    y_mean = y.mean()
    y_sd = y.std()
    y_centered = (y - y_mean) / y_sd

    # Time points scaled to [0, 1]
    t = df["days_since_start"].values
    t_scaled = t / t.max() if t.max() > 0 else t

    result = {
        "N": len(df),
        "t": t_scaled,
        "y": y_centered,
        "_y_mean": y_mean,
        "_y_sd": y_sd,
        "_t_max": t.max(),
        "_dates": df["date"].dt.strftime("%Y-%m-%d").tolist() if include_date_index else [],
    }

    # Add additional covariates if desired
    # Could add other variables as covariates matrix

    return result


if __name__ == "__main__":
    # Test loading
    df = load_daily_metrics()
    print(f"Loaded {len(df)} daily records")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nDate range:", df["date"].min(), "to", df["date"].max())

    # Check missingness
    print("\nMissing values per column:")
    print(df.isnull().sum())