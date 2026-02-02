"""Load VO2 max and fitness age data from Garmin MetricsMaxMetData files."""
import json
from pathlib import Path

import pandas as pd


def load_vo2max_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load VO2 max and fitness metrics from Garmin MetricsMaxMetData files.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with columns: date, vo2_max, fitness_age, fitness_age_description,
        max_met, max_met_category, analyzer_method, calibrated_data, device_id, etc.
    """
    data_dir = Path(data_dir)
    metrics_dir = data_dir / "DI_CONNECT/DI-Connect-Metrics"

    # Find all MetricsMaxMetData files
    metric_files = list(metrics_dir.glob("MetricsMaxMetData_*.json"))
    if not metric_files:
        raise FileNotFoundError(f"No MetricsMaxMetData files found in {metrics_dir}")

    records = []
    for filepath in metric_files:
        with open(filepath) as f:
            data = json.load(f)

        for entry in data:
            # Extract date
            date_str = entry.get("calendarDate")
            if not date_str:
                continue

            # VO2 max is the primary metric (mL/kg/min)
            vo2_max = entry.get("vo2MaxValue")
            if vo2_max is None:
                continue  # Skip entries without VO2 max

            # Fitness age (string like "23") - convert to numeric if possible
            fitness_age_str = entry.get("fitnessAge")
            fitness_age = None
            if fitness_age_str and fitness_age_str.isdigit():
                fitness_age = int(fitness_age_str)

            record = {
                "date": pd.to_datetime(date_str),
                "vo2_max": vo2_max,
                "fitness_age": fitness_age,
                "fitness_age_description": entry.get("fitnessAgeDescription"),
                "max_met": entry.get("maxMet"),
                "max_met_category": entry.get("maxMetCategory"),
                "analyzer_method": entry.get("analyzerMethod"),
                "calibrated_data": entry.get("calibratedData"),
                "device_id": entry.get("deviceId"),
                "update_timestamp": pd.to_datetime(entry.get("updateTimestamp")) if entry.get("updateTimestamp") else None,
            }
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Remove duplicate dates (keep most recent measurement if multiple per day)
    # Based on update_timestamp if available, otherwise keep first
    if "update_timestamp" in df.columns and df["update_timestamp"].notnull().any():
        # Sort by update timestamp within each date group
        df = df.sort_values(["date", "update_timestamp"], ascending=[True, False])
    df = df.drop_duplicates(subset="date", keep="first")

    return df


def merge_vo2max_with_weight(
    data_dir: Path | str = "data",
    weight_aggregation: str = "mean",
) -> pd.DataFrame:
    """Merge VO2 max data with weight data.

    Args:
        data_dir: Path to data directory.
        weight_aggregation: Which weight statistic to use as primary weight variable.

    Returns:
        DataFrame with columns date, weight_* (aggregations), vo2_max, etc.
        Rows are aligned by date; missing days in either dataset are dropped (inner join).
    """
    from .weight import load_weight_data
    from .align import aggregate_weight_to_daily

    # Load data
    df_weight = load_weight_data(data_dir)
    df_vo2max = load_vo2max_data(data_dir)

    # Aggregate weight to daily
    df_weight_daily = aggregate_weight_to_daily(df_weight)

    # Merge on date (inner join to keep only dates with both weight and VO2 max)
    merged = pd.merge(df_weight_daily, df_vo2max, on="date", how="inner")

    # Add derived columns
    merged["weight_variable"] = merged[f"weight_{weight_aggregation}"]
    merged["days_since_start"] = (merged["date"] - merged["date"].min()).dt.days

    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


if __name__ == "__main__":
    # Test loading
    df = load_vo2max_data()
    print(f"Loaded {len(df)} VO2 max records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nColumns:", df.columns.tolist())

    # Show missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Show summary statistics
    print("\nVO2 max summary:")
    print(df["vo2_max"].describe())

    # Check frequency
    date_diff = df["date"].diff().dt.days
    print("\nTypical measurement interval (days):")
    print(f"  Mean: {date_diff.mean():.1f}")
    print(f"  Median: {date_diff.median():.1f}")
    print(f"  Min: {date_diff.min():.0f}")
    print(f"  Max: {date_diff.max():.0f}")

    # Merge with weight to test
    try:
        merged = merge_vo2max_with_weight()
        print(f"\nMerged dataset shape: {merged.shape}")
        print(f"Dates with both weight and VO2 max: {len(merged)}")
        if len(merged) > 0:
            corr = merged[["weight_mean", "vo2_max"]].corr().iloc[0, 1]
            print(f"Correlation (weight_mean vs VO2 max): {corr:.3f}")
    except Exception as e:
        print(f"\nCould not merge with weight: {e}")