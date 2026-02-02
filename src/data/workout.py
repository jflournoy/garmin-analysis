"""Load and process workout data from Garmin fitness export."""
import json
from pathlib import Path
from typing import Literal

import pandas as pd


def load_workout_data(
    data_dir: Path | str = "data",
    activity_type: str | list[str] = "strength_training",
    include_exercise_details: bool = True,
) -> pd.DataFrame:
    """Load workout data from Garmin summarized activities export.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.
        activity_type: Which activity types to include. Can be a string or list.
                      Common types: 'strength_training', 'walking', 'running', etc.
        include_exercise_details: Whether to include detailed exercise set information.

    Returns:
        DataFrame with columns:
        - activity_id, activity_type, name, start_time_gmt, start_time_local,
        - duration, calories, avg_hr, max_hr, min_hr, steps, total_reps, total_sets,
        - total_volume (optional), active_sets, exercise_details (if include_exercise_details)
        - date (derived from start_time_local)
    """
    data_dir = Path(data_dir)
    activities_path = data_dir / "DI_CONNECT/DI-Connect-Fitness/jocoflo@pm.me_0_summarizedActivities.json"

    with open(activities_path) as f:
        data = json.load(f)

    # Get activities list
    activities = data[0].get("summarizedActivitiesExport", [])

    # Normalize activity_type to list
    if isinstance(activity_type, str):
        activity_type = [activity_type]

    # Filter by activity type
    filtered_activities = [
        act for act in activities
        if act.get("activityType") in activity_type
    ]

    records = []
    for act in filtered_activities:
        # Extract basic info
        activity_id = act.get("activityId")
        activity_type_name = act.get("activityType")
        name = act.get("name", "")

        # Timestamps (milliseconds since epoch)
        start_time_gmt_ms = act.get("startTimeGmt")
        start_time_local_ms = act.get("startTimeLocal")

        # Convert to datetime
        start_time_gmt = pd.to_datetime(start_time_gmt_ms, unit="ms") if start_time_gmt_ms else None
        start_time_local = pd.to_datetime(start_time_local_ms, unit="ms") if start_time_local_ms else None

        # Extract date from local time
        date = start_time_local.date() if start_time_local else None

        # Duration in seconds
        duration = act.get("duration")

        # Fitness metrics
        calories = act.get("calories")
        avg_hr = act.get("avgHr")
        max_hr = act.get("maxHr")
        min_hr = act.get("minHr")
        steps = act.get("steps")

        # Strength-specific metrics
        total_reps = act.get("totalReps")
        total_sets = act.get("totalSets")
        active_sets = act.get("activeSets")

        # Calculate total volume from exercise sets (if available)
        total_volume = 0
        exercise_details = []

        if include_exercise_details:
            exercise_sets = act.get("summarizedExerciseSets", [])
            for ex_set in exercise_sets:
                volume = ex_set.get("volume", 0)
                total_volume += volume

                # Store exercise details
                exercise_details.append({
                    "category": ex_set.get("category"),
                    "sub_category": ex_set.get("subCategory"),
                    "reps": ex_set.get("reps"),
                    "volume": volume,
                    "duration": ex_set.get("duration"),
                    "sets": ex_set.get("sets"),
                    "max_weight": ex_set.get("maxWeight"),
                })

        # Create record
        record = {
            "activity_id": activity_id,
            "activity_type": activity_type_name,
            "name": name,
            "start_time_gmt": start_time_gmt,
            "start_time_local": start_time_local,
            "date": pd.to_datetime(date) if date else None,
            "duration": duration,
            "calories": calories,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "min_hr": min_hr,
            "steps": steps,
            "total_reps": total_reps,
            "total_sets": total_sets,
            "active_sets": active_sets,
            "total_volume": total_volume if include_exercise_details else None,
        }

        if include_exercise_details:
            record["exercise_details"] = exercise_details

        records.append(record)

    df = pd.DataFrame(records)

    # Sort by date
    if "date" in df.columns and not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def prepare_workout_aggregates(
    df_workouts: pd.DataFrame,
    aggregation: Literal["daily", "weekly"] = "daily",
    metric: Literal["count", "volume", "reps", "sets", "calories", "duration"] = "count",
) -> pd.DataFrame:
    """Aggregate workout data to regular time intervals.

    Args:
        df_workouts: DataFrame from load_workout_data()
        aggregation: Time interval for aggregation ('daily' or 'weekly')
        metric: Which metric to aggregate ('count', 'volume', 'reps', 'sets', 'calories', 'duration')

    Returns:
        DataFrame with columns: date, workout_metric (aggregated)
    """
    if df_workouts.empty:
        return pd.DataFrame(columns=["date", "workout_metric"])

    # Ensure date column exists
    if "date" not in df_workouts.columns:
        raise ValueError("DataFrame must have 'date' column")

    # Remove rows without date
    df = df_workouts.dropna(subset=["date"]).copy()

    # Determine aggregation column based on metric
    metric_columns = {
        "count": None,  # Count of workouts
        "volume": "total_volume",
        "reps": "total_reps",
        "sets": "total_sets",
        "calories": "calories",
        "duration": "duration",
    }

    if metric not in metric_columns:
        raise ValueError(f"Unknown metric: {metric}. Must be one of {list(metric_columns.keys())}")

    # Create aggregation dataframe
    if metric == "count":
        # Count workouts per day/week
        df["workout_metric"] = 1
    else:
        col = metric_columns[metric]
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame for metric '{metric}'")
        df["workout_metric"] = df[col]

    # Group by date or week
    if aggregation == "daily":
        # Group by date
        aggregated = df.groupby("date")["workout_metric"].sum().reset_index()
    elif aggregation == "weekly":
        # Group by week starting Monday
        df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
        aggregated = df.groupby("week_start")["workout_metric"].sum().reset_index()
        aggregated = aggregated.rename(columns={"week_start": "date"})
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Must be 'daily' or 'weekly'")

    # Sort by date
    aggregated = aggregated.sort_values("date").reset_index(drop=True)

    # Rename column to be specific
    aggregated = aggregated.rename(columns={"workout_metric": f"workout_{metric}"})

    return aggregated


def prepare_workout_for_stan(
    df_workouts: pd.DataFrame,
    metric: str = "count",
    aggregation: str = "daily",
    fill_missing: bool = True,
) -> dict:
    """Prepare workout data for Stan modeling.

    Args:
        df_workouts: DataFrame from load_workout_data()
        metric: Which metric to use ('count', 'volume', 'reps', 'sets', 'calories', 'duration')
        aggregation: Time interval ('daily' or 'weekly')
        fill_missing: Whether to fill missing days with zeros

    Returns:
        Dictionary with Stan data fields.
    """
    # Aggregate workouts
    df_agg = prepare_workout_aggregates(df_workouts, aggregation=aggregation, metric=metric)

    if df_agg.empty:
        raise ValueError("No workout data after aggregation")

    # If fill_missing, create complete date range
    if fill_missing and aggregation == "daily":
        date_range = pd.date_range(df_agg["date"].min(), df_agg["date"].max(), freq="D")
        df_complete = pd.DataFrame({"date": date_range})
        df_complete = df_complete.merge(df_agg, on="date", how="left")
        df_complete[f"workout_{metric}"] = df_complete[f"workout_{metric}"].fillna(0)
        df_agg = df_complete

    # Create days since start
    df_agg["days_since_start"] = (df_agg["date"] - df_agg["date"].min()).dt.days

    # Extract metric values
    metric_col = f"workout_{metric}"
    y = df_agg[metric_col].values

    # Optional: scale the metric
    y_mean = y.mean()
    y_sd = y.std()
    y_scaled = (y - y_mean) / y_sd if y_sd > 0 else y - y_mean

    # Time points scaled to [0, 1]
    t = df_agg["days_since_start"].values
    t_max = t.max() if t.max() > 0 else 1
    t_scaled = t / t_max

    result = {
        "N": len(df_agg),
        "t": t_scaled.tolist(),
        "y": y_scaled.tolist(),
        "_y_mean": y_mean,
        "_y_sd": y_sd,
        "_t_max": t_max,
        "_dates": df_agg["date"].dt.strftime("%Y-%m-%d").tolist(),
        "_metric": metric,
        "_aggregation": aggregation,
        "_metric_values": y.tolist(),  # Original values
    }

    return result


def load_strength_training_data(
    data_dir: Path | str = "data",
    aggregation: str = "daily",
    metric: str = "count",
) -> tuple[pd.DataFrame, dict]:
    """Convenience function to load strength training data for modeling.

    Args:
        data_dir: Path to data directory.
        aggregation: Time interval for aggregation.
        metric: Which metric to use.

    Returns:
        Tuple of (DataFrame with workout data, Stan data dictionary)
    """
    # Load raw workout data
    df_workouts = load_workout_data(
        data_dir=data_dir,
        activity_type="strength_training",
        include_exercise_details=True,
    )

    # Prepare for Stan
    stan_data = prepare_workout_for_stan(
        df_workouts,
        metric=metric,
        aggregation=aggregation,
        fill_missing=True,
    )

    # Also return aggregated DataFrame for inspection
    df_agg = prepare_workout_aggregates(df_workouts, aggregation=aggregation, metric=metric)

    return df_agg, stan_data


if __name__ == "__main__":
    # Test the module
    print("=== Testing workout data loader ===")

    # Load raw workout data
    df_workouts = load_workout_data()
    print(f"Loaded {len(df_workouts)} workouts")
    print(f"Activity types: {df_workouts['activity_type'].unique().tolist()}")

    # Show strength training only
    df_strength = df_workouts[df_workouts["activity_type"] == "strength_training"]
    print(f"\nStrength training workouts: {len(df_strength)}")

    if len(df_strength) > 0:
        print("\nSample strength training workouts:")
        print(df_strength[["date", "name", "duration", "calories", "total_reps", "total_sets"]].head())

        # Test aggregation
        print("\n=== Testing aggregation ===")
        for agg_metric in ["count", "volume", "reps", "sets"]:
            df_agg = prepare_workout_aggregates(
                df_strength,
                aggregation="daily",
                metric=agg_metric
            )
            print(f"\nDaily {agg_metric}: {len(df_agg)} days")
            print(f"  Date range: {df_agg['date'].min().date()} to {df_agg['date'].max().date()}")
            print(f"  Total {agg_metric}: {df_agg[f'workout_{agg_metric}'].sum()}")

        # Test Stan data preparation
        print("\n=== Testing Stan data preparation ===")
        stan_data = prepare_workout_for_stan(
            df_strength,
            metric="count",
            aggregation="daily"
        )
        print(f"Stan data keys: {list(stan_data.keys())}")
        print(f"N: {stan_data['N']}")
        print(f"Date range in Stan data: {len(stan_data['_dates'])} days")

    else:
        print("No strength training workouts found!")