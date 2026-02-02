"""Load and process weight data from Garmin export."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Conversion factor
GRAMS_TO_LBS = 0.00220462


def load_weight_data(data_dir: Path | str = "data") -> pd.DataFrame:
    """Load weight measurements from Garmin biometrics export.

    Args:
        data_dir: Path to the data directory containing DI_CONNECT folder.

    Returns:
        DataFrame with columns: date, timestamp, weight_lbs, days_since_start
    """
    data_dir = Path(data_dir)
    biometrics_path = data_dir / "DI_CONNECT/DI-Connect-Wellness/114762117_userBioMetrics.json"

    with open(biometrics_path) as f:
        data = json.load(f)

    # Extract entries with weight data
    records = []
    for entry in data:
        if "weight" not in entry or not entry["weight"]:
            continue

        weight_info = entry["weight"]
        # Use timestampGMT if available, otherwise calendarDate
        timestamp_str = weight_info.get("timestampGMT") or entry["metaData"]["calendarDate"]
        date_str = entry["metaData"]["calendarDate"][:10]
        weight_lbs = weight_info["weight"] * GRAMS_TO_LBS  # Convert from grams to lbs

        records.append({
            "date": pd.to_datetime(date_str),
            "timestamp": pd.to_datetime(timestamp_str),
            "weight_lbs": weight_lbs,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add days since first measurement for modeling (based on timestamp)
    df["days_since_start"] = (df["timestamp"] - df["timestamp"].min()).dt.days

    return df


def prepare_stan_data(
    df: pd.DataFrame,
    include_hour_info: bool = True,
    include_weekly_info: bool = False,
    fourier_harmonics: int = 2,
    weekly_harmonics: int = 2,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data dictionary for Stan model.

    Args:
        df: DataFrame from load_weight_data()
        include_hour_info: Whether to include hour-of-day information for cyclic models
        include_weekly_info: Whether to include day-of-week information for weekly cyclic models
        fourier_harmonics: Number of Fourier harmonics for spline model (K parameter)
        weekly_harmonics: Number of Fourier harmonics for weekly spline model (L parameter)
        use_sparse: Whether to include sparse GP parameters (for optimized model)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Dictionary with Stan data fields.
    """
    # Standardize time to help with model convergence
    t = df["days_since_start"].values
    t_scaled = t / t.max()  # Scale to [0, 1]

    # Center weight for better sampling
    y = df["weight_lbs"].values
    y_mean = y.mean()
    y_sd = y.std()
    y_centered = (y - y_mean) / y_sd

    result = {
        "N": len(df),
        "t": t_scaled,
        "y": y_centered,
        # Store scaling parameters for back-transformation
        "_y_mean": y_mean,
        "_y_sd": y_sd,
        "_t_max": t.max(),
    }

    if include_hour_info and "timestamp" in df.columns:
        # Calculate continuous time in hours since first measurement
        t_hours = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600.0

        # Calculate hour of day (0-24) as float
        hour_of_day = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0

        # Calculate period for daily cycles (24 hours scaled to [0,1] range)
        total_hours = t_hours.max() - t_hours.min()
        period_daily = 24.0 / total_hours if total_hours > 0 else 0.0

        # Add hour information for cyclic models
        result.update({
            "t_hours": t_hours.values,
            "hour_of_day": hour_of_day.values,
            "period_daily": period_daily,
            "_t_hours_max": total_hours,
            "K": fourier_harmonics,
        })

    if include_weekly_info and "timestamp" in df.columns:
        # Calculate day of week (0-6.999... representing Monday-Sunday)
        # Monday = 0, Tuesday = 1, ..., Sunday = 6
        day_of_week = df["timestamp"].dt.dayofweek + df["timestamp"].dt.hour / 24.0 + df["timestamp"].dt.minute / (24.0 * 60.0)

        # Add weekly information for weekly cyclic models
        result.update({
            "day_of_week": day_of_week.values,
            "L": weekly_harmonics,
        })

    # Sparse GP configuration (for optimized model)
    if use_sparse:
        # Validate number of inducing points
        if n_inducing_points <= 0:
            raise ValueError("n_inducing_points must be positive")
        if n_inducing_points > len(t_scaled):
            n_inducing_points = len(t_scaled)
            print(f"Warning: n_inducing_points reduced to N={n_inducing_points}")

        # Select inducing points based on method
        if inducing_point_method == "uniform":
            # Uniform spacing across time range
            indices = np.linspace(0, len(t_scaled) - 1, n_inducing_points, dtype=int)
            t_inducing = t_scaled[indices]
        elif inducing_point_method == "kmeans":
            # K-means clustering on time points
            from sklearn.cluster import KMeans
            # Reshape for sklearn
            X = t_scaled.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=10)
            kmeans.fit(X)
            # Use cluster centers as inducing points
            t_inducing = kmeans.cluster_centers_.flatten()
            # Sort by time
            t_inducing = np.sort(t_inducing)
        elif inducing_point_method == "random":
            # Random selection (without replacement)
            indices = np.random.choice(len(t_scaled), n_inducing_points, replace=False)
            indices = np.sort(indices)
            t_inducing = t_scaled[indices]
        else:
            raise ValueError(f"Unknown inducing_point_method: {inducing_point_method}. "
                           f"Must be 'uniform', 'kmeans', or 'random'.")

        # Add sparse GP parameters to result
        result.update({
            "use_sparse": 1,
            "M": n_inducing_points,
            "t_inducing": t_inducing.tolist(),
        })
    else:
        # Full GP mode (default) - set inducing points to observed points for consistent prediction
        result.update({
            "use_sparse": 0,
            "M": len(t_scaled),
            "t_inducing": t_scaled.tolist(),
        })

    # Prediction grid (optional)
    if include_prediction_grid:
        # Validate prediction hour (if using single hour)
        if prediction_hour_step is None and not (0 <= prediction_hour <= 24):
            raise ValueError(f"prediction_hour must be between 0 and 24, got {prediction_hour}")

        # Validate prediction hour step (if using multiple hours)
        if prediction_hour_step is not None:
            if prediction_hour_step <= 0 or prediction_hour_step > 24:
                raise ValueError(f"prediction_hour_step must be between 0 and 24, got {prediction_hour_step}")

        # Create integer days from min(t) to max(t) with step prediction_step_days
        t_min = t.min()
        t_max = t.max()
        t_pred_days = np.arange(t_min, t_max + prediction_step_days, prediction_step_days)
        # Ensure we don't exceed max due to floating point
        t_pred_days = t_pred_days[t_pred_days <= t_max]

        # Generate prediction hours
        if prediction_hour_step is None:
            # Single hour per day
            prediction_hours = [prediction_hour]
        else:
            # Multiple hours per day (0 inclusive, 24 exclusive)
            prediction_hours = np.arange(0, 24, prediction_hour_step)
            # Ensure we don't include 24.0 exactly
            prediction_hours = prediction_hours[prediction_hours < 24]

        # Create cartesian product of days and hours
        t_pred_days_grid = np.repeat(t_pred_days, len(prediction_hours))
        t_pred = t_pred_days_grid / t_max  # scale using same scaling factor

        hour_of_day_pred = np.tile(prediction_hours, len(t_pred_days))

        # Calculate day of week for prediction points
        if include_weekly_info:
            # Create datetime objects for prediction days
            start_date = df["timestamp"].min()
            prediction_dates = [start_date + pd.Timedelta(days=float(day)) for day in t_pred_days_grid]

            # Calculate day of week for each prediction date/hour combination
            day_of_week_pred = []
            for date, hour in zip(prediction_dates, hour_of_day_pred):
                # Create datetime with specific hour
                dt = date + pd.Timedelta(hours=float(hour))
                # Calculate day of week (0-6.999...)
                dow = dt.dayofweek + dt.hour / 24.0 + dt.minute / (24.0 * 60.0)
                day_of_week_pred.append(dow)

            day_of_week_pred = np.array(day_of_week_pred)
        else:
            day_of_week_pred = []

        result.update({
            "N_pred": len(t_pred),
            "t_pred": t_pred.tolist(),
            "hour_of_day_pred": hour_of_day_pred.tolist(),
            "day_of_week_pred": day_of_week_pred.tolist() if include_weekly_info else [],
        })
    else:
        result.update({
            "N_pred": 0,
            "t_pred": [],
            "hour_of_day_pred": [],
        })

    return result
