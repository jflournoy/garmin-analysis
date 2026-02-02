"""Align weight data with daily metrics for correlation analysis."""
from pathlib import Path

import pandas as pd
import numpy as np

from .weight import load_weight_data
from .activity import load_daily_metrics


def aggregate_weight_to_daily(df_weight: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weight measurements to daily statistics.

    Args:
        df_weight: DataFrame from load_weight_data() with columns timestamp, weight_lbs.

    Returns:
        DataFrame with columns date, weight_mean, weight_std, weight_count,
        weight_min, weight_max, weight_median, weight_first, weight_last.
    """
    # Extract date (without time)
    df = df_weight.copy()
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # Group by date
    grouped = df.groupby("date")["weight_lbs"].agg([
        ("weight_mean", "mean"),
        ("weight_std", "std"),
        ("weight_count", "count"),
        ("weight_min", "min"),
        ("weight_max", "max"),
        ("weight_median", "median"),
        ("weight_first", lambda x: x.iloc[0]),
        ("weight_last", lambda x: x.iloc[-1]),
    ]).reset_index()

    # Fill NaN std with 0 for single measurements
    grouped["weight_std"] = grouped["weight_std"].fillna(0.0)

    return grouped


def merge_weight_with_daily_metrics(
    data_dir: Path | str = "data",
    weight_aggregation: str = "mean",
) -> pd.DataFrame:
    """Merge weight data (aggregated daily) with daily metrics.

    Args:
        data_dir: Path to data directory.
        weight_aggregation: Which weight statistic to use as primary weight variable.
            Options: 'mean', 'median', 'first', 'last', 'min', 'max'.

    Returns:
        DataFrame with columns date, weight_* (all aggregations), plus all daily metrics.
        Rows are aligned by date; missing days in either dataset are dropped (inner join).
    """
    # Load data
    df_weight = load_weight_data(data_dir)
    df_daily = load_daily_metrics(data_dir)

    # Aggregate weight to daily
    df_weight_daily = aggregate_weight_to_daily(df_weight)

    # Merge on date (inner join to keep only dates with both weight and daily metrics)
    merged = pd.merge(df_weight_daily, df_daily, on="date", how="inner")

    # Add derived columns
    merged["weight_variable"] = merged[f"weight_{weight_aggregation}"]
    merged["weight_day_of_week"] = merged["date"].dt.dayofweek  # Monday=0
    merged["weight_day_of_year"] = merged["date"].dt.dayofyear

    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def prepare_bivariate_stan_data(
    df: pd.DataFrame,
    weight_var: str = "weight_mean",
    other_var: str = "resting_heart_rate",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data for bivariate GP model (weight + another variable).

    Args:
        df: Merged DataFrame from merge_weight_with_daily_metrics().
        weight_var: Weight variable column name.
        other_var: Other variable column name (must be numeric).
        use_sparse: Whether to include sparse GP parameters.
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True).
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random").
        include_prediction_grid: Whether to include prediction grid for unobserved days.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data fields for bivariate model.
    """
    # Ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)

    # Check for missing values
    if df[weight_var].isnull().any() or df[other_var].isnull().any():
        raise ValueError(f"Missing values in {weight_var} or {other_var}")

    # Create days since start
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    # Extract variables
    y_weight = df[weight_var].values
    y_other = df[other_var].values

    # Scale variables
    y_weight_mean = y_weight.mean()
    y_weight_sd = y_weight.std()
    y_weight_scaled = (y_weight - y_weight_mean) / y_weight_sd

    y_other_mean = y_other.mean()
    y_other_sd = y_other.std()
    y_other_scaled = (y_other - y_other_mean) / y_other_sd

    # Time points scaled to [0, 1]
    t = df["days_since_start"].values
    t_scaled = t / t.max() if t.max() > 0 else t

    # Prepare result
    result = {
        "N": len(df),
        "t": t_scaled,
        "y_weight": y_weight_scaled,
        "y_other": y_other_scaled,
        "_y_weight_mean": y_weight_mean,
        "_y_weight_sd": y_weight_sd,
        "_y_other_mean": y_other_mean,
        "_y_other_sd": y_other_sd,
        "_t_max": t.max(),
        "_dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
    }

    # Sparse GP configuration
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
        # Full GP mode - set inducing points to observed points
        result.update({
            "use_sparse": 0,
            "M": len(t_scaled),
            "t_inducing": t_scaled.tolist(),
        })

    # Prediction grid (optional)
    if include_prediction_grid:
        # Create integer days from min(t) to max(t) with step prediction_step_days
        t_min = t.min()
        t_max = t.max()
        t_pred_days = np.arange(t_min, t_max + prediction_step_days, prediction_step_days)
        # Ensure we don't exceed max due to floating point
        t_pred_days = t_pred_days[t_pred_days <= t_max]
        t_pred = t_pred_days / t_max  # scale using same scaling factor

        result.update({
            "N_pred": len(t_pred),
            "t_pred": t_pred.tolist(),
        })
    else:
        result.update({
            "N_pred": 0,
            "t_pred": [],
        })

    return result


def prepare_bivariate_stan_data_mismatched(
    df_weight: pd.DataFrame,
    df_other: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    other_time_col: str = "timestamp",
    other_value_col: str = "value",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data for bivariate GP model with mismatched observation times.

    Args:
        df_weight: DataFrame with weight observations (must have timestamp and value columns).
        df_other: DataFrame with other variable observations (must have timestamp and value columns).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        other_time_col: Name of timestamp column in df_other.
        other_value_col: Name of value column in df_other.
        use_sparse: Whether to include sparse GP parameters.
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True).
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random").
        include_prediction_grid: Whether to include prediction grid for unobserved days.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data fields for mismatched bivariate model.
    """
    # Ensure sorted by time
    df_weight = df_weight.sort_values(weight_time_col).reset_index(drop=True)
    df_other = df_other.sort_values(other_time_col).reset_index(drop=True)

    # Check for missing values
    if df_weight[weight_value_col].isnull().any():
        raise ValueError(f"Missing values in weight column {weight_value_col}")
    if df_other[other_value_col].isnull().any():
        raise ValueError(f"Missing values in other column {other_value_col}")

    # Convert timestamps to datetime if not already
    t_weight = pd.to_datetime(df_weight[weight_time_col])
    t_other = pd.to_datetime(df_other[other_time_col])

    # Find global time range
    global_t_min = min(t_weight.min(), t_other.min())
    global_t_max = max(t_weight.max(), t_other.max())

    # Compute days since global min
    df_weight["days_since_start"] = (t_weight - global_t_min).dt.days
    df_other["days_since_start"] = (t_other - global_t_min).dt.days

    # Extract variables
    y_weight = df_weight[weight_value_col].values
    y_other = df_other[other_value_col].values

    # Scale variables (zero mean, unit variance)
    y_weight_mean = y_weight.mean()
    y_weight_sd = y_weight.std()
    y_weight_scaled = (y_weight - y_weight_mean) / y_weight_sd

    y_other_mean = y_other.mean()
    y_other_sd = y_other.std()
    y_other_scaled = (y_other - y_other_mean) / y_other_sd

    # Time points scaled to [0, 1] based on global range
    t_weight_days = df_weight["days_since_start"].values
    t_other_days = df_other["days_since_start"].values
    t_max = (global_t_max - global_t_min).days
    if t_max == 0:
        t_max = 1  # avoid division by zero

    t_weight_scaled = t_weight_days / t_max
    t_other_scaled = t_other_days / t_max

    # Prepare result
    result = {
        "N_weight": len(df_weight),
        "t_weight": t_weight_scaled.tolist(),
        "y_weight": y_weight_scaled.tolist(),
        "N_other": len(df_other),
        "t_other": t_other_scaled.tolist(),
        "y_other": y_other_scaled.tolist(),
        "_y_weight_mean": y_weight_mean,
        "_y_weight_sd": y_weight_sd,
        "_y_other_mean": y_other_mean,
        "_y_other_sd": y_other_sd,
        "_t_max": t_max,
        "_global_t_min": global_t_min.strftime("%Y-%m-%d"),
        "_global_t_max": global_t_max.strftime("%Y-%m-%d"),
    }

    # Sparse GP configuration
    # We need inducing points across the global time range [0, 1]
    if use_sparse:
        # Validate number of inducing points
        if n_inducing_points <= 0:
            raise ValueError("n_inducing_points must be positive")
        # Use union of scaled time points as candidate locations
        t_all_scaled = np.unique(np.concatenate([t_weight_scaled, t_other_scaled]))
        if n_inducing_points > len(t_all_scaled):
            n_inducing_points = len(t_all_scaled)
            print(f"Warning: n_inducing_points reduced to N={n_inducing_points}")

        # Select inducing points based on method
        if inducing_point_method == "uniform":
            # Uniform spacing across [0, 1]
            t_inducing = np.linspace(0, 1, n_inducing_points)
        elif inducing_point_method == "kmeans":
            # K-means clustering on all scaled time points
            from sklearn.cluster import KMeans
            X = t_all_scaled.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=10)
            kmeans.fit(X)
            t_inducing = kmeans.cluster_centers_.flatten()
            t_inducing = np.sort(t_inducing)
        elif inducing_point_method == "random":
            # Random selection from union of time points
            t_inducing = np.random.choice(t_all_scaled, n_inducing_points, replace=False)
            t_inducing = np.sort(t_inducing)
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
        # Full GP mode - set inducing points to union of observed points
        t_all_scaled = np.unique(np.concatenate([t_weight_scaled, t_other_scaled]))
        result.update({
            "use_sparse": 0,
            "M": len(t_all_scaled),
            "t_inducing": t_all_scaled.tolist(),
        })

    # Prediction grid (optional) - shared grid across global time range
    if include_prediction_grid:
        # Create integer days from 0 to t_max with step prediction_step_days
        t_pred_days = np.arange(0, t_max + prediction_step_days, prediction_step_days)
        # Ensure we don't exceed max due to floating point
        t_pred_days = t_pred_days[t_pred_days <= t_max]
        t_pred = t_pred_days / t_max  # scale using same scaling factor

        result.update({
            "N_pred": len(t_pred),
            "t_pred": t_pred.tolist(),
        })
    else:
        result.update({
            "N_pred": 0,
            "t_pred": [],
        })

    return result


def prepare_crosslagged_stan_data(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    lag_days: float = 2.0,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data for cross-lagged GP model (weight depends on lagged workouts).

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        lag_days: Lag in days (workouts at time t affect weight at time t+lag).
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data fields for cross-lagged model.
    """
    # Use the existing mismatched data preparation
    stan_data = prepare_bivariate_stan_data_mismatched(
        df_weight=df_weight,
        df_other=df_workout,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        other_time_col=workout_time_col,
        other_value_col=workout_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Add lag parameter (converted to scaled time units)
    t_max = stan_data["_t_max"]
    if t_max == 0:
        t_max = 1  # avoid division by zero

    lag_scaled = lag_days / t_max

    # Ensure lag is within reasonable bounds (0 to t_max)
    if lag_scaled < 0:
        print(f"Warning: lag_days={lag_days} results in negative scaled lag, setting to 0")
        lag_scaled = 0.0
    elif lag_scaled > 1.0:
        print(f"Warning: lag_days={lag_days} exceeds time range, capping at t_max={t_max}")
        lag_scaled = 1.0

    stan_data["lag_scaled"] = lag_scaled

    # Rename keys for cross-lagged model
    # Stan model expects N_workout, t_workout, y_workout (not N_other, t_other, y_other)
    if "N_other" in stan_data:
        stan_data["N_workout"] = stan_data["N_other"]
        stan_data["t_workout"] = stan_data["t_other"]
        stan_data["y_workout"] = stan_data["y_other"]
        # Also rename scaling parameters
        if "_y_other_mean" in stan_data:
            stan_data["_y_workout_mean"] = stan_data["_y_other_mean"]
            stan_data["_y_workout_sd"] = stan_data["_y_other_sd"]
        # Remove old keys to avoid confusion
        del stan_data["N_other"]
        del stan_data["t_other"]
        del stan_data["y_other"]
        if "_y_other_mean" in stan_data:
            del stan_data["_y_other_mean"]
            del stan_data["_y_other_sd"]

    return stan_data


def prepare_crosslagged_stan_data_estimated(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data for cross-lagged GP model with estimated lag parameter.

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data fields for cross-lagged model with estimated lag.
    """
    # Use the existing mismatched data preparation
    stan_data = prepare_bivariate_stan_data_mismatched(
        df_weight=df_weight,
        df_other=df_workout,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        other_time_col=workout_time_col,
        other_value_col=workout_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Add t_max parameter (needed for lag scaling in estimated lag model)
    t_max = stan_data["_t_max"]
    stan_data["t_max"] = t_max  # Add t_max (without underscore) for Stan data

    # Rename keys for cross-lagged model
    # Stan model expects N_workout, t_workout, y_workout (not N_other, t_other, y_other)
    if "N_other" in stan_data:
        stan_data["N_workout"] = stan_data["N_other"]
        stan_data["t_workout"] = stan_data["t_other"]
        stan_data["y_workout"] = stan_data["y_other"]
        # Also rename scaling parameters
        if "_y_other_mean" in stan_data:
            stan_data["_y_workout_mean"] = stan_data["_y_other_mean"]
            stan_data["_y_workout_sd"] = stan_data["_y_other_sd"]
        # Remove old keys to avoid confusion
        del stan_data["N_other"]
        del stan_data["t_other"]
        del stan_data["y_other"]
        if "_y_other_mean" in stan_data:
            del stan_data["_y_other_mean"]
            del stan_data["_y_other_sd"]

    return stan_data


def prepare_crosslagged_stan_data_cumulative(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    lag_days_list: list[float] = [1.0, 2.0, 3.0],
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> dict:
    """Prepare data for cross-lagged GP model with cumulative lag effects.

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        lag_days_list: List of lag values in days (e.g., [1, 2, 3] for 1,2,3 day lags).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data fields for cross-lagged model with cumulative lags.
    """
    # Use the existing mismatched data preparation
    stan_data = prepare_bivariate_stan_data_mismatched(
        df_weight=df_weight,
        df_other=df_workout,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        other_time_col=workout_time_col,
        other_value_col=workout_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Add lag parameters (converted to scaled time units)
    t_max = stan_data["_t_max"]
    if t_max == 0:
        t_max = 1  # avoid division by zero

    # Convert lag days to scaled lags
    lag_scaled = [lag_days / t_max for lag_days in lag_days_list]
    # Ensure lags are non-negative and within reasonable bounds
    lag_scaled = [max(0.0, min(1.0, lag)) for lag in lag_scaled]

    stan_data["K"] = len(lag_days_list)
    stan_data["lag_scaled"] = lag_scaled

    # Rename keys for cross-lagged model
    # Stan model expects N_workout, t_workout, y_workout (not N_other, t_other, y_other)
    if "N_other" in stan_data:
        stan_data["N_workout"] = stan_data["N_other"]
        stan_data["t_workout"] = stan_data["t_other"]
        stan_data["y_workout"] = stan_data["y_other"]
        # Also rename scaling parameters
        if "_y_other_mean" in stan_data:
            stan_data["_y_workout_mean"] = stan_data["_y_other_mean"]
            stan_data["_y_workout_sd"] = stan_data["_y_other_sd"]
        # Remove old keys to avoid confusion
        del stan_data["N_other"]
        del stan_data["t_other"]
        del stan_data["y_other"]
        if "_y_other_mean" in stan_data:
            del stan_data["_y_other_mean"]
            del stan_data["_y_other_sd"]

    return stan_data


if __name__ == "__main__":
    # Test merging
    merged = merge_weight_with_daily_metrics()
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    print("\nColumns:", merged.columns.tolist())

    # Show correlation between weight_mean and resting_heart_rate
    corr = merged[["weight_mean", "resting_heart_rate"]].corr().iloc[0, 1]
    print(f"\nCorrelation (weight_mean vs resting_heart_rate): {corr:.3f}")

    # Show sample rows
    print("\nSample rows (first 5):")
    print(merged[["date", "weight_mean", "resting_heart_rate", "total_steps"]].head())

    # Test bivariate data preparation
    stan_data = prepare_bivariate_stan_data(merged, use_sparse=True, n_inducing_points=30)
    print(f"\nStan data keys: {list(stan_data.keys())}")
    print(f"N: {stan_data['N']}")
    print(f"M: {stan_data['M']}")
    print(f"use_sparse: {stan_data['use_sparse']}")
    print(f"y_weight mean (scaled): {stan_data['y_weight'].mean():.2f}")
    print(f"y_other mean (scaled): {stan_data['y_other'].mean():.2f}")