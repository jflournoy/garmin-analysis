"""Compute workout intensity from heart rate data.

This module provides functions to calculate workout intensity using the formula:
intensity = duration × (avg_hr - resting_hr) / (max_hr - resting_hr)

Where:
- duration: workout duration in seconds
- avg_hr: average heart rate during workout (bpm)
- resting_hr: resting heart rate for that day (bpm)
- max_hr: estimated maximum heart rate (bpm)

The intensity represents the relative cardiovascular load of a workout.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from src.data.workout import load_workout_data
from src.data.health_metrics import load_combined_health_data


def compute_workout_intensity(
    df_workouts: pd.DataFrame,
    df_health: pd.DataFrame,
    max_hr: float = 185.0,
    intensity_col: str = "intensity",
) -> pd.DataFrame:
    """Compute workout intensity for each workout and aggregate by day.

    Args:
        df_workouts: DataFrame with workout data from load_workout_data.
                     Must contain columns: 'date', 'duration', 'avg_hr'.
        df_health: DataFrame with daily health metrics from load_combined_health_data.
                   Must contain column: 'resting_heart_rate'.
        max_hr: Estimated maximum heart rate (default: 185, typical for age 35).
        intensity_col: Name for the intensity column in output.

    Returns:
        DataFrame with columns: 'date', intensity_col (daily summed intensity).
    """
    # Ensure date columns are datetime
    df_workouts = df_workouts.copy()
    df_health = df_health.copy()

    if 'date' in df_workouts.columns:
        df_workouts['date'] = pd.to_datetime(df_workouts['date'])
    if 'date' in df_health.columns:
        df_health['date'] = pd.to_datetime(df_health['date'])

    # Merge resting HR into workouts
    # We need resting_heart_rate for each workout's date
    resting_df = df_health[['date', 'resting_heart_rate']].copy()

    # Forward fill resting HR for days without measurement
    resting_df = resting_df.set_index('date')
    resting_df = resting_df.asfreq('D').ffill()
    resting_df = resting_df.reset_index()

    # Merge with workouts
    df_merged = pd.merge(
        df_workouts,
        resting_df,
        on='date',
        how='left'
    )

    # Filter workouts with heart rate data
    has_hr = df_merged['avg_hr'].notna() & df_merged['resting_heart_rate'].notna()
    df_with_hr = df_merged[has_hr].copy()

    if len(df_with_hr) == 0:
        print("WARNING: No workouts with both avg_hr and resting_heart_rate data")
        # Return empty daily intensity dataframe
        empty_df = pd.DataFrame(columns=['date', intensity_col])
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        return empty_df

    # Compute intensity for each workout
    # Intensity = duration * (avg_hr - resting_hr) / (max_hr - resting_hr)
    # Ensure resting_hr < max_hr to avoid division by zero or negative
    valid_hr = df_with_hr['resting_heart_rate'] < max_hr
    df_with_hr = df_with_hr[valid_hr].copy()

    if len(df_with_hr) == 0:
        print(f"WARNING: No workouts with resting_hr < max_hr ({max_hr})")
        empty_df = pd.DataFrame(columns=['date', intensity_col])
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        return empty_df

    df_with_hr[intensity_col] = (
        df_with_hr['duration'] *
        (df_with_hr['avg_hr'] - df_with_hr['resting_heart_rate']) /
        (max_hr - df_with_hr['resting_heart_rate'])
    )

    # Sum intensity by date
    daily_intensity = df_with_hr.groupby('date')[intensity_col].sum().reset_index()

    # Create full date range to ensure continuous daily series
    if not daily_intensity.empty:
        date_range = pd.date_range(
            start=daily_intensity['date'].min(),
            end=daily_intensity['date'].max(),
            freq='D'
        )
        full_df = pd.DataFrame({'date': date_range})
        daily_intensity = pd.merge(full_df, daily_intensity, on='date', how='left')
        daily_intensity[intensity_col] = daily_intensity[intensity_col].fillna(0.0)
    else:
        daily_intensity = pd.DataFrame(columns=['date', intensity_col])
        daily_intensity['date'] = pd.to_datetime(daily_intensity['date'])

    return daily_intensity


def load_intensity_data(
    data_dir: Union[str, Path] = "data",
    activity_types: Optional[List[str]] = None,
    max_hr: float = 185.0,
    intensity_col: str = "intensity",
) -> pd.DataFrame:
    """Load workout data and compute daily intensity.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.
        activity_types: List of activity types to include.
                       If None, includes all types.
        max_hr: Estimated maximum heart rate.
        intensity_col: Name for intensity column.

    Returns:
        DataFrame with columns: 'date', intensity_col.
    """
    data_dir = Path(data_dir)

    # Load health data for resting HR
    print("Loading health data for resting heart rate...")
    df_health = load_combined_health_data(data_dir)

    # Load workout data
    print("Loading workout data...")
    if activity_types is None:
        # Load all activity types by passing empty list (workout module defaults to strength_training)
        # Actually need to load multiple types. Let's load strength_training, walking, cycling
        activity_types = ['strength_training', 'walking', 'cycling']

    all_workouts = []
    for activity_type in activity_types:
        print(f"  Loading {activity_type}...")
        try:
            df_act = load_workout_data(
                data_dir=data_dir,
                activity_type=activity_type,
                include_exercise_details=False,
            )
            if len(df_act) > 0:
                all_workouts.append(df_act)
        except Exception as e:
            print(f"    Error loading {activity_type}: {e}")

    if not all_workouts:
        print("No workout data loaded")
        empty_df = pd.DataFrame(columns=['date', intensity_col])
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        return empty_df

    df_workouts = pd.concat(all_workouts, ignore_index=True)

    # Compute intensity
    print("Computing workout intensity...")
    df_intensity = compute_workout_intensity(
        df_workouts=df_workouts,
        df_health=df_health,
        max_hr=max_hr,
        intensity_col=intensity_col,
    )

    return df_intensity


def prepare_state_space_data(
    df_weight: pd.DataFrame,
    df_intensity: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    intensity_time_col: str = "date",
    intensity_value_col: str = "intensity",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
) -> Dict[str, Any]:
    """Prepare data for state-space model.

    Args:
        df_weight: DataFrame with weight observations.
        df_intensity: DataFrame with daily intensity values.
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        intensity_time_col: Name of date column in df_intensity.
        intensity_value_col: Name of intensity column in df_intensity.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.

    Returns:
        Dictionary with Stan data format for weight_state_space.stan.
    """
    # Ensure datetime
    df_weight = df_weight.copy()
    df_intensity = df_intensity.copy()

    df_weight[weight_time_col] = pd.to_datetime(df_weight[weight_time_col])
    df_intensity[intensity_time_col] = pd.to_datetime(df_intensity[intensity_time_col])

    # Create full date range covering both weight and intensity data
    start_datetime = min(
        df_weight[weight_time_col].min(),
        df_intensity[intensity_time_col].min()
    )
    end_datetime = max(
        df_weight[weight_time_col].max(),
        df_intensity[intensity_time_col].max()
    )

    # Normalize to midnight for date range (intensity data uses daily aggregates)
    start_date = start_datetime.normalize()
    end_date = end_datetime.normalize()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    D = len(date_range)  # Number of days

    # Create day index mapping using date objects (for alignment with weight timestamps)
    date_to_idx = {date.date(): i + 1 for i, date in enumerate(date_range)}

    # Prepare intensity vector (standardized)
    # Merge intensity data with full date range
    intensity_full = pd.DataFrame({'date': date_range})
    intensity_full = pd.merge(
        intensity_full,
        df_intensity[[intensity_time_col, intensity_value_col]],
        left_on='date',
        right_on=intensity_time_col,
        how='left'
    )
    intensity_full[intensity_value_col] = intensity_full[intensity_value_col].fillna(0.0)

    # Standardize intensity
    intensity_values = intensity_full[intensity_value_col].values
    intensity_mean = np.mean(intensity_values)
    intensity_std = np.std(intensity_values)
    if intensity_std > 0:
        intensity_standardized = (intensity_values - intensity_mean) / intensity_std
    else:
        intensity_standardized = intensity_values - intensity_mean

    # Prepare weight data
    df_weight = df_weight.sort_values(weight_time_col)
    t_weight_days = (df_weight[weight_time_col] - start_datetime).dt.total_seconds() / (24 * 3600)
    t_weight_scaled = t_weight_days.values / t_weight_days.max() if t_weight_days.max() > 0 else t_weight_days.values

    # Standardize weight
    weight_values = df_weight[weight_value_col].values
    weight_mean = np.mean(weight_values)
    weight_std = np.std(weight_values)
    if weight_std > 0:
        weight_standardized = (weight_values - weight_mean) / weight_std
    else:
        weight_standardized = weight_values - weight_mean

    # Map weight observations to day index
    day_idx = []
    for ts in df_weight[weight_time_col]:
        # Find closest date in date_range
        date = ts.date()
        if date in date_to_idx:
            day_idx.append(date_to_idx[date])
        else:
            # Find nearest date
            days_diff = [(d.date() - date).days for d in date_range]
            nearest_idx = np.argmin(np.abs(days_diff))
            day_idx.append(nearest_idx + 1)

    # Prepare sparse GP inducing points if needed
    t_inducing = np.array([])
    if use_sparse:
        # Uniform inducing points across time range
        if n_inducing_points > 0:
            t_inducing = np.linspace(0, 1, n_inducing_points)

    # Prepare prediction grid (optional)
    N_pred = 0
    t_pred = np.array([])

    stan_data = {
        'D': D,
        'intensity': intensity_standardized.astype(float),
        'N_weight': len(weight_standardized),
        't_weight': t_weight_scaled.astype(float),
        'y_weight': weight_standardized.astype(float),
        'day_idx': np.array(day_idx, dtype=int),
        'use_sparse': 1 if use_sparse else 0,
        'M': len(t_inducing),
        't_inducing': t_inducing.astype(float),
        'N_pred': N_pred,
        't_pred': t_pred.astype(float),
        'weight_mean': float(weight_mean),
        'weight_std': float(weight_std),
        '_y_mean': float(weight_mean),
        '_y_sd': float(weight_std),
        'intensity_mean': float(intensity_mean),
        'intensity_std': float(intensity_std),
        '_start_date': start_datetime.isoformat(),  # For interpolation
    }

    return stan_data


def compute_cumulative_intensity(
    df_intensity: pd.DataFrame,
    window_days: int = 7,
    intensity_col: str = "intensity",
    time_col: str = "date",
) -> pd.DataFrame:
    """Compute cumulative intensity over a rolling window.

    Args:
        df_intensity: DataFrame with daily intensity values.
        window_days: Number of days to include in cumulative window.
        intensity_col: Name of intensity column.
        time_col: Name of time column.

    Returns:
        DataFrame with original columns plus 'cumulative_intensity_{window_days}d'.
    """
    df = df_intensity.copy()
    df = df.sort_values(time_col)

    # Ensure continuous daily series
    date_range = pd.date_range(
        start=df[time_col].min(),
        end=df[time_col].max(),
        freq='D'
    )
    df_full = pd.DataFrame({time_col: date_range})
    df_full = pd.merge(df_full, df[[time_col, intensity_col]], on=time_col, how='left')
    df_full[intensity_col] = df_full[intensity_col].fillna(0.0)

    # Compute cumulative intensity over window
    cum_col = f'cumulative_intensity_{window_days}d'
    df_full[cum_col] = df_full[intensity_col].rolling(window=window_days, min_periods=1).sum()

    # Merge back with original df (keeping only dates present in original)
    df_result = pd.merge(
        df,
        df_full[[time_col, cum_col]],
        on=time_col,
        how='left'
    )

    # Forward fill any missing cumulative values
    df_result[cum_col] = df_result[cum_col].ffill().bfill()

    return df_result


def compute_log_intensity(
    df_intensity: pd.DataFrame,
    intensity_col: str = "intensity",
    offset: float = 1.0,
) -> pd.DataFrame:
    """Compute log-transformed intensity to reduce skew.

    Args:
        df_intensity: DataFrame with daily intensity values.
        intensity_col: Name of intensity column.
        offset: Value to add before log transformation (log(intensity + offset)).

    Returns:
        DataFrame with original columns plus 'log_intensity'.
    """
    df = df_intensity.copy()
    df['log_intensity'] = np.log(df[intensity_col] + offset)
    return df


def prepare_state_space_data_with_spline(
    df_weight: pd.DataFrame,
    df_intensity: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    intensity_time_col: str = "date",
    intensity_value_col: str = "intensity",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    fourier_harmonics: int = 2,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_step_days: int = 1,
) -> Dict[str, Any]:
    """Prepare data for state-space model with daily spline component.

    Args:
        df_weight: DataFrame with weight observations.
        df_intensity: DataFrame with daily intensity values.
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        intensity_time_col: Name of date column in df_intensity.
        intensity_value_col: Name of intensity column in df_intensity.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        fourier_harmonics: Number of Fourier harmonics for spline model (K parameter).
        include_prediction_grid: Whether to include prediction grid.
        prediction_hour: Hour of day (0-24) for prediction points.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Dictionary with Stan data format for weight_state_space_impulse_spline.stan.
    """
    # Ensure datetime
    df_weight = df_weight.copy()
    df_intensity = df_intensity.copy()

    df_weight[weight_time_col] = pd.to_datetime(df_weight[weight_time_col])
    df_intensity[intensity_time_col] = pd.to_datetime(df_intensity[intensity_time_col])

    # Create full date range covering both weight and intensity data
    start_datetime = min(
        df_weight[weight_time_col].min(),
        df_intensity[intensity_time_col].min()
    )
    end_datetime = max(
        df_weight[weight_time_col].max(),
        df_intensity[intensity_time_col].max()
    )

    # Normalize to midnight for date range (intensity data uses daily aggregates)
    start_date = start_datetime.normalize()
    end_date = end_datetime.normalize()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    D = len(date_range)  # Number of days

    # Create day index mapping using date objects (for alignment with weight timestamps)
    date_to_idx = {date.date(): i + 1 for i, date in enumerate(date_range)}

    # Prepare intensity vector (standardized)
    # Merge intensity data with full date range
    intensity_full = pd.DataFrame({'date': date_range})
    intensity_full = pd.merge(
        intensity_full,
        df_intensity[[intensity_time_col, intensity_value_col]],
        left_on='date',
        right_on=intensity_time_col,
        how='left'
    )
    intensity_full[intensity_value_col] = intensity_full[intensity_value_col].fillna(0.0)

    # Standardize intensity
    intensity_values = intensity_full[intensity_value_col].values
    intensity_mean = np.mean(intensity_values)
    intensity_std = np.std(intensity_values)
    if intensity_std > 0:
        intensity_standardized = (intensity_values - intensity_mean) / intensity_std
    else:
        intensity_standardized = intensity_values - intensity_mean

    # Prepare weight data
    df_weight = df_weight.sort_values(weight_time_col)
    t_weight_days = (df_weight[weight_time_col] - start_datetime).dt.total_seconds() / (24 * 3600)
    t_weight_scaled = t_weight_days.values / t_weight_days.max() if t_weight_days.max() > 0 else t_weight_days.values

    # Standardize weight
    weight_values = df_weight[weight_value_col].values
    weight_mean = np.mean(weight_values)
    weight_std = np.std(weight_values)
    if weight_std > 0:
        weight_standardized = (weight_values - weight_mean) / weight_std
    else:
        weight_standardized = weight_values - weight_mean

    # Map weight observations to day index
    day_idx = []
    for ts in df_weight[weight_time_col]:
        # Find closest date in date_range
        date = ts.date()
        if date in date_to_idx:
            day_idx.append(date_to_idx[date])
        else:
            # Find nearest date
            days_diff = [(d.date() - date).days for d in date_range]
            nearest_idx = np.argmin(np.abs(days_diff))
            day_idx.append(nearest_idx + 1)

    # Calculate hour of day (0-24) as float
    hour_of_day = df_weight[weight_time_col].dt.hour + df_weight[weight_time_col].dt.minute / 60.0

    # Prepare sparse GP inducing points if needed
    t_inducing = np.array([])
    if use_sparse:
        # Uniform inducing points across time range
        if n_inducing_points > 0:
            t_inducing = np.linspace(0, 1, n_inducing_points)

    # Prepare prediction grid (optional)
    if include_prediction_grid:
        # Create prediction time points
        pred_days = np.arange(0, t_weight_days.max() + prediction_step_days, prediction_step_days)
        t_pred_scaled = pred_days / t_weight_days.max() if t_weight_days.max() > 0 else pred_days

        # Create prediction hour array (all at same hour for simplicity)
        hour_of_day_pred = np.full(len(t_pred_scaled), prediction_hour)

        N_pred = len(t_pred_scaled)
        t_pred = t_pred_scaled.astype(float)
        hour_of_day_pred = hour_of_day_pred.astype(float)
    else:
        N_pred = 0
        t_pred = np.array([])
        hour_of_day_pred = np.array([])

    stan_data = {
        'D': D,
        'intensity': intensity_standardized.astype(float),
        'N_weight': len(weight_standardized),
        't_weight': t_weight_scaled.astype(float),
        'y_weight': weight_standardized.astype(float),
        'day_idx': np.array(day_idx, dtype=int),
        'hour_of_day': hour_of_day.values.astype(float),
        'K': fourier_harmonics,
        'use_sparse': 1 if use_sparse else 0,
        'M': len(t_inducing),
        't_inducing': t_inducing.astype(float),
        'N_pred': N_pred,
        't_pred': t_pred,
        'hour_of_day_pred': hour_of_day_pred,
        'weight_mean': float(weight_mean),
        'weight_std': float(weight_std),
        '_y_mean': float(weight_mean),
        '_y_sd': float(weight_std),
        'intensity_mean': float(intensity_mean),
        'intensity_std': float(intensity_std),
        '_start_date': start_datetime.isoformat(),  # For interpolation
    }

    return stan_data


def prepare_state_space_data_cumulative(
    df_weight: pd.DataFrame,
    df_intensity: pd.DataFrame,
    cumulative_window: int = 7,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    intensity_time_col: str = "date",
    intensity_value_col: str = "intensity",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
) -> Dict[str, Any]:
    """Prepare data for state-space model with cumulative intensity.

    Similar to prepare_state_space_data but uses cumulative intensity
    over a window of days instead of single-day intensity.

    Args:
        df_weight: DataFrame with weight observations.
        df_intensity: DataFrame with daily intensity values.
        cumulative_window: Number of days to include in cumulative intensity.
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        intensity_time_col: Name of date column in df_intensity.
        intensity_value_col: Name of intensity column in df_intensity.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.

    Returns:
        Dictionary with Stan data format for modified weight_state_space.stan
        that expects cumulative intensity.
    """
    # Compute cumulative intensity
    df_intensity_cum = compute_cumulative_intensity(
        df_intensity,
        window_days=cumulative_window,
        intensity_col=intensity_value_col,
        time_col=intensity_time_col
    )

    # Use cumulative intensity column
    cum_col = f'cumulative_intensity_{cumulative_window}d'

    # Prepare data using the standard function but with cumulative intensity
    df_intensity_modified = df_intensity_cum[[intensity_time_col, cum_col]].copy()
    df_intensity_modified = df_intensity_modified.rename(columns={cum_col: intensity_value_col})

    # Call original prepare_state_space_data with modified intensity
    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity_modified,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        intensity_time_col=intensity_time_col,
        intensity_value_col=intensity_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
    )

    # Add cumulative window info to stan_data for reference
    stan_data['cumulative_window'] = cumulative_window

    return stan_data


def load_intensity_by_activity(
    data_dir: Union[str, Path] = "data",
    activity_types: Optional[List[str]] = None,
    max_hr: float = 185.0,
) -> pd.DataFrame:
    """Load workout data and compute daily intensity separated by activity type.

    Args:
        data_dir: Path to data directory containing DI_CONNECT folder.
        activity_types: List of activity types to include.
                       If None, includes ['strength_training', 'walking', 'cycling'].
        max_hr: Estimated maximum heart rate.

    Returns:
        DataFrame with columns: 'date', plus columns for each activity type
        with intensity values for that activity (0 on days without that activity).
    """
    data_dir = Path(data_dir)

    if activity_types is None:
        activity_types = ['strength_training', 'walking', 'cycling']

    # Load health data for resting HR
    print("Loading health data for resting heart rate...")
    df_health = load_combined_health_data(data_dir)

    # Initialize result dataframe with full date range
    result_df = None

    for activity_type in activity_types:
        print(f"  Loading {activity_type} intensity...")
        try:
            # Load workout data for this activity type only
            df_workouts = load_workout_data(
                data_dir=data_dir,
                activity_type=activity_type,
                include_exercise_details=False,
            )

            if len(df_workouts) == 0:
                print(f"    No {activity_type} data found")
                continue

            # Compute intensity for this activity type
            df_intensity = compute_workout_intensity(
                df_workouts=df_workouts,
                df_health=df_health,
                max_hr=max_hr,
                intensity_col=activity_type,  # Use activity type as column name
            )

            if result_df is None:
                result_df = df_intensity.rename(columns={activity_type: 'total'})
                result_df[activity_type] = result_df['total']
                result_df = result_df.drop(columns=['total'])
            else:
                # Merge with existing result_df
                result_df = pd.merge(
                    result_df,
                    df_intensity.rename(columns={activity_type: activity_type}),
                    on='date',
                    how='outer'
                )

        except Exception as e:
            print(f"    Error processing {activity_type}: {e}")

    if result_df is None:
        print("No intensity data loaded for any activity type")
        empty_df = pd.DataFrame(columns=['date'] + activity_types)
        empty_df['date'] = pd.to_datetime(empty_df['date'])
        return empty_df

    # Fill missing values with 0
    for activity_type in activity_types:
        if activity_type in result_df.columns:
            result_df[activity_type] = result_df[activity_type].fillna(0.0)
        else:
            result_df[activity_type] = 0.0

    # Ensure date column is datetime and sort
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df = result_df.sort_values('date').reset_index(drop=True)

    print(f"Loaded intensity data for {len(result_df)} days with activities: {[col for col in result_df.columns if col != 'date']}")
    return result_df