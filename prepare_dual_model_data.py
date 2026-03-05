#!/usr/bin/env python3
"""Prepare data for dual state-space model with separate strength and aerobic training.

This script:
1. Loads weight data
2. Computes aerobic intensity (HR-based) for walking and cycling
3. Computes strength volume (weight-based) from strength training
4. Standardizes all inputs
5. Creates Stan data structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity, compute_workout_intensity
from src.data.workout import load_workout_data
from src.data.health_metrics import load_combined_health_data

def load_aerobic_intensity(data_dir: Path, activity_types: List[str] = None) -> pd.DataFrame:
    """Load aerobic intensity data (HR-based) for specified activity types.

    Args:
        data_dir: Path to data directory.
        activity_types: List of aerobic activity types (default: ['walking', 'cycling']).

    Returns:
        DataFrame with daily aerobic intensity.
    """
    if activity_types is None:
        activity_types = ['walking', 'cycling']

    print(f"Loading aerobic intensity for activities: {activity_types}")

    # Load intensity by activity
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=activity_types,
        max_hr=185.0,
    )

    if len(df_intensity) == 0:
        print("WARNING: No aerobic intensity data found")
        return pd.DataFrame(columns=['date', 'aerobic_intensity'])

    # Sum across aerobic activities
    aerobic_cols = [col for col in df_intensity.columns if col in activity_types and col != 'date']
    df_intensity['aerobic_intensity'] = df_intensity[aerobic_cols].sum(axis=1)

    # Keep only date and aerobic_intensity
    result = df_intensity[['date', 'aerobic_intensity']].copy()

    print(f"  Loaded {len(result)} days of aerobic intensity data")
    print(f"  Non-zero days: {(result['aerobic_intensity'] > 0).sum()}")
    print(f"  Mean intensity: {result['aerobic_intensity'].mean():.2f}")

    return result

def load_strength_volume(data_dir: Path) -> pd.DataFrame:
    """Load strength training volume data.

    Args:
        data_dir: Path to data directory.

    Returns:
        DataFrame with daily strength volume.
    """
    print("Loading strength training volume...")

    # First try to load from our extracted metrics
    strength_metrics_path = data_dir.parent / "output" / "strength_analysis" / "daily_strength_metrics.csv"

    if strength_metrics_path.exists():
        df_strength = pd.read_csv(strength_metrics_path)
        df_strength['date'] = pd.to_datetime(df_strength['date'])

        # Use volume in grams (already in df_strength['strength_volume'])
        df_strength = df_strength[['date', 'strength_volume']].copy()
        df_strength = df_strength.rename(columns={'strength_volume': 'strength_volume_raw'})

        print(f"  Loaded {len(df_strength)} days from precomputed strength metrics")
    else:
        # Fall back to loading from workout data
        print("  Precomputed strength metrics not found, loading from workout data...")

        # Load strength training workouts
        df_workouts = load_workout_data(
            data_dir=data_dir,
            activity_type='strength_training',
            include_exercise_details=False,
        )

        if len(df_workouts) == 0:
            print("  No strength training workouts found")
            return pd.DataFrame(columns=['date', 'strength_volume_raw'])

        # For now, use duration as proxy for volume (since we don't have exercise details)
        # In practice, you'd want to extract volume from summarizedExerciseSets
        df_workouts['strength_volume_raw'] = df_workouts['duration']

        # Group by date
        df_strength = df_workouts.groupby('date')['strength_volume_raw'].sum().reset_index()
        df_strength['date'] = pd.to_datetime(df_strength['date'])

        print(f"  Loaded {len(df_strength)} days from workout data (using duration as volume proxy)")

    print(f"  Non-zero days: {(df_strength['strength_volume_raw'] > 0).sum()}")
    print(f"  Mean volume: {df_strength['strength_volume_raw'].mean():.2f}")

    return df_strength

def prepare_stan_data(
    df_weight: pd.DataFrame,
    df_aerobic: pd.DataFrame,
    df_strength: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_inducing_points: int = 50,
    fourier_harmonics: int = 2,
) -> Dict[str, Any]:
    """Prepare data for Stan model.

    Args:
        df_weight: DataFrame with weight data.
        df_aerobic: DataFrame with aerobic intensity data.
        df_strength: DataFrame with strength volume data.
        start_date: Start date for analysis (YYYY-MM-DD).
        end_date: End date for analysis (YYYY-MM-DD).
        n_inducing_points: Number of inducing points for sparse GP.
        fourier_harmonics: Number of Fourier harmonics for daily cycle.

    Returns:
        Dictionary with Stan data.
    """
    print("\nPreparing Stan data...")

    # Filter by date range if specified
    if start_date:
        start_date = pd.to_datetime(start_date)
        df_weight = df_weight[df_weight['timestamp'] >= start_date]
        df_aerobic = df_aerobic[df_aerobic['date'] >= start_date]
        df_strength = df_strength[df_strength['date'] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        df_weight = df_weight[df_weight['timestamp'] <= end_date]
        df_aerobic = df_aerobic[df_aerobic['date'] <= end_date]
        df_strength = df_strength[df_strength['date'] <= end_date]

    # Determine date range
    all_dates = set()
    if len(df_weight) > 0:
        all_dates.update(df_weight['timestamp'].dt.date)
    if len(df_aerobic) > 0:
        all_dates.update(df_aerobic['date'].dt.date)
    if len(df_strength) > 0:
        all_dates.update(df_strength['date'].dt.date)

    if not all_dates:
        raise ValueError("No data available for the specified date range")

    # Create full date range
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)  # Number of days

    print(f"  Date range: {min_date} to {max_date} ({D} days)")

    # Create daily data frame
    df_daily = pd.DataFrame({'date': date_range})

    # Merge aerobic intensity
    df_daily = pd.merge(df_daily, df_aerobic, on='date', how='left')
    df_daily['aerobic_intensity'] = df_daily['aerobic_intensity'].fillna(0.0)

    # Merge strength volume
    df_daily = pd.merge(df_daily, df_strength, on='date', how='left')
    df_daily['strength_volume_raw'] = df_daily['strength_volume_raw'].fillna(0.0)

    # Standardize inputs but ensure non-negative intensity
    # Workout intensity should be >= 0 (0 = no workout, >0 = workout)
    # Instead of z-scoring which can produce negative values, we'll:
    # 1. Center by subtracting minimum (so min becomes 0)
    # 2. Scale by standard deviation (optional)

    # For aerobic intensity (HR-based, already 0-1 scale)
    aerobic_min = df_daily['aerobic_intensity'].min()
    aerobic_std = df_daily['aerobic_intensity'].std()
    if aerobic_std > 0:
        # Shift so minimum is 0, then optionally scale
        df_daily['aerobic_intensity_std'] = (df_daily['aerobic_intensity'] - aerobic_min) / aerobic_std
    else:
        df_daily['aerobic_intensity_std'] = df_daily['aerobic_intensity'] - aerobic_min

    # For strength volume (grams or duration, non-negative)
    strength_min = df_daily['strength_volume_raw'].min()
    strength_std = df_daily['strength_volume_raw'].std()
    if strength_std > 0:
        df_daily['strength_volume_std'] = (df_daily['strength_volume_raw'] - strength_min) / strength_std
    else:
        df_daily['strength_volume_std'] = df_daily['strength_volume_raw'] - strength_min

    print(f"  Aerobic intensity: min={aerobic_min:.2f}, std={aerobic_std:.2f}")
    print(f"  Strength volume: min={strength_min:.2f}, std={strength_std:.2f}")
    print(f"  Note: Intensity scaled to be non-negative (min shifted to 0)")

    # Prepare weight data
    df_weight = df_weight.copy()
    df_weight['timestamp'] = pd.to_datetime(df_weight['timestamp'])

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"  Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}  # 1-indexed for Stan

    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    missing_days = df_weight['day_idx'].isna().sum()
    if missing_days > 0:
        print(f"  WARNING: {missing_days} weight observations outside date range, dropping")
        df_weight = df_weight[df_weight['day_idx'].notna()]

    # Scale time to [0, 1] for GP
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()

    if time_range > 0:
        df_weight['t_scaled'] = (df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
    else:
        df_weight['t_scaled'] = 0.0

    # Hour of day for daily cycle
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Create inducing points for sparse GP
    if n_inducing_points > 0:
        t_inducing = np.linspace(0, 1, n_inducing_points)
    else:
        t_inducing = np.array([])

    # Prepare Stan data
    stan_data = {
        # Daily data
        'D': D,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'strength_volume': df_daily['strength_volume_std'].values.astype(float),

        # Weight observations
        'N_weight': len(df_weight),
        't_weight': df_weight['t_scaled'].values.astype(float),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),

        # Daily cycle
        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': fourier_harmonics,

        # Sparse GP
        'use_sparse': 1 if n_inducing_points > 0 else 0,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),

        # Prediction (empty for now)
        'N_pred': 0,
        't_pred': np.array([]).astype(float),
        'hour_of_day_pred': np.array([]).astype(float),
    }

    # Add standardization parameters for later transformation
    stan_data['_standardization'] = {
        'weight_mean': float(weight_mean),
        'weight_std': float(weight_std),
        'aerobic_min': float(aerobic_min),
        'aerobic_std': float(aerobic_std),
        'strength_min': float(strength_min),
        'strength_std': float(strength_std),
        'min_time': min_time.isoformat(),
        'max_time': max_time.isoformat(),
    }

    print(f"\nStan data prepared:")
    print(f"  Days (D): {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Aerobic intensity days > 0: {(df_daily['aerobic_intensity'] > 0).sum()}")
    print(f"  Strength volume days > 0: {(df_daily['strength_volume_raw'] > 0).sum()}")
    print(f"  Inducing points: {n_inducing_points}")
    print(f"  Fourier harmonics: {fourier_harmonics}")

    return stan_data, df_weight, df_daily

def main():
    """Main function to prepare data for dual model."""
    data_dir = Path("data")
    output_dir = Path("output/dual_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 70)
        print("PREPARING DATA FOR DUAL STATE-SPACE MODEL")
        print("=" * 70)

        # Load data
        print("\n1. Loading weight data...")
        df_weight = load_weight_data(data_dir)
        print(f"   Loaded {len(df_weight)} weight measurements")

        print("\n2. Loading aerobic intensity data...")
        df_aerobic = load_aerobic_intensity(data_dir, activity_types=['walking', 'cycling'])

        print("\n3. Loading strength volume data...")
        df_strength = load_strength_volume(data_dir)

        # Prepare Stan data
        print("\n4. Preparing Stan data...")
        stan_data, df_weight_processed, df_daily = prepare_stan_data(
            df_weight=df_weight,
            df_aerobic=df_aerobic,
            df_strength=df_strength,
            n_inducing_points=50,
            fourier_harmonics=2,
        )

        # Save data
        print("\n5. Saving data...")

        # Save Stan data as JSON
        # Convert numpy arrays to lists for JSON serialization
        stan_data_json = {}
        for key, value in stan_data.items():
            if key == '_standardization':
                stan_data_json[key] = value
            elif hasattr(value, 'tolist'):
                stan_data_json[key] = value.tolist()
            else:
                stan_data_json[key] = value

        with open(output_dir / "stan_data.json", 'w') as f:
            json.dump(stan_data_json, f, indent=2)

        # Save processed dataframes
        df_weight_processed.to_csv(output_dir / "weight_processed.csv", index=False)
        df_daily.to_csv(output_dir / "daily_data.csv", index=False)

        # Save summary
        summary = {
            'n_days': int(stan_data['D']),
            'n_weight_obs': int(stan_data['N_weight']),
            'n_aerobic_days': int((df_daily['aerobic_intensity'] > 0).sum()),
            'n_strength_days': int((df_daily['strength_volume_raw'] > 0).sum()),
            'date_range_start': df_daily['date'].min().isoformat()[:10],
            'date_range_end': df_daily['date'].max().isoformat()[:10],
            'weight_mean': float(df_weight['weight_lbs'].mean()),
            'weight_std': float(df_weight['weight_lbs'].std()),
            'aerobic_min': float(df_daily['aerobic_intensity'].min()),
            'aerobic_std': float(df_daily['aerobic_intensity'].std()),
            'strength_min': float(df_daily['strength_volume_raw'].min()),
            'strength_std': float(df_daily['strength_volume_raw'].std()),
        }

        with open(output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nData saved to: {output_dir}/")
        print(f"  stan_data.json - Stan model input data")
        print(f"  weight_processed.csv - Processed weight data")
        print(f"  daily_data.csv - Daily intensity/volume data")
        print(f"  data_summary.json - Summary statistics")

        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()