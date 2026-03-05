#!/usr/bin/env python3
"""Simple test of training decay model with minimal configuration."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import arviz as az

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def test_simple():
    """Simple test with minimal configuration."""
    print("Simple test of strength-only training decay model...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load only strength training intensity
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
        max_hr=185.0,
    )

    # Create simple date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_intensity['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    print(f"  Date range: {min_date} to {max_date} ({D} days)")

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    df_act = df_intensity[['date', 'strength_training']].copy()
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')
    df_daily['strength_intensity'] = df_daily['strength_training'].fillna(0.0)

    # Simple standardization
    intensity_mean = df_daily['strength_intensity'].mean()
    intensity_std = df_daily['strength_intensity'].std()
    if intensity_std > 0:
        df_daily['strength_intensity_std'] = (df_daily['strength_intensity'] - intensity_mean) / intensity_std
    else:
        df_daily['strength_intensity_std'] = df_daily['strength_intensity'] - intensity_mean

    print(f"  Strength intensity: mean={intensity_mean:.2f}, std={intensity_std:.2f}")

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"  Weight: mean={weight_mean:.2f}, std={weight_std:.2f}")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Scale time to [0, 1] for GP
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()
    if time_range > 0:
        df_weight['t_scaled'] = (df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
    else:
        df_weight['t_scaled'] = 0.0

    # Hour of day
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Simple inducing points
    n_inducing_points = 20
    t_inducing = np.linspace(0, 1, n_inducing_points)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        't_weight': df_weight['t_scaled'].values.astype(float),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': 1,  # Fewer harmonics
        'use_sparse': 1,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),
    }

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    # Compile and fit model with fewer iterations
    model_path = Path("stan/weight_state_space_training_decay_strength_only.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("\nFitting model with minimal iterations...")
    fit = model.sample(
        data=stan_data,
        chains=2,  # Fewer chains
        iter_warmup=100,  # Minimal warmup
        iter_sampling=100,  # Minimal sampling
        adapt_delta=0.8,  # Lower adapt_delta
        max_treedepth=8,  # Lower max_treedepth
        show_progress=True,
        seed=12345,
    )

    print(f"\nSampling completed!")
    print(f"  Divergent transitions: {fit.num_divergent}")

    # Check for any errors
    if fit.num_divergent > 0:
        print(f"  WARNING: {fit.num_divergent} divergent transitions")

    # Try to extract some parameters
    try:
        # Get summary of key parameters
        summary = fit.summary()
        print("\nParameter summary:")
        params_of_interest = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
        for param in params_of_interest:
            if param in summary.index:
                mean = summary.loc[param, 'Mean']
                sd = summary.loc[param, 'StdDev']
                print(f"  {param}: {mean:.3f} ± {sd:.3f}")
    except Exception as e:
        print(f"Could not extract parameter summary: {e}")

    return fit


if __name__ == "__main__":
    test_simple()