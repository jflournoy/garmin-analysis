#!/usr/bin/env python3
"""Quick test of the fixed aerobic model."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def test_fixed_model():
    """Test the fixed aerobic model."""
    print("Testing fixed aerobic model with tighter priors...")

    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load both strength and aerobic intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],
        max_hr=185.0,
    )

    # Create full date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_intensity['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    df_act = df_intensity[['date', 'strength_training', 'walking', 'cycling']].copy()
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')

    # Fill missing values with 0
    df_daily['strength_training'] = df_daily['strength_training'].fillna(0.0)
    df_daily['walking'] = df_daily['walking'].fillna(0.0)
    df_daily['cycling'] = df_daily['cycling'].fillna(0.0)

    # Combine walking and cycling into aerobic intensity
    df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']

    # Standardize intensity (shift so min=0)
    for intensity_type in ['strength_training', 'aerobic_intensity']:
        min_val = df_daily[intensity_type].min()
        std = df_daily[intensity_type].std()
        if std > 0:
            df_daily[f'{intensity_type}_std'] = (df_daily[intensity_type] - min_val) / std
        else:
            df_daily[f'{intensity_type}_std'] = df_daily[intensity_type] - min_val

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print("Compiling fixed aerobic model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_aerobic_fixed.stan")

    print("Fitting model with tighter priors...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        adapt_delta=0.99,  # Higher adapt_delta for better convergence
        max_treedepth=15,  # Higher max_treedepth
        show_progress=True,
        seed=12345,
    )

    # Check for divergent transitions
    print("\nChecking for divergent transitions...")
    try:
        diagnose = fit.diagnose()
        print(diagnose)
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    # Check summary
    print("\nModel summary:")
    summary = fit.summary()
    print(summary)

    # Check if there were any divergent transitions
    divergent = fit.diagnostic()['divergent__']
    if divergent is not None:
        print(f"\nDivergent transitions: {divergent.sum()} out of {len(divergent)}")
    else:
        print("\nNo divergent transition information available")

    # Extract key parameters
    draws_df = fit.draws_pd()

    print("\nKey parameter means:")
    for param in ['alpha_d_s', 'alpha_m_s', 'beta_s', 'alpha_d_a', 'alpha_m_a', 'beta_a',
                  'weight_intercept', 'gamma_s', 'gamma_a', 'sigma_w']:
        if param in draws_df.columns:
            mean_val = draws_df[param].mean()
            std_val = draws_df[param].std()
            print(f"  {param}: {mean_val:.3f} ± {std_val:.3f}")

    # Calculate predictions
    print("\nCalculating predictions...")

    # Get fitness samples
    fitness_cols_s = [col for col in draws_df.columns if col.startswith('strength_fitness_stored[')]
    fitness_cols_a = [col for col in draws_df.columns if col.startswith('aerobic_fitness_stored[')]

    if fitness_cols_s and fitness_cols_a:
        # For simplicity, just check a few key metrics
        print("Model appears to have run successfully with fitness states extracted")

        # Check for NaN values
        nan_count = draws_df.isna().sum().sum()
        print(f"NaN values in draws: {nan_count}")

        if nan_count == 0:
            print("✓ No NaN values detected")
        else:
            print("⚠ NaN values detected in draws")

    return fit


if __name__ == "__main__":
    test_fixed_model()