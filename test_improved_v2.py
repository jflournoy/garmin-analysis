#!/usr/bin/env python3
"""Test the improved v2 model with simpler initialization."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

def test_improved_v2():
    """Test the improved v2 model."""
    print("\n" + "=" * 70)
    print("TESTING IMPROVED V2 MODEL")
    print("=" * 70)

    # Configuration
    output_dir = Path("output/four_fitness_improved_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    chains = 4
    iter_warmup = 500  # Start with smaller warmup
    iter_sampling = 500  # Start with smaller sampling
    adapt_delta = 0.99
    max_treedepth = 12
    fourier_harmonics = 2
    n_inducing_points = 50

    print("Loading data...")

    # Load weight data
    df_weight = load_weight_data(data_dir)
    print(f"  Loaded {len(df_weight)} weight measurements")

    # Load aerobic intensity (walking + cycling)
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['walking', 'cycling'],
        max_hr=185.0,
    )

    # Sum walking and cycling
    df_intensity['aerobic_intensity'] = df_intensity.get('walking', 0) + df_intensity.get('cycling', 0)
    df_aerobic = df_intensity[['date', 'aerobic_intensity']].copy()

    # Load strength training intensity
    df_strength_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
        max_hr=185.0,
    )

    if len(df_strength_intensity) > 0 and 'strength_training' in df_strength_intensity.columns:
        df_strength_daily = df_strength_intensity[['date', 'strength_training']].copy()
        df_strength_daily = df_strength_daily.rename(columns={'strength_training': 'strength_intensity'})
    else:
        df_strength_daily = pd.DataFrame(columns=['date', 'strength_intensity'])

    # Create full date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_aerobic['date'].dt.date)
    if len(df_strength_daily) > 0:
        all_dates.update(df_strength_daily['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    print(f"Date range for analysis: {min_date} to {max_date} ({D} days)")

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge aerobic intensity
    df_daily = pd.merge(df_daily, df_aerobic, on='date', how='left')
    df_daily['aerobic_intensity'] = df_daily['aerobic_intensity'].fillna(0.0)

    # Merge strength intensity
    if len(df_strength_daily) > 0:
        df_daily = pd.merge(df_daily, df_strength_daily, on='date', how='left')
    df_daily['strength_intensity'] = df_daily['strength_intensity'].fillna(0.0)

    # Standardize inputs
    for col in ['aerobic_intensity', 'strength_intensity']:
        min_val = df_daily[col].min()
        std = df_daily[col].std()
        if std > 0:
            df_daily[f'{col}_std'] = (df_daily[col] - min_val) / std
        else:
            df_daily[f'{col}_std'] = df_daily[col] - min_val

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

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

    # Hour of day for daily cycle
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Create inducing points
    t_inducing = np.linspace(0, 1, n_inducing_points)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),

        'N_weight': len(df_weight),
        't_weight': df_weight['t_scaled'].values.astype(float),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),

        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': fourier_harmonics,

        'use_sparse': 1,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),

        'N_pred': 0,
        't_pred': np.array([]).astype(float),
        'hour_of_day_pred': np.array([]).astype(float),
    }

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")

    # Fit model
    print("\n" + "=" * 70)
    print("FITTING IMPROVED V2 MODEL")
    print("=" * 70)

    print(f"Configuration:")
    print(f"  Chains: {chains}")
    print(f"  Warmup: {iter_warmup}")
    print(f"  Sampling: {iter_sampling}")
    print(f"  Adapt delta: {adapt_delta}")
    print(f"  Max treedepth: {max_treedepth}")

    # Compile model
    model_path = Path("stan/weight_state_space_four_fitness_improved_v2.stan")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    # Fit model with NO custom inits (let Stan handle it)
    print("\nFitting model...")
    try:
        fit = model.sample(
            data=stan_data,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            show_progress=True,
            seed=12345,
            save_warmup=False,
            # No custom inits - let Stan generate them
        )

        print(f"\nModel fitting completed!")
        print(f"  Samples: {fit.num_draws_sampling}")
        print(f"  Chains: {fit.chains}")

        # Save samples
        fit.save_csvfiles(dir=output_dir)
        print(f"  Results saved to: {output_dir}")

        # Check diagnostics
        print("\n" + "=" * 70)
        print("DIAGNOSTICS")
        print("=" * 70)

        diag = fit.diagnose()
        print(diag)

    except Exception as e:
        print(f"\nERROR during sampling: {e}")
        print("\nTrying with show_console=True to see Stan output...")

        # Try again with console output
        fit = model.sample(
            data=stan_data,
            chains=1,  # Just one chain for debugging
            iter_warmup=100,
            iter_sampling=100,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            show_progress=True,
            seed=12345,
            save_warmup=False,
            show_console=True,
        )

if __name__ == "__main__":
    test_improved_v2()