#!/usr/bin/env python3
"""Test the simplified training decay model (no GP, no daily cycle)."""

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


def test_simple_model():
    """Test the simplified model."""
    print("Testing simplified training decay model (no GP, no daily cycle)...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load only strength training intensity
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
        max_hr=185.0,
    )

    # Create date range
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

    # Standardize intensity: (value - min)/std
    min_val = df_daily['strength_intensity'].min()
    std = df_daily['strength_intensity'].std()
    if std > 0:
        df_daily['strength_intensity_std'] = (df_daily['strength_intensity'] - min_val) / std
    else:
        df_daily['strength_intensity_std'] = df_daily['strength_intensity'] - min_val

    print(f"  Strength intensity: min={min_val:.2f}, std={std:.2f}, scaled mean={df_daily['strength_intensity_std'].mean():.4f}, max={df_daily['strength_intensity_std'].max():.4f}")

    # Standardize weight: (value - mean)/std
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"  Weight: mean={weight_mean:.2f}, std={weight_std:.2f}")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    # Compile and fit model
    model_path = Path("stan/weight_state_space_training_decay_simple.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("\nFitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        adapt_delta=0.8,
        max_treedepth=8,
        show_progress=True,
        seed=12345,
    )

    print(f"\nSampling completed!")

    # Try to get parameter summary
    try:
        # Get summary
        summary = fit.summary()
        print("\nParameter summary:")
        params_of_interest = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
        for param in params_of_interest:
            if param in summary.index:
                mean = summary.loc[param, 'Mean']
                sd = summary.loc[param, 'StdDev']
                rhat = summary.loc[param, 'R_hat'] if 'R_hat' in summary.columns else 'N/A'
                print(f"  {param}: {mean:.3f} ± {sd:.3f} (Rhat: {rhat})")

        # Check Rhat values
        if 'R_hat' in summary.columns:
            high_rhat = summary[summary['R_hat'] > 1.1]
            if len(high_rhat) > 0:
                print(f"\nWARNING: {len(high_rhat)} parameters with Rhat > 1.1")
                for param in high_rhat.index[:5]:  # Show first 5
                    print(f"  {param}: Rhat = {high_rhat.loc[param, 'R_hat']:.3f}")

    except Exception as e:
        print(f"Could not extract parameter summary: {e}")

    # Convert to ArviZ for analysis
    try:
        idata = az.from_cmdstanpy(
            posterior=fit,
            posterior_predictive='y_weight_rep',
            log_likelihood='log_lik_weight',
        )

        # Check effective sample size
        ess = az.ess(idata)
        print(f"\nEffective sample size (min): {ess.min().values:.0f}")

        return fit, idata

    except Exception as e:
        print(f"Could not convert to InferenceData: {e}")
        return fit, None


if __name__ == "__main__":
    test_simple_model()