#!/usr/bin/env python3
"""Simple test of Gamma model to check implementation."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def test_gamma_model_simple():
    """Test Gamma model with minimal data."""
    print("Testing Gamma model implementation...")

    # Load minimal data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Take just first 10 observations for quick test
    df_weight = df_weight.head(10).copy()

    # Create simple synthetic data
    D = 30  # 30 days
    dates = pd.date_range(start='2024-01-01', periods=D, freq='D')

    # Create simple intensity data
    np.random.seed(123)
    strength_intensity = np.zeros(D)
    aerobic_intensity = np.zeros(D)

    # Add some training days
    strength_intensity[5] = 1.0
    strength_intensity[10] = 2.0
    strength_intensity[15] = 1.5

    aerobic_intensity[3] = 1.0
    aerobic_intensity[8] = 2.0
    aerobic_intensity[20] = 1.0

    # Create positive weight data
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    y_weight = np.random.gamma(shape=2.0, scale=1.0, size=len(df_weight)) + 1.0

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': strength_intensity.astype(float),
        'aerobic_intensity': aerobic_intensity.astype(float),
        'N_weight': len(y_weight),
        'y_weight': y_weight.astype(float),
        'day_idx': np.random.randint(1, D+1, size=len(y_weight)).astype(int),
    }

    print(f"\nTest data: {D} days, {len(y_weight)} weight observations")
    print(f"Weight range: [{y_weight.min():.3f}, {y_weight.max():.3f}]")

    # Test Gamma model
    print("\nTesting Gamma model...")
    try:
        model = cmdstanpy.CmdStanModel(
            stan_file="stan/weight_state_space_training_decay_aerobic_symmetric_gamma.stan"
        )

        # Run with very few iterations for quick test
        fit = model.sample(
            data=stan_data,
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            show_progress=True,
            seed=12345,
        )

        print("Gamma model ran successfully!")

        # Check key parameters
        draws_df = fit.draws_pd()
        print("\nKey parameters from Gamma model:")
        for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'gamma_shape']:
            if param in draws_df.columns:
                mean_val = draws_df[param].mean()
                print(f"  {param}: {mean_val:.3f}")

    except Exception as e:
        print(f"ERROR in Gamma model: {e}")
        import traceback
        traceback.print_exc()

    # Test log-normal model
    print("\nTesting Log-Normal model...")
    try:
        model = cmdstanpy.CmdStanModel(
            stan_file="stan/weight_state_space_training_decay_aerobic_symmetric_lognormal.stan"
        )

        # Run with very few iterations for quick test
        fit = model.sample(
            data=stan_data,
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            show_progress=True,
            seed=12345,
        )

        print("Log-Normal model ran successfully!")

        # Check key parameters
        draws_df = fit.draws_pd()
        print("\nKey parameters from Log-Normal model:")
        for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'sigma_w']:
            if param in draws_df.columns:
                mean_val = draws_df[param].mean()
                print(f"  {param}: {mean_val:.3f}")

    except Exception as e:
        print(f"ERROR in Log-Normal model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gamma_model_simple()