"""Test state-space model for weight prediction using workout intensity."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.intensity import prepare_state_space_data
from src.models.fit_weight import fit_state_space_model


def generate_synthetic_data(n_days=100, n_weight_obs=50):
    """Generate synthetic weight and intensity data for testing."""
    # Generate date range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate intensity (some days with workouts)
    np.random.seed(42)
    intensity = np.random.exponential(scale=10, size=n_days)
    # Make most days zero (no workout)
    intensity[np.random.rand(n_days) < 0.7] = 0

    # Generate weight observations at random times
    weight_times = [start_date + timedelta(days=np.random.rand() * n_days) for _ in range(n_weight_obs)]
    weight_times.sort()

    # Generate weight values with some trend and noise
    base_weight = 170.0
    trend = 0.02  # slight upward trend
    noise = np.random.normal(0, 0.5, n_weight_obs)

    # Simple effect: intensity reduces weight with 2-day lag
    weight_values = []
    for i, wt in enumerate(weight_times):
        day_idx = int((wt - start_date).days)
        # Cumulative effect of past 3 days intensity
        past_days = max(0, day_idx - 3)
        intensity_effect = -0.1 * np.sum(intensity[past_days:day_idx]) if day_idx > 0 else 0
        trend_effect = trend * day_idx
        weight = base_weight + trend_effect + intensity_effect + noise[i]
        weight_values.append(weight)

    # Create DataFrames
    df_weight = pd.DataFrame({
        'timestamp': weight_times,
        'weight_lbs': weight_values
    })

    df_intensity = pd.DataFrame({
        'date': dates,
        'intensity': intensity
    })

    return df_weight, df_intensity


def test_data_preparation():
    """Test that we can prepare state-space Stan data from synthetic data."""
    df_weight, df_intensity = generate_synthetic_data()

    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity,
        use_sparse=True,
        n_inducing_points=30,
    )

    required = [
        'D', 'intensity', 'N_weight', 't_weight', 'y_weight', 'day_idx',
        'use_sparse', 'M', 't_inducing'
    ]
    for key in required:
        assert key in stan_data, f"Missing {key}"

    # Check dimensions
    assert stan_data['D'] == 100  # n_days
    assert stan_data['N_weight'] == 50
    assert len(stan_data['intensity']) == 100
    assert len(stan_data['y_weight']) == 50
    assert len(stan_data['day_idx']) == 50
    assert stan_data['M'] == 30  # n_inducing_points

    print(f"Data preparation OK: D={stan_data['D']}, N_weight={stan_data['N_weight']}, M={stan_data['M']}")


def test_model_compilation():
    """Test that the state-space Stan model compiles."""
    from cmdstanpy import CmdStanModel
    import os

    stan_file = "stan/weight_state_space.stan"
    if not os.path.exists(stan_file):
        raise FileNotFoundError(f"Stan file not found: {stan_file}")

    # Try compilation (might take a while)
    print("Compiling state-space Stan model...")
    model = CmdStanModel(stan_file=stan_file)
    print(f"Model compiled successfully: {model}")
    return model


def test_fit_minimal_synthetic():
    """Run minimal fitting on synthetic data to ensure no errors."""
    # Generate synthetic data
    df_weight, df_intensity = generate_synthetic_data(n_days=60, n_weight_obs=30)

    # Use minimal settings
    fit, idata, df_weight_out, df_intensity_out, stan_data = fit_state_space_model(
        df_weight=df_weight,  # Override data loading
        df_intensity=df_intensity,
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True,
    )
    assert fit is not None
    assert idata is not None
    assert len(df_weight_out) == 30
    assert len(df_intensity_out) == 60
    print(f"Minimal fitting successful: {stan_data['N_weight']} weight observations, {stan_data['D']} days")
    return fit, idata


if __name__ == "__main__":
    print("Testing state-space model...")
    test_data_preparation()
    model = test_model_compilation()
    # Optional: run minimal fit (can be slow)
    # Uncomment to test sampling
    fit, idata = test_fit_minimal_synthetic()
    print("All tests passed.")