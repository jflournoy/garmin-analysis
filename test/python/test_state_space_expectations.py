#!/usr/bin/env python3
"""Test state-space model expectation visualizations.

This test verifies that we can create visualizations of model expectations
for fitness and weight with data overlays.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.intensity import prepare_state_space_data
from src.models.fit_weight import fit_state_space_model


def generate_synthetic_state_space_data(n_days=60, n_weight_obs=30):
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


def test_plot_state_space_expectations_function_exists():
    """Test that plot_state_space_expectations function can be imported."""
    # This test will fail initially because module/function doesn't exist
    from src.models.plot_cyclic import plot_state_space_expectations
    assert plot_state_space_expectations is not None
    print("✓ plot_state_space_expectations function exists")


def test_plot_state_space_expectations_creates_figure():
    """Test that plot_state_space_expectations creates a matplotlib figure."""
    from src.models.plot_cyclic import plot_state_space_expectations

    # Generate synthetic data
    df_weight, df_intensity = generate_synthetic_state_space_data(n_days=60, n_weight_obs=30)

    # Prepare Stan data
    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity,
        use_sparse=True,
        n_inducing_points=30,
    )

    # Fit model with minimal settings
    fit, idata, df_weight_out, df_intensity_out, stan_data = fit_state_space_model(
        df_weight=df_weight,
        df_intensity=df_intensity,
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True,
    )

    # Call plotting function
    fig = plot_state_space_expectations(
        idata=idata,
        df_weight=df_weight_out,
        df_intensity=df_intensity_out,
        stan_data=stan_data,
        model_name="Test State-Space Model",
        output_path=None,  # Don't save for test
        show_ci=True,
    )

    # Verify figure was created
    assert fig is not None
    assert hasattr(fig, 'axes')
    assert len(fig.axes) >= 2  # Should have at least 2 subplots (fitness and weight)

    # Check axes labels
    ax1, ax2 = fig.axes[:2]
    assert ax1.get_xlabel() != ''
    assert ax1.get_ylabel() != ''
    assert ax2.get_xlabel() != ''
    assert ax2.get_ylabel() != ''

    print("✓ plot_state_space_expectations creates figure with 2 subplots")


def test_plot_state_space_expectations_saves_file():
    """Test that plot_state_space_expectations can save to file."""
    from src.models.plot_cyclic import plot_state_space_expectations
    import tempfile
    import os

    # Generate minimal synthetic data for speed
    df_weight, df_intensity = generate_synthetic_state_space_data(n_days=30, n_weight_obs=15)

    # Prepare Stan data
    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity,
        use_sparse=True,
        n_inducing_points=20,
    )

    # Fit model with minimal settings
    fit, idata, df_weight_out, df_intensity_out, stan_data = fit_state_space_model(
        df_weight=df_weight,
        df_intensity=df_intensity,
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=False,
        force_refit=True,
    )

    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_expectations.png")
        fig = plot_state_space_expectations(
            idata=idata,
            df_weight=df_weight_out,
            df_intensity=df_intensity_out,
            stan_data=stan_data,
            model_name="Test Model",
            output_path=output_path,
            show_ci=True,
        )

        # Verify file was created
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        print(f"✓ plot_state_space_expectations saves file to {output_path}")


if __name__ == "__main__":
    print("Testing state-space expectation visualizations...")
    print("=" * 60)

    # Test 1: Function exists
    try:
        test_plot_state_space_expectations_function_exists()
        function_exists = True
    except ImportError as e:
        print(f"✗ plot_state_space_expectations function not found: {e}")
        print("  (Expected failure - function needs to be created)")
        function_exists = False

    # Only run other tests if function exists
    if function_exists:
        # Test 2: Creates figure
        try:
            test_plot_state_space_expectations_creates_figure()
        except Exception as e:
            print(f"✗ plot_state_space_expectations_creates_figure failed: {e}")
            print("  (Check function implementation)")

        # Test 3: Saves file
        try:
            test_plot_state_space_expectations_saves_file()
        except Exception as e:
            print(f"✗ test_plot_state_space_expectations_saves_file failed: {e}")
            print("  (Check function implementation)")

    print("=" * 60)
    print("Expectation visualization tests completed.")
    print("Note: Some failures are expected during TDD red phase.")