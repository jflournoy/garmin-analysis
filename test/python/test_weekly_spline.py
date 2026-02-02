"""Test weekly spline model for days of the week."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.weight import load_weight_data, prepare_stan_data
from src.models.fit_weight import fit_weight_model_spline_weekly


def test_prepare_stan_data_includes_day_of_week():
    """Test that prepare_stan_data includes day-of-week information."""
    # Create a simple test dataframe
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "timestamp": dates,
        "weight_lbs": np.random.normal(150, 5, 10),
        "days_since_start": np.arange(10)
    })

    # Test with include_weekly_info=True
    stan_data = prepare_stan_data(df, include_weekly_info=True, weekly_harmonics=2)

    # Check that day_of_week field is included
    assert "day_of_week" in stan_data, "day_of_week should be in stan_data when include_weekly_info=True"
    assert "L" in stan_data, "L (weekly harmonics) should be in stan_data"
    assert stan_data["L"] == 2, "L should match weekly_harmonics parameter"

    # Check day_of_week values are between 0 and 7
    day_of_week = stan_data["day_of_week"]
    assert len(day_of_week) == len(df), "day_of_week should have same length as data"
    assert np.all(day_of_week >= 0) and np.all(day_of_week < 7), "day_of_week should be in [0, 7)"

    # Check that Monday=0, Tuesday=1, etc.
    # First date is Monday 2024-01-01
    assert day_of_week[0] == 0.0, "Monday should be 0.0"

    # Test without weekly info
    stan_data_no_weekly = prepare_stan_data(df, include_weekly_info=False)
    assert "day_of_week" not in stan_data_no_weekly, "day_of_week should not be included when include_weekly_info=False"
    assert "L" not in stan_data_no_weekly, "L should not be included when include_weekly_info=False"


def test_weekly_spline_model_compilation():
    """Test that the weekly spline Stan model compiles."""
    # This test will fail initially because the model doesn't exist yet
    stan_file = Path("stan/weight_gp_spline_weekly.stan")

    # Check if file exists (will fail initially)
    if not stan_file.exists():
        pytest.fail(f"Weekly spline model file {stan_file} does not exist")

    # Try to compile the model (will fail if syntax errors)
    try:
        from cmdstanpy import CmdStanModel
        model = CmdStanModel(stan_file=stan_file)
        assert model is not None, "Model should compile successfully"
    except Exception as e:
        pytest.fail(f"Model compilation failed: {e}")


def test_weekly_spline_model_has_weekly_parameters():
    """Test that the weekly spline model includes weekly parameters."""
    # This test will check the Stan model structure
    stan_file = Path("stan/weight_gp_spline_weekly.stan")

    if not stan_file.exists():
        pytest.skip("Weekly spline model file not found")

    # Read the Stan file to check for required components
    with open(stan_file, 'r') as f:
        content = f.read()

    # Check for weekly parameters
    required_params = [
        "L",  # weekly harmonics parameter
        "sigma_fourier_weekly",  # weekly coefficient scale
        "b_sin_raw",  # raw weekly sine coefficients
        "b_cos_raw",  # raw weekly cosine coefficients
        "f_weekly",  # weekly component
    ]

    for param in required_params:
        assert param in content, f"Weekly parameter '{param}' not found in Stan model"


def test_fit_weight_model_spline_weekly_function_exists():
    """Test that the weekly spline fitting function exists and has correct signature."""
    # This will fail initially because the function doesn't exist
    try:
        # Check if function exists in module
        from src.models.fit_weight import fit_weight_model_spline_weekly
        assert callable(fit_weight_model_spline_weekly), "fit_weight_model_spline_weekly should be callable"
    except ImportError as e:
        pytest.fail(f"fit_weight_model_spline_weekly function not found: {e}")


def test_weekly_spline_model_runs_with_real_data():
    """Test that the weekly spline model runs successfully with real data."""
    # This test verifies the model works end-to-end with actual data
    data_dir = Path("data")

    if not (data_dir / "DI_CONNECT").exists():
        pytest.skip("Data directory not found, skipping integration test")

    # Load data
    load_weight_data(data_dir)

    # Fit weekly model
    fit_weekly, idata_weekly, df_weekly, stan_data_weekly = fit_weight_model_spline_weekly(
        data_dir=data_dir,
        fourier_harmonics=2,
        weekly_harmonics=1,
        use_sparse=True,
        n_inducing_points=50,
        chains=1,  # Use minimal chains for test
        iter_warmup=100,
        iter_sampling=100,
        cache=False,
        force_refit=True,
    )

    # Check that model ran successfully
    assert fit_weekly is not None, "Model fit should not be None"
    assert idata_weekly is not None, "InferenceData should not be None"
    assert len(df_weekly) > 0, "DataFrame should not be empty"

    # Check that weekly parameters are in the posterior
    assert "weekly_amplitude" in idata_weekly.posterior, "weekly_amplitude should be in posterior"
    assert "prop_variance_weekly" in idata_weekly.posterior, "prop_variance_weekly should be in posterior"
    assert "f_weekly" in idata_weekly.posterior, "f_weekly should be in posterior"

    # Check that weekly amplitude is computed (could be small if no weekly pattern)
    weekly_amplitude = idata_weekly.posterior["weekly_amplitude"].values.mean()
    assert weekly_amplitude >= 0, f"Weekly amplitude should be non-negative, got {weekly_amplitude:.4f}"

    # Check that proportion of variance from weekly component is computed
    prop_variance_weekly = idata_weekly.posterior["prop_variance_weekly"].values.mean()
    assert 0 <= prop_variance_weekly <= 1, f"prop_variance_weekly should be between 0 and 1, got {prop_variance_weekly:.4f}"

    # Check that all three components are present
    assert "f_trend" in idata_weekly.posterior, "f_trend should be in posterior"
    assert "f_daily" in idata_weekly.posterior, "f_daily should be in posterior"
    assert "f_weekly" in idata_weekly.posterior, "f_weekly should be in posterior"

    print("\nWeekly model test results:")
    print(f"  Weekly amplitude: {weekly_amplitude:.4f}")
    print(f"  Proportion of variance from weekly component: {prop_variance_weekly:.4f}")
    print(f"  Sigma (residual std): {idata_weekly.posterior['sigma'].values.mean():.4f}")


if __name__ == "__main__":
    # Run tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))