"""Test prior predictive checks for weight GP model."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import arviz as az

from src.models.fit_weight import generate_prior_predictive


def test_generate_prior_predictive_exists():
    """Test that generate_prior_predictive function exists and has correct signature."""
    # This test will fail because generate_prior_predictive doesn't exist yet
    # This is the RED phase of TDD
    assert hasattr(generate_prior_predictive, '__call__'), \
        "generate_prior_predictive function should exist"

    # Check that it returns expected outputs
    # We'll create minimal test data
    N = 10
    t = np.linspace(0, 1, N)

    # This should fail because function doesn't exist
    result = generate_prior_predictive(t=t, n_samples=100)

    # Check structure
    assert 'y_prior_rep' in result, "Should return prior predictive samples"
    assert 'alpha_samples' in result, "Should return prior parameter samples"
    assert 'rho_samples' in result, "Should return prior parameter samples"
    assert 'sigma_samples' in result, "Should return prior parameter samples"

    # Check shapes
    assert result['y_prior_rep'].shape == (100, N), \
        f"Expected shape (100, {N}), got {result['y_prior_rep'].shape}"

    print("✓ Test for generate_prior_predictive passes (this shouldn't happen in RED phase!)")


def test_prior_predictive_plotting():
    """Test that prior predictive can be plotted."""
    from src.models.plot_weight_cli import plot_weight_enhanced

    # Create mock data
    N = 20
    dates = pd.date_range('2023-01-01', periods=N, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'weight_lbs': np.random.normal(150, 5, N)
    })

    # Create mock InferenceData with prior predictive
    # This will fail because we don't have prior predictive in idata
    idata = az.from_dict(
        posterior={
            'alpha': np.random.normal(0, 1, (1, 100)),
            'rho': np.random.gamma(5, 1, (1, 100)),
            'sigma': np.abs(np.random.normal(0, 0.5, (1, 100))),
            'f': np.random.normal(0, 1, (1, 100, N)),
        },
        prior_predictive={
            'y_prior_rep': np.random.normal(0, 1, (1, 100, N))
        }
    )

    stan_data = {
        '_y_mean': df['weight_lbs'].mean(),
        '_y_sd': df['weight_lbs'].std(),
        'N': N
    }

    # Try to plot with prior predictive - this should work if implemented
    fig, axes = plot_weight_enhanced(
        idata=idata,
        df=df,
        stan_data=stan_data,
        show_prior_predictive=True,
        prior_predictive_level=0.95
    )

    assert fig is not None, "Figure should be created"
    print("✓ Prior predictive plotting test passes (this shouldn't happen in RED phase!)")


if __name__ == "__main__":
    print("Running prior predictive tests...")
    try:
        test_generate_prior_predictive_exists()
        print("WARNING: Test passed in RED phase - this shouldn't happen!")
    except (ImportError, AttributeError, AssertionError) as e:
        print(f"✓ Test correctly fails in RED phase: {e}")

    try:
        test_prior_predictive_plotting()
        print("WARNING: Test passed in RED phase - this shouldn't happen!")
    except Exception as e:
        print(f"✓ Test correctly fails in RED phase: {e}")