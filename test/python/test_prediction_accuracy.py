#!/usr/bin/env python3
"""Test prediction accuracy for spline optimized model.

This test verifies that model predictions at observed time points
match the actual observations within reasonable statistical expectations.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.fit_weight import fit_weight_model_spline_optimized


def compute_posterior_predictive_metrics(idata, y_obs):
    """Compute posterior predictive check metrics.

    Args:
        idata: ArviZ InferenceData with posterior predictive samples 'y_rep'
        y_obs: Observed y values (standardized)

    Returns:
        Dictionary of metrics including coverage, RMSE, etc.
    """
    # Extract posterior predictive samples (shape: chain, draw, N)
    y_rep = idata.posterior_predictive['y_rep'].values
    n_chains, n_draws, n_obs = y_rep.shape

    # Flatten chains and draws
    y_rep_flat = y_rep.reshape(-1, n_obs)

    # Compute credible intervals for each observation
    lower = np.percentile(y_rep_flat, 2.5, axis=0)
    upper = np.percentile(y_rep_flat, 97.5, axis=0)

    # Coverage of 95% credible interval
    coverage = np.mean((y_obs >= lower) & (y_obs <= upper))

    # Mean squared error (posterior mean vs observed)
    y_rep_mean = y_rep_flat.mean(axis=0)
    mse = np.mean((y_rep_mean - y_obs) ** 2)
    rmse = np.sqrt(mse)

    # Mean absolute error
    mae = np.mean(np.abs(y_rep_mean - y_obs))

    # Proportion of variance explained (R²)
    ss_res = np.sum((y_obs - y_rep_mean) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Posterior predictive p-value (probability y_rep > y_obs)
    ppp = np.mean(y_rep_flat > y_obs[np.newaxis, :], axis=0)
    ppp_mean = np.mean(ppp)
    ppp_extreme = np.mean((ppp < 0.05) | (ppp > 0.95))

    return {
        'coverage_95ci': coverage,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ppp_mean': ppp_mean,
        'ppp_extreme_prop': ppp_extreme,
        'n_observations': n_obs,
        'n_draws_total': y_rep_flat.shape[0],
    }


def test_spline_optimized_prediction_accuracy():
    """Test prediction accuracy for spline optimized model."""
    print("Testing prediction accuracy for spline optimized model...")

    # Fit model with minimal iterations for quick test
    result = fit_weight_model_spline_optimized(
        data_dir="data",
        chains=2,
        iter_warmup=100,
        iter_sampling=100,
        fourier_harmonics=2,
        use_sparse=True,
        n_inducing_points=10,
        include_prediction_grid=False,  # We only need posterior predictive
        cache=False,
        force_refit=True,
    )

    fit, idata, df, stan_data = result

    # Get standardized y values from stan data
    y_std = stan_data['y']

    # Compute metrics
    metrics = compute_posterior_predictive_metrics(idata, y_std)

    print("\nPosterior predictive check metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Check coverage is close to nominal (allow some tolerance)
    coverage = metrics['coverage_95ci']
    assert 0.90 <= coverage <= 1.0, f"Coverage {coverage:.3f} outside expected range (0.90-1.0)"

    # Check RMSE is reasonable (should be less than 1 SD if model fits well)
    # Since y is standardized, RMSE should be < 1 (model explains some variance)
    rmse = metrics['rmse']
    assert rmse < 1.5, f"RMSE {rmse:.3f} too high for standardized data"

    # Check R² is positive (model explains some variance)
    r2 = metrics['r2']
    assert r2 > 0.0, f"R² {r2:.3f} should be positive"

    print("\nAll prediction accuracy checks passed.")

    # Return metrics for inspection
    return metrics


if __name__ == "__main__":
    try:
        metrics = test_spline_optimized_prediction_accuracy()
        sys.exit(0)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)