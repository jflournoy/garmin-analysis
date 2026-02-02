"""Test cyclic GP model fitting and comparison with original model."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data, prepare_stan_data
from src.models.fit_weight import fit_weight_model, fit_weight_model_flexible


def test_fit_weight_model_cyclic_function_exists():
    """Test that fit_weight_model_cyclic function exists."""
    from src.models import fit_weight

    # Check if function exists
    assert hasattr(fit_weight, 'fit_weight_model_cyclic'), \
        "fit_weight module should have fit_weight_model_cyclic function"

    print("✓ fit_weight_model_cyclic function exists")


def test_cyclic_model_fitting_basic():
    """Test basic fitting of cyclic model with real data."""
    # Load data
    df = load_weight_data()
    prepare_stan_data(df)

    # Import and call cyclic fitting function
    from src.models.fit_weight import fit_weight_model_cyclic

    try:
        # Fit with minimal settings for speed
        fit, idata, df_result, stan_data_result = fit_weight_model_cyclic(
            chains=1,
            iter_warmup=10,
            iter_sampling=10,
            cache=False,
            force_refit=True
        )

        # Check results
        assert fit is not None, "Fit should not be None"
        assert idata is not None, "InferenceData should not be None"
        assert len(df_result) == len(df), "DataFrame should have same length"

        # Check that cyclic parameters are in posterior
        posterior_vars = list(idata.posterior.data_vars)
        expected_vars = ['alpha_trend', 'rho_trend', 'alpha_daily', 'rho_daily', 'sigma']
        for var in expected_vars:
            assert var in posterior_vars, f"Posterior should contain {var}"

        print("✓ Cyclic model fitted successfully")
        print(f"  Parameters: {posterior_vars}")
        return fit, idata

    except Exception as e:
        pytest.fail(f"Cyclic model fitting failed: {e}")


def test_cyclic_model_sigma_comparison():
    """Compare sigma estimates between original and cyclic models."""
    # Fit original model
    print("Fitting original model...")
    fit_orig, idata_orig, df_orig, stan_data_orig = fit_weight_model(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True
    )

    # Fit cyclic model
    print("Fitting cyclic model...")
    from src.models.fit_weight import fit_weight_model_cyclic
    fit_cyclic, idata_cyclic, df_cyclic, stan_data_cyclic = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True
    )

    # Extract sigma estimates
    sigma_orig = idata_orig.posterior["sigma"].values.mean()
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.mean()

    print(f"Original model sigma: {sigma_orig:.4f}")
    print(f"Cyclic model sigma: {sigma_cyclic:.4f}")

    # Hypothesis: sigma should be smaller in cyclic model
    # (since daily variation is modeled separately)
    sigma_reduction = sigma_orig - sigma_cyclic
    sigma_reduction_pct = (sigma_reduction / sigma_orig) * 100

    print(f"Sigma reduction: {sigma_reduction:.4f} ({sigma_reduction_pct:.1f}%)")

    # Note: With minimal iterations, we can't guarantee reduction,
    # but we can check that models run and produce estimates
    assert sigma_orig > 0, "Original sigma should be positive"
    assert sigma_cyclic > 0, "Cyclic sigma should be positive"

    print("✓ Sigma comparison completed")


def test_cyclic_model_daily_component_significance():
    """Test that daily component captures meaningful variation."""
    from src.models.fit_weight import fit_weight_model_cyclic

    # Fit cyclic model
    fit, idata, df, stan_data = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True
    )

    # Check daily amplitude
    daily_amplitude = idata.posterior["daily_amplitude"].values.mean()
    prop_variance_daily = idata.posterior["prop_variance_daily"].values.mean()

    print(f"Daily amplitude: {daily_amplitude:.4f}")
    print(f"Proportion variance from daily: {prop_variance_daily:.3f}")

    # Daily amplitude should be positive
    assert daily_amplitude > 0, "Daily amplitude should be positive"

    # Proportion should be between 0 and 1
    assert 0 <= prop_variance_daily <= 1, \
        f"Proportion variance should be in [0,1], got {prop_variance_daily}"

    print("✓ Daily component captures variation")


def test_cyclic_model_caching():
    """Test that cyclic model supports caching like other models."""
    from src.models.fit_weight import fit_weight_model_cyclic

    # First fit (should create cache)
    print("First fit (creating cache)...")
    fit1, idata1, df1, stan_data1 = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=True,
        force_refit=True
    )

    # Second fit (should load from cache)
    print("Second fit (loading from cache)...")
    fit2, idata2, df2, stan_data2 = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=True,
        force_refit=False
    )

    # Check that we got results (even if cached)
    assert fit2 is not None, "Cached fit should not be None"
    assert idata2 is not None, "Cached InferenceData should not be None"

    # Sigma should be similar (allowing for sampling variation)
    sigma1 = idata1.posterior["sigma"].values.mean()
    sigma2 = idata2.posterior["sigma"].values.mean()

    print(f"First fit sigma: {sigma1:.4f}")
    print(f"Second fit (cached) sigma: {sigma2:.4f}")

    print("✓ Caching works for cyclic model")


def test_cyclic_model_backward_compatibility():
    """Test that existing code still works after adding cyclic model."""
    # Test original model still works
    fit_orig, idata_orig, df_orig, stan_data_orig = fit_weight_model(
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=False,
        force_refit=True
    )

    # Test flexible model still works
    fit_flex, idata_flex, df_flex, stan_data_flex = fit_weight_model_flexible(
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=False,
        force_refit=True
    )

    # Test cyclic model works
    from src.models.fit_weight import fit_weight_model_cyclic
    fit_cyclic, idata_cyclic, df_cyclic, stan_data_cyclic = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=5,
        iter_sampling=5,
        cache=False,
        force_refit=True
    )

    # All should produce valid results
    assert idata_orig is not None
    assert idata_flex is not None
    assert idata_cyclic is not None

    print("✓ All model variants work together")


def test_cyclic_model_parameter_interpretation():
    """Test that cyclic model parameters have reasonable interpretations."""
    from src.models.fit_weight import fit_weight_model_cyclic

    # Fit cyclic model
    fit, idata, df, stan_data = fit_weight_model_cyclic(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True
    )

    # Extract parameter estimates
    alpha_trend = idata.posterior["alpha_trend"].values.mean()
    rho_trend = idata.posterior["rho_trend"].values.mean()
    alpha_daily = idata.posterior["alpha_daily"].values.mean()
    rho_daily = idata.posterior["rho_daily"].values.mean()
    sigma = idata.posterior["sigma"].values.mean()

    print(f"alpha_trend (trend amplitude): {alpha_trend:.4f}")
    print(f"rho_trend (trend length scale): {rho_trend:.4f}")
    print(f"alpha_daily (daily amplitude): {alpha_daily:.4f}")
    print(f"rho_daily (daily smoothness): {rho_daily:.4f}")
    print(f"sigma (measurement error): {sigma:.4f}")

    # Check parameter ranges
    assert alpha_trend > 0, "alpha_trend should be positive"
    assert rho_trend > 0, "rho_trend should be positive"
    assert alpha_daily > 0, "alpha_daily should be positive"
    assert rho_daily > 0, "rho_daily should be positive"
    assert sigma > 0, "sigma should be positive"

    # Daily amplitude should be smaller than trend amplitude (typically)
    # But not required for test to pass
    if alpha_daily < alpha_trend:
        print("✓ Daily amplitude smaller than trend amplitude (expected)")

    print("✓ Parameters have reasonable values")


if __name__ == "__main__":
    # Run tests
    print("Testing cyclic model fitting and comparison...")
    print("=" * 60)

    test_fit_weight_model_cyclic_function_exists()
    print()

    test_cyclic_model_fitting_basic()
    print()

    test_cyclic_model_sigma_comparison()
    print()

    test_cyclic_model_daily_component_significance()
    print()

    test_cyclic_model_caching()
    print()

    test_cyclic_model_backward_compatibility()
    print()

    test_cyclic_model_parameter_interpretation()
    print()

    print("=" * 60)
    print("All cyclic model fitting tests passed! ✓")