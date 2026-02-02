"""Test bivariate GP model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import merge_weight_with_daily_metrics, prepare_bivariate_stan_data
from src.models.fit_weight import fit_bivariate_model


def test_data_loading():
    """Test that we can load and prepare bivariate data."""
    df = merge_weight_with_daily_metrics()
    assert len(df) > 0
    assert "weight_mean" in df.columns
    assert "resting_heart_rate" in df.columns

    stan_data = prepare_bivariate_stan_data(df, use_sparse=True, n_inducing_points=30)
    required = ["N", "t", "y_weight", "y_other", "use_sparse", "M", "t_inducing"]
    for key in required:
        assert key in stan_data, f"Missing {key}"
    print(f"Data loading OK: N={stan_data['N']}, M={stan_data['M']}")


def test_model_compilation():
    """Test that the bivariate Stan model compiles."""
    from cmdstanpy import CmdStanModel
    import os

    stan_file = "stan/weight_gp_bivariate.stan"
    if not os.path.exists(stan_file):
        raise FileNotFoundError(f"Stan file not found: {stan_file}")

    # Try compilation (might take a while)
    print("Compiling bivariate Stan model...")
    model = CmdStanModel(stan_file=stan_file)
    print(f"Model compiled successfully: {model}")
    return model


def test_fit_minimal():
    """Run minimal fitting (small chains, few iterations) to ensure no errors."""
    # Use minimal settings
    fit, idata, df, stan_data = fit_bivariate_model(
        chains=1,
        iter_warmup=10,
        iter_sampling=10,
        cache=False,
        force_refit=True,
    )
    assert fit is not None
    assert idata is not None
    print(f"Minimal fitting successful: {len(df)} observations")
    return fit, idata


if __name__ == "__main__":
    print("Testing bivariate GP model...")
    test_data_loading()
    model = test_model_compilation()
    # Optional: run minimal fit (can be slow)
    # Uncomment to test sampling
    # fit, idata = test_fit_minimal()
    print("All tests passed.")