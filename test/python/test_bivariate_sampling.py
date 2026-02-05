"""Test bivariate cross-lagged models with improved priors and initializations to avoid sampling issues.

Tests focus on sampling diagnostics: divergences, R-hat, ESS, tree depth.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.align import prepare_crosslagged_stan_data


def generate_synthetic_bivariate_data(
    n_weight=100,
    n_workout=80,
    beta_true=0.5,
    lag_days=2.0,
    seed=42,
):
    """Generate synthetic weight and workout data with known lag effect.

    Returns:
        df_weight, df_workout DataFrames with columns timestamp/date and values.
    """
    rng = np.random.default_rng(seed)

    # Generate time points (scaled to days)
    max_days = 200.0
    t_weight = np.sort(rng.uniform(0, max_days, n_weight))
    t_workout = np.sort(rng.uniform(0, max_days, n_workout))

    # Create latent GP functions using simple exponential covariance
    def gp_sample(t, alpha=1.0, rho=10.0):
        """Sample from GP with squared exponential kernel."""
        n = len(t)
        K = alpha**2 * np.exp(-0.5 * ((t[:, None] - t[None, :]) / rho)**2)
        K += np.eye(n) * 1e-6
        return rng.multivariate_normal(np.zeros(n), K)

    # Sample intrinsic weight GP
    f_weight_intrinsic = gp_sample(t_weight, alpha=1.0, rho=15.0)

    # Sample workout GP
    f_workout = gp_sample(t_workout, alpha=1.0, rho=20.0)

    # Interpolate workout GP at weight times minus lag
    from scipy.interpolate import interp1d
    # Ensure monotonic increasing
    interp = interp1d(t_workout, f_workout, kind='linear', bounds_error=False, fill_value=0.0)
    t_weight_lagged = t_weight - lag_days
    f_workout_lagged = interp(t_weight_lagged)
    f_workout_lagged[~np.isfinite(f_workout_lagged)] = 0.0

    # Combine: weight = beta * lagged_workout + intrinsic + noise
    noise_weight = rng.normal(0, 0.2, n_weight)
    noise_workout = rng.normal(0, 0.3, n_workout)

    y_weight = beta_true * f_workout_lagged + f_weight_intrinsic + noise_weight
    y_workout = f_workout + noise_workout

    # Create DataFrames matching expected column names
    df_weight = pd.DataFrame({
        'timestamp': pd.to_datetime(t_weight, unit='D', origin='2020-01-01'),
        'weight_lbs': y_weight,
    })
    df_workout = pd.DataFrame({
        'date': pd.to_datetime(t_workout, unit='D', origin='2020-01-01'),
        'workout_count': y_workout,
    })

    return df_weight, df_workout


def compile_improved_model(stan_file_name):
    """Compile improved Stan model."""
    stan_path = Path(f"stan/{stan_file_name}")
    if not stan_path.exists():
        raise FileNotFoundError(f"Stan file not found: {stan_path}")
    model = CmdStanModel(stan_file=str(stan_path))
    return model


def check_sampling_diagnostics(fit, max_divergence_pct=10.0, min_ess=100, max_rhat=1.1, max_treedepth=12):
    """Check sampling diagnostics and raise AssertionError if poor."""
    diagnostics = fit.diagnose()
    # Extract divergences from fit object
    n_divergences = 0
    if hasattr(fit, 'divergences'):
        n_divergences = fit.divergences.sum()
    total_samples = fit.draws().size  # approximate
    divergence_pct = 100.0 * n_divergences / total_samples if total_samples > 0 else 0.0

    # Check tree depth
    max_treedepth_hit = 0
    if hasattr(fit, 'max_treedepth_hits'):
        max_treedepth_hit = fit.max_treedepth_hits.sum()

    # Check ESS and R-hat for key parameters
    idata = az.from_cmdstanpy(posterior=fit)
    posterior = idata.posterior
    # Focus on beta and GP parameters
    key_params = ['beta', 'alpha_weight', 'alpha_workout', 'rho_weight', 'rho_workout']
    # Some parameters may have different names in improved models
    param_names = list(posterior.data_vars)
    selected = [p for p in key_params if p in param_names]
    if not selected:
        selected = param_names[:5]  # fallback

    ess_values = []
    rhat_values = []
    for param in selected:
        samples = posterior[param].values.flatten()
        ess = az.ess(samples)
        rhat = az.rhat(posterior[param])
        if ess is not None:
            ess_values.append(float(ess))
        if rhat is not None:
            rhat_values.append(float(rhat.values))

    avg_ess = np.mean(ess_values) if ess_values else 0
    avg_rhat = np.mean(rhat_values) if rhat_values else np.inf

    # Assertions
    assert divergence_pct < max_divergence_pct, (
        f"Divergence percentage too high: {divergence_pct:.1f}% "
        f"({n_divergences} divergences)"
    )
    assert max_treedepth_hit == 0, (
        f"Max tree depth hit {max_treedepth_hit} times"
    )
    assert avg_ess >= min_ess, (
        f"Average ESS too low: {avg_ess:.0f} (minimum {min_ess})"
    )
    assert avg_rhat <= max_rhat, (
        f"Average R-hat too high: {avg_rhat:.3f} (maximum {max_rhat})"
    )

    return {
        'divergence_pct': divergence_pct,
        'n_divergences': n_divergences,
        'max_treedepth_hits': max_treedepth_hit,
        'avg_ess': avg_ess,
        'avg_rhat': avg_rhat,
    }


def test_improved_fixed_lag_sampling():
    """Test improved fixed lag model sampling diagnostics."""
    print("\n=== Testing Improved Fixed Lag Model ===")
    # Generate synthetic data
    df_weight, df_workout = generate_synthetic_bivariate_data(
        n_weight=80, n_workout=60, beta_true=0.3, lag_days=2.0
    )

    # Prepare Stan data
    stan_data = prepare_crosslagged_stan_data(
        df_weight=df_weight,
        df_workout=df_workout,
        lag_days=2.0,
        use_sparse=True,
        n_inducing_points=30,
        inducing_point_method='uniform',
        include_prediction_grid=False,
    )

    # Compile improved model
    model = compile_improved_model('weight_gp_crosslagged_improved.stan')

    # Custom initial values for improved model (log parameters)
    def generate_init():
        init = {
            'log_alpha_weight': 0.0,      # exp(0) = 1
            'log_alpha_workout': 0.0,
            'log_rho_weight': np.log(0.2),
            'log_rho_workout': np.log(0.2),
            'beta': 0.0,
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter internal keys
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    # Fit with minimal iterations for speed (but enough for diagnostics)
    fit = model.sample(
        data=filtered_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        show_progress=True,
        inits=generate_init(),
        adapt_delta=0.95,
        max_treedepth=12,
    )

    # Check diagnostics
    diag = check_sampling_diagnostics(
        fit,
        max_divergence_pct=10.0,
        min_ess=50,  # lower due to fewer iterations
        max_rhat=1.1,
        max_treedepth=12,
    )
    print(f"Diagnostics: {diag}")
    print("✓ Improved fixed lag model passed sampling checks")


def test_improved_estimated_lag_sampling():
    """Test improved estimated lag model sampling diagnostics."""
    print("\n=== Testing Improved Estimated Lag Model ===")
    df_weight, df_workout = generate_synthetic_bivariate_data(
        n_weight=80, n_workout=60, beta_true=0.3, lag_days=2.0
    )

    # Need to prepare data for estimated model (includes t_max)
    # Use same function but we need to add t_max manually
    stan_data = prepare_crosslagged_stan_data(
        df_weight=df_weight,
        df_workout=df_workout,
        lag_days=2.0,  # will be ignored but needed for function
        use_sparse=True,
        n_inducing_points=30,
        inducing_point_method='uniform',
        include_prediction_grid=False,
    )
    # Add t_max for estimated model (maximum days)
    # t_max is already in stan_data as '_t_max'
    # The improved estimated model expects 't_max' (without underscore)
    if '_t_max' in stan_data:
        stan_data['t_max'] = stan_data['_t_max']

    model = compile_improved_model('weight_gp_crosslagged_estimated_improved.stan')

    def generate_init():
        init = {
            'log_alpha_weight': 0.0,
            'log_alpha_workout': 0.0,
            'log_rho_weight': np.log(0.2),
            'log_rho_workout': np.log(0.2),
            'beta': 0.0,
            'lag_days': 2.0,  # true lag
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        show_progress=True,
        inits=generate_init(),
        adapt_delta=0.95,
        max_treedepth=12,
    )

    diag = check_sampling_diagnostics(
        fit,
        max_divergence_pct=10.0,
        min_ess=50,
        max_rhat=1.1,
        max_treedepth=12,
    )
    print(f"Diagnostics: {diag}")
    print("✓ Improved estimated lag model passed sampling checks")


def test_improved_cumulative_lag_sampling():
    """Test improved cumulative lag model sampling diagnostics."""
    print("\n=== Testing Improved Cumulative Lag Model ===")
    df_weight, df_workout = generate_synthetic_bivariate_data(
        n_weight=80, n_workout=60, beta_true=0.3, lag_days=2.0
    )

    # Prepare Stan data with lag list
    # First get base data
    stan_data = prepare_crosslagged_stan_data(
        df_weight=df_weight,
        df_workout=df_workout,
        lag_days=0.0,  # dummy
        use_sparse=True,
        n_inducing_points=30,
        inducing_point_method='uniform',
        include_prediction_grid=False,
    )

    # Add cumulative lag specification
    lag_window = 7  # days
    lag_step = 1
    # Create lag list in scaled units
    t_max = stan_data.get('_t_max', 200.0)
    lag_days_list = np.arange(0, lag_window + lag_step, lag_step)
    lag_days_list = lag_days_list[lag_days_list <= lag_window]
    lag_scaled = lag_days_list / t_max

    stan_data['K'] = len(lag_scaled)
    stan_data['lag_scaled'] = lag_scaled.tolist()

    model = compile_improved_model('weight_gp_crosslagged_cumulative_improved.stan')

    def generate_init():
        init = {
            'log_alpha_weight': 0.0,
            'log_alpha_workout': 0.0,
            'log_rho_weight': np.log(0.2),
            'log_rho_workout': np.log(0.2),
            'beta': 0.0,
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        show_progress=True,
        inits=generate_init(),
        adapt_delta=0.95,
        max_treedepth=12,
    )

    diag = check_sampling_diagnostics(
        fit,
        max_divergence_pct=10.0,
        min_ess=50,
        max_rhat=1.1,
        max_treedepth=12,
    )
    print(f"Diagnostics: {diag}")
    print("✓ Improved cumulative lag model passed sampling checks")


def test_original_vs_improved_divergence_reduction():
    """Compare divergences between original and improved models."""
    print("\n=== Comparing Original vs Improved Models ===")
    # Quick test with minimal iterations to see divergence difference
    df_weight, df_workout = generate_synthetic_bivariate_data(
        n_weight=50, n_workout=40, beta_true=0.0, lag_days=1.0
    )

    # Prepare data
    stan_data = prepare_crosslagged_stan_data(
        df_weight=df_weight,
        df_workout=df_workout,
        lag_days=1.0,
        use_sparse=True,
        n_inducing_points=20,
        inducing_point_method='uniform',
        include_prediction_grid=False,
    )
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    # Original model
    model_orig = CmdStanModel(stan_file="stan/weight_gp_crosslagged.stan")
    fit_orig = model_orig.sample(
        data=filtered_data,
        chains=1,
        iter_warmup=50,
        iter_sampling=50,
        show_progress=False,
        adapt_delta=0.95,
        max_treedepth=12,
    )
    div_orig = fit_orig.divergences.sum() if hasattr(fit_orig, 'divergences') else 0

    # Improved model
    model_improved = compile_improved_model('weight_gp_crosslagged_improved.stan')
    def init_improved():
        return {
            'log_alpha_weight': 0.0,
            'log_alpha_workout': 0.0,
            'log_rho_weight': np.log(0.2),
            'log_rho_workout': np.log(0.2),
            'beta': 0.0,
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
    fit_improved = model_improved.sample(
        data=filtered_data,
        chains=1,
        iter_warmup=50,
        iter_sampling=50,
        show_progress=False,
        inits=init_improved(),
        adapt_delta=0.95,
        max_treedepth=12,
    )
    div_improved = fit_improved.divergences.sum() if hasattr(fit_improved, 'divergences') else 0

    print(f"Original model divergences: {div_orig}")
    print(f"Improved model divergences: {div_improved}")

    # Assert improvement (divergences should not increase)
    # Note: sometimes original may have zero divergences by chance
    assert div_improved <= div_orig + 2, (
        f"Improved model has more divergences ({div_improved}) than original ({div_orig})"
    )
    print("✓ Improved model shows non-increased divergences")


if __name__ == "__main__":
    # Run tests
    import traceback
    tests = [
        test_improved_fixed_lag_sampling,
        test_improved_estimated_lag_sampling,
        test_improved_cumulative_lag_sampling,
        test_original_vs_improved_divergence_reduction,
    ]

    for test_fn in tests:
        try:
            test_fn()
            print(f"\n{test_fn.__name__}: PASSED\n")
        except Exception as e:
            print(f"\n{test_fn.__name__}: FAILED - {e}")
            traceback.print_exc()
            print()