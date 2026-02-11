"""Fit Bayesian weight model using CmdStanPy."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, CmdStanMCMC, from_csv
from typing import Optional

from src.data.weight import load_weight_data, prepare_stan_data
from src.data.align import merge_weight_with_daily_metrics, prepare_bivariate_stan_data, prepare_bivariate_stan_data_mismatched
from src.data.intensity import prepare_state_space_data


def _get_data_file_mtime(data_dir: Path) -> float:
    """Get modification time of weight data file."""
    data_file = Path(data_dir) / "DI_CONNECT/DI-Connect-Wellness/114762117_userBioMetrics.json"
    return data_file.stat().st_mtime if data_file.exists() else 0


def _compute_cache_key(
    data_dir: Path,
    stan_file: Path,
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    adapt_delta: float = None,
    max_treedepth: int = None,
    alpha_prior_sd: float = None,
    rho_prior_shape: float = None,
    rho_prior_scale: float = None,
    sigma_prior_sd: float = None,
    fourier_harmonics: int = None,
    weekly_harmonics: int = None,
    use_sparse: int = None,
    n_inducing_points: int = None,
    inducing_point_method: str = None,
    include_prediction_grid: bool = None,
    prediction_hour: float = None,
    prediction_hour_step: float = None,
    prediction_step_days: int = None,
) -> tuple[str, dict]:
    """Compute SHA256 hash of model configuration for caching.

    Returns:
        Tuple of (cache_key, config_dict) where config_dict contains all
        parameters used to compute the hash.
    """
    # Include file modification times
    data_mtime = _get_data_file_mtime(data_dir)
    stan_mtime = Path(stan_file).stat().st_mtime if Path(stan_file).exists() else 0

    config = {
        "data_dir": str(data_dir),
        "data_mtime": data_mtime,
        "stan_file": str(stan_file),
        "stan_mtime": stan_mtime,
        "chains": chains,
        "iter_warmup": iter_warmup,
        "iter_sampling": iter_sampling,
    }

    if adapt_delta is not None:
        config["adapt_delta"] = adapt_delta
    if max_treedepth is not None:
        config["max_treedepth"] = max_treedepth

    # Include hyperparameters if provided (for flexible model)
    if alpha_prior_sd is not None:
        config["alpha_prior_sd"] = alpha_prior_sd
    if rho_prior_shape is not None:
        config["rho_prior_shape"] = rho_prior_shape
    if rho_prior_scale is not None:
        config["rho_prior_scale"] = rho_prior_scale
    if sigma_prior_sd is not None:
        config["sigma_prior_sd"] = sigma_prior_sd
    if fourier_harmonics is not None:
        config["fourier_harmonics"] = fourier_harmonics
    if weekly_harmonics is not None:
        config["weekly_harmonics"] = weekly_harmonics
    if use_sparse is not None:
        config["use_sparse"] = use_sparse
    if n_inducing_points is not None:
        config["n_inducing_points"] = n_inducing_points
    if inducing_point_method is not None:
        config["inducing_point_method"] = inducing_point_method
    if include_prediction_grid is not None:
        config["include_prediction_grid"] = include_prediction_grid
    if prediction_hour is not None:
        config["prediction_hour"] = prediction_hour
    if prediction_hour_step is not None:
        config["prediction_hour_step"] = prediction_hour_step
    if prediction_step_days is not None:
        config["prediction_step_days"] = prediction_step_days

    # Convert to JSON and hash
    config_str = json.dumps(config, sort_keys=True, default=str)
    cache_key = hashlib.sha256(config_str.encode()).hexdigest()
    return cache_key, config


def _save_cache(
    cache_dir: Path,
    fit: CmdStanMCMC,
    idata: az.InferenceData,
    df: pd.DataFrame,
    stan_data: dict,
    config: dict,
):
    """Save model results to cache directory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    csv_dir = cache_dir / "csv"
    csv_dir.mkdir(exist_ok=True)
    for csv_file in fit.runset.csv_files:
        shutil.copy2(csv_file, csv_dir / Path(csv_file).name)

    # Save InferenceData as netcdf
    idata.to_netcdf(cache_dir / "idata.nc")

    # Save DataFrame as parquet
    df.to_parquet(cache_dir / "df.parquet")

    # Save stan_data as JSON (convert numpy arrays)
    stan_data_serializable = {}
    for key, value in stan_data.items():
        if isinstance(value, np.ndarray):
            stan_data_serializable[key] = value.tolist()
        elif isinstance(value, np.generic):
            # Convert numpy scalars to Python scalars (e.g., np.float64, np.int64, np.bool_)
            stan_data_serializable[key] = value.item()
        elif isinstance(value, (int, float, str, list, dict, bool)) or value is None:
            stan_data_serializable[key] = value
        else:
            # Convert other types to string
            stan_data_serializable[key] = str(value)

    with open(cache_dir / "stan_data.json", "w") as f:
        json.dump(stan_data_serializable, f, indent=2)

    # Save metadata
    metadata = {
        "config": config,
        "created": datetime.now().isoformat(),
        "csv_files": [Path(f).name for f in fit.runset.csv_files],
        "n_chains": fit.chains,
        "n_samples": fit.num_draws_sampling,
    }

    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model results cached to {cache_dir}")


def _load_cache(cache_dir: Path, stan_file: Path):
    """Load model results from cache directory."""
    cache_dir = Path(cache_dir)

    # Load metadata
    with open(cache_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load DataFrame
    df = pd.read_parquet(cache_dir / "df.parquet")

    # Load stan_data
    with open(cache_dir / "stan_data.json", "r") as f:
        stan_data_loaded = json.load(f)

    # Convert lists back to numpy arrays for numeric fields
    stan_data = {}
    for key, value in stan_data_loaded.items():
        if isinstance(value, list) and key not in ["t", "y"]:  # t and y are lists in Stan data
            # Check if it's a numeric list
            if value and isinstance(value[0], (int, float)):
                stan_data[key] = np.array(value)
            else:
                stan_data[key] = value
        else:
            stan_data[key] = value

    # Load InferenceData
    idata = az.from_netcdf(cache_dir / "idata.nc")
    idata.load()  # Ensure data is loaded into memory
    idata.close()  # Close file handle to allow overwriting

    # Reconstruct CmdStanMCMC object from CSV files
    # Need to compile model first (fast if already compiled)
    CmdStanModel(stan_file=stan_file)
    csv_files = [str(cache_dir / "csv" / fname) for fname in metadata["csv_files"]]
    fit = from_csv(csv_files)

    print(f"Loaded cached model results from {cache_dir}")
    return fit, idata, df, stan_data


def _cache_exists(cache_dir: Path) -> bool:
    """Check if cache directory contains all required files."""
    required = [
        cache_dir / "metadata.json",
        cache_dir / "idata.nc",
        cache_dir / "df.parquet",
        cache_dir / "stan_data.json",
        cache_dir / "csv",
    ]
    return all(path.exists() for path in required)


def extract_predictions(idata, stan_data):
    """Extract predictions from InferenceData and back-transform to original scale.

    Args:
        idata: ArviZ InferenceData object from fitted model
        stan_data: Stan data dictionary with scaling parameters

    Returns:
        Dictionary with prediction results if available, else empty dict.
        Keys include:
        - t_pred: prediction time points (days since start)
        - t_pred_scaled: scaled time points
        - hour_of_day_pred: hour of day for predictions
        - f_pred_mean, f_pred_lower, f_pred_upper: mean and 95% CI for f_pred
        - y_pred_mean, y_pred_lower, y_pred_upper: mean and 95% CI for y_pred
        All values in original scale (lbs).
    """
    N_pred = stan_data.get("N_pred", 0)
    if N_pred == 0 or "f_pred" not in idata.posterior_predictive:
        return {}

    # Ensure numeric types (defensive conversion for cached data)
    y_mean = float(stan_data["_y_mean"])
    y_sd = float(stan_data["_y_sd"])
    t_max = float(stan_data["_t_max"])

    # Back-transform time
    t_pred_scaled = np.array(stan_data.get("t_pred", []))
    t_pred = t_pred_scaled * t_max

    # Extract samples
    f_pred_samples = idata.posterior_predictive["f_pred"].values  # shape (chain, draw, N_pred)
    y_pred_samples = idata.posterior_predictive["y_pred"].values

    # Compute summary statistics
    f_pred_mean = f_pred_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_pred_lower = np.percentile(f_pred_samples, 2.5, axis=(0, 1)) * y_sd + y_mean
    f_pred_upper = np.percentile(f_pred_samples, 97.5, axis=(0, 1)) * y_sd + y_mean

    y_pred_mean = y_pred_samples.mean(axis=(0, 1)) * y_sd + y_mean
    y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=(0, 1)) * y_sd + y_mean
    y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=(0, 1)) * y_sd + y_mean

    return {
        "t_pred": t_pred,
        "t_pred_scaled": t_pred_scaled,
        "hour_of_day_pred": np.array(stan_data.get("hour_of_day_pred", [])),
        "f_pred_mean": f_pred_mean,
        "f_pred_lower": f_pred_lower,
        "f_pred_upper": f_pred_upper,
        "y_pred_mean": y_pred_mean,
        "y_pred_lower": y_pred_lower,
        "y_pred_upper": y_pred_upper,
    }


def fit_weight_model(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    cache: bool = True,
    force_refit: bool = False,
) -> tuple:
    """Fit the GP weight model and return results.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            # No hyperparameters for basic model
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(df)
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Compile and fit model
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting model...")
    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": stan_data["y"]},
        coords={"obs": np.arange(stan_data["N"])},
        dims={"f": ["obs"], "y_rep": ["obs"], "y": ["obs"], "log_lik": ["obs"]},
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def plot_weight_fit(idata, df, stan_data, output_path: Path | str = None):
    """Plot the fitted weight model with uncertainty."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Back-transform predictions to original scale
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Extract posterior mean and credible intervals for f
    f_samples = idata.posterior["f"].values
    f_mean = f_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_lower = np.percentile(f_samples, 2.5, axis=(0, 1)) * y_sd + y_mean
    f_upper = np.percentile(f_samples, 97.5, axis=(0, 1)) * y_sd + y_mean

    # Plot 1: Fit with uncertainty
    ax = axes[0]
    ax.scatter(df["date"], df["weight_lbs"], alpha=0.5, s=20, label="Observations")
    ax.plot(df["date"], f_mean, "k-", linewidth=2, label="GP mean")
    ax.fill_between(df["date"], f_lower, f_upper, alpha=0.3, color="blue", label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Weight Over Time - Gaussian Process Fit")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Posterior predictive check
    ax = axes[1]
    az.plot_ppc(idata, ax=ax, num_pp_samples=50)
    ax.set_title("Posterior Predictive Check")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def print_summary(idata, stan_data):
    """Print model summary statistics."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    # Key parameters
    summary = az.summary(idata, var_names=["alpha", "rho", "sigma", "trend_change"])
    print("\nKey parameters:")
    print(summary)

    # Diagnostics
    print("\nDiagnostics:")
    print(f"  R-hat max: {summary['r_hat'].max():.3f}")
    print(f"  ESS min: {summary['ess_bulk'].min():.0f}")

    # Back-transform trend change
    y_sd = stan_data["_y_sd"]
    trend_change = idata.posterior["trend_change"].values.mean() * y_sd
    print(f"\nTrend change (original scale): {trend_change:.2f} lbs")


def fit_weight_model_flexible(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_flexible.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    alpha_prior_sd: float = 1.0,
    rho_prior_shape: float = 5.0,
    rho_prior_scale: float = 1.0,
    sigma_prior_sd: float = 0.5,
    cache: bool = True,
    force_refit: bool = False,
) -> tuple:
    """Fit the GP weight model with flexible priors and return results.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (should be the flexible version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        alpha_prior_sd: Standard deviation for alpha ~ normal(0, sd)
        rho_prior_shape: Shape parameter for rho ~ inv_gamma(shape, scale)
        rho_prior_scale: Scale parameter for rho ~ inv_gamma(shape, scale)
        sigma_prior_sd: Standard deviation for sigma ~ normal(0, sd)
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with hyperparameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            alpha_prior_sd=alpha_prior_sd,
            rho_prior_shape=rho_prior_shape,
            rho_prior_scale=rho_prior_scale,
            sigma_prior_sd=sigma_prior_sd,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(df)
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Add hyperparameters to Stan data
    stan_data["alpha_prior_sd"] = alpha_prior_sd
    stan_data["rho_prior_shape"] = rho_prior_shape
    stan_data["rho_prior_scale"] = rho_prior_scale
    stan_data["sigma_prior_sd"] = sigma_prior_sd

    # Compile and fit model
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting model...")
    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": stan_data["y"]},
        coords={"obs": np.arange(stan_data["N"])},
        dims={"f": ["obs"], "y_rep": ["obs"], "y": ["obs"], "log_lik": ["obs"]},
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def generate_prior_predictive(
    t,
    n_samples=100,
    alpha_prior_sd=1.0,
    rho_prior_shape=5.0,
    rho_prior_scale=1.0,
    sigma_prior_sd=0.5,
    seed=None,
):
    """Generate prior predictive samples for the GP weight model.

    Samples parameters from priors, then generates GP function values and
    prior predictive observations.

    Args:
        t: Array of time points (scaled to [0,1])
        n_samples: Number of prior samples to generate
        alpha_prior_sd: Standard deviation for alpha ~ normal(0, sd)
        rho_prior_shape: Shape parameter for rho ~ inv_gamma(shape, scale)
        rho_prior_scale: Scale parameter for rho ~ inv_gamma(shape, scale)
        sigma_prior_sd: Standard deviation for sigma ~ half-normal(0, sd)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with prior predictive samples and parameter samples:
        - y_prior_rep: Prior predictive samples (n_samples, N)
        - f_prior_rep: GP function samples (n_samples, N)
        - alpha_samples: Alpha prior samples (n_samples,)
        - rho_samples: Rho prior samples (n_samples,)
        - sigma_samples: Sigma prior samples (n_samples,)
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(t)
    # Sample parameters from priors
    alpha_samples = np.random.normal(0, alpha_prior_sd, n_samples)
    # Inverse gamma samples: using gamma distribution transformation
    # If X ~ Gamma(shape, scale), then 1/X ~ InvGamma(shape, 1/scale)
    # Actually inv_gamma(shape, scale) has pdf: scale^shape / Gamma(shape) * x^(-shape-1) * exp(-scale/x)
    # We can sample as scale / np.random.gamma(shape, 1, n_samples)
    rho_samples = rho_prior_scale / np.random.gamma(rho_prior_shape, 1, n_samples)
    # Half-normal for sigma (positive only)
    sigma_samples = np.abs(np.random.normal(0, sigma_prior_sd, n_samples))

    # Precompute pairwise squared distances (N x N)
    # Efficient computation using broadcasting
    t_col = t[:, np.newaxis]
    t_row = t[np.newaxis, :]
    sq_dist = (t_col - t_row) ** 2  # shape (N, N)

    # Initialize arrays for GP function and prior predictive
    f_prior_rep = np.zeros((n_samples, N))
    y_prior_rep = np.zeros((n_samples, N))

    for i in range(n_samples):
        alpha = alpha_samples[i]
        rho = rho_samples[i]
        sigma = sigma_samples[i]

        # Compute covariance matrix
        K = alpha**2 * np.exp(-sq_dist / (2 * rho**2))
        # Add jitter for numerical stability (same as Stan model)
        np.fill_diagonal(K, K.diagonal() + alpha**2 * 1e-4 + 1e-6)

        # Cholesky decomposition
        try:
            L_K = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more jitter and retry
            np.fill_diagonal(K, K.diagonal() + alpha**2 * 1e-3 + 1e-5)
            L_K = np.linalg.cholesky(K)

        # Sample standardized GP values
        eta = np.random.normal(0, 1, N)
        f = L_K @ eta  # GP function values

        # Sample prior predictive observations (add observation noise)
        y = f + np.random.normal(0, sigma, N)

        f_prior_rep[i, :] = f
        y_prior_rep[i, :] = y

    return {
        'y_prior_rep': y_prior_rep,
        'f_prior_rep': f_prior_rep,
        'alpha_samples': alpha_samples,
        'rho_samples': rho_samples,
        'sigma_samples': sigma_samples,
    }


def fit_weight_model_cyclic(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_cyclic.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    cache: bool = True,
    force_refit: bool = False,
) -> tuple:
    """Fit the cyclic GP weight model (trend + daily) and return results.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (cyclic version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with period_daily
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            # No hyperparameters for cyclic model (uses default priors)
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(df)  # Includes hour info by default
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for cyclic model
    required_fields = ["period_daily"]
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Cyclic model requires {field} in stan_data. "
                           f"Make sure prepare_stan_data() includes hour info.")

    # Compile and fit model
    print("Compiling cyclic Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting cyclic model...")
    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": stan_data["y"]},
        coords={"obs": np.arange(stan_data["N"])},
        dims={
            "f_trend": ["obs"],
            "f_daily": ["obs"],
            "f_total": ["obs"],
            "y_rep": ["obs"],
            "y": ["obs"],
            "f_trend_std": ["obs"],
            "f_daily_std": ["obs"],
            "log_lik": ["obs"],
        },
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_spline(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_spline.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    fourier_harmonics: int = 2,
    cache: bool = True,
    force_refit: bool = False,
) -> tuple:
    """Fit the spline GP weight model (trend + Fourier spline for daily cycles) and return results.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (spline version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        fourier_harmonics: Number of Fourier harmonics (K parameter)
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with hour_of_day and K
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            fourier_harmonics=fourier_harmonics,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(df, fourier_harmonics=fourier_harmonics)  # Includes hour info and K
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for spline model
    required_fields = ["hour_of_day", "K"]
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Spline model requires {field} in stan_data. "
                           f"Make sure prepare_stan_data() includes hour info.")

    # Compile and fit model
    print("Compiling spline Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting spline model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_trend': 1.0,
            'rho_trend': 0.2,
            'sigma_fourier': 0.1,
            'sigma': 0.1,
            'a_sin_raw': np.zeros(stan_data['K']),
            'a_cos_raw': np.zeros(stan_data['K']),
            'eta_trend': np.zeros(stan_data['N']),
        }
        return init

    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
        "fourier_coeff": np.arange(stan_data["K"]),
    }
    dims = {
        "f_trend": ["obs"],
        "f_daily": ["obs"],
        "f_total": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "f_trend_std": ["obs"],
        "f_daily_std": ["obs"],
        "a_sin": ["fourier_coeff"],
        "a_cos": ["fourier_coeff"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_trend_pred": ["pred"],
            "f_daily_pred": ["pred"],
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred", "f_trend_pred", "f_daily_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_spline_optimized(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_spline_optimized.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    fourier_harmonics: int = 2,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit the OPTIMIZED spline GP weight model with cov_exp_quad, optional sparse GP, and Student-t likelihood for robustness.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (optimized spline version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        fourier_harmonics: Number of Fourier harmonics (K parameter)
        use_sparse: Whether to use sparse GP approximation (default: False = full GP)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with sparse GP parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            fourier_harmonics=fourier_harmonics,
            use_sparse=1 if use_sparse else 0,
            n_inducing_points=n_inducing_points if use_sparse else None,
            inducing_point_method=inducing_point_method if use_sparse else None,
            include_prediction_grid=include_prediction_grid,
            prediction_hour=prediction_hour,
            prediction_hour_step=prediction_hour_step,
            prediction_step_days=prediction_step_days,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(
        df,
        fourier_harmonics=fourier_harmonics,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_hour=prediction_hour,
        prediction_hour_step=prediction_hour_step,
        prediction_step_days=prediction_step_days,
    )
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for optimized spline model
    required_fields = ["hour_of_day", "K"]
    if use_sparse:
        required_fields.extend(["use_sparse", "M", "t_inducing"])
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Optimized spline model requires {field} in stan_data. "
                           f"Make sure prepare_stan_data() includes sparse GP options if needed.")

    # Compile and fit model
    print("Compiling optimized spline Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting optimized spline model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_trend': 1.0,
            'rho_trend': 0.2,
            'sigma_fourier': 0.1,
            'sigma': 0.1,
            'nu': 10.0,  # Student-t degrees of freedom
            'a_sin_raw': np.zeros(stan_data['K']),
            'a_cos_raw': np.zeros(stan_data['K']),
            'eta_trend': np.zeros(stan_data['N']),
            'eta_inducing': np.zeros(stan_data['M']),
        }
        return init

    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
        "fourier_coeff": np.arange(stan_data["K"]),
    }
    dims = {
        "f_trend": ["obs"],
        "f_daily": ["obs"],
        "f_total": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "f_trend_std": ["obs"],
        "f_daily_std": ["obs"],
        "a_sin": ["fourier_coeff"],
        "a_cos": ["fourier_coeff"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_trend_pred": ["pred"],
            "f_daily_pred": ["pred"],
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred", "f_trend_pred", "f_daily_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_spline_weekly(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_spline_weekly.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    fourier_harmonics: int = 2,
    weekly_harmonics: int = 2,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit the WEEKLY spline GP weight model with trend + daily + weekly Fourier components and Student-t likelihood for robustness.

    This model extends the optimized spline model by adding weekly cyclic component
    to capture day-of-week patterns (e.g., weekend vs weekday effects).

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (weekly spline version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        fourier_harmonics: Number of Fourier harmonics for daily cycles (K parameter)
        weekly_harmonics: Number of Fourier harmonics for weekly cycles (L parameter)
        use_sparse: Whether to use sparse GP approximation (default: False = full GP)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with weekly parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            fourier_harmonics=fourier_harmonics,
            weekly_harmonics=weekly_harmonics,
            use_sparse=1 if use_sparse else 0,
            n_inducing_points=n_inducing_points if use_sparse else None,
            inducing_point_method=inducing_point_method if use_sparse else None,
            include_prediction_grid=include_prediction_grid,
            prediction_hour=prediction_hour,
            prediction_hour_step=prediction_hour_step,
            prediction_step_days=prediction_step_days,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(
        df,
        fourier_harmonics=fourier_harmonics,
        weekly_harmonics=weekly_harmonics,
        include_weekly_info=True,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_hour=prediction_hour,
        prediction_hour_step=prediction_hour_step,
        prediction_step_days=prediction_step_days,
    )
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for weekly spline model
    required_fields = ["hour_of_day", "K", "day_of_week", "L"]
    if use_sparse:
        required_fields.extend(["use_sparse", "M", "t_inducing"])
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Weekly spline model requires {field} in stan_data. "
                           f"Make sure prepare_stan_data() includes weekly info and sparse GP options if needed.")

    # Compile and fit model
    print("Compiling weekly spline Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting weekly spline model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_trend': 1.0,
            'rho_trend': 0.2,
            'sigma_fourier_daily': 0.1,
            'sigma_fourier_weekly': 0.1,
            'sigma': 0.1,
            'nu': 10.0,  # Student-t degrees of freedom
            'a_sin_raw': np.zeros(stan_data['K']),
            'a_cos_raw': np.zeros(stan_data['K']),
            'b_sin_raw': np.zeros(stan_data['L']),
            'b_cos_raw': np.zeros(stan_data['L']),
            'eta_trend': np.zeros(stan_data['N']),
            'eta_inducing': np.zeros(stan_data['M']),
        }
        return init

    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
        "fourier_coeff_daily": np.arange(stan_data["K"]),
        "fourier_coeff_weekly": np.arange(stan_data["L"]),
    }
    dims = {
        "f_trend": ["obs"],
        "f_daily": ["obs"],
        "f_weekly": ["obs"],
        "f_total": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "f_trend_std": ["obs"],
        "f_daily_std": ["obs"],
        "f_weekly_std": ["obs"],
        "a_sin": ["fourier_coeff_daily"],
        "a_cos": ["fourier_coeff_daily"],
        "b_sin": ["fourier_coeff_weekly"],
        "b_cos": ["fourier_coeff_weekly"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_trend_pred": ["pred"],
            "f_daily_pred": ["pred"],
            "f_weekly_pred": ["pred"],
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred", "f_trend_pred", "f_daily_pred", "f_weekly_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_optimized(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_optimized.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit the OPTIMIZED original GP weight model with cov_exp_quad and optional sparse GP.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (optimized original version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        use_sparse: Whether to use sparse GP approximation (default: False = full GP)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with sparse GP parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            use_sparse=1 if use_sparse else 0,
            n_inducing_points=n_inducing_points if use_sparse else None,
            inducing_point_method=inducing_point_method if use_sparse else None,
            include_prediction_grid=include_prediction_grid,
            prediction_hour=prediction_hour,
            prediction_step_days=prediction_step_days,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(
        df,
        include_hour_info=False,  # Original model doesn't need hour info
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_hour=prediction_hour,
        prediction_hour_step=prediction_hour_step,
        prediction_step_days=prediction_step_days,
    )
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for optimized model
    if use_sparse:
        required_fields = ["use_sparse", "M", "t_inducing"]
        for field in required_fields:
            if field not in stan_data:
                raise ValueError(f"Optimized model requires {field} in stan_data. "
                               f"Make sure prepare_stan_data() includes sparse GP options if needed.")

    # Compile and fit model
    print("Compiling optimized original Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting optimized original model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha': 1.0,
            'rho': 0.2,
            'sigma': 0.1,
            'eta': np.zeros(stan_data['N']),
            'eta_inducing': np.zeros(stan_data['M']),
        }
        return init

    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
    }
    dims = {
        "f": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_cyclic_optimized(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_cyclic_optimized.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    adapt_delta: float = 0.99,
    max_treedepth: int = 12,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit the OPTIMIZED cyclic GP weight model with cov_exp_quad, cov_periodic, and optional sparse GP.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (optimized cyclic version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        adapt_delta: Adapt delta parameter for NUTS (default: 0.99)
        max_treedepth: Maximum tree depth for NUTS (default: 12)
        use_sparse: Whether to use sparse GP approximation (default: False = full GP)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with sparse GP parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            use_sparse=1 if use_sparse else 0,
            n_inducing_points=n_inducing_points if use_sparse else None,
            inducing_point_method=inducing_point_method if use_sparse else None,
            include_prediction_grid=include_prediction_grid,
            prediction_hour=prediction_hour,
            prediction_step_days=prediction_step_days,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(
        df,
        include_hour_info=True,  # Cyclic model needs period_daily
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_hour=prediction_hour,
        prediction_hour_step=prediction_hour_step,
        prediction_step_days=prediction_step_days,
    )
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Verify required fields for optimized cyclic model
    required_fields = ["period_daily"]
    if use_sparse:
        required_fields.extend(["use_sparse", "M", "t_inducing"])
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Optimized cyclic model requires {field} in stan_data. "
                           f"Make sure prepare_stan_data() includes sparse GP options if needed.")

    # Compile and fit model
    print("Compiling optimized cyclic Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting optimized cyclic model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_trend': 1.0,
            'rho_trend': 0.2,
            'alpha_daily': 0.1,
            'rho_daily': 0.5,
            'sigma': 0.1,
            'eta_trend': np.zeros(stan_data['N']),
            'eta_daily': np.zeros(stan_data['N']),
            'eta_inducing': np.zeros(stan_data['M']),
        }
        return init

    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
    }
    dims = {
        "f_trend": ["obs"],
        "f_daily": ["obs"],
        "f_total": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "f_trend_std": ["obs"],
        "f_daily_std": ["obs"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_trend_pred": ["pred"],
            "f_daily_pred": ["pred"],
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred", "f_trend_pred", "f_daily_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def fit_weight_model_flexible_optimized(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_flexible_optimized.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    alpha_prior_sd: float = 1.0,
    rho_prior_shape: float = 5.0,
    rho_prior_scale: float = 1.0,
    sigma_prior_sd: float = 0.5,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit the OPTIMIZED flexible GP weight model with cov_exp_quad, customizable priors, and optional sparse GP.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file (optimized flexible version)
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        alpha_prior_sd: Standard deviation for alpha ~ normal(0, sd)
        rho_prior_shape: Shape parameter for rho ~ inv_gamma(shape, scale)
        rho_prior_scale: Scale parameter for rho ~ inv_gamma(shape, scale)
        sigma_prior_sd: Standard deviation for sigma ~ normal(0, sd)
        use_sparse: Whether to use sparse GP approximation (default: False = full GP)
        n_inducing_points: Number of inducing points for sparse GP (if use_sparse=True)
        inducing_point_method: Method for selecting inducing points ("uniform", "kmeans", "random")
        cache: Whether to cache model results for faster reloading
        force_refit: Force re-fitting even if cached results exist
        include_prediction_grid: Whether to include prediction grid for unobserved days
        prediction_hour: Hour of day (0-24) for prediction points (default 8.0 = 8 AM)
        prediction_hour_step: Step size in hours for multiple prediction hours per day (default None = single hour)
        prediction_step_days: Step size in days for prediction grid (default 1 = daily)

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary with hyperparameters and sparse GP parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup
    cache_dir = None
    cache_config = None
    if cache:
        cache_key, cache_config = _compute_cache_key(
            data_dir=data_dir,
            stan_file=stan_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            alpha_prior_sd=alpha_prior_sd,
            rho_prior_shape=rho_prior_shape,
            rho_prior_scale=rho_prior_scale,
            sigma_prior_sd=sigma_prior_sd,
            use_sparse=1 if use_sparse else 0,
            n_inducing_points=n_inducing_points if use_sparse else None,
            inducing_point_method=inducing_point_method if use_sparse else None,
            include_prediction_grid=include_prediction_grid,
            prediction_hour=prediction_hour,
            prediction_step_days=prediction_step_days,
        )
        cache_dir = output_dir / "cache" / cache_key

        if not force_refit and _cache_exists(cache_dir):
            return _load_cache(cache_dir, stan_file)

    # Load and prepare data
    print("Loading weight data...")
    df = load_weight_data(data_dir)
    stan_data = prepare_stan_data(
        df,
        include_hour_info=False,  # Flexible model doesn't need hour info
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_hour=prediction_hour,
        prediction_hour_step=prediction_hour_step,
        prediction_step_days=prediction_step_days,
    )
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")

    # Add hyperparameters to stan_data
    stan_data.update({
        "alpha_prior_sd": alpha_prior_sd,
        "rho_prior_shape": rho_prior_shape,
        "rho_prior_scale": rho_prior_scale,
        "sigma_prior_sd": sigma_prior_sd,
    })

    # Verify required fields for optimized flexible model
    if use_sparse:
        required_fields = ["use_sparse", "M", "t_inducing"]
        for field in required_fields:
            if field not in stan_data:
                raise ValueError(f"Optimized flexible model requires {field} in stan_data. "
                               f"Make sure prepare_stan_data() includes sparse GP options if needed.")

    # Compile and fit model
    print("Compiling optimized flexible Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting optimized flexible model...")
    fit = model.sample(
        data={k: v for k, v in stan_data.items() if not k.startswith("_")},
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
    }
    dims = {
        "f": ["obs"],
        "y_rep": ["obs"],
        "y": ["obs"],
        "log_lik": ["obs"],
    }
    posterior_predictive_vars = ["y_rep"]

    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })
        posterior_predictive_vars.extend(["y_pred", "f_pred"])

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=posterior_predictive_vars,
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled
    if cache_dir is not None:
        _save_cache(cache_dir, fit, idata, df, stan_data, cache_config)

    return fit, idata, df, stan_data


def compare_models_sigma(
    idata_original,
    idata_cyclic,
    stan_data=None,
    print_summary: bool = True,
) -> dict:
    """Compare sigma estimates between original and cyclic models.

    Args:
        idata_original: InferenceData from original model
        idata_cyclic: InferenceData from cyclic model
        stan_data: Stan data dictionary for back-transformation
        print_summary: Whether to print comparison summary

    Returns:
        Dictionary with comparison metrics
    """
    # Extract sigma estimates
    sigma_orig = idata_original.posterior["sigma"].values.mean()
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.mean()

    # Calculate reduction
    sigma_reduction = sigma_orig - sigma_cyclic
    sigma_reduction_pct = (sigma_reduction / sigma_orig) * 100 if sigma_orig > 0 else 0

    # Extract daily component metrics
    daily_amplitude = idata_cyclic.posterior["daily_amplitude"].values.mean()
    prop_variance_daily = idata_cyclic.posterior["prop_variance_daily"].values.mean()

    # Back-transform if stan_data provided
    if stan_data is not None and "_y_sd" in stan_data:
        y_sd = stan_data["_y_sd"]
        sigma_orig_lbs = sigma_orig * y_sd
        sigma_cyclic_lbs = sigma_cyclic * y_sd
        daily_amplitude_lbs = daily_amplitude * y_sd
    else:
        sigma_orig_lbs = sigma_cyclic_lbs = daily_amplitude_lbs = None

    comparison = {
        "sigma_original": sigma_orig,
        "sigma_cyclic": sigma_cyclic,
        "sigma_reduction": sigma_reduction,
        "sigma_reduction_pct": sigma_reduction_pct,
        "daily_amplitude": daily_amplitude,
        "prop_variance_daily": prop_variance_daily,
        "sigma_original_lbs": sigma_orig_lbs,
        "sigma_cyclic_lbs": sigma_cyclic_lbs,
        "daily_amplitude_lbs": daily_amplitude_lbs,
    }

    if print_summary:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON: Original vs Cyclic")
        print("=" * 60)

        print("\nSigma (measurement error + residual):")
        print(f"  Original model:  {sigma_orig:.4f}")
        print(f"  Cyclic model:    {sigma_cyclic:.4f}")
        print(f"  Reduction:       {sigma_reduction:.4f} ({sigma_reduction_pct:.1f}%)")

        if sigma_orig_lbs is not None:
            print("\nIn pounds (original scale):")
            print(f"  Original sigma: {sigma_orig_lbs:.2f} lbs")
            print(f"  Cyclic sigma:   {sigma_cyclic_lbs:.2f} lbs")
            print(f"  Daily amplitude: {daily_amplitude_lbs:.2f} lbs")

        print("\nDaily component:")
        print(f"  Amplitude: {daily_amplitude:.4f}")
        print(f"  Proportion of variance: {prop_variance_daily:.3f}")

        if sigma_reduction > 0:
            print("\n✓ Sigma reduced by modeling daily cycles")
            if prop_variance_daily > 0.05:
                print(f"✓ Daily component captures meaningful variation ({prop_variance_daily:.1%})")
        else:
            print("\n⚠ Sigma not reduced (daily cycles may be minimal)")

    return comparison

def compare_models_all(
    idata_original,
    idata_cyclic,
    idata_spline=None,
    stan_data=None,
    print_summary: bool = True,
) -> dict:
    """Compare sigma estimates and daily components across all three models.

    Args:
        idata_original: InferenceData from original model
        idata_cyclic: InferenceData from cyclic model
        idata_spline: InferenceData from spline model (optional)
        stan_data: Stan data dictionary for back-transformation
        print_summary: Whether to print comparison summary

    Returns:
        Dictionary with comparison metrics for all available models
    """
    # Extract sigma estimates
    sigma_orig = idata_original.posterior["sigma"].values.mean()
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.mean()

    # Calculate reduction original vs cyclic
    sigma_reduction_oc = sigma_orig - sigma_cyclic
    sigma_reduction_pct_oc = (sigma_reduction_oc / sigma_orig) * 100 if sigma_orig > 0 else 0

    # Extract daily component metrics from cyclic model
    daily_amplitude_cyclic = idata_cyclic.posterior["daily_amplitude"].values.mean()
    prop_variance_daily_cyclic = idata_cyclic.posterior["prop_variance_daily"].values.mean()

    # Initialize spline metrics
    sigma_spline = daily_amplitude_spline = prop_variance_daily_spline = None
    sigma_reduction_cs = sigma_reduction_pct_cs = None
    sigma_reduction_os = sigma_reduction_pct_os = None

    if idata_spline is not None:
        sigma_spline = idata_spline.posterior["sigma"].values.mean()
        daily_amplitude_spline = idata_spline.posterior["daily_amplitude"].values.mean()
        prop_variance_daily_spline = idata_spline.posterior["prop_variance_daily"].values.mean()

        # Calculate reductions involving spline model
        if sigma_cyclic > 0:
            sigma_reduction_cs = sigma_cyclic - sigma_spline
            sigma_reduction_pct_cs = (sigma_reduction_cs / sigma_cyclic) * 100

        sigma_reduction_os = sigma_orig - sigma_spline
        sigma_reduction_pct_os = (sigma_reduction_os / sigma_orig) * 100 if sigma_orig > 0 else 0

    # Back-transform if stan_data provided
    if stan_data is not None and "_y_sd" in stan_data:
        y_sd = stan_data["_y_sd"]
        sigma_orig_lbs = sigma_orig * y_sd
        sigma_cyclic_lbs = sigma_cyclic * y_sd
        daily_amplitude_cyclic_lbs = daily_amplitude_cyclic * y_sd

        if idata_spline is not None:
            sigma_spline_lbs = sigma_spline * y_sd
            daily_amplitude_spline_lbs = daily_amplitude_spline * y_sd
        else:
            sigma_spline_lbs = daily_amplitude_spline_lbs = None
    else:
        sigma_orig_lbs = sigma_cyclic_lbs = daily_amplitude_cyclic_lbs = None
        sigma_spline_lbs = daily_amplitude_spline_lbs = None

    comparison = {
        # Sigma values (standardized scale)
        "sigma_original": sigma_orig,
        "sigma_cyclic": sigma_cyclic,
        "sigma_spline": sigma_spline,

        # Sigma reductions
        "sigma_reduction_original_cyclic": sigma_reduction_oc,
        "sigma_reduction_pct_original_cyclic": sigma_reduction_pct_oc,
        "sigma_reduction_cyclic_spline": sigma_reduction_cs,
        "sigma_reduction_pct_cyclic_spline": sigma_reduction_pct_cs,
        "sigma_reduction_original_spline": sigma_reduction_os,
        "sigma_reduction_pct_original_spline": sigma_reduction_pct_os,

        # Daily component metrics
        "daily_amplitude_cyclic": daily_amplitude_cyclic,
        "prop_variance_daily_cyclic": prop_variance_daily_cyclic,
        "daily_amplitude_spline": daily_amplitude_spline,
        "prop_variance_daily_spline": prop_variance_daily_spline,

        # Original scale (lbs) if available
        "sigma_original_lbs": sigma_orig_lbs,
        "sigma_cyclic_lbs": sigma_cyclic_lbs,
        "sigma_spline_lbs": sigma_spline_lbs,
        "daily_amplitude_cyclic_lbs": daily_amplitude_cyclic_lbs,
        "daily_amplitude_spline_lbs": daily_amplitude_spline_lbs,
    }

    if print_summary:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: Original vs Cyclic vs Spline")
        print("=" * 70)

        print("\nSigma (measurement error + residual):")
        print(f"  Original model:  {sigma_orig:.4f}")
        print(f"  Cyclic model:    {sigma_cyclic:.4f}")
        if sigma_spline is not None:
            print(f"  Spline model:    {sigma_spline:.4f}")

        if sigma_orig_lbs is not None:
            print("\nIn pounds (original scale):")
            print(f"  Original sigma: {sigma_orig_lbs:.2f} lbs")
            print(f"  Cyclic sigma:   {sigma_cyclic_lbs:.2f} lbs")
            if sigma_spline_lbs is not None:
                print(f"  Spline sigma:   {sigma_spline_lbs:.2f} lbs")

        print("\nReduction from original to cyclic:")
        print(f"  Absolute: {sigma_reduction_oc:.4f} ({sigma_reduction_pct_oc:.1f}%)")

        if sigma_spline is not None:
            print("\nReduction from cyclic to spline:")
            if sigma_reduction_cs is not None:
                print(f"  Absolute: {sigma_reduction_cs:.4f} ({sigma_reduction_pct_cs:.1f}%)")
            else:
                print("  Could not calculate (sigma_cyclic <= 0)")

            print("\nTotal reduction from original to spline:")
            print(f"  Absolute: {sigma_reduction_os:.4f} ({sigma_reduction_pct_os:.1f}%)")

        print("\nDaily component amplitude:")
        print(f"  Cyclic model: {daily_amplitude_cyclic:.4f}")
        if daily_amplitude_spline is not None:
            print(f"  Spline model: {daily_amplitude_spline:.4f}")

        print("\nProportion of variance from daily component:")
        print(f"  Cyclic model: {prop_variance_daily_cyclic:.3f}")
        if prop_variance_daily_spline is not None:
            print(f"  Spline model: {prop_variance_daily_spline:.3f}")

        # Interpretation notes
        if sigma_reduction_oc > 0:
            print("\n✓ Sigma reduced by modeling daily cycles (cyclic model)")
            if prop_variance_daily_cyclic > 0.05:
                print(f"✓ Cyclic model captures meaningful daily variation ({prop_variance_daily_cyclic:.1%})")

        if sigma_spline is not None and sigma_reduction_cs is not None and sigma_reduction_cs > 0:
            print("✓ Further reduction with spline model (Fourier harmonics)")
            if prop_variance_daily_spline is not None and prop_variance_daily_spline > prop_variance_daily_cyclic:
                print("✓ Spline model captures more daily variation than cyclic model")
        elif sigma_spline is not None and sigma_reduction_cs is not None and sigma_reduction_cs <= 0:
            print("⚠ Spline model shows minimal improvement over cyclic model")

    return comparison


def compare_models_waic_loo(
    idata_original=None,
    idata_flexible=None,
    idata_cyclic=None,
    idata_spline=None,
    idata_spline_optimized=None,
    idata_original_optimized=None,
    idata_flexible_optimized=None,
    idata_cyclic_optimized=None,
    model_names=None,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Compare models using WAIC and LOO-CV information criteria.

    Computes WAIC (Widely Applicable Information Criterion) and LOO-CV
    (Leave-One-Out Cross-Validation) for each provided model, then calculates
    model weights based on information criteria differences.

    Args:
        idata_original: InferenceData from original GP model (optional)
        idata_flexible: InferenceData from flexible prior model (optional)
        idata_cyclic: InferenceData from cyclic model (optional)
        idata_spline: InferenceData from spline model (optional)
        idata_spline_optimized: InferenceData from optimized spline model (optional)
        idata_original_optimized: InferenceData from optimized original model (optional)
        idata_flexible_optimized: InferenceData from optimized flexible model (optional)
        idata_cyclic_optimized: InferenceData from optimized cyclic model (optional)
        model_names: Custom names for models (list of strings matching order
                     of provided InferenceData objects). If None, uses default
                     names: dictionary keys (original, flexible, cyclic, spline, spline_optimized,
                     original_optimized, flexible_optimized, cyclic_optimized) based on provided models
        print_summary: Whether to print comparison table and diagnostics

    Returns:
        pandas DataFrame with columns:
        - waic: WAIC value (lower is better)
        - waic_se: Standard error of WAIC
        - p_waic: Effective number of parameters (WAIC)
        - loo: LOO value (lower is better)
        - loo_se: Standard error of LOO
        - p_loo: Effective number of parameters (LOO)
        - waic_weight: Model weight based on WAIC (higher is better)
        - loo_weight: Model weight based on LOO (higher is better)

    Raises:
        ValueError: If no InferenceData objects provided
        RuntimeError: If WAIC/LOO computation fails (e.g., high Pareto k)
    """
    # Collect non-None InferenceData objects
    idata_dict = {}
    if idata_original is not None:
        idata_dict["original"] = idata_original
    if idata_flexible is not None:
        idata_dict["flexible"] = idata_flexible
    if idata_cyclic is not None:
        idata_dict["cyclic"] = idata_cyclic
    if idata_spline is not None:
        idata_dict["spline"] = idata_spline
    if idata_spline_optimized is not None:
        idata_dict["spline_optimized"] = idata_spline_optimized
    if idata_original_optimized is not None:
        idata_dict["original_optimized"] = idata_original_optimized
    if idata_flexible_optimized is not None:
        idata_dict["flexible_optimized"] = idata_flexible_optimized
    if idata_cyclic_optimized is not None:
        idata_dict["cyclic_optimized"] = idata_cyclic_optimized

    if not idata_dict:
        raise ValueError("At least one InferenceData object must be provided")

    # Use custom model names if provided
    if model_names is not None:
        if len(model_names) != len(idata_dict):
            raise ValueError(
                f"model_names length ({len(model_names)}) must match "
                f"number of models ({len(idata_dict)})"
            )
        # Create new dictionary with custom names
        custom_dict = {}
        for (old_name, idata), new_name in zip(idata_dict.items(), model_names):
            custom_dict[new_name] = idata
        idata_dict = custom_dict

    # Compute WAIC and LOO for each model
    waic_results = {}
    loo_results = {}
    warnings = []

    for name, idata in idata_dict.items():
        try:
            waic = az.waic(idata)
            waic_results[name] = waic
        except Exception as e:
            warnings.append(f"WAIC computation failed for {name}: {e}")
            waic_results[name] = None

        try:
            loo = az.loo(idata)
            loo_results[name] = loo
        except Exception as e:
            warnings.append(f"LOO computation failed for {name}: {e}")
            loo_results[name] = None

    # Create comparison DataFrame
    rows = []
    for name in idata_dict.keys():
        waic = waic_results.get(name)
        loo = loo_results.get(name)

        row = {
            "model": name,
            "waic": -2 * waic.elpd_waic if waic is not None else None,
            "waic_se": 2 * waic.se if waic is not None else None,
            "p_waic": waic.p_waic if waic is not None else None,
            "loo": -2 * loo.elpd_loo if loo is not None else None,
            "loo_se": 2 * loo.se if loo is not None else None,
            "p_loo": loo.p_loo if loo is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    # Compute model weights (Akaike weights for WAIC, LOO weights)
    # Weight = exp(-0.5 * Δ) / sum(exp(-0.5 * Δ))
    def compute_weights(values):
        """Compute Akaike weights from information criterion values."""
        # Handle missing values
        valid_mask = pd.notna(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            return pd.Series([None] * len(values), index=values.index)

        # Compute differences from minimum
        min_val = valid_values.min()
        deltas = valid_values - min_val
        # Compute weights
        weights = np.exp(-0.5 * deltas)
        weights = weights / weights.sum()

        # Create full series with NaN for missing
        full_weights = pd.Series([None] * len(values), index=values.index)
        full_weights[valid_mask] = weights
        return full_weights

    if df["waic"].notna().any():
        df["waic_weight"] = compute_weights(df["waic"])
    if df["loo"].notna().any():
        df["loo_weight"] = compute_weights(df["loo"])

    # Print summary if requested
    if print_summary:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON: WAIC and LOO-CV")
        print("=" * 70)

        # Print warnings if any
        if warnings:
            print("\n⚠ WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")

        # Print comparison table
        print("\nInformation Criteria (lower is better):")
        print(df.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"))

        # Print interpretation
        print("\nINTERPRETATION:")
        print("  • WAIC: Widely Applicable Information Criterion (fully Bayesian)")
        print("  • LOO-CV: Leave-One-Out Cross-Validation (approximate)")
        print("  • p_waic/p_loo: Effective number of parameters")
        print("  • Weight: Probability that model is best given data")

        # Highlight best model by each criterion
        for criterion in ["waic", "loo"]:
            weight_col = f"{criterion}_weight"
            if weight_col in df.columns and df[weight_col].notna().any():
                best_idx = df[weight_col].idxmax()
                best_weight = df.loc[best_idx, weight_col]
                if pd.notna(best_weight):
                    print(f"\n  Best model by {criterion.upper()}: {best_idx} "
                          f"(weight={best_weight:.3f})")

        # Check for high Pareto k values (LOO diagnostics)
        print("\nLOO DIAGNOSTICS (Pareto k estimates):")
        print("  Thresholds: k > 0.57 (warning), k > 0.7 (high), k > 1.0 (very high)")
        for name, loo in loo_results.items():
            if loo is not None and hasattr(loo, "pareto_k"):
                k_values = loo.pareto_k
                if k_values is not None:
                    # Convert to numpy array if xarray DataArray
                    if hasattr(k_values, 'values'):
                        k_values = k_values.values
                    n_warning = np.sum(k_values > 0.57)
                    n_high = np.sum(k_values > 0.7)
                    n_very_high = np.sum(k_values > 1.0)
                    if n_warning > 0:
                        print(f"  {name}: {n_warning} observations with k > 0.57 "
                              f"({n_high} with k > 0.7, {n_very_high} with k > 1.0)")
                        # Identify indices of observations with k > 0.7 for further inspection
                        if n_high > 0:
                            high_indices = np.where(k_values > 0.7)[0]
                            # Print first few indices
                            if len(high_indices) <= 10:
                                indices_str = ', '.join(map(str, high_indices))
                                print(f"    Observations with k > 0.7: {indices_str}")
                            else:
                                print(f"    Observations with k > 0.7: first 10 of {len(high_indices)}: {', '.join(map(str, high_indices[:10]))}")
                        # Provide recommendations
                        print("    Recommendations: Check data quality at high-k observations; consider robust modeling approaches.")
                    else:
                        print(f"  {name}: All Pareto k < 0.57 (good)")

    return df


def get_pareto_k_diagnostics(idata, threshold_warning=0.57, threshold_high=0.7, threshold_very_high=1.0):
    """Extract Pareto k diagnostics from LOO results for a single model.

    Args:
        idata: ArviZ InferenceData object with LOO results
        threshold_warning: Pareto k threshold for warning (default 0.57)
        threshold_high: Pareto k threshold for high (default 0.7)
        threshold_very_high: Pareto k threshold for very high (default 1.0)

    Returns:
        dict with keys:
        - 'k_values': array of Pareto k estimates
        - 'n_warning': number of observations with k > threshold_warning
        - 'n_high': number of observations with k > threshold_high
        - 'n_very_high': number of observations with k > threshold_very_high
        - 'warning_indices': indices where k > threshold_warning
        - 'high_indices': indices where k > threshold_high
        - 'very_high_indices': indices where k > threshold_very_high
        - 'max_k': maximum Pareto k value
        - 'mean_k': mean Pareto k value
        - 'summary': textual summary
    """
    import numpy as np

    # Compute LOO if not already present
    if not hasattr(idata, 'loo') or idata.loo is None:
        import arviz as az
        loo_result = az.loo(idata)
    else:
        loo_result = idata.loo

    if not hasattr(loo_result, 'pareto_k') or loo_result.pareto_k is None:
        raise ValueError("LOO result does not have Pareto k estimates")

    k_values = loo_result.pareto_k
    if hasattr(k_values, 'values'):
        k_values = k_values.values

    # Compute diagnostics
    warning_mask = k_values > threshold_warning
    high_mask = k_values > threshold_high
    very_high_mask = k_values > threshold_very_high

    n_warning = np.sum(warning_mask)
    n_high = np.sum(high_mask)
    n_very_high = np.sum(very_high_mask)

    warning_indices = np.where(warning_mask)[0]
    high_indices = np.where(high_mask)[0]
    very_high_indices = np.where(very_high_mask)[0]

    max_k = np.max(k_values) if len(k_values) > 0 else 0.0
    mean_k = np.mean(k_values) if len(k_values) > 0 else 0.0

    summary = (
        f"Pareto k diagnostics: {len(k_values)} observations, "
        f"{n_warning} with k > {threshold_warning}, "
        f"{n_high} with k > {threshold_high}, "
        f"{n_very_high} with k > {threshold_very_high}. "
        f"Max k = {max_k:.3f}, mean k = {mean_k:.3f}."
    )

    return {
        'k_values': k_values,
        'n_warning': n_warning,
        'n_high': n_high,
        'n_very_high': n_very_high,
        'warning_indices': warning_indices,
        'high_indices': high_indices,
        'very_high_indices': very_high_indices,
        'max_k': max_k,
        'mean_k': mean_k,
        'summary': summary,
    }


def get_high_k_observations(idata, df, threshold_high=0.7):
    """Get DataFrame rows corresponding to observations with high Pareto k values.

    Args:
        idata: ArviZ InferenceData object
        df: Original DataFrame with observations (must match order in idata)
        threshold_high: Pareto k threshold for "high" (default 0.7)

    Returns:
        DataFrame containing rows from df where Pareto k > threshold_high,
        with additional column 'pareto_k' containing the k value.

    Raises:
        ValueError: If df length doesn't match number of observations in idata
    """

    # Get Pareto k diagnostics
    diag = get_pareto_k_diagnostics(idata, threshold_warning=threshold_high)

    if len(df) != len(diag['k_values']):
        raise ValueError(
            f"DataFrame length ({len(df)}) doesn't match number of observations "
            f"in InferenceData ({len(diag['k_values'])})"
        )

    # Get indices of high-k observations
    high_indices = diag['high_indices']

    if len(high_indices) == 0:
        return pd.DataFrame()  # Empty DataFrame

    # Extract corresponding rows from df
    high_obs = df.iloc[high_indices].copy()

    # Add Pareto k values as a column
    high_obs['pareto_k'] = diag['k_values'][high_indices]

    # Sort by Pareto k (highest first)
    high_obs = high_obs.sort_values('pareto_k', ascending=False)

    return high_obs


def check_loo_reliability(idata, threshold_warning=0.57, threshold_high=0.7):
    """Check if LOO-CV results are reliable based on Pareto k diagnostics.

    Args:
        idata: ArviZ InferenceData object
        threshold_warning: Warning threshold for Pareto k (default 0.57)
        threshold_high: High threshold for Pareto k (default 0.7)

    Returns:
        dict with keys:
        - 'is_reliable': bool indicating if LOO is reliable
        - 'reliability_level': str ('good', 'warning', 'high', 'very_high')
        - 'n_warning', 'n_high', 'n_very_high': counts
        - 'max_k', 'mean_k': statistics
        - 'recommendations': list of recommendation strings
    """
    diag = get_pareto_k_diagnostics(
        idata,
        threshold_warning=threshold_warning,
        threshold_high=threshold_high,
        threshold_very_high=1.0
    )

    n_obs = len(diag['k_values'])
    n_warning = diag['n_warning']
    n_high = diag['n_high']
    n_very_high = diag['n_very_high']
    max_k = diag['max_k']
    mean_k = diag['mean_k']

    # Determine reliability level
    reliability_level = 'good'
    if n_very_high > 0.1 * n_obs:  # >10% with k > 1.0
        reliability_level = 'very_high'
    elif n_high > 0.1 * n_obs:  # >10% with k > 0.7
        reliability_level = 'high'
    elif n_warning > 0.1 * n_obs:  # >10% with k > 0.57
        reliability_level = 'warning'

    # Default to reliable unless many high-k observations
    is_reliable = reliability_level in ['good', 'warning']

    # Generate recommendations based on severity
    recommendations = []

    if reliability_level == 'good':
        recommendations.append("LOO-CV is reliable for model comparison.")
    elif reliability_level == 'warning':
        recommendations.append(f"LOO-CV has warning: {n_warning}/{n_obs} observations with k > {threshold_warning}.")
        recommendations.append("Consider using WAIC instead of LOO for model comparison.")
        recommendations.append("Check data quality at high-k observations using get_high_k_observations().")
    elif reliability_level == 'high':
        recommendations.append(f"LOO-CV is unreliable: {n_high}/{n_obs} observations with k > {threshold_high}.")
        recommendations.append("Use WAIC for model comparison instead of LOO.")
        recommendations.append("Examine high-k observations for data quality issues.")
        recommendations.append("Consider robust modeling approaches (e.g., Student-t likelihood).")
    elif reliability_level == 'very_high':
        recommendations.append(f"LOO-CV is very unreliable: {n_very_high}/{n_obs} observations with k > 1.0.")
        recommendations.append("LOO-CV should not be used for model comparison.")
        recommendations.append("Use WAIC or consider model re-specification.")
        recommendations.append("Check if model has converged properly (chains, iterations).")

    # Additional general recommendations
    if max_k > 1.0:
        recommendations.append(f"Maximum Pareto k = {max_k:.3f} indicates very influential observations.")
    if mean_k > 0.5:
        recommendations.append(f"Mean Pareto k = {mean_k:.3f} suggests many influential observations.")

    return {
        'is_reliable': is_reliable,
        'reliability_level': reliability_level,
        'n_warning': n_warning,
        'n_high': n_high,
        'n_very_high': n_very_high,
        'max_k': max_k,
        'mean_k': mean_k,
        'recommendations': recommendations,
    }


def compare_models_with_fallback(idata_dict, df, print_summary=True):
    """Compare models using LOO-CV with fallback to WAIC when LOO is unreliable.

    This function automatically checks LOO reliability for each model and uses
    WAIC for models where LOO is unreliable.

    Args:
        idata_dict: Dictionary of {model_name: InferenceData} objects
        df: Original DataFrame (used for checking high-k observations)
        print_summary: Whether to print comparison summary

    Returns:
        pandas DataFrame with comparison results
    """
    import pandas as pd

    # Check LOO reliability for each model
    reliability_info = {}
    for name, idata in idata_dict.items():
        reliability_info[name] = check_loo_reliability(idata)

    # Determine which models can use LOO
    models_for_loo = [name for name, info in reliability_info.items()
                      if info['is_reliable']]
    models_for_waic = [name for name in idata_dict.keys()
                       if name not in models_for_loo]

    if print_summary:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON WITH RELIABILITY CHECK")
        print("=" * 70)

        print("\nLOO-CV Reliability Assessment:")
        for name, info in reliability_info.items():
            level = info['reliability_level']
            color = {
                'good': '✓',
                'warning': '⚠',
                'high': '⚠',
                'very_high': '✗'
            }.get(level, '?')
            print(f"  {color} {name}: {level} "
                  f"(k > 0.57: {info['n_warning']}, "
                  f"k > 0.7: {info['n_high']}, "
                  f"k > 1.0: {info['n_very_high']})")

        if models_for_waic:
            print(f"\nUsing WAIC for models: {', '.join(models_for_waic)} "
                  f"(LOO-CV unreliable)")
        if models_for_loo:
            print(f"Using LOO-CV for models: {', '.join(models_for_loo)}")

    # Prepare models for comparison
    # We need to handle the case where some models use LOO and some use WAIC
    # For simplicity, we'll use WAIC for all models if any model has unreliable LOO
    # This ensures consistent comparison
    use_waic_for_all = len(models_for_waic) > 0

    if use_waic_for_all and print_summary:
        print("\nNote: Using WAIC for all models for consistent comparison "
              "(some models have unreliable LOO-CV)")

    # Run comparison using WAIC (more reliable when LOO has issues)
    # We'll modify compare_models_waic_loo to use only WAIC
    # For now, just use compare_models_waic_loo with default behavior
    # but print additional warnings

    # Call the original compare_models_waic_loo
    # Extract InferenceData objects in order
    idata_list = []
    model_names_list = []
    for name, idata in idata_dict.items():
        idata_list.append(idata)
        model_names_list.append(name)

    # We need to pass the InferenceData objects as separate arguments
    # This is a bit hacky but works with the existing function signature
    # Since we don't know which models are which type, we'll use the
    # original function and rely on its warnings

    # For simplicity, just call compare_models_waic_loo with print_summary=False
    # and add our own summary

    # We'll create a modified version that emphasizes WAIC when LOO is unreliable
    if print_summary:
        print("\n" + "-" * 70)
        print("MODEL COMPARISON RESULTS")
        print("-" * 70)

        # Print recommendations for each model
        for name, info in reliability_info.items():
            if not info['is_reliable']:
                print(f"\nRecommendations for {name} model:")
                for rec in info['recommendations']:
                    print(f"  • {rec}")

    # Use the standard comparison function
    # Note: This requires all InferenceData objects to be passed as named arguments
    # We'll use a simpler approach: compute WAIC for each and create comparison table

    rows = []
    for name, idata in idata_dict.items():
        try:
            waic = az.waic(idata)
            waic_elpd = -2 * waic.elpd_waic
            waic_se = 2 * waic.se
            p_waic = waic.p_waic
        except Exception as e:
            if print_summary:
                print(f"⚠ WAIC computation failed for {name}: {e}")
            waic_elpd = waic_se = p_waic = None

        # Try LOO if reliable
        loo_elpd = loo_se = p_loo = None
        if reliability_info[name]['is_reliable']:
            try:
                loo = az.loo(idata)
                loo_elpd = -2 * loo.elpd_loo
                loo_se = 2 * loo.se
                p_loo = loo.p_loo
            except Exception as e:
                if print_summary:
                    print(f"⚠ LOO computation failed for {name}: {e}")

        rows.append({
            'model': name,
            'waic': waic_elpd,
            'waic_se': waic_se,
            'p_waic': p_waic,
            'loo': loo_elpd,
            'loo_se': loo_se,
            'p_loo': p_loo,
            'loo_reliable': reliability_info[name]['is_reliable'],
        })

    df_compare = pd.DataFrame(rows).set_index('model')

    # Compute weights based on WAIC (more reliable when LOO has issues)
    if df_compare['waic'].notna().any():
        # Use WAIC for weights
        valid_mask = pd.notna(df_compare['waic'])
        valid_values = df_compare['waic'][valid_mask]
        if len(valid_values) > 0:
            min_val = valid_values.min()
            deltas = valid_values - min_val
            weights = np.exp(-0.5 * deltas)
            weights = weights / weights.sum()

            df_compare['weight'] = pd.Series([None] * len(df_compare), index=df_compare.index)
            df_compare.loc[valid_mask, 'weight'] = weights
        else:
            df_compare['weight'] = None
    else:
        df_compare['weight'] = None

    if print_summary:
        print(f"\nComparison ({'WAIC' if use_waic_for_all else 'LOO-CV'}):")
        print(df_compare.to_string(float_format=lambda x: f"{x:.1f}" if pd.notna(x) else "NaN"))

        if 'weight' in df_compare.columns and df_compare['weight'].notna().any():
            best_idx = df_compare['weight'].idxmax()
            best_weight = df_compare.loc[best_idx, 'weight']
            if pd.notna(best_weight):
                print(f"\nBest model: {best_idx} (weight={best_weight:.3f})")

    return df_compare


def fit_bivariate_model(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp_bivariate.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    weight_var: str = "weight_mean",
    other_var: str = "resting_heart_rate",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit bivariate GP model for weight and another variable.

    Args:
        data_dir: Path to data directory.
        stan_file: Path to Stan model file (bivariate version).
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        weight_var: Weight variable column name.
        other_var: Other variable column name.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Merged DataFrame with weight and other variable
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - reuse existing cache logic but need to extend)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Load and prepare data
    print("Loading weight and daily metrics...")
    df = merge_weight_with_daily_metrics(data_dir)
    print(f"  {len(df)} days with both weight and daily metrics")

    stan_data = prepare_bivariate_stan_data(
        df,
        weight_var=weight_var,
        other_var=other_var,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Verify required fields
    required_fields = ["y_weight", "y_other", "t", "use_sparse", "M", "t_inducing"]
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"Bivariate model requires {field} in stan_data.")

    # Compile and fit model
    print("Compiling bivariate Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting bivariate model...")
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha': 1.0,
            'rho': 0.2,
            'sigma_B': [0.5, 0.5],
            'L_corr': [[1.0, 0.0], [0.0, 1.0]],  # identity Cholesky factor
            'sigma': [0.1, 0.1],
            'nu': [10.0, 10.0],
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith("_")}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
        "output": ["weight", "other"],
    }
    dims = {
        "f_trend": ["obs", "output"],
        "y_weight": ["obs"],
        "y_other": ["obs"],
        "y_weight_rep": ["obs"],
        "y_other_rep": ["obs"],
        "log_lik": ["obs", "output"],
    }
    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred", "output"],
            "y_pred": ["pred", "output"],
        })

    # Need to extract posterior variables appropriately
    # For simplicity, we'll create InferenceData with default mapping
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_weight_rep", "y_other_rep"],
        observed_data={"y_weight": stan_data["y_weight"], "y_other": stan_data["y_other"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    # Save to cache if caching is enabled (TODO)
    if cache_dir is not None:
        pass

    return fit, idata, df, stan_data


def fit_gp_simple(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    value_col: str = "value",
    stan_file: Path | str = "stan/gp_simple.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
    t_pred_days: Optional[np.ndarray] = None,
) -> tuple:
    """Fit simple GP model to any time series.

    Args:
        df: DataFrame with timestamp and value columns.
        time_col: Name of timestamp column.
        value_col: Name of value column.
        stan_file: Path to Stan model file (simple GP version).
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.
        t_pred_days: Optional array of absolute days for prediction grid (overrides include_prediction_grid and prediction_step_days).

    Returns:
        Tuple of (fit, idata, df_prepared, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df_prepared: Prepared DataFrame with scaled time and values
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified)
    if cache:
        # Skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Prepare data
    print(f"Preparing data for {value_col}...")
    df_prepared = df.copy()
    df_prepared = df_prepared.sort_values(time_col).reset_index(drop=True)

    # Create days since start
    t_min = df_prepared[time_col].min()
    df_prepared["days_since_start"] = (df_prepared[time_col] - t_min).dt.days
    t = df_prepared["days_since_start"].values
    t_scaled = t / t.max() if t.max() > 0 else t

    # Extract and scale values
    y = df_prepared[value_col].values
    if np.any(pd.isna(y)):
        raise ValueError(f"Value column '{value_col}' contains missing values")

    y_mean = y.mean()
    y_sd = y.std()
    y_scaled = (y - y_mean) / y_sd

    # Prepare Stan data dictionary
    stan_data = {
        "N": len(df_prepared),
        "t": t_scaled,
        "y": y_scaled,
        "_y_mean": y_mean,
        "_y_sd": y_sd,
        "_t_max": t.max(),
        "_dates": df_prepared[time_col].dt.strftime("%Y-%m-%d").tolist(),
    }

    # Sparse GP configuration
    if use_sparse:
        if n_inducing_points <= 0:
            raise ValueError("n_inducing_points must be positive")
        if n_inducing_points > len(t_scaled):
            n_inducing_points = len(t_scaled)
            print(f"Warning: n_inducing_points reduced to N={n_inducing_points}")

        # Select inducing points
        if inducing_point_method == "uniform":
            indices = np.linspace(0, len(t_scaled) - 1, n_inducing_points, dtype=int)
            t_inducing = t_scaled[indices]
        elif inducing_point_method == "kmeans":
            from sklearn.cluster import KMeans
            X = t_scaled.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_inducing_points, random_state=42, n_init=10)
            kmeans.fit(X)
            t_inducing = kmeans.cluster_centers_.flatten()
            t_inducing = np.sort(t_inducing)
        elif inducing_point_method == "random":
            indices = np.random.choice(len(t_scaled), n_inducing_points, replace=False)
            indices = np.sort(indices)
            t_inducing = t_scaled[indices]
        else:
            raise ValueError(f"Unknown inducing_point_method: {inducing_point_method}")

        stan_data.update({
            "use_sparse": 1,
            "M": n_inducing_points,
            "t_inducing": t_inducing.tolist(),
        })
    else:
        stan_data.update({
            "use_sparse": 0,
            "M": len(t_scaled),
            "t_inducing": t_scaled.tolist(),
        })

    # Prediction grid
    t_max = t.max()
    if t_pred_days is not None:
        # Use provided prediction grid
        t_pred_days_arr = np.asarray(t_pred_days)
        t_pred = t_pred_days_arr / t_max
        stan_data.update({
            "N_pred": len(t_pred),
            "t_pred": t_pred.tolist(),
            "_t_pred_days": t_pred_days_arr.tolist(),
        })
    elif include_prediction_grid:
        t_min = t.min()
        t_pred_days = np.arange(t_min, t_max + prediction_step_days, prediction_step_days)
        t_pred_days = t_pred_days[t_pred_days <= t_max]
        t_pred = t_pred_days / t_max

        stan_data.update({
            "N_pred": len(t_pred),
            "t_pred": t_pred.tolist(),
            "_t_pred_days": t_pred_days.tolist(),
        })
    else:
        stan_data.update({
            "N_pred": 0,
            "t_pred": [],
        })

    # Compile and fit model
    print("Compiling simple GP model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting simple GP model...")
    def generate_init():
        return {
            'alpha': 1.0,
            'rho': 0.2,
            'sigma': 0.1,
            'nu': 10.0,
            'eta_inducing_raw': np.zeros(stan_data['M']),
        }

    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith("_")}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs": np.arange(stan_data["N"]),
    }
    dims = {
        "f_trend": ["obs"],
        "y": ["obs"],
        "y_rep": ["obs"],
        "log_lik": ["obs"],
    }
    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred"],
            "y_pred": ["pred"],
        })

    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_rep"],
        observed_data={"y": stan_data["y"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik",
    )

    return fit, idata, df_prepared, stan_data



def fit_bivariate_model_mismatched(
    df_weight: pd.DataFrame,
    df_other: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    other_time_col: str = "timestamp",
    other_value_col: str = "value",
    stan_file: Path | str = "stan/weight_gp_bivariate_mismatched.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    '''Fit bivariate GP model for weight and another variable with mismatched observation times.

    Args:
        df_weight: DataFrame with weight observations.
        df_other: DataFrame with other variable observations.
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        other_time_col: Name of timestamp column in df_other.
        other_value_col: Name of value column in df_other.
        stan_file: Path to Stan model file (mismatched bivariate version).
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - stan_data: Stan data dictionary
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - reuse existing cache logic but need to extend)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Prepare data using mismatched data preparation
    print('Preparing mismatched bivariate Stan data...')
    stan_data = prepare_bivariate_stan_data_mismatched(
        df_weight=df_weight,
        df_other=df_other,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        other_time_col=other_time_col,
        other_value_col=other_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Verify required fields
    required_fields = ['N_weight', 'N_other', 't_weight', 't_other', 'y_weight', 'y_other',
                       'use_sparse', 'M', 't_inducing']
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f'Mismatched bivariate model requires {field} in stan_data.')

    # Compile and fit model
    print('Compiling mismatched bivariate Stan model...')
    model = CmdStanModel(stan_file=stan_file)

    print('Fitting mismatched bivariate model...')
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha': 1.0,
            'rho': 0.2,
            'sigma_B': [0.5, 0.5],
            'L_corr': [[1.0, 0.0], [0.0, 1.0]],  # identity Cholesky factor
            'sigma': [0.1, 0.1],
            'nu': [10.0, 10.0],
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print('Creating ArviZ InferenceData...')
    N_pred = stan_data.get('N_pred', 0)
    coords = {
        'obs_weight': np.arange(stan_data['N_weight']),
        'obs_other': np.arange(stan_data['N_other']),
        'output': ['weight', 'other'],
    }
    dims = {
        'f_weight': ['obs_weight'],
        'f_other': ['obs_other'],
        'y_weight': ['obs_weight'],
        'y_other': ['obs_other'],
        'y_weight_rep': ['obs_weight'],
        'y_other_rep': ['obs_other'],
        'log_lik_weight': ['obs_weight'],
        'log_lik_other': ['obs_other'],
    }
    if N_pred > 0:
        coords['pred'] = np.arange(N_pred)
        dims.update({
            'f_pred': ['pred', 'output'],
            'y_pred': ['pred', 'output'],
        })

    # Need to extract posterior variables appropriately
    # For simplicity, we'll create InferenceData with default mapping
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=['y_weight_rep', 'y_other_rep'],
        observed_data={'y_weight': stan_data['y_weight'], 'y_other': stan_data['y_other']},
        coords=coords,
        dims=dims,
        log_likelihood=None,
    )

    # Save to cache if caching is enabled (TODO)
    if cache_dir is not None:
        pass

    return fit, idata, stan_data


def fit_crosslagged_model(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    lag_days: float = 2.0,
    stan_file: Path | str = "stan/weight_gp_crosslagged.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit cross-lagged GP model for weight depending on lagged workouts.

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        lag_days: Lag in days (workouts at time t affect weight at time t+lag).
        stan_file: Path to cross-lagged Stan model file.
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - reuse existing cache logic)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Prepare data using cross-lagged data preparation
    print('Preparing cross-lagged Stan data...')
    from src.data.align import prepare_crosslagged_stan_data
    stan_data = prepare_crosslagged_stan_data(
        df_weight=df_weight,
        df_workout=df_workout,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        workout_time_col=workout_time_col,
        workout_value_col=workout_value_col,
        lag_days=lag_days,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Verify required fields (including lag_scaled)
    required_fields = ['N_weight', 'N_workout', 't_weight', 't_workout',
                       'y_weight', 'y_workout', 'lag_scaled',
                       'use_sparse', 'M', 't_inducing']

    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f'Cross-lagged model requires {field} in stan_data.')

    # Compile and fit model
    print('Compiling cross-lagged Stan model...')
    model = CmdStanModel(stan_file=stan_file)

    print('Fitting cross-lagged model...')
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_weight': 1.0,
            'alpha_workout': 1.0,
            'rho_weight': 0.2,
            'rho_workout': 0.2,
            'beta': 0.0,  # Start with no effect
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print('Creating ArviZ InferenceData...')
    N_pred = stan_data.get('N_pred', 0)
    coords = {
        'obs_weight': np.arange(stan_data['N_weight']),
        'obs_workout': np.arange(stan_data['N_workout']),
        'output': ['weight', 'workout'],
    }
    dims = {
        'f_weight': ['obs_weight'],
        'f_workout': ['obs_workout'],
        'y_weight': ['obs_weight'],
        'y_workout': ['obs_workout'],
        'y_weight_rep': ['obs_weight'],
        'y_workout_rep': ['obs_workout'],
        'log_lik_weight': ['obs_weight'],
        'log_lik_workout': ['obs_workout'],
    }
    if N_pred > 0:
        coords['pred'] = np.arange(N_pred)
        dims.update({
            'f_pred': ['pred', 'output'],
            'y_pred': ['pred', 'output'],
        })

    # Need to handle the fact that Stan outputs may use different names
    # We'll use default mapping and adjust as needed
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=['y_weight_rep', 'y_workout_rep'],
        observed_data={
            'y_weight': stan_data['y_weight'],
            'y_workout': stan_data['y_workout']
        },
        coords=coords,
        dims=dims,
        log_likelihood=None,
    )

    # Convert beta to original units for interpretation
    # β_original = β_scaled * (weight_sd / workout_sd)
    if '_y_weight_sd' in stan_data and '_y_workout_sd' in stan_data:
        weight_sd = stan_data['_y_weight_sd']
        workout_sd = stan_data['_y_workout_sd']
        if workout_sd > 0:
            beta_scaled = idata.posterior['beta'] * (weight_sd / workout_sd)
            # Add as new variable
            idata.posterior['beta_original_units'] = beta_scaled

    # Save to cache if caching is enabled (TODO)
    if cache_dir is not None:
        pass

    return fit, idata, stan_data


def fit_crosslagged_model_estimated(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    stan_file: Path | str = "stan/weight_gp_crosslagged_estimated.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit cross-lagged GP model with estimated lag parameter.

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        stan_file: Path to cross-lagged Stan model file with estimated lag.
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - reuse existing cache logic)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Prepare data using cross-lagged data preparation for estimated lag
    print('Preparing cross-lagged Stan data (estimated lag)...')
    from src.data.align import prepare_crosslagged_stan_data_estimated
    stan_data = prepare_crosslagged_stan_data_estimated(
        df_weight=df_weight,
        df_workout=df_workout,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        workout_time_col=workout_time_col,
        workout_value_col=workout_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Verify required fields (including t_max, excluding lag_scaled)
    required_fields = ['N_weight', 'N_workout', 't_weight', 't_workout',
                       'y_weight', 'y_workout', 't_max',
                       'use_sparse', 'M', 't_inducing']

    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f'Cross-lagged estimated model requires {field} in stan_data.')

    # Compile and fit model
    print('Compiling cross-lagged Stan model (estimated lag)...')
    model = CmdStanModel(stan_file=stan_file)

    print('Fitting cross-lagged model (estimated lag)...')
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_weight': 1.0,
            'alpha_workout': 1.0,
            'rho_weight': 0.2,
            'rho_workout': 0.2,
            'beta': 0.0,  # Start with no effect
            'lag_days': 2.0,  # Start with 2 days lag (prior mean 3)
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print('Creating ArviZ InferenceData...')
    N_pred = stan_data.get('N_pred', 0)
    coords = {
        'obs_weight': np.arange(stan_data['N_weight']),
        'obs_workout': np.arange(stan_data['N_workout']),
        'output': ['weight', 'workout'],
    }
    dims = {
        'f_weight': ['obs_weight'],
        'f_workout': ['obs_workout'],
        'y_weight': ['obs_weight'],
        'y_workout': ['obs_workout'],
        'y_weight_rep': ['obs_weight'],
        'y_workout_rep': ['obs_workout'],
        'log_lik_weight': ['obs_weight'],
        'log_lik_workout': ['obs_workout'],
    }
    if N_pred > 0:
        coords['pred'] = np.arange(N_pred)
        dims.update({
            'f_pred': ['pred', 'output'],
            'y_pred': ['pred', 'output'],
        })

    # Need to handle the fact that Stan outputs may use different names
    # We'll use default mapping and adjust as needed
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=['y_weight_rep', 'y_workout_rep'],
        observed_data={
            'y_weight': stan_data['y_weight'],
            'y_workout': stan_data['y_workout']
        },
        coords=coords,
        dims=dims,
        log_likelihood=None,
    )

    # Convert beta to original units for interpretation
    # β_original = β_scaled * (weight_sd / workout_sd)
    if '_y_weight_sd' in stan_data and '_y_workout_sd' in stan_data:
        weight_sd = stan_data['_y_weight_sd']
        workout_sd = stan_data['_y_workout_sd']
        if workout_sd > 0:
            beta_scaled = idata.posterior['beta'] * (weight_sd / workout_sd)
            # Add as new variable
            idata.posterior['beta_original_units'] = beta_scaled

    # Save to cache if caching is enabled (TODO)
    if cache_dir is not None:
        pass

    return fit, idata, stan_data


def fit_crosslagged_model_cumulative(
    df_weight: pd.DataFrame,
    df_workout: pd.DataFrame,
    lag_days_list: list[float] = [1.0, 2.0, 3.0],
    weight_time_col: str = "timestamp",
    weight_value_col: str = "weight_lbs",
    workout_time_col: str = "date",
    workout_value_col: str = "workout_count",
    stan_file: Path | str = "stan/weight_gp_crosslagged_cumulative.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit cross-lagged GP model with cumulative lag effects.

    Args:
        df_weight: DataFrame with weight observations.
        df_workout: DataFrame with workout metric observations (daily aggregates).
        lag_days_list: List of lag values in days (e.g., [1, 2, 3] for 1,2,3 day lags).
        weight_time_col: Name of timestamp column in df_weight.
        weight_value_col: Name of value column in df_weight.
        workout_time_col: Name of timestamp column in df_workout.
        workout_value_col: Name of value column in df_workout.
        stan_file: Path to cross-lagged Stan model file with cumulative lags.
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - reuse existing cache logic)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Prepare data using cross-lagged data preparation for cumulative lags
    print('Preparing cross-lagged Stan data (cumulative lags)...')
    from src.data.align import prepare_crosslagged_stan_data_cumulative
    stan_data = prepare_crosslagged_stan_data_cumulative(
        df_weight=df_weight,
        df_workout=df_workout,
        lag_days_list=lag_days_list,
        weight_time_col=weight_time_col,
        weight_value_col=weight_value_col,
        workout_time_col=workout_time_col,
        workout_value_col=workout_value_col,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
        inducing_point_method=inducing_point_method,
        include_prediction_grid=include_prediction_grid,
        prediction_step_days=prediction_step_days,
    )

    # Verify required fields (including K and lag_scaled)
    required_fields = ['N_weight', 'N_workout', 't_weight', 't_workout',
                       'y_weight', 'y_workout', 'K', 'lag_scaled',
                       'use_sparse', 'M', 't_inducing']

    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f'Cross-lagged cumulative model requires {field} in stan_data.')

    # Compile and fit model
    print('Compiling cross-lagged Stan model (cumulative lags)...')
    model = CmdStanModel(stan_file=stan_file)

    print('Fitting cross-lagged model (cumulative lags)...')
    # Generate sensible initial values
    def generate_init():
        init = {
            'alpha_weight': 1.0,
            'alpha_workout': 1.0,
            'rho_weight': 0.2,
            'rho_workout': 0.2,
            'beta': 0.0,  # Start with no effect
            'sigma_weight': 0.1,
            'sigma_workout': 0.1,
            'nu_weight': 10.0,
            'nu_workout': 10.0,
            'eta_inducing_raw': np.zeros((stan_data['M'], 2)),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith('_')}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print('Creating ArviZ InferenceData...')
    N_pred = stan_data.get('N_pred', 0)
    coords = {
        'obs_weight': np.arange(stan_data['N_weight']),
        'obs_workout': np.arange(stan_data['N_workout']),
        'output': ['weight', 'workout'],
    }
    dims = {
        'f_weight': ['obs_weight'],
        'f_workout': ['obs_workout'],
        'y_weight': ['obs_weight'],
        'y_workout': ['obs_workout'],
        'y_weight_rep': ['obs_weight'],
        'y_workout_rep': ['obs_workout'],
        'log_lik_weight': ['obs_weight'],
        'log_lik_workout': ['obs_workout'],
    }
    if N_pred > 0:
        coords['pred'] = np.arange(N_pred)
        dims.update({
            'f_pred': ['pred', 'output'],
            'y_pred': ['pred', 'output'],
        })

    # Need to handle the fact that Stan outputs may use different names
    # We'll use default mapping and adjust as needed
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=['y_weight_rep', 'y_workout_rep'],
        observed_data={
            'y_weight': stan_data['y_weight'],
            'y_workout': stan_data['y_workout']
        },
        coords=coords,
        dims=dims,
        log_likelihood=None,
    )

    # Convert beta to original units for interpretation
    # β_original = β_scaled * (weight_sd / workout_sd)
    if '_y_weight_sd' in stan_data and '_y_workout_sd' in stan_data:
        weight_sd = stan_data['_y_weight_sd']
        workout_sd = stan_data['_y_workout_sd']
        if workout_sd > 0:
            beta_scaled = idata.posterior['beta'] * (weight_sd / workout_sd)
            # Add as new variable
            idata.posterior['beta_original_units'] = beta_scaled

    # Save to cache if caching is enabled (TODO)
    if cache_dir is not None:
        pass

    return fit, idata, stan_data


def fit_state_space_model(
    data_dir: Path | str = "data",
    df_weight: pd.DataFrame = None,
    df_intensity: pd.DataFrame = None,
    stan_file: Path | str = "stan/weight_state_space.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    activity_types: list[str] = None,
    max_hr: float = 185.0,
    intensity_col: str = "intensity",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit state-space model for weight prediction using workout intensity.

    Args:
        data_dir: Path to data directory (used only if df_weight or df_intensity not provided).
        df_weight: Optional DataFrame with weight observations.
                   If None, weight data is loaded from data_dir.
        df_intensity: Optional DataFrame with daily intensity values.
                      If None, intensity data is loaded from data_dir.
        stan_file: Path to state-space Stan model file.
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        adapt_delta: Target acceptance probability for NUTS (default: 0.95).
        max_treedepth: Maximum tree depth for NUTS (default: 12).
        activity_types: List of activity types to include for intensity calculation.
                       If None, includes ['strength_training', 'walking', 'cycling'].
        max_hr: Estimated maximum heart rate for intensity calculation.
        intensity_col: Name for intensity column.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, df_weight, df_intensity, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df_weight: DataFrame with weight observations
        - df_intensity: DataFrame with daily intensity values
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - skip for now like other models)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Load or use provided weight data
    if df_weight is None:
        print("Loading weight data...")
        df_weight = load_weight_data(data_dir)
    else:
        print("Using provided weight DataFrame")
        df_weight = df_weight.copy()

    # Load or use provided intensity data
    if df_intensity is None:
        print("Loading workout intensity data...")
        from src.data.intensity import load_intensity_data
        df_intensity = load_intensity_data(
            data_dir=data_dir,
            activity_types=activity_types,
            max_hr=max_hr,
            intensity_col=intensity_col,
        )
    else:
        print("Using provided intensity DataFrame")
        df_intensity = df_intensity.copy()

    # Prepare state-space Stan data
    print("Preparing state-space Stan data...")
    from src.data.intensity import prepare_state_space_data
    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
    )

    # Verify required fields
    required_fields = [
        'D', 'intensity', 'N_weight', 't_weight', 'y_weight', 'day_idx',
        'use_sparse', 'M', 't_inducing'
    ]
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"State-space model requires {field} in stan_data.")

    # Compile model
    print("Compiling state-space Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting state-space model...")
    # Generate initial values at prior centers (weakly informative priors)
    def generate_init():
        init = {
            'alpha': 0.5,        # beta(2,2) prior mean (also mode)
            'beta': 0.0,         # normal(0,1) prior mean and mode
            'gamma': 0.0,        # normal(0,1) prior mean and mode
            'sigma_f': 0.5,      # exponential(1) prior mean = 1, but start smaller for stability
            'sigma_w': 0.5,      # exponential(1) prior mean = 1, but start smaller for stability
            'alpha_gp': 0.5,     # normal(0,1) truncated [0.01,5] - start at reasonable value
            'rho_gp': 0.25,      # inv_gamma(3,0.5) prior mean = 0.25 (more stable than mode at 0.125)
            'fitness_raw': np.zeros(stan_data['D']),
            'eta_inducing_raw': np.zeros(stan_data['M']) if stan_data['M'] > 0 else np.array([]),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith("_")}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs_weight": np.arange(stan_data["N_weight"]),
        "day": np.arange(stan_data["D"]),
    }
    dims = {
        "fitness": ["day"],
        "f_gp": ["obs_weight"],
        "y_weight": ["obs_weight"],
        "y_weight_rep": ["obs_weight"],
        "log_lik_weight": ["obs_weight"],
    }
    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred", "component"],  # component: fitness_effect, gp_component
            "y_pred": ["pred", "component"],
        })

    # Extract posterior variables
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_weight_rep"],
        observed_data={"y_weight": stan_data["y_weight"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik_weight",
    )

    # Add fitness states to InferenceData
    # The Stan model stores fitness in 'fitness_stored' and GP in 'f_gp_stored'
    # Extract these from the fit object and add to posterior with proper dimensions
    import xarray as xr
    if 'fitness_stored' in fit.stan_variables() and 'fitness_stored' not in idata.posterior:
        fitness_data = fit.stan_variable('fitness_stored')  # shape (chain, draw, day)
        idata.posterior['fitness_stored'] = xr.DataArray(
            fitness_data,
            dims=("chain", "draw", "day"),
            coords={
                "chain": idata.posterior.coords["chain"],
                "draw": idata.posterior.coords["draw"],
                "day": idata.posterior.coords["day"]
            }
        )
    if 'f_gp_stored' in fit.stan_variables() and 'f_gp_stored' not in idata.posterior:
        f_gp_data = fit.stan_variable('f_gp_stored')  # shape (chain, draw, obs_weight)
        idata.posterior['f_gp_stored'] = xr.DataArray(
            f_gp_data,
            dims=("chain", "draw", "obs_weight"),
            coords={
                "chain": idata.posterior.coords["chain"],
                "draw": idata.posterior.coords["draw"],
                "obs_weight": idata.posterior.coords["obs_weight"]
            }
        )

    # Convert parameters to original units for interpretation
    # gamma_original = gamma_scaled * (weight_std / intensity_std)
    if 'weight_std' in stan_data and 'intensity_std' in stan_data:
        weight_std = stan_data['weight_std']
        intensity_std = stan_data['intensity_std']
        if intensity_std > 0:
            gamma_scaled = idata.posterior['gamma'] * (weight_std / intensity_std)
            idata.posterior['gamma_original_units'] = gamma_scaled

    # Save to cache if caching is enabled (skipped for now)
    if cache_dir is not None:
        pass  # Cache saving skipped (TODO: implement)

    return fit, idata, df_weight, df_intensity, stan_data


def fit_state_space_model_impulse(
    data_dir: Path | str = "data",
    df_weight: pd.DataFrame = None,
    df_intensity: pd.DataFrame = None,
    stan_file: Path | str = "stan/weight_state_space_impulse.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    activity_types: list[str] = None,
    max_hr: float = 185.0,
    intensity_col: str = "intensity",
    use_sparse: bool = True,
    n_inducing_points: int = 50,
    inducing_point_method: str = "uniform",
    cache: bool = True,
    force_refit: bool = False,
    include_prediction_grid: bool = False,
    prediction_step_days: int = 1,
) -> tuple:
    """Fit state-space model with impulse-response fitness.

    Same as fit_state_space_model but uses impulse-response model where:
    - impulse[t] = psi·impulse[t-1] + intensity[t]
    - fitness[t] = alpha·fitness[t-1] + beta·impulse[t-1] + ε_f[t]

    Args:
        data_dir: Path to data directory (used only if df_weight or df_intensity not provided).
        df_weight: Optional DataFrame with weight observations.
                   If None, weight data is loaded from data_dir.
        df_intensity: Optional DataFrame with daily intensity values.
                      If None, intensity data is loaded from data_dir.
        stan_file: Path to impulse-response state-space Stan model file.
        output_dir: Directory for output files.
        chains: Number of MCMC chains.
        iter_warmup: Warmup iterations per chain.
        iter_sampling: Sampling iterations per chain.
        adapt_delta: Target acceptance probability for NUTS (default: 0.95).
        max_treedepth: Maximum tree depth for NUTS (default: 12).
        activity_types: List of activity types to include for intensity calculation.
                       If None, includes ['strength_training', 'walking', 'cycling'].
        max_hr: Estimated maximum heart rate for intensity calculation.
        intensity_col: Name for intensity column.
        use_sparse: Whether to use sparse GP approximation.
        n_inducing_points: Number of inducing points for sparse GP.
        inducing_point_method: Method for selecting inducing points.
        cache: Whether to cache model results.
        force_refit: Force re-fitting even if cached results exist.
        include_prediction_grid: Whether to include prediction grid.
        prediction_step_days: Step size in days for prediction grid.

    Returns:
        Tuple of (fit, idata, df_weight, df_intensity, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df_weight: DataFrame with weight observations
        - df_intensity: DataFrame with daily intensity values
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Cache setup (simplified - skip for now like other models)
    cache_dir = None
    if cache:
        # For simplicity, skip cache for now (TODO: extend _compute_cache_key)
        pass

    # Load or use provided weight data
    if df_weight is None:
        print("Loading weight data...")
        df_weight = load_weight_data(data_dir)
    else:
        print("Using provided weight DataFrame")
        df_weight = df_weight.copy()

    # Load or use provided intensity data
    if df_intensity is None:
        print("Loading workout intensity data...")
        from src.data.intensity import load_intensity_data
        df_intensity = load_intensity_data(
            data_dir=data_dir,
            activity_types=activity_types,
            max_hr=max_hr,
            intensity_col=intensity_col,
        )
    else:
        print("Using provided intensity DataFrame")
        df_intensity = df_intensity.copy()

    # Prepare state-space Stan data
    print("Preparing state-space Stan data...")
    from src.data.intensity import prepare_state_space_data
    stan_data = prepare_state_space_data(
        df_weight=df_weight,
        df_intensity=df_intensity,
        use_sparse=use_sparse,
        n_inducing_points=n_inducing_points,
    )

    # Verify required fields
    required_fields = [
        'D', 'intensity', 'N_weight', 't_weight', 'y_weight', 'day_idx',
        'use_sparse', 'M', 't_inducing'
    ]
    for field in required_fields:
        if field not in stan_data:
            raise ValueError(f"State-space model requires {field} in stan_data.")

    # Compile model
    print("Compiling state-space Stan model...")
    model = CmdStanModel(stan_file=stan_file)

    print("Fitting state-space model...")
    # Generate initial values at prior centers (weakly informative priors)
    def generate_init():
        init = {
            'alpha': 0.8,        # beta(8,2) prior mode ~0.8
            'psi': 0.7,          # beta(5,2) prior mode ~0.7
            'beta': 0.3,         # normal(0.3,0.2) prior mean
            'gamma': -0.5,       # normal(-0.5,0.2) prior mean
            'sigma_f': 0.5,      # exponential(1) prior mean = 1, but start smaller for stability
            'sigma_w': 0.5,      # exponential(1) prior mean = 1, but start smaller for stability
            'alpha_gp': 0.5,     # exponential(5) prior mean = 0.2, but start at reasonable value
            'rho_gp': 0.25,      # inv_gamma(8,1) prior mean = 0.143, start at reasonable value
            'fitness_raw': np.zeros(stan_data['D']),
            'eta_inducing_raw': np.zeros(stan_data['M']) if stan_data['M'] > 0 else np.array([]),
        }
        return init

    # Filter out internal keys starting with underscore
    filtered_data = {k: v for k, v in stan_data.items() if not k.startswith("_")}

    fit = model.sample(
        data=filtered_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
        inits=generate_init(),
    )

    # Convert to ArviZ
    print("Creating ArviZ InferenceData...")
    N_pred = stan_data.get("N_pred", 0)
    coords = {
        "obs_weight": np.arange(stan_data["N_weight"]),
        "day": np.arange(stan_data["D"]),
    }
    dims = {
        "fitness": ["day"],
        "f_gp": ["obs_weight"],
        "y_weight": ["obs_weight"],
        "y_weight_rep": ["obs_weight"],
        "log_lik_weight": ["obs_weight"],
        "impulse": ["day"],  # Added for impulse model
    }
    if N_pred > 0:
        coords["pred"] = np.arange(N_pred)
        dims.update({
            "f_pred": ["pred", "component"],  # component: fitness_effect, gp_component
            "y_pred": ["pred", "component"],
        })

    # Extract posterior variables
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive=["y_weight_rep"],
        observed_data={"y_weight": stan_data["y_weight"]},
        coords=coords,
        dims=dims,
        log_likelihood="log_lik_weight",
    )

    # Add fitness and impulse states to InferenceData
    import xarray as xr
    if 'fitness_stored' in fit.stan_variables() and 'fitness_stored' not in idata.posterior:
        fitness_data = fit.stan_variable('fitness_stored')  # shape (chain, draw, day)
        idata.posterior['fitness_stored'] = xr.DataArray(
            fitness_data,
            dims=("chain", "draw", "day"),
            coords={
                "chain": idata.posterior.coords["chain"],
                "draw": idata.posterior.coords["draw"],
                "day": idata.posterior.coords["day"]
            }
        )
    if 'impulse_stored' in fit.stan_variables() and 'impulse_stored' not in idata.posterior:
        impulse_data = fit.stan_variable('impulse_stored')  # shape (chain, draw, day)
        idata.posterior['impulse_stored'] = xr.DataArray(
            impulse_data,
            dims=("chain", "draw", "day"),
            coords={
                "chain": idata.posterior.coords["chain"],
                "draw": idata.posterior.coords["draw"],
                "day": idata.posterior.coords["day"]
            }
        )
    if 'f_gp_stored' in fit.stan_variables() and 'f_gp_stored' not in idata.posterior:
        f_gp_data = fit.stan_variable('f_gp_stored')  # shape (chain, draw, obs_weight)
        idata.posterior['f_gp_stored'] = xr.DataArray(
            f_gp_data,
            dims=("chain", "draw", "obs_weight"),
            coords={
                "chain": idata.posterior.coords["chain"],
                "draw": idata.posterior.coords["draw"],
                "obs_weight": idata.posterior.coords["obs_weight"]
            }
        )

    # Convert parameters to original units for interpretation
    # gamma_original = gamma_scaled * (weight_std / intensity_std)
    if 'weight_std' in stan_data and 'intensity_std' in stan_data:
        weight_std = stan_data['weight_std']
        intensity_std = stan_data['intensity_std']
        if intensity_std > 0:
            gamma_scaled = idata.posterior['gamma'] * (weight_std / intensity_std)
            idata.posterior['gamma_original_units'] = gamma_scaled

    # Save to cache if caching is enabled (skipped for now)
    if cache_dir is not None:
        pass  # Cache saving skipped (TODO: implement)

    return fit, idata, df_weight, df_intensity, stan_data


if __name__ == "__main__":
    # Run the analysis
    fit, idata, df, stan_data = fit_weight_model()
    print_summary(idata, stan_data)
    plot_weight_fit(idata, df, stan_data, "output/weight_fit.png")
    plt.show()
