"""Fit Bayesian weight model using CmdStanPy."""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from cmdstanpy import CmdStanModel

from src.data.weight import load_weight_data, prepare_stan_data


def fit_weight_model(
    data_dir: Path | str = "data",
    stan_file: Path | str = "stan/weight_gp.stan",
    output_dir: Path | str = "output",
    chains: int = 4,
    iter_warmup: int = 500,
    iter_sampling: int = 500,
) -> tuple:
    """Fit the GP weight model and return results.

    Args:
        data_dir: Path to data directory
        stan_file: Path to Stan model file
        output_dir: Directory for output files
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain

    Returns:
        Tuple of (fit, idata, df, stan_data) where:
        - fit: CmdStanMCMC object
        - idata: ArviZ InferenceData object
        - df: Original DataFrame
        - stan_data: Stan data dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

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
        dims={"f": ["obs"], "y_rep": ["obs"], "y": ["obs"]},
    )

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


if __name__ == "__main__":
    # Run the analysis
    fit, idata, df, stan_data = fit_weight_model()
    print_summary(idata, stan_data)
    plot_weight_fit(idata, df, stan_data, "output/weight_fit.png")
    plt.show()
