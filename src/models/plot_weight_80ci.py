"""Create weight trajectory plot with 80% credible interval."""
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from src.models.fit_weight import fit_weight_model


def plot_weight_80ci(idata, df, stan_data, output_path: Path | str = None):
    """Plot the fitted weight model with 80% credible interval.

    Args:
        idata: ArviZ InferenceData object
        df: Original DataFrame with weight data
        stan_data: Stan data dictionary with scaling parameters
        output_path: Optional path to save plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Back-transform predictions to original scale
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Extract posterior samples for f (GP function)
    f_samples = idata.posterior["f"].values

    # Compute posterior mean and 80% credible interval (10% - 90%)
    f_mean = f_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_lower = np.percentile(f_samples, 10, axis=(0, 1)) * y_sd + y_mean
    f_upper = np.percentile(f_samples, 90, axis=(0, 1)) * y_sd + y_mean

    # Plot observations
    ax.scatter(df["date"], df["weight_lbs"], alpha=0.5, s=20,
               label="Observations", color="black")

    # Plot GP mean (expected trajectory)
    ax.plot(df["date"], f_mean, "r-", linewidth=2,
            label="Expected trajectory (GP mean)")

    # Plot 80% credible interval
    ax.fill_between(df["date"], f_lower, f_upper, alpha=0.3,
                    color="red", label="80% credible interval")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Weight Over Time - Gaussian Process Fit with 80% CI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, ax


def main():
    """Run model and create 80% CI plot."""
    # Fit the model (or load existing results if available)
    print("Fitting weight model...")
    fit, idata, df, stan_data = fit_weight_model()

    # Create plot with 80% CI
    print("Creating plot with 80% credible interval...")
    fig, ax = plot_weight_80ci(idata, df, stan_data, "output/weight_fit_80ci.png")

    # Show summary statistics
    print("\n" + "=" * 60)
    print("MODEL SUMMARY WITH 80% CI")
    print("=" * 60)

    # Key parameters
    summary = az.summary(idata, var_names=["alpha", "rho", "sigma", "trend_change"])
    print("\nKey parameters:")
    print(summary)

    # Back-transform trend change
    y_sd = stan_data["_y_sd"]
    trend_change = idata.posterior["trend_change"].values.mean() * y_sd
    print(f"\nTrend change (original scale): {trend_change:.2f} lbs")

    print("\nPlot saved to output/weight_fit_80ci.png")
    plt.show()


if __name__ == "__main__":
    main()