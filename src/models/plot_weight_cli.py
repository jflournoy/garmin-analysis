"""Command-line interface for weight trajectory plotting with enhanced visualization."""
import argparse
from pathlib import Path
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from src.models.fit_weight import (
    fit_weight_model,
    fit_weight_model_flexible,
    fit_weight_model_cyclic,
    fit_weight_model_spline,
    generate_prior_predictive,
)


def plot_weight_enhanced(
    idata,
    df,
    stan_data,
    output_path: Path | str = None,
    ci_levels: list = [50, 80, 95],
    show_stddev: bool = False,
    show_multiple_intervals: bool = False,
    show_posterior_predictive: bool = False,
    posterior_predictive_level: float = 0.95,
    show_prior_predictive: bool = False,
    prior_predictive_level: float = 0.95,
):
    """Plot the fitted weight model with enhanced error visualization.

    Args:
        idata: ArviZ InferenceData object
        df: Original DataFrame with weight data
        stan_data: Stan data dictionary with scaling parameters
        output_path: Optional path to save plot
        ci_levels: List of credible interval levels to display (e.g., [50, 80, 95])
        show_stddev: Whether to plot standard deviation over time
        show_multiple_intervals: Whether to show multiple credible intervals
        show_posterior_predictive: Whether to show posterior predictive envelope
        posterior_predictive_level: Probability level for posterior predictive envelope (e.g., 0.95)
        show_prior_predictive: Whether to show prior predictive distribution
        prior_predictive_level: Probability level for prior predictive envelope (e.g., 0.95)
    """
    # Create figure with subplots
    if show_stddev:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        ax_main, ax_std = axes
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(12, 6))
        ax_std = None

    # Back-transform predictions to original scale
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Determine which parameter contains the GP function
    if "f_total" in idata.posterior:
        f_param = "f_total"
    elif "f" in idata.posterior:
        f_param = "f"
    else:
        raise ValueError("Could not find GP function parameter (f or f_total) in posterior")

    # Extract posterior samples for f (GP function)
    f_samples = idata.posterior[f_param].values  # shape: (chain, draw, obs)

    # Compute posterior mean and credible intervals
    f_mean = f_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_std = f_samples.std(axis=(0, 1)) * y_sd  # Standard deviation in original units

    # Plot observations
    ax_main.scatter(
        df["date"],
        df["weight_lbs"],
        alpha=0.5,
        s=20,
        label="Observations",
        color="black",
        zorder=10,
    )

    # Plot posterior predictive envelope if requested
    if show_posterior_predictive:
        try:
            # Extract posterior predictive samples
            y_rep_samples = idata.posterior_predictive["y_rep"].values  # shape: (chain, draw, obs)

            # Back-transform to original scale
            y_rep_samples = y_rep_samples * y_sd + y_mean

            # Compute credible interval for posterior predictive
            lower_percentile = (1 - posterior_predictive_level) / 2 * 100
            upper_percentile = 100 - lower_percentile

            y_rep_lower = np.percentile(y_rep_samples, lower_percentile, axis=(0, 1))
            y_rep_upper = np.percentile(y_rep_samples, upper_percentile, axis=(0, 1))

            # Plot as shaded region
            ax_main.fill_between(
                df["date"],
                y_rep_lower,
                y_rep_upper,
                alpha=0.15,
                color="blue",
                label=f"{posterior_predictive_level*100:.0f}% posterior predictive",
                edgecolor="none",
                zorder=1,
            )
        except KeyError:
            print("Warning: y_rep not found in posterior predictive samples. Posterior predictive envelope not plotted.")
        except Exception as e:
            print(f"Warning: Error plotting posterior predictive envelope: {e}")

    # Plot prior predictive envelope if requested
    if show_prior_predictive:
        try:
            # Check if prior predictive already exists in idata
            if hasattr(idata, 'prior_predictive') and 'y_prior_rep' in idata.prior_predictive:
                y_prior_rep_samples = idata.prior_predictive['y_prior_rep'].values
                # Reshape if needed (chain, draw, obs) -> (chain*draw, obs)
                if y_prior_rep_samples.ndim == 3:
                    y_prior_rep_samples = y_prior_rep_samples.reshape(-1, y_prior_rep_samples.shape[-1])
            else:
                # Generate prior predictive samples
                # Get time points (scaled) from stan_data
                t = stan_data.get('t')
                if t is None:
                    # Create scaled time from 0 to 1 based on date range
                    dates_numeric = (df['date'] - df['date'].min()).dt.total_seconds()
                    t = dates_numeric / dates_numeric.max()
                # Ensure t is a numpy array (stan_data may store as list)
                t = np.array(t, dtype=np.float64).flatten()

                # Get hyperparameters from stan_data (if using flexible model)
                alpha_prior_sd = stan_data.get('alpha_prior_sd', 1.0)
                rho_prior_shape = stan_data.get('rho_prior_shape', 5.0)
                rho_prior_scale = stan_data.get('rho_prior_scale', 1.0)
                sigma_prior_sd = stan_data.get('sigma_prior_sd', 0.5)

                # Generate prior predictive samples
                prior_results = generate_prior_predictive(
                    t=t,
                    n_samples=500,  # Reasonable number for visualization
                    alpha_prior_sd=alpha_prior_sd,
                    rho_prior_shape=rho_prior_shape,
                    rho_prior_scale=rho_prior_scale,
                    sigma_prior_sd=sigma_prior_sd,
                )
                y_prior_rep_samples = prior_results['y_prior_rep']

            # Back-transform to original scale
            y_prior_rep_samples = y_prior_rep_samples * y_sd + y_mean

            # Compute credible interval for prior predictive
            lower_percentile = (1 - prior_predictive_level) / 2 * 100
            upper_percentile = 100 - lower_percentile

            y_prior_lower = np.percentile(y_prior_rep_samples, lower_percentile, axis=0)
            y_prior_upper = np.percentile(y_prior_rep_samples, upper_percentile, axis=0)

            # Plot as shaded region (use purple color to distinguish from posterior)
            ax_main.fill_between(
                df["date"],
                y_prior_lower,
                y_prior_upper,
                alpha=0.1,
                color="purple",
                label=f"{prior_predictive_level*100:.0f}% prior predictive",
                edgecolor="none",
                zorder=0,
            )
        except KeyError:
            print("Warning: Could not generate prior predictive samples. Prior predictive envelope not plotted.")
        except Exception as e:
            print(f"Warning: Error plotting prior predictive envelope: {e}")

    # Plot multiple credible intervals if requested
    if show_multiple_intervals and ci_levels:
        # Sort levels descending for proper layering (wider intervals first)
        sorted_levels = sorted(ci_levels, reverse=True)
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(sorted_levels)))

        for level, color in zip(sorted_levels, colors):
            lower_percentile = (100 - level) / 2
            upper_percentile = 100 - lower_percentile

            f_lower = (
                np.percentile(f_samples, lower_percentile, axis=(0, 1)) * y_sd + y_mean
            )
            f_upper = (
                np.percentile(f_samples, upper_percentile, axis=(0, 1)) * y_sd + y_mean
            )

            ax_main.fill_between(
                df["date"],
                f_lower,
                f_upper,
                alpha=0.2,
                color=color,
                label=f"{level}% CI",
                edgecolor="none",
            )
    else:
        # Default: plot single 80% CI
        f_lower = np.percentile(f_samples, 10, axis=(0, 1)) * y_sd + y_mean
        f_upper = np.percentile(f_samples, 90, axis=(0, 1)) * y_sd + y_mean
        ax_main.fill_between(
            df["date"],
            f_lower,
            f_upper,
            alpha=0.3,
            color="red",
            label="80% credible interval",
        )

    # Plot GP mean (expected trajectory)
    ax_main.plot(
        df["date"],
        f_mean,
        "r-",
        linewidth=2,
        label="Expected trajectory (GP mean)",
        zorder=5,
    )

    ax_main.set_xlabel("Date")
    ax_main.set_ylabel("Weight (lbs)")
    title_parts = ["Weight Over Time - Gaussian Process Fit"]
    if show_multiple_intervals and ci_levels:
        title_parts.append(f"CI: {', '.join(str(l) + '%' for l in ci_levels)}")
    else:
        title_parts.append("80% CI")
    ax_main.set_title(" - ".join(title_parts))
    ax_main.legend(loc="best")
    ax_main.grid(True, alpha=0.3)

    # Plot standard deviation over time if requested
    if show_stddev and ax_std is not None:
        ax_std.plot(df["date"], f_std, "b-", linewidth=1.5, alpha=0.7)
        ax_std.fill_between(df["date"], 0, f_std, alpha=0.3, color="blue")

        # Compute and display summary statistics
        mean_std = f_std.mean()
        max_std = f_std.max()
        min_std = f_std.min()

        ax_std.axhline(y=mean_std, color="green", linestyle="--", alpha=0.7, linewidth=1)
        ax_std.text(
            df["date"].iloc[0],
            mean_std * 1.05,
            f"Mean: {mean_std:.2f} lbs",
            fontsize=9,
            color="green",
        )

        ax_std.set_xlabel("Date")
        ax_std.set_ylabel("Uncertainty (lbs)")
        ax_std.set_title(f"Uncertainty Over Time (Std Dev) - Range: {min_std:.2f} to {max_std:.2f} lbs")
        ax_std.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig, (ax_main, ax_std) if show_stddev else (ax_main,)


def main():
    """Command-line interface for weight trajectory plotting."""
    parser = argparse.ArgumentParser(
        description="Plot weight trajectory with Gaussian Process fit and uncertainty visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/weight_fit_enhanced.png",
        help="Output file path for the plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively (just save to file)",
    )

    # Credible interval options
    parser.add_argument(
        "--ci-levels",
        type=int,
        nargs="+",
        default=[80],
        help="Credible interval levels to display (e.g., 50 80 95)",
    )
    parser.add_argument(
        "--multiple-intervals",
        action="store_true",
        help="Show multiple credible intervals simultaneously",
    )

    # Error visualization options
    parser.add_argument(
        "--show-stddev",
        action="store_true",
        help="Show standard deviation over time in a separate subplot",
    )
    parser.add_argument(
        "--show-posterior-predictive",
        action="store_true",
        help="Show posterior predictive envelope (includes observation noise)",
    )
    parser.add_argument(
        "--posterior-predictive-level",
        type=float,
        default=0.95,
        help="Probability level for posterior predictive envelope (e.g., 0.95 for 95%%)",
    )
    parser.add_argument(
        "--show-prior-predictive",
        action="store_true",
        help="Show prior predictive distribution (requires prior sampling)",
    )
    parser.add_argument(
        "--prior-predictive-level",
        type=float,
        default=0.95,
        help="Probability level for prior predictive envelope (e.g., 0.95 for 95%%)",
    )

    # Model fitting options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--stan-file",
        type=str,
        default="stan/weight_gp.stan",
        help="Path to Stan model file",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Warmup iterations per chain",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Sampling iterations per chain",
    )
    parser.add_argument(
        "--alpha-prior-sd",
        type=float,
        default=1.0,
        help="Standard deviation for alpha prior (normal distribution)",
    )
    parser.add_argument(
        "--rho-prior-shape",
        type=float,
        default=5.0,
        help="Shape parameter for rho prior (inverse gamma)",
    )
    parser.add_argument(
        "--rho-prior-scale",
        type=float,
        default=1.0,
        help="Scale parameter for rho prior (inverse gamma)",
    )
    parser.add_argument(
        "--sigma-prior-sd",
        type=float,
        default=0.5,
        help="Standard deviation for sigma prior (normal distribution)",
    )
    parser.add_argument(
        "--use-cyclic",
        action="store_true",
        help="Use the cyclic Stan model (weight_gp_cyclic.stan) with trend + daily components",
    )
    parser.add_argument(
        "--use-spline",
        action="store_true",
        help="Use the spline Stan model (weight_gp_spline.stan) with Fourier harmonics",
    )
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics for spline model (default: 2)",
    )
    parser.add_argument(
        "--use-flexible",
        action="store_true",
        help="Use the flexible Stan model (weight_gp_flexible.stan)",
    )

    args = parser.parse_args()

    # Validate model selection (only one model type allowed)
    model_flags = [args.use_flexible, args.use_spline, args.use_cyclic]
    if sum(model_flags) > 1:
        print("Error: Only one of --use-flexible, --use-spline, --use-cyclic can be specified", file=sys.stderr)
        sys.exit(1)

    # Validate CI levels
    for level in args.ci_levels:
        if not 0 < level < 100:
            print(f"Error: CI level {level} must be between 0 and 100", file=sys.stderr)
            sys.exit(1)

    # Fit the model
    print("Fitting weight model...")
    try:
        # Determine which model to use
        use_flexible = args.use_flexible or \
                      args.alpha_prior_sd != 1.0 or \
                      args.rho_prior_shape != 5.0 or \
                      args.rho_prior_scale != 1.0 or \
                      args.sigma_prior_sd != 0.5

        if use_flexible:
            print("Using flexible model with custom priors...")
            fit, idata, df, stan_data = fit_weight_model_flexible(
                data_dir=args.data_dir,
                stan_file="stan/weight_gp_flexible.stan",
                output_dir="output",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                alpha_prior_sd=args.alpha_prior_sd,
                rho_prior_shape=args.rho_prior_shape,
                rho_prior_scale=args.rho_prior_scale,
                sigma_prior_sd=args.sigma_prior_sd,
            )
        elif args.use_spline:
            print("Using spline model with Fourier harmonics...")
            fit, idata, df, stan_data = fit_weight_model_spline(
                data_dir=args.data_dir,
                stan_file="stan/weight_gp_spline.stan",
                output_dir="output",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                fourier_harmonics=args.fourier_harmonics,
            )
        elif args.use_cyclic:
            print("Using cyclic model with trend + daily components...")
            fit, idata, df, stan_data = fit_weight_model_cyclic(
                data_dir=args.data_dir,
                stan_file="stan/weight_gp_cyclic.stan",
                output_dir="output",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
            )
        else:
            fit, idata, df, stan_data = fit_weight_model(
                data_dir=args.data_dir,
                stan_file=args.stan_file,
                output_dir="output",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
            )
    except Exception as e:
        print(f"Error fitting model: {e}", file=sys.stderr)
        sys.exit(1)

    # Create enhanced plot
    print("Creating enhanced plot...")
    try:
        fig, axes = plot_weight_enhanced(
            idata,
            df,
            stan_data,
            output_path=args.output,
            ci_levels=args.ci_levels,
            show_stddev=args.show_stddev,
            show_multiple_intervals=args.multiple_intervals,
            show_posterior_predictive=args.show_posterior_predictive,
            posterior_predictive_level=args.posterior_predictive_level,
            show_prior_predictive=args.show_prior_predictive,
            prior_predictive_level=args.prior_predictive_level,
        )
    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    # Determine model type for summary
    y_sd = stan_data["_y_sd"]

    # Select appropriate parameters based on model type
    if args.use_cyclic:
        var_names = ["alpha_trend", "rho_trend", "alpha_daily", "rho_daily", "sigma", "trend_change", "daily_amplitude", "prop_variance_daily"]
    elif args.use_spline:
        var_names = ["alpha_trend", "rho_trend", "sigma", "sigma_fourier", "trend_change", "daily_amplitude", "prop_variance_daily"]
    else:
        var_names = ["alpha", "rho", "sigma", "trend_change"]

    # Print parameter summary
    summary = az.summary(idata, var_names=var_names)
    print("\nKey parameters:")
    print(summary)

    # Back-transform trend change
    trend_change = idata.posterior["trend_change"].values.mean() * y_sd
    print(f"\nTrend change (original scale): {trend_change:.2f} lbs")

    # Determine which parameter contains the GP function for uncertainty stats
    if "f_total" in idata.posterior:
        f_param = "f_total"
    elif "f" in idata.posterior:
        f_param = "f"
    else:
        raise ValueError("Could not find GP function parameter (f or f_total) in posterior")

    # Uncertainty statistics
    f_samples = idata.posterior[f_param].values
    f_std = f_samples.std(axis=(0, 1)) * y_sd
    print("\nUncertainty statistics (standard deviation):")
    print(f"  Mean: {f_std.mean():.2f} lbs")
    print(f"  Min: {f_std.min():.2f} lbs")
    print(f"  Max: {f_std.max():.2f} lbs")
    print(f"  Range: {f_std.max() - f_std.min():.2f} lbs")

    print(f"\nPlot saved to {args.output}")

    # Show plot unless --no-show flag is set
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()