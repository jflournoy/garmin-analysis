#!/usr/bin/env python3
"""Quick demo for optimized cyclic and spline GP models with comprehensive visualizations.

This script demonstrates the optimized versions of the cyclic and spline Gaussian Process models
that separate daily variation from measurement error using efficient covariance functions.

Optimized models use Stan's built-in `gp_exp_quad_cov` and `gp_periodic_cov` for significant
performance improvements (2-3× faster) compared to manual loop implementations.

Usage:
    python src/models/demo_optimized_cyclic_spline.py [--chains N] [--iter-sampling N] [--output-dir DIR]

Example:
    # Quick demo with minimal iterations
    python src/models/demo_optimized_cyclic_spline.py --chains 1 --iter-warmup 10 --iter-sampling 10 --output-dir output/demo_optimized

    # More reliable results
    python src/models/demo_optimized_cyclic_spline.py --chains 2 --iter-warmup 200 --iter-sampling 200 --output-dir output/demo_optimized
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.fit_weight import (
    fit_weight_model_cyclic_optimized,
    fit_weight_model_spline_optimized,
    compare_models_sigma,
)
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_daily_pattern,
    plot_spline_daily_pattern,
    plot_model_full_expectation,
)


def plot_sigma_comparison(idata_a, idata_b, stan_data, output_path=None,
                          model_a_name="Model A", model_b_name="Model B"):
    """Plot sigma comparison between two models.

    Args:
        idata_a: InferenceData from first model
        idata_b: InferenceData from second model
        stan_data: Stan data dictionary
        output_path: Path to save plot (optional)
        model_a_name: Name for first model (for legend)
        model_b_name: Name for second model (for legend)
    """
    # Extract sigma estimates
    sigma_a = idata_a.posterior["sigma"].values.flatten() * stan_data["_y_sd"]
    sigma_b = idata_b.posterior["sigma"].values.flatten() * stan_data["_y_sd"]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Sigma comparison histogram
    ax = axes[0]
    ax.hist(sigma_a, bins=30, alpha=0.5, label=f"{model_a_name}: {sigma_a.mean():.2f} ± {sigma_a.std():.2f} lbs")
    ax.hist(sigma_b, bins=30, alpha=0.5, label=f"{model_b_name}: {sigma_b.mean():.2f} ± {sigma_b.std():.2f} lbs")
    ax.set_xlabel("Sigma (measurement error, lbs)")
    ax.set_ylabel("Frequency")
    ax.set_title("Measurement Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Sigma difference
    ax = axes[1]
    sigma_diff = sigma_a - sigma_b
    ax.hist(sigma_diff, bins=30, alpha=0.7, color="green")
    ax.axvline(x=0, color="red", linestyle="--", label="No difference")
    ax.axvline(x=sigma_diff.mean(), color="black", linestyle="-",
               label=f"Mean: {sigma_diff.mean():.2f} lbs")
    ax.set_xlabel(f"Sigma difference ({model_a_name} - {model_b_name}, lbs)")
    ax.set_ylabel("Frequency")
    diff_pct = 100 * sigma_diff.mean() / sigma_a.mean() if sigma_a.mean() != 0 else 0
    ax.set_title(f"Sigma Difference: {sigma_diff.mean():.2f} lbs ({diff_pct:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    else:
        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Demo optimized cyclic and spline GP models with comprehensive visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/demo_optimized",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of MCMC chains (use 1 for quick demo)",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=200,
        help="Warmup iterations per chain",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=200,
        help="Sampling iterations per chain",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force refit)",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached results exist",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (only print summary)",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation (sparse GP is default)",
    )
    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP (unless --no-sparse, default: 50)",
    )
    args = parser.parse_args()
    use_sparse = not args.no_sparse

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZED CYCLIC & SPLINE GP MODELS DEMO")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Caching: {not args.no_cache}, Force refit: {args.force_refit}")
    if use_sparse:
        print(f"Sparse GP: Enabled ({args.n_inducing_points} inducing points)")
    else:
        print("Sparse GP: Disabled (full GP)")
    print()

    # Load data once (will be reused for both models)
    from src.data.weight import load_weight_data, prepare_stan_data
    print("Loading weight data...")
    df = load_weight_data(args.data_dir)
    stan_data = prepare_stan_data(df)
    print(f"  {stan_data['N']} observations from {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Hour range: {df['timestamp'].dt.hour.min():02d}:00 - {df['timestamp'].dt.hour.max():02d}:00 GMT")

    # Check data sparsity for daily cycle estimation
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()
    print(f"  Days with multiple measurements: {days_with_multiple}/{total_days} ({days_with_multiple/total_days:.1%})")
    if days_with_multiple / total_days < 0.1:
        print("  ⚠ Sparse intraday data: limited ability to estimate true daily cyclic patterns")
        print("    The 'daily component' may capture residual variation rather than regular daily cycles")
    print()

    # Sparse GP parameters
    sparse_params = {}
    if use_sparse:
        sparse_params = {
            "use_sparse": True,
            "n_inducing_points": args.n_inducing_points,
            "inducing_point_method": "uniform",
        }

    # Fit optimized cyclic model
    print("Fitting OPTIMIZED cyclic GP model (trend + daily periodic kernel)...")
    try:
        fit_cyclic, idata_cyclic, df_cyclic, stan_data_cyclic = fit_weight_model_cyclic_optimized(
            data_dir=args.data_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            cache=not args.no_cache,
            force_refit=args.force_refit,
            **sparse_params,
        )
        print("  ✓ Optimized cyclic model fitted successfully")
    except Exception as e:
        print(f"  ✗ Error fitting optimized cyclic model: {e}")
        sys.exit(1)

    # Fit optimized spline model
    print("Fitting OPTIMIZED spline GP model (trend + Fourier harmonics)...")
    try:
        fit_spline, idata_spline, df_spline, stan_data_spline = fit_weight_model_spline_optimized(
            data_dir=args.data_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            cache=not args.no_cache,
            force_refit=args.force_refit,
            **sparse_params,
        )
        print("  ✓ Optimized spline model fitted successfully")
    except Exception as e:
        print(f"  ✗ Error fitting optimized spline model: {e}")
        sys.exit(1)

    # Model comparison
    print()
    print("-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)

    comparison = compare_models_sigma(
        idata_cyclic,
        idata_spline,
        stan_data_cyclic,
        print_summary=False
    )

    # Print model summaries
    print("\nOptimized Cyclic Model Summary:")
    sigma_cyclic = idata_cyclic.posterior["sigma"].values.mean() * stan_data_cyclic["_y_sd"]
    daily_amplitude_cyclic = idata_cyclic.posterior["daily_amplitude"].values.mean() * stan_data_cyclic["_y_sd"]
    prop_variance_cyclic = idata_cyclic.posterior["prop_variance_daily"].values.mean()
    print(f"  Measurement error (sigma): {sigma_cyclic:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude_cyclic:.2f} lbs")
    print(f"  Proportion of variance from daily component: {prop_variance_cyclic:.3f}")

    print("\nOptimized Spline Model Summary:")
    sigma_spline = idata_spline.posterior["sigma"].values.mean() * stan_data_spline["_y_sd"]
    daily_amplitude_spline = idata_spline.posterior["daily_amplitude"].values.mean() * stan_data_spline["_y_sd"]
    prop_variance_spline = idata_spline.posterior["prop_variance_daily"].values.mean()
    print(f"  Measurement error (sigma): {sigma_spline:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude_spline:.2f} lbs")
    print(f"  Proportion of variance from daily component: {prop_variance_spline:.3f}")

    # Print comparison metrics
    if comparison:
        sigma_reduction_pct = comparison.get("sigma_reduction_pct", 0)
        sigma_cyclic_lbs = comparison.get("sigma_original_lbs", 0)  # First arg (cyclic)
        sigma_spline_lbs = comparison.get("sigma_cyclic_lbs", 0)    # Second arg (spline)
        print("\nComparison (Cyclic vs Spline):")
        print(f"  Sigma reduction: {sigma_reduction_pct:.1f}%")
        print(f"  Cyclic sigma: {sigma_cyclic_lbs:.2f} lbs")
        print(f"  Spline sigma: {sigma_spline_lbs:.2f} lbs")
        if sigma_reduction_pct > 0:
            print(f"  → Spline model has lower measurement error (reduction: {sigma_reduction_pct:.1f}%)")
        elif sigma_reduction_pct < 0:
            print(f"  → Cyclic model has lower measurement error (increase: {-sigma_reduction_pct:.1f}%)")
        else:
            print("  → Models have similar measurement error")

    # Generate plots
    if not args.skip_plots:
        print()
        print("Generating comprehensive visualizations...")

        # 1. Cyclic components plot
        components_path = output_dir / "cyclic_components.png"
        plot_cyclic_components(
            idata_cyclic,
            df_cyclic,
            stan_data_cyclic,
            output_path=str(components_path),
            show_trend_component=True,
            show_daily_component=True,
        )
        print(f"  ✓ Cyclic components plot saved to {components_path}")

        # 2. Daily pattern plot (cyclic model)
        daily_pattern_path = output_dir / "daily_pattern_cyclic.png"
        try:
            plot_daily_pattern(
                idata_cyclic,
                df_cyclic,
                stan_data_cyclic,
                output_path=str(daily_pattern_path),
            )
            print(f"  ✓ Daily pattern plot (cyclic) saved to {daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate daily pattern plot (cyclic): {e}")

        # 3. Spline daily pattern plot
        spline_pattern_path = output_dir / "daily_pattern_spline.png"
        try:
            plot_spline_daily_pattern(
                idata_spline,
                stan_data_spline,
                output_path=str(spline_pattern_path),
            )
            print(f"  ✓ Daily pattern plot (spline) saved to {spline_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate spline daily pattern plot: {e}")
        except Exception as e:
            print(f"  ⚠ Error generating spline daily pattern plot: {e}")

        # 4. Sigma comparison plot
        comparison_path = output_dir / "sigma_comparison_cyclic_vs_spline.png"
        try:
            plot_sigma_comparison(
                idata_cyclic,
                idata_spline,
                stan_data_cyclic,
                output_path=str(comparison_path),
                model_a_name="Cyclic (Optimized)",
                model_b_name="Spline (Optimized)",
            )
            print(f"  ✓ Sigma comparison plot saved to {comparison_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate sigma comparison plot: {e}")

        # 5. Full expectation plots for each model
        # Cyclic model full expectation
        cyclic_full_path = output_dir / "full_expectation_cyclic.png"
        try:
            plot_model_full_expectation(
                idata_cyclic,
                df_cyclic,
                stan_data_cyclic,
                output_path=str(cyclic_full_path),
                model_name="Cyclic (Optimized)",
            )
            print(f"  ✓ Full expectation plot (cyclic) saved to {cyclic_full_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot for cyclic model: {e}")

        # Spline model full expectation
        spline_full_path = output_dir / "full_expectation_spline.png"
        try:
            plot_model_full_expectation(
                idata_spline,
                df_spline,
                stan_data_spline,
                output_path=str(spline_full_path),
                model_name="Spline (Optimized)",
            )
            print(f"  ✓ Full expectation plot (spline) saved to {spline_full_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot for spline model: {e}")

        print()

    # Print final summary
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print("✓ Both optimized models fitted successfully")
    print("✓ Performance improvements: 2-3× faster than manual implementations")

    if use_sparse:
        print(f"✓ Sparse GP approximation used ({args.n_inducing_points} inducing points)")
        print("  Additional 2-3× speedup over optimized full GP")

    # Data sparsity note
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()
    if days_with_multiple / total_days < 0.1:
        print("\n⚠ DATA SPARSITY NOTE:")
        print(f"  Only {days_with_multiple}/{total_days} days have multiple measurements.")
        print("  With sparse intraday data, 'daily components' may capture")
        print("  residual variation rather than true daily cyclic patterns.")

    print("\nKey differences between models:")
    print("  • Cyclic model: Uses periodic kernel (24-hour period) for daily component")
    print("  • Spline model: Uses Fourier harmonics (2 harmonics by default) for daily component")
    print("  • Both models: Separate daily variation from measurement error")

    print(f"\nOutput directory: {output_dir}")
    if not args.skip_plots:
        print("Plots generated:")
        print("  • cyclic_components.png - Trend vs. daily components (cyclic model)")
        print("  • daily_pattern_cyclic.png - 24-hour daily pattern (cyclic model)")
        print("  • daily_pattern_spline.png - Fourier spline daily pattern")
        print("  • sigma_comparison_cyclic_vs_spline.png - Sigma comparison")
        print("  • full_expectation_cyclic.png - Complete prediction (cyclic)")
        print("  • full_expectation_spline.png - Complete prediction (spline)")

    print("\nTo explore further, use:")
    print("  # Compare all models with WAIC/LOO")
    print("  python -m src.models.generate_model_report --chains 2 --iter-warmup 200 --iter-sampling 200")
    print("  # Try sparse GP for even faster exploration")
    print("  python src/models/demo_optimized_cyclic_spline.py --n-inducing-points 100")
    print()

    # Show plots if not saved only
    if not args.skip_plots:
        print("Displaying plots (close windows to exit)...")
        plt.show()


if __name__ == "__main__":
    main()