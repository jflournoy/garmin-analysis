#!/usr/bin/env python3
"""Quick CLI demo for the cyclic GP model with daily cycle and comparison.

This script demonstrates the new cyclic Gaussian Process model that separates
daily variation from measurement error, and compares it with the original model.

Usage:
    python src/models/demo_cyclic.py [--chains N] [--iter-sampling N] [--output-dir DIR]

Example:
    python src/models/demo_cyclic.py --chains 2 --iter-sampling 200 --output-dir output/demo
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.fit_weight import (
    fit_weight_model,
    fit_weight_model_cyclic,
    fit_weight_model_spline,
    compare_models_sigma,
    compare_models_all,
)
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_daily_pattern,
    plot_model_comparison,
    plot_spline_daily_pattern,
    plot_models_comparison_all,
)


def main():
    parser = argparse.ArgumentParser(
        description="Demo cyclic GP model with daily cycle and comparison",
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
        default="output/demo_cyclic",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of MCMC chains (use 2 for quick demo)",
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
        "--skip-original",
        action="store_true",
        help="Skip fitting original model (use only cyclic)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (only print summary)",
    )
    parser.add_argument(
        "--include-spline",
        action="store_true",
        help="Include Fourier spline model in comparison",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CYCLIC GP MODEL DEMO - Daily Cycle Identification")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Caching: {not args.no_cache}, Force refit: {args.force_refit}")
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

    # Fit original model (if not skipped)
    if not args.skip_original:
        print("Fitting original GP model...")
        try:
            fit_orig, idata_orig, df_orig, stan_data_orig = fit_weight_model(
                data_dir=args.data_dir,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                cache=not args.no_cache,
                force_refit=args.force_refit,
            )
            print("  ✓ Original model fitted successfully")
        except Exception as e:
            print(f"  ✗ Error fitting original model: {e}")
            sys.exit(1)
    else:
        idata_orig = None
        print("  Skipping original model (--skip-original)")

    # Initialize spline model variables
    fit_spline = idata_spline = df_spline = stan_data_spline = None

    # Fit cyclic model
    print("Fitting cyclic GP model (trend + daily components)...")
    try:
        fit_cyclic, idata_cyclic, df_cyclic, stan_data_cyclic = fit_weight_model_cyclic(
            data_dir=args.data_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            cache=not args.no_cache,
            force_refit=args.force_refit,
        )
        print("  ✓ Cyclic model fitted successfully")
    except Exception as e:
        print(f"  ✗ Error fitting cyclic model: {e}")
        sys.exit(1)

    # Fit spline model (if included)
    if args.include_spline:
        print("Fitting spline GP model (trend + Fourier spline)...")
        try:
            fit_spline, idata_spline, df_spline, stan_data_spline = fit_weight_model_spline(
                data_dir=args.data_dir,
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                cache=not args.no_cache,
                force_refit=args.force_refit,
            )
            print("  ✓ Spline model fitted successfully")
        except Exception as e:
            print(f"  ✗ Error fitting spline model: {e}")
            sys.exit(1)

    # Model comparison
    print()
    print("-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)

    # Choose comparison function based on available models
    if idata_orig is not None:
        if idata_spline is not None:
            # Three-way comparison: original, cyclic, spline
            comparison = compare_models_all(
                idata_original=idata_orig,
                idata_cyclic=idata_cyclic,
                idata_spline=idata_spline,
                stan_data=stan_data_cyclic,
                print_summary=True,
            )
        else:
            # Two-way comparison: original vs cyclic
            comparison = compare_models_sigma(idata_orig, idata_cyclic, stan_data_cyclic)
    else:
        # Just print cyclic model summary (no original model)
        print("\nCyclic model summary:")
        sigma_cyclic = idata_cyclic.posterior["sigma"].values.mean() * stan_data_cyclic["_y_sd"]
        daily_amplitude = idata_cyclic.posterior["daily_amplitude"].values.mean() * stan_data_cyclic["_y_sd"]
        prop_variance = idata_cyclic.posterior["prop_variance_daily"].values.mean()
        print(f"  Measurement error (sigma): {sigma_cyclic:.2f} lbs")
        print(f"  Daily amplitude: {daily_amplitude:.2f} lbs")
        print(f"  Proportion of variance from daily component: {prop_variance:.3f}")

        # If spline available, print its summary separately
        if idata_spline is not None:
            print("\nSpline model summary:")
            sigma_spline = idata_spline.posterior["sigma"].values.mean() * stan_data_spline["_y_sd"]
            daily_amplitude_spline = idata_spline.posterior["daily_amplitude"].values.mean() * stan_data_spline["_y_sd"]
            prop_variance_spline = idata_spline.posterior["prop_variance_daily"].values.mean()
            print(f"  Measurement error (sigma): {sigma_spline:.2f} lbs")
            print(f"  Daily amplitude: {daily_amplitude_spline:.2f} lbs")
            print(f"  Proportion of variance from daily component: {prop_variance_spline:.3f}")

        comparison = None

    # Generate plots
    if not args.skip_plots:
        print()
        print("Generating visualizations...")

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

        # 2. Daily pattern plot
        daily_pattern_path = output_dir / "daily_pattern.png"
        try:
            plot_daily_pattern(
                idata_cyclic,
                df_cyclic,
                stan_data_cyclic,
                output_path=str(daily_pattern_path),
            )
            print(f"  ✓ Daily pattern plot saved to {daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate daily pattern plot: {e}")

        # 3. Spline daily pattern plot (if spline model available)
        if idata_spline is not None:
            spline_pattern_path = output_dir / "spline_daily_pattern.png"
            try:
                plot_spline_daily_pattern(
                    idata_spline,
                    stan_data_spline,
                    output_path=str(spline_pattern_path),
                )
                print(f"  ✓ Spline daily pattern plot saved to {spline_pattern_path}")
            except ValueError as e:
                print(f"  ⚠ Could not generate spline daily pattern plot: {e}")
            except Exception as e:
                print(f"  ⚠ Error generating spline daily pattern plot: {e}")

        # 4. Model comparison plot (if original model available)
        if idata_orig is not None:
            comparison_path = output_dir / "model_comparison.png"
            if idata_spline is not None:
                # Three-way comparison plot
                plot_models_comparison_all(
                    idata_original=idata_orig,
                    idata_cyclic=idata_cyclic,
                    idata_spline=idata_spline,
                    stan_data=stan_data_cyclic,
                    output_path=str(comparison_path),
                )
            else:
                # Two-way comparison plot
                plot_model_comparison(
                    idata_orig,
                    idata_cyclic,
                    stan_data_cyclic,
                    output_path=str(comparison_path),
                )
            print(f"  ✓ Model comparison plot saved to {comparison_path}")

        print()

    # Print final summary
    print("=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print("Cyclic model successfully separates daily variation from measurement error.")

    # Data sparsity note
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()
    if days_with_multiple / total_days < 0.1:
        print("\n⚠ DATA SPARSITY NOTE:")
        print(f"  Only {days_with_multiple}/{total_days} days have multiple measurements.")
        print("  With sparse intraday data, the 'daily component' may capture")
        print("  residual variation rather than true daily cyclic patterns.")
        print("  Consider collecting more frequent measurements for robust daily cycle estimation.")

    if comparison:
        sigma_reduction_pct = comparison.get("sigma_reduction_pct", 0)
        sigma_cyclic_lbs = comparison.get("sigma_cyclic_lbs", 0)
        daily_amplitude_lbs = comparison.get("daily_amplitude_lbs", 0)
        print(f"\n• Measurement error reduction: {sigma_reduction_pct:.1f}%")
        print(f"• Estimated measurement error (cyclic): {sigma_cyclic_lbs:.2f} lbs")
        print(f"• Daily variation amplitude: {daily_amplitude_lbs:.2f} lbs")
        if sigma_reduction_pct > 0:
            print("✓ Daily cycles improve model identifiability")
        else:
            print("⚠ Minimal daily variation detected in this dataset")
    print(f"\nPlots saved to: {output_dir}")
    print("To explore further, use:")
    print("  python src/models/plot_weight_cli.py --ci-levels 50 80 95 --show-stddev")
    print("  python src/models/plot_weight_cli.py --use-flexible --show-prior-predictive")
    print()

    # Show plots if not saved only
    if not args.skip_plots and not args.skip_original:
        print("Displaying plots (close windows to exit)...")
        plt.show()


if __name__ == "__main__":
    main()