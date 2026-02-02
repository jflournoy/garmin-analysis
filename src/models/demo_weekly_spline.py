"""Demo script for weekly spline model with trend + daily + weekly Fourier components."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.fit_weight import fit_weight_model_spline_weekly
from src.models.plot_cyclic import plot_model_predictions, plot_weekly_zoomed_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Fit weekly spline model (trend + daily + weekly Fourier components)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/weekly-spline-demo",
        help="Output directory (default: output/weekly-spline-demo)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Warmup iterations per chain (default: 500)",
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Sampling iterations per chain (default: 500)",
    )
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics for daily cycles (K parameter, default: 2)",
    )
    parser.add_argument(
        "--weekly-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics for weekly cycles (L parameter, default: 2)",
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
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force re-fitting even if cached results exist",
    )
    parser.add_argument(
        "--include-prediction-grid",
        action="store_true",
        help="Include prediction grid for unobserved days",
    )
    parser.add_argument(
        "--prediction-hour",
        type=float,
        default=8.0,
        help="Hour of day for prediction points (default: 8.0 = 8 AM)",
    )
    parser.add_argument(
        "--prediction-hour-step",
        type=float,
        default=None,
        help="Step size in hours for multiple prediction hours per day (default: None = single hour)",
    )
    parser.add_argument(
        "--prediction-step-days",
        type=int,
        default=1,
        help="Step size in days for prediction grid (default: 1 = daily)",
    )
    parser.add_argument(
        "--zoom-to",
        type=str,
        choices=["last_week", "last_month", "last_year", "all"],
        default="last_week",
        help="Zoom preset for weekly zoomed plot (default: last_week)",
    )

    args = parser.parse_args()
    use_sparse = not args.no_sparse

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WEEKLY SPLINE MODEL DEMO")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Daily Fourier harmonics (K): {args.fourier_harmonics}")
    print(f"Weekly Fourier harmonics (L): {args.weekly_harmonics}")
    print(f"Sparse GP: {use_sparse}")
    if use_sparse:
        print(f"  Inducing points: {args.n_inducing_points}")
    print(f"Prediction grid: {args.include_prediction_grid}")
    if args.include_prediction_grid:
        print(f"  Prediction hour: {args.prediction_hour}")
        if args.prediction_hour_step:
            print(f"  Prediction hour step: {args.prediction_hour_step}")
        print(f"  Prediction step days: {args.prediction_step_days}")
    print()

    # Fit the weekly spline model
    fit, idata, df, stan_data = fit_weight_model_spline_weekly(
        data_dir=args.data_dir,
        output_dir=output_dir,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        fourier_harmonics=args.fourier_harmonics,
        weekly_harmonics=args.weekly_harmonics,
        use_sparse=use_sparse,
        n_inducing_points=args.n_inducing_points,
        cache=True,
        force_refit=args.force_refit,
        include_prediction_grid=args.include_prediction_grid,
        prediction_hour=args.prediction_hour,
        prediction_hour_step=args.prediction_hour_step,
        prediction_step_days=args.prediction_step_days,
    )

    # Print model summary
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    # Extract key parameters
    sigma = idata.posterior["sigma"].values.mean()
    weekly_amplitude = idata.posterior["weekly_amplitude"].values.mean()
    daily_amplitude = idata.posterior["daily_amplitude"].values.mean()
    prop_variance_weekly = idata.posterior["prop_variance_weekly"].values.mean()
    prop_variance_daily = idata.posterior["prop_variance_daily"].values.mean()

    # Back-transform to original scale (lbs)
    y_sd = stan_data["_y_sd"]
    sigma_lbs = sigma * y_sd
    weekly_amplitude_lbs = weekly_amplitude * y_sd
    daily_amplitude_lbs = daily_amplitude * y_sd

    print("\nKey diagnostics (standardized scale):")
    print(f"  Sigma (residual std): {sigma:.4f}")
    print(f"  Daily amplitude: {daily_amplitude:.4f}")
    print(f"  Weekly amplitude: {weekly_amplitude:.4f}")
    print(f"  Proportion of variance from daily component: {prop_variance_daily:.4f}")
    print(f"  Proportion of variance from weekly component: {prop_variance_weekly:.4f}")

    print("\nKey diagnostics (original scale - lbs):")
    print(f"  Sigma (residual std): {sigma_lbs:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude_lbs:.2f} lbs")
    print(f"  Weekly amplitude: {weekly_amplitude_lbs:.2f} lbs")

    # Interpretation
    print("\nINTERPRETATION:")
    if weekly_amplitude > 0.05:  # Arbitrary threshold for "meaningful"
        print("  ✓ Weekly component captures meaningful variation")
        if prop_variance_weekly > 0.05:
            print(f"  ✓ Weekly component explains {prop_variance_weekly:.1%} of total variance")
    else:
        print(f"  ⚠ Weekly amplitude is small ({weekly_amplitude:.4f}), may not be meaningful")

    if daily_amplitude > 0.05:
        print("  ✓ Daily component captures meaningful variation")
        if prop_variance_daily > 0.05:
            print(f"  ✓ Daily component explains {prop_variance_daily:.1%} of total variance")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Main prediction plot
    if args.include_prediction_grid:
        predictions_path = output_dir / "weekly_spline_predictions.png"
        print(f"Creating prediction plot: {predictions_path}")
        fig = plot_model_predictions(
            idata, df, stan_data,
            title="Weekly Spline Model: Trend + Daily + Weekly Components",
            output_path=predictions_path,
        )
        plt.close(fig)

        # 2. Weekly zoomed plot
        if args.prediction_hour_step is None or args.prediction_hour_step >= 1:
            # Only create weekly zoomed plot if we have at least hourly predictions
            weekly_zoom_path = output_dir / "weekly_spline_zoomed.png"
            print(f"Creating weekly zoomed plot: {weekly_zoom_path}")

            # Determine zoom range
            if args.zoom_to == "last_week":
                # Zoom to last week of data
                zoom_end = df["date"].max()
                zoom_start = zoom_end - pd.Timedelta(days=7)
            elif args.zoom_to == "last_month":
                zoom_end = df["date"].max()
                zoom_start = zoom_end - pd.Timedelta(days=30)
            elif args.zoom_to == "last_year":
                zoom_end = df["date"].max()
                zoom_start = zoom_end - pd.Timedelta(days=365)
            else:  # "all"
                zoom_start = df["date"].min()
                zoom_end = df["date"].max()

            fig = plot_weekly_zoomed_predictions(
                idata, df, stan_data,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                title=f"Weekly Spline Model - Zoomed to {args.zoom_to.replace('_', ' ')}",
                output_path=weekly_zoom_path,
            )
            plt.close(fig)

    # 3. Component decomposition plot
    print("Creating component decomposition plot...")
    component_path = output_dir / "weekly_spline_components.png"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Back-transform components
    y_mean = stan_data["_y_mean"]
    y_sd = stan_data["_y_sd"]

    # Extract components
    f_trend_samples = idata.posterior["f_trend"].values
    f_daily_samples = idata.posterior["f_daily"].values
    f_weekly_samples = idata.posterior["f_weekly"].values

    f_trend_mean = f_trend_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_daily_mean = f_daily_samples.mean(axis=(0, 1)) * y_sd + y_mean
    f_weekly_mean = f_weekly_samples.mean(axis=(0, 1)) * y_sd + y_mean

    # Plot trend component
    ax = axes[0]
    ax.plot(df["date"], f_trend_mean, "b-", linewidth=2)
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Trend Component")
    ax.grid(True, alpha=0.3)

    # Plot daily component
    ax = axes[1]
    ax.plot(df["date"], f_daily_mean, "g-", linewidth=2)
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Daily Component (24-hour cycles)")
    ax.grid(True, alpha=0.3)

    # Plot weekly component
    ax = axes[2]
    ax.plot(df["date"], f_weekly_mean, "r-", linewidth=2)
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Weekly Component (7-day cycles)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(component_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to: {component_path}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"All outputs saved to: {output_dir}")
    print("\nTo explore the model further:")
    print(f"  - Check component plots in {component_path}")
    if args.include_prediction_grid:
        print(f"  - Check prediction plots in {output_dir}/")
    print("\nModel diagnostics:")
    print(f"  Weekly amplitude: {weekly_amplitude_lbs:.2f} lbs")
    print(f"  Daily amplitude: {daily_amplitude_lbs:.2f} lbs")
    print(f"  Residual std: {sigma_lbs:.2f} lbs")


if __name__ == "__main__":
    # Need to import pandas here to avoid circular imports
    import pandas as pd
    main()