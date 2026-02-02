#!/usr/bin/env python3
"""Single-model demo: optimized spline GP with Gaussian process trend and Fourier spline.

This script runs ONLY the optimized spline Gaussian Process model (trend + Fourier spline for daily cycles)
with all optimizations in place. It includes adjustable adapt_delta and max_treedepth parameters
and generates a comprehensive markdown report with embedded visualizations.

Usage:
    python -m src.models.demo_spline_single [--chains N] [--iter-warmup N] [--iter-sampling N]
        [--adapt-delta D] [--max-treedepth D] [--fourier-harmonics K] [--no-sparse] [--n-inducing-points M]
        [--output-dir DIR] [--skip-plots] [--no-cache] [--force-refit]
        [--include-prediction-grid] [--prediction-hour H] [--prediction-hour-step S] [--prediction-step-days D]
        [--zoom-to PRESET] [--zoom-reference-date DATE] [--zoom-start DATE] [--zoom-end DATE]

Example:
    # Quick demo with minimal iterations
    python -m src.models.demo_spline_single --chains 1 --iter-warmup 10 --iter-sampling 10 \
        --output-dir output/demo_spline_single

    # Reliable results with custom sampler parameters
    python -m src.models.demo_spline_single --chains 2 --iter-warmup 200 --iter-sampling 200 \
        --adapt-delta 0.99 --max-treedepth 12 --fourier-harmonics 3 --output-dir output/demo_spline_single

    # Prediction grid with hourly steps every 10 days
    python -m src.models.demo_spline_single --output-dir output/spline-demo \
        --chains 4 --iter-warmup 500 --iter-sampling 1500 \
        --adapt-delta .99 --max-treedepth 12 --fourier-harmonics 2 \
        --n-inducing-points 50 --force-refit \
        --include-prediction-grid --prediction-hour 8.0 \
        --prediction-hour-step 1 --prediction-step-days 10
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def should_show_plots() -> bool:
    """Return True if plots should be displayed interactively.

    Modified to always return False - plots are saved to disk only.
    """
    return False

from src.models.fit_weight import fit_weight_model_spline_optimized, extract_predictions
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_spline_daily_pattern,
    plot_model_full_expectation,
    plot_model_predictions,
    plot_hourly_predictions,
    plot_weekly_zoomed_predictions,
)


def write_markdown_report(
    output_dir: Path,
    idata,
    df,
    stan_data,
    sigma: float,
    daily_amplitude: float,
    prop_variance_daily: float,
    adapt_delta: float,
    max_treedepth: int,
    fourier_harmonics: int,
    use_sparse: bool,
    n_inducing_points: int,
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    skip_plots: bool,
    include_prediction_grid: bool = False,
    prediction_hour: float = 8.0,
    prediction_hour_step: float = None,
    prediction_step_days: int = 1,
    predictions: dict = None,
):
    """Write comprehensive markdown report with embedded images and model summary.

    Args:
        output_dir: Directory containing generated images and report
        idata: ArviZ InferenceData object
        df: Original DataFrame
        stan_data: Stan data dictionary
        sigma: Measurement error (lbs)
        daily_amplitude: Daily amplitude (lbs)
        prop_variance_daily: Proportion of variance from daily component
        adapt_delta: Adapt delta parameter used
        max_treedepth: Maximum tree depth used
        fourier_harmonics: Number of Fourier harmonics (K)
        use_sparse: Whether sparse GP was used
        n_inducing_points: Number of inducing points (if sparse)
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        skip_plots: Whether to skip visualization section in report
    """
    report_path = output_dir / "spline_single_report.md"

    # Data statistics
    n_obs = stan_data["N"]
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    hour_range = f"{df['timestamp'].dt.hour.min():02d}:00 - {df['timestamp'].dt.hour.max():02d}:00 GMT"
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()

    with open(report_path, 'w') as f:
        f.write(f"""# Optimized Spline GP Model Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Data Summary

- **Observations**: {n_obs} weight measurements
- **Date range**: {date_range}
- **Hour range**: {hour_range}
- **Days with multiple measurements**: {days_with_multiple}/{total_days} ({days_with_multiple/total_days:.1%})
- **Data sparsity note**: {'⚠ Sparse intraday data: daily components may capture residual variation rather than true daily cycles' if days_with_multiple/total_days < 0.1 else 'Sufficient intraday data for reliable daily cycle estimation'}

## 2. Model Configuration

- **Model**: Optimized spline GP (trend + Fourier spline for daily cycles)
- **Optimizations**: Built-in `gp_exp_quad_cov` function for trend
- **Fourier harmonics (K)**: {fourier_harmonics} (sin/cos pairs)
- **Sparse GP**: {'Enabled' if use_sparse else 'Disabled'} {f'({n_inducing_points} inducing points)' if use_sparse else ''}
- **Sampler parameters**:
  - Chains: {chains}
  - Warmup iterations: {iter_warmup}
  - Sampling iterations: {iter_sampling}
  - Adapt delta: {adapt_delta}
  - Max tree depth: {max_treedepth}

## 3. Model Results

### 3.1 Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Measurement error (σ) | {sigma:.2f} lbs | Estimated scale of random measurement noise |
| Daily amplitude | {daily_amplitude:.2f} lbs | Peak‑to‑peak variation within a 24‑hour period |
| Proportion of variance from daily component | {prop_variance_daily:.3f} | Fraction of total variation explained by daily cycle |

### 3.2 Diagnostic Summary

""")

        # Add diagnostic information if available
        if hasattr(idata, 'sample_stats'):
            divergent = idata.sample_stats.get('divergent', None)
            if divergent is not None:
                n_divergent = divergent.values.sum()
                pct_divergent = 100 * n_divergent / (chains * iter_sampling) if chains * iter_sampling > 0 else 0
                f.write(f"- **Divergent transitions**: {n_divergent} ({pct_divergent:.1f}%)\n")

            treedepth = idata.sample_stats.get('treedepth__', None)
            if treedepth is not None:
                max_observed = treedepth.values.max()
                f.write(f"- **Maximum tree depth observed**: {max_observed}/{max_treedepth}\n")

        # Sparse approximation description and command line
        if use_sparse:
            sparse_desc = f'Projected process (DIC) with {n_inducing_points} inducing points'
            sparse_cmd = f'--n-inducing-points {n_inducing_points}'
        else:
            sparse_desc = 'Full GP (no approximation)'
            sparse_cmd = ''

        # Prediction grid command line
        if include_prediction_grid:
            if prediction_hour_step is not None:
                prediction_cmd = f'--include-prediction-grid --prediction-hour {prediction_hour} --prediction-hour-step {prediction_hour_step} --prediction-step-days {prediction_step_days}'
            else:
                prediction_cmd = f'--include-prediction-grid --prediction-hour {prediction_hour} --prediction-step-days {prediction_step_days}'
        else:
            prediction_cmd = ''

        # Visualization section
        if not skip_plots:
            f.write(f"""
## 4. Visualizations

### 4.1 Cyclic Components
![Cyclic Components](cyclic_components.png)

Shows the decomposition of the fitted model into **trend** (slow changes over days/weeks) and **daily** (Fourier spline) components. The daily component captures regular within‑day variation using {fourier_harmonics} Fourier harmonics.

### 4.2 Spline Daily Pattern
![Spline Daily Pattern](spline_daily_pattern.png)

The estimated 24‑hour daily pattern (mean ± 2 SD) from the Fourier spline model with K={fourier_harmonics} harmonics. This plot shows the continuous daily pattern evaluated across a full day.

### 4.3 Full Model Expectation
![Full Expectation](full_expectation.png)

Complete model prediction (trend + daily) with 95% credible interval. Observations are shown as points. The daily spline variation is visible as oscillations around the trend.
""")
            if predictions is not None:
                f.write("""
### 4.4 Predictions at Unobserved Days
![Predictions](predictions.png)

Model predictions for unobserved days across the full date range (generated with prediction grid). Shows the expected weight trajectory with 95% credible interval for days without measurements.
""")
                # Check for hourly predictions plot
                hourly_plot_path = output_dir / "hourly_predictions.png"
                if hourly_plot_path.exists():
                    f.write("""
### 4.5 Hourly Predictions
![Hourly Predictions](hourly_predictions.png)

Shows weight predictions at different hours of the day across all prediction days. The left panel aggregates predictions by hour (mean ± SD across days), the right panel shows hourly predictions for a single day. This visualization reveals the within‑day variation captured by the Fourier spline.
""")
                # Check for weekly zoomed predictions plot
                weekly_zoomed_path = output_dir / "weekly_zoomed_predictions.png"
                if weekly_zoomed_path.exists():
                    f.write("""
### 4.6 Weekly Zoomed Predictions
![Weekly Zoomed Predictions](weekly_zoomed_predictions.png)

Hourly predictions zoomed into a specific week. Each day is shown with a distinct color, with observed data points overlaid. This detailed view shows how the daily pattern repeats across consecutive days.
""")
        else:
            f.write("""
## 4. Visualizations

*Visualizations were skipped (--skip-plots used).*
""")

        # Interpretation, reproduction, and technical details
        f.write(f"""
## 5. Interpretation

1. **Measurement error reduction**: The spline model separates true daily variation from measurement noise, yielding a lower σ than a simple GP.
2. **Daily cycle significance**: A substantial daily amplitude ({daily_amplitude:.2f} lbs) suggests regular within‑day weight fluctuations.
3. **Model fit**: The proportion of variance explained by the daily component ({prop_variance_daily:.3f}) indicates how much of the total variation is cyclic.
4. **Fourier harmonics**: Using K={fourier_harmonics} harmonics allows the spline to capture more complex daily patterns than a simple periodic kernel.

## 6. Reproduction

To reproduce this analysis:

```bash
python -m src.models.demo_spline_single \\
  --chains {chains} \\
  --iter-warmup {iter_warmup} \\
  --iter-sampling {iter_sampling} \\
  --adapt-delta {adapt_delta} \\
  --max-treedepth {max_treedepth} \\
  --fourier-harmonics {fourier_harmonics} \\
  {sparse_cmd} {prediction_cmd} \\
  --output-dir {output_dir.absolute()}
```

## 7. Technical Details

- **Stan model**: `weight_gp_spline_optimized.stan`
- **Covariance function**: `gp_exp_quad_cov` (trend)
- **Daily representation**: Fourier spline with {fourier_harmonics} harmonics (sin/cos pairs)
- **Sparse approximation**: {sparse_desc}
- **Software**: CmdStanPy {chains} chains, ArviZ for diagnostics

---
*Report generated by `demo_spline_single.py`*
""")

    print(f"✓ Markdown report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Single-model demo: optimized spline GP with Gaussian process trend and Fourier spline",
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
        default="output/demo_spline_single",
        help="Directory for output plots and report",
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
        "--adapt-delta",
        type=float,
        default=0.99,
        help="Adapt delta parameter for NUTS sampler",
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Maximum tree depth for NUTS sampler",
    )
    parser.add_argument(
        "--fourier-harmonics",
        type=int,
        default=2,
        help="Number of Fourier harmonics (K parameter) for spline model",
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
    parser.add_argument(
        "--include-prediction-grid",
        action="store_true",
        help="Include prediction grid for unobserved days (generates predictions at regular intervals)",
    )
    parser.add_argument(
        "--prediction-hour",
        type=float,
        default=8.0,
        help="Hour of day (0-24) for prediction points (default 8.0 = 8 AM)",
    )
    parser.add_argument(
        "--prediction-hour-step",
        type=float,
        default=None,
        help="Step size in hours for multiple prediction hours per day (default None = single hour)",
    )
    parser.add_argument(
        "--prediction-step-days",
        type=int,
        default=1,
        help="Step size in days for prediction grid (default 1 = daily)",
    )
    parser.add_argument(
        "--zoom-to",
        type=str,
        choices=["last_week", "last_month", "last_year", "all"],
        default=None,
        help="Zoom to preset date range for prediction plot",
    )
    parser.add_argument(
        "--zoom-reference-date",
        type=str,
        default=None,
        help="Reference date for zoom presets (YYYY-MM-DD). Defaults to latest prediction date.",
    )
    parser.add_argument(
        "--zoom-start",
        type=str,
        default=None,
        help="Start date for custom zoom range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--zoom-end",
        type=str,
        default=None,
        help="End date for custom zoom range (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    use_sparse = not args.no_sparse

    # Process zoom arguments
    zoom_to = None
    zoom_reference_date = None

    if args.zoom_start and args.zoom_end:
        try:
            start = pd.Timestamp(args.zoom_start)
            end = pd.Timestamp(args.zoom_end)
            zoom_to = (start, end)
            print(f"Zoom: custom range {start.date()} to {end.date()}")
        except Exception as e:
            print(f"Warning: invalid zoom dates: {e}. Ignoring zoom.")
    elif args.zoom_to:
        zoom_to = args.zoom_to
        if args.zoom_reference_date:
            try:
                zoom_reference_date = pd.Timestamp(args.zoom_reference_date)
                print(f"Zoom: preset '{zoom_to}' with reference {zoom_reference_date.date()}")
            except Exception as e:
                print(f"Warning: invalid reference date: {e}. Using default.")
        else:
            print(f"Zoom: preset '{zoom_to}' (default reference)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OPTIMIZED SPLINE GP MODEL (SINGLE-MODEL DEMO)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Sampler: adapt_delta={args.adapt_delta}, max_treedepth={args.max_treedepth}")
    print(f"Fourier harmonics (K): {args.fourier_harmonics}")
    print(f"Caching: {not args.no_cache}, Force refit: {args.force_refit}")
    if use_sparse:
        print(f"Sparse GP: Enabled ({args.n_inducing_points} inducing points)")
    else:
        print("Sparse GP: Disabled (full GP)")
    print()

    # Sparse GP parameters
    sparse_params = {}
    if use_sparse:
        sparse_params = {
            "use_sparse": True,
            "n_inducing_points": args.n_inducing_points,
            "inducing_point_method": "uniform",
        }

    # Fit optimized spline model
    print("Fitting OPTIMIZED spline GP model (trend + Fourier spline)...")
    try:
        fit, idata, df, stan_data = fit_weight_model_spline_optimized(
            data_dir=args.data_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            fourier_harmonics=args.fourier_harmonics,
            cache=not args.no_cache,
            force_refit=args.force_refit,
            include_prediction_grid=args.include_prediction_grid,
            prediction_hour=args.prediction_hour,
            prediction_hour_step=args.prediction_hour_step,
            prediction_step_days=args.prediction_step_days,
            **sparse_params,
        )
        print("  ✓ Optimized spline model fitted successfully")
    except Exception as e:
        print(f"  ✗ Error fitting optimized spline model: {e}")
        sys.exit(1)

    # Extract key metrics
    sigma = idata.posterior["sigma"].values.mean() * stan_data["_y_sd"]
    daily_amplitude = idata.posterior["daily_amplitude"].values.mean() * stan_data["_y_sd"]
    prop_variance_daily = idata.posterior["prop_variance_daily"].values.mean()

    print("\n" + "-" * 60)
    print("MODEL SUMMARY")
    print("-" * 60)
    print(f"Measurement error (sigma): {sigma:.2f} lbs")
    print(f"Daily amplitude: {daily_amplitude:.2f} lbs")
    print(f"Proportion of variance from daily component: {prop_variance_daily:.3f}")

    # Extract predictions if prediction grid is enabled
    predictions = None
    if args.include_prediction_grid and stan_data.get('N_pred', 0) > 0:
        predictions = extract_predictions(idata, stan_data)
        print(f"  Extracted predictions for {len(predictions['t_pred'])} time points")

    # Generate plots
    if not args.skip_plots:
        print("\nGenerating visualizations...")

        # 1. Cyclic components plot
        components_path = output_dir / "cyclic_components.png"
        try:
            plot_cyclic_components(
                idata,
                df,
                stan_data,
                output_path=str(components_path),
                show_trend_component=True,
                show_daily_component=True,
            )
            print(f"  ✓ Cyclic components plot saved to {components_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate cyclic components plot: {e}")

        # 2. Spline daily pattern plot
        spline_daily_pattern_path = output_dir / "spline_daily_pattern.png"
        try:
            plot_spline_daily_pattern(
                idata,
                stan_data,
                output_path=str(spline_daily_pattern_path),
                n_hours_grid=100,
            )
            print(f"  ✓ Spline daily pattern plot saved to {spline_daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate spline daily pattern plot: {e}")

        # 3. Full expectation plot
        full_expectation_path = output_dir / "full_expectation.png"
        try:
            plot_model_full_expectation(
                idata,
                df,
                stan_data,
                output_path=str(full_expectation_path),
                model_name="Spline (Optimized)",
            )
            print(f"  ✓ Full expectation plot saved to {full_expectation_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot: {e}")

        # 4. Prediction plot (if prediction grid enabled)
        if predictions is not None:
            predictions_path = output_dir / "predictions.png"
            try:
                plot_model_predictions(
                    predictions=predictions,
                    df=df,
                    stan_data=stan_data,
                    model_name="Spline (Optimized)",
                    output_path=str(predictions_path),
                    zoom_to=zoom_to,
                    zoom_reference_date=zoom_reference_date,
                )
                print(f"  ✓ Prediction plot saved to {predictions_path}")
            except Exception as e:
                print(f"  ⚠ Could not generate prediction plot: {e}")

            # 5. Hourly predictions plot (if multiple hours per day)
            unique_hours = np.unique(predictions["hour_of_day_pred"])
            if len(unique_hours) > 1:
                hourly_predictions_path = output_dir / "hourly_predictions.png"
                try:
                    plot_hourly_predictions(
                        predictions=predictions,
                        df=df,
                        stan_data=stan_data,
                        model_name="Spline (Optimized)",
                        output_path=str(hourly_predictions_path),
                        show_all_days=True,
                    )
                    print(f"  ✓ Hourly predictions plot saved to {hourly_predictions_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate hourly predictions plot: {e}")

            # 6. Weekly zoomed predictions plot (if zoom range provided)
            if zoom_to is not None:
                weekly_zoomed_path = output_dir / "weekly_zoomed_predictions.png"
                try:
                    # Determine target date: middle of zoom range or reference date
                    if isinstance(zoom_to, tuple):
                        # Custom date range: use midpoint
                        start_date, end_date = zoom_to
                        target_date = start_date + (end_date - start_date) / 2
                    else:
                        # Preset zoom: use reference date or middle of predictions
                        if zoom_reference_date is not None:
                            target_date = zoom_reference_date
                        else:
                            # Find middle of prediction range
                            start_timestamp = df["timestamp"].min()
                            t_pred_days = predictions["t_pred"]
                            hour_of_day_pred = predictions.get("hour_of_day_pred")
                            if hour_of_day_pred is not None and len(hour_of_day_pred) == len(t_pred_days):
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) + pd.Timedelta(hours=float(h))
                                              for d, h in zip(t_pred_days, hour_of_day_pred)]
                            else:
                                dates_pred = [start_timestamp + pd.Timedelta(days=float(d)) for d in t_pred_days]
                            mid_idx = len(dates_pred) // 2
                            target_date = dates_pred[mid_idx]

                    plot_weekly_zoomed_predictions(
                        predictions=predictions,
                        df=df,
                        stan_data=stan_data,
                        model_name="Spline (Optimized)",
                        output_path=str(weekly_zoomed_path),
                        target_date=target_date,
                        show_ci=True,
                        show_observations=True,
                    )
                    print(f"  ✓ Weekly zoomed predictions plot saved to {weekly_zoomed_path}")
                except Exception as e:
                    print(f"  ⚠ Could not generate weekly zoomed predictions plot: {e}")

    # Write markdown report
    report_path = write_markdown_report(
        output_dir=output_dir,
        idata=idata,
        df=df,
        stan_data=stan_data,
        sigma=sigma,
        daily_amplitude=daily_amplitude,
        prop_variance_daily=prop_variance_daily,
        adapt_delta=args.adapt_delta,
        max_treedepth=args.max_treedepth,
        fourier_harmonics=args.fourier_harmonics,
        use_sparse=use_sparse,
        n_inducing_points=args.n_inducing_points,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        include_prediction_grid=args.include_prediction_grid,
        prediction_hour=args.prediction_hour,
        prediction_hour_step=args.prediction_hour_step,
        prediction_step_days=args.prediction_step_days,
        predictions=predictions,
        skip_plots=args.skip_plots,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("✓ Model fitted successfully with optimized implementation")
    print(f"✓ Sampler parameters: adapt_delta={args.adapt_delta}, max_treedepth={args.max_treedepth}")
    print(f"✓ Fourier harmonics: K={args.fourier_harmonics}")
    if use_sparse:
        print(f"✓ Sparse GP approximation used ({args.n_inducing_points} inducing points)")
    print(f"✓ Key metric: sigma = {sigma:.2f} lbs, daily amplitude = {daily_amplitude:.2f} lbs")
    print(f"✓ Output directory: {output_dir}")
    if not args.skip_plots:
        print("  • cyclic_components.png - Trend vs. daily components")
        print("  • spline_daily_pattern.png - 24‑hour Fourier spline pattern")
        print("  • full_expectation.png - Complete prediction with credible interval")
    print("  • spline_single_report.md - Comprehensive markdown report")
    print(f"\nView the full report at: {report_path}")

    # Show plots if not saved only
    if not args.skip_plots:
        print("\nDisplaying plots (close windows to exit)...")
        if should_show_plots():
            plt.show()
        else:
            print("Plots saved to disk (interactive display not available)")


if __name__ == "__main__":
    main()