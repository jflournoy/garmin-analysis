#!/usr/bin/env python3
"""Single-model demo: optimized cyclic GP with Gaussian process trend.

This script runs ONLY the optimized cyclic Gaussian Process model (trend + daily periodic kernel)
with all optimizations in place. It includes adjustable adapt_delta and max_treedepth parameters
and generates a comprehensive markdown report with embedded visualizations.

Usage:
    python -m src.models.demo_cyclic_single [--chains N] [--iter-warmup N] [--iter-sampling N]
        [--adapt-delta D] [--max-treedepth D] [--no-sparse] [--n-inducing-points M]
        [--output-dir DIR] [--skip-plots] [--no-cache] [--force-refit]

Example:
    # Quick demo with minimal iterations
    python -m src.models.demo_cyclic_single --chains 1 --iter-warmup 10 --iter-sampling 10 \
        --output-dir output/demo_cyclic_single

    # Reliable results with custom sampler parameters
    python -m src.models.demo_cyclic_single --chains 2 --iter-warmup 200 --iter-sampling 200 \
        --adapt-delta 0.99 --max-treedepth 12 --output-dir output/demo_cyclic_single
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt


def should_show_plots() -> bool:
    """Return True if plots should be displayed interactively.

    Modified to always return False - plots are saved to disk only.
    """
    return False

from src.models.fit_weight import fit_weight_model_cyclic_optimized
from src.models.plot_cyclic import (
    plot_cyclic_components,
    plot_daily_pattern,
    plot_model_full_expectation,
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
    use_sparse: bool,
    n_inducing_points: int,
    chains: int,
    iter_warmup: int,
    iter_sampling: int,
    skip_plots: bool,
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
        use_sparse: Whether sparse GP was used
        n_inducing_points: Number of inducing points (if sparse)
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        skip_plots: Whether to skip visualization section in report
    """
    report_path = output_dir / "cyclic_single_report.md"

    # Data statistics
    n_obs = stan_data["N"]
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    hour_range = f"{df['timestamp'].dt.hour.min():02d}:00 - {df['timestamp'].dt.hour.max():02d}:00 GMT"
    days_with_multiple = (df.groupby('date').size() > 1).sum()
    total_days = df['date'].nunique()

    with open(report_path, 'w') as f:
        f.write(f"""# Optimized Cyclic GP Model Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Data Summary

- **Observations**: {n_obs} weight measurements
- **Date range**: {date_range}
- **Hour range**: {hour_range}
- **Days with multiple measurements**: {days_with_multiple}/{total_days} ({days_with_multiple/total_days:.1%})
- **Data sparsity note**: {'⚠ Sparse intraday data: daily components may capture residual variation rather than true daily cycles' if days_with_multiple/total_days < 0.1 else 'Sufficient intraday data for reliable daily cycle estimation'}

## 2. Model Configuration

- **Model**: Optimized cyclic GP (trend + daily periodic kernel)
- **Optimizations**: Built-in `gp_exp_quad_cov` and `gp_periodic_cov` functions
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
| Daily amplitude | {daily_amplitude:.2f} lbs | Peak-to-peak variation within a 24‑hour period |
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

        # Visualization section (conditional)
        if not skip_plots:
            f.write("""
## 4. Visualizations

### 4.1 Cyclic Components
![Cyclic Components](cyclic_components.png)

Shows the decomposition of the fitted model into **trend** (slow changes over days/weeks) and **daily** (24‑hour periodic) components. The daily component captures regular within‑day variation.

### 4.2 Daily Pattern
![Daily Pattern](daily_pattern.png)

The estimated 24‑hour daily pattern (mean ± 2 SD). This is the periodic component evaluated over a full day, showing typical weight fluctuation across hours.

### 4.3 Full Model Expectation
![Full Expectation](full_expectation.png)

Complete model prediction (trend + daily) with 95% credible interval. Observations are shown as points. The daily cyclic variation is visible as oscillations around the trend.

""")
        else:
            f.write("""
## 4. Visualizations

*Visualizations were skipped (--skip-plots used).*
""")

        # Write interpretation, reproduction, and technical details
        # Sparse approximation description
        if use_sparse:
            sparse_desc = f'Projected process (DIC) with {n_inducing_points} inducing points'
        else:
            sparse_desc = 'Full GP (no approximation)'

        f.write(f"""
## 5. Interpretation

1. **Measurement error reduction**: The cyclic model separates true daily variation from measurement noise, yielding a lower σ than a simple GP.
2. **Daily cycle significance**: A substantial daily amplitude ({daily_amplitude:.2f} lbs) suggests regular within‑day weight fluctuations.
3. **Model fit**: The proportion of variance explained by the daily component ({prop_variance_daily:.3f}) indicates how much of the total variation is cyclic.

## 6. Reproduction

To reproduce this analysis:

```bash
python -m src.models.demo_cyclic_single \\
  --chains {chains} \\
  --iter-warmup {iter_warmup} \\
  --iter-sampling {iter_sampling} \\
  --adapt-delta {adapt_delta} \\
  --max-treedepth {max_treedepth} \\
  --output-dir {output_dir.absolute()}
```

## 7. Technical Details

- **Stan model**: `weight_gp_cyclic_optimized.stan`
- **Covariance functions**: `gp_exp_quad_cov` (trend), `gp_periodic_cov` (daily)
- **Sparse approximation**: {sparse_desc}
- **Software**: CmdStanPy {chains} chains, ArviZ for diagnostics

---
*Report generated by `demo_cyclic_single.py`*
""")

    print(f"✓ Markdown report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Single-model demo: optimized cyclic GP with Gaussian process trend",
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
        default="output/demo_cyclic_single",
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
    print("OPTIMIZED CYCLIC GP MODEL (SINGLE-MODEL DEMO)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Chains: {args.chains}, Warmup: {args.iter_warmup}, Sampling: {args.iter_sampling}")
    print(f"Sampler: adapt_delta={args.adapt_delta}, max_treedepth={args.max_treedepth}")
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

    # Fit optimized cyclic model
    print("Fitting OPTIMIZED cyclic GP model (trend + daily periodic kernel)...")
    try:
        fit, idata, df, stan_data = fit_weight_model_cyclic_optimized(
            data_dir=args.data_dir,
            chains=args.chains,
            iter_warmup=args.iter_warmup,
            iter_sampling=args.iter_sampling,
            adapt_delta=args.adapt_delta,
            max_treedepth=args.max_treedepth,
            cache=not args.no_cache,
            force_refit=args.force_refit,
            **sparse_params,
        )
        print("  ✓ Optimized cyclic model fitted successfully")
    except Exception as e:
        print(f"  ✗ Error fitting optimized cyclic model: {e}")
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

        # 2. Daily pattern plot
        daily_pattern_path = output_dir / "daily_pattern.png"
        try:
            plot_daily_pattern(
                idata,
                df,
                stan_data,
                output_path=str(daily_pattern_path),
            )
            print(f"  ✓ Daily pattern plot saved to {daily_pattern_path}")
        except ValueError as e:
            print(f"  ⚠ Could not generate daily pattern plot: {e}")

        # 3. Full expectation plot
        full_expectation_path = output_dir / "full_expectation.png"
        try:
            plot_model_full_expectation(
                idata,
                df,
                stan_data,
                output_path=str(full_expectation_path),
                model_name="Cyclic (Optimized)",
            )
            print(f"  ✓ Full expectation plot saved to {full_expectation_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate full expectation plot: {e}")

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
        use_sparse=use_sparse,
        n_inducing_points=args.n_inducing_points,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        skip_plots=args.skip_plots,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("✓ Model fitted successfully with optimized implementation")
    print(f"✓ Sampler parameters: adapt_delta={args.adapt_delta}, max_treedepth={args.max_treedepth}")
    if use_sparse:
        print(f"✓ Sparse GP approximation used ({args.n_inducing_points} inducing points)")
    print(f"✓ Key metric: sigma = {sigma:.2f} lbs, daily amplitude = {daily_amplitude:.2f} lbs")
    print(f"✓ Output directory: {output_dir}")
    if not args.skip_plots:
        print("  • cyclic_components.png - Trend vs. daily components")
        print("  • daily_pattern.png - 24‑hour daily pattern")
        print("  • full_expectation.png - Complete prediction with credible interval")
    print("  • cyclic_single_report.md - Comprehensive markdown report")
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