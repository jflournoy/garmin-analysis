#!/usr/bin/env python3
"""Demo script for cross-lagged Gaussian Process models.

This script runs cross-lagged GP models for analyzing causal relationships
between weight and workout/activity variables with time lags.

Supported model types:
1. Fixed lag: Lag τ specified as data parameter (compare multiple τ values)
2. Estimated lag: Lag τ estimated as parameter with prior distribution
3. Cumulative lag: Average effect over a window of lag values

Usage:
    # Fixed lag comparison across multiple lag values
    python -m src.models.demo_bivariate --model-type fixed \
        --workout-vars strength_training,cardio \
        --lags 0,1,2,3,7 \
        --output-dir output/demo_bivariate_fixed

    # Estimated lag model
    python -m src.models.demo_bivariate --model-type estimated \
        --workout-vars strength_training \
        --lag-prior-mean 2.0 --lag-prior-sd 1.0 \
        --output-dir output/demo_bivariate_estimated

    # Cumulative lag model
    python -m src.models.demo_bivariate --model-type cumulative \
        --workout-vars strength_training \
        --lag-window 7 --lag-step 1 \
        --output-dir output/demo_bivariate_cumulative

Example (quick test):
    python -m src.models.demo_bivariate --model-type fixed \
        --workout-vars strength_training \
        --lags 0,2 \
        --chains 2 --iter-warmup 100 --iter-sampling 100 \
        --output-dir output/test
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from src.models.fit_weight import (
    fit_crosslagged_model,
    fit_crosslagged_model_estimated,
    fit_crosslagged_model_cumulative,
)
from src.data import load_weight_data
from src.data.workout import load_workout_data, prepare_workout_aggregates


def should_show_plots() -> bool:
    """Return True if plots should be displayed interactively.

    Modified to always return False - plots are saved to disk only.
    """
    return False


def parse_comma_list(value: str) -> List[str]:
    """Parse comma-separated list string into list of strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_comma_float_list(value: str) -> List[float]:
    """Parse comma-separated list of floats."""
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _compute_ess(da) -> float:
    """Compute effective sample size from DataArray."""
    try:
        ess = az.ess(da)
        if ess is None:
            return np.nan
        # Extract scalar if possible
        val = ess.values
        return float(val.item() if val.size == 1 else val.mean())
    except Exception:
        return np.nan


def _compute_rhat(da) -> float:
    """Compute R-hat from DataArray."""
    try:
        rhat = az.rhat(da)
        if rhat is None:
            return np.nan
        val = rhat.values
        return float(val.item() if val.size == 1 else val.mean())
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for cross-lagged Gaussian Process models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["fixed", "estimated", "cumulative"],
        default="fixed",
        help="Type of cross-lagged model: 'fixed' (compare lag values), "
             "'estimated' (estimate lag parameter), 'cumulative' (average over lag window)"
    )

    parser.add_argument(
        "--workout-vars",
        type=str,
        required=True,
        help="Comma-separated list of workout/activity variables to analyze. "
             "Examples: 'strength_training,cardio,total_active_minutes'"
    )

    # Fixed lag model arguments
    parser.add_argument(
        "--lags",
        type=str,
        default="0,1,2,3,7",
        help="Comma-separated list of lag values in days (for fixed lag model only)"
    )

    # Estimated lag model arguments
    parser.add_argument(
        "--lag-prior-mean",
        type=float,
        default=2.0,
        help="Prior mean for lag parameter in days (for estimated lag model only)"
    )
    parser.add_argument(
        "--lag-prior-sd",
        type=float,
        default=1.0,
        help="Prior standard deviation for lag parameter in days (for estimated lag model only)"
    )

    # Cumulative lag model arguments
    parser.add_argument(
        "--lag-window",
        type=int,
        default=7,
        help="Window size for cumulative lag effect in days (for cumulative model only)"
    )
    parser.add_argument(
        "--lag-step",
        type=float,
        default=1.0,
        help="Step between lags in cumulative window in days (for cumulative model only)"
    )

    # Data and output
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/demo_bivariate",
        help="Directory for output plots and reports"
    )

    # MCMC parameters
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains"
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Warmup iterations per chain"
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Sampling iterations per chain"
    )
    parser.add_argument(
        "--adapt-delta",
        type=float,
        default=0.99,
        help="Adapt delta parameter for NUTS sampler"
    )
    parser.add_argument(
        "--max-treedepth",
        type=int,
        default=12,
        help="Maximum tree depth for NUTS sampler"
    )

    # Sparse GP configuration
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation (sparse GP is default)"
    )
    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP (unless --no-sparse)"
    )

    # Caching
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached results exist"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force refit)"
    )

    # Prediction grid
    parser.add_argument(
        "--include-prediction-grid",
        action="store_true",
        help="Include prediction grid for unobserved time points"
    )
    parser.add_argument(
        "--prediction-step-days",
        type=float,
        default=1.0,
        help="Step size in days for prediction grid"
    )

    # Visualization
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (only print summary)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Parse comma-separated lists
    workout_vars = parse_comma_list(args.workout_vars)
    if not workout_vars:
        parser.error("--workout-vars must contain at least one variable")

    lag_values = parse_comma_float_list(args.lags) if args.model_type == "fixed" else []

    # Validate arguments based on model type
    if args.model_type == "fixed" and not lag_values:
        parser.error("--lags must contain at least one lag value for fixed lag model")
    elif args.model_type == "estimated":
        if args.lag_prior_mean <= 0:
            parser.error("--lag-prior-mean must be positive")
        if args.lag_prior_sd <= 0:
            parser.error("--lag-prior-sd must be positive")
    elif args.model_type == "cumulative":
        if args.lag_window <= 0:
            parser.error("--lag-window must be positive")
        if args.lag_step <= 0:
            parser.error("--lag-step must be positive")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up sparse GP parameters
    use_sparse = not args.no_sparse
    sparse_params = {
        "use_sparse": use_sparse,
        "n_inducing_points": args.n_inducing_points if use_sparse else 0,
    }

    # Set up caching
    cache = not args.no_cache
    force_refit = args.force_refit or args.no_cache

    # Print configuration
    print("=" * 70)
    print("CROSS-LAGGED GP MODEL DEMO")
    print("=" * 70)
    print(f"Model type: {args.model_type}")
    print(f"Workout variables: {', '.join(workout_vars)}")
    print(f"Output directory: {output_dir}")
    print(f"MCMC: {args.chains} chains, {args.iter_warmup} warmup, {args.iter_sampling} sampling")
    print(f"Sparse GP: {use_sparse} ({args.n_inducing_points} inducing points)" if use_sparse else "Sparse GP: disabled")
    print(f"Caching: {cache} (force refit: {force_refit})")

    if args.model_type == "fixed":
        print(f"Lag values: {lag_values} days")
    elif args.model_type == "estimated":
        print(f"Lag prior: N({args.lag_prior_mean}, {args.lag_prior_sd}²) days")
    elif args.model_type == "cumulative":
        print(f"Cumulative lag window: {args.lag_window} days (step: {args.lag_step} days)")

    # Dispatch based on model type
    if args.model_type == "fixed":
        results = run_fixed_lag_comparison(
            args, workout_vars, lag_values, output_dir, args.skip_plots
        )
        create_comparison_report(results, output_dir, "fixed")

    elif args.model_type == "estimated":
        results = run_estimated_lag_analysis(
            args, workout_vars, output_dir, args.skip_plots
        )
        create_comparison_report(results, output_dir, "estimated")

    elif args.model_type == "cumulative":
        results = run_cumulative_lag_analysis(
            args, workout_vars, output_dir, args.skip_plots
        )
        create_comparison_report(results, output_dir, "cumulative")

    print(f"\nDemo complete. Results saved to {output_dir}")


def run_fixed_lag_comparison(
    args,
    workout_vars: List[str],
    lag_values: List[float],
    output_dir: Path,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run fixed lag model comparison across multiple lag values and variables."""
    print("\n" + "=" * 70)
    print("FIXED LAG MODEL COMPARISON")
    print("=" * 70)

    # Load weight data (common for all variables)
    print("\n1. Loading weight data...")
    df_weight = load_weight_data(args.data_dir)
    print(f"   Weight measurements: {len(df_weight)}")

    # Set up sparse GP parameters
    use_sparse = not args.no_sparse
    sparse_params = {
        "use_sparse": use_sparse,
        "n_inducing_points": args.n_inducing_points if use_sparse else 0,
    }

    # Set up caching
    cache = not args.no_cache
    force_refit = args.force_refit or args.no_cache

    results = {}

    for var_name in workout_vars:
        print(f"\n{'='*60}")
        print(f"Analyzing variable: {var_name}")
        print(f"{'='*60}")

        # Load workout data for this variable
        print(f"2. Loading workout data for '{var_name}'...")
        df_workouts_raw = load_workout_data(
            data_dir=args.data_dir,
            activity_type=var_name,
            include_exercise_details=False,
        )
        print(f"   Raw workout records: {len(df_workouts_raw)}")

        # Aggregate workouts to daily count
        df_workouts_agg = prepare_workout_aggregates(
            df_workouts_raw,
            aggregation="daily",
            metric="count"
        )
        print(f"   Workout days (with workouts): {len(df_workouts_agg)}")
        if len(df_workouts_agg) == 0:
            print(f"   WARNING: No workout data for '{var_name}'. Skipping.")
            continue

        var_results = []
        var_output_dir = output_dir / var_name
        var_output_dir.mkdir(parents=True, exist_ok=True)

        for lag_days in lag_values:
            print(f"\n{'='*40}")
            print(f"  Lag τ = {lag_days} days")
            print(f"{'='*40}")

            try:
                # Fit model
                fit, idata, stan_data = fit_crosslagged_model(
                    df_weight=df_weight,
                    df_workout=df_workouts_agg,
                    workout_value_col="workout_count",
                    lag_days=lag_days,
                    chains=args.chains,
                    iter_warmup=args.iter_warmup,
                    iter_sampling=args.iter_sampling,
                    use_sparse=use_sparse,
                    n_inducing_points=args.n_inducing_points,
                    inducing_point_method="uniform",
                    cache=cache,
                    force_refit=force_refit,
                    include_prediction_grid=args.include_prediction_grid,
                    prediction_step_days=args.prediction_step_days,
                )

                # Extract posterior summary for beta
                posterior = idata.posterior
                if 'beta' in posterior:
                    beta_samples = posterior['beta'].values.flatten()
                    beta_mean = np.mean(beta_samples)
                    beta_std = np.std(beta_samples)
                    beta_ci_low = np.percentile(beta_samples, 2.5)
                    beta_ci_high = np.percentile(beta_samples, 97.5)

                    # Convert to original units if available
                    if 'beta_original_units' in posterior:
                        beta_orig_samples = posterior['beta_original_units'].values.flatten()
                        beta_orig_mean = np.mean(beta_orig_samples)
                        beta_orig_std = np.std(beta_orig_samples)
                        beta_orig_ci_low = np.percentile(beta_orig_samples, 2.5)
                        beta_orig_ci_high = np.percentile(beta_orig_samples, 97.5)
                    else:
                        beta_orig_mean = beta_orig_std = beta_orig_ci_low = beta_orig_ci_high = np.nan

                    # Compute WAIC and LOO (weight likelihood only)
                    try:
                        waic = az.waic(idata, var_name="log_lik_weight")
                        loo = az.loo(idata, var_name="log_lik_weight")
                        waic_value = waic.waic
                        loo_value = loo.loo
                    except Exception as e:
                        print(f"     Warning: Could not compute WAIC/LOO: {e}")
                        waic_value = loo_value = np.nan

                    # Store results
                    result = {
                        'variable': var_name,
                        'lag_days': lag_days,
                        'beta_mean': beta_mean,
                        'beta_std': beta_std,
                        'beta_ci_low': beta_ci_low,
                        'beta_ci_high': beta_ci_high,
                        'beta_orig_mean': beta_orig_mean,
                        'beta_orig_std': beta_orig_std,
                        'beta_orig_ci_low': beta_orig_ci_low,
                        'beta_orig_ci_high': beta_orig_ci_high,
                        'waic': waic_value,
                        'loo': loo_value,
                        'n_eff_beta': _compute_ess(posterior['beta']) if 'beta' in posterior else np.nan,
                        'rhat_beta': _compute_rhat(posterior['beta']) if 'beta' in posterior else np.nan,
                        'n_weight_obs': len(df_weight),
                        'n_workout_obs': len(df_workouts_agg),
                    }
                    var_results.append(result)

                    print(f"     β (standardized): {beta_mean:.3f} [{beta_ci_low:.3f}, {beta_ci_high:.3f}]")
                    if not np.isnan(beta_orig_mean):
                        print(f"     β (original units): {beta_orig_mean:.3f} [{beta_orig_ci_low:.3f}, {beta_orig_ci_high:.3f}]")
                    print(f"     WAIC: {waic_value:.1f}" if not np.isnan(waic_value) else "     WAIC: N/A")
                    print(f"     LOO: {loo_value:.1f}" if not np.isnan(loo_value) else "     LOO: N/A")

                    # Save model results
                    lag_dir = var_output_dir / f"lag_{lag_days}days"
                    lag_dir.mkdir(exist_ok=True)

                    # Save posterior summary
                    summary = az.summary(idata, round_to=3)
                    summary.to_csv(lag_dir / "posterior_summary.csv")

                    # Save beta posterior samples
                    beta_df = pd.DataFrame({'beta': beta_samples})
                    beta_df.to_csv(lag_dir / "beta_posterior.csv", index=False)

                    # Plot beta posterior if not skipping plots
                    if not skip_plots:
                        plt.figure(figsize=(8, 5))
                        plt.hist(beta_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
                        plt.axvline(0, color='red', linestyle='--', label='Zero effect')
                        plt.xlabel('β (causal effect: workouts → weight)')
                        plt.ylabel('Density')
                        plt.title(f'Posterior of β ({var_name}, lag τ={lag_days} days)')
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(lag_dir / "beta_posterior.png", dpi=150)
                        plt.close()
                        print(f"     Saved plot: {lag_dir / 'beta_posterior.png'}")

                    print(f"     Saved results to {lag_dir}")

                else:
                    print("     Warning: 'beta' not found in posterior")
                    var_results.append({
                        'variable': var_name,
                        'lag_days': lag_days,
                        'error': 'beta not found'
                    })

            except Exception as e:
                print(f"     Error fitting model with lag {lag_days}: {e}")
                import traceback
                traceback.print_exc()
                var_results.append({
                    'variable': var_name,
                    'lag_days': lag_days,
                    'error': str(e)
                })

        # Create summary table for this variable
        if var_results and any('beta_mean' in r for r in var_results):
            valid_results = [r for r in var_results if 'beta_mean' in r]
            if valid_results:
                df_var_results = pd.DataFrame(valid_results)
                # Reorder columns
                cols = ['lag_days', 'beta_mean', 'beta_std', 'beta_ci_low', 'beta_ci_high',
                        'beta_orig_mean', 'beta_orig_std', 'beta_orig_ci_low', 'beta_orig_ci_high',
                        'waic', 'loo', 'n_eff_beta', 'rhat_beta', 'n_weight_obs', 'n_workout_obs']
                df_var_results = df_var_results[[c for c in cols if c in df_var_results.columns]]
                df_var_results.to_csv(var_output_dir / "lag_comparison.csv", index=False)

                # Plot β vs lag for this variable if not skipping plots
                if not skip_plots and len(valid_results) > 1:
                    plt.figure(figsize=(10, 6))
                    lags_vals = [r['lag_days'] for r in valid_results]
                    beta_means = [r['beta_mean'] for r in valid_results]
                    beta_cis_low = [r['beta_ci_low'] for r in valid_results]
                    beta_cis_high = [r['beta_ci_high'] for r in valid_results]

                    plt.errorbar(lags_vals, beta_means,
                                yerr=[np.array(beta_means) - np.array(beta_cis_low),
                                      np.array(beta_cis_high) - np.array(beta_means)],
                                fmt='o-', capsize=5, capthick=2)
                    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    plt.xlabel('Lag τ (days)')
                    plt.ylabel('β (causal effect)')
                    plt.title(f'Causal effect of {var_name} on weight across different lags')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(var_output_dir / "beta_vs_lag.png", dpi=150)
                    plt.close()
                    print(f"   Saved comparison plot: {var_output_dir / 'beta_vs_lag.png'}")

                # Find best lag by WAIC (if available)
                if 'waic' in df_var_results.columns and not df_var_results['waic'].isna().all():
                    best_idx = df_var_results['waic'].idxmin()
                    best_lag = df_var_results.loc[best_idx, 'lag_days']
                    best_waic = df_var_results.loc[best_idx, 'waic']
                    print(f"\n   Best lag for {var_name} by WAIC: τ = {best_lag} days (WAIC = {best_waic:.1f})")

        results[var_name] = var_results

    # Create overall comparison across variables
    if results:
        all_valid_results = []
        for var_name, var_results in results.items():
            for r in var_results:
                if 'beta_mean' in r:
                    all_valid_results.append(r)

        if all_valid_results:
            df_all = pd.DataFrame(all_valid_results)
            df_all.to_csv(output_dir / "overall_comparison.csv", index=False)

            # Create combined plot across variables if not skipping plots
            if not skip_plots and len(workout_vars) > 1:
                plt.figure(figsize=(12, 8))
                colors = plt.cm.tab10(np.linspace(0, 1, len(workout_vars)))

                for idx, var_name in enumerate(workout_vars):
                    var_results = results.get(var_name, [])
                    valid_var_results = [r for r in var_results if 'beta_mean' in r]
                    if not valid_var_results:
                        continue

                    lags_vals = [r['lag_days'] for r in valid_var_results]
                    beta_means = [r['beta_mean'] for r in valid_var_results]
                    beta_cis_low = [r['beta_ci_low'] for r in valid_var_results]
                    beta_cis_high = [r['beta_ci_high'] for r in valid_var_results]

                    plt.errorbar(lags_vals, beta_means,
                                yerr=[np.array(beta_means) - np.array(beta_cis_low),
                                      np.array(beta_cis_high) - np.array(beta_means)],
                                fmt='o-', capsize=5, capthick=2, color=colors[idx],
                                label=var_name, alpha=0.8)

                plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                plt.xlabel('Lag τ (days)')
                plt.ylabel('β (causal effect)')
                plt.title('Causal effect of different workout types on weight across lags')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "beta_vs_lag_all_variables.png", dpi=150)
                plt.close()
                print(f"\nSaved combined plot: {output_dir / 'beta_vs_lag_all_variables.png'}")

    return results


def run_estimated_lag_analysis(
    args,
    workout_vars: List[str],
    output_dir: Path,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run estimated lag model analysis for each variable."""
    print("\n" + "=" * 70)
    print("ESTIMATED LAG MODEL ANALYSIS")
    print("=" * 70)

    # Load weight data (common for all variables)
    print("\n1. Loading weight data...")
    df_weight = load_weight_data(args.data_dir)
    print(f"   Weight measurements: {len(df_weight)}")

    # Set up sparse GP parameters
    use_sparse = not args.no_sparse
    sparse_params = {
        "use_sparse": use_sparse,
        "n_inducing_points": args.n_inducing_points if use_sparse else 0,
    }

    # Set up caching
    cache = not args.no_cache
    force_refit = args.force_refit or args.no_cache

    results = {}

    for var_name in workout_vars:
        print(f"\n{'='*60}")
        print(f"Analyzing variable: {var_name}")
        print(f"{'='*60}")

        # Load workout data for this variable
        print(f"2. Loading workout data for '{var_name}'...")
        df_workouts_raw = load_workout_data(
            data_dir=args.data_dir,
            activity_type=var_name,
            include_exercise_details=False,
        )
        print(f"   Raw workout records: {len(df_workouts_raw)}")

        # Aggregate workouts to daily count
        df_workouts_agg = prepare_workout_aggregates(
            df_workouts_raw,
            aggregation="daily",
            metric="count"
        )
        print(f"   Workout days (with workouts): {len(df_workouts_agg)}")
        if len(df_workouts_agg) == 0:
            print(f"   WARNING: No workout data for '{var_name}'. Skipping.")
            continue

        var_output_dir = output_dir / var_name
        var_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Fit estimated lag model
            print(f"3. Fitting estimated lag model for '{var_name}'...")
            fit, idata, stan_data = fit_crosslagged_model_estimated(
                df_weight=df_weight,
                df_workout=df_workouts_agg,
                workout_value_col="workout_count",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                use_sparse=use_sparse,
                n_inducing_points=args.n_inducing_points,
                inducing_point_method="uniform",
                cache=cache,
                force_refit=force_refit,
                include_prediction_grid=args.include_prediction_grid,
                prediction_step_days=args.prediction_step_days,
            )

            # Extract posterior summaries
            posterior = idata.posterior
            result = {
                'variable': var_name,
                'n_weight_obs': len(df_weight),
                'n_workout_obs': len(df_workouts_agg),
            }

            # Extract beta (causal effect)
            if 'beta' in posterior:
                beta_samples = posterior['beta'].values.flatten()
                result['beta_mean'] = np.mean(beta_samples)
                result['beta_std'] = np.std(beta_samples)
                result['beta_ci_low'] = np.percentile(beta_samples, 2.5)
                result['beta_ci_high'] = np.percentile(beta_samples, 97.5)
                result['n_eff_beta'] = _compute_ess(posterior['beta'])
                result['rhat_beta'] = _compute_rhat(posterior['beta'])
                print(f"   β (standardized): {result['beta_mean']:.3f} [{result['beta_ci_low']:.3f}, {result['beta_ci_high']:.3f}]")
            else:
                print("   Warning: 'beta' not found in posterior")

            # Extract lag_days parameter
            if 'lag_days' in posterior:
                lag_samples = posterior['lag_days'].values.flatten()
                result['lag_days_mean'] = np.mean(lag_samples)
                result['lag_days_std'] = np.std(lag_samples)
                result['lag_days_ci_low'] = np.percentile(lag_samples, 2.5)
                result['lag_days_ci_high'] = np.percentile(lag_samples, 97.5)
                result['n_eff_lag'] = _compute_ess(posterior['lag_days'])
                result['rhat_lag'] = _compute_rhat(posterior['lag_days'])
                print(f"   Lag τ (days): {result['lag_days_mean']:.2f} [{result['lag_days_ci_low']:.2f}, {result['lag_days_ci_high']:.2f}]")
            else:
                print("   Warning: 'lag_days' not found in posterior")

            # Extract beta in original units if available
            if 'beta_original_units' in posterior:
                beta_orig_samples = posterior['beta_original_units'].values.flatten()
                result['beta_orig_mean'] = np.mean(beta_orig_samples)
                result['beta_orig_std'] = np.std(beta_orig_samples)
                result['beta_orig_ci_low'] = np.percentile(beta_orig_samples, 2.5)
                result['beta_orig_ci_high'] = np.percentile(beta_orig_samples, 97.5)
                print(f"   β (original units): {result['beta_orig_mean']:.3f} [{result['beta_orig_ci_low']:.3f}, {result['beta_orig_ci_high']:.3f}]")

            # Compute WAIC and LOO (weight likelihood only)
            try:
                waic = az.waic(idata, var_name="log_lik_weight")
                loo = az.loo(idata, var_name="log_lik_weight")
                result['waic'] = waic.waic
                result['loo'] = loo.loo
                print(f"   WAIC: {result['waic']:.1f}")
                print(f"   LOO: {result['loo']:.1f}")
            except Exception as e:
                print(f"   Warning: Could not compute WAIC/LOO: {e}")
                result['waic'] = np.nan
                result['loo'] = np.nan

            # Save posterior summary
            summary = az.summary(idata, round_to=3)
            summary.to_csv(var_output_dir / "posterior_summary.csv")

            # Save beta and lag posterior samples
            if 'beta' in posterior:
                beta_df = pd.DataFrame({'beta': beta_samples})
                beta_df.to_csv(var_output_dir / "beta_posterior.csv", index=False)
            if 'lag_days' in posterior:
                lag_df = pd.DataFrame({'lag_days': lag_samples})
                lag_df.to_csv(var_output_dir / "lag_days_posterior.csv", index=False)

            # Generate plots if not skipping
            if not skip_plots:
                # Beta posterior plot
                if 'beta' in posterior:
                    plt.figure(figsize=(8, 5))
                    plt.hist(beta_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
                    plt.axvline(0, color='red', linestyle='--', label='Zero effect')
                    plt.xlabel('β (causal effect: workouts → weight)')
                    plt.ylabel('Density')
                    plt.title(f'Posterior of β ({var_name}, estimated lag)')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(var_output_dir / "beta_posterior.png", dpi=150)
                    plt.close()
                    print(f"   Saved beta posterior plot: {var_output_dir / 'beta_posterior.png'}")

                # Lag posterior plot
                if 'lag_days' in posterior:
                    plt.figure(figsize=(8, 5))
                    plt.hist(lag_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
                    plt.axvline(3.0, color='red', linestyle='--', label='Prior mean (3 days)')
                    plt.xlabel('Lag τ (days)')
                    plt.ylabel('Density')
                    plt.title(f'Posterior of lag τ ({var_name})')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(var_output_dir / "lag_days_posterior.png", dpi=150)
                    plt.close()
                    print(f"   Saved lag posterior plot: {var_output_dir / 'lag_days_posterior.png'}")

                # Joint posterior scatter (beta vs lag)
                if 'beta' in posterior and 'lag_days' in posterior:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(lag_samples, beta_samples, alpha=0.5, s=10)
                    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
                    plt.axvline(3.0, color='red', linestyle='--', alpha=0.5, label='Prior mean lag')
                    plt.xlabel('Lag τ (days)')
                    plt.ylabel('β (causal effect)')
                    plt.title(f'Joint posterior: β vs lag τ ({var_name})')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(var_output_dir / "beta_vs_lag_scatter.png", dpi=150)
                    plt.close()
                    print(f"   Saved joint posterior plot: {var_output_dir / 'beta_vs_lag_scatter.png'}")

            results[var_name] = result
            print(f"   Results saved to {var_output_dir}")

        except Exception as e:
            print(f"   Error fitting estimated lag model for {var_name}: {e}")
            import traceback
            traceback.print_exc()
            results[var_name] = {'variable': var_name, 'error': str(e)}

    # Create summary table across variables
    if results:
        valid_results = []
        for var_name, result in results.items():
            if 'beta_mean' in result:
                valid_results.append(result)

        if valid_results:
            df_results = pd.DataFrame(valid_results)
            # Reorder columns
            cols = ['variable', 'beta_mean', 'beta_std', 'beta_ci_low', 'beta_ci_high',
                    'beta_orig_mean', 'beta_orig_std', 'beta_orig_ci_low', 'beta_orig_ci_high',
                    'lag_days_mean', 'lag_days_std', 'lag_days_ci_low', 'lag_days_ci_high',
                    'waic', 'loo', 'n_eff_beta', 'rhat_beta', 'n_eff_lag', 'rhat_lag',
                    'n_weight_obs', 'n_workout_obs']
            df_results = df_results[[c for c in cols if c in df_results.columns]]
            df_results.to_csv(output_dir / "estimated_lag_summary.csv", index=False)
            print(f"\nSaved summary table: {output_dir / 'estimated_lag_summary.csv'}")

    return results


def run_cumulative_lag_analysis(
    args,
    workout_vars: List[str],
    output_dir: Path,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run cumulative lag model analysis for each variable."""
    print("\n" + "=" * 70)
    print("CUMULATIVE LAG MODEL ANALYSIS")
    print("=" * 70)

    # Generate lag days list based on window and step
    lag_days_list = np.arange(0, args.lag_window + args.lag_step, args.lag_step)
    lag_days_list = lag_days_list[lag_days_list <= args.lag_window]  # ensure up to window
    lag_days_list = lag_days_list[lag_days_list > 0]  # exclude zero lag? include zero if window starts at 0
    # Include zero lag if step starts at 0
    if 0 in lag_days_list:
        lag_days_list = lag_days_list[lag_days_list >= 0]
    print(f"Lag window: {args.lag_window} days, step: {args.lag_step} days")
    print(f"Lags to include: {lag_days_list.tolist()} days")

    # Load weight data (common for all variables)
    print("\n1. Loading weight data...")
    df_weight = load_weight_data(args.data_dir)
    print(f"   Weight measurements: {len(df_weight)}")

    # Set up sparse GP parameters
    use_sparse = not args.no_sparse
    sparse_params = {
        "use_sparse": use_sparse,
        "n_inducing_points": args.n_inducing_points if use_sparse else 0,
    }

    # Set up caching
    cache = not args.no_cache
    force_refit = args.force_refit or args.no_cache

    results = {}

    for var_name in workout_vars:
        print(f"\n{'='*60}")
        print(f"Analyzing variable: {var_name}")
        print(f"{'='*60}")

        # Load workout data for this variable
        print(f"2. Loading workout data for '{var_name}'...")
        df_workouts_raw = load_workout_data(
            data_dir=args.data_dir,
            activity_type=var_name,
            include_exercise_details=False,
        )
        print(f"   Raw workout records: {len(df_workouts_raw)}")

        # Aggregate workouts to daily count
        df_workouts_agg = prepare_workout_aggregates(
            df_workouts_raw,
            aggregation="daily",
            metric="count"
        )
        print(f"   Workout days (with workouts): {len(df_workouts_agg)}")
        if len(df_workouts_agg) == 0:
            print(f"   WARNING: No workout data for '{var_name}'. Skipping.")
            continue

        var_output_dir = output_dir / var_name
        var_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Fit cumulative lag model
            print(f"3. Fitting cumulative lag model for '{var_name}'...")
            print(f"   Using lags: {lag_days_list.tolist()} days")
            fit, idata, stan_data = fit_crosslagged_model_cumulative(
                df_weight=df_weight,
                df_workout=df_workouts_agg,
                lag_days_list=lag_days_list.tolist(),
                workout_value_col="workout_count",
                chains=args.chains,
                iter_warmup=args.iter_warmup,
                iter_sampling=args.iter_sampling,
                use_sparse=use_sparse,
                n_inducing_points=args.n_inducing_points,
                inducing_point_method="uniform",
                cache=cache,
                force_refit=force_refit,
                include_prediction_grid=args.include_prediction_grid,
                prediction_step_days=args.prediction_step_days,
            )

            # Extract posterior summaries
            posterior = idata.posterior
            result = {
                'variable': var_name,
                'lag_window': args.lag_window,
                'lag_step': args.lag_step,
                'lag_days_list': lag_days_list.tolist(),
                'n_weight_obs': len(df_weight),
                'n_workout_obs': len(df_workouts_agg),
            }

            # Extract beta (causal effect) - single beta for cumulative model
            if 'beta' in posterior:
                beta_samples = posterior['beta'].values.flatten()
                result['beta_mean'] = np.mean(beta_samples)
                result['beta_std'] = np.std(beta_samples)
                result['beta_ci_low'] = np.percentile(beta_samples, 2.5)
                result['beta_ci_high'] = np.percentile(beta_samples, 97.5)
                result['n_eff_beta'] = _compute_ess(posterior['beta'])
                result['rhat_beta'] = _compute_rhat(posterior['beta'])
                print(f"   β (standardized): {result['beta_mean']:.3f} [{result['beta_ci_low']:.3f}, {result['beta_ci_high']:.3f}]")
            else:
                print("   Warning: 'beta' not found in posterior")

            # Extract beta in original units if available
            if 'beta_original_units' in posterior:
                beta_orig_samples = posterior['beta_original_units'].values.flatten()
                result['beta_orig_mean'] = np.mean(beta_orig_samples)
                result['beta_orig_std'] = np.std(beta_orig_samples)
                result['beta_orig_ci_low'] = np.percentile(beta_orig_samples, 2.5)
                result['beta_orig_ci_high'] = np.percentile(beta_orig_samples, 97.5)
                print(f"   β (original units): {result['beta_orig_mean']:.3f} [{result['beta_orig_ci_low']:.3f}, {result['beta_orig_ci_high']:.3f}]")

            # Compute WAIC and LOO (weight likelihood only)
            try:
                waic = az.waic(idata, var_name="log_lik_weight")
                loo = az.loo(idata, var_name="log_lik_weight")
                result['waic'] = waic.waic
                result['loo'] = loo.loo
                print(f"   WAIC: {result['waic']:.1f}")
                print(f"   LOO: {result['loo']:.1f}")
            except Exception as e:
                print(f"   Warning: Could not compute WAIC/LOO: {e}")
                result['waic'] = np.nan
                result['loo'] = np.nan

            # Save posterior summary
            summary = az.summary(idata, round_to=3)
            summary.to_csv(var_output_dir / "posterior_summary.csv")

            # Save beta posterior samples
            if 'beta' in posterior:
                beta_df = pd.DataFrame({'beta': beta_samples})
                beta_df.to_csv(var_output_dir / "beta_posterior.csv", index=False)

            # Generate plots if not skipping
            if not skip_plots:
                # Beta posterior plot
                if 'beta' in posterior:
                    plt.figure(figsize=(8, 5))
                    plt.hist(beta_samples, bins=30, alpha=0.7, density=True, edgecolor='black')
                    plt.axvline(0, color='red', linestyle='--', label='Zero effect')
                    plt.xlabel('β (causal effect: workouts → weight)')
                    plt.ylabel('Density')
                    plt.title(f'Posterior of β ({var_name}, cumulative lags {args.lag_window}d window)')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(var_output_dir / "beta_posterior.png", dpi=150)
                    plt.close()
                    print(f"   Saved beta posterior plot: {var_output_dir / 'beta_posterior.png'}")

                # Optional: Plot of lag contributions if model provides them
                # (Cumulative model may not have individual lag contributions)

            results[var_name] = result
            print(f"   Results saved to {var_output_dir}")

        except Exception as e:
            print(f"   Error fitting cumulative lag model for {var_name}: {e}")
            import traceback
            traceback.print_exc()
            results[var_name] = {'variable': var_name, 'error': str(e)}

    # Create summary table across variables
    if results:
        valid_results = []
        for var_name, result in results.items():
            if 'beta_mean' in result:
                valid_results.append(result)

        if valid_results:
            df_results = pd.DataFrame(valid_results)
            # Reorder columns
            cols = ['variable', 'lag_window', 'lag_step', 'lag_days_list',
                    'beta_mean', 'beta_std', 'beta_ci_low', 'beta_ci_high',
                    'beta_orig_mean', 'beta_orig_std', 'beta_orig_ci_low', 'beta_orig_ci_high',
                    'waic', 'loo', 'n_eff_beta', 'rhat_beta',
                    'n_weight_obs', 'n_workout_obs']
            df_results = df_results[[c for c in cols if c in df_results.columns]]
            df_results.to_csv(output_dir / "cumulative_lag_summary.csv", index=False)
            print(f"\nSaved summary table: {output_dir / 'cumulative_lag_summary.csv'}")

    return results


def create_comparison_report(
    results: Dict[str, Any],
    output_dir: Path,
    model_type: str,
) -> None:
    """Create markdown report comparing results across variables/lags."""
    report_path = output_dir / f"{model_type}_comparison_report.md"

    with open(report_path, 'w') as f:
        f.write(f"# Cross-Lagged Model Comparison Report\n\n")
        f.write(f"**Model type**: {model_type}\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if model_type == "fixed":
            _write_fixed_lag_report(f, results, output_dir)
        elif model_type == "estimated":
            _write_estimated_lag_report(f, results, output_dir)
        elif model_type == "cumulative":
            _write_cumulative_lag_report(f, results, output_dir)
        else:
            f.write("Unknown model type\n")

        f.write("\n---\n")
        f.write("*Report generated by demo_bivariate.py*\n")

    print(f"Report saved to {report_path}")


def _write_fixed_lag_report(f, results: Dict[str, Any], output_dir: Path):
    """Write fixed lag model report section."""
    if not results:
        f.write("No results available.\n")
        return

    f.write("## Fixed Lag Model Comparison\n\n")

    # Collect all valid results
    all_valid_results = []
    for var_name, var_results in results.items():
        for r in var_results:
            if 'beta_mean' in r:
                all_valid_results.append(r)

    if not all_valid_results:
        f.write("No valid model results found.\n")
        return

    df_all = pd.DataFrame(all_valid_results)

    # Summary table
    f.write("### Summary Table\n\n")
    f.write("| Variable | Lag (days) | β (std) | 95% CI | β (orig) | 95% CI | WAIC | LOO |\n")
    f.write("|----------|------------|---------|--------|----------|--------|------|-----|\n")

    for _, row in df_all.iterrows():
        beta_std_ci = f"{row['beta_mean']:.3f} [{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}]"
        if pd.notna(row['beta_orig_mean']):
            beta_orig_ci = f"{row['beta_orig_mean']:.3f} [{row['beta_orig_ci_low']:.3f}, {row['beta_orig_ci_high']:.3f}]"
        else:
            beta_orig_ci = "N/A"
        waic_str = f"{row['waic']:.1f}" if pd.notna(row['waic']) else "N/A"
        loo_str = f"{row['loo']:.1f}" if pd.notna(row['loo']) else "N/A"
        f.write(f"| {row['variable']} | {row['lag_days']} | {beta_std_ci} | {beta_orig_ci} | {waic_str} | {loo_str} |\n")

    f.write("\n")

    # Best lag by WAIC for each variable
    f.write("### Best Lag Selection (by WAIC)\n\n")
    variables = df_all['variable'].unique()
    for var in variables:
        df_var = df_all[df_all['variable'] == var]
        if df_var['waic'].notna().any():
            best_idx = df_var['waic'].idxmin()
            best_row = df_var.loc[best_idx]
            f.write(f"- **{var}**: Best lag τ = {best_row['lag_days']} days (WAIC = {best_row['waic']:.1f})\n")
            f.write(f"  - β (std): {best_row['beta_mean']:.3f} [{best_row['beta_ci_low']:.3f}, {best_row['beta_ci_high']:.3f}]\n")
            if pd.notna(best_row['beta_orig_mean']):
                f.write(f"  - β (orig): {best_row['beta_orig_mean']:.3f} [{best_row['beta_orig_ci_low']:.3f}, {best_row['beta_orig_ci_high']:.3f}]\n")
        else:
            f.write(f"- **{var}**: WAIC not available\n")
    f.write("\n")

    # Interpretation
    f.write("### Interpretation\n\n")
    f.write("1. **β (standardized)**: Weight change in standard deviations per 1 SD increase in workout metric.\n")
    f.write("2. **β (original units)**: Weight change in lbs per unit increase in workout metric.\n")
    f.write("3. **Lag τ**: Days between workout and weight measurement (workouts at time t affect weight at time t+τ).\n")
    f.write("4. **WAIC/LOO**: Lower values indicate better model fit (WAIC weights can be computed).\n")
    f.write("\n")
    f.write("**Effect direction**:\n")
    f.write("- β > 0: Workouts cause weight gain (muscle mass accumulation)\n")
    f.write("- β < 0: Workouts cause weight loss (fat loss)\n")
    f.write("- β ≈ 0: No causal effect detected\n")
    f.write("\n")
    f.write("**Assumptions**:\n")
    f.write("- No unmeasured confounding\n")
    f.write("- Linear effect of workouts on weight\n")
    f.write("- Gaussian process captures intrinsic weight dynamics\n")
    f.write("\n")

    # Visualizations
    f.write("### Visualizations\n\n")
    # Individual variable plots
    for var in variables:
        var_dir = output_dir / var
        beta_vs_lag_path = var_dir / "beta_vs_lag.png"
        if beta_vs_lag_path.exists():
            f.write(f"#### {var}\n\n")
            f.write(f"![β vs Lag for {var}]({var}/beta_vs_lag.png)\n\n")
            f.write(f"*Causal effect of {var} on weight across different lags*\n\n")

    # Combined plot
    combined_path = output_dir / "beta_vs_lag_all_variables.png"
    if combined_path.exists():
        f.write("#### All Variables Combined\n\n")
        f.write("![β vs Lag for all variables](beta_vs_lag_all_variables.png)\n\n")
        f.write("*Comparison of causal effects across different workout types*\n\n")

    # Data summary
    f.write("### Data Summary\n\n")
    sample_row = df_all.iloc[0]
    f.write(f"- Weight observations: {sample_row['n_weight_obs']}\n")
    # Count unique workout observations per variable
    for var in variables:
        df_var = df_all[df_all['variable'] == var]
        if len(df_var) > 0:
            n_workout = df_var.iloc[0]['n_workout_obs']
            f.write(f"- {var} workout days: {n_workout}\n")
    f.write("\n")


def _write_estimated_lag_report(f, results: Dict[str, Any], output_dir: Path):
    """Write estimated lag model report section."""
    f.write("## Estimated Lag Model Analysis\n\n")

    if not results:
        f.write("No results available.\n")
        return

    # Collect valid results
    valid_results = []
    for var_name, result in results.items():
        if 'beta_mean' in result and 'lag_days_mean' in result:
            valid_results.append(result)

    if not valid_results:
        f.write("No valid model results found.\n")
        return

    df = pd.DataFrame(valid_results)

    # Summary table
    f.write("### Summary Table\n\n")
    f.write("| Variable | β (std) | 95% CI | β (orig) | 95% CI | Lag τ (days) | 95% CI | WAIC | LOO |\n")
    f.write("|----------|---------|--------|----------|--------|--------------|--------|------|-----|\n")

    for _, row in df.iterrows():
        beta_std_ci = f"{row['beta_mean']:.3f} [{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}]"
        if 'beta_orig_mean' in row and pd.notna(row['beta_orig_mean']):
            beta_orig_ci = f"{row['beta_orig_mean']:.3f} [{row['beta_orig_ci_low']:.3f}, {row['beta_orig_ci_high']:.3f}]"
        else:
            beta_orig_ci = "N/A"
        lag_ci = f"{row['lag_days_mean']:.2f} [{row['lag_days_ci_low']:.2f}, {row['lag_days_ci_high']:.2f}]"
        waic_str = f"{row['waic']:.1f}" if 'waic' in row and pd.notna(row['waic']) else "N/A"
        loo_str = f"{row['loo']:.1f}" if 'loo' in row and pd.notna(row['loo']) else "N/A"
        f.write(f"| {row['variable']} | {beta_std_ci} | {beta_orig_ci} | {lag_ci} | {waic_str} | {loo_str} |\n")

    f.write("\n")

    # Interpretation
    f.write("### Interpretation\n\n")
    f.write("1. **β (standardized)**: Weight change in standard deviations per 1 SD increase in workout metric.\n")
    f.write("2. **β (original units)**: Weight change in lbs per unit increase in workout metric.\n")
    f.write("3. **Lag τ**: Estimated days between workout and weight measurement (posterior mean with 95% CI).\n")
    f.write("4. **Prior**: Lag follows exponential(1/3) prior (mean = 3 days).\n")
    f.write("5. **WAIC/LOO**: Lower values indicate better model fit.\n")
    f.write("\n")
    f.write("**Effect direction**:\n")
    f.write("- β > 0: Workouts cause weight gain (muscle mass accumulation)\n")
    f.write("- β < 0: Workouts cause weight loss (fat loss)\n")
    f.write("- β ≈ 0: No causal effect detected\n")
    f.write("\n")
    f.write("**Lag interpretation**:\n")
    f.write("- τ ≈ 0: Immediate effect (same day)\n")
    f.write("- τ ≈ 1-3 days: Short-term effect\n")
    f.write("- τ > 7 days: Longer-term delayed effect\n")
    f.write("\n")

    # Visualizations
    f.write("### Visualizations\n\n")
    for var_name in df['variable'].unique():
        var_dir = output_dir / var_name
        beta_plot_path = var_dir / "beta_posterior.png"
        lag_plot_path = var_dir / "lag_days_posterior.png"
        scatter_plot_path = var_dir / "beta_vs_lag_scatter.png"

        if beta_plot_path.exists():
            f.write(f"#### {var_name}\n\n")
            f.write(f"![β posterior for {var_name}]({var_name}/beta_posterior.png)\n\n")
            f.write(f"*Posterior distribution of causal effect β for {var_name}*\n\n")

        if lag_plot_path.exists():
            f.write(f"![Lag τ posterior for {var_name}]({var_name}/lag_days_posterior.png)\n\n")
            f.write(f"*Posterior distribution of estimated lag τ for {var_name}*\n\n")

        if scatter_plot_path.exists():
            f.write(f"![β vs τ scatter for {var_name}]({var_name}/beta_vs_lag_scatter.png)\n\n")
            f.write(f"*Joint posterior of β and τ for {var_name}*\n\n")

    # Data summary
    f.write("### Data Summary\n\n")
    sample_row = df.iloc[0]
    f.write(f"- Weight observations: {sample_row['n_weight_obs']}\n")
    for _, row in df.iterrows():
        f.write(f"- {row['variable']} workout days: {row['n_workout_obs']}\n")
    f.write("\n")


def _write_cumulative_lag_report(f, results: Dict[str, Any], output_dir: Path):
    """Write cumulative lag model report section."""
    f.write("## Cumulative Lag Model Analysis\n\n")

    if not results:
        f.write("No results available.\n")
        return

    # Collect valid results
    valid_results = []
    for var_name, result in results.items():
        if 'beta_mean' in result:
            valid_results.append(result)

    if not valid_results:
        f.write("No valid model results found.\n")
        return

    df = pd.DataFrame(valid_results)

    # Summary table
    f.write("### Summary Table\n\n")
    f.write("| Variable | Lag Window | Lag Step | β (std) | 95% CI | β (orig) | 95% CI | WAIC | LOO |\n")
    f.write("|----------|------------|----------|---------|--------|----------|--------|------|-----|\n")

    for _, row in df.iterrows():
        lag_window = row.get('lag_window', 'N/A')
        lag_step = row.get('lag_step', 'N/A')
        beta_std_ci = f"{row['beta_mean']:.3f} [{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}]"
        if 'beta_orig_mean' in row and pd.notna(row['beta_orig_mean']):
            beta_orig_ci = f"{row['beta_orig_mean']:.3f} [{row['beta_orig_ci_low']:.3f}, {row['beta_orig_ci_high']:.3f}]"
        else:
            beta_orig_ci = "N/A"
        waic_str = f"{row['waic']:.1f}" if 'waic' in row and pd.notna(row['waic']) else "N/A"
        loo_str = f"{row['loo']:.1f}" if 'loo' in row and pd.notna(row['loo']) else "N/A"
        f.write(f"| {row['variable']} | {lag_window} days | {lag_step} days | {beta_std_ci} | {beta_orig_ci} | {waic_str} | {loo_str} |\n")

    f.write("\n")

    # Interpretation
    f.write("### Interpretation\n\n")
    f.write("1. **Cumulative effect**: Average causal effect over multiple lag values.\n")
    f.write("2. **Lag window**: Window of lag values considered (e.g., 0-7 days).\n")
    f.write("3. **Lag step**: Step between lag values in the window.\n")
    f.write("4. **β (standardized)**: Average weight change in standard deviations per 1 SD increase in workout metric.\n")
    f.write("5. **β (original units)**: Average weight change in lbs per unit increase in workout metric.\n")
    f.write("6. **WAIC/LOO**: Lower values indicate better model fit.\n")
    f.write("\n")
    f.write("**Effect direction**:\n")
    f.write("- β > 0: Workouts cause weight gain (muscle mass accumulation)\n")
    f.write("- β < 0: Workouts cause weight loss (fat loss)\n")
    f.write("- β ≈ 0: No causal effect detected\n")
    f.write("\n")
    f.write("**Lag window interpretation**:\n")
    f.write("- Window of 0-7 days captures immediate to week-delayed effects.\n")
    f.write("- Cumulative effect averages across all lags in window.\n")
    f.write("- Model assumes equal weighting across lags (uniform average).\n")
    f.write("\n")

    # Visualizations
    f.write("### Visualizations\n\n")
    for var_name in df['variable'].unique():
        var_dir = output_dir / var_name
        beta_plot_path = var_dir / "beta_posterior.png"

        if beta_plot_path.exists():
            f.write(f"#### {var_name}\n\n")
            f.write(f"![β posterior for {var_name}]({var_name}/beta_posterior.png)\n\n")
            f.write(f"*Posterior distribution of cumulative causal effect β for {var_name}*\n\n")

    # Data summary
    f.write("### Data Summary\n\n")
    sample_row = df.iloc[0]
    f.write(f"- Weight observations: {sample_row['n_weight_obs']}\n")
    for _, row in df.iterrows():
        f.write(f"- {row['variable']} workout days: {row['n_workout_obs']}\n")
    f.write("\n")


if __name__ == "__main__":
    main()