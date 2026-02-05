#!/usr/bin/env python3
"""Compare different cross-lagged modeling approaches.

This script runs and compares three cross-lagged GP model types:
1. Fixed lag: Compare multiple lag values (τ=0,1,2,3,7 days)
2. Estimated lag: Estimate lag τ as parameter with exponential prior
3. Cumulative lag: Average effect over window of lag values

For each workout variable, it runs all three models, collects results,
computes model comparison metrics (WAIC/LOO), and generates comprehensive
visualizations and reports.

Usage:
    # Compare all three models for strength_training
    python -m src.analysis.compare_crosslagged_models --workout-vars strength_training \
        --output-dir output/crosslagged_comparison

    # Compare multiple workout types with custom lags
    python -m src.analysis.compare_crosslagged_models \
        --workout-vars strength_training,walking,cycling \
        --fixed-lags 0,1,2,3,5,7 \
        --estimated-lag-prior-mean 3.0 --estimated-lag-prior-sd 2.0 \
        --cumulative-window 7 --cumulative-step 1 \
        --output-dir output/full_comparison

    # Quick test with minimal MCMC
    python -m src.analysis.compare_crosslagged_models \
        --workout-vars strength_training \
        --chains 2 --iter-warmup 100 --iter-sampling 100 \
        --skip-plots --output-dir output/test
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.demo_bivariate import (
    run_fixed_lag_comparison,
    run_estimated_lag_analysis,
    run_cumulative_lag_analysis,
    _compute_ess,
    _compute_rhat,
)
from src.data import load_weight_data
from src.data.workout import load_workout_data, prepare_workout_aggregates


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


class CrossLaggedModelComparison:
    """Compare different cross-lagged modeling approaches."""

    def __init__(
        self,
        workout_vars: List[str],
        data_dir: str = "data",
        output_dir: str = "output/crosslagged_comparison",
        fixed_lags: List[float] = [0, 1, 2, 3, 7],
        estimated_lag_prior_mean: float = 3.0,
        estimated_lag_prior_sd: float = 2.0,
        cumulative_window: int = 7,
        cumulative_step: float = 1.0,
        chains: int = 4,
        iter_warmup: int = 500,
        iter_sampling: int = 500,
        use_sparse: bool = True,
        n_inducing_points: int = 50,
        skip_plots: bool = False,
        force_refit: bool = False,
        no_cache: bool = False,
    ):
        """Initialize comparison with configuration.

        Args:
            workout_vars: List of workout/activity variables to analyze.
            data_dir: Path to data directory.
            output_dir: Directory for output files.
            fixed_lags: List of lag values for fixed lag model (days).
            estimated_lag_prior_mean: Prior mean for estimated lag model (days).
            estimated_lag_prior_sd: Prior SD for estimated lag model (days).
            cumulative_window: Window size for cumulative lag model (days).
            cumulative_step: Step between lags in cumulative window (days).
            chains: Number of MCMC chains.
            iter_warmup: Warmup iterations per chain.
            iter_sampling: Sampling iterations per chain.
            use_sparse: Whether to use sparse GP approximation.
            n_inducing_points: Number of inducing points for sparse GP.
            skip_plots: Skip generating plots.
            force_refit: Force refit even if cached results exist.
            no_cache: Disable caching (force refit).
        """
        self.workout_vars = workout_vars
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fixed_lags = fixed_lags
        self.estimated_lag_prior_mean = estimated_lag_prior_mean
        self.estimated_lag_prior_sd = estimated_lag_prior_sd
        self.cumulative_window = cumulative_window
        self.cumulative_step = cumulative_step
        self.chains = chains
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.use_sparse = use_sparse
        self.n_inducing_points = n_inducing_points
        self.skip_plots = skip_plots
        self.force_refit = force_refit
        self.no_cache = no_cache

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_dir = self.output_dir / "fixed_lag"
        self.estimated_dir = self.output_dir / "estimated_lag"
        self.cumulative_dir = self.output_dir / "cumulative_lag"
        self.fixed_dir.mkdir(exist_ok=True)
        self.estimated_dir.mkdir(exist_ok=True)
        self.cumulative_dir.mkdir(exist_ok=True)

        # Store results
        self.fixed_results = {}
        self.estimated_results = {}
        self.cumulative_results = {}
        self.comparison_results = []

        # Load common data
        self.df_weight = None
        self.workout_data = {}

    def load_data(self) -> None:
        """Load weight and workout data for all variables."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        # Load weight data (common for all variables)
        print("\n1. Loading weight data...")
        self.df_weight = load_weight_data(self.data_dir)
        print(f"   Weight measurements: {len(self.df_weight)}")

        # Load workout data for each variable
        for var_name in self.workout_vars:
            print(f"\n2. Loading workout data for '{var_name}'...")
            df_workouts_raw = load_workout_data(
                data_dir=self.data_dir,
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
                print(f"   WARNING: No workout data for '{var_name}'. Skipping variable.")
                continue

            self.workout_data[var_name] = df_workouts_agg

    def run_fixed_lag_analysis(self) -> None:
        """Run fixed lag model comparison."""
        print("\n" + "=" * 70)
        print("FIXED LAG MODEL ANALYSIS")
        print("=" * 70)

        # Create args object for run_fixed_lag_comparison
        class Args:
            pass

        args = Args()
        args.data_dir = str(self.data_dir)
        args.no_sparse = not self.use_sparse
        args.n_inducing_points = self.n_inducing_points
        args.chains = self.chains
        args.iter_warmup = self.iter_warmup
        args.iter_sampling = self.iter_sampling
        args.no_cache = self.no_cache
        args.force_refit = self.force_refit
        args.include_prediction_grid = False
        args.prediction_step_days = 1.0
        args.skip_plots = self.skip_plots

        # Run fixed lag analysis using demo_bivariate function
        self.fixed_results = run_fixed_lag_comparison(
            args=args,
            workout_vars=self.workout_vars,
            lag_values=self.fixed_lags,
            output_dir=self.fixed_dir,
            skip_plots=self.skip_plots,
        )

        # Extract comparison results
        for var_name, var_results in self.fixed_results.items():
            if not var_results:
                continue
            for result in var_results:
                if 'beta_mean' in result:
                    self.comparison_results.append({
                        'variable': var_name,
                        'model_type': 'fixed',
                        'lag_value': result.get('lag_days', np.nan),
                        'beta_mean': result.get('beta_mean', np.nan),
                        'beta_std': result.get('beta_std', np.nan),
                        'beta_ci_low': result.get('beta_ci_low', np.nan),
                        'beta_ci_high': result.get('beta_ci_high', np.nan),
                        'beta_orig_mean': result.get('beta_orig_mean', np.nan),
                        'beta_orig_ci_low': result.get('beta_orig_ci_low', np.nan),
                        'beta_orig_ci_high': result.get('beta_orig_ci_high', np.nan),
                        'waic': result.get('waic', np.nan),
                        'loo': result.get('loo', np.nan),
                        'n_eff_beta': result.get('n_eff_beta', np.nan),
                        'rhat_beta': result.get('rhat_beta', np.nan),
                        'n_weight_obs': result.get('n_weight_obs', np.nan),
                        'n_workout_obs': result.get('n_workout_obs', np.nan),
                    })

    def run_estimated_lag_analysis(self) -> None:
        """Run estimated lag model analysis."""
        print("\n" + "=" * 70)
        print("ESTIMATED LAG MODEL ANALYSIS")
        print("=" * 70)

        # Create args object for run_estimated_lag_analysis
        class Args:
            pass

        args = Args()
        args.data_dir = str(self.data_dir)
        args.no_sparse = not self.use_sparse
        args.n_inducing_points = self.n_inducing_points
        args.chains = self.chains
        args.iter_warmup = self.iter_warmup
        args.iter_sampling = self.iter_sampling
        args.no_cache = self.no_cache
        args.force_refit = self.force_refit
        args.include_prediction_grid = False
        args.prediction_step_days = 1.0
        args.skip_plots = self.skip_plots
        args.lag_prior_mean = self.estimated_lag_prior_mean
        args.lag_prior_sd = self.estimated_lag_prior_sd

        # Run estimated lag analysis using demo_bivariate function
        self.estimated_results = run_estimated_lag_analysis(
            args=args,
            workout_vars=self.workout_vars,
            output_dir=self.estimated_dir,
            skip_plots=self.skip_plots,
        )

        # Extract comparison results
        for var_name, result in self.estimated_results.items():
            if 'beta_mean' in result:
                self.comparison_results.append({
                    'variable': var_name,
                    'model_type': 'estimated',
                    'lag_value': result.get('lag_days_mean', np.nan),
                    'beta_mean': result.get('beta_mean', np.nan),
                    'beta_std': result.get('beta_std', np.nan),
                    'beta_ci_low': result.get('beta_ci_low', np.nan),
                    'beta_ci_high': result.get('beta_ci_high', np.nan),
                    'beta_orig_mean': result.get('beta_orig_mean', np.nan),
                    'beta_orig_ci_low': result.get('beta_orig_ci_low', np.nan),
                    'beta_orig_ci_high': result.get('beta_orig_ci_high', np.nan),
                    'waic': result.get('waic', np.nan),
                    'loo': result.get('loo', np.nan),
                    'n_eff_beta': result.get('n_eff_beta', np.nan),
                    'rhat_beta': result.get('rhat_beta', np.nan),
                    'n_weight_obs': result.get('n_weight_obs', np.nan),
                    'n_workout_obs': result.get('n_workout_obs', np.nan),
                    'lag_mean': result.get('lag_days_mean', np.nan),
                    'lag_ci_low': result.get('lag_days_ci_low', np.nan),
                    'lag_ci_high': result.get('lag_days_ci_high', np.nan),
                    'n_eff_lag': result.get('n_eff_lag', np.nan),
                    'rhat_lag': result.get('rhat_lag', np.nan),
                })

    def run_cumulative_lag_analysis(self) -> None:
        """Run cumulative lag model analysis."""
        print("\n" + "=" * 70)
        print("CUMULATIVE LAG MODEL ANALYSIS")
        print("=" * 70)

        # Create args object for run_cumulative_lag_analysis
        class Args:
            pass

        args = Args()
        args.data_dir = str(self.data_dir)
        args.no_sparse = not self.use_sparse
        args.n_inducing_points = self.n_inducing_points
        args.chains = self.chains
        args.iter_warmup = self.iter_warmup
        args.iter_sampling = self.iter_sampling
        args.no_cache = self.no_cache
        args.force_refit = self.force_refit
        args.include_prediction_grid = False
        args.prediction_step_days = 1.0
        args.skip_plots = self.skip_plots
        args.lag_window = self.cumulative_window
        args.lag_step = self.cumulative_step

        # Run cumulative lag analysis using demo_bivariate function
        self.cumulative_results = run_cumulative_lag_analysis(
            args=args,
            workout_vars=self.workout_vars,
            output_dir=self.cumulative_dir,
            skip_plots=self.skip_plots,
        )

        # Extract comparison results
        for var_name, result in self.cumulative_results.items():
            if 'beta_mean' in result:
                self.comparison_results.append({
                    'variable': var_name,
                    'model_type': 'cumulative',
                    'lag_value': f"window_{self.cumulative_window}d",
                    'beta_mean': result.get('beta_mean', np.nan),
                    'beta_std': result.get('beta_std', np.nan),
                    'beta_ci_low': result.get('beta_ci_low', np.nan),
                    'beta_ci_high': result.get('beta_ci_high', np.nan),
                    'beta_orig_mean': result.get('beta_orig_mean', np.nan),
                    'beta_orig_ci_low': result.get('beta_orig_ci_low', np.nan),
                    'beta_orig_ci_high': result.get('beta_orig_ci_high', np.nan),
                    'waic': result.get('waic', np.nan),
                    'loo': result.get('loo', np.nan),
                    'n_eff_beta': result.get('n_eff_beta', np.nan),
                    'rhat_beta': result.get('rhat_beta', np.nan),
                    'n_weight_obs': result.get('n_weight_obs', np.nan),
                    'n_workout_obs': result.get('n_workout_obs', np.nan),
                    'lag_window': result.get('lag_window', self.cumulative_window),
                    'lag_step': result.get('lag_step', self.cumulative_step),
                    'lag_days_list': str(result.get('lag_days_list', []))
                })

    def create_comparison_tables(self) -> pd.DataFrame:
        """Create comparison tables from all model results."""
        print("\n" + "=" * 70)
        print("CREATING COMPARISON TABLES")
        print("=" * 70)

        if not self.comparison_results:
            print("No results to compare.")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(self.comparison_results)

        # Save raw comparison table
        df.to_csv(self.output_dir / "model_comparison_raw.csv", index=False)
        print(f"Saved raw comparison table: {self.output_dir / 'model_comparison_raw.csv'}")

        # Create summary table (best model per variable by WAIC)
        summary_rows = []
        for var_name in df['variable'].unique():
            df_var = df[df['variable'] == var_name]

            # Find best model by WAIC (lowest)
            if df_var['waic'].notna().any():
                best_idx = df_var['waic'].idxmin()
                best_row = df_var.loc[best_idx]

                summary_rows.append({
                    'variable': var_name,
                    'best_model': best_row['model_type'],
                    'best_waic': best_row['waic'],
                    'best_beta': best_row['beta_mean'],
                    'beta_ci': f"[{best_row['beta_ci_low']:.3f}, {best_row['beta_ci_high']:.3f}]",
                    'best_lag': best_row.get('lag_value', best_row.get('lag_mean', 'N/A')),
                    'n_weight': int(best_row['n_weight_obs']),
                    'n_workout': int(best_row['n_workout_obs']),
                })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(self.output_dir / "model_comparison_summary.csv", index=False)
            print(f"Saved summary table: {self.output_dir / 'model_comparison_summary.csv'}")

            # Print summary
            print("\nBest Model by WAIC for Each Variable:")
            print("-" * 80)
            print(f"{'Variable':<20} {'Best Model':<15} {'WAIC':<10} {'β':<10} {'95% CI':<20} {'Lag':<15}")
            print("-" * 80)
            for _, row in df_summary.iterrows():
                print(f"{row['variable']:<20} {row['best_model']:<15} {row['best_waic']:<10.1f} "
                      f"{row['best_beta']:<10.3f} {row['beta_ci']:<20} {str(row['best_lag']):<15}")

        return df

    def create_comparison_plots(self, df: pd.DataFrame) -> None:
        """Create comparison visualizations."""
        if self.skip_plots or df.empty:
            return

        print("\n" + "=" * 70)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("=" * 70)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # 1. β comparison across models for each variable
        for var_name in df['variable'].unique():
            df_var = df[df['variable'] == var_name]

            plt.figure(figsize=(12, 8))

            # Separate fixed lag points
            df_fixed = df_var[df_var['model_type'] == 'fixed']
            df_other = df_var[df_var['model_type'] != 'fixed']

            # Plot fixed lag points with line
            if not df_fixed.empty:
                # Sort by lag
                df_fixed = df_fixed.sort_values('lag_value')
                plt.errorbar(
                    df_fixed['lag_value'],
                    df_fixed['beta_mean'],
                    yerr=[df_fixed['beta_mean'] - df_fixed['beta_ci_low'],
                          df_fixed['beta_ci_high'] - df_fixed['beta_mean']],
                    fmt='o-', capsize=5, capthick=2, linewidth=2,
                    label='Fixed lag (different τ values)', alpha=0.8
                )

            # Plot estimated and cumulative as points
            model_colors = {'estimated': 'red', 'cumulative': 'green'}
            model_labels = {'estimated': 'Estimated τ', 'cumulative': f'Cumulative ({self.cumulative_window}d)'}

            for model_type in ['estimated', 'cumulative']:
                df_model = df_other[df_other['model_type'] == model_type]
                if not df_model.empty:
                    row = df_model.iloc[0]
                    x_pos = {'estimated': 10, 'cumulative': 12}.get(model_type, 8)

                    plt.errorbar(
                        x_pos,
                        row['beta_mean'],
                        yerr=[[row['beta_mean'] - row['beta_ci_low']],
                              [row['beta_ci_high'] - row['beta_mean']]],
                        fmt='s', capsize=8, capthick=2, markersize=10,
                        color=model_colors.get(model_type, 'blue'),
                        label=model_labels.get(model_type, model_type)
                    )

            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Lag τ (days) / Model Type', fontsize=12)
            plt.ylabel('β (causal effect: workouts → weight)', fontsize=12)
            plt.title(f'Cross-Lagged Model Comparison: {var_name}', fontsize=14, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"model_comparison_{var_name}.png", dpi=150)
            plt.close()
            print(f"  Saved comparison plot: {self.output_dir / f'model_comparison_{var_name}.png'}")

        # 2. WAIC comparison across models
        plt.figure(figsize=(10, 6))

        # Create pivot table for WAIC
        df_waic = df.pivot_table(
            index='variable',
            columns='model_type',
            values='waic',
            aggfunc='first'
        )

        # Plot WAIC as grouped bar chart
        df_waic.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Workout Variable', fontsize=12)
        plt.ylabel('WAIC (lower is better)', fontsize=12)
        plt.title('Model Comparison by WAIC', fontsize=14, fontweight='bold')
        plt.legend(title='Model Type')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / "waic_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved WAIC comparison: {self.output_dir / 'waic_comparison.png'}")

        # 3. Lag estimates from estimated model
        df_estimated = df[df['model_type'] == 'estimated']
        if not df_estimated.empty:
            plt.figure(figsize=(10, 6))

            variables = df_estimated['variable'].unique()
            x_pos = np.arange(len(variables))

            lag_means = []
            lag_cis_low = []
            lag_cis_high = []

            for var_name in variables:
                row = df_estimated[df_estimated['variable'] == var_name].iloc[0]
                lag_means.append(row.get('lag_mean', np.nan))
                lag_cis_low.append(row.get('lag_ci_low', np.nan))
                lag_cis_high.append(row.get('lag_ci_high', np.nan))

            plt.errorbar(
                x_pos, lag_means,
                yerr=[np.array(lag_means) - np.array(lag_cis_low),
                      np.array(lag_cis_high) - np.array(lag_means)],
                fmt='o', capsize=8, capthick=2, markersize=8
            )
            plt.axhline(y=self.estimated_lag_prior_mean, color='red',
                       linestyle='--', label=f'Prior mean ({self.estimated_lag_prior_mean} days)')

            plt.xticks(x_pos, variables, rotation=45)
            plt.xlabel('Workout Variable', fontsize=12)
            plt.ylabel('Estimated Lag τ (days)', fontsize=12)
            plt.title('Estimated Lag Parameters', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "estimated_lags.png", dpi=150)
            plt.close()
            print(f"  Saved estimated lags plot: {self.output_dir / 'estimated_lags.png'}")

    def generate_report(self, df: pd.DataFrame) -> None:
        """Generate comprehensive markdown report."""
        report_path = self.output_dir / "crosslagged_comparison_report.md"

        with open(report_path, 'w') as f:
            f.write("# Cross-Lagged Model Comparison Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Workout variables**: {', '.join(self.workout_vars)}\n")
            f.write(f"**Weight observations**: {len(self.df_weight)}\n\n")

            f.write("## Model Configuration\n\n")
            f.write("### Fixed Lag Model\n")
            f.write(f"- Lag values: {self.fixed_lags} days\n")
            f.write(f"- MCMC: {self.chains} chains, {self.iter_warmup} warmup, {self.iter_sampling} sampling\n\n")

            f.write("### Estimated Lag Model\n")
            f.write(f"- Lag prior: N({self.estimated_lag_prior_mean}, {self.estimated_lag_prior_sd}²) days\n")
            f.write(f"- MCMC: {self.chains} chains, {self.iter_warmup} warmup, {self.iter_sampling} sampling\n\n")

            f.write("### Cumulative Lag Model\n")
            f.write(f"- Lag window: {self.cumulative_window} days\n")
            f.write(f"- Lag step: {self.cumulative_step} days\n")
            f.write(f"- MCMC: {self.chains} chains, {self.iter_warmup} warmup, {self.iter_sampling} sampling\n\n")

            # Results summary
            f.write("## Results Summary\n\n")

            if not df.empty:
                # Best models by WAIC
                f.write("### Best Models by WAIC\n\n")
                f.write("| Variable | Best Model | WAIC | β | 95% CI | Lag | Weight Obs | Workout Obs |\n")
                f.write("|----------|------------|------|---|--------|-----|------------|-------------|\n")

                for var_name in df['variable'].unique():
                    df_var = df[df['variable'] == var_name]
                    if df_var['waic'].notna().any():
                        best_idx = df_var['waic'].idxmin()
                        best_row = df_var.loc[best_idx]

                        beta_ci = f"[{best_row['beta_ci_low']:.3f}, {best_row['beta_ci_high']:.3f}]"
                        lag_str = str(best_row.get('lag_value', best_row.get('lag_mean', 'N/A')))

                        f.write(f"| {var_name} | {best_row['model_type']} | {best_row['waic']:.1f} | "
                                f"{best_row['beta_mean']:.3f} | {beta_ci} | {lag_str} | "
                                f"{int(best_row['n_weight_obs'])} | {int(best_row['n_workout_obs'])} |\n")

                f.write("\n")

                # Model comparison
                f.write("### Model Comparison\n\n")
                f.write("Key findings:\n")
                f.write("\n")

                for var_name in df['variable'].unique():
                    df_var = df[df['variable'] == var_name]
                    if len(df_var) > 0:
                        f.write(f"#### {var_name}\n\n")

                        for _, row in df_var.iterrows():
                            model_type = row['model_type']
                            if model_type == 'fixed':
                                f.write(f"- **Fixed lag (τ={row['lag_value']} days)**: β = {row['beta_mean']:.3f} "
                                        f"[{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}], WAIC = {row['waic']:.1f}\n")
                            elif model_type == 'estimated':
                                f.write(f"- **Estimated lag**: τ = {row.get('lag_mean', 'N/A'):.2f} days, "
                                        f"β = {row['beta_mean']:.3f} [{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}], "
                                        f"WAIC = {row['waic']:.1f}\n")
                            elif model_type == 'cumulative':
                                f.write(f"- **Cumulative lag ({self.cumulative_window}d window)**: "
                                        f"β = {row['beta_mean']:.3f} [{row['beta_ci_low']:.3f}, {row['beta_ci_high']:.3f}], "
                                        f"WAIC = {row['waic']:.1f}\n")
                        f.write("\n")

            # Visualizations
            f.write("## Visualizations\n\n")

            # List available plots
            plot_files = [
                "waic_comparison.png",
                "estimated_lags.png",
            ]

            for var_name in self.workout_vars:
                plot_files.append(f"model_comparison_{var_name}.png")

            for plot_file in plot_files:
                plot_path = self.output_dir / plot_file
                if plot_path.exists():
                    f.write(f"### {plot_file.replace('_', ' ').replace('.png', '')}\n\n")
                    f.write(f"![{plot_file}]({plot_file})\n\n")

            # Interpretation
            f.write("## Interpretation\n\n")
            f.write("### Model Selection\n")
            f.write("1. **WAIC (Watanabe-Akaike Information Criterion)**: Lower values indicate better predictive performance.\n")
            f.write("2. **LOO (Leave-One-Out cross-validation)**: Alternative to WAIC, also lower is better.\n")
            f.write("3. **Parameter estimates**: Posterior means with 95% credible intervals.\n")
            f.write("\n")

            f.write("### Effect Interpretation\n")
            f.write("- **β > 0**: Workouts cause weight gain (likely muscle mass accumulation)\n")
            f.write("- **β < 0**: Workouts cause weight loss (likely fat loss)\n")
            f.write("- **β ≈ 0**: No clear causal effect detected\n")
            f.write("\n")

            f.write("### Lag Interpretation\n")
            f.write("- **τ = 0**: Immediate effect (same day)\n")
            f.write("- **τ = 1-3 days**: Short-term delayed effect\n")
            f.write("- **τ > 7 days**: Longer-term delayed effect\n")
            f.write("- **Cumulative model**: Average effect over multiple lag values\n")
            f.write("\n")

            f.write("---\n")
            f.write("*Report generated by compare_crosslagged_models.py*\n")

        print(f"Generated report: {report_path}")

    def run_comparison(self) -> None:
        """Run full comparison pipeline."""
        print("\n" + "=" * 70)
        print("CROSS-LAGGED MODEL COMPARISON")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Workout variables: {', '.join(self.workout_vars)}")
        print(f"Output directory: {self.output_dir}")
        print(f"MCMC: {self.chains} chains, {self.iter_warmup} warmup, {self.iter_sampling} sampling")
        print(f"Sparse GP: {self.use_sparse} ({self.n_inducing_points} inducing points)")
        print(f"Fixed lags: {self.fixed_lags}")
        print(f"Estimated lag prior: N({self.estimated_lag_prior_mean}, {self.estimated_lag_prior_sd}²) days")
        print(f"Cumulative window: {self.cumulative_window} days (step: {self.cumulative_step} days)")
        print("=" * 70)

        # Load data
        self.load_data()

        if not self.workout_data:
            print("ERROR: No workout data loaded. Exiting.")
            return

        # Run analyses
        print("\n" + "=" * 70)
        print("RUNNING ANALYSES")
        print("=" * 70)

        self.run_fixed_lag_analysis()
        self.run_estimated_lag_analysis()
        self.run_cumulative_lag_analysis()

        # Create comparison
        print("\n" + "=" * 70)
        print("COMPARING RESULTS")
        print("=" * 70)

        df = self.create_comparison_tables()

        if not df.empty:
            self.create_comparison_plots(df)
            self.generate_report(df)

        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {self.output_dir}")
        print(f"Report: {self.output_dir / 'crosslagged_comparison_report.md'}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare different cross-lagged modeling approaches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--workout-vars",
        type=str,
        required=True,
        help="Comma-separated list of workout/activity variables to analyze. "
             "Examples: 'strength_training,walking,cycling'"
    )

    # Model configurations
    parser.add_argument(
        "--fixed-lags",
        type=str,
        default="0,1,2,3,7",
        help="Comma-separated list of lag values for fixed lag model (days)"
    )

    parser.add_argument(
        "--estimated-lag-prior-mean",
        type=float,
        default=3.0,
        help="Prior mean for estimated lag parameter (days)"
    )

    parser.add_argument(
        "--estimated-lag-prior-sd",
        type=float,
        default=2.0,
        help="Prior standard deviation for estimated lag parameter (days)"
    )

    parser.add_argument(
        "--cumulative-window",
        type=int,
        default=7,
        help="Window size for cumulative lag effect (days)"
    )

    parser.add_argument(
        "--cumulative-step",
        type=float,
        default=1.0,
        help="Step between lags in cumulative window (days)"
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
        default="output/crosslagged_comparison",
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

    # Sparse GP
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation"
    )

    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP"
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

    # Visualization
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots (only create tables and reports)"
    )

    args = parser.parse_args()

    # Parse comma-separated lists
    workout_vars = parse_comma_list(args.workout_vars)
    fixed_lags = parse_comma_float_list(args.fixed_lags)

    if not workout_vars:
        parser.error("--workout-vars must contain at least one variable")

    # Create and run comparison
    comparison = CrossLaggedModelComparison(
        workout_vars=workout_vars,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fixed_lags=fixed_lags,
        estimated_lag_prior_mean=args.estimated_lag_prior_mean,
        estimated_lag_prior_sd=args.estimated_lag_prior_sd,
        cumulative_window=args.cumulative_window,
        cumulative_step=args.cumulative_step,
        chains=args.chains,
        iter_warmup=args.iter_warmup,
        iter_sampling=args.iter_sampling,
        use_sparse=not args.no_sparse,
        n_inducing_points=args.n_inducing_points,
        skip_plots=args.skip_plots,
        force_refit=args.force_refit,
        no_cache=args.no_cache,
    )

    comparison.run_comparison()


if __name__ == "__main__":
    main()