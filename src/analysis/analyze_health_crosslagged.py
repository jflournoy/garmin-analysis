#!/usr/bin/env python3
"""Analyze cross-lagged effects of workouts on health metrics.

This script extends cross-lagged analysis to health metrics:
- Sleep quality (duration, efficiency, stages)
- Stress levels (average, intensity)
- Resting heart rate
- Body Battery (charged/drained values)
- Respiration rates
- Activity metrics (steps, calories, distance)

For each health metric, it runs cross-lagged models to estimate:
1. How workouts affect health metrics with time lags
2. Optimal lag times for different health outcomes
3. Strength and direction of effects

Usage:
    # Analyze all health metrics for strength training
    python -m src.analysis.analyze_health_crosslagged \
        --workout-vars strength_training \
        --health-metrics sleep_efficiency,resting_heart_rate,avg_stress \
        --output-dir output/health_crosslagged

    # Analyze specific categories
    python -m src.analysis.analyze_health_crosslagged \
        --workout-vars strength_training,walking,cycling \
        --categories sleep,stress,heart \
        --output-dir output/health_full_analysis
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import load_weight_data
from src.data.workout import load_workout_data, prepare_workout_aggregates
from src.data.health_metrics import load_combined_health_data, get_available_health_metrics
from src.models.demo_bivariate import (
    run_fixed_lag_comparison,
    run_estimated_lag_analysis,
    run_cumulative_lag_analysis,
    _compute_ess,
    _compute_rhat,
)


def parse_comma_list(value: str) -> List[str]:
    """Parse comma-separated list string into list of strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class HealthCrossLaggedAnalyzer:
    """Analyze cross-lagged effects of workouts on health metrics."""

    def __init__(
        self,
        workout_vars: List[str],
        health_metrics: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        data_dir: str = "data",
        output_dir: str = "output/health_crosslagged",
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
        max_metrics: int = 10,
    ):
        """Initialize analyzer with configuration.

        Args:
            workout_vars: List of workout/activity variables to analyze.
            health_metrics: Specific health metrics to analyze (if None, uses categories).
            categories: Health metric categories to analyze (sleep, stress, heart, etc.).
            data_dir: Path to data directory.
            output_dir: Directory for output files.
            fixed_lags: List of fixed lag values to test.
            estimated_lag_prior_mean: Prior mean for estimated lag.
            estimated_lag_prior_sd: Prior standard deviation for estimated lag.
            cumulative_window: Window size for cumulative lag models.
            cumulative_step: Step size for cumulative lag models.
            chains: Number of MCMC chains.
            iter_warmup: Number of warmup iterations per chain.
            iter_sampling: Number of sampling iterations per chain.
            use_sparse: Whether to use sparse GP approximation.
            n_inducing_points: Number of inducing points for sparse GP.
            skip_plots: Skip generating plots.
            force_refit: Force refit even if cached results exist.
            no_cache: Disable caching of results.
            max_metrics: Maximum number of health metrics to analyze.
        """
        self.workout_vars = workout_vars
        self.health_metrics = health_metrics
        self.categories = categories
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
        self.max_metrics = max_metrics

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.weight_df = None
        self.workout_df = None
        self.health_df = None
        self.available_metrics = None

        # Store results
        self.results = {}

    def load_data(self) -> None:
        """Load all required data."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        # Load weight data
        print("\nLoading weight data...")
        self.weight_df = load_weight_data(self.data_dir)
        print(f"  Loaded {len(self.weight_df)} weight measurements")

        # Load health metrics
        print("\nLoading health metrics...")
        self.health_df = load_combined_health_data(self.data_dir)
        print(f"  Loaded {len(self.health_df)} days of health data")
        print(f"  Available metrics: {len(self.health_df.columns)}")

        # Get available metrics
        self.available_metrics = get_available_health_metrics()

        # Select health metrics to analyze
        self.selected_metrics = self._select_health_metrics()
        print(f"\nSelected {len(self.selected_metrics)} health metrics for analysis:")
        for metric in self.selected_metrics:
            print(f"  - {metric}")

        # Load workout data for each variable
        self.workout_data = {}
        for var_name in self.workout_vars:
            print(f"\nLoading workout data for '{var_name}'...")
            df_workouts_raw = load_workout_data(
                data_dir=self.data_dir,
                activity_type=var_name,
                include_exercise_details=False,
            )
            print(f"  Raw workout records: {len(df_workouts_raw)}")

            # Aggregate workouts to daily count
            df_workouts_agg = prepare_workout_aggregates(
                df_workouts_raw,
                aggregation="daily",
                metric="count"
            )
            print(f"  Workout days (with workouts): {len(df_workouts_agg)}")

            if len(df_workouts_agg) == 0:
                print(f"  WARNING: No workout data for '{var_name}'. Skipping variable.")
                continue

            self.workout_data[var_name] = df_workouts_agg

        # Merge data
        self._merge_data()

    def _select_health_metrics(self) -> List[str]:
        """Select health metrics to analyze based on user input."""
        all_metrics = []

        # Get all metrics from categories if specified
        if self.categories:
            for category in self.categories:
                if category in self.available_metrics.get('categories', {}):
                    all_metrics.extend(self.available_metrics['categories'][category])

        # Add specific metrics if provided
        if self.health_metrics:
            all_metrics.extend(self.health_metrics)

        # If neither categories nor specific metrics provided, use all
        if not all_metrics:
            for category_metrics in self.available_metrics.get('categories', {}).values():
                all_metrics.extend(category_metrics)

        # Remove duplicates and limit to max_metrics
        unique_metrics = list(dict.fromkeys(all_metrics))

        # Filter out non-numeric columns and date columns
        numeric_metrics = []
        for metric in unique_metrics:
            if metric in self.health_df.columns:
                if pd.api.types.is_numeric_dtype(self.health_df[metric]):
                    # Check for sufficient non-missing values
                    non_missing = self.health_df[metric].notna().sum()
                    if non_missing >= 100:  # At least 100 observations
                        numeric_metrics.append(metric)

        # Limit to max_metrics
        return numeric_metrics[:self.max_metrics]

    def _merge_data(self) -> None:
        """Merge weight, workout, and health data."""
        print("\nMerging data...")

        # Start with weight data
        self.merged_df = self.weight_df.copy()

        # Add workout aggregates for each variable
        for workout_var in self.workout_vars:
            if workout_var in self.workout_data:
                # Get workout data for this variable
                workout_df = self.workout_data[workout_var].copy()
                # Rename 'workout_count' to the variable name
                workout_df = workout_df.rename(columns={'workout_count': workout_var})

                # Merge workout data
                self.merged_df = pd.merge(
                    self.merged_df,
                    workout_df[['date', workout_var]],
                    on='date',
                    how='left'
                )
                # Fill missing with 0
                self.merged_df[workout_var] = self.merged_df[workout_var].fillna(0)
                print(f"  Added {workout_var}: {(self.merged_df[workout_var] > 0).sum()} workout days")

        # Add health metrics
        health_subset = self.health_df[['date'] + self.selected_metrics].copy()
        self.merged_df = pd.merge(
            self.merged_df,
            health_subset,
            on='date',
            how='left'
        )

        print(f"\n  Merged dataset: {len(self.merged_df)} rows")
        print(f"  Columns: {len(self.merged_df.columns)}")

        # Check for missing values
        missing = self.merged_df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            print(f"\n  Missing values:")
            for col, count in missing_cols.items():
                pct = count / len(self.merged_df) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")

    def run_analysis(self) -> None:
        """Run cross-lagged analysis for all workout-health metric pairs."""
        print("\n" + "=" * 70)
        print("RUNNING CROSS-LAGGED ANALYSIS")
        print("=" * 70)

        total_analyses = len(self.workout_vars) * len(self.selected_metrics)
        print(f"\nTotal analyses to run: {total_analyses}")
        print(f"Workout variables: {self.workout_vars}")
        print(f"Health metrics: {self.selected_metrics}")

        analysis_count = 0

        for workout_var in self.workout_vars:
            for health_metric in self.selected_metrics:
                analysis_count += 1
                print(f"\n{'='*60}")
                print(f"Analysis {analysis_count}/{total_analyses}: {workout_var} → {health_metric}")
                print(f"{'='*60}")

                try:
                    result = self._analyze_pair(workout_var, health_metric)
                    self.results[f"{workout_var}_{health_metric}"] = result
                except Exception as e:
                    print(f"  Error analyzing {workout_var} → {health_metric}: {e}")
                    continue

        print(f"\nCompleted {len(self.results)}/{total_analyses} analyses")

    def _analyze_pair(self, workout_var: str, health_metric: str) -> Dict[str, Any]:
        """Analyze cross-lagged effect for a single workout-health metric pair."""

        # Prepare data for this pair
        pair_df = self.merged_df[['date', workout_var, health_metric]].copy()
        pair_df = pair_df.dropna(subset=[workout_var, health_metric])

        if len(pair_df) < 50:
            raise ValueError(f"Insufficient data: only {len(pair_df)} complete observations")

        print(f"  Data: {len(pair_df)} complete observations")
        print(f"  Workout days: {(pair_df[workout_var] > 0).sum()}")
        print(f"  Health metric range: [{pair_df[health_metric].min():.2f}, {pair_df[health_metric].max():.2f}]")

        # Create output directory for this pair
        pair_dir = self.output_dir / f"{workout_var}_{health_metric}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Run fixed lag comparison
        print(f"  Running fixed lag comparison...")
        fixed_results = run_fixed_lag_comparison(
            x=pair_df[workout_var].values,
            y=pair_df[health_metric].values,
            dates=pair_df['date'].values,
            lag_values=self.fixed_lags,
            output_dir=str(pair_dir / "fixed_lag"),
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            use_sparse=self.use_sparse,
            n_inducing_points=self.n_inducing_points,
            skip_plots=self.skip_plots,
            force_refit=self.force_refit,
            no_cache=self.no_cache,
        )

        # Run estimated lag analysis
        print(f"  Running estimated lag analysis...")
        estimated_results = run_estimated_lag_analysis(
            x=pair_df[workout_var].values,
            y=pair_df[health_metric].values,
            dates=pair_df['date'].values,
            lag_prior_mean=self.estimated_lag_prior_mean,
            lag_prior_sd=self.estimated_lag_prior_sd,
            output_dir=str(pair_dir / "estimated_lag"),
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            use_sparse=self.use_sparse,
            n_inducing_points=self.n_inducing_points,
            skip_plots=self.skip_plots,
            force_refit=self.force_refit,
            no_cache=self.no_cache,
        )

        # Run cumulative lag analysis
        print(f"  Running cumulative lag analysis...")
        cumulative_results = run_cumulative_lag_analysis(
            x=pair_df[workout_var].values,
            y=pair_df[health_metric].values,
            dates=pair_df['date'].values,
            window=self.cumulative_window,
            step=self.cumulative_step,
            output_dir=str(pair_dir / "cumulative_lag"),
            chains=self.chains,
            iter_warmup=self.iter_warmup,
            iter_sampling=self.iter_sampling,
            use_sparse=self.use_sparse,
            n_inducing_points=self.n_inducing_points,
            skip_plots=self.skip_plots,
            force_refit=self.force_refit,
            no_cache=self.no_cache,
        )

        # Compile results
        result = {
            'workout_var': workout_var,
            'health_metric': health_metric,
            'n_obs': len(pair_df),
            'n_workout_days': (pair_df[workout_var] > 0).sum(),
            'health_mean': pair_df[health_metric].mean(),
            'health_std': pair_df[health_metric].std(),
            'fixed_results': fixed_results,
            'estimated_results': estimated_results,
            'cumulative_results': cumulative_results,
            'analysis_date': datetime.now().isoformat(),
        }

        # Save individual results
        result_path = pair_dir / "analysis_results.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  Results saved to: {result_path}")

        return result

    def generate_summary_report(self) -> None:
        """Generate summary report of all analyses."""
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY REPORT")
        print("=" * 70)

        if not self.results:
            print("No results to summarize")
            return

        # Create summary dataframe
        summary_rows = []

        for key, result in self.results.items():
            # Extract key metrics
            workout_var = result['workout_var']
            health_metric = result['health_metric']

            # Get effect sizes from different models
            fixed_betas = []
            if result['fixed_results']:
                for lag_result in result['fixed_results'].values():
                    if 'beta_mean' in lag_result:
                        fixed_betas.append(lag_result['beta_mean'])

            estimated_beta = None
            if result['estimated_results'] and 'beta_mean' in result['estimated_results']:
                estimated_beta = result['estimated_results']['beta_mean']

            cumulative_beta = None
            if result['cumulative_results'] and 'beta_mean' in result['cumulative_results']:
                cumulative_beta = result['cumulative_results']['beta_mean']

            # Calculate summary statistics
            all_betas = [b for b in fixed_betas if b is not None]
            if estimated_beta is not None:
                all_betas.append(estimated_beta)
            if cumulative_beta is not None:
                all_betas.append(cumulative_beta)

            if all_betas:
                beta_mean = np.mean(all_betas)
                beta_std = np.std(all_betas)
                beta_positive = sum(1 for b in all_betas if b > 0)
                beta_negative = sum(1 for b in all_betas if b < 0)
                beta_zero = sum(1 for b in all_betas if abs(b) < 0.1)
            else:
                beta_mean = beta_std = np.nan
                beta_positive = beta_negative = beta_zero = 0

            summary_rows.append({
                'workout_var': workout_var,
                'health_metric': health_metric,
                'n_obs': result['n_obs'],
                'n_workout_days': result['n_workout_days'],
                'health_mean': result['health_mean'],
                'health_std': result['health_std'],
                'beta_mean': beta_mean,
                'beta_std': beta_std,
                'beta_positive': beta_positive,
                'beta_negative': beta_negative,
                'beta_zero': beta_zero,
                'n_models': len(all_betas),
            })

        # Create dataframe
        summary_df = pd.DataFrame(summary_rows)

        # Save summary
        summary_path = self.output_dir / "health_crosslagged_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary to: {summary_path}")

        # Generate HTML report
        self._generate_html_report(summary_df)

        # Print top findings
        self._print_top_findings(summary_df)

    def _generate_html_report(self, summary_df: pd.DataFrame) -> None:
        """Generate HTML report of findings."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Health Cross-Lagged Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .neutral {{ color: gray; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Health Cross-Lagged Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Output directory: {self.output_dir}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total analyses: {len(summary_df)}</p>
                <p>Workout variables: {', '.join(self.workout_vars)}</p>
                <p>Health metrics analyzed: {len(self.selected_metrics)}</p>
                <p>Date range: {self.available_metrics['date_range']['start']} to {self.available_metrics['date_range']['end']}</p>
            </div>

            <h2>Results Summary</h2>
            <table>
                <tr>
                    <th>Workout</th>
                    <th>Health Metric</th>
                    <th>Observations</th>
                    <th>Workout Days</th>
                    <th>Health Mean</th>
                    <th>β Mean</th>
                    <th>β ±</th>
                    <th>β +</th>
                    <th>β -</th>
                    <th>Effect Direction</th>
                </tr>
        """

        for _, row in summary_df.iterrows():
            # Determine effect direction
            if row['beta_mean'] > 0.1:
                direction_class = "positive"
                direction_text = "Positive"
            elif row['beta_mean'] < -0.1:
                direction_class = "negative"
                direction_text = "Negative"
            else:
                direction_class = "neutral"
                direction_text = "Neutral"

            html_content += f"""
                <tr>
                    <td>{row['workout_var']}</td>
                    <td>{row['health_metric']}</td>
                    <td>{row['n_obs']}</td>
                    <td>{row['n_workout_days']}</td>
                    <td>{row['health_mean']:.2f}</td>
                    <td>{row['beta_mean']:.3f}</td>
                    <td>{row['beta_std']:.3f}</td>
                    <td>{row['beta_positive']}</td>
                    <td>{row['beta_negative']}</td>
                    <td class="{direction_class}">{direction_text}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>β > 0</strong>: Workouts cause increase in health metric</li>
                <li><strong>β < 0</strong>: Workouts cause decrease in health metric</li>
                <li><strong>β ≈ 0</strong>: No clear effect detected</li>
                <li><strong>Effect examples</strong>:
                    <ul>
                        <li>Sleep efficiency: Positive β = workouts improve sleep</li>
                        <li>Resting heart rate: Negative β = workouts lower resting HR</li>
                        <li>Stress: Negative β = workouts reduce stress</li>
                        <li>Body Battery charged: Positive β = workouts increase energy</li>
                    </ul>
                </li>
            </ul>

            <footer>
                <p>Report generated by analyze_health_crosslagged.py</p>
            </footer>
        </body>
        </html>
        """

        report_path = self.output_dir / "health_crosslagged_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"Generated HTML report: {report_path}")

    def _print_top_findings(self, summary_df: pd.DataFrame) -> None:
        """Print top findings from analysis."""
        print("\n" + "=" * 70)
        print("TOP FINDINGS")
        print("=" * 70)

        if summary_df.empty:
            print("No results to analyze")
            return

        # Find strongest positive effects
        positive_effects = summary_df[summary_df['beta_mean'] > 0.1].copy()
        if not positive_effects.empty:
            positive_effects = positive_effects.sort_values('beta_mean', ascending=False)
            print("\nStrongest Positive Effects (workouts increase health metric):")
            for _, row in positive_effects.head(5).iterrows():
                print(f"  {row['workout_var']} → {row['health_metric']}: β = {row['beta_mean']:.3f} ± {row['beta_std']:.3f}")

        # Find strongest negative effects
        negative_effects = summary_df[summary_df['beta_mean'] < -0.1].copy()
        if not negative_effects.empty:
            negative_effects = negative_effects.sort_values('beta_mean', ascending=True)
            print("\nStrongest Negative Effects (workouts decrease health metric):")
            for _, row in negative_effects.head(5).iterrows():
                print(f"  {row['workout_var']} → {row['health_metric']}: β = {row['beta_mean']:.3f} ± {row['beta_std']:.3f}")

        # Find most consistent effects
        summary_df['consistency'] = summary_df.apply(
            lambda row: abs(row['beta_positive'] - row['beta_negative']) / row['n_models']
            if row['n_models'] > 0 else 0,
            axis=1
        )
        consistent_effects = summary_df[summary_df['consistency'] > 0.7].copy()
        if not consistent_effects.empty:
            consistent_effects = consistent_effects.sort_values('consistency', ascending=False)
            print("\nMost Consistent Effects (same direction across models):")
            for _, row in consistent_effects.head(5).iterrows():
                direction = "positive" if row['beta_positive'] > row['beta_negative'] else "negative"
                print(f"  {row['workout_var']} → {row['health_metric']}: {direction} in {row['consistency']:.0%} of models")

        print(f"\nComplete results saved to: {self.output_dir}/health_crosslagged_summary.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze cross-lagged effects of workouts on health metrics"
    )

    # Required arguments
    parser.add_argument(
        "--workout-vars",
        type=str,
        required=True,
        help="Comma-separated list of workout variables to analyze"
    )

    # Health metric selection
    parser.add_argument(
        "--health-metrics",
        type=str,
        default="",
        help="Comma-separated list of specific health metrics to analyze"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated list of health metric categories (sleep,stress,heart,activity,respiration,body_battery)"
    )
    parser.add_argument(
        "--max-metrics",
        type=int,
        default=10,
        help="Maximum number of health metrics to analyze (default: 10)"
    )

    # Model configuration
    parser.add_argument(
        "--fixed-lags",
        type=str,
        default="0,1,2,3,7",
        help="Comma-separated list of fixed lag values (default: 0,1,2,3,7)"
    )
    parser.add_argument(
        "--estimated-lag-prior-mean",
        type=float,
        default=3.0,
        help="Prior mean for estimated lag (default: 3.0)"
    )
    parser.add_argument(
        "--estimated-lag-prior-sd",
        type=float,
        default=2.0,
        help="Prior standard deviation for estimated lag (default: 2.0)"
    )
    parser.add_argument(
        "--cumulative-window",
        type=int,
        default=7,
        help="Window size for cumulative lag models (default: 7)"
    )
    parser.add_argument(
        "--cumulative-step",
        type=float,
        default=1.0,
        help="Step size for cumulative lag models (default: 1.0)"
    )

    # MCMC settings
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)"
    )
    parser.add_argument(
        "--iter-warmup",
        type=int,
        default=500,
        help="Number of warmup iterations per chain (default: 500)"
    )
    parser.add_argument(
        "--iter-sampling",
        type=int,
        default=500,
        help="Number of sampling iterations per chain (default: 500)"
    )

    # GP settings
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse GP approximation"
    )
    parser.add_argument(
        "--n-inducing-points",
        type=int,
        default=50,
        help="Number of inducing points for sparse GP (default: 50)"
    )

    # Output and caching
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/health_crosslagged",
        help="Output directory for results (default: output/health_crosslagged)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Force refit even if cached results exist"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of results"
    )

    args = parser.parse_args()

    # Parse lists
    workout_vars = parse_comma_list(args.workout_vars)
    health_metrics = parse_comma_list(args.health_metrics)
    categories = parse_comma_list(args.categories)
    fixed_lags = [float(x) for x in parse_comma_list(args.fixed_lags)]

    # Create analyzer
    analyzer = HealthCrossLaggedAnalyzer(
        workout_vars=workout_vars,
        health_metrics=health_metrics if health_metrics else None,
        categories=categories if categories else None,
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
        max_metrics=args.max_metrics,
    )

    try:
        # Run analysis
        analyzer.load_data()
        analyzer.run_analysis()
        analyzer.generate_summary_report()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {analyzer.output_dir}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()