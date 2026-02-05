#!/usr/bin/env python3
"""Explore correlations between workouts and health metrics.

This script provides a simpler alternative to full cross-lagged analysis:
1. Loads workout and health data
2. Computes lagged correlations (0-7 day lags)
3. Identifies strongest relationships
4. Generates visualizations and summary reports

Usage:
    # Explore all health metrics for strength training
    python -m src.analysis.explore_health_workout_correlations \
        --workout-vars strength_training \
        --max-lag 7 \
        --output-dir output/health_correlations
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.workout import load_workout_data, prepare_workout_aggregates
from src.data.health_metrics import load_combined_health_data, get_available_health_metrics


def parse_comma_list(value: str) -> List[str]:
    """Parse comma-separated list string into list of strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class HealthWorkoutCorrelationAnalyzer:
    """Analyze correlations between workouts and health metrics."""

    def __init__(
        self,
        workout_vars: List[str],
        max_lag: int = 7,
        data_dir: str = "data",
        output_dir: str = "output/health_correlations",
        max_metrics: int = 20,
        min_observations: int = 50,
    ):
        """Initialize analyzer.

        Args:
            workout_vars: List of workout variables to analyze.
            max_lag: Maximum lag to compute correlations for (0-7 days).
            data_dir: Path to data directory.
            output_dir: Directory for output files.
            max_metrics: Maximum number of health metrics to analyze.
            min_observations: Minimum number of observations required.
        """
        self.workout_vars = workout_vars
        self.max_lag = max_lag
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_metrics = max_metrics
        self.min_observations = min_observations

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.workout_data = {}
        self.health_df = None
        self.available_metrics = None
        self.selected_metrics = []

        # Results storage
        self.correlation_results = {}

    def load_data(self) -> None:
        """Load all required data."""
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        # Load health metrics
        print("\nLoading health metrics...")
        self.health_df = load_combined_health_data(self.data_dir)
        print(f"  Loaded {len(self.health_df)} days of health data")
        print(f"  Available metrics: {len(self.health_df.columns)}")

        # Get available metrics
        self.available_metrics = get_available_health_metrics()

        # Select health metrics to analyze
        self._select_health_metrics()
        print(f"\nSelected {len(self.selected_metrics)} health metrics for analysis:")
        for metric in self.selected_metrics[:10]:  # Show first 10
            print(f"  - {metric}")
        if len(self.selected_metrics) > 10:
            print(f"  ... and {len(self.selected_metrics) - 10} more")

        # Load workout data for each variable
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

    def _select_health_metrics(self) -> None:
        """Select health metrics to analyze."""
        # Get all metrics from all categories
        all_metrics = []
        for category_metrics in self.available_metrics.get('categories', {}).values():
            all_metrics.extend(category_metrics)

        # Remove duplicates
        unique_metrics = list(dict.fromkeys(all_metrics))

        # Filter out non-numeric columns and date columns
        numeric_metrics = []
        for metric in unique_metrics:
            if metric in self.health_df.columns:
                if pd.api.types.is_numeric_dtype(self.health_df[metric]):
                    # Check for sufficient non-missing values
                    non_missing = self.health_df[metric].notna().sum()
                    if non_missing >= self.min_observations:
                        numeric_metrics.append(metric)

        # Limit to max_metrics
        self.selected_metrics = numeric_metrics[:self.max_metrics]

    def compute_correlations(self) -> None:
        """Compute lagged correlations between workouts and health metrics."""
        print("\n" + "=" * 70)
        print("COMPUTING CORRELATIONS")
        print("=" * 70)

        total_analyses = len(self.workout_vars) * len(self.selected_metrics)
        print(f"\nTotal analyses to compute: {total_analyses}")

        for workout_var in self.workout_vars:
            if workout_var not in self.workout_data:
                print(f"\nSkipping {workout_var}: no workout data")
                continue

            print(f"\n{'='*60}")
            print(f"Analyzing: {workout_var}")
            print(f"{'='*60}")

            # Get workout data
            workout_df = self.workout_data[workout_var].copy()
            workout_df = workout_df.rename(columns={'workout_count': 'workout'})

            # Merge with health data
            merged_df = pd.merge(
                self.health_df[['date'] + self.selected_metrics],
                workout_df[['date', 'workout']],
                on='date',
                how='left'
            )
            # Fill missing workouts with 0
            merged_df['workout'] = merged_df['workout'].fillna(0)

            # Store results for this workout variable
            self.correlation_results[workout_var] = {}

            for health_metric in self.selected_metrics:
                # Skip if insufficient data
                valid_data = merged_df[[health_metric, 'workout']].dropna()
                if len(valid_data) < self.min_observations:
                    continue

                # Compute correlations for different lags
                lag_results = {}
                for lag in range(self.max_lag + 1):
                    if lag == 0:
                        # Same-day correlation
                        corr, p_value = self._compute_correlation(
                            merged_df, 'workout', health_metric, lag=0
                        )
                    else:
                        # Lagged correlation: workout today → health metric tomorrow
                        corr, p_value = self._compute_correlation(
                            merged_df, 'workout', health_metric, lag=lag
                        )

                    if corr is not None:
                        lag_results[lag] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'n_obs': len(valid_data)
                        }

                if lag_results:
                    self.correlation_results[workout_var][health_metric] = lag_results

                    # Print strongest correlation
                    strongest_lag = max(
                        lag_results.items(),
                        key=lambda x: abs(x[1]['correlation'])
                    )
                    lag, result = strongest_lag
                    direction = "positive" if result['correlation'] > 0 else "negative"
                    sig = "**" if result['significant'] else ""
                    print(f"  {health_metric}: lag {lag}d, r = {result['correlation']:.3f} ({direction}){sig}")

        print(f"\nCompleted correlation analysis")

    def _compute_correlation(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        lag: int = 0
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute correlation between x and y with optional lag.

        For lag > 0: x(t) correlated with y(t+lag)
        """
        # Create lagged series
        if lag == 0:
            x_series = df[x_col]
            y_series = df[y_col]
        else:
            # Shift y forward by lag days
            x_series = df[x_col].iloc[:-lag] if lag > 0 else df[x_col]
            y_series = df[y_col].iloc[lag:] if lag > 0 else df[y_col]

        # Align series
        aligned_df = pd.DataFrame({
            'x': x_series.reset_index(drop=True),
            'y': y_series.reset_index(drop=True)
        }).dropna()

        if len(aligned_df) < self.min_observations:
            return None, None

        # Compute Pearson correlation
        try:
            corr, p_value = stats.pearsonr(aligned_df['x'], aligned_df['y'])
            return corr, p_value
        except Exception:
            return None, None

    def generate_summary_report(self) -> None:
        """Generate summary report of correlation findings."""
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY REPORT")
        print("=" * 70)

        if not self.correlation_results:
            print("No correlation results to summarize")
            return

        # Create summary dataframe
        summary_rows = []

        for workout_var, health_results in self.correlation_results.items():
            for health_metric, lag_results in health_results.items():
                # Find strongest correlation (absolute value)
                if not lag_results:
                    continue

                strongest = max(
                    lag_results.items(),
                    key=lambda x: abs(x[1]['correlation'])
                )
                lag, result = strongest

                summary_rows.append({
                    'workout_var': workout_var,
                    'health_metric': health_metric,
                    'strongest_lag': lag,
                    'correlation': result['correlation'],
                    'p_value': result['p_value'],
                    'significant': result['significant'],
                    'n_obs': result['n_obs'],
                    'direction': 'positive' if result['correlation'] > 0 else 'negative',
                    'abs_correlation': abs(result['correlation'])
                })

        # Create dataframe
        summary_df = pd.DataFrame(summary_rows)

        if summary_df.empty:
            print("No significant correlations found")
            return

        # Sort by absolute correlation strength
        summary_df = summary_df.sort_values('abs_correlation', ascending=False)

        # Save summary
        summary_path = self.output_dir / "health_workout_correlations.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved correlation summary to: {summary_path}")

        # Generate HTML report
        self._generate_html_report(summary_df)

        # Print top findings
        self._print_top_findings(summary_df)

        # Generate visualizations
        self._generate_visualizations(summary_df)

    def _generate_html_report(self, summary_df: pd.DataFrame) -> None:
        """Generate HTML report of findings."""
        # Group by significance
        significant = summary_df[summary_df['significant']]
        not_significant = summary_df[~summary_df['significant']]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Health-Workout Correlation Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                h3 {{ color: #777; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .positive {{ color: green; font-weight: bold; }}
                .negative {{ color: red; font-weight: bold; }}
                .sig-positive {{ background-color: #e8f8e8; }}
                .sig-negative {{ background-color: #f8e8e8; }}
            </style>
        </head>
        <body>
            <h1>Health-Workout Correlation Analysis</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Output directory: {self.output_dir}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total analyses: {len(summary_df)}</p>
                <p>Significant correlations (p < 0.05): {len(significant)}</p>
                <p>Workout variables: {', '.join(self.workout_vars)}</p>
                <p>Health metrics analyzed: {len(self.selected_metrics)}</p>
                <p>Maximum lag analyzed: {self.max_lag} days</p>
            </div>
        """

        # Add significant correlations table
        if not significant.empty:
            html_content += f"""
            <h2>Significant Correlations (p < 0.05)</h2>
            <table>
                <tr>
                    <th>Workout</th>
                    <th>Health Metric</th>
                    <th>Lag (days)</th>
                    <th>Correlation (r)</th>
                    <th>p-value</th>
                    <th>Observations</th>
                    <th>Interpretation</th>
                </tr>
            """

            for _, row in significant.iterrows():
                direction_class = "sig-positive" if row['correlation'] > 0 else "sig-negative"
                direction_text = "Positive" if row['correlation'] > 0 else "Negative"

                # Simple interpretation
                if 'sleep' in row['health_metric'].lower():
                    interpretation = "Workouts → better sleep" if row['correlation'] > 0 else "Workouts → worse sleep"
                elif 'stress' in row['health_metric'].lower():
                    interpretation = "Workouts → more stress" if row['correlation'] > 0 else "Workouts → less stress"
                elif 'heart' in row['health_metric'].lower():
                    interpretation = "Workouts → higher HR" if row['correlation'] > 0 else "Workouts → lower HR"
                else:
                    interpretation = f"Workouts → increase in {row['health_metric']}" if row['correlation'] > 0 else f"Workouts → decrease in {row['health_metric']}"

                html_content += f"""
                <tr class="{direction_class}">
                    <td>{row['workout_var']}</td>
                    <td>{row['health_metric']}</td>
                    <td>{row['strongest_lag']}</td>
                    <td>{row['correlation']:.3f}</td>
                    <td>{row['p_value']:.4f}</td>
                    <td>{row['n_obs']}</td>
                    <td>{interpretation}</td>
                </tr>
                """

            html_content += "</table>"

        # Add all correlations table (collapsible)
        html_content += f"""
            <h2>All Correlations</h2>
            <details>
                <summary>Click to show/hide all {len(summary_df)} correlations</summary>
                <table>
                    <tr>
                        <th>Workout</th>
                        <th>Health Metric</th>
                        <th>Lag</th>
                        <th>Correlation</th>
                        <th>p-value</th>
                        <th>Significant</th>
                    </tr>
        """

        for _, row in summary_df.iterrows():
            sig_class = "positive" if row['significant'] and row['correlation'] > 0 else \
                       "negative" if row['significant'] and row['correlation'] < 0 else ""
            sig_text = "Yes" if row['significant'] else "No"

            html_content += f"""
                <tr>
                    <td>{row['workout_var']}</td>
                    <td>{row['health_metric']}</td>
                    <td>{row['strongest_lag']}</td>
                    <td class="{sig_class}">{row['correlation']:.3f}</td>
                    <td>{row['p_value']:.4f}</td>
                    <td>{sig_text}</td>
                </tr>
            """

        html_content += """
                </table>
            </details>

            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>Positive correlation (r > 0)</strong>: Higher workout counts associated with higher health metric values</li>
                <li><strong>Negative correlation (r < 0)</strong>: Higher workout counts associated with lower health metric values</li>
                <li><strong>Lag</strong>: Days between workout and health measurement (0 = same day)</li>
                <li><strong>p-value < 0.05</strong>: Statistically significant relationship</li>
                <li><strong>Correlation strength</strong>:
                    <ul>
                        <li>|r| > 0.5: Strong relationship</li>
                        <li>0.3 < |r| < 0.5: Moderate relationship</li>
                        <li>0.1 < |r| < 0.3: Weak relationship</li>
                        <li>|r| < 0.1: Very weak or no relationship</li>
                    </ul>
                </li>
            </ul>

            <footer>
                <p>Report generated by explore_health_workout_correlations.py</p>
            </footer>
        </body>
        </html>
        """

        report_path = self.output_dir / "health_correlation_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"Generated HTML report: {report_path}")

    def _print_top_findings(self, summary_df: pd.DataFrame) -> None:
        """Print top findings from analysis."""
        print("\n" + "=" * 70)
        print("TOP FINDINGS")
        print("=" * 70)

        # Top positive correlations
        positive = summary_df[summary_df['correlation'] > 0].copy()
        if not positive.empty:
            positive = positive.sort_values('correlation', ascending=False)
            print("\nStrongest Positive Correlations:")
            for _, row in positive.head(5).iterrows():
                sig = "**" if row['significant'] else ""
                print(f"  {row['workout_var']} → {row['health_metric']} (lag {row['strongest_lag']}d): r = {row['correlation']:.3f}{sig}")

        # Top negative correlations
        negative = summary_df[summary_df['correlation'] < 0].copy()
        if not negative.empty:
            negative = negative.sort_values('correlation', ascending=True)
            print("\nStrongest Negative Correlations:")
            for _, row in negative.head(5).iterrows():
                sig = "**" if row['significant'] else ""
                print(f"  {row['workout_var']} → {row['health_metric']} (lag {row['strongest_lag']}d): r = {row['correlation']:.3f}{sig}")

        # Most significant correlations
        significant = summary_df[summary_df['significant']].copy()
        if not significant.empty:
            print(f"\nTotal significant correlations: {len(significant)}")
            print("Most significant (lowest p-values):")
            significant = significant.sort_values('p_value')
            for _, row in significant.head(5).iterrows():
                print(f"  {row['workout_var']} → {row['health_metric']}: p = {row['p_value']:.4f}, r = {row['correlation']:.3f}")

        print(f"\nComplete results saved to: {self.output_dir}/health_workout_correlations.csv")

    def _generate_visualizations(self, summary_df: pd.DataFrame) -> None:
        """Generate visualization plots."""
        print("\nGenerating visualizations...")

        # Create visualizations directory
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. Correlation strength by health metric category
        self._plot_correlation_strength(summary_df, viz_dir)

        # 2. Lag distribution
        self._plot_lag_distribution(summary_df, viz_dir)

        # 3. Top correlations heatmap
        self._plot_top_correlations_heatmap(summary_df, viz_dir)

        print(f"Visualizations saved to: {viz_dir}")

    def _plot_correlation_strength(self, summary_df: pd.DataFrame, viz_dir: Path) -> None:
        """Plot correlation strength by health metric category."""
        # Categorize health metrics
        def categorize_metric(metric: str) -> str:
            metric_lower = metric.lower()
            if any(keyword in metric_lower for keyword in ['sleep', 'rem']):
                return 'Sleep'
            elif 'stress' in metric_lower:
                return 'Stress'
            elif any(keyword in metric_lower for keyword in ['heart', 'hr']):
                return 'Heart'
            elif any(keyword in metric_lower for keyword in ['step', 'calori', 'distance']):
                return 'Activity'
            elif 'respir' in metric_lower:
                return 'Respiration'
            elif 'battery' in metric_lower:
                return 'Body Battery'
            else:
                return 'Other'

        summary_df['category'] = summary_df['health_metric'].apply(categorize_metric)

        # Create plot
        plt.figure(figsize=(12, 8))

        # Box plot of correlations by category
        categories = summary_df['category'].unique()
        data = [summary_df[summary_df['category'] == cat]['correlation'].values for cat in categories]

        plt.boxplot(data, labels=categories)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Correlation Strength by Health Metric Category')
        plt.ylabel('Correlation (r)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plot_path = viz_dir / "correlation_by_category.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _plot_lag_distribution(self, summary_df: pd.DataFrame, viz_dir: Path) -> None:
        """Plot distribution of optimal lags."""
        plt.figure(figsize=(10, 6))

        # Histogram of lags
        plt.hist(summary_df['strongest_lag'], bins=range(self.max_lag + 2),
                edgecolor='black', alpha=0.7)
        plt.title('Distribution of Optimal Lag Days')
        plt.xlabel('Lag (days)')
        plt.ylabel('Count')
        plt.xticks(range(self.max_lag + 1))

        plot_path = viz_dir / "lag_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _plot_top_correlations_heatmap(self, summary_df: pd.DataFrame, viz_dir: Path) -> None:
        """Create heatmap of top correlations."""
        # Get top 20 correlations by absolute value
        top_n = min(20, len(summary_df))
        top_df = summary_df.nlargest(top_n, 'abs_correlation').copy()

        # Create pivot table for heatmap
        pivot_data = top_df.pivot_table(
            values='correlation',
            index='health_metric',
            columns='workout_var',
            aggfunc='first'
        )

        if pivot_data.empty:
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, cbar_kws={'label': 'Correlation (r)'})
        plt.title(f'Top {top_n} Health-Workout Correlations')
        plt.tight_layout()

        plot_path = viz_dir / "top_correlations_heatmap.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore correlations between workouts and health metrics"
    )

    # Required arguments
    parser.add_argument(
        "--workout-vars",
        type=str,
        required=True,
        help="Comma-separated list of workout variables to analyze"
    )

    # Analysis parameters
    parser.add_argument(
        "--max-lag",
        type=int,
        default=7,
        help="Maximum lag to compute correlations for (default: 7)"
    )
    parser.add_argument(
        "--max-metrics",
        type=int,
        default=20,
        help="Maximum number of health metrics to analyze (default: 20)"
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=50,
        help="Minimum number of observations required (default: 50)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/health_correlations",
        help="Output directory for results (default: output/health_correlations)"
    )

    args = parser.parse_args()

    # Parse lists
    workout_vars = parse_comma_list(args.workout_vars)

    # Create analyzer
    analyzer = HealthWorkoutCorrelationAnalyzer(
        workout_vars=workout_vars,
        max_lag=args.max_lag,
        output_dir=args.output_dir,
        max_metrics=args.max_metrics,
        min_observations=args.min_observations,
    )

    try:
        # Run analysis
        analyzer.load_data()
        analyzer.compute_correlations()
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