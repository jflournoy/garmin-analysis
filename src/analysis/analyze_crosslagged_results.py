#!/usr/bin/env python3
"""Analyze and visualize cross-lagged model results.

This script loads results from cross-lagged model comparisons and creates
comprehensive visualizations and summary reports.

Usage:
    python -m src.analysis.analyze_crosslagged_results --results-dir output/full_comparison
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CrossLaggedResultsAnalyzer:
    """Analyze results from cross-lagged model comparisons."""

    def __init__(self, results_dir: str = "output/full_comparison"):
        """Initialize analyzer with results directory.

        Args:
            results_dir: Path to directory containing cross-lagged model results.
        """
        self.results_dir = Path(results_dir)

        # Check if directory exists
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {self.results_dir}")

        # Load comparison tables if they exist
        self.comparison_df = None
        self.summary_df = None

        # Store loaded results
        self.fixed_results = {}
        self.estimated_results = {}
        self.cumulative_results = {}

    def load_comparison_tables(self) -> None:
        """Load comparison tables from results directory."""
        raw_path = self.results_dir / "model_comparison_raw.csv"
        summary_path = self.results_dir / "analysis_summary.csv"

        if raw_path.exists():
            self.comparison_df = pd.read_csv(raw_path)
            print(f"Loaded comparison table: {len(self.comparison_df)} rows")
        else:
            print(f"Warning: Comparison table not found: {raw_path}")

        if summary_path.exists():
            self.summary_df = pd.read_csv(summary_path)
            print(f"Loaded summary table: {len(self.summary_df)} rows")
        else:
            print(f"Warning: Summary table not found: {summary_path}")

    def load_individual_results(self) -> None:
        """Load individual model results from subdirectories."""
        print("\n" + "=" * 70)
        print("LOADING INDIVIDUAL MODEL RESULTS")
        print("=" * 70)

        # Load fixed lag results
        fixed_dir = self.results_dir / "fixed_lag"
        if fixed_dir.exists():
            print(f"\nLoading fixed lag results from: {fixed_dir}")
            self._load_fixed_results(fixed_dir)

        # Load estimated lag results
        estimated_dir = self.results_dir / "estimated_lag"
        if estimated_dir.exists():
            print(f"\nLoading estimated lag results from: {estimated_dir}")
            self._load_estimated_results(estimated_dir)

        # Load cumulative lag results
        cumulative_dir = self.results_dir / "cumulative_lag"
        if cumulative_dir.exists():
            print(f"\nLoading cumulative lag results from: {cumulative_dir}")
            self._load_cumulative_results(cumulative_dir)

    def _load_fixed_results(self, fixed_dir: Path) -> None:
        """Load fixed lag model results."""
        for var_dir in fixed_dir.iterdir():
            if var_dir.is_dir():
                var_name = var_dir.name
                self.fixed_results[var_name] = {}

                for lag_dir in var_dir.iterdir():
                    if lag_dir.is_dir() and "lag_" in lag_dir.name:
                        # Extract lag value from directory name
                        lag_str = lag_dir.name.replace("lag_", "").replace("days", "")
                        try:
                            lag_value = float(lag_str)
                        except ValueError:
                            continue

                        # Load posterior summary
                        summary_path = lag_dir / "posterior_summary.csv"
                        if summary_path.exists():
                            try:
                                summary_df = pd.read_csv(summary_path)
                                self.fixed_results[var_name][lag_value] = summary_df
                            except Exception as e:
                                print(f"  Error loading {summary_path}: {e}")

    def _load_estimated_results(self, estimated_dir: Path) -> None:
        """Load estimated lag model results."""
        for var_dir in estimated_dir.iterdir():
            if var_dir.is_dir():
                var_name = var_dir.name

                # Load posterior summary
                summary_path = var_dir / "posterior_summary.csv"
                if summary_path.exists():
                    try:
                        summary_df = pd.read_csv(summary_path)
                        self.estimated_results[var_name] = summary_df
                    except Exception as e:
                        print(f"  Error loading {summary_path}: {e}")

    def _load_cumulative_results(self, cumulative_dir: Path) -> None:
        """Load cumulative lag model results."""
        for var_dir in cumulative_dir.iterdir():
            if var_dir.is_dir():
                var_name = var_dir.name

                # Load posterior summary
                summary_path = var_dir / "posterior_summary.csv"
                if summary_path.exists():
                    try:
                        summary_df = pd.read_csv(summary_path)
                        self.cumulative_results[var_name] = summary_df
                    except Exception as e:
                        print(f"  Error loading {summary_path}: {e}")

    def create_summary_report(self) -> pd.DataFrame:
        """Create comprehensive summary report."""
        print("\n" + "=" * 70)
        print("CREATING SUMMARY REPORT")
        print("=" * 70)

        if self.comparison_df is None or self.comparison_df.empty:
            print("No comparison data available.")
            return pd.DataFrame()

        # Create summary statistics
        summary_rows = []

        for var_name in self.comparison_df['variable'].unique():
            df_var = self.comparison_df[self.comparison_df['variable'] == var_name]

            # Basic statistics
            n_models = len(df_var)
            n_fixed = len(df_var[df_var['model_type'] == 'fixed'])
            n_estimated = len(df_var[df_var['model_type'] == 'estimated'])
            n_cumulative = len(df_var[df_var['model_type'] == 'cumulative'])

            # Effect sizes
            beta_means = df_var['beta_mean'].dropna()
            beta_positive = (beta_means > 0).sum()
            beta_negative = (beta_means < 0).sum()
            beta_zero = (beta_means.abs() < 0.1).sum()

            # Credible intervals
            ci_widths = df_var['beta_ci_high'] - df_var['beta_ci_low']
            avg_ci_width = ci_widths.mean() if not ci_widths.empty else np.nan

            # Model comparison
            if df_var['waic'].notna().any():
                best_waic = df_var['waic'].min()
                best_model = df_var.loc[df_var['waic'].idxmin(), 'model_type']
                best_beta = df_var.loc[df_var['waic'].idxmin(), 'beta_mean']
            else:
                best_waic = np.nan
                best_model = "N/A"
                best_beta = np.nan

            summary_rows.append({
                'variable': var_name,
                'n_models': n_models,
                'n_fixed': n_fixed,
                'n_estimated': n_estimated,
                'n_cumulative': n_cumulative,
                'beta_mean_avg': beta_means.mean(),
                'beta_std_avg': beta_means.std(),
                'beta_positive': beta_positive,
                'beta_negative': beta_negative,
                'beta_zero': beta_zero,
                'avg_ci_width': avg_ci_width,
                'best_model': best_model,
                'best_waic': best_waic,
                'best_beta': best_beta,
            })

        summary_df = pd.DataFrame(summary_rows)

        # Save summary report
        report_path = self.results_dir / "analysis_summary.csv"
        summary_df.to_csv(report_path, index=False)
        print(f"Saved summary report: {report_path}")

        # Print summary
        print("\nSummary Statistics:")
        print("-" * 100)
        print(f"{'Variable':<20} {'Models':<8} {'β avg':<10} {'β ±':<10} {'Best':<12} {'WAIC':<10} {'CI Width':<10}")
        print("-" * 100)

        for _, row in summary_df.iterrows():
            print(f"{row['variable']:<20} {row['n_models']:<8} {row['beta_mean_avg']:<10.3f} "
                  f"{row['beta_std_avg']:<10.3f} {row['best_model']:<12} {row['best_waic']:<10.1f} "
                  f"{row['avg_ci_width']:<10.3f}")

        return summary_df

    def create_visualizations(self) -> None:
        """Create comprehensive visualizations."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        if self.comparison_df is None or self.comparison_df.empty:
            print("No data available for visualizations.")
            return

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Create visualization directory
        viz_dir = self.results_dir / "analysis_visualizations"
        viz_dir.mkdir(exist_ok=True)

        # 1. Effect size comparison across variables
        self._plot_effect_sizes(viz_dir)

        # 2. Lag-effect relationship for fixed models
        self._plot_lag_effects(viz_dir)

        # 3. Model comparison by WAIC
        self._plot_model_comparison(viz_dir)

        # 4. Credible interval widths
        self._plot_ci_widths(viz_dir)

        # 5. Posterior distributions for key parameters
        self._plot_posterior_distributions(viz_dir)

        print(f"\nVisualizations saved to: {viz_dir}")

    def _plot_effect_sizes(self, viz_dir: Path) -> None:
        """Plot effect sizes across variables and models."""
        df = self.comparison_df

        plt.figure(figsize=(14, 8))

        # Create grouped bar chart
        variables = df['variable'].unique()
        model_types = df['model_type'].unique()

        x_pos = np.arange(len(variables))
        bar_width = 0.8 / len(model_types)

        for i, model_type in enumerate(model_types):
            df_model = df[df['model_type'] == model_type]

            # Align by variable
            beta_means = []
            beta_errors_low = []
            beta_errors_high = []

            for var_name in variables:
                df_var = df_model[df_model['variable'] == var_name]
                if not df_var.empty:
                    # For fixed models, take average across lags
                    if model_type == 'fixed':
                        beta_mean = df_var['beta_mean'].mean()
                        beta_ci_low = df_var['beta_ci_low'].mean()
                        beta_ci_high = df_var['beta_ci_high'].mean()
                    else:
                        beta_mean = df_var['beta_mean'].iloc[0]
                        beta_ci_low = df_var['beta_ci_low'].iloc[0]
                        beta_ci_high = df_var['beta_ci_high'].iloc[0]

                    beta_means.append(beta_mean)
                    beta_errors_low.append(beta_mean - beta_ci_low)
                    beta_errors_high.append(beta_ci_high - beta_mean)
                else:
                    beta_means.append(np.nan)
                    beta_errors_low.append(np.nan)
                    beta_errors_high.append(np.nan)

            # Plot bars with error bars
            offset = (i - len(model_types)/2 + 0.5) * bar_width
            plt.bar(x_pos + offset, beta_means, bar_width,
                   label=model_type, alpha=0.8)

            # Add error bars
            plt.errorbar(x_pos + offset, beta_means,
                        yerr=[beta_errors_low, beta_errors_high],
                        fmt='none', color='black', capsize=3, alpha=0.7)

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xticks(x_pos, variables, rotation=45)
        plt.xlabel('Workout Variable', fontsize=12)
        plt.ylabel('β (causal effect: workouts → weight)', fontsize=12)
        plt.title('Cross-Lagged Effect Sizes by Variable and Model Type', fontsize=14, fontweight='bold')
        plt.legend(title='Model Type')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(viz_dir / "effect_sizes_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved effect sizes plot: {viz_dir / 'effect_sizes_comparison.png'}")

    def _plot_lag_effects(self, viz_dir: Path) -> None:
        """Plot lag-effect relationship for fixed models."""
        df_fixed = self.comparison_df[self.comparison_df['model_type'] == 'fixed']

        if df_fixed.empty:
            return

        plt.figure(figsize=(12, 8))

        # Plot each variable separately
        variables = df_fixed['variable'].unique()

        for var_name in variables:
            df_var = df_fixed[df_fixed['variable'] == var_name]

            # Sort by lag
            df_var = df_var.sort_values('lag_value')

            # Plot with error bars
            plt.errorbar(df_var['lag_value'], df_var['beta_mean'],
                        yerr=[df_var['beta_mean'] - df_var['beta_ci_low'],
                              df_var['beta_ci_high'] - df_var['beta_mean']],
                        fmt='o-', capsize=5, capthick=2, linewidth=2,
                        label=var_name, alpha=0.8, markersize=8)

        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Lag τ (days)', fontsize=12)
        plt.ylabel('β (causal effect: workouts → weight)', fontsize=12)
        plt.title('Fixed Lag Model: Effect Size vs. Lag', fontsize=14, fontweight='bold')
        plt.legend(title='Workout Variable')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "lag_effect_relationship.png", dpi=150)
        plt.close()
        print(f"  Saved lag-effect plot: {viz_dir / 'lag_effect_relationship.png'}")

    def _plot_model_comparison(self, viz_dir: Path) -> None:
        """Plot model comparison by WAIC."""
        df = self.comparison_df

        if df['waic'].isna().all():
            return

        plt.figure(figsize=(12, 8))

        # Create pivot table for WAIC
        df_waic = df.pivot_table(
            index='variable',
            columns='model_type',
            values='waic',
            aggfunc='first'
        )

        # Plot WAIC as heatmap
        plt.imshow(df_waic.values, cmap='viridis', aspect='auto')
        plt.colorbar(label='WAIC (lower is better)')

        # Add labels
        plt.xticks(range(len(df_waic.columns)), df_waic.columns, rotation=45)
        plt.yticks(range(len(df_waic.index)), df_waic.index)
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Workout Variable', fontsize=12)
        plt.title('Model Comparison by WAIC', fontsize=14, fontweight='bold')

        # Add text values
        for i in range(len(df_waic.index)):
            for j in range(len(df_waic.columns)):
                waic_value = df_waic.iloc[i, j]
                if not np.isnan(waic_value):
                    plt.text(j, i, f'{waic_value:.1f}',
                            ha='center', va='center',
                            color='white' if waic_value > df_waic.values.mean() else 'black',
                            fontweight='bold')

        plt.tight_layout()
        plt.savefig(viz_dir / "waic_heatmap.png", dpi=150)
        plt.close()
        print(f"  Saved WAIC heatmap: {viz_dir / 'waic_heatmap.png'}")

    def _plot_ci_widths(self, viz_dir: Path) -> None:
        """Plot credible interval widths."""
        df = self.comparison_df

        # Calculate CI widths
        df['ci_width'] = df['beta_ci_high'] - df['beta_ci_low']

        plt.figure(figsize=(12, 8))

        # Group by variable and model type
        df_grouped = df.groupby(['variable', 'model_type'])['ci_width'].mean().unstack()

        # Plot as grouped bar chart
        df_grouped.plot(kind='bar', figsize=(12, 8))

        plt.xlabel('Workout Variable', fontsize=12)
        plt.ylabel('95% Credible Interval Width', fontsize=12)
        plt.title('Uncertainty in Effect Estimates by Variable and Model', fontsize=14, fontweight='bold')
        plt.legend(title='Model Type')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(viz_dir / "ci_widths_comparison.png", dpi=150)
        plt.close()
        print(f"  Saved CI widths plot: {viz_dir / 'ci_widths_comparison.png'}")

    def _plot_posterior_distributions(self, viz_dir: Path) -> None:
        """Plot posterior distributions for key parameters."""
        # This would require loading the full posterior samples
        # For now, we'll create a simpler version using summary statistics

        if not self.fixed_results and not self.estimated_results and not self.cumulative_results:
            return

        print("  Note: Full posterior distribution plots require loading posterior samples.")
        print("  Consider running with --load-samples flag when samples are available.")

    def generate_html_report(self) -> None:
        """Generate HTML report with interactive visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING HTML REPORT")
        print("=" * 70)

        if self.comparison_df is None or self.comparison_df.empty:
            print("No data available for HTML report.")
            return

        # Create HTML report
        report_path = self.results_dir / "analysis_report.html"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Lagged Model Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                h2 {{ color: #555; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .image {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Cross-Lagged Model Analysis Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Results directory: {self.results_dir}</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total variables analyzed: {len(self.comparison_df['variable'].unique())}</p>
                <p>Total model runs: {len(self.comparison_df)}</p>
                <p>Model types: Fixed lag, Estimated lag, Cumulative lag</p>
            </div>

            <h2>Best Models by WAIC</h2>
        """

        if self.summary_df is not None and not self.summary_df.empty:
            html_content += """
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Models</th>
                    <th>β Avg</th>
                    <th>β ±</th>
                    <th>β +</th>
                    <th>β -</th>
                    <th>Best Model</th>
                    <th>Avg CI Width</th>
                </tr>
            """

            for _, row in self.summary_df.iterrows():
                html_content += f"""
                <tr>
                    <td>{row['variable']}</td>
                    <td>{row['n_models']}</td>
                    <td>{row['beta_mean_avg']:.3f}</td>
                    <td>{row['beta_std_avg']:.3f}</td>
                    <td>{row['beta_positive']}</td>
                    <td>{row['beta_negative']}</td>
                    <td>{row['best_model']}</td>
                    <td>{row['avg_ci_width']:.3f}</td>
                </tr>
                """

            html_content += "</table>"

        # Add image references
        viz_dir = self.results_dir / "analysis_visualizations"
        if viz_dir.exists():
            html_content += "<h2>Visualizations</h2>"

            for img_file in viz_dir.glob("*.png"):
                img_name = img_file.name.replace("_", " ").replace(".png", "")
                html_content += f"""
                <h3>{img_name}</h3>
                <img class="image" src="{img_file.relative_to(self.results_dir)}" alt="{img_name}">
                """

        html_content += """
            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>β > 0</strong>: Workouts cause weight gain (likely muscle mass accumulation)</li>
                <li><strong>β < 0</strong>: Workouts cause weight loss (likely fat loss)</li>
                <li><strong>β ≈ 0</strong>: No clear causal effect detected</li>
                <li><strong>WAIC</strong>: Lower values indicate better predictive performance</li>
                <li><strong>Credible Interval Width</strong>: Narrower intervals indicate more precise estimates</li>
            </ul>

            <footer>
                <p>Report generated by analyze_crosslagged_results.py</p>
            </footer>
        </body>
        </html>
        """

        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"Generated HTML report: {report_path}")

    def run_analysis(self, load_individual: bool = False) -> None:
        """Run complete analysis pipeline.

        Args:
            load_individual: Whether to load individual model results (slower).
        """
        print("=" * 70)
        print("CROSS-LAGGED RESULTS ANALYSIS")
        print("=" * 70)
        print(f"Results directory: {self.results_dir}")
        print("=" * 70)

        # Load comparison tables
        self.load_comparison_tables()

        # Load individual results if requested
        if load_individual:
            self.load_individual_results()

        # Create summary report
        summary_df = self.create_summary_report()

        # Create visualizations
        self.create_visualizations()

        # Generate HTML report
        self.generate_html_report()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {self.results_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize cross-lagged model results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="output/full_comparison",
        help="Directory containing cross-lagged model results"
    )

    parser.add_argument(
        "--load-individual",
        action="store_true",
        help="Load individual model results (slower but more detailed)"
    )

    args = parser.parse_args()

    # Create and run analyzer
    analyzer = CrossLaggedResultsAnalyzer(args.results_dir)
    analyzer.run_analysis(load_individual=args.load_individual)


if __name__ == "__main__":
    main()