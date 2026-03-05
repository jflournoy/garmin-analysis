#!/usr/bin/env python3
"""Visualize fitness time series from four-fitness model with credible intervals.

This script:
1. Loads the four-fitness model results
2. Extracts fitness state time series with credible intervals
3. Creates visualizations showing how fitness perturbations map to weight
4. Generates an interactive HTML report with time series plots
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pio.templates.default = "plotly_white"


class FitnessTimeSeriesVisualizer:
    """Visualize fitness time series from four-fitness model."""

    def __init__(self, output_dir: str = "output/fitness_time_series"):
        """Initialize visualizer.

        Args:
            output_dir: Path to directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model results
        self.idata = self._load_inference_data()
        self.standardization = self._load_standardization()
        self.df_daily = self._load_daily_data()

        if self.idata is None:
            raise ValueError("Could not load inference data. Model may need to be re-run.")

    def _load_inference_data(self):
        """Load ArviZ InferenceData from four-fitness model."""
        import pickle

        # Try to load from the new full output directory first
        full_output_dir = Path("output/four_fitness_full")

        # Check for pickle file
        pickle_path = full_output_dir / "inference_data.pkl"
        if pickle_path.exists():
            print(f"Loading from pickle: {pickle_path}")
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            return data['idata']

        # Check for NetCDF file
        nc_path = full_output_dir / "inference_data.nc"
        if nc_path.exists():
            print(f"Loading from NetCDF: {nc_path}")
            return az.from_netcdf(str(nc_path))

        # Try to load from the previous analysis directory
        prev_output_dir = Path("output/four_fitness_analysis")
        pickle_path = prev_output_dir / "inference_data.pkl"
        if pickle_path.exists():
            print(f"Loading from pickle: {pickle_path}")
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            return data['idata']

        print("No inference data found. Model may need to be re-run with run_four_fitness_with_states.py")
        return None

    def _load_standardization(self):
        """Load standardization information."""
        std_path = Path("output/four_fitness_analysis/standardization.json")
        if std_path.exists():
            with open(std_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_daily_data(self):
        """Load daily data used in the model."""
        # This would need to match how the model was run
        # For now, return empty dataframe
        return pd.DataFrame()

    def extract_fitness_time_series(self):
        """Extract fitness state time series from model."""
        print("\nExtracting fitness time series...")

        # Check what variables are available
        print("Available variables in posterior:")
        for var in self.idata.posterior.data_vars:
            print(f"  {var}: {self.idata.posterior[var].dims}")

        # Look for fitness state variables
        fitness_vars = [
            'fitness_a_short_stored', 'fitness_s_short_stored',
            'fitness_a_long_stored', 'fitness_s_long_stored'
        ]

        available_vars = [v for v in fitness_vars if v in self.idata.posterior]
        print(f"\nFound {len(available_vars)} fitness variables: {available_vars}")

        if not available_vars:
            print("No fitness state variables found in posterior.")
            print("Model may not have been compiled with stored states.")
            return None

        # Extract time series
        fitness_series = {}
        for var in available_vars:
            # Get posterior samples: shape should be (chain, draw, time)
            samples = self.idata.posterior[var].values

            # Reshape to (sample, time)
            n_chains, n_draws, n_time = samples.shape
            samples_flat = samples.reshape(-1, n_time)

            # Compute statistics
            mean = samples_flat.mean(axis=0)
            std = samples_flat.std(axis=0)
            ci_low = np.percentile(samples_flat, 2.5, axis=0)
            ci_high = np.percentile(samples_flat, 97.5, axis=0)

            fitness_series[var] = {
                'mean': mean,
                'std': std,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'samples': samples_flat
            }

            print(f"  {var}: {n_time} time points, {samples_flat.shape[0]} samples")

        return fitness_series

    def compute_weight_contributions(self, fitness_series):
        """Compute weight contributions from each fitness component."""
        print("\nComputing weight contributions...")

        # Get gamma parameters
        gamma_params = {
            'gamma_a_short': None, 'gamma_s_short': None,
            'gamma_a_long': None, 'gamma_s_long': None
        }

        for param in gamma_params:
            if param in self.idata.posterior:
                samples = self.idata.posterior[param].values.flatten()
                gamma_params[param] = {
                    'mean': samples.mean(),
                    'std': samples.std(),
                    'samples': samples
                }
                print(f"  {param}: mean={samples.mean():.3f}, std={samples.std():.3f}")

        # Compute weight contributions
        weight_contributions = {}

        for fitness_var, fitness_data in fitness_series.items():
            # Map fitness variable to gamma parameter
            if 'a_short' in fitness_var:
                gamma_key = 'gamma_a_short'
                label = 'Aerobic Short-term'
            elif 's_short' in fitness_var:
                gamma_key = 'gamma_s_short'
                label = 'Strength Short-term'
            elif 'a_long' in fitness_var:
                gamma_key = 'gamma_a_long'
                label = 'Aerobic Long-term'
            elif 's_long' in fitness_var:
                gamma_key = 'gamma_s_long'
                label = 'Strength Long-term'
            else:
                continue

            if gamma_params[gamma_key] is None:
                print(f"  Warning: No {gamma_key} parameter found")
                continue

            # Get gamma samples
            gamma_samples = gamma_params[gamma_key]['samples']
            fitness_samples = fitness_data['samples']

            # Ensure compatible dimensions
            n_gamma = len(gamma_samples)
            n_fitness_samples, n_time = fitness_samples.shape

            # Use first n_gamma fitness samples if needed
            if n_fitness_samples > n_gamma:
                fitness_samples = fitness_samples[:n_gamma, :]
            elif n_gamma > n_fitness_samples:
                gamma_samples = gamma_samples[:n_fitness_samples]

            # Compute contribution: gamma * fitness
            n_samples = min(n_gamma, n_fitness_samples)
            contrib_samples = np.zeros((n_samples, n_time))

            for i in range(n_samples):
                contrib_samples[i, :] = gamma_samples[i] * fitness_samples[i, :]

            # Compute statistics
            mean = contrib_samples.mean(axis=0)
            std = contrib_samples.std(axis=0)
            ci_low = np.percentile(contrib_samples, 2.5, axis=0)
            ci_high = np.percentile(contrib_samples, 97.5, axis=0)

            weight_contributions[label] = {
                'mean': mean,
                'std': std,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'samples': contrib_samples,
                'fitness_var': fitness_var,
                'gamma_mean': gamma_params[gamma_key]['mean']
            }

            print(f"  {label}: contribution range [{mean.min():.3f}, {mean.max():.3f}]")

        return weight_contributions

    def create_time_series_plots(self, fitness_series, weight_contributions):
        """Create time series plots of fitness states and weight contributions."""
        print("\nCreating time series plots...")

        if not fitness_series:
            print("No fitness series data to plot")
            return

        # Create time index (days)
        n_time = len(next(iter(fitness_series.values()))['mean'])
        time_idx = np.arange(n_time)

        # 1. Plot fitness states
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
        axes1 = axes1.flatten()

        plot_idx = 0
        for var_name, data in fitness_series.items():
            if plot_idx >= len(axes1):
                break

            ax = axes1[plot_idx]

            # Plot mean with credible interval
            ax.fill_between(time_idx, data['ci_low'], data['ci_high'],
                           alpha=0.3, label='95% CI')
            ax.plot(time_idx, data['mean'], linewidth=2, label='Mean')

            # Formatting
            title = var_name.replace('_', ' ').replace('stored', '').strip().title()
            ax.set_title(f'Fitness State: {title}')
            ax.set_xlabel('Day')
            ax.set_ylabel('Fitness (standardized)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Set y-axis limits based on data range with some padding
            y_min = min(data['ci_low'].min(), data['mean'].min())
            y_max = max(data['ci_high'].max(), data['mean'].max())
            padding = (y_max - y_min) * 0.1  # 10% padding
            ax.set_ylim(y_min - padding, y_max + padding)

            plot_idx += 1

        # Remove empty subplots
        for i in range(plot_idx, len(axes1)):
            fig1.delaxes(axes1[i])

        plt.tight_layout()
        fig1.savefig(self.output_dir / "fitness_states_time_series.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  Fitness states plot saved: {self.output_dir / 'fitness_states_time_series.png'}")

        # 2. Plot weight contributions
        if weight_contributions:
            fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
            axes2 = axes2.flatten()

            plot_idx = 0
            for label, data in weight_contributions.items():
                if plot_idx >= len(axes2):
                    break

                ax = axes2[plot_idx]

                # Plot contribution with credible interval
                ax.fill_between(time_idx, data['ci_low'], data['ci_high'],
                               alpha=0.3, label='95% CI')
                ax.plot(time_idx, data['mean'], linewidth=2, label='Mean')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero')

                # Formatting
                ax.set_title(f'Weight Contribution: {label}\n(γ = {data["gamma_mean"]:.3f})')
                ax.set_xlabel('Day')
                ax.set_ylabel('Weight Contribution (standardized)')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Set y-axis limits based on data range with some padding
                y_min = min(data['ci_low'].min(), data['mean'].min(), 0)
                y_max = max(data['ci_high'].max(), data['mean'].max(), 0)
                padding = max((y_max - y_min) * 0.1, 0.1)  # 10% padding, minimum 0.1
                ax.set_ylim(y_min - padding, y_max + padding)

                plot_idx += 1

            # Remove empty subplots
            for i in range(plot_idx, len(axes2)):
                fig2.delaxes(axes2[i])

            plt.tight_layout()
            fig2.savefig(self.output_dir / "weight_contributions_time_series.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Weight contributions plot saved: {self.output_dir / 'weight_contributions_time_series.png'}")

        # 3. Combined plot
        fig3, ax3 = plt.subplots(figsize=(15, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, len(weight_contributions)))

        for (label, data), color in zip(weight_contributions.items(), colors):
            ax3.plot(time_idx, data['mean'], label=label, linewidth=2, color=color)
            ax3.fill_between(time_idx, data['ci_low'], data['ci_high'],
                           alpha=0.2, color=color)

        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Combined Weight Contributions from Fitness Components')
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Weight Contribution (standardized)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Set y-axis limits for combined plot
        all_ci_low = np.concatenate([data['ci_low'] for data in weight_contributions.values()])
        all_ci_high = np.concatenate([data['ci_high'] for data in weight_contributions.values()])
        all_mean = np.concatenate([data['mean'] for data in weight_contributions.values()])

        y_min = min(all_ci_low.min(), all_mean.min(), 0)
        y_max = max(all_ci_high.max(), all_mean.max(), 0)
        padding = max((y_max - y_min) * 0.1, 0.1)  # 10% padding, minimum 0.1
        ax3.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()
        fig3.savefig(self.output_dir / "combined_contributions.png", dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Combined contributions plot saved: {self.output_dir / 'combined_contributions.png'}")

    def create_interactive_report(self, fitness_series, weight_contributions):
        """Create interactive HTML report with Plotly."""
        print("\nCreating interactive HTML report...")

        if not fitness_series:
            print("No data for interactive report")
            return

        # Create time index
        n_time = len(next(iter(fitness_series.values()))['mean'])
        time_idx = np.arange(n_time)

        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Aerobic Short-term Fitness',
                'Strength Short-term Fitness',
                'Aerobic Long-term Fitness',
                'Strength Long-term Fitness'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Plot fitness states
        row_col_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        fitness_labels = {
            'fitness_a_short_stored': 'Aerobic Short-term',
            'fitness_s_short_stored': 'Strength Short-term',
            'fitness_a_long_stored': 'Aerobic Long-term',
            'fitness_s_long_stored': 'Strength Long-term'
        }

        for (var_name, label), (row, col) in zip(fitness_labels.items(), row_col_positions):
            if var_name in fitness_series:
                data = fitness_series[var_name]

                # Add mean line
                fig.add_trace(
                    go.Scatter(
                        x=time_idx,
                        y=data['mean'],
                        mode='lines',
                        name=f'{label} (Mean)',
                        line=dict(width=2),
                        legendgroup=label
                    ),
                    row=row, col=col
                )

                # Add credible interval
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_idx, time_idx[::-1]]),
                        y=np.concatenate([data['ci_high'], data['ci_low'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(0, 100, 255, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        name=f'{label} (95% CI)',
                        showlegend=False,
                        legendgroup=label
                    ),
                    row=row, col=col
                )

        # Update layout
        fig.update_layout(
            title_text="Four-Fitness Model: Fitness State Time Series",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes
        for i in range(1, 5):
            fig.update_xaxes(title_text="Day", row=(i+1)//2, col=2 if i%2==0 else 1)
            fig.update_yaxes(title_text="Fitness (standardized)", row=(i+1)//2, col=2 if i%2==0 else 1)

        # Save interactive plot
        html_path = self.output_dir / "fitness_time_series_interactive.html"
        fig.write_html(str(html_path))
        print(f"  Interactive plot saved: {html_path}")

        # Create weight contributions interactive plot
        if weight_contributions:
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=list(weight_contributions.keys()),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            for (label, data), (row, col), color in zip(
                weight_contributions.items(), row_col_positions, colors
            ):
                # Add mean line
                fig2.add_trace(
                    go.Scatter(
                        x=time_idx,
                        y=data['mean'],
                        mode='lines',
                        name=f'{label} (Mean)',
                        line=dict(width=2, color=color),
                        legendgroup=label
                    ),
                    row=row, col=col
                )

                # Add credible interval
                fig2.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_idx, time_idx[::-1]]),
                        y=np.concatenate([data['ci_high'], data['ci_low'][::-1]]),
                        fill='toself',
                        fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        name=f'{label} (95% CI)',
                        showlegend=False,
                        legendgroup=label
                    ),
                    row=row, col=col
                )

                # Add zero line
                fig2.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5,
                              row=row, col=col)

            fig2.update_layout(
                title_text="Weight Contributions from Fitness Components",
                height=800,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            for i in range(1, 5):
                fig2.update_xaxes(title_text="Day", row=(i+1)//2, col=2 if i%2==0 else 1)
                fig2.update_yaxes(title_text="Weight Contribution", row=(i+1)//2, col=2 if i%2==0 else 1)

            html_path2 = self.output_dir / "weight_contributions_interactive.html"
            fig2.write_html(str(html_path2))
            print(f"  Weight contributions interactive plot saved: {html_path2}")

    def run(self):
        """Run full visualization pipeline."""
        print("=" * 70)
        print("FITNESS TIME SERIES VISUALIZATION")
        print("=" * 70)

        # Extract fitness time series
        fitness_series = self.extract_fitness_time_series()

        if fitness_series is None:
            print("\nCould not extract fitness time series.")
            print("The model may need to be re-run with stored states.")
            return

        # Compute weight contributions
        weight_contributions = self.compute_weight_contributions(fitness_series)

        # Create visualizations
        self.create_time_series_plots(fitness_series, weight_contributions)
        self.create_interactive_report(fitness_series, weight_contributions)

        print("\n" + "=" * 70)
        print("VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {self.output_dir}")
        print("\nFiles created:")
        print(f"  • fitness_states_time_series.png - Fitness state time series")
        print(f"  • weight_contributions_time_series.png - Weight contribution time series")
        print(f"  • combined_contributions.png - Combined contributions")
        print(f"  • fitness_time_series_interactive.html - Interactive fitness plot")
        print(f"  • weight_contributions_interactive.html - Interactive contributions plot")


def main():
    """Main function."""
    visualizer = FitnessTimeSeriesVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()