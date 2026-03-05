#!/usr/bin/env python3
"""Improved visualization showing fitness indexes mapping to weight perturbations.

This script creates a comprehensive visualization showing:
1. Actual weight time series with credible intervals
2. Four fitness indexes over time with credible intervals
3. Weight contributions from each fitness component
4. Clear mapping between fitness states and weight changes
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import json
from datetime import datetime, timedelta
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ImprovedFitnessWeightVisualizer:
    """Create improved visualizations showing fitness-weight mapping."""

    def __init__(self, output_dir: str = "output/improved_fitness_weight"):
        """Initialize visualizer.

        Args:
            output_dir: Path to directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model results
        self.idata = self._load_inference_data()
        self.standardization = self._load_standardization()
        self.weight_data = self._load_weight_data()
        self.workout_data = self._load_workout_data()
        self.impulse_data = self._extract_impulse_data()

        if self.idata is None:
            raise ValueError("Could not load inference data. Model may need to be re-run.")

    def _load_inference_data(self):
        """Load ArviZ InferenceData from four-fitness model."""
        # Try to load from the full output directory
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

        print("No inference data found.")
        return None

    def _load_standardization(self):
        """Load standardization information."""
        std_path = Path("output/four_fitness_full/standardization.json")
        if std_path.exists():
            with open(std_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_weight_data(self):
        """Load actual weight data."""
        try:
            weight_df = load_weight_data()
            return weight_df
        except Exception as e:
            print(f"Warning: Could not load weight data: {e}")
            return None

    def _load_workout_data(self):
        """Load workout event data."""
        try:
            workout_path = Path("output/workout_report/workouts_daily_count.csv")
            if workout_path.exists():
                workout_df = pd.read_csv(workout_path)
                workout_df['date'] = pd.to_datetime(workout_df['date'])
                return workout_df
            else:
                print(f"Warning: Workout data not found at {workout_path}")
                return None
        except Exception as e:
            print(f"Warning: Could not load workout data: {e}")
            return None

    def _extract_impulse_data(self):
        """Extract impulse data from model."""
        print("\nExtracting impulse data...")

        impulse_vars = [
            'impulse_a_short', 'impulse_s_short',
            'impulse_a_long', 'impulse_s_long'
        ]

        impulse_data = {}
        for var in impulse_vars:
            if var in self.idata.posterior:
                samples = self.idata.posterior[var].values
                n_chains, n_draws, n_time = samples.shape
                samples_flat = samples.reshape(-1, n_time)

                # Compute mean impulse
                mean_impulse = samples_flat.mean(axis=0)

                impulse_data[var] = {
                    'mean': mean_impulse,
                    'samples': samples_flat,
                    'n_time': n_time
                }

                # Find days with non-zero impulses (workout events)
                non_zero_days = np.where(mean_impulse > 0.1)[0]  # Threshold for significant impulses
                print(f"  {var}: {len(non_zero_days)} days with significant impulses")

        return impulse_data

    def _get_workout_event_days(self, dates):
        """Get days with workout events."""
        workout_days = []
        if self.workout_data is not None and dates is not None:
            workout_dates = pd.to_datetime(self.workout_data['date'])
            # Convert dates list to pandas DatetimeIndex for comparison
            dates_pd = pd.DatetimeIndex(dates)
            for workout_date in workout_dates:
                # Find closest date in dates list
                idx = np.argmin(np.abs(dates_pd - workout_date))
                workout_days.append(idx)
        return workout_days

    def _get_impulse_event_days(self):
        """Get days with significant impulse events from model."""
        impulse_events = {}
        if self.impulse_data:
            for impulse_key, data in self.impulse_data.items():
                mean_impulse = data['mean']
                # Find days with significant impulses (above threshold)
                significant_days = np.where(mean_impulse > 0.1)[0]  # Adjust threshold as needed
                impulse_events[impulse_key] = significant_days
        return impulse_events

    def _add_impulse_markers(self, ax, time_idx, impulse_events):
        """Add impulse event markers to a plot as vertical lines scaled by intensity."""
        if impulse_events and self.impulse_data:
            # Define colors for each activity type
            impulse_colors = {
                'impulse_a_short': '#1f77b4',  # Blue for aerobic short
                'impulse_s_short': '#ff7f0e',  # Orange for strength short
                'impulse_a_long': '#2ca02c',   # Green for aerobic long
                'impulse_s_long': '#d62728'    # Red for strength long
            }

            impulse_labels = {
                'impulse_a_short': 'Aerobic Short',
                'impulse_s_short': 'Strength Short',
                'impulse_a_long': 'Aerobic Long',
                'impulse_s_long': 'Strength Long'
            }

            # Track which labels we've added to legend
            added_labels = set()

            # Get plot y-range for scaling
            ylim = ax.get_ylim()
            y_range = ylim[1] - ylim[0]

            for impulse_key, days in impulse_events.items():
                if impulse_key in self.impulse_data and days.size > 0:
                    color = impulse_colors.get(impulse_key, 'gray')
                    label = impulse_labels.get(impulse_key, 'Impulse')

                    # Get impulse intensities for these days
                    mean_impulse = self.impulse_data[impulse_key]['mean']

                    for day in days:
                        intensity = mean_impulse[day]
                        if intensity > 0.1:  # Only plot significant impulses
                            # Scale line height appropriately - use a fixed percentage of plot height
                            # Impulse intensities are typically 0-50, fitness values are -1 to 1
                            # So scale impulses to be visible but not overwhelming
                            line_height = min(0.15 * y_range, intensity * 0.02 * y_range)

                            # Draw vertical line from bottom to scaled height
                            ax.plot([day, day], [ylim[0], ylim[0] + line_height],
                                   color=color, alpha=0.6, linewidth=1.5,
                                   solid_capstyle='round', zorder=1)

                            # Add label to legend if not already added
                            if impulse_key not in added_labels:
                                ax.plot([], [], color=color, linewidth=2,
                                       label=f'{label} Impulse', alpha=0.7)
                                added_labels.add(impulse_key)

    def extract_fitness_time_series(self):
        """Extract fitness state time series from model."""
        print("\nExtracting fitness time series...")

        # Look for fitness state variables
        fitness_vars = [
            'fitness_a_short_stored', 'fitness_s_short_stored',
            'fitness_a_long_stored', 'fitness_s_long_stored'
        ]

        available_vars = [v for v in fitness_vars if v in self.idata.posterior]
        print(f"Found {len(available_vars)} fitness variables: {available_vars}")

        if not available_vars:
            print("No fitness state variables found in posterior.")
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

    def extract_weight_predictions(self):
        """Extract weight predictions from model."""
        print("\nExtracting weight predictions...")

        # The model should have weight predictions
        # Look for variables that might contain weight predictions
        weight_vars = ['weight_pred', 'y_pred', 'f_gp_stored', 'f_daily_stored']

        for var in weight_vars:
            if var in self.idata.posterior:
                print(f"Found weight variable: {var}")
                samples = self.idata.posterior[var].values

                # Reshape to (sample, time)
                if len(samples.shape) == 3:  # (chain, draw, time)
                    n_chains, n_draws, n_time = samples.shape
                    samples_flat = samples.reshape(-1, n_time)
                elif len(samples.shape) == 2:  # (chain*draw, time)
                    samples_flat = samples
                    n_time = samples.shape[1]
                else:
                    continue

                # Compute statistics
                mean = samples_flat.mean(axis=0)
                std = samples_flat.std(axis=0)
                ci_low = np.percentile(samples_flat, 2.5, axis=0)
                ci_high = np.percentile(samples_flat, 97.5, axis=0)

                return {
                    'mean': mean,
                    'std': std,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'samples': samples_flat,
                    'n_time': n_time
                }

        print("No weight prediction variables found.")
        return None

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
                color = '#1f77b4'  # blue
            elif 's_short' in fitness_var:
                gamma_key = 'gamma_s_short'
                label = 'Strength Short-term'
                color = '#ff7f0e'  # orange
            elif 'a_long' in fitness_var:
                gamma_key = 'gamma_a_long'
                label = 'Aerobic Long-term'
                color = '#2ca02c'  # green
            elif 's_long' in fitness_var:
                gamma_key = 'gamma_s_long'
                label = 'Strength Long-term'
                color = '#d62728'  # red
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
                'gamma_mean': gamma_params[gamma_key]['mean'],
                'color': color
            }

            print(f"  {label}: contribution range [{mean.min():.3f}, {mean.max():.3f}]")

        return weight_contributions

    def create_comprehensive_plot(self, fitness_series, weight_contributions, weight_predictions=None):
        """Create comprehensive plot showing fitness-weight mapping with impulses."""
        print("\nCreating comprehensive fitness-weight mapping plot with impulses...")

        if not fitness_series:
            print("No fitness series data to plot")
            return

        # Create time index (days)
        n_time = len(next(iter(fitness_series.values()))['mean'])
        time_idx = np.arange(n_time)

        # Create date range if we have standardization info
        if 'date_range_start' in self.standardization:
            start_date = datetime.strptime(self.standardization['date_range_start'], '%Y-%m-%d')
            dates = [start_date + timedelta(days=i) for i in range(n_time)]
            date_labels = [d.strftime('%Y-%m-%d') for d in dates]
            # Use every 30th date for x-axis labels to avoid clutter
            date_ticks = dates[::30]
            date_tick_labels = [d.strftime('%Y-%m-%d') for d in date_ticks]
        else:
            dates = None
            date_labels = None
            date_ticks = None
            date_tick_labels = None

        # Get workout event days
        workout_days = self._get_workout_event_days(dates)

        # Get impulse event days (significant impulses from model)
        impulse_events = self._get_impulse_event_days()

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 14))

        # 1. Top: Actual weight data (if available)
        if self.weight_data is not None and dates is not None:
            ax1 = plt.subplot(5, 1, 1)

            # Plot actual weight measurements
            weight_dates = pd.to_datetime(self.weight_data['date'])
            weight_values = self.weight_data['weight_lbs']

            ax1.scatter(weight_dates, weight_values, alpha=0.6, s=20,
                       color='black', label='Actual Weight')

            # Add smoothed trend line
            from scipy import signal
            if len(weight_values) > 10:
                # Use Savitzky-Golay filter for smoothing
                window_length = min(31, len(weight_values) // 2 * 2 + 1)  # odd number
                if window_length > 2:
                    smoothed = signal.savgol_filter(weight_values, window_length, 3)
                    ax1.plot(weight_dates, smoothed, 'r-', linewidth=2,
                            alpha=0.7, label='Smoothed Trend')

            # Add workout event markers
            if workout_days:
                workout_dates = [dates[i] for i in workout_days]
                # Add vertical lines for workout days
                for workout_date in workout_dates:
                    ax1.axvline(x=workout_date, color='blue', alpha=0.3, linestyle=':', linewidth=0.5)

            ax1.set_title('Actual Weight Measurements with Workout Events', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Weight (lbs)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Fitness states (4 components) with impulse markers
        ax2 = plt.subplot(5, 1, 2)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        labels = ['Aerobic Short-term', 'Strength Short-term',
                 'Aerobic Long-term', 'Strength Long-term']
        fitness_keys = ['fitness_a_short_stored', 'fitness_s_short_stored',
                       'fitness_a_long_stored', 'fitness_s_long_stored']

        for i, (key, label, color) in enumerate(zip(fitness_keys, labels, colors)):
            if key in fitness_series:
                data = fitness_series[key]
                ax2.plot(time_idx, data['mean'], label=label,
                        color=color, linewidth=2, alpha=0.8)
                # Use thinner CI for better visibility
                ax2.fill_between(time_idx, data['ci_low'], data['ci_high'],
                               alpha=0.15, color=color)

        # Add impulse markers
        self._add_impulse_markers(ax2, time_idx, impulse_events)

        ax2.set_title('Fitness States Over Time with Impulse Events (95% Credible Intervals)',
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Fitness (standardized)', fontsize=12)
        ax2.legend(loc='upper left', ncol=2, fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. Weight contributions from fitness components with impulse markers
        ax3 = plt.subplot(5, 1, 3)

        if weight_contributions:
            for label, data in weight_contributions.items():
                color = data['color']
                ax3.plot(time_idx, data['mean'], label=label,
                        color=color, linewidth=2, alpha=0.8)
                # Use thinner CI for better visibility
                ax3.fill_between(time_idx, data['ci_low'], data['ci_high'],
                               alpha=0.15, color=color)

            # Add zero line
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

        # Add impulse markers
        self._add_impulse_markers(ax3, time_idx, impulse_events)

        ax3.set_title('Weight Contributions from Fitness Components with Impulse Events (γ × Fitness)',
                     fontsize=14, fontweight='bold')
        ax3.set_ylabel('Weight Contribution\n(standardized)', fontsize=12)
        ax3.legend(loc='upper left', ncol=2, fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Combined effect and comparison
        ax4 = plt.subplot(5, 1, 4)

        # Plot combined weight contributions
        if weight_contributions:
            combined_mean = np.zeros(n_time)
            combined_ci_low = np.zeros(n_time)
            combined_ci_high = np.zeros(n_time)

            # Stack all contribution samples
            all_contrib_samples = []
            for data in weight_contributions.values():
                all_contrib_samples.append(data['samples'])

            if all_contrib_samples:
                # Sum contributions across components
                # We need to align samples properly
                min_samples = min([s.shape[0] for s in all_contrib_samples])
                aligned_samples = [s[:min_samples, :] for s in all_contrib_samples]
                combined_samples = np.sum(aligned_samples, axis=0)

                combined_mean = combined_samples.mean(axis=0)
                combined_ci_low = np.percentile(combined_samples, 2.5, axis=0)
                combined_ci_high = np.percentile(combined_samples, 97.5, axis=0)

                ax4.plot(time_idx, combined_mean, label='Total Fitness Effect',
                        color='purple', linewidth=3, alpha=0.9)
                # Use thinner CI for better visibility
                ax4.fill_between(time_idx, combined_ci_low, combined_ci_high,
                               alpha=0.2, color='purple', label='95% CI')

        # Add impulse markers
        self._add_impulse_markers(ax4, time_idx, impulse_events)

        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax4.set_title('Combined Fitness Effect on Weight with Impulse Events',
                     fontsize=14, fontweight='bold')
        ax4.set_ylabel('Standardized Effect', fontsize=12)
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. Impulse strength over time
        ax5 = plt.subplot(5, 1, 5)

        if self.impulse_data:
            impulse_colors = {'impulse_a_short': '#1f77b4', 'impulse_s_short': '#ff7f0e',
                            'impulse_a_long': '#2ca02c', 'impulse_s_long': '#d62728'}
            impulse_labels = {'impulse_a_short': 'Aerobic Short', 'impulse_s_short': 'Strength Short',
                            'impulse_a_long': 'Aerobic Long', 'impulse_s_long': 'Strength Long'}

            # Plot impulse strengths as vertical bars
            max_impulse = 0
            for impulse_key, color in impulse_colors.items():
                if impulse_key in self.impulse_data:
                    mean_impulse = self.impulse_data[impulse_key]['mean']
                    # Only plot significant impulses
                    significant_idx = np.where(mean_impulse > 0.1)[0]
                    if len(significant_idx) > 0:
                        # Plot vertical bars for each impulse
                        for idx in significant_idx:
                            intensity = mean_impulse[idx]
                            ax5.bar(idx, intensity, color=color, alpha=0.7, width=1.0,
                                   label=impulse_labels[impulse_key] if idx == significant_idx[0] else "")
                            max_impulse = max(max_impulse, intensity)

            # Set y-axis limit with some padding
            if max_impulse > 0:
                ax5.set_ylim(0, max_impulse * 1.2)

            ax5.set_title('Impulse Strength from Workout Events (Vertical Bars)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Day', fontsize=12)
            ax5.set_ylabel('Impulse Strength', fontsize=12)
            # Remove duplicate labels from legend
            handles, labels = ax5.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax5.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)
            ax5.grid(True, alpha=0.3, axis='y')

        # Set x-axis labels for bottom plot
        if dates is not None and date_ticks is not None:
            tick_positions = [i for i, d in enumerate(dates) if d in date_ticks]
            ax5.set_xticks(tick_positions)
            ax5.set_xticklabels(date_tick_labels, rotation=45, ha='right')

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / "fitness_weight_mapping_comprehensive_with_impulses.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  Comprehensive plot with impulses saved: {output_path}")

        # Create zoomed versions
        self.create_zoomed_plots(fitness_series, weight_contributions, dates, workout_days, impulse_events)

        # Create additional focused plots
        self.create_focused_plots(fitness_series, weight_contributions, dates, workout_days, impulse_events)

    def create_zoomed_plots(self, fitness_series, weight_contributions, dates, workout_days, impulse_events):
        """Create zoomed-in versions of plots to show detail."""
        print("\nCreating zoomed plots...")

        if not fitness_series:
            return

        n_time = len(next(iter(fitness_series.values()))['mean'])
        time_idx = np.arange(n_time)

        # Create several zoomed regions (e.g., first 100 days, middle 100 days, last 100 days)
        zoom_regions = [
            (0, 100, "first_100_days"),
            (n_time//2 - 50, n_time//2 + 50, "middle_100_days"),
            (n_time - 100, n_time, "last_100_days"),
            (0, 50, "first_50_days"),  # Extra zoomed
            (n_time - 50, n_time, "last_50_days")  # Extra zoomed
        ]

        for start_idx, end_idx, region_name in zoom_regions:
            if start_idx < 0 or end_idx > n_time:
                continue

            # Create zoomed figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))

            # Zoomed time indices
            zoom_time_idx = time_idx[start_idx:end_idx]

            # 1. Fitness states (zoomed)
            ax1 = axes[0]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            labels = ['Aerobic Short-term', 'Strength Short-term',
                     'Aerobic Long-term', 'Strength Long-term']
            fitness_keys = ['fitness_a_short_stored', 'fitness_s_short_stored',
                           'fitness_a_long_stored', 'fitness_s_long_stored']

            for i, (key, label, color) in enumerate(zip(fitness_keys, labels, colors)):
                if key in fitness_series:
                    data = fitness_series[key]
                    ax1.plot(zoom_time_idx, data['mean'][start_idx:end_idx],
                            label=label, color=color, linewidth=2, alpha=0.8)
                    # Very thin CI for zoomed view
                    ax1.fill_between(zoom_time_idx,
                                    data['ci_low'][start_idx:end_idx],
                                    data['ci_high'][start_idx:end_idx],
                                    alpha=0.1, color=color)

            # Add impulse markers
            self._add_impulse_markers_zoomed(ax1, zoom_time_idx, start_idx, end_idx, impulse_events)

            ax1.set_title(f'Fitness States ({region_name.replace("_", " ").title()})', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Fitness', fontsize=10)
            ax1.legend(loc='upper left', ncol=2, fontsize=8)
            ax1.grid(True, alpha=0.3)

            # 2. Weight contributions (zoomed)
            ax2 = axes[1]
            if weight_contributions:
                for label, data in weight_contributions.items():
                    color = data['color']
                    ax2.plot(zoom_time_idx, data['mean'][start_idx:end_idx],
                            label=label, color=color, linewidth=2, alpha=0.8)
                    # Very thin CI for zoomed view
                    ax2.fill_between(zoom_time_idx,
                                    data['ci_low'][start_idx:end_idx],
                                    data['ci_high'][start_idx:end_idx],
                                    alpha=0.1, color=color)

                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

            # Add impulse markers
            self._add_impulse_markers_zoomed(ax2, zoom_time_idx, start_idx, end_idx, impulse_events)

            ax2.set_title(f'Weight Contributions ({region_name.replace("_", " ").title()})', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Contribution', fontsize=10)
            ax2.legend(loc='upper left', ncol=2, fontsize=8)
            ax2.grid(True, alpha=0.3)

            # 3. Combined effect (zoomed)
            ax3 = axes[2]
            if weight_contributions:
                # Compute combined effect for zoomed region
                all_contrib_samples = []
                for data in weight_contributions.values():
                    all_contrib_samples.append(data['samples'])

                if all_contrib_samples:
                    min_samples = min([s.shape[0] for s in all_contrib_samples])
                    aligned_samples = [s[:min_samples, start_idx:end_idx] for s in all_contrib_samples]
                    combined_samples = np.sum(aligned_samples, axis=0)

                    combined_mean = combined_samples.mean(axis=0)
                    combined_ci_low = np.percentile(combined_samples, 2.5, axis=0)
                    combined_ci_high = np.percentile(combined_samples, 97.5, axis=0)

                    ax3.plot(zoom_time_idx, combined_mean, label='Total Effect',
                            color='purple', linewidth=2, alpha=0.9)
                    # Very thin CI for zoomed view
                    ax3.fill_between(zoom_time_idx, combined_ci_low, combined_ci_high,
                                   alpha=0.15, color='purple', label='95% CI')

            # Add impulse markers
            self._add_impulse_markers_zoomed(ax3, zoom_time_idx, start_idx, end_idx, impulse_events)

            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax3.set_title(f'Combined Fitness Effect ({region_name.replace("_", " ").title()})', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Day', fontsize=10)
            ax3.set_ylabel('Effect', fontsize=10)
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save zoomed figure
            output_path = self.output_dir / f"zoomed_{region_name}.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"  Zoomed plot saved: {output_path}")

    def _add_impulse_markers_zoomed(self, ax, zoom_time_idx, start_idx, end_idx, impulse_events):
        """Add impulse event markers to zoomed plot as vertical lines scaled by intensity."""
        if impulse_events and self.impulse_data:
            # Define colors for each activity type
            impulse_colors = {
                'impulse_a_short': '#1f77b4',  # Blue for aerobic short
                'impulse_s_short': '#ff7f0e',  # Orange for strength short
                'impulse_a_long': '#2ca02c',   # Green for aerobic long
                'impulse_s_long': '#d62728'    # Red for strength long
            }

            impulse_labels = {
                'impulse_a_short': 'Aerobic Short',
                'impulse_s_short': 'Strength Short',
                'impulse_a_long': 'Aerobic Long',
                'impulse_s_long': 'Strength Long'
            }

            # Track which labels we've added to legend
            added_labels = set()

            # Get plot y-range for scaling
            ylim = ax.get_ylim()
            y_range = ylim[1] - ylim[0]

            for impulse_key, days in impulse_events.items():
                if impulse_key in self.impulse_data:
                    color = impulse_colors.get(impulse_key, 'gray')
                    label = impulse_labels.get(impulse_key, 'Impulse')

                    # Get impulse intensities
                    mean_impulse = self.impulse_data[impulse_key]['mean']

                    for day in days:
                        if start_idx <= day < end_idx:
                            intensity = mean_impulse[day]
                            if intensity > 0.1:  # Only plot significant impulses
                                rel_day = day - start_idx

                                # Scale line height appropriately for zoomed view
                                # Use smaller scaling for zoomed plots
                                line_height = min(0.1 * y_range, intensity * 0.015 * y_range)

                                # Draw vertical line from bottom to scaled height
                                ax.plot([rel_day, rel_day], [ylim[0], ylim[0] + line_height],
                                       color=color, alpha=0.7, linewidth=1.2,
                                       solid_capstyle='round', zorder=1)

                                # Add label to legend if not already added
                                if impulse_key not in added_labels:
                                    ax.plot([], [], color=color, linewidth=2,
                                           label=f'{label} Impulse', alpha=0.7)
                                    added_labels.add(impulse_key)

    def create_focused_plots(self, fitness_series, weight_contributions, dates=None, workout_days=None, impulse_events=None):
        """Create additional focused plots for detailed analysis with impulse markers."""
        print("\nCreating focused analysis plots with impulses...")

        n_time = len(next(iter(fitness_series.values()))['mean'])
        time_idx = np.arange(n_time)

        # 1. Plot showing relationship between fitness states and contributions
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
        axes1 = axes1.flatten()

        fitness_keys = ['fitness_a_short_stored', 'fitness_s_short_stored',
                       'fitness_a_long_stored', 'fitness_s_long_stored']
        contribution_labels = ['Aerobic Short-term', 'Strength Short-term',
                              'Aerobic Long-term', 'Strength Long-term']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i, (fitness_key, contrib_label, color, ax) in enumerate(
            zip(fitness_keys, contribution_labels, colors, axes1)):

            if fitness_key in fitness_series and contrib_label in weight_contributions:
                fitness_data = fitness_series[fitness_key]
                contrib_data = weight_contributions[contrib_label]

                # Plot fitness state
                ax.plot(time_idx, fitness_data['mean'], color=color,
                       linewidth=2, alpha=0.7, label='Fitness State')
                ax.fill_between(time_idx, fitness_data['ci_low'], fitness_data['ci_high'],
                              alpha=0.2, color=color)

                # Plot weight contribution on secondary y-axis
                ax2 = ax.twinx()
                ax2.plot(time_idx, contrib_data['mean'], color='black',
                        linewidth=2, alpha=0.7, linestyle='--', label='Weight Contribution')
                ax2.fill_between(time_idx, contrib_data['ci_low'], contrib_data['ci_high'],
                               alpha=0.1, color='gray')

                # Add impulse markers
                if impulse_events:
                    self._add_impulse_markers(ax, time_idx, impulse_events)

                # Formatting
                ax.set_title(f'{contrib_label}\nFitness State vs. Weight Contribution',
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Day', fontsize=10)
                ax.set_ylabel('Fitness (standardized)', fontsize=10, color=color)
                ax2.set_ylabel('Weight Contribution', fontsize=10, color='black')

                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                         loc='upper left', fontsize=8)

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig1.savefig(self.output_dir / "fitness_vs_contributions.png",
                    dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  Fitness vs contributions plot saved: {self.output_dir / 'fitness_vs_contributions.png'}")

        # 2. Plot showing cumulative contributions
        fig2, ax2 = plt.subplots(figsize=(15, 6))

        if weight_contributions:
            cumulative_contributions = {}

            for label, data in weight_contributions.items():
                # Compute cumulative sum of contributions
                cumsum_samples = np.cumsum(data['samples'], axis=1)
                cumsum_mean = cumsum_samples.mean(axis=0)
                cumsum_ci_low = np.percentile(cumsum_samples, 2.5, axis=0)
                cumsum_ci_high = np.percentile(cumsum_samples, 97.5, axis=0)

                cumulative_contributions[label] = {
                    'mean': cumsum_mean,
                    'ci_low': cumsum_ci_low,
                    'ci_high': cumsum_ci_high,
                    'color': data['color']
                }

                ax2.plot(time_idx, cumsum_mean, label=f'Cumulative {label}',
                        color=data['color'], linewidth=2, alpha=0.8)
                ax2.fill_between(time_idx, cumsum_ci_low, cumsum_ci_high,
                               alpha=0.2, color=data['color'])

            # Add impulse markers
            if impulse_events:
                self._add_impulse_markers(ax2, time_idx, impulse_events)

            ax2.set_title('Cumulative Weight Contributions from Fitness Components',
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Day', fontsize=12)
            ax2.set_ylabel('Cumulative Weight Effect', fontsize=12)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fig2.savefig(self.output_dir / "cumulative_contributions.png",
                        dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Cumulative contributions plot saved: {self.output_dir / 'cumulative_contributions.png'}")

    def create_summary_statistics(self, fitness_series, weight_contributions):
        """Create summary statistics and insights."""
        print("\nCreating summary statistics...")

        summary = {
            'fitness_components': {},
            'weight_contributions': {},
            'insights': []
        }

        # Analyze fitness components
        for var_name, data in fitness_series.items():
            mean_vals = data['mean']
            summary['fitness_components'][var_name] = {
                'mean_overall': float(mean_vals.mean()),
                'std_overall': float(mean_vals.std()),
                'min_value': float(mean_vals.min()),
                'max_value': float(mean_vals.max()),
                'trend': 'increasing' if mean_vals[-1] > mean_vals[0] else 'decreasing'
            }

        # Analyze weight contributions
        if weight_contributions:
            for label, data in weight_contributions.items():
                mean_contrib = data['mean']
                summary['weight_contributions'][label] = {
                    'mean_effect': float(mean_contrib.mean()),
                    'std_effect': float(mean_contrib.std()),
                    'min_effect': float(mean_contrib.min()),
                    'max_effect': float(mean_contrib.max()),
                    'gamma_coefficient': float(data['gamma_mean']),
                    'predominant_sign': 'positive' if mean_contrib.mean() > 0 else 'negative'
                }

        # Generate insights
        if weight_contributions:
            # Find which component has largest average effect
            avg_effects = {label: data['mean'].mean()
                          for label, data in weight_contributions.items()}
            max_effect_label = max(avg_effects.items(), key=lambda x: abs(x[1]))[0]
            max_effect_value = avg_effects[max_effect_label]

            summary['insights'].append(
                f"Largest weight effect: {max_effect_label} "
                f"(average effect: {max_effect_value:.3f})"
            )

            # Check if effects are consistent
            consistent_effects = all(abs(effect) > 0.01 for effect in avg_effects.values())
            summary['insights'].append(
                f"Effects are {'consistent' if consistent_effects else 'inconsistent'} "
                f"across all components"
            )

            # Check temporal patterns
            for label, data in weight_contributions.items():
                trend = 'increasing' if data['mean'][-1] > data['mean'][0] else 'decreasing'
                summary['insights'].append(
                    f"{label} contribution shows {trend} trend over time"
                )

        # Save summary to JSON
        summary_path = self.output_dir / "summary_statistics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Summary statistics saved: {summary_path}")

        # Print key insights
        print("\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)
        for insight in summary['insights']:
            print(f"• {insight}")

        if weight_contributions:
            print("\nWeight Contribution Summary:")
            for label, stats in summary['weight_contributions'].items():
                sign = "+" if stats['mean_effect'] > 0 else ""
                print(f"  {label}: {sign}{stats['mean_effect']:.3f} "
                      f"(γ = {stats['gamma_coefficient']:.3f})")

    def run(self):
        """Run full visualization pipeline."""
        print("="*70)
        print("IMPROVED FITNESS-WEIGHT MAPPING VISUALIZATION")
        print("="*70)

        # Extract fitness time series
        fitness_series = self.extract_fitness_time_series()

        if fitness_series is None:
            print("\nCould not extract fitness time series.")
            return

        # Extract weight predictions
        weight_predictions = self.extract_weight_predictions()

        # Compute weight contributions
        weight_contributions = self.compute_weight_contributions(fitness_series)

        # Create visualizations
        self.create_comprehensive_plot(fitness_series, weight_contributions, weight_predictions)

        # Create summary statistics
        self.create_summary_statistics(fitness_series, weight_contributions)

        print("\n" + "="*70)
        print("VISUALIZATION COMPLETE")
        print("="*70)
        print(f"Results saved to: {self.output_dir}")
        print("\nFiles created:")
        print(f"  • fitness_weight_mapping_comprehensive.png - Main comprehensive plot")
        print(f"  • fitness_vs_contributions.png - Fitness vs weight contributions")
        print(f"  • cumulative_contributions.png - Cumulative weight effects")
        print(f"  • summary_statistics.json - Summary statistics and insights")


def main():
    """Main function."""
    visualizer = ImprovedFitnessWeightVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()