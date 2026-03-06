#!/usr/bin/env python3
"""Regenerate plots from existing predictions with proper data-driven scaling.

This script reads existing component predictions from CSV files and
regenerates the visualizations with proper y-axis scaling (data-driven,
not including 0). No model fitting is required.

Usage:
  uv run python regenerate_plots_with_proper_scaling.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.weight import load_weight_data

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_component_predictions(predictions_dir="docs/component_predictions", use_lbs=True):
    """Load component predictions from CSV files."""
    print(f"Loading component predictions from {predictions_dir}/...")

    predictions_dir = Path(predictions_dir)

    # Determine suffix based on scale
    suffix = "_lbs" if use_lbs else ""

    # Load the combined file
    combined_path = predictions_dir / f"all_component_predictions{suffix}.csv"
    if not combined_path.exists():
        print(f"  ✗ Combined file not found: {combined_path}")
        # Try without suffix as fallback
        combined_path = predictions_dir / "all_component_predictions.csv"
        if not combined_path.exists():
            raise FileNotFoundError(f"Could not find component predictions in {predictions_dir}/")
        print(f"  ✓ Using fallback file: {combined_path}")
        suffix = ""  # Update suffix

    df_all = pd.read_csv(combined_path)
    df_all['date'] = pd.to_datetime(df_all['date'])

    print(f"  ✓ Loaded {len(df_all)} predictions")
    print(f"  Date range: {df_all['date'].min().date()} to {df_all['date'].max().date()}")
    print(f"  Hours: {sorted(df_all['hour'].unique())}")

    return df_all, suffix


def extract_components_from_dataframe(df_all, suffix=""):
    """Extract component data from the combined dataframe."""
    print("\nExtracting component data...")

    # Get unique dates and hours
    dates = np.sort(df_all['date'].unique())
    hours = np.sort(df_all['hour'].unique())

    D = len(dates)
    H = len(hours)

    # Create date and hour index mappings
    date_to_idx = {date: i for i, date in enumerate(dates)}
    hour_to_idx = {hour: i for i, hour in enumerate(hours)}

    # Initialize component arrays
    component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total', 'full_model']
    components = {}

    for component_name in component_names:
        mean = np.zeros((D, H))
        lower = np.zeros((D, H))
        upper = np.zeros((D, H))

        # Fill arrays
        for _, row in df_all.iterrows():
            d_idx = date_to_idx[row['date']]
            h_idx = hour_to_idx[row['hour']]

            mean[d_idx, h_idx] = row[f'{component_name}_mean']
            lower[d_idx, h_idx] = row[f'{component_name}_lower']
            upper[d_idx, h_idx] = row[f'{component_name}_upper']

        components[component_name] = (mean, lower, upper)
        print(f"  ✓ {component_name}: {D} days × {H} hours")

    return components, dates, hours


def create_component_visualizations(components, df_weight, dates, hours, output_dir, suffix=""):
    """Create visualizations with proper data-driven scaling."""
    print(f"\nCreating visualizations with proper scaling...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Convert to numpy arrays for convenience
    dates_np = np.array(dates)
    hours_np = np.array(hours)

    # Find noon index
    noon_idx = np.where(hours_np == 12.0)[0]
    if len(noon_idx) == 0:
        # Find closest to 12
        noon_idx = np.argmin(np.abs(hours_np - 12.0))
    else:
        noon_idx = noon_idx[0]

    # 1. Time series of each component at noon (12:00) with actual weight data
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total', 'full_model']
    titles = [
        'Intercept Component',
        'Strength Fitness Component',
        'Aerobic Fitness Component',
        'Daily Spline Component',
        'Total Prediction (All Components)',
        'Full Model Prediction (from Stan)'
    ]

    # Prepare actual weight data for noon (filter weights measured around noon)
    noon_start = 11.5  # 11:30
    noon_end = 12.5    # 12:30
    df_weight_noon = df_weight[(df_weight['hour_of_day'] >= noon_start) &
                               (df_weight['hour_of_day'] <= noon_end)].copy()
    df_weight_noon['timestamp'] = pd.to_datetime(df_weight_noon['date'].astype(str) + ' ' +
                                                 df_weight_noon['hour_of_day'].astype(str).str.replace('.', ':'))

    for idx, (component_name, title) in enumerate(zip(component_names, titles)):
        ax = axes[idx]
        mean, lower, upper = components[component_name]

        # Plot mean with credible interval
        ax.fill_between(dates_np, lower[:, noon_idx], upper[:, noon_idx],
                       alpha=0.3, color='skyblue', label='95% CI')
        ax.plot(dates_np, mean[:, noon_idx], 'b-', linewidth=1.5, label='Mean Prediction')

        # Add actual weight data (noon measurements)
        if len(df_weight_noon) > 0:
            ax.scatter(df_weight_noon['timestamp'], df_weight_noon['weight_lbs'],
                      alpha=0.6, s=30, color='red', edgecolor='black', linewidth=0.5,
                      label='Actual Weight (noon)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (lbs)' if 'lbs' in suffix else 'Standardized Weight')
        ax.set_title(f'{title} at 12:00')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data (not including 0)
        all_y_data = np.concatenate([
            mean[:, noon_idx],
            lower[:, noon_idx],
            upper[:, noon_idx]
        ])
        if len(df_weight_noon) > 0:
            all_y_data = np.concatenate([all_y_data, df_weight_noon['weight_lbs'].values])

        # Calculate data range with 5% padding
        y_min = np.min(all_y_data)
        y_max = np.max(all_y_data)
        y_range = y_max - y_min
        y_padding = y_range * 0.05  # 5% padding

        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Format x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_filename = f'component_time_series_noon{suffix}.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 2. Heatmap of total predictions over time
    fig, ax = plt.subplots(figsize=(14, 8))

    mean_total, _, _ = components['total']

    # Create heatmap - use ordinal dates for plotting
    # Convert dates to ordinal numbers (days since 0001-01-01)
    dates_ordinal = np.array([pd.Timestamp(d).toordinal() for d in dates_np])

    im = ax.imshow(mean_total.T, aspect='auto', cmap='viridis',
                   extent=[dates_ordinal[0], dates_ordinal[-1],
                           hours_np[0], hours_np[-1]],
                   origin='lower')

    # Add actual weight data points on heatmap
    for _, row in df_weight.iterrows():
        date_ordinal = pd.Timestamp(row['date']).toordinal()
        ax.scatter(date_ordinal, row['hour_of_day'],
                  s=20, color='red', edgecolor='white', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Total Predictions Heatmap with Actual Weight Measurements')

    # Format x-axis with dates
    date_ticks = np.linspace(dates_ordinal[0], dates_ordinal[-1], 8)
    date_labels = [pd.Timestamp.fromordinal(int(d)).strftime('%Y-%m-%d') for d in date_ticks]
    ax.set_xticks(date_ticks)
    ax.set_xticklabels(date_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (lbs)' if 'lbs' in suffix else 'Standardized Weight')

    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_filename = f'total_predictions_heatmap{suffix}.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 3. Component contributions at sample dates
    sample_indices = [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, -1]
    sample_dates = dates_np[sample_indices]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (sample_date, date_idx) in enumerate(zip(sample_dates, sample_indices)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get component values at this date
        components_at_date = {}
        for component_name in ['intercept', 'strength', 'aerobic', 'spline', 'total']:
            mean, _, _ = components[component_name]
            components_at_date[component_name] = mean[date_idx, :]

        # Plot line plot
        for component_name, color, label in [
            ('intercept', 'gray', 'Intercept'),
            ('strength', 'blue', 'Strength'),
            ('aerobic', 'green', 'Aerobic'),
            ('spline', 'orange', 'Spline')
        ]:
            values = components_at_date[component_name]
            ax.plot(hours_np, values, 'o-', linewidth=2, markersize=4, label=label, color=color)

        # Add actual weight measurements for this date
        df_date_weights = df_weight[df_weight['date'] == pd.Timestamp(sample_date).date()]
        if len(df_date_weights) > 0:
            ax.scatter(df_date_weights['hour_of_day'], df_date_weights['weight_lbs'],
                      s=40, color='red', edgecolor='black', linewidth=1,
                      label='Actual Weight', zorder=5)

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Weight (lbs)' if 'lbs' in suffix else 'Standardized Weight')
        ax.set_title(f'Components at {pd.Timestamp(sample_date).date()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data
        all_y_data = []
        for component_name in ['intercept', 'strength', 'aerobic', 'spline', 'total']:
            all_y_data.append(components_at_date[component_name])

        if len(df_date_weights) > 0:
            all_y_data.append(df_date_weights['weight_lbs'].values)

        all_y_data_flat = np.concatenate([d.flatten() for d in all_y_data])

        y_min = np.min(all_y_data_flat)
        y_max = np.max(all_y_data_flat)
        y_range = y_max - y_min
        y_padding = y_range * 0.05

        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Remove empty subplots
    for idx in range(len(sample_dates), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_filename = f'component_contributions_sample_dates{suffix}.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 4. Daily patterns analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean daily pattern
    ax = axes[0]
    mean_total, _, _ = components['total']
    mean_daily = mean_total.mean(axis=0)

    ax.plot(hours_np, mean_daily, 'o-', linewidth=3, markersize=6,
            color='darkblue', label='Mean Daily Pattern')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)' if 'lbs' in suffix else 'Standardized Weight')
    ax.set_title('Mean Daily Weight Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    y_min = np.min(mean_daily)
    y_max = np.max(mean_daily)
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Right: Daily pattern variability
    ax = axes[1]
    n_days_to_plot = min(50, len(dates))
    day_indices = np.linspace(0, len(dates)-1, n_days_to_plot, dtype=int)

    for day_idx in day_indices:
        ax.plot(hours_np, mean_total[day_idx, :], 'gray', alpha=0.1, linewidth=0.5)

    ax.plot(hours_np, mean_daily, 'b-', linewidth=3, label='Mean Pattern')

    # Add all actual weight data
    ax.scatter(df_weight['hour_of_day'], df_weight['weight_lbs'],
              alpha=0.3, s=20, color='red', label='Actual Weight')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)' if 'lbs' in suffix else 'Standardized Weight')
    ax.set_title(f'Daily Pattern Variability ({n_days_to_plot} sample days)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits
    all_y_data = []
    for day_idx in day_indices:
        all_y_data.append(mean_total[day_idx, :])
    all_y_data.append(mean_daily)
    all_y_data.append(df_weight['weight_lbs'].values)

    all_y_data_flat = np.concatenate([d.flatten() for d in all_y_data])

    y_min = np.min(all_y_data_flat)
    y_max = np.max(all_y_data_flat)
    y_range = y_max - y_min
    y_padding = y_range * 0.05
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    output_filename = f'daily_patterns_analysis{suffix}.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")

    # 5. Prediction vs actual scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get predictions at actual measurement times
    predicted_weights = []
    actual_weights = []

    for _, row in df_weight.iterrows():
        # Find closest date and hour
        date_diffs = np.abs(dates_np - pd.Timestamp(row['date']).to_datetime64())
        date_idx = np.argmin(date_diffs)

        hour_diffs = np.abs(hours_np - row['hour_of_day'])
        hour_idx = np.argmin(hour_diffs)

        predicted_weight = mean_total[date_idx, hour_idx]
        actual_weight = row['weight_lbs']

        predicted_weights.append(predicted_weight)
        actual_weights.append(actual_weight)

    ax.scatter(actual_weights, predicted_weights, alpha=0.6, s=20)

    # Add diagonal line
    min_val = min(min(actual_weights), min(predicted_weights))
    max_val = max(max(actual_weights), max(predicted_weights))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

    ax.set_xlabel('Actual Weight (lbs)' if 'lbs' in suffix else 'Actual Weight (std)')
    ax.set_ylabel('Predicted Weight (lbs)' if 'lbs' in suffix else 'Predicted Weight (std)')
    ax.set_title('Prediction vs Actual Weight')

    # Add correlation coefficient
    correlation = np.corrcoef(actual_weights, predicted_weights)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = f'prediction_vs_actual_scatter{suffix}.png'
    plt.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{output_filename}")


def main():
    """Main function to regenerate plots with proper scaling."""
    print("="*60)
    print("Regenerating Plots with Proper Data-Driven Scaling")
    print("="*60)
    print("\nThis script reads existing predictions and regenerates plots")
    print("with proper y-axis scaling (data-driven, not including 0).")
    print("No model fitting is required.\n")

    # Ask user which scale to use
    use_lbs = True  # Default to lbs scale since that's what we're fixing

    # Load weight data
    print("Loading weight data...")
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Prepare weight data
    df_weight['date'] = df_weight['timestamp'].dt.date
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    print(f"  ✓ Loaded {len(df_weight)} weight measurements")
    print(f"  Weight range: {df_weight['weight_lbs'].min():.1f} to {df_weight['weight_lbs'].max():.1f} lbs")

    # Load component predictions
    try:
        df_all, suffix = load_component_predictions(use_lbs=use_lbs)
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run one of the component prediction scripts first:")
        print("  uv run python generate_component_predictions.py")
        print("  uv run python generate_component_predictions_with_raw_weights.py")
        return

    # Extract components from dataframe
    components, dates, hours = extract_components_from_dataframe(df_all, suffix)

    # Create output directory
    output_dir = "docs/component_predictions"

    # Create visualizations with proper scaling
    create_component_visualizations(components, df_weight, dates, hours, output_dir, suffix)

    print("\n" + "="*60)
    print("Plot Regeneration Complete")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Source: Existing predictions from docs/component_predictions/")
    print(f"  - Scale: {'lbs' if 'lbs' in suffix else 'standardized'}")
    print(f"  - Output directory: {output_dir}/")
    print(f"  - Files: 5 plots with proper data-driven scaling")
    print(f"\nKey improvements:")
    print(f"  1. Y-axis scaled to data range (not including 0)")
    print(f"  2. 5% padding around data for visual clarity")
    print(f"  3. No model refitting required")
    print(f"\nTo regenerate plots with different scaling, modify the script")
    print(f"or run the appropriate component prediction script.")


if __name__ == "__main__":
    main()