#!/usr/bin/env python3
"""Generate predictions at 4-hour intervals for each model component with raw lbs scale and actual weight data."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_data_with_4h_intervals():
    """Prepare data with 4-hour prediction intervals (0, 4, 8, 12, 16, 20, 24 hours)."""
    print("Preparing data with 4-hour prediction intervals...")

    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],
        max_hr=185.0,
    )

    # Create full date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_intensity['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    df_act = df_intensity[['date', 'strength_training', 'walking', 'cycling']].copy()
    df_daily = df_daily.merge(df_act, on='date', how='left')
    df_daily = df_daily.fillna(0)

    # Standardize intensity data
    strength_mean = df_daily['strength_training'].mean()
    strength_std = df_daily['strength_training'].std()
    aerobic_mean = (df_daily['walking'] + df_daily['cycling']).mean()
    aerobic_std = (df_daily['walking'] + df_daily['cycling']).std()

    df_daily['strength_intensity_std'] = (df_daily['strength_training'] - strength_mean) / strength_std
    df_daily['aerobic_intensity_std'] = ((df_daily['walking'] + df_daily['cycling']) - aerobic_mean) / aerobic_std

    # Prepare weight data
    df_weight['date'] = df_weight['timestamp'].dt.date
    # Convert to same type for merge
    df_daily['date_date'] = df_daily['date'].dt.date
    df_weight = df_weight.merge(df_daily[['date_date']].rename(columns={'date_date': 'date'}), on='date', how='inner')

    # Standardize weight - SAVE THESE PARAMETERS FOR CONVERSION BACK
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Create day index mapping
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)

    # Extract hour of day
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Prepare Stan data with 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)
    H = 7  # 7 time points: 0, 4, 8, 12, 16, 20, 24
    pred_hours = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    pred_hours_scaled = pred_hours / 24.0

    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values,
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values,
        'day_idx': df_weight['day_idx'].values.astype(int),
        'hour_of_day': df_weight['hour_of_day'].values,
        'K': 2,  # 24h and 12h cycles
        'H': H,
        'pred_hours_scaled': pred_hours_scaled
    }

    # Return standardization parameters for conversion back to lbs
    standardization_params = {
        'weight_mean': weight_mean,
        'weight_std': weight_std
    }

    return stan_data, df_weight, df_daily, date_range, pred_hours, standardization_params


def run_constrained_model(stan_data):
    """Run the constrained AR(1) model."""
    print("\nRunning constrained AR(1) model for predictions...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan"

    try:
        model = cmdstanpy.CmdStanModel(stan_file=model_path)
        print(f"✓ Model compiled successfully")
    except Exception as e:
        print(f"✗ Model compilation failed: {e}")
        print(f"  Model path: {model_path}")
        print(f"  Current directory: {Path.cwd()}")
        return None

    # Run MCMC sampling
    print("Running MCMC sampling...")
    try:
        fit = model.sample(
            data=stan_data,
            chains=4,
            parallel_chains=4,
            iter_warmup=500,
            iter_sampling=500,
            show_progress=True,
            seed=12345
        )
        print("✓ MCMC sampling completed")
        return fit
    except Exception as e:
        print(f"✗ MCMC sampling failed: {e}")
        return None


def extract_component_predictions(fit, date_range, pred_hours):
    """Extract component predictions from Stan fit."""
    print("\nExtracting component predictions...")

    D = len(date_range)
    H = len(pred_hours)

    # Extract posterior samples for each component
    intercept_samples = fit.stan_variable('intercept').reshape(-1, D, H)
    strength_samples = fit.stan_variable('strength_fitness').reshape(-1, D, H)
    aerobic_samples = fit.stan_variable('aerobic_fitness').reshape(-1, D, H)
    spline_samples = fit.stan_variable('spline_component').reshape(-1, D, H)
    total_samples = intercept_samples + strength_samples + aerobic_samples + spline_samples
    full_model_samples = fit.stan_variable('pred_weight').reshape(-1, D, H)

    # Compute means and 95% credible intervals
    components = {}
    component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total', 'full_model']
    samples_list = [intercept_samples, strength_samples, aerobic_samples, spline_samples, total_samples, full_model_samples]

    for name, samples in zip(component_names, samples_list):
        mean = samples.mean(axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        components[name] = (mean, lower, upper)

    print(f"✓ Extracted predictions for {D} days × {H} time points")
    return components


def convert_to_lbs(components, standardization_params):
    """Convert standardized predictions back to raw lbs scale."""
    print("Converting predictions to raw lbs scale...")

    weight_mean = standardization_params['weight_mean']
    weight_std = standardization_params['weight_std']

    components_lbs = {}

    for component_name, (mean, lower, upper) in components.items():
        # Convert mean and credible intervals
        mean_lbs = mean * weight_std + weight_mean
        lower_lbs = lower * weight_std + weight_mean
        upper_lbs = upper * weight_std + weight_mean

        components_lbs[component_name] = (mean_lbs, lower_lbs, upper_lbs)

    return components_lbs


def save_component_predictions(components_lbs, date_range, pred_hours, output_dir):
    """Save component predictions to CSV files in lbs scale."""
    print(f"\nSaving component predictions to {output_dir}/...")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create a comprehensive dataframe for each hour
    for h_idx, hour in enumerate(pred_hours):
        hour_str = f"{int(hour):02d}"
        data = {'date': date_range, 'hour': hour}

        for component_name, (mean, lower, upper) in components_lbs.items():
            data[f'{component_name}_mean'] = mean[:, h_idx]
            data[f'{component_name}_lower'] = lower[:, h_idx]
            data[f'{component_name}_upper'] = upper[:, h_idx]

        df_hour = pd.DataFrame(data)
        csv_path = output_dir / f'component_predictions_hour_{hour_str}.csv'
        df_hour.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Create a combined long-format file
    all_data = []
    for h_idx, hour in enumerate(pred_hours):
        for d_idx, date in enumerate(date_range):
            row = {
                'date': date,
                'hour': hour,
                'day_index': d_idx + 1,
                'hour_index': h_idx + 1
            }
            for component_name, (mean, lower, upper) in components_lbs.items():
                row[f'{component_name}_mean'] = mean[d_idx, h_idx]
                row[f'{component_name}_lower'] = lower[d_idx, h_idx]
                row[f'{component_name}_upper'] = upper[d_idx, h_idx]
            all_data.append(row)

    df_all = pd.DataFrame(all_data)
    csv_path = output_dir / 'all_component_predictions.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"  Saved combined file: {csv_path}")

    return output_dir


def create_component_visualizations(components_lbs, df_weight, date_range, pred_hours, output_dir):
    """Create visualizations of component predictions over time with actual weight data."""
    print("\nCreating component visualizations with actual weight data...")

    output_dir = Path(output_dir)

    # 1. Time series of each component at noon (12:00) with actual weight data
    noon_idx = 3  # 12:00 is index 3 in [0, 4, 8, 12, 16, 20, 24]

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
    df_weight_noon = df_weight[(df_weight['hour_of_day'] >= noon_start) & (df_weight['hour_of_day'] <= noon_end)].copy()
    df_weight_noon['timestamp'] = pd.to_datetime(df_weight_noon['date'].astype(str) + ' ' +
                                                 df_weight_noon['hour_of_day'].astype(str).str.replace('.', ':'))

    for idx, (component_name, title) in enumerate(zip(component_names, titles)):
        ax = axes[idx]
        mean, lower, upper = components_lbs[component_name]

        # Plot mean with credible interval
        ax.fill_between(date_range, lower[:, noon_idx], upper[:, noon_idx],
                       alpha=0.3, color='skyblue', label='95% CI')
        ax.plot(date_range, mean[:, noon_idx], 'b-', linewidth=1.5, label='Mean Prediction')

        # Add actual weight data (noon measurements)
        if len(df_weight_noon) > 0:
            ax.scatter(df_weight_noon['timestamp'], df_weight_noon['weight_lbs'],
                      alpha=0.6, s=30, color='red', edgecolor='black', linewidth=0.5,
                      label='Actual Weight (noon)')

        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title(f'{title} at 12:00')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data (not including 0)
        # Combine prediction data and actual weight data
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
    plt.savefig(output_dir / 'component_time_series_noon_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/component_time_series_noon_with_data.png")

    # 2. Heatmap of total predictions over time
    fig, ax = plt.subplots(figsize=(14, 8))

    mean_total, _, _ = components_lbs['total']

    # Create heatmap
    im = ax.imshow(mean_total.T, aspect='auto', cmap='viridis',
                   extent=[date_range[0].toordinal(), date_range[-1].toordinal(),
                           pred_hours[0], pred_hours[-1]],
                   origin='lower')

    # Add actual weight data points on heatmap
    for _, row in df_weight.iterrows():
        date_ordinal = row['date'].toordinal()
        ax.scatter(date_ordinal, row['hour_of_day'],
                  s=20, color='red', edgecolor='white', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Total Predictions Heatmap with Actual Weight Measurements')

    # Format x-axis with dates
    date_ticks = np.linspace(date_range[0].toordinal(), date_range[-1].toordinal(), 8)
    date_labels = [datetime.fromordinal(int(d)).strftime('%Y-%m-%d') for d in date_ticks]
    ax.set_xticks(date_ticks)
    ax.set_xticklabels(date_labels)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (lbs)')

    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'total_predictions_heatmap_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/total_predictions_heatmap_with_data.png")

    # 3. Component contributions at sample dates
    sample_dates = [
        date_range[0],  # First date
        date_range[len(date_range)//4],  # 25%
        date_range[len(date_range)//2],  # 50%
        date_range[3*len(date_range)//4],  # 75%
        date_range[-1]   # Last date
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, sample_date in enumerate(sample_dates):
        if idx >= len(axes):
            break

        ax = axes[idx]
        date_idx = np.where(date_range == sample_date)[0][0]

        # Get component values at this date
        components_at_date = {}
        for component_name in ['intercept', 'strength', 'aerobic', 'spline', 'total']:
            mean, _, _ = components_lbs[component_name]
            components_at_date[component_name] = mean[date_idx, :]

        # Plot stacked area or line plot
        x = pred_hours
        bottom = np.zeros_like(x)

        for component_name, color, label in [
            ('intercept', 'gray', 'Intercept'),
            ('strength', 'blue', 'Strength'),
            ('aerobic', 'green', 'Aerobic'),
            ('spline', 'orange', 'Spline')
        ]:
            values = components_at_date[component_name]
            ax.plot(x, values, 'o-', linewidth=2, markersize=4, label=label, color=color)

        # Add actual weight measurements for this date
        df_date_weights = df_weight[df_weight['date'] == sample_date.date()]
        if len(df_date_weights) > 0:
            ax.scatter(df_date_weights['hour_of_day'], df_date_weights['weight_lbs'],
                      s=40, color='red', edgecolor='black', linewidth=1,
                      label='Actual Weight', zorder=5)

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title(f'Components at {sample_date.date()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data (not including 0)
        # Combine all component data and actual weight data
        all_y_data = []
        for component_name in ['intercept', 'strength', 'aerobic', 'spline', 'total']:
            all_y_data.append(components_at_date[component_name])

        if len(df_date_weights) > 0:
            all_y_data.append(df_date_weights['weight_lbs'].values)

        all_y_data_flat = np.concatenate([d.flatten() if hasattr(d, 'flatten') else np.array(d) for d in all_y_data])

        # Calculate data range with 5% padding
        y_min = np.min(all_y_data_flat)
        y_max = np.max(all_y_data_flat)
        y_range = y_max - y_min
        y_padding = y_range * 0.05  # 5% padding

        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Remove empty subplot if needed
    for idx in range(len(sample_dates), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'component_contributions_sample_dates_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/component_contributions_sample_dates_with_data.png")

    # 4. Daily patterns analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean daily pattern
    ax = axes[0]
    mean_total, _, _ = components_lbs['total']
    mean_daily = mean_total.mean(axis=0)

    ax.plot(pred_hours, mean_daily, 'o-', linewidth=3, markersize=6,
            color='darkblue', label='Mean Daily Pattern')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Mean Daily Weight Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits based on data (not including 0)
    y_min = np.min(mean_daily)
    y_max = np.max(mean_daily)
    y_range = y_max - y_min
    y_padding = y_range * 0.05  # 5% padding
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Right: Daily pattern variability
    ax = axes[1]
    n_days_to_plot = min(50, len(date_range))
    day_indices = np.linspace(0, len(date_range)-1, n_days_to_plot, dtype=int)

    for day_idx in day_indices:
        ax.plot(pred_hours, mean_total[day_idx, :], 'gray', alpha=0.1, linewidth=0.5)

    ax.plot(pred_hours, mean_daily, 'b-', linewidth=3, label='Mean Pattern')

    # Add all actual weight data (transparent)
    ax.scatter(df_weight['hour_of_day'], df_weight['weight_lbs'],
              alpha=0.3, s=20, color='red', label='Actual Weight')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title(f'Daily Pattern Variability ({n_days_to_plot} sample days)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set y-axis limits based on data (not including 0)
    # Combine all daily patterns and actual weight data
    all_y_data = []
    for day_idx in day_indices:
        all_y_data.append(mean_total[day_idx, :])
    all_y_data.append(mean_daily)
    all_y_data.append(df_weight['weight_lbs'].values)

    all_y_data_flat = np.concatenate([d.flatten() if hasattr(d, 'flatten') else np.array(d) for d in all_y_data])

    y_min = np.min(all_y_data_flat)
    y_max = np.max(all_y_data_flat)
    y_range = y_max - y_min
    y_padding = y_range * 0.05  # 5% padding
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()
    plt.savefig(output_dir / 'daily_patterns_analysis_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/daily_patterns_analysis_with_data.png")

    # 5. Prediction vs actual scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get predictions at actual measurement times
    predicted_weights = []
    actual_weights = []

    for _, row in df_weight.iterrows():
        date_idx = np.where(date_range == pd.Timestamp(row['date']))[0][0]

        # Find closest prediction hour
        hour_diffs = np.abs(pred_hours - row['hour_of_day'])
        hour_idx = np.argmin(hour_diffs)

        mean_total, _, _ = components_lbs['total']
        predicted_weight = mean_total[date_idx, hour_idx]
        actual_weight = row['weight_lbs']

        predicted_weights.append(predicted_weight)
        actual_weights.append(actual_weight)

    ax.scatter(actual_weights, predicted_weights, alpha=0.6, s=20)

    # Add diagonal line (perfect predictions)
    min_val = min(min(actual_weights), min(predicted_weights))
    max_val = max(max(actual_weights), max(predicted_weights))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

    ax.set_xlabel('Actual Weight (lbs)')
    ax.set_ylabel('Predicted Weight (lbs)')
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
    plt.savefig(output_dir / 'prediction_vs_actual_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/prediction_vs_actual_scatter.png")


def create_prediction_summary(components_lbs, df_weight, date_range, pred_hours, standardization_params, output_dir):
    """Create a summary report of the predictions in lbs scale."""
    print(f"\nCreating prediction summary report...")

    output_dir = Path(output_dir)

    summary_path = output_dir / 'prediction_summary.md'

    with open(summary_path, 'w') as f:
        f.write("# Component Predictions Summary (lbs scale)\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Information\n\n")
        f.write("- **Model**: `weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan`\n")
        f.write("- **AR(1) component**: Excluded from predictions (measurement-time specific)\n")
        f.write("- **Prediction intervals**: 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)\n")
        f.write(f"- **Time range**: {date_range[0].date()} to {date_range[-1].date()} ({len(date_range)} days)\n")
        f.write(f"- **Weight observations**: {len(df_weight)} measurements\n\n")

        f.write("## Standardization Parameters\n\n")
        f.write(f"- **Weight mean**: {standardization_params['weight_mean']:.2f} lbs\n")
        f.write(f"- **Weight std**: {standardization_params['weight_std']:.2f} lbs\n\n")

        f.write("## Components Predicted\n\n")
        f.write("1. **Intercept**: Baseline weight level\n")
        f.write("2. **Strength Fitness Component**: `γ_s × strength_fitness[t]`\n")
        f.write("3. **Aerobic Fitness Component**: `γ_a × aerobic_fitness[t]`\n")
        f.write("4. **Daily Spline Component**: Fourier basis for intraday variations\n")
        f.write("5. **Total Prediction**: Sum of all components (intercept + strength + aerobic + spline)\n")
        f.write("6. **Full Model Prediction**: Direct output from Stan (for validation)\n\n")

        f.write("## Component Magnitudes in lbs (Mean Absolute Values)\n\n")

        for component_name in ['intercept', 'strength', 'aerobic', 'spline', 'total']:
            mean, _, _ = components_lbs[component_name]
            mean_abs = np.mean(np.abs(mean))
            f.write(f"- **{component_name.capitalize()}**: {mean_abs:.2f} lbs\n")

        f.write("\n## Data Files Generated\n\n")

        f.write("### CSV Files (lbs scale)\n")
        f.write("1. **Per-hour files**: `component_predictions_hour_XX.csv` (XX = 00, 04, 08, 12, 16, 20, 24)\n")
        f.write("   - Columns: date, hour, plus `{component}_mean`, `{component}_lower`, `{component}_upper`\n")
        f.write("2. **Combined file**: `all_component_predictions.csv`\n")
        f.write("   - Long format with all hours and days\n")
        f.write("   - Additional columns: day_index, hour_index\n\n")

        f.write("### Visualization Files (with actual weight data)\n")
        f.write("1. `component_time_series_noon_with_data.png` - Time series at 12:00 with actual weights\n")
        f.write("2. `total_predictions_heatmap_with_data.png` - Heatmap with actual weight points\n")
        f.write("3. `component_contributions_sample_dates_with_data.png` - Components at sample dates with actual weights\n")
        f.write("4. `daily_patterns_analysis_with_data.png` - Daily patterns with actual weights\n")
        f.write("5. `prediction_vs_actual_scatter.png` - Scatter plot of predictions vs actual\n\n")

        f.write("## Usage Notes\n\n")
        f.write("1. **Predictions are in raw lbs scale** (converted from standardized units)\n")
        f.write("2. **AR(1) component is excluded** as it models measurement-time residual correlation\n")
        f.write("3. **Credible intervals (95%)** provided for uncertainty quantification\n")
        f.write("4. **Components are additive**: Total = Intercept + Strength + Aerobic + Spline\n")
        f.write("5. **Actual weight data points** are shown in red on all visualizations\n\n")

        f.write("## Key Statistics (lbs scale)\n\n")

        # Compute some key statistics
        mean_total, lower_total, upper_total = components_lbs['total']

        total_range = mean_total.max() - mean_total.min()
        f.write(f"- **Total prediction range**: {total_range:.2f} lbs\n")
        f.write(f"- **Average intercept**: {np.mean(components_lbs['intercept'][0]):.2f} lbs\n")
        f.write(f"- **Average strength contribution**: {np.mean(components_lbs['strength'][0]):.2f} lbs\n")
        f.write(f"- **Average aerobic contribution**: {np.mean(components_lbs['aerobic'][0]):.2f} lbs\n")
        f.write(f"- **Average spline amplitude**: {np.mean(np.abs(components_lbs['spline'][0])):.2f} lbs\n")

        # Prediction accuracy metrics
        predicted_at_measurements = []
        actual_weights = df_weight['weight_lbs'].values

        for _, row in df_weight.iterrows():
            date_idx = np.where(date_range == pd.Timestamp(row['date']))[0][0]
            hour_diffs = np.abs(pred_hours - row['hour_of_day'])
            hour_idx = np.argmin(hour_diffs)
            predicted_at_measurements.append(mean_total[date_idx, hour_idx])

        predicted_at_measurements = np.array(predicted_at_measurements)
        rmse = np.sqrt(np.mean((actual_weights - predicted_at_measurements) ** 2))
        mae = np.mean(np.abs(actual_weights - predicted_at_measurements))

        f.write(f"- **RMSE**: {rmse:.2f} lbs\n")
        f.write(f"- **MAE**: {mae:.2f} lbs\n")
        f.write(f"- **Correlation**: {np.corrcoef(actual_weights, predicted_at_measurements)[0, 1]:.3f}\n")

    print(f"  Saved: {summary_path}")


def main():
    """Main function to generate component predictions."""
    print("="*60)
    print("Generating Component Predictions at 4-Hour Intervals (lbs scale)")
    print("="*60)

    # Prepare data
    stan_data, df_weight, df_daily, date_range, pred_hours, standardization_params = prepare_data_with_4h_intervals()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {len(date_range)}")
    print(f"  Number of weight observations: {len(df_weight)}")
    print(f"  Prediction hours: {pred_hours.tolist()}")
    print(f"  Time range: {date_range[0].date()} to {date_range[-1].date()}")
    print(f"  Weight mean: {standardization_params['weight_mean']:.2f} lbs")
    print(f"  Weight std: {standardization_params['weight_std']:.2f} lbs")

    # Run model
    fit = run_constrained_model(stan_data)
    if fit is None:
        print("✗ Model fitting failed. Exiting.")
        return

    # Extract component predictions (in standardized units)
    components_std = extract_component_predictions(fit, date_range, pred_hours)

    # Convert to lbs scale
    components_lbs = convert_to_lbs(components_std, standardization_params)

    # Save predictions to CSV files (lbs scale)
    output_dir = "docs/component_predictions_with_data"
    save_component_predictions(components_lbs, date_range, pred_hours, output_dir)

    # Create visualizations with actual weight data
    create_component_visualizations(components_lbs, df_weight, date_range, pred_hours, output_dir)

    # Create summary report
    create_prediction_summary(components_lbs, df_weight, date_range, pred_hours, standardization_params, output_dir)

    print("\n" + "="*60)
    print("Component Predictions Generation Complete (lbs scale)")
    print("="*60)

    print(f"\nSummary:")
    print(f"  - Model: Constrained AR(1) model (AR component excluded)")
    print(f"  - Scale: Raw lbs (converted from standardized units)")
    print(f"  - Prediction intervals: {pred_hours.tolist()} hours")
    print(f"  - Components: Intercept, Strength, Aerobic, Spline, Total")
    print(f"  - Output directory: {output_dir}/")
    print(f"  - Files: CSV predictions in lbs, visualizations with weight data, summary report")
    print(f"\nKey features:")
    print(f"  1. All predictions converted to raw lbs scale")
    print(f"  2. Actual weight data points shown on all plots")
    print(f"  3. Comprehensive summary report with accuracy metrics")


if __name__ == "__main__":
    main()