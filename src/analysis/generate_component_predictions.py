#!/usr/bin/env python3
"""Generate predictions at 4-hour intervals with raw weight data in lbs scale."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cmdstanpy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        return None

    # Run sampling with conservative settings
    print("Running MCMC sampling...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=500,
        iter_sampling=500,
        seed=12345,
        adapt_delta=0.95,
        max_treedepth=12,
        show_console=False,
        show_progress=True
    )

    return fit


def extract_component_predictions(fit, date_range, pred_hours):
    """Extract predictions for each model component at 4-hour intervals."""
    print("\nExtracting component predictions...")

    # Get posterior samples
    n_samples = fit.num_draws_sampling
    D = len(date_range)
    H = len(pred_hours)

    # Extract component predictions from generated quantities
    # Note: We need to compute components from parameters since they're not directly stored

    # Get parameters
    weight_intercept_samples = fit.stan_variable('weight_intercept')
    gamma_s_samples = fit.stan_variable('gamma_s')
    gamma_a_samples = fit.stan_variable('gamma_a')

    # Get fitness states
    strength_fitness_samples = fit.stan_variable('strength_fitness_stored')  # [samples, D]
    aerobic_fitness_samples = fit.stan_variable('aerobic_fitness_stored')    # [samples, D]

    # Get Fourier coefficients
    a_sin_samples = fit.stan_variable('a_sin_stored')  # [samples, K]
    a_cos_samples = fit.stan_variable('a_cos_stored')  # [samples, K]

    # Get daily predictions (without AR component)
    mu_pred_all_days_no_ar_samples = fit.stan_variable('mu_pred_all_days_no_ar')  # [samples, D, H]

    # Compute component contributions for each sample
    print("Computing component contributions...")

    # Initialize arrays for component predictions
    intercept_component = np.zeros((n_samples, D, H))
    strength_component = np.zeros((n_samples, D, H))
    aerobic_component = np.zeros((n_samples, D, H))
    spline_component = np.zeros((n_samples, D, H))

    # For each MCMC sample
    for s in range(n_samples):
        if s % 100 == 0:
            print(f"  Processing sample {s+1}/{n_samples}...")

        # Extract parameters for this sample
        intercept = weight_intercept_samples[s]
        gamma_s = gamma_s_samples[s]
        gamma_a = gamma_a_samples[s]
        strength_fitness = strength_fitness_samples[s, :]  # [D]
        aerobic_fitness = aerobic_fitness_samples[s, :]    # [D]
        a_sin = a_sin_samples[s, :]  # [K]
        a_cos = a_cos_samples[s, :]  # [K]

        # For each day and hour
        for d in range(D):
            for h_idx, hour_scaled in enumerate(np.array(pred_hours) / 24.0):
                # Intercept component
                intercept_component[s, d, h_idx] = intercept

                # Strength fitness component
                strength_component[s, d, h_idx] = gamma_s * strength_fitness[d]

                # Aerobic fitness component
                aerobic_component[s, d, h_idx] = gamma_a * aerobic_fitness[d]

                # Daily spline component
                spline_value = 0.0
                for k in range(len(a_sin)):
                    freq = 2.0 * np.pi * (k + 1)  # k is 0-indexed, harmonics are 1-indexed
                    spline_value += a_sin[k] * np.sin(freq * hour_scaled) + a_cos[k] * cos(freq * hour_scaled)
                spline_component[s, d, h_idx] = spline_value

    # Compute total prediction (intercept + strength + aerobic + spline)
    total_prediction = intercept_component + strength_component + aerobic_component + spline_component

    # Compute summary statistics (mean and 95% credible interval)
    def compute_summary(component_array):
        """Compute mean and 95% CI for a component array."""
        mean = np.mean(component_array, axis=0)  # [D, H]
        lower = np.percentile(component_array, 2.5, axis=0)
        upper = np.percentile(component_array, 97.5, axis=0)
        return mean, lower, upper

    print("Computing summary statistics...")
    components = {
        'intercept': compute_summary(intercept_component),
        'strength': compute_summary(strength_component),
        'aerobic': compute_summary(aerobic_component),
        'spline': compute_summary(spline_component),
        'total': compute_summary(total_prediction),
        'full_model': compute_summary(mu_pred_all_days_no_ar_samples)  # From Stan directly
    }

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

        # Create dataframe for this hour
        data = {
            'date': date_range,
            'hour': hour
        }

        # Add each component in lbs
        for component_name, (mean, lower, upper) in components_lbs.items():
            data[f'{component_name}_mean_lbs'] = mean[:, h_idx]
            data[f'{component_name}_lower_lbs'] = lower[:, h_idx]
            data[f'{component_name}_upper_lbs'] = upper[:, h_idx]

        df_hour = pd.DataFrame(data)

        # Save to CSV
        csv_path = output_dir / f'component_predictions_hour_{hour_str}_lbs.csv'
        df_hour.to_csv(csv_path, index=False)
        print(f"  Saved hour {hour_str}: {csv_path}")

    # Also create a summary file with all hours combined (long format)
    all_data = []
    for h_idx, hour in enumerate(pred_hours):
        for d_idx, date in enumerate(date_range):
            row = {
                'date': date,
                'hour': hour,
                'day_index': d_idx + 1,
                'hour_index': h_idx + 1
            }

            # Add component values in lbs
            for component_name, (mean, lower, upper) in components_lbs.items():
                row[f'{component_name}_mean_lbs'] = mean[d_idx, h_idx]
                row[f'{component_name}_lower_lbs'] = lower[d_idx, h_idx]
                row[f'{component_name}_upper_lbs'] = upper[d_idx, h_idx]

            all_data.append(row)

    df_all = pd.DataFrame(all_data)
    csv_path = output_dir / 'all_component_predictions_lbs.csv'
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
    df_weight_noon = df_weight.copy()
    # Add hour of day to weight data
    df_weight_noon['hour'] = df_weight_noon['timestamp'].dt.hour + df_weight_noon['timestamp'].dt.minute / 60.0
    # Filter for weights measured between 11 AM and 1 PM (noon ± 1 hour)
    df_weight_noon_filtered = df_weight_noon[(df_weight_noon['hour'] >= 11) & (df_weight_noon['hour'] <= 13)].copy()

    for idx, (component_name, title) in enumerate(zip(component_names, titles)):
        ax = axes[idx]
        mean, lower, upper = components_lbs[component_name]

        # Plot component mean with credible interval
        ax.fill_between(date_range, lower[:, noon_idx], upper[:, noon_idx],
                       alpha=0.3, color='skyblue', label='95% CI')
        ax.plot(date_range, mean[:, noon_idx], 'b-', linewidth=1.5, label=f'{title} Prediction')

        # Add actual weight data (noon measurements)
        if len(df_weight_noon_filtered) > 0:
            ax.scatter(df_weight_noon_filtered['timestamp'], df_weight_noon_filtered['weight_lbs'],
                      color='red', s=20, alpha=0.6, label='Actual Weight (11AM-1PM)', zorder=5)

        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title(f'{title} at 12:00\nwith Actual Weight Data')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on data (not including 0)
        # Combine prediction data and actual weight data
        all_y_data = np.concatenate([
            mean[:, noon_idx],
            lower[:, noon_idx],
            upper[:, noon_idx]
        ])
        if len(df_weight_noon_filtered) > 0:
            all_y_data = np.concatenate([all_y_data, df_weight_noon_filtered['weight_lbs'].values])

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

    # 2. Heatmap of total predictions over time and hours
    fig, ax = plt.subplots(figsize=(14, 8))

    mean_total = components_lbs['total'][0]  # [D, H]

    # Create heatmap
    im = ax.imshow(mean_total.T, aspect='auto', cmap='viridis',
                  extent=[date_range[0], date_range[-1], pred_hours[-1], pred_hours[0]])

    # Add actual weight data points on heatmap
    # Convert weight timestamps to coordinates
    for _, row in df_weight.iterrows():
        timestamp = row['timestamp']
        hour = row['hour_of_day']
        weight = row['weight_lbs']

        # Find closest date in date_range
        date_diff = np.abs((date_range - timestamp).total_seconds())
        if date_diff.min() < 12*3600:  # Within 12 hours
            date_idx = np.argmin(date_diff)
            ax.plot(timestamp, hour, 'ro', markersize=3, alpha=0.6, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Total Predictions Heatmap with Actual Weight Measurements\n(Red dots = actual measurements)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight (lbs)')

    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'total_predictions_heatmap_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Component contributions at specific dates with weight data for those dates
    sample_dates = [
        date_range[0],  # First date
        date_range[len(date_range)//4],  # ~25%
        date_range[len(date_range)//2],  # ~50%
        date_range[3*len(date_range)//4],  # ~75%
        date_range[-1]   # Last date
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, sample_date in enumerate(sample_dates):
        if idx >= len(axes):
            break

        ax = axes[idx]
        date_idx = np.where(date_range == sample_date)[0][0]

        # Get component means for this date
        component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total']
        component_means = [
            components_lbs['intercept'][0][date_idx, :],
            components_lbs['strength'][0][date_idx, :],
            components_lbs['aerobic'][0][date_idx, :],
            components_lbs['spline'][0][date_idx, :],
            components_lbs['total'][0][date_idx, :]
        ]

        # Plot each component
        for comp_idx, (label, means) in enumerate(zip(component_labels, component_means)):
            ax.plot(pred_hours, means, 'o-', linewidth=2, markersize=4, label=label)

        # Add actual weight measurements for this date
        date_str = sample_date.date()
        df_date_weights = df_weight[df_weight['date'] == date_str]

        if len(df_date_weights) > 0:
            ax.scatter(df_date_weights['hour_of_day'], df_date_weights['weight_lbs'],
                      color='black', s=50, alpha=0.8, label='Actual Weight', zorder=10,
                      edgecolors='white', linewidth=1)

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title(f'Component Contributions: {sample_date.date()}\nwith Actual Weight Data')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pred_hours)

    # Remove empty subplot if needed
    if len(sample_dates) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'component_contributions_with_data.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Daily patterns (average across all days) with all weight data
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean daily pattern
    ax = axes[0]
    component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total']
    component_arrays = [
        components_lbs['intercept'][0],  # [D, H]
        components_lbs['strength'][0],
        components_lbs['aerobic'][0],
        components_lbs['spline'][0],
        components_lbs['total'][0]
    ]

    for label, arr in zip(component_labels, component_arrays):
        daily_mean = np.mean(arr, axis=0)  # Average across days
        ax.plot(pred_hours, daily_mean, 'o-', linewidth=2, markersize=4, label=label)

    # Add all actual weight data (transparent)
    ax.scatter(df_weight['hour_of_day'], df_weight['weight_lbs'],
              color='gray', s=10, alpha=0.2, label='All Weight Measurements')

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Average Daily Pattern (All Days)\nwith All Weight Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pred_hours)

    # Right: Model predictions vs actual measurements
    ax = axes[1]

    # Get total predictions at measurement times
    # For each weight measurement, find closest prediction hour
    predicted_weights = []
    actual_weights = []

    for _, row in df_weight.iterrows():
        actual_weight = row['weight_lbs']
        hour = row['hour_of_day']
        date = row['date']

        # Find date index
        date_idx = np.where(date_range.date == date)[0][0]

        # Find closest prediction hour
        hour_idx = np.argmin(np.abs(pred_hours - hour))

        # Get prediction for this date and hour
        predicted_weight = components_lbs['total'][0][date_idx, hour_idx]

        predicted_weights.append(predicted_weight)
        actual_weights.append(actual_weight)

    # Scatter plot: predicted vs actual
    ax.scatter(actual_weights, predicted_weights, alpha=0.6, s=20)

    # Add diagonal line (perfect prediction)
    min_val = min(min(actual_weights), min(predicted_weights))
    max_val = max(max(actual_weights), max(predicted_weights))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')

    ax.set_xlabel('Actual Weight (lbs)')
    ax.set_ylabel('Predicted Weight (lbs)')
    ax.set_title('Model Predictions vs Actual Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = np.corrcoef(actual_weights, predicted_weights)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'model_vs_actual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}/")


def create_prediction_summary(components_lbs, df_weight, date_range, pred_hours, standardization_params, output_dir):
    """Create a summary report of the predictions in lbs scale."""
    report_path = Path(output_dir) / 'prediction_summary_lbs.md'

    with open(report_path, 'w') as f:
        f.write("# Component Predictions Summary (lbs scale)\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Information\n\n")
        f.write("- **Model**: `weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan`\n")
        f.write("- **AR(1) component**: Excluded from predictions (measurement-time specific)\n")
        f.write("- **Prediction intervals**: 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)\n")
        f.write(f"- **Time range**: {date_range[0].date()} to {date_range[-1].date()} ({len(date_range)} days)\n")
        f.write(f"- **Weight measurements**: {len(df_weight)} observations\n\n")

        f.write("## Standardization Parameters\n\n")
        f.write(f"- **Weight mean**: {standardization_params['weight_mean']:.2f} lbs\n")
        f.write(f"- **Weight std**: {standardization_params['weight_std']:.2f} lbs\n\n")

        f.write("## Components Predicted (in lbs)\n\n")
        f.write("1. **Intercept**: Baseline weight level\n")
        f.write("2. **Strength Fitness Component**: `γ_s × strength_fitness[t]`\n")
        f.write("3. **Aerobic Fitness Component**: `γ_a × aerobic_fitness[t]`\n")
        f.write("4. **Daily Spline Component**: Fourier basis for intraday variations\n")
        f.write("5. **Total Prediction**: Sum of all components (intercept + strength + aerobic + spline)\n")
        f.write("6. **Full Model Prediction**: Direct output from Stan (for validation)\n\n")

        f.write("## Component Magnitudes (Mean Absolute Values in lbs)\n\n")

        # Compute average magnitudes
        component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total']
        component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total']

        for name, label in zip(component_names, component_labels):
            mean_component = components_lbs[name][0]  # [D, H]
            avg_magnitude = np.mean(np.abs(mean_component))
            f.write(f"- **{label}**: {avg_magnitude:.2f} lbs\n")

        f.write("\n## Data Files Generated\n\n")
        f.write("### CSV Files (lbs scale)\n")
        f.write("1. **Per-hour files**: `component_predictions_hour_XX_lbs.csv` (XX = 00, 04, 08, 12, 16, 20, 24)\n")
        f.write("   - Columns: date, hour, plus `{component}_mean_lbs`, `{component}_lower_lbs`, `{component}_upper_lbs`\n")
        f.write("2. **Combined file**: `all_component_predictions_lbs.csv`\n")
        f.write("   - Long format with all hours and days\n")
        f.write("   - Additional columns: day_index, hour_index\n\n")

        f.write("### Visualization Files (with actual weight data)\n")
        f.write("1. `component_time_series_noon_with_data.png` - Time series at 12:00 with actual weights\n")
        f.write("2. `total_predictions_heatmap_with_data.png` - Heatmap with actual measurements (red dots)\n")
        f.write("3. `component_contributions_with_data.png` - Component contributions with weight data\n")
        f.write("4. `model_vs_actual_analysis.png` - Predictions vs actual measurements scatter plot\n\n")

        f.write("## Key Statistics (lbs scale)\n\n")

        # Total prediction range
        total_mean = components_lbs['total'][0]
        total_min = np.min(total_mean)
        total_max = np.max(total_mean)
        total_range = total_max - total_min

        f.write(f"- **Total prediction range**: {total_range:.1f} lbs (min: {total_min:.1f}, max: {total_max:.1f})\n")

        # Component contributions
        intercept_mean = components_lbs['intercept'][0]
        strength_mean = components_lbs['strength'][0]
        aerobic_mean = components_lbs['aerobic'][0]
        spline_mean = components_lbs['spline'][0]

        f.write(f"- **Average intercept**: {np.mean(intercept_mean):.1f} lbs\n")
        f.write(f"- **Average strength contribution**: {np.mean(strength_mean):.1f} lbs\n")
        f.write(f"- **Average aerobic contribution**: {np.mean(aerobic_mean):.1f} lbs\n")
        f.write(f"- **Average spline amplitude**: {np.mean(np.abs(spline_mean)):.1f} lbs\n")

        # Daily pattern strength
        spline_daily_range = np.max(spline_mean, axis=1) - np.min(spline_mean, axis=1)
        avg_daily_range = np.mean(spline_daily_range)
        f.write(f"- **Average daily spline range**: {avg_daily_range:.1f} lbs\n")

        # Actual weight statistics
        f.write(f"\n## Actual Weight Data Statistics\n\n")
        f.write(f"- **Number of measurements**: {len(df_weight)}\n")
        f.write(f"- **Actual weight range**: {df_weight['weight_lbs'].min():.1f} to {df_weight['weight_lbs'].max():.1f} lbs\n")
        f.write(f"- **Actual weight mean**: {df_weight['weight_lbs'].mean():.1f} lbs\n")
        f.write(f"- **Actual weight std**: {df_weight['weight_lbs'].std():.1f} lbs\n")

        # Model performance
        # Calculate predictions at measurement times
        predicted_at_measurements = []
        for _, row in df_weight.iterrows():
            hour = row['hour_of_day']
            date = row['date']

            # Find date index
            date_idx = np.where(date_range.date == date)[0][0]

            # Find closest prediction hour
            hour_idx = np.argmin(np.abs(pred_hours - hour))

            # Get prediction
            predicted_at_measurements.append(components_lbs['total'][0][date_idx, hour_idx])

        # Calculate RMSE
        actual_weights = df_weight['weight_lbs'].values
        rmse = np.sqrt(np.mean((actual_weights - predicted_at_measurements) ** 2))
        mae = np.mean(np.abs(actual_weights - predicted_at_measurements))

        f.write(f"\n## Model Performance at Measurement Times\n\n")
        f.write(f"- **RMSE**: {rmse:.2f} lbs\n")
        f.write(f"- **MAE**: {mae:.2f} lbs\n")
        f.write(f"- **Correlation**: {np.corrcoef(actual_weights, predicted_at_measurements)[0, 1]:.3f}\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main function to generate component predictions with raw weight data."""
    print("Generating Component Predictions at 4-Hour Intervals (lbs scale)")
    print("=" * 60)

    # Prepare data with 4-hour intervals
    stan_data, df_weight, df_daily, date_range, pred_hours, standardization_params = prepare_data_with_4h_intervals()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {len(date_range)}")
    print(f"  Number of weight observations: {len(df_weight)}")
    print(f"  Prediction hours: {pred_hours.tolist()}")
    print(f"  Time range: {date_range[0].date()} to {date_range[-1].date()}")
    print(f"  Weight mean: {standardization_params['weight_mean']:.2f} lbs")
    print(f"  Weight std: {standardization_params['weight_std']:.2f} lbs")

    # Run the constrained model
    fit = run_constrained_model(stan_data)

    if fit is None:
        print("Failed to run model. Exiting.")
        return

    # Extract component predictions (in standardized units)
    components_std = extract_component_predictions(fit, date_range, pred_hours)

    # Convert to lbs scale
    components_lbs = convert_to_lbs(components_std, standardization_params)

    # Save predictions to CSV files (lbs scale)
    output_dir = "docs/component_predictions_lbs"
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

    print(f"\nKey outputs:")
    print(f"  1. CSV files with component predictions in lbs at each hour")
    print(f"  2. Visualizations with actual weight data overlaid")
    print(f"  3. Model performance statistics (RMSE, MAE, correlation)")
    print(f"  4. Summary report with all statistics in lbs scale")


if __name__ == "__main__":
    main()