#!/usr/bin/env python3
"""Generate predictions at 4-hour intervals for each model component (excluding AR(1))."""

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

    # Standardize weight
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

    # Return standardization parameters for converting back to lbs
    standardization_params = {
        'weight_mean': weight_mean,
        'weight_std': weight_std,
        'strength_mean': strength_mean,
        'strength_std': strength_std,
        'aerobic_mean': aerobic_mean,
        'aerobic_std': aerobic_std
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

    # Get sigma_fourier
    sigma_fourier_samples = fit.stan_variable('sigma_fourier')

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
                    spline_value += a_sin[k] * np.sin(freq * hour_scaled) + a_cos[k] * np.cos(freq * hour_scaled)
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


def convert_to_lbs_scale(components, standardization_params):
    """Convert component predictions from standardized units back to lbs scale."""
    print("\nConverting predictions to lbs scale...")

    weight_mean = standardization_params['weight_mean']
    weight_std = standardization_params['weight_std']

    def convert_component(mean, lower, upper):
        """Convert a single component from standardized to lbs scale."""
        mean_lbs = mean * weight_std + weight_mean
        lower_lbs = lower * weight_std + weight_mean
        upper_lbs = upper * weight_std + weight_mean
        return mean_lbs, lower_lbs, upper_lbs

    # Convert all components
    components_lbs = {}
    for component_name, (mean, lower, upper) in components.items():
        components_lbs[component_name] = convert_component(mean, lower, upper)

    return components_lbs


def save_component_predictions(components, date_range, pred_hours, output_dir, lbs_scale=False):
    """Save component predictions to CSV files."""
    print(f"\nSaving component predictions to {output_dir}/...")
    if lbs_scale:
        print("  Saving in lbs scale")

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

        # Add each component
        for component_name, (mean, lower, upper) in components.items():
            data[f'{component_name}_mean'] = mean[:, h_idx]
            data[f'{component_name}_lower'] = lower[:, h_idx]
            data[f'{component_name}_upper'] = upper[:, h_idx]

        df_hour = pd.DataFrame(data)

        # Save to CSV
        suffix = "_lbs" if lbs_scale else ""
        csv_path = output_dir / f'component_predictions_hour_{hour_str}{suffix}.csv'
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

            # Add component values
            for component_name, (mean, lower, upper) in components.items():
                row[f'{component_name}_mean'] = mean[d_idx, h_idx]
                row[f'{component_name}_lower'] = lower[d_idx, h_idx]
                row[f'{component_name}_upper'] = upper[d_idx, h_idx]

            all_data.append(row)

    df_all = pd.DataFrame(all_data)
    suffix = "_lbs" if lbs_scale else ""
    csv_path = output_dir / f'all_component_predictions{suffix}.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"  Saved combined file: {csv_path}")

    return output_dir


def create_component_visualizations(components, date_range, pred_hours, output_dir, df_weight=None, lbs_scale=False):
    """Create visualizations of component predictions over time."""
    print("\nCreating component visualizations...")
    if lbs_scale:
        print("  Creating visualizations in lbs scale")
    if df_weight is not None:
        print("  Adding actual weight data to plots")

    output_dir = Path(output_dir)

    # 1. Time series of each component at noon (12:00)
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

    for idx, (component_name, title) in enumerate(zip(component_names, titles)):
        ax = axes[idx]
        mean, lower, upper = components[component_name]

        # Plot mean with credible interval
        ax.fill_between(date_range, lower[:, noon_idx], upper[:, noon_idx],
                       alpha=0.3, color='skyblue', label='95% CI')
        ax.plot(date_range, mean[:, noon_idx], 'b-', linewidth=1.5, label='Mean')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)

        # Add actual weight data if available (for total and full_model plots)
        if df_weight is not None and component_name in ['total', 'full_model']:
            # Group weight data by date
            df_weight['date_only'] = df_weight['timestamp'].dt.date
            daily_weights = df_weight.groupby('date_only')['weight_lbs'].mean()

            # Plot actual weights as scatter points
            for date, weight in daily_weights.items():
                if date in date_range.date:
                    ax.scatter(date, weight, color='red', s=10, alpha=0.5, zorder=5, label='Actual Weight' if idx == 0 else "")

        ax.set_xlabel('Date')
        ylabel = 'Weight (lbs)' if lbs_scale else 'Standardized Weight'
        ax.set_ylabel(ylabel)
        title_suffix = ' (lbs scale)' if lbs_scale else ''
        ax.set_title(f'{title} at 12:00{title_suffix}')
        if idx == 0 and df_weight is not None:
            ax.legend()
        else:
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    suffix = "_lbs" if lbs_scale else ""
    plt.savefig(output_dir / f'component_time_series_noon{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Heatmap of total predictions over time and hours
    fig, ax = plt.subplots(figsize=(14, 8))

    mean_total = components['total'][0]  # [D, H]

    # Create heatmap
    im = ax.imshow(mean_total.T, aspect='auto', cmap='viridis',
                  extent=[date_range[0], date_range[-1], pred_hours[-1], pred_hours[0]])

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    title_suffix = ' (lbs scale)' if lbs_scale else ''
    ax.set_title(f'Total Predictions Heatmap (All Components){title_suffix}')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar_label = 'Weight (lbs)' if lbs_scale else 'Standardized Weight'
    cbar.set_label(cbar_label)

    # Format x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    suffix = "_lbs" if lbs_scale else ""
    plt.savefig(output_dir / f'total_predictions_heatmap{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Component contributions at specific dates
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
        components_data = []
        component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline']
        component_means = [
            components['intercept'][0][date_idx, :],
            components['strength'][0][date_idx, :],
            components['aerobic'][0][date_idx, :],
            components['spline'][0][date_idx, :]
        ]

        # Plot each component
        for comp_idx, (label, means) in enumerate(zip(component_labels, component_means)):
            ax.plot(pred_hours, means, 'o-', linewidth=2, markersize=4, label=label)

        ax.set_xlabel('Hour of Day')
        ylabel = 'Weight (lbs)' if lbs_scale else 'Standardized Weight'
        ax.set_ylabel(ylabel)
        title_suffix = ' (lbs scale)' if lbs_scale else ''
        ax.set_title(f'Component Contributions: {sample_date.date()}{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pred_hours)

    # Remove empty subplot if needed
    if len(sample_dates) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    suffix = "_lbs" if lbs_scale else ""
    plt.savefig(output_dir / f'component_contributions_sample_dates{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Daily patterns (average across all days)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean daily pattern
    ax = axes[0]
    component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total']
    component_arrays = [
        components['intercept'][0],  # [D, H]
        components['strength'][0],
        components['aerobic'][0],
        components['spline'][0],
        components['total'][0]
    ]

    for label, arr in zip(component_labels, component_arrays):
        daily_mean = np.mean(arr, axis=0)  # Average across days
        ax.plot(pred_hours, daily_mean, 'o-', linewidth=2, markersize=4, label=label)

    ax.set_xlabel('Hour of Day')
    ylabel = 'Weight (lbs)' if lbs_scale else 'Standardized Weight'
    ax.set_ylabel(ylabel)
    title_suffix = ' (lbs scale)' if lbs_scale else ''
    ax.set_title(f'Average Daily Pattern (All Days){title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pred_hours)

    # Right: Variability in daily patterns
    ax = axes[1]
    total_daily = components['total'][0]  # [D, H]

    # Plot individual days (sample of 50 days)
    n_days_to_plot = min(50, len(date_range))
    day_indices = np.linspace(0, len(date_range)-1, n_days_to_plot, dtype=int)

    for day_idx in day_indices:
        ax.plot(pred_hours, total_daily[day_idx, :], 'gray', alpha=0.1, linewidth=0.5)

    # Plot mean pattern
    mean_daily = np.mean(total_daily, axis=0)
    ax.plot(pred_hours, mean_daily, 'b-', linewidth=3, label='Mean Pattern')

    ax.set_xlabel('Hour of Day')
    ylabel = 'Weight (lbs)' if lbs_scale else 'Standardized Weight'
    ax.set_ylabel(ylabel)
    title_suffix = ' (lbs scale)' if lbs_scale else ''
    ax.set_title(f'Daily Pattern Variability ({n_days_to_plot} sample days){title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pred_hours)

    plt.tight_layout()
    suffix = "_lbs" if lbs_scale else ""
    plt.savefig(output_dir / f'daily_patterns_analysis{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualizations saved to {output_dir}/")


def create_prediction_summary(components, date_range, pred_hours, output_dir, lbs_scale=False, standardization_params=None):
    """Create a summary report of the predictions."""
    suffix = "_lbs" if lbs_scale else ""
    report_path = Path(output_dir) / f'prediction_summary{suffix}.md'

    with open(report_path, 'w') as f:
        f.write("# Component Predictions Summary\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Information\n\n")
        f.write("- **Model**: `weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_constrained.stan`\n")
        f.write("- **AR(1) component**: Excluded from predictions (measurement-time specific)\n")
        f.write("- **Prediction intervals**: 4-hour intervals (0, 4, 8, 12, 16, 20, 24 hours)\n")
        f.write(f"- **Time range**: {date_range[0].date()} to {date_range[-1].date()} ({len(date_range)} days)\n\n")

        f.write("## Components Predicted\n\n")
        f.write("1. **Intercept**: Baseline weight level\n")
        f.write("2. **Strength Fitness Component**: `γ_s × strength_fitness[t]`\n")
        f.write("3. **Aerobic Fitness Component**: `γ_a × aerobic_fitness[t]`\n")
        f.write("4. **Daily Spline Component**: Fourier basis for intraday variations\n")
        f.write("5. **Total Prediction**: Sum of all components (intercept + strength + aerobic + spline)\n")
        f.write("6. **Full Model Prediction**: Direct output from Stan (for validation)\n\n")

        f.write("## Component Magnitudes (Mean Absolute Values)\n\n")

        # Compute average magnitudes
        component_names = ['intercept', 'strength', 'aerobic', 'spline', 'total']
        component_labels = ['Intercept', 'Strength', 'Aerobic', 'Spline', 'Total']

        for name, label in zip(component_names, component_labels):
            mean_component = components[name][0]  # [D, H]
            avg_magnitude = np.mean(np.abs(mean_component))
            f.write(f"- **{label}**: {avg_magnitude:.4f}\n")

        f.write("\n## Data Files Generated\n\n")
        f.write("### CSV Files\n")
        f.write("1. **Per-hour files**: `component_predictions_hour_XX.csv` (XX = 00, 04, 08, 12, 16, 20, 24)\n")
        f.write("   - Columns: date, hour, plus `{component}_mean`, `{component}_lower`, `{component}_upper`\n")
        f.write("2. **Combined file**: `all_component_predictions.csv`\n")
        f.write("   - Long format with all hours and days\n")
        f.write("   - Additional columns: day_index, hour_index\n\n")

        f.write("### Visualization Files\n")
        f.write("1. `component_time_series_noon.png` - Time series of each component at 12:00\n")
        f.write("2. `total_predictions_heatmap.png` - Heatmap of total predictions over time and hours\n")
        f.write("3. `component_contributions_sample_dates.png` - Component contributions at sample dates\n")
        f.write("4. `daily_patterns_analysis.png` - Daily patterns and variability\n\n")

        f.write("## Usage Notes\n\n")
        if lbs_scale:
            f.write("1. **Predictions are in lbs scale** (converted from standardized units)\n")
            f.write("2. **Standardization parameters**: Mean = {:.2f} lbs, Std = {:.2f} lbs\n".format(
                standardization_params['weight_mean'], standardization_params['weight_std']))
        else:
            f.write("1. **Predictions are in standardized units** (mean=0, std=1 of original weight data)\n")
        f.write("{}2. **AR(1) component is excluded** as it models measurement-time residual correlation\n".format("3. " if lbs_scale else ""))
        f.write("{}3. **Credible intervals (95%)** provided for uncertainty quantification\n".format("4. " if lbs_scale else ""))
        f.write("{}4. **Components are additive**: Total = Intercept + Strength + Aerobic + Spline\n".format("5. " if lbs_scale else ""))

        # Add key statistics
        f.write("\n## Key Statistics\n\n")

        # Total prediction range
        total_mean = components['total'][0]
        total_min = np.min(total_mean)
        total_max = np.max(total_mean)
        total_range = total_max - total_min

        f.write(f"- **Total prediction range**: {total_range:.3f} (min: {total_min:.3f}, max: {total_max:.3f})\n")

        # Component contributions
        intercept_mean = components['intercept'][0]
        strength_mean = components['strength'][0]
        aerobic_mean = components['aerobic'][0]
        spline_mean = components['spline'][0]

        f.write(f"- **Average intercept**: {np.mean(intercept_mean):.3f}\n")
        f.write(f"- **Average strength contribution**: {np.mean(strength_mean):.3f}\n")
        f.write(f"- **Average aerobic contribution**: {np.mean(aerobic_mean):.3f}\n")
        f.write(f"- **Average spline amplitude**: {np.mean(np.abs(spline_mean)):.3f}\n")

        # Daily pattern strength
        spline_daily_range = np.max(spline_mean, axis=1) - np.min(spline_mean, axis=1)
        avg_daily_range = np.mean(spline_daily_range)
        f.write(f"- **Average daily spline range**: {avg_daily_range:.3f}\n")

    print(f"✓ Summary report saved to {report_path}")


def main():
    """Main function to generate component predictions."""
    print("Generating Component Predictions at 4-Hour Intervals")
    print("=" * 60)

    # Prepare data with 4-hour intervals
    stan_data, df_weight, df_daily, date_range, pred_hours, standardization_params = prepare_data_with_4h_intervals()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {len(date_range)}")
    print(f"  Number of weight observations: {len(df_weight)}")
    print(f"  Prediction hours: {pred_hours.tolist()}")
    print(f"  Time range: {date_range[0].date()} to {date_range[-1].date()}")

    # Run the constrained model
    fit = run_constrained_model(stan_data)

    if fit is None:
        print("Failed to run model. Exiting.")
        return

    # Extract component predictions
    components = extract_component_predictions(fit, date_range, pred_hours)

    # Save predictions to CSV files (standardized units)
    output_dir = "docs/component_predictions"
    save_component_predictions(components, date_range, pred_hours, output_dir, lbs_scale=False)

    # Create visualizations (standardized units)
    create_component_visualizations(components, date_range, pred_hours, output_dir,
                                   df_weight=df_weight, lbs_scale=False)

    # Create summary report (standardized units)
    create_prediction_summary(components, date_range, pred_hours, output_dir,
                             lbs_scale=False, standardization_params=standardization_params)

    # Convert to lbs scale
    components_lbs = convert_to_lbs_scale(components, standardization_params)

    # Save predictions to CSV files (lbs scale)
    save_component_predictions(components_lbs, date_range, pred_hours, output_dir, lbs_scale=True)

    # Create visualizations (lbs scale)
    create_component_visualizations(components_lbs, date_range, pred_hours, output_dir,
                                   df_weight=df_weight, lbs_scale=True)

    # Create summary report (lbs scale)
    create_prediction_summary(components_lbs, date_range, pred_hours, output_dir,
                             lbs_scale=True, standardization_params=standardization_params)

    print("\n" + "="*60)
    print("Component Predictions Generation Complete")
    print("="*60)

    print(f"\nSummary:")
    print(f"  - Model: Constrained AR(1) model (AR component excluded)")
    print(f"  - Prediction intervals: {pred_hours.tolist()} hours")
    print(f"  - Components: Intercept, Strength, Aerobic, Spline, Total")
    print(f"  - Output directory: {output_dir}/")
    print(f"  - Files: CSV predictions, visualizations, summary report")

    print(f"\nKey outputs:")
    print(f"  1. CSV files with component predictions at each hour")
    print(f"  2. Visualizations of component time series and patterns")
    print(f"  3. Summary report with statistics and usage notes")


if __name__ == "__main__":
    main()