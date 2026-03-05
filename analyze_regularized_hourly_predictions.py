#!/usr/bin/env python3
"""Analyze horseshoe model with hourly predictions for all days."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import arviz as az
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_data_with_hourly_predictions():
    """Prepare data with hour_of_day for spline model and hourly predictions."""
    print("Preparing data with hourly predictions...")

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

    # Define hours to predict at (0, 6, 12, 18, 24 hours)
    H = 5
    pred_hours = np.array([0.0, 6.0, 12.0, 18.0, 24.0])
    pred_hours_scaled = pred_hours / 24.0

    # Prepare Stan data
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

    return stan_data, df_weight, df_daily, pred_hours, date_range


def run_model(stan_data, chains=4, iter_warmup=500, iter_sampling=500):
    """Run the regularized model."""
    print("\nRunning regularized model...")

    model_path = "stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline_regularized.stan"
    model = cmdstanpy.CmdStanModel(stan_file=model_path)

    print(f"  Chains: {chains}, Warmup: {iter_warmup}, Sampling: {iter_sampling}")

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        show_progress=True,
        seed=12345
    )

    return fit


def extract_hourly_predictions(fit, date_range, pred_hours):
    """Extract hourly predictions from model fit."""
    print("\nExtracting hourly predictions...")

    # Get predictions for all days at all hours
    # fit.stan_variable() returns array of shape (draws, D, H)
    y_pred_all_days = fit.stan_variable('y_pred_all_days')

    # Convert to numpy array for easier manipulation
    # Shape: (draws, D, H)
    n_draws = y_pred_all_days.shape[0]
    D = y_pred_all_days.shape[1]
    H = y_pred_all_days.shape[2]

    print(f"  Draws: {n_draws}, Days: {D}, Hours: {H}")

    # Calculate summary statistics
    # Mean across draws for each day-hour combination
    y_pred_mean = np.mean(y_pred_all_days, axis=0)  # Shape: (D, H)
    y_pred_std = np.std(y_pred_all_days, axis=0)    # Shape: (D, H)

    # Calculate 90% credible intervals
    y_pred_lower = np.percentile(y_pred_all_days, 5, axis=0)   # 5th percentile
    y_pred_upper = np.percentile(y_pred_all_days, 95, axis=0)  # 95th percentile

    # Create DataFrame with predictions
    pred_data = []
    for day_idx in range(D):
        date = date_range[day_idx]
        for hour_idx in range(H):
            hour = pred_hours[hour_idx]
            pred_data.append({
                'date': date,
                'hour': hour,
                'pred_mean': y_pred_mean[day_idx, hour_idx],
                'pred_std': y_pred_std[day_idx, hour_idx],
                'pred_lower': y_pred_lower[day_idx, hour_idx],
                'pred_upper': y_pred_upper[day_idx, hour_idx]
            })

    df_pred = pd.DataFrame(pred_data)

    return df_pred, y_pred_all_days


def visualize_hourly_predictions(df_pred, df_weight, output_dir="docs/regularized_hourly_predictions"):
    """Create visualizations of hourly predictions."""
    print(f"\nCreating visualizations in {output_dir}...")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Daily pattern for a specific day
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Select a few representative days
    sample_dates = df_pred['date'].unique()
    if len(sample_dates) >= 4:
        plot_dates = [sample_dates[0], sample_dates[len(sample_dates)//3],
                     sample_dates[2*len(sample_dates)//3], sample_dates[-1]]
    else:
        plot_dates = sample_dates[:min(4, len(sample_dates))]

    for idx, plot_date in enumerate(plot_dates):
        if idx >= len(axes):
            break

        date_pred = df_pred[df_pred['date'] == plot_date]

        ax = axes[idx]
        ax.plot(date_pred['hour'], date_pred['pred_mean'], 'b-', linewidth=2, label='Mean prediction')
        ax.fill_between(date_pred['hour'], date_pred['pred_lower'], date_pred['pred_upper'],
                       alpha=0.3, color='b', label='90% CI')

        # Add actual weight measurements for this day if available
        date_str = plot_date.strftime('%Y-%m-%d')
        day_weights = df_weight[df_weight['date'] == pd.Timestamp(plot_date).date()]
        if not day_weights.empty:
            ax.scatter(day_weights['hour_of_day'], day_weights['weight_std'],
                      color='red', s=50, zorder=5, label='Actual measurements')

        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Weight (standardized)')
        ax.set_title(f'Daily Pattern: {date_str}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/daily_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Time series of predictions at a specific hour (e.g., noon)
    fig, ax = plt.subplots(figsize=(12, 6))

    noon_pred = df_pred[df_pred['hour'] == 12.0].sort_values('date')

    ax.plot(noon_pred['date'], noon_pred['pred_mean'], 'b-', linewidth=1.5, label='Noon prediction')
    ax.fill_between(noon_pred['date'], noon_pred['pred_lower'], noon_pred['pred_upper'],
                   alpha=0.3, color='b', label='90% CI')

    # Add actual weight measurements
    ax.scatter(df_weight['timestamp'], df_weight['weight_std'],
              color='red', s=20, alpha=0.6, label='Actual measurements')

    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (standardized)')
    ax.set_title('Time Series: Predictions at Noon vs Actual Measurements')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/noon_time_series.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Heatmap of daily patterns over time
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create pivot table for heatmap
    heatmap_data = df_pred.pivot_table(index='date', columns='hour', values='pred_mean')

    im = ax.imshow(heatmap_data.T, aspect='auto', cmap='viridis',
                  extent=[mdates.date2num(heatmap_data.index[0]),
                         mdates.date2num(heatmap_data.index[-1]),
                         heatmap_data.columns[-1], heatmap_data.columns[0]])

    ax.set_xlabel('Date')
    ax.set_ylabel('Hour of Day')
    ax.set_title('Heatmap: Daily Weight Patterns Over Time')

    # Format x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.colorbar(im, ax=ax, label='Weight (standardized)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_daily_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualizations to {output_dir}/")


def analyze_ar_component(fit, output_dir="docs/regularized_hourly_predictions"):
    """Analyze the AR(1) component shrinkage."""
    print("\nAnalyzing AR(1) component shrinkage...")

    # Extract AR(1) parameters
    rho_samples = fit.stan_variable('rho')
    sigma_epsilon_samples = fit.stan_variable('sigma_epsilon')

    # Create summary
    ar_summary = {
        'rho_mean': np.mean(rho_samples),
        'rho_std': np.std(rho_samples),
        'rho_90ci_lower': np.percentile(rho_samples, 5),
        'rho_90ci_upper': np.percentile(rho_samples, 95),
        'sigma_epsilon_mean': np.mean(sigma_epsilon_samples),
        'sigma_epsilon_std': np.std(sigma_epsilon_samples)
    }

    print("  AR(1) Parameter Summary:")
    print(f"    ρ (autocorrelation): {ar_summary['rho_mean']:.3f} [{ar_summary['rho_90ci_lower']:.3f}, {ar_summary['rho_90ci_upper']:.3f}]")
    print(f"    σ_ε (innovation scale): {ar_summary['sigma_epsilon_mean']:.3f} ± {ar_summary['sigma_epsilon_std']:.3f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rho distribution
    axes[0].hist(rho_samples, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(ar_summary['rho_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {ar_summary["rho_mean"]:.3f}')
    axes[0].set_xlabel('ρ (AR(1) autocorrelation)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of AR(1) Autocorrelation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot sigma_epsilon distribution
    axes[1].hist(sigma_epsilon_samples, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].axvline(ar_summary['sigma_epsilon_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {ar_summary["sigma_epsilon_mean"]:.3f}')
    axes[1].set_xlabel('σ_ε (Innovation scale)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of AR(1) Innovation Scale')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ar_component_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

    return ar_summary


def main():
    """Main analysis function."""
    print("Analyzing regularized model with hourly predictions")
    print("=" * 60)

    # Prepare data
    stan_data, df_weight, df_daily, pred_hours, date_range = prepare_data_with_hourly_predictions()

    print(f"\nData dimensions:")
    print(f"  Number of days (D): {stan_data['D']}")
    print(f"  Number of weight observations: {stan_data['N_weight']}")
    print(f"  Fourier harmonics (K): {stan_data['K']}")
    print(f"  Prediction hours (H): {stan_data['H']}")
    print(f"  Hours: {pred_hours}")

    # Run model (with reduced iterations for testing)
    fit = run_model(stan_data, chains=2, iter_warmup=200, iter_sampling=200)

    # Extract hourly predictions
    df_pred, y_pred_all_days = extract_hourly_predictions(fit, date_range, pred_hours)

    # Create visualizations
    output_dir = "docs/regularized_hourly_predictions"
    visualize_hourly_predictions(df_pred, df_weight, output_dir)

    # Analyze AR component
    ar_summary = analyze_ar_component(fit, output_dir)

    # Save predictions to CSV
    df_pred.to_csv(f"{output_dir}/hourly_predictions.csv", index=False)
    print(f"\n✓ Saved hourly predictions to {output_dir}/hourly_predictions.csv")

    # Create summary report
    with open(f"{output_dir}/analysis_summary.md", 'w') as f:
        f.write("# Regularized Model with Hourly Predictions Analysis\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n")
        f.write(f"- Days (D): {stan_data['D']}\n")
        f.write(f"- Weight observations: {stan_data['N_weight']}\n")
        f.write(f"- Fourier harmonics (K): {stan_data['K']}\n")
        f.write(f"- Prediction hours (H): {stan_data['H']}\n")
        f.write(f"- Hours predicted at: {list(pred_hours)}\n\n")

        f.write("## AR(1) Component Analysis\n")
        f.write(f"- ρ (autocorrelation): {ar_summary['rho_mean']:.3f} [{ar_summary['rho_90ci_lower']:.3f}, {ar_summary['rho_90ci_upper']:.3f}]\n")
        f.write(f"- σ_ε (innovation scale): {ar_summary['sigma_epsilon_mean']:.3f} ± {ar_summary['sigma_epsilon_std']:.3f}\n")
        f.write(f"- Regularizing prior: sigma_epsilon ~ normal(0, 0.1)\n")
        f.write(f"- Informative prior: rho ~ normal(0, 0.2)\n\n")

        f.write("## Generated Files\n")
        f.write("1. `daily_patterns.png` - Daily weight patterns for sample days\n")
        f.write("2. `noon_time_series.png` - Time series of noon predictions vs actual measurements\n")
        f.write("3. `heatmap_daily_patterns.png` - Heatmap of daily patterns over time\n")
        f.write("4. `ar_component_analysis.png` - Distribution of AR(1) parameters\n")
        f.write("5. `hourly_predictions.csv` - All hourly predictions\n")
        f.write("6. `analysis_summary.md` - This summary file\n")

    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")
    print(f"✓ Summary report: {output_dir}/analysis_summary.md")


if __name__ == "__main__":
    main()