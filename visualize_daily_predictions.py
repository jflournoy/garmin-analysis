#!/usr/bin/env python3
"""Visualize DAILY weight predictions (not just measurement days)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_prepare_data():
    """Load and prepare data for visualization."""
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
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
    df_act = df_intensity[['date', 'strength_training']].copy()
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')
    df_daily['strength_training'] = df_daily['strength_training'].fillna(0.0)

    # Standardize intensity (shift so min=0)
    min_val = df_daily['strength_training'].min()
    std = df_daily['strength_training'].std()
    if std > 0:
        df_daily['strength_intensity_std'] = (df_daily['strength_training'] - min_val) / std
    else:
        df_daily['strength_intensity_std'] = df_daily['strength_training'] - min_val

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    return df_weight, df_daily, weight_mean, weight_std, date_range


def fit_intercept_model(df_weight, df_daily):
    """Fit the intercept model."""
    D = len(df_daily)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print("Compiling intercept model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_intercept.stan")

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        adapt_delta=0.95,
        max_treedepth=12,
        show_progress=True,
        seed=12345,
    )

    return fit


def extract_posterior_samples(fit, D):
    """Extract posterior samples from cmdstanpy fit."""
    # Get all draws as pandas DataFrame
    draws_df = fit.draws_pd()

    # Extract parameter samples
    samples = {}

    # Core parameters
    param_names = ['alpha_d', 'alpha_m', 'beta', 'weight_intercept', 'gamma', 'sigma_w']
    for param in param_names:
        if param in draws_df.columns:
            samples[param] = draws_df[param].values

    # Extract fitness states for ALL days
    fitness_cols = [col for col in draws_df.columns if col.startswith('fitness_stored[')]
    if fitness_cols:
        # Parse indices from column names like 'fitness_stored[1]'
        fitness_values = {}
        for col in fitness_cols:
            # Extract index from column name
            idx_str = col.split('[')[1].split(']')[0]
            idx = int(idx_str) - 1  # Convert to 0-index
            fitness_values[idx] = draws_df[col].values

        # Sort by index and create array
        sorted_indices = sorted(fitness_values.keys())
        fitness_array = np.column_stack([fitness_values[idx] for idx in sorted_indices])
        samples['fitness_stored'] = fitness_array

        # Verify we have all days
        if len(sorted_indices) == D:
            print(f"✓ Extracted fitness states for all {D} days")
        else:
            print(f"⚠ Only extracted {len(sorted_indices)}/{D} days of fitness states")

    return samples


def create_daily_prediction_plots(samples, df_daily, df_weight, weight_mean, weight_std, date_range):
    """Create plots showing DAILY predictions (not just measurement days)."""
    output_dir = Path("output/daily_predictions")
    output_dir.mkdir(exist_ok=True, parents=True)

    D = len(date_range)

    # Plot 1: DAILY weight predictions with measurement points
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle("DAILY Weight Predictions (All 924 Days, Not Just 147 Measurement Days)", fontsize=16)

    # Panel 1: Training intensity
    ax = axes[0]
    ax.plot(date_range, df_daily['strength_intensity_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['strength_intensity_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity")
    ax.set_title("Daily Training Intensity")
    ax.grid(True, alpha=0.3)

    # Panel 2: DAILY fitness predictions
    ax = axes[1]
    if 'fitness_stored' in samples:
        fitness_samples = samples['fitness_stored']
        fitness_mean = np.mean(fitness_samples, axis=0)
        fitness_ci_lower = np.percentile(fitness_samples, 5, axis=0)
        fitness_ci_upper = np.percentile(fitness_samples, 95, axis=0)

        # Plot DAILY fitness with CI
        ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Mean fitness')
        ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='blue', label='90% CI')

        ax.set_ylabel("Fitness State")
        ax.set_title(f"Daily Fitness Predictions (All {D} Days)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: DAILY weight predictions
    ax = axes[2]
    if 'fitness_stored' in samples and 'weight_intercept' in samples and 'gamma' in samples:
        intercept_samples = samples['weight_intercept']
        gamma_samples = samples['gamma']
        fitness_samples = samples['fitness_stored']

        # Calculate DAILY weight predictions for ALL days
        daily_weight_samples = np.zeros((len(gamma_samples), D))

        for day in range(D):
            for s in range(len(gamma_samples)):
                # Weight = baseline + gamma * fitness
                weight_std_units = intercept_samples[s] + gamma_samples[s] * fitness_samples[s, day]
                daily_weight_samples[s, day] = weight_mean + weight_std_units * weight_std

        daily_weight_mean = np.mean(daily_weight_samples, axis=0)
        daily_weight_ci_lower = np.percentile(daily_weight_samples, 5, axis=0)
        daily_weight_ci_upper = np.percentile(daily_weight_samples, 95, axis=0)

        # Plot DAILY weight predictions with CI
        ax.plot(date_range, daily_weight_mean, 'g-', linewidth=2, label='Predicted weight (mean)')
        ax.fill_between(date_range, daily_weight_ci_lower, daily_weight_ci_upper,
                       alpha=0.3, color='green', label='90% CI')

        # Overlay ACTUAL weight measurements
        weight_dates = df_weight['timestamp']
        y_obs = df_weight['weight_lbs'].values
        ax.scatter(weight_dates, y_obs, s=50, alpha=0.8, color='red',
                  label=f'Actual measurements ({len(df_weight)} points)', zorder=5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title(f"Daily Weight Predictions vs Actual Measurements")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics
        ax.text(0.02, 0.98, f"Daily predictions: {D} days\nMeasurements: {len(df_weight)} points\nCorrelation: {np.corrcoef(y_obs, daily_weight_mean[df_weight['day_idx'].values-1])[0,1]:.3f}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "daily_predictions_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Zoomed view of predictions vs measurements
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Zoomed View: Daily Predictions vs Actual Measurements", fontsize=16)

    # Panel 1: 90-day zoom
    ax = axes[0]
    if 'fitness_stored' in samples and 'weight_intercept' in samples and 'gamma' in samples:
        # Zoom to last 90 days
        zoom_days = 90
        zoom_start = max(0, D - zoom_days)
        zoom_dates = date_range[zoom_start:]
        zoom_weight_mean = daily_weight_mean[zoom_start:]
        zoom_weight_ci_lower = daily_weight_ci_lower[zoom_start:]
        zoom_weight_ci_upper = daily_weight_ci_upper[zoom_start:]

        # Filter measurements in zoom period
        zoom_mask = (df_weight['timestamp'] >= zoom_dates[0]) & (df_weight['timestamp'] <= zoom_dates[-1])
        zoom_measurements = df_weight[zoom_mask]

        ax.plot(zoom_dates, zoom_weight_mean, 'g-', linewidth=2, label='Predicted (mean)')
        ax.fill_between(zoom_dates, zoom_weight_ci_lower, zoom_weight_ci_upper,
                       alpha=0.3, color='green', label='90% CI')

        if len(zoom_measurements) > 0:
            ax.scatter(zoom_measurements['timestamp'], zoom_measurements['weight_lbs'],
                      s=60, alpha=0.9, color='red', label='Measurements', zorder=5)

        ax.set_ylabel("Weight (lbs)")
        ax.set_title(f"Last {zoom_days} Days (Zoomed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 2: Prediction error analysis
    ax = axes[1]
    if 'fitness_stored' in samples and 'weight_intercept' in samples and 'gamma' in samples:
        # Get predictions at measurement times
        pred_at_measurements = daily_weight_mean[df_weight['day_idx'].values - 1]
        y_obs = df_weight['weight_lbs'].values

        # Calculate errors
        errors = y_obs - pred_at_measurements
        abs_errors = np.abs(errors)

        # Scatter plot of predictions vs observations
        ax.scatter(y_obs, pred_at_measurements, s=40, alpha=0.7, color='blue')

        # Perfect prediction line
        min_val = min(y_obs.min(), pred_at_measurements.min())
        max_val = max(y_obs.max(), pred_at_measurements.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')

        # Statistics
        correlation = np.corrcoef(y_obs, pred_at_measurements)[0, 1]
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(abs_errors)

        ax.set_xlabel("Observed Weight (lbs)")
        ax.set_ylabel("Predicted Weight (lbs)")
        ax.set_title(f"Prediction Accuracy at Measurement Times\nCorrelation: {correlation:.3f}, RMSE: {rmse:.2f} lbs, MAE: {mae:.2f} lbs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add error distribution inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
        ax_inset.hist(errors, bins=20, density=True, alpha=0.7, color='purple')
        ax_inset.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax_inset.set_xlabel("Error (lbs)")
        ax_inset.set_ylabel("Density")
        ax_inset.set_title("Error Distribution")
        ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "zoomed_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Weight decomposition over time (DAILY)
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle("Daily Weight Decomposition (Baseline + Fitness Contribution)", fontsize=16)

    if 'fitness_stored' in samples and 'weight_intercept' in samples and 'gamma' in samples:
        # Calculate components DAILY
        baseline_daily = np.zeros((len(gamma_samples), D))
        fitness_contrib_daily = np.zeros((len(gamma_samples), D))
        total_daily = np.zeros((len(gamma_samples), D))

        for day in range(D):
            for s in range(len(gamma_samples)):
                baseline = weight_mean + intercept_samples[s] * weight_std
                fitness_contrib = gamma_samples[s] * fitness_samples[s, day] * weight_std

                baseline_daily[s, day] = baseline
                fitness_contrib_daily[s, day] = fitness_contrib
                total_daily[s, day] = baseline + fitness_contrib

        # Panel 1: Baseline weight (constant)
        ax = axes[0]
        baseline_mean = np.mean(baseline_daily, axis=0)
        baseline_ci_lower = np.percentile(baseline_daily, 5, axis=0)
        baseline_ci_upper = np.percentile(baseline_daily, 95, axis=0)

        ax.plot(date_range, baseline_mean, 'purple', linewidth=2, label='Baseline weight')
        ax.fill_between(date_range, baseline_ci_lower, baseline_ci_upper,
                       alpha=0.3, color='purple', label='90% CI')

        ax.set_ylabel("Weight (lbs)")
        ax.set_title(f"Daily Baseline Weight (Intercept): {np.mean(baseline_mean):.1f} ± {np.std(baseline_mean):.1f} lbs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Fitness contribution DAILY
        ax = axes[1]
        fitness_contrib_mean = np.mean(fitness_contrib_daily, axis=0)
        fitness_contrib_ci_lower = np.percentile(fitness_contrib_daily, 5, axis=0)
        fitness_contrib_ci_upper = np.percentile(fitness_contrib_daily, 95, axis=0)

        ax.plot(date_range, fitness_contrib_mean, 'green', linewidth=2, label='Fitness contribution')
        ax.fill_between(date_range, fitness_contrib_ci_lower, fitness_contrib_ci_upper,
                       alpha=0.3, color='green', label='90% CI')

        ax.set_ylabel("Weight (lbs)")
        ax.set_title(f"Daily Fitness Contribution: Mean = {np.mean(fitness_contrib_mean):.1f} lbs, Max = {np.max(fitness_contrib_mean):.1f} lbs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Total predicted weight
        ax = axes[2]
        total_mean = np.mean(total_daily, axis=0)
        total_ci_lower = np.percentile(total_daily, 5, axis=0)
        total_ci_upper = np.percentile(total_daily, 95, axis=0)

        ax.plot(date_range, total_mean, 'blue', linewidth=2, label='Total predicted weight')
        ax.fill_between(date_range, total_ci_lower, total_ci_upper,
                       alpha=0.3, color='blue', label='90% CI')

        # Overlay measurements
        ax.scatter(df_weight['timestamp'], df_weight['weight_lbs'],
                  s=40, alpha=0.7, color='red', label='Measurements', zorder=5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title("Total Daily Weight Prediction = Baseline + Fitness Contribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "daily_weight_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll DAILY prediction plots saved to {output_dir}/")

    # Print summary statistics
    print("\n" + "="*80)
    print("DAILY PREDICTION SUMMARY")
    print("="*80)

    if 'fitness_stored' in samples and 'weight_intercept' in samples and 'gamma' in samples:
        # Daily statistics
        daily_weight_mean_all = np.mean(daily_weight_samples, axis=0)

        print(f"\nDaily Weight Predictions ({D} days):")
        print(f"  Mean predicted weight: {np.mean(daily_weight_mean_all):.1f} lbs")
        print(f"  Std of predictions: {np.std(daily_weight_mean_all):.1f} lbs")
        print(f"  Range: {np.min(daily_weight_mean_all):.1f} to {np.max(daily_weight_mean_all):.1f} lbs")

        # Fitness contribution statistics
        fitness_contrib_mean_all = np.mean(fitness_contrib_daily, axis=0)
        print(f"\nDaily Fitness Contribution:")
        print(f"  Mean contribution: {np.mean(fitness_contrib_mean_all):.1f} lbs")
        print(f"  Max contribution: {np.max(fitness_contrib_mean_all):.1f} lbs")
        print(f"  Min contribution: {np.min(fitness_contrib_mean_all):.1f} lbs")

        # Model fit at measurement times
        pred_at_meas = daily_weight_mean[df_weight['day_idx'].values - 1]
        y_obs = df_weight['weight_lbs'].values
        errors = y_obs - pred_at_meas

        print(f"\nModel Fit at {len(df_weight)} Measurement Times:")
        print(f"  Correlation: {np.corrcoef(y_obs, pred_at_meas)[0,1]:.3f}")
        print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f} lbs")
        print(f"  MAE: {np.mean(np.abs(errors)):.2f} lbs")
        print(f"  Mean error: {np.mean(errors):.2f} lbs")
        print(f"  Std of errors: {np.std(errors):.2f} lbs")

        # Compare with weight variability
        print(f"\nWeight Statistics:")
        print(f"  Observed weight mean: {weight_mean:.1f} lbs")
        print(f"  Observed weight std: {weight_std:.2f} lbs")
        print(f"  Model explains {np.corrcoef(y_obs, pred_at_meas)[0,1]**2:.1%} of variance")

    print("\n" + "="*80)


def main():
    """Main function to create daily prediction visualizations."""
    print("Creating DAILY weight prediction visualizations (all 924 days, not just 147 measurement days)...")

    # Load and prepare data
    df_weight, df_daily, weight_mean, weight_std, date_range = load_and_prepare_data()

    D = len(date_range)
    print(f"Data loaded: {D} days total, {len(df_weight)} weight measurements")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")
    print(f"Date range: {date_range[0].date()} to {date_range[-1].date()}")

    # Fit intercept model
    fit = fit_intercept_model(df_weight, df_daily)

    # Extract posterior samples
    samples = extract_posterior_samples(fit, D)

    # Create daily prediction plots
    create_daily_prediction_plots(samples, df_daily, df_weight, weight_mean, weight_std, date_range)

    print("\nDone! Created visualizations showing DAILY predictions for all days.")


if __name__ == "__main__":
    main()