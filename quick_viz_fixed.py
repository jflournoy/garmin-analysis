#!/usr/bin/env python3
"""Quick visualization of training decay model predictions - simplified version."""

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


def fit_simple_model(df_weight, df_daily):
    """Fit the simple training decay model."""
    D = len(df_daily)

    # Prepare Stan data for simple model
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print("Compiling simple training decay model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_simple.stan")

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        adapt_delta=0.9,
        max_treedepth=10,
        show_progress=True,
        seed=12345,
    )

    return fit


def extract_posterior_samples(fit):
    """Extract posterior samples from cmdstanpy fit."""
    # Get all draws as pandas DataFrame
    draws_df = fit.draws_pd()

    # Extract parameter samples
    samples = {}

    # Core parameters
    param_names = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
    for param in param_names:
        if param in draws_df.columns:
            samples[param] = draws_df[param].values

    # Extract fitness states if available
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

    return samples


def create_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range):
    """Create comprehensive visualizations."""
    output_dir = Path("output/quick_visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Intensity and fitness over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("Training Intensity and Fitness Over Time", fontsize=16)

    # Strength intensity
    ax = axes[0]
    ax.plot(date_range, df_daily['strength_intensity_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['strength_intensity_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity (standardized)")
    ax.set_title("Strength Training Intensity")
    ax.grid(True, alpha=0.3)

    # Fitness state
    ax = axes[1]
    if 'fitness_stored' in samples:
        fitness_samples = samples['fitness_stored']
        fitness_mean = np.mean(fitness_samples, axis=0)
        fitness_ci_lower = np.percentile(fitness_samples, 5, axis=0)
        fitness_ci_upper = np.percentile(fitness_samples, 95, axis=0)

        ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='blue', label='90% CI')
        ax.set_ylabel("Fitness State (standardized)")
        ax.set_title("Fitness State (with 90% Credible Intervals)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Training days indicator
    ax = axes[2]
    trained_days = df_daily['strength_intensity_std'] > 0
    ax.plot(date_range, trained_days.astype(float), 'g-', alpha=0.7, linewidth=1, drawstyle='steps-post')
    ax.set_ylabel("Trained (1=yes, 0=no)")
    ax.set_title("Training Days")
    ax.set_xlabel("Date")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "intensity_fitness_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Parameter Posterior Distributions", fontsize=16)

    param_names = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
    param_titles = [
        'Decay (no training)',
        'Training reduces decay',
        'Gain per intensity',
        'Weight effect',
        'Weight noise std'
    ]

    for idx, (param_name, param_title) in enumerate(zip(param_names, param_titles)):
        ax = axes[idx // 3, idx % 3]
        if param_name in samples:
            param_samples = samples[param_name]

            ax.hist(param_samples, bins=30, density=True, alpha=0.7, color='steelblue')
            ax.axvline(np.mean(param_samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(param_samples):.3f}')

            # Add 90% CI
            ci_lower = np.percentile(param_samples, 5)
            ci_upper = np.percentile(param_samples, 95)
            ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'90% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

            ax.set_xlabel(param_name)
            ax.set_ylabel("Density")
            ax.set_title(param_title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Weight effect of fitness over time
    if 'fitness_stored' in samples and 'gamma' in samples:
        fig, ax = plt.subplots(figsize=(15, 8))

        # Calculate weight effect = fitness * gamma * weight_std
        gamma_samples = samples['gamma']
        fitness_samples = samples['fitness_stored']

        # For each sample, calculate weight effect
        weight_effect_samples = np.zeros((len(gamma_samples), len(date_range)))
        for i in range(len(gamma_samples)):
            weight_effect_samples[i, :] = fitness_samples[i, :] * gamma_samples[i] * weight_std

        weight_effect_mean = np.mean(weight_effect_samples, axis=0)
        weight_effect_ci_lower = np.percentile(weight_effect_samples, 5, axis=0)
        weight_effect_ci_upper = np.percentile(weight_effect_samples, 95, axis=0)

        ax.plot(date_range, weight_effect_mean, 'g-', linewidth=2, label='Mean weight effect')
        ax.fill_between(date_range, weight_effect_ci_lower, weight_effect_ci_upper, alpha=0.3, color='green', label='90% CI')

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight effect (lbs)")
        ax.set_title("Estimated Weight Effect from Strength Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        max_effect = np.max(weight_effect_mean)
        avg_effect = np.mean(weight_effect_mean)
        min_effect = np.min(weight_effect_mean)
        ax.text(0.02, 0.98, f"Max: {max_effect:.2f} lbs\nAvg: {avg_effect:.2f} lbs\nMin: {min_effect:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "weight_effect_over_time.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 4: Model predictions vs observed weight
    fig, ax = plt.subplots(figsize=(15, 8))

    # Calculate predicted weight = weight_mean + gamma * fitness * weight_std
    if 'fitness_stored' in samples and 'gamma' in samples:
        weight_dates = df_weight['timestamp']
        y_obs = df_weight['weight_lbs'].values

        # For each weight observation, calculate predictions
        pred_samples = np.zeros((len(gamma_samples), len(df_weight)))

        for i, row in df_weight.iterrows():
            day_idx = row['day_idx'] - 1  # Convert to 0-index
            for s in range(len(gamma_samples)):
                pred_samples[s, i] = weight_mean + gamma_samples[s] * fitness_samples[s, day_idx] * weight_std

        pred_mean = np.mean(pred_samples, axis=0)
        pred_ci_lower = np.percentile(pred_samples, 5, axis=0)
        pred_ci_upper = np.percentile(pred_samples, 95, axis=0)

        ax.scatter(weight_dates, y_obs, s=40, alpha=0.7, color='blue', label='Observed')
        ax.scatter(weight_dates, pred_mean, s=40, alpha=0.7, color='red', label='Predicted (mean)')
        # Calculate error bars (ensure non-negative)
        yerr_lower = np.abs(pred_mean - pred_ci_lower)
        yerr_upper = np.abs(pred_ci_upper - pred_mean)
        ax.errorbar(weight_dates, pred_mean, yerr=[yerr_lower, yerr_upper],
                   fmt='none', alpha=0.3, color='red', capsize=3, label='90% CI')

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title("Observed vs Predicted Weight (with 90% Credible Intervals)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation text
        correlation = np.corrcoef(y_obs, pred_mean)[0, 1]
        rmse = np.sqrt(np.mean((y_obs - pred_mean) ** 2))
        ax.text(0.02, 0.98, f"Correlation: {correlation:.3f}\nRMSE: {rmse:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "weight_predictions.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"All visualizations saved to {output_dir}/")

    # Print parameter summary
    print("\nParameter Summary (mean ± 90% CI):")
    for param_name, param_title in zip(param_names, param_titles):
        if param_name in samples:
            param_samples = samples[param_name]
            mean_val = np.mean(param_samples)
            ci_lower = np.percentile(param_samples, 5)
            ci_upper = np.percentile(param_samples, 95)
            print(f"  {param_name}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] - {param_title}")

    # Special interpretation for alpha parameters
    if 'alpha_d' in samples and 'alpha_m' in samples:
        alpha_d_mean = np.mean(samples['alpha_d'])
        alpha_m_mean = np.mean(samples['alpha_m'])
        print(f"\nInterpretation:")
        print(f"  Without training: fitness decays to {alpha_d_mean:.3f} of previous day")
        print(f"  With training: fitness decays to {alpha_d_mean + alpha_m_mean:.3f} of previous day")
        print(f"  Training reduces decay by: {alpha_m_mean:.3f} ({alpha_m_mean/alpha_d_mean*100:.1f}% of base decay)")

    if 'gamma' in samples:
        gamma_mean = np.mean(samples['gamma'])
        print(f"  Weight effect: {gamma_mean:.3f} standardized units per fitness unit")
        print(f"  In lbs: {gamma_mean * weight_std:.2f} lbs per fitness unit")


def main():
    """Main function to create quick visualizations."""
    print("Creating quick visualizations for training decay model...")

    # Load and prepare data
    df_weight, df_daily, weight_mean, weight_std, date_range = load_and_prepare_data()

    print(f"Data loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Fit simple model
    fit = fit_simple_model(df_weight, df_daily)

    # Extract posterior samples
    samples = extract_posterior_samples(fit)

    # Create visualizations
    create_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range)

    print("Done!")


if __name__ == "__main__":
    main()