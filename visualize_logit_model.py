#!/usr/bin/env python3
"""Comprehensive visualizations for the logit-parameterized training decay model."""

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


def fit_logit_model(df_weight, df_daily):
    """Fit the logit-parameterized decay model."""
    D = len(df_daily)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print("Compiling logit-parameterized decay model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_logit.stan")

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        adapt_delta=0.95,
        max_treedepth=12,
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


def create_comprehensive_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range):
    """Create comprehensive visualizations with all predictions and CIs."""
    output_dir = Path("output/logit_model_comprehensive")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Training intensity over time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Training Intensity and Patterns", fontsize=16)

    # Strength intensity over time
    ax = axes[0]
    ax.plot(date_range, df_daily['strength_intensity_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['strength_intensity_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity (standardized)")
    ax.set_title("Strength Training Intensity Over Time")
    ax.grid(True, alpha=0.3)

    # Training frequency histogram
    ax = axes[1]
    # Calculate weekly training frequency
    df_daily['week'] = (df_daily['date'] - df_daily['date'].min()).dt.days // 7
    weekly_training = df_daily.groupby('week')['strength_intensity_std'].apply(lambda x: (x > 0).sum())

    ax.bar(weekly_training.index, weekly_training.values, alpha=0.7, color='green')
    ax.set_xlabel("Week")
    ax.set_ylabel("Training Sessions per Week")
    ax.set_title("Weekly Training Frequency")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=weekly_training.mean(), color='red', linestyle='--', label=f'Mean: {weekly_training.mean():.1f} sessions/week')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_intensity_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Fitness state over time (with CIs)
    if 'fitness_stored' in samples:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Fitness State Evolution (with 90% Credible Intervals)", fontsize=16)

        # Fitness state
        ax = axes[0]
        fitness_samples = samples['fitness_stored']
        fitness_mean = np.mean(fitness_samples, axis=0)
        fitness_ci_lower = np.percentile(fitness_samples, 5, axis=0)
        fitness_ci_upper = np.percentile(fitness_samples, 95, axis=0)

        ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Mean fitness')
        ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='blue', label='90% CI')
        ax.set_ylabel("Fitness State (standardized)")
        ax.set_title("Fitness State Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Fitness accumulation vs training
        ax = axes[1]
        # Calculate cumulative training
        cumulative_training = df_daily['strength_intensity_std'].cumsum()

        ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Fitness (mean)')
        ax2 = ax.twinx()
        ax2.plot(date_range, cumulative_training, 'g-', alpha=0.7, linewidth=1, label='Cumulative training')

        ax.set_xlabel("Date")
        ax.set_ylabel("Fitness State", color='blue')
        ax2.set_ylabel("Cumulative Training Intensity", color='green')
        ax.set_title("Fitness Accumulation vs Cumulative Training")
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()
        plt.savefig(output_dir / "fitness_state_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 3: Weight decomposition
    if 'fitness_stored' in samples and 'gamma' in samples:
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle("Weight Decomposition (with 90% Credible Intervals)", fontsize=16)

        # Get weight observation times
        weight_dates = df_weight['timestamp']
        y_obs = df_weight['weight_lbs'].values

        # Calculate fitness contribution to weight
        gamma_samples = samples['gamma']
        fitness_samples = samples['fitness_stored']

        # For each weight observation, calculate fitness contribution
        fitness_contrib_samples = np.zeros((len(gamma_samples), len(df_weight)))
        pred_samples = np.zeros((len(gamma_samples), len(df_weight)))

        for i, row in df_weight.iterrows():
            day_idx = row['day_idx'] - 1  # Convert to 0-index
            for s in range(len(gamma_samples)):
                fitness_contrib = gamma_samples[s] * fitness_samples[s, day_idx] * weight_std
                fitness_contrib_samples[s, i] = fitness_contrib
                pred_samples[s, i] = weight_mean + fitness_contrib

        # Panel 1: Observed vs predicted weight
        ax = axes[0]
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

        ax.set_ylabel("Weight (lbs)")
        ax.set_title("Observed vs Predicted Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation text
        correlation = np.corrcoef(y_obs, pred_mean)[0, 1]
        rmse = np.sqrt(np.mean((y_obs - pred_mean) ** 2))
        ax.text(0.02, 0.98, f"Correlation: {correlation:.3f}\nRMSE: {rmse:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 2: Fitness contribution to weight
        ax = axes[1]
        fitness_contrib_mean = np.mean(fitness_contrib_samples, axis=0)
        fitness_contrib_ci_lower = np.percentile(fitness_contrib_samples, 5, axis=0)
        fitness_contrib_ci_upper = np.percentile(fitness_contrib_samples, 95, axis=0)

        ax.scatter(weight_dates, fitness_contrib_mean, s=40, alpha=0.7, color='green')

        # Calculate error bars
        yerr_lower = np.abs(fitness_contrib_mean - fitness_contrib_ci_lower)
        yerr_upper = np.abs(fitness_contrib_ci_upper - fitness_contrib_mean)
        ax.errorbar(weight_dates, fitness_contrib_mean, yerr=[yerr_lower, yerr_upper],
                   fmt='none', alpha=0.3, color='green', capsize=3)

        ax.set_ylabel("Weight due to fitness (lbs)")
        ax.set_title("Fitness Contribution to Weight")
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        mean_effect = np.mean(fitness_contrib_mean)
        max_effect = np.max(fitness_contrib_mean)
        ax.text(0.02, 0.98, f"Mean: {mean_effect:.2f} lbs\nMax: {max_effect:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 3: Residuals (observed - predicted)
        ax = axes[2]
        residuals = y_obs - pred_mean

        ax.scatter(weight_dates, residuals, s=40, alpha=0.7, color='purple')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Add 95% prediction interval
        residual_std = np.std(residuals)
        ax.axhline(y=2*residual_std, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=-2*residual_std, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between([weight_dates.min(), weight_dates.max()],
                       -2*residual_std, 2*residual_std, alpha=0.1, color='gray')

        ax.set_xlabel("Date")
        ax.set_ylabel("Residual (obs - pred, lbs)")
        ax.set_title("Model Residuals (±2σ bands)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "weight_decomposition.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 4: Parameter posterior distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Parameter Posterior Distributions (with 90% Credible Intervals)", fontsize=16)

    param_names = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
    param_titles = [
        'Base retention (no training)',
        'Additional retention when trained',
        'Gain per intensity',
        'Weight effect',
        'Weight noise std'
    ]

    for idx, (param_name, param_title) in enumerate(zip(param_names, param_titles)):
        ax = axes[idx // 3, idx % 3]
        if param_name in samples:
            param_samples = samples[param_name]

            # Histogram
            ax.hist(param_samples, bins=30, density=True, alpha=0.7, color='steelblue')

            # Mean line
            mean_val = np.mean(param_samples)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')

            # 90% CI
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

    # Hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Model interpretation and predictions
    if 'alpha_d' in samples and 'alpha_m' in samples and 'beta' in samples:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Interpretation and Predictions", fontsize=16)

        # Panel 1: Decay rates
        ax = axes[0, 0]
        alpha_d_samples = samples['alpha_d']
        alpha_m_samples = samples['alpha_m']

        # Calculate total retention when trained
        alpha_total_samples = alpha_d_samples + (1 - alpha_d_samples) * alpha_m_samples

        # Plot distributions
        ax.hist(alpha_d_samples, bins=30, density=True, alpha=0.5, color='blue', label='No training')
        ax.hist(alpha_total_samples, bins=30, density=True, alpha=0.5, color='green', label='With training')

        ax.set_xlabel("Daily retention rate")
        ax.set_ylabel("Density")
        ax.set_title("Fitness Retention: With vs Without Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Half-life calculation
        ax = axes[0, 1]

        def half_life(retention_rate):
            """Calculate half-life in days."""
            if retention_rate <= 0:
                return 0
            if retention_rate >= 1:
                return float('inf')
            return np.log(0.5) / np.log(retention_rate)

        half_life_no_train = np.array([half_life(r) for r in alpha_d_samples])
        half_life_with_train = np.array([half_life(r) for r in alpha_total_samples])

        ax.hist(half_life_no_train, bins=30, density=True, alpha=0.5, color='blue', label='No training')
        ax.hist(half_life_with_train, bins=30, density=True, alpha=0.5, color='green', label='With training')

        ax.set_xlabel("Half-life (days)")
        ax.set_ylabel("Density")
        ax.set_title("Fitness Half-life Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: Steady-state fitness prediction
        ax = axes[1, 0]
        beta_samples = samples['beta']

        # Calculate steady-state fitness for different training intensities
        training_intensities = np.linspace(0.5, 5, 20)
        steady_state_samples = np.zeros((len(beta_samples), len(training_intensities)))

        for i, intensity in enumerate(training_intensities):
            for s in range(len(beta_samples)):
                steady_state = (beta_samples[s] * intensity) / (1 - alpha_total_samples[s])
                steady_state_samples[s, i] = steady_state

        steady_state_mean = np.mean(steady_state_samples, axis=0)
        steady_state_ci_lower = np.percentile(steady_state_samples, 5, axis=0)
        steady_state_ci_upper = np.percentile(steady_state_samples, 95, axis=0)

        ax.plot(training_intensities, steady_state_mean, 'b-', linewidth=2)
        ax.fill_between(training_intensities, steady_state_ci_lower, steady_state_ci_upper, alpha=0.3, color='blue')

        ax.set_xlabel("Training intensity (standardized)")
        ax.set_ylabel("Steady-state fitness")
        ax.set_title("Steady-state Fitness vs Training Intensity")
        ax.grid(True, alpha=0.3)

        # Panel 4: Weight effect prediction
        ax = axes[1, 1]
        if 'gamma' in samples:
            gamma_samples = samples['gamma']

            # Calculate steady-state weight effect
            weight_effect_samples = np.zeros((len(gamma_samples), len(training_intensities)))

            for i, intensity in enumerate(training_intensities):
                for s in range(len(gamma_samples)):
                    fitness = (beta_samples[s] * intensity) / (1 - alpha_total_samples[s])
                    weight_effect = fitness * gamma_samples[s] * weight_std
                    weight_effect_samples[s, i] = weight_effect

            weight_effect_mean = np.mean(weight_effect_samples, axis=0)
            weight_effect_ci_lower = np.percentile(weight_effect_samples, 5, axis=0)
            weight_effect_ci_upper = np.percentile(weight_effect_samples, 95, axis=0)

            ax.plot(training_intensities, weight_effect_mean, 'g-', linewidth=2)
            ax.fill_between(training_intensities, weight_effect_ci_lower, weight_effect_ci_upper, alpha=0.3, color='green')

            ax.set_xlabel("Training intensity (standardized)")
            ax.set_ylabel("Steady-state weight effect (lbs)")
            ax.set_title("Steady-state Weight Effect vs Training Intensity")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "model_interpretation.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nAll visualizations saved to {output_dir}/")

    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL SUMMARY")
    print("="*80)

    if 'alpha_d' in samples and 'alpha_m' in samples:
        alpha_d_mean = np.mean(samples['alpha_d'])
        alpha_m_mean = np.mean(samples['alpha_m'])
        alpha_total_mean = alpha_d_mean + (1 - alpha_d_mean) * alpha_m_mean

        print(f"\n1. FITNESS DECAY CHARACTERISTICS:")
        print(f"   Without training: {alpha_d_mean:.1%} retained per day ({1-alpha_d_mean:.1%} decay)")
        print(f"   With training:    {alpha_total_mean:.1%} retained per day ({1-alpha_total_mean:.1%} decay)")
        print(f"   Training effect:  adds {alpha_m_mean:.1%} of remaining decay")

        # Half-life
        half_life_no_train = np.log(0.5) / np.log(alpha_d_mean)
        half_life_with_train = np.log(0.5) / np.log(alpha_total_mean)
        print(f"   Half-life without training: {half_life_no_train:.1f} days")
        print(f"   Half-life with training:    {half_life_with_train:.1f} days")

    if 'beta' in samples:
        beta_mean = np.mean(samples['beta'])
        print(f"\n2. FITNESS GAIN:")
        print(f"   Gain per unit intensity: {beta_mean:.3f} fitness units")

    if 'gamma' in samples:
        gamma_mean = np.mean(samples['gamma'])
        print(f"\n3. WEIGHT EFFECT:")
        print(f"   Weight effect: {gamma_mean:.3f} standardized units per fitness unit")
        print(f"   In lbs: {gamma_mean * weight_std:.2f} lbs per fitness unit")

        # Steady-state calculation
        if 'alpha_d' in samples and 'alpha_m' in samples and 'beta' in samples:
            avg_intensity = np.mean(df_daily['strength_intensity_std'][df_daily['strength_intensity_std'] > 0])
            avg_training_freq = (df_daily['strength_intensity_std'] > 0).mean()

            # Effective retention rate given training frequency
            effective_retention = alpha_d_mean * (1 - avg_training_freq) + alpha_total_mean * avg_training_freq

            steady_state_fitness = (beta_mean * avg_intensity * avg_training_freq) / (1 - effective_retention)
            steady_state_weight = steady_state_fitness * gamma_mean * weight_std

            print(f"\n4. STEADY-STATE PREDICTIONS (current training pattern):")
            print(f"   Average training intensity: {avg_intensity:.2f}")
            print(f"   Training frequency: {avg_training_freq:.1%} of days")
            print(f"   Effective retention rate: {effective_retention:.1%}")
            print(f"   Steady-state fitness: {steady_state_fitness:.2f} units")
            print(f"   Steady-state weight effect: {steady_state_weight:.2f} lbs")

    print(f"\n5. MODEL FIT:")
    if 'fitness_stored' in samples and 'gamma' in samples:
        # Calculate predictions for all weight observations
        pred_samples_all = np.zeros((len(gamma_samples), len(df_weight)))
        for i, row in df_weight.iterrows():
            day_idx = row['day_idx'] - 1
            for s in range(len(gamma_samples)):
                pred_samples_all[s, i] = weight_mean + gamma_samples[s] * fitness_samples[s, day_idx] * weight_std

        pred_mean_all = np.mean(pred_samples_all, axis=0)
        y_obs_all = df_weight['weight_lbs'].values

        correlation = np.corrcoef(y_obs_all, pred_mean_all)[0, 1]
        rmse = np.sqrt(np.mean((y_obs_all - pred_mean_all) ** 2))
        mae = np.mean(np.abs(y_obs_all - pred_mean_all))

        print(f"   Correlation (obs vs pred): {correlation:.3f}")
        print(f"   RMSE: {rmse:.2f} lbs")
        print(f"   MAE: {mae:.2f} lbs")
        print(f"   Weight std: {weight_std:.2f} lbs")
        print(f"   Model explains {correlation**2:.1%} of weight variance")

    print("\n" + "="*80)


def main():
    """Main function to create comprehensive visualizations."""
    print("Creating comprehensive visualizations for logit-parameterized training decay model...")

    # Load and prepare data
    df_weight, df_daily, weight_mean, weight_std, date_range = load_and_prepare_data()

    print(f"Data loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Fit logit model
    fit = fit_logit_model(df_weight, df_daily)

    # Extract posterior samples
    samples = extract_posterior_samples(fit)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range)

    print("\nDone! All visualizations created successfully.")


if __name__ == "__main__":
    main()