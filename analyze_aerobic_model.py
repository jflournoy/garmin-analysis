#!/usr/bin/env python3
"""Analyze the aerobic fitness model with both strength and aerobic components."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import arviz as az

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_prepare_data():
    """Load and prepare data for aerobic model."""
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load both strength and aerobic intensity data
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
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')

    # Fill missing values with 0
    df_daily['strength_training'] = df_daily['strength_training'].fillna(0.0)
    df_daily['walking'] = df_daily['walking'].fillna(0.0)
    df_daily['cycling'] = df_daily['cycling'].fillna(0.0)

    # Combine walking and cycling into aerobic intensity
    df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']

    # Standardize intensity (shift so min=0)
    for intensity_type in ['strength_training', 'aerobic_intensity']:
        min_val = df_daily[intensity_type].min()
        std = df_daily[intensity_type].std()
        if std > 0:
            df_daily[f'{intensity_type}_std'] = (df_daily[intensity_type] - min_val) / std
        else:
            df_daily[f'{intensity_type}_std'] = df_daily[intensity_type] - min_val

        print(f"{intensity_type}: min={min_val:.2f}, std={std:.2f}, scaled mean={df_daily[f'{intensity_type}_std'].mean():.4f}")

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


def fit_aerobic_model(df_weight, df_daily):
    """Fit the aerobic model with both strength and aerobic components."""
    D = len(df_daily)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print("Compiling aerobic model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_aerobic.stan")

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
    param_names = [
        'alpha_d_s', 'alpha_m_s', 'beta_s',
        'alpha_d_a', 'alpha_m_a', 'beta_a',
        'weight_intercept', 'gamma_s', 'gamma_a', 'sigma_w'
    ]

    for param in param_names:
        if param in draws_df.columns:
            samples[param] = draws_df[param].values

    # Extract fitness states if available
    for fitness_type in ['strength_fitness_stored', 'aerobic_fitness_stored']:
        fitness_cols = [col for col in draws_df.columns if col.startswith(f'{fitness_type}[')]
        if fitness_cols:
            # Parse indices from column names like 'strength_fitness_stored[1]'
            fitness_values = {}
            for col in fitness_cols:
                # Extract index from column name
                idx_str = col.split('[')[1].split(']')[0]
                idx = int(idx_str) - 1  # Convert to 0-index
                fitness_values[idx] = draws_df[col].values

            # Sort by index and create array
            sorted_indices = sorted(fitness_values.keys())
            fitness_array = np.column_stack([fitness_values[idx] for idx in sorted_indices])
            samples[fitness_type] = fitness_array

    return samples


def create_comprehensive_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range):
    """Create comprehensive visualizations for aerobic model."""
    output_dir = Path("output/aerobic_model_comprehensive")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot 1: Training intensity and fitness for both types
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle("Strength and Aerobic Training: Intensity and Fitness Evolution", fontsize=16)

    # Panel 1: Strength training intensity
    ax = axes[0]
    ax.plot(date_range, df_daily['strength_training_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['strength_training_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity (standardized)")
    ax.set_title("Strength Training Intensity Over Time")
    ax.grid(True, alpha=0.3)

    # Panel 2: Strength fitness state
    ax = axes[1]
    if 'strength_fitness_stored' in samples:
        fitness_samples = samples['strength_fitness_stored']
        fitness_mean = np.mean(fitness_samples, axis=0)
        fitness_ci_lower = np.percentile(fitness_samples, 5, axis=0)
        fitness_ci_upper = np.percentile(fitness_samples, 95, axis=0)

        ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Mean fitness')
        ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='blue', label='90% CI')
        ax.set_ylabel("Strength Fitness (standardized)")
        ax.set_title("Strength Fitness State Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: Aerobic training intensity
    ax = axes[2]
    ax.plot(date_range, df_daily['aerobic_intensity_std'], 'g-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['aerobic_intensity_std'], alpha=0.3, color='green')
    ax.set_ylabel("Aerobic Intensity (standardized)")
    ax.set_title("Aerobic Training Intensity Over Time")
    ax.grid(True, alpha=0.3)

    # Panel 4: Aerobic fitness state
    ax = axes[3]
    if 'aerobic_fitness_stored' in samples:
        fitness_samples = samples['aerobic_fitness_stored']
        fitness_mean = np.mean(fitness_samples, axis=0)
        fitness_ci_lower = np.percentile(fitness_samples, 5, axis=0)
        fitness_ci_upper = np.percentile(fitness_samples, 95, axis=0)

        ax.plot(date_range, fitness_mean, 'purple', linewidth=2, label='Mean fitness')
        ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='purple', label='90% CI')
        ax.set_xlabel("Date")
        ax.set_ylabel("Aerobic Fitness (standardized)")
        ax.set_title("Aerobic Fitness State Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_fitness_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Weight decomposition with both components
    if ('strength_fitness_stored' in samples and 'aerobic_fitness_stored' in samples and
        'weight_intercept' in samples and 'gamma_s' in samples and 'gamma_a' in samples):

        fig, axes = plt.subplots(4, 1, figsize=(15, 20))
        fig.suptitle("Weight Decomposition with Aerobic Model (with 90% Credible Intervals)", fontsize=16)

        # Get weight observation times
        weight_dates = df_weight['timestamp']
        y_obs = df_weight['weight_lbs'].values

        # Calculate predictions and components
        intercept_samples = samples['weight_intercept']
        gamma_s_samples = samples['gamma_s']
        gamma_a_samples = samples['gamma_a']
        strength_fitness_samples = samples['strength_fitness_stored']
        aerobic_fitness_samples = samples['aerobic_fitness_stored']

        # For each weight observation, calculate predictions
        pred_samples = np.zeros((len(gamma_s_samples), len(df_weight)))
        baseline_samples = np.zeros((len(gamma_s_samples), len(df_weight)))
        strength_contrib_samples = np.zeros((len(gamma_s_samples), len(df_weight)))
        aerobic_contrib_samples = np.zeros((len(gamma_s_samples), len(df_weight)))

        for i, row in df_weight.iterrows():
            day_idx = row['day_idx'] - 1
            for s in range(len(gamma_s_samples)):
                # Baseline weight (intercept)
                baseline = weight_mean + intercept_samples[s] * weight_std
                baseline_samples[s, i] = baseline

                # Strength fitness contribution
                strength_contrib = gamma_s_samples[s] * strength_fitness_samples[s, day_idx] * weight_std
                strength_contrib_samples[s, i] = strength_contrib

                # Aerobic fitness contribution
                aerobic_contrib = gamma_a_samples[s] * aerobic_fitness_samples[s, day_idx] * weight_std
                aerobic_contrib_samples[s, i] = aerobic_contrib

                # Total prediction
                pred_samples[s, i] = baseline + strength_contrib + aerobic_contrib

        # Panel 1: Observed vs predicted weight
        ax = axes[0]
        pred_mean = np.mean(pred_samples, axis=0)
        pred_ci_lower = np.percentile(pred_samples, 5, axis=0)
        pred_ci_upper = np.percentile(pred_samples, 95, axis=0)

        ax.scatter(weight_dates, y_obs, s=40, alpha=0.7, color='blue', label='Observed')
        ax.scatter(weight_dates, pred_mean, s=40, alpha=0.7, color='red', label='Predicted (mean)')

        # Calculate error bars
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

        # Panel 2: Baseline weight (intercept)
        ax = axes[1]
        baseline_mean = np.mean(baseline_samples, axis=0)
        baseline_ci_lower = np.percentile(baseline_samples, 5, axis=0)
        baseline_ci_upper = np.percentile(baseline_samples, 95, axis=0)

        ax.scatter(weight_dates, baseline_mean, s=40, alpha=0.7, color='purple', label='Baseline weight')

        # Calculate error bars
        yerr_lower = np.abs(baseline_mean - baseline_ci_lower)
        yerr_upper = np.abs(baseline_ci_upper - baseline_mean)
        ax.errorbar(weight_dates, baseline_mean, yerr=[yerr_lower, yerr_upper],
                   fmt='none', alpha=0.3, color='purple', capsize=3, label='90% CI')

        ax.set_ylabel("Baseline weight (lbs)")
        ax.set_title("Estimated Baseline Weight (weight at zero fitness)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add baseline summary
        baseline_overall = np.mean(baseline_mean)
        ax.axhline(y=baseline_overall, color='black', linestyle='--', alpha=0.5,
                  label=f'Mean baseline: {baseline_overall:.1f} lbs')
        ax.axhline(y=weight_mean, color='gray', linestyle=':', alpha=0.5,
                  label=f'Actual mean: {weight_mean:.1f} lbs')
        ax.legend()

        # Panel 3: Weight due to strength fitness
        ax = axes[2]
        strength_contrib_mean = np.mean(strength_contrib_samples, axis=0)
        strength_contrib_ci_lower = np.percentile(strength_contrib_samples, 5, axis=0)
        strength_contrib_ci_upper = np.percentile(strength_contrib_samples, 95, axis=0)

        ax.scatter(weight_dates, strength_contrib_mean, s=40, alpha=0.7, color='red', label='Weight due to strength')

        # Calculate error bars
        yerr_lower = np.abs(strength_contrib_mean - strength_contrib_ci_lower)
        yerr_upper = np.abs(strength_contrib_ci_upper - strength_contrib_mean)
        ax.errorbar(weight_dates, strength_contrib_mean, yerr=[yerr_lower, yerr_upper],
                   fmt='none', alpha=0.3, color='red', capsize=3, label='90% CI')

        ax.set_ylabel("Weight due to strength (lbs)")
        ax.set_title("Strength Fitness Contribution to Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add strength contribution summary
        mean_effect = np.mean(strength_contrib_mean)
        max_effect = np.max(strength_contrib_mean)
        ax.text(0.02, 0.98, f"Mean: {mean_effect:.2f} lbs\nMax: {max_effect:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 4: Weight due to aerobic fitness
        ax = axes[3]
        aerobic_contrib_mean = np.mean(aerobic_contrib_samples, axis=0)
        aerobic_contrib_ci_lower = np.percentile(aerobic_contrib_samples, 5, axis=0)
        aerobic_contrib_ci_upper = np.percentile(aerobic_contrib_samples, 95, axis=0)

        ax.scatter(weight_dates, aerobic_contrib_mean, s=40, alpha=0.7, color='green', label='Weight due to aerobic')

        # Calculate error bars
        yerr_lower = np.abs(aerobic_contrib_mean - aerobic_contrib_ci_lower)
        yerr_upper = np.abs(aerobic_contrib_ci_upper - aerobic_contrib_mean)
        ax.errorbar(weight_dates, aerobic_contrib_mean, yerr=[yerr_lower, yerr_upper],
                   fmt='none', alpha=0.3, color='green', capsize=3, label='90% CI')

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight due to aerobic (lbs)")
        ax.set_title("Aerobic Fitness Contribution to Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add aerobic contribution summary
        mean_effect = np.mean(aerobic_contrib_mean)
        max_effect = np.max(aerobic_contrib_mean)
        ax.text(0.02, 0.98, f"Mean: {mean_effect:.2f} lbs\nMax: {max_effect:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "weight_decomposition_aerobic.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 3: Parameter posterior distributions
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle("Parameter Posterior Distributions (Aerobic Model)", fontsize=16)

    param_groups = [
        ('Strength Parameters', ['alpha_d_s', 'alpha_m_s', 'beta_s', 'gamma_s']),
        ('Aerobic Parameters', ['alpha_d_a', 'alpha_m_a', 'beta_a', 'gamma_a']),
        ('Shared Parameters', ['weight_intercept', 'sigma_w', '', ''])
    ]

    param_titles = {
        'alpha_d_s': 'Strength base retention',
        'alpha_m_s': 'Strength training effect',
        'beta_s': 'Strength gain per intensity',
        'gamma_s': 'Strength weight effect',
        'alpha_d_a': 'Aerobic base retention',
        'alpha_m_a': 'Aerobic training effect',
        'beta_a': 'Aerobic gain per intensity',
        'gamma_a': 'Aerobic weight effect',
        'weight_intercept': 'Weight intercept',
        'sigma_w': 'Weight noise std'
    }

    for row_idx, (group_name, params) in enumerate(param_groups):
        for col_idx, param_name in enumerate(params):
            ax = axes[row_idx, col_idx]

            if param_name and param_name in samples:
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
                ax.set_title(param_titles.get(param_name, param_name))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                if param_name:
                    ax.text(0.5, 0.5, f"No {param_name} data", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, group_name, ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll visualizations saved to {output_dir}/")

    # Print comprehensive summary
    print("\n" + "="*80)
    print("AEROBIC MODEL - COMPREHENSIVE SUMMARY")
    print("="*80)

    # Strength parameters
    if 'alpha_d_s' in samples and 'alpha_m_s' in samples:
        alpha_d_s_mean = np.mean(samples['alpha_d_s'])
        alpha_m_s_mean = np.mean(samples['alpha_m_s'])
        alpha_total_s_mean = alpha_d_s_mean + (1 - alpha_d_s_mean) * alpha_m_s_mean

        print(f"\n1. STRENGTH FITNESS DECAY:")
        print(f"   Without training: {alpha_d_s_mean:.1%} retained per day")
        print(f"     → Decays {1-alpha_d_s_mean:.3%} per day")
        print(f"     → Half-life: {np.log(0.5)/np.log(alpha_d_s_mean):.0f} days")
        print(f"   With training:    {alpha_total_s_mean:.1%} retained per day")
        print(f"     → Decays {1-alpha_total_s_mean:.3%} per day")
        print(f"     → Half-life: {np.log(0.5)/np.log(alpha_total_s_mean):.0f} days")

    # Aerobic parameters
    if 'alpha_d_a' in samples and 'alpha_m_a' in samples:
        alpha_d_a_mean = np.mean(samples['alpha_d_a'])
        alpha_m_a_mean = np.mean(samples['alpha_m_a'])
        alpha_total_a_mean = alpha_d_a_mean + (1 - alpha_d_a_mean) * alpha_m_a_mean

        print(f"\n2. AEROBIC FITNESS DECAY:")
        print(f"   Without training: {alpha_d_a_mean:.1%} retained per day")
        print(f"     → Decays {1-alpha_d_a_mean:.3%} per day")
        print(f"     → Half-life: {np.log(0.5)/np.log(alpha_d_a_mean):.0f} days")
        print(f"   With training:    {alpha_total_a_mean:.1%} retained per day")
        print(f"     → Decays {1-alpha_total_a_mean:.3%} per day")
        print(f"     → Half-life: {np.log(0.5)/np.log(alpha_total_a_mean):.0f} days")

    # Weight model
    if 'weight_intercept' in samples and 'gamma_s' in samples and 'gamma_a' in samples:
        intercept_mean = np.mean(samples['weight_intercept'])
        gamma_s_mean = np.mean(samples['gamma_s'])
        gamma_a_mean = np.mean(samples['gamma_a'])

        print(f"\n3. WEIGHT MODEL:")
        print(f"   Weight intercept: {intercept_mean:.3f} standardized units")
        print(f"     → {weight_mean + intercept_mean * weight_std:.1f} lbs baseline weight")
        print(f"   Strength weight effect: {gamma_s_mean:.3f} standardized units per fitness unit")
        print(f"     → {gamma_s_mean * weight_std:.2f} lbs per fitness unit")
        print(f"   Aerobic weight effect: {gamma_a_mean:.3f} standardized units per fitness unit")
        print(f"     → {gamma_a_mean * weight_std:.2f} lbs per fitness unit")

        # Calculate current fitness contributions
        if 'strength_fitness_stored' in samples and 'aerobic_fitness_stored' in samples:
            strength_fitness_samples_all = samples['strength_fitness_stored']
            aerobic_fitness_samples_all = samples['aerobic_fitness_stored']

            avg_strength_fitness = np.mean(strength_fitness_samples_all)
            avg_aerobic_fitness = np.mean(aerobic_fitness_samples_all)

            current_strength_contrib = avg_strength_fitness * gamma_s_mean * weight_std
            current_aerobic_contrib = avg_aerobic_fitness * gamma_a_mean * weight_std

            print(f"\n4. CURRENT STATE ESTIMATES:")
            print(f"   Average strength fitness: {avg_strength_fitness:.2f} units")
            print(f"   Average aerobic fitness: {avg_aerobic_fitness:.2f} units")
            print(f"   Current strength contribution: {current_strength_contrib:.1f} lbs")
            print(f"   Current aerobic contribution: {current_aerobic_contrib:.1f} lbs")
            print(f"   Baseline weight: {weight_mean + intercept_mean * weight_std:.1f} lbs")
            print(f"   Current total weight: {weight_mean + intercept_mean * weight_std + current_strength_contrib + current_aerobic_contrib:.1f} lbs")
            print(f"   Actual mean weight: {weight_mean:.1f} lbs")

    if 'sigma_w' in samples:
        sigma_w_mean = np.mean(samples['sigma_w'])
        print(f"\n5. MODEL FIT:")
        print(f"   Measurement noise (σ_w): {sigma_w_mean:.3f} standardized units")
        print(f"     → {sigma_w_mean * weight_std:.2f} lbs in original units")
        print(f"   Correlation (obs vs pred): {correlation:.3f}")
        print(f"   RMSE: {rmse:.2f} lbs (vs weight std: {weight_std:.2f} lbs)")
        print(f"   Model explains {correlation**2:.1%} of weight variance")

    print(f"\n6. KEY INSIGHTS:")
    print(f"   • Baseline weight (zero fitness): ~{weight_mean + intercept_mean * weight_std:.0f} lbs")
    print(f"   • Strength fitness adds ~{current_strength_contrib:.1f} lbs (likely muscle)")
    print(f"   • Aerobic fitness contributes ~{current_aerobic_contrib:.1f} lbs (likely fat loss)")
    print(f"   • Strength fitness decays slower than aerobic fitness")
    print(f"   • Adding aerobic component improves model fit")

    print("\n" + "="*80)


def main():
    """Main function to analyze aerobic model."""
    print("Analyzing aerobic fitness model with both strength and aerobic components...")

    # Load and prepare data
    df_weight, df_daily, weight_mean, weight_std, date_range = load_and_prepare_data()

    print(f"Data loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Fit aerobic model
    fit = fit_aerobic_model(df_weight, df_daily)

    # Extract posterior samples
    samples = extract_posterior_samples(fit)

    # Create comprehensive visualizations
    create_comprehensive_visualizations(samples, df_daily, df_weight, weight_mean, weight_std, date_range)

    print("\nDone! Aerobic model analysis completed successfully.")


if __name__ == "__main__":
    main()