#!/usr/bin/env python3
"""Test the logit-parameterized decay model."""

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


def main():
    """Test logit-parameterized decay model."""
    print("Testing logit-parameterized decay model...")

    # Load data
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

    # Prepare Stan data
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"Data prepared: {D} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Fit logit model
    model_path = Path("stan/weight_state_space_training_decay_logit.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

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

    # Extract posterior samples
    draws_df = fit.draws_pd()

    # Extract parameter samples
    param_names = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
    samples = {}
    for param in param_names:
        if param in draws_df.columns:
            samples[param] = draws_df[param].values

    # Extract fitness states
    fitness_cols = [col for col in draws_df.columns if col.startswith('fitness_stored[')]
    if fitness_cols:
        fitness_values = {}
        for col in fitness_cols:
            idx_str = col.split('[')[1].split(']')[0]
            idx = int(idx_str) - 1
            fitness_values[idx] = draws_df[col].values

        sorted_indices = sorted(fitness_values.keys())
        fitness_array = np.column_stack([fitness_values[idx] for idx in sorted_indices])
        samples['fitness_stored'] = fitness_array

    # Print parameter summary
    print("\nParameter Summary (mean ± 90% CI):")
    param_titles = [
        'Base retention (no training)',
        'Additional retention when trained',
        'Gain per intensity',
        'Weight effect',
        'Weight noise std'
    ]

    for param_name, param_title in zip(param_names, param_titles):
        if param_name in samples:
            param_samples = samples[param_name]
            mean_val = np.mean(param_samples)
            ci_lower = np.percentile(param_samples, 5)
            ci_upper = np.percentile(param_samples, 95)
            print(f"  {param_name}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] - {param_title}")

    # Special interpretation
    if 'alpha_d' in samples and 'alpha_m' in samples:
        alpha_d_mean = np.mean(samples['alpha_d'])
        alpha_m_mean = np.mean(samples['alpha_m'])
        alpha_total_mean = alpha_d_mean + (1 - alpha_d_mean) * alpha_m_mean

        print(f"\nInterpretation:")
        print(f"  Without training: fitness retained = {alpha_d_mean:.3f}")
        print(f"    (decays by {1-alpha_d_mean:.3f} per day)")
        print(f"  With training: fitness retained = {alpha_total_mean:.3f}")
        print(f"    (decays by {1-alpha_total_mean:.3f} per day)")
        print(f"  Training effect: adds {alpha_m_mean:.3f} of remaining decay")

        # Calculate half-life
        def half_life(retention_rate):
            """Calculate half-life in days given daily retention rate."""
            if retention_rate <= 0:
                return 0
            if retention_rate >= 1:
                return float('inf')
            return np.log(0.5) / np.log(retention_rate)

        half_life_no_train = half_life(alpha_d_mean)
        half_life_with_train = half_life(alpha_total_mean)

        print(f"\nHalf-life of fitness:")
        print(f"  Without training: {half_life_no_train:.1f} days")
        print(f"  With training: {half_life_with_train:.1f} days")

        # Calculate steady-state fitness for constant training
        print(f"\nSteady-state analysis:")
        print(f"  With constant training at average intensity:")
        avg_intensity = np.mean(df_daily['strength_intensity_std'][df_daily['strength_intensity_std'] > 0])
        print(f"    Average training intensity: {avg_intensity:.3f}")

        if 'beta' in samples:
            beta_mean = np.mean(samples['beta'])
            steady_state = (beta_mean * avg_intensity) / (1 - alpha_total_mean)
            print(f"    Steady-state fitness: {steady_state:.3f} units")

            if 'gamma' in samples:
                gamma_mean = np.mean(samples['gamma'])
                weight_effect = steady_state * gamma_mean * weight_std
                print(f"    Steady-state weight effect: {weight_effect:.2f} lbs")

    if 'gamma' in samples:
        gamma_mean = np.mean(samples['gamma'])
        print(f"\nWeight effect: {gamma_mean:.3f} standardized units per fitness unit")
        print(f"  In lbs: {gamma_mean * weight_std:.2f} lbs per fitness unit")

    # Create visualization
    output_dir = Path("output/logit_model_test")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot fitness over time
    if 'fitness_stored' in samples:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Logit Model: Fitness Over Time", fontsize=16)

        # Fitness state
        ax = axes[0]
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

        # Training days and intensity
        ax = axes[1]
        # Plot intensity as bars
        ax.bar(date_range, df_daily['strength_intensity_std'], alpha=0.5, color='green', width=1.0)
        ax.set_ylabel("Training Intensity")
        ax.set_title("Training Days and Intensity")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fitness_over_time.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to {output_dir}/fitness_over_time.png")

    # Check for divergent transitions
    print("\nModel diagnostics:")
    print(f"  Divergent transitions: Chain 1: {fit.diagnose().get('num_divergent', 0)}, Chain 2: {fit.diagnose().get('num_divergent', 0)}")


if __name__ == "__main__":
    main()