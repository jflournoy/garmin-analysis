#!/usr/bin/env python3
"""Test the symmetric aerobic model with same priors for strength and aerobic effects."""

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


def test_symmetric_model():
    """Test the symmetric aerobic model."""
    print("Testing symmetric aerobic model with same priors for strength and aerobic effects...")
    print("="*80)

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

        print(f"{intensity_type}: min={min_val:.2f}, std={std:.2f}")

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
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"\nData loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    print("\nCompiling symmetric aerobic model...")
    model = cmdstanpy.CmdStanModel(stan_file="stan/weight_state_space_training_decay_aerobic_symmetric.stan")

    print("Fitting model with symmetric priors...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        adapt_delta=0.99,
        max_treedepth=15,
        show_progress=True,
        seed=12345,
    )

    # Check for issues
    print("\n" + "="*80)
    print("MODEL DIAGNOSTICS")
    print("="*80)

    try:
        diagnose = fit.diagnose()
        print(diagnose[:2000])  # Print first 2000 chars
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    # Extract results
    draws_df = fit.draws_pd()

    print("\n" + "="*80)
    print("KEY PARAMETER RESULTS (Symmetric Priors)")
    print("="*80)

    # Key parameters to check
    key_params = ['gamma_s', 'gamma_a', 'weight_intercept', 'sigma_w',
                  'alpha_d_s', 'alpha_m_s', 'beta_s',
                  'alpha_d_a', 'alpha_m_a', 'beta_a']

    results = {}
    for param in key_params:
        if param in draws_df.columns:
            mean_val = draws_df[param].mean()
            std_val = draws_df[param].std()
            ci_lower = np.percentile(draws_df[param], 2.5)
            ci_upper = np.percentile(draws_df[param], 97.5)
            results[param] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

            # Report credible intervals for gamma parameters
            if param in ['gamma_s', 'gamma_a']:
                sign = "positive" if mean_val > 0 else "negative"
                print(f"{param}: {mean_val:.3f} ± {std_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  → 95% credible interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  → Posterior mean is {sign}")
            else:
                print(f"{param}: {mean_val:.3f} ± {std_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

    # Calculate weight effects in original units
    if 'gamma_s' in results and 'gamma_a' in results:
        gamma_s_lbs = results['gamma_s']['mean'] * weight_std
        gamma_a_lbs = results['gamma_a']['mean'] * weight_std

        print(f"\nWeight effects in original units (lbs per fitness unit):")
        print(f"  Strength: {gamma_s_lbs:.3f} lbs")
        print(f"  Aerobic:  {gamma_a_lbs:.3f} lbs")

    # Compare with original model results
    print("\n" + "="*80)
    print("COMPARISON WITH ORIGINAL MODEL (Asymmetric Priors)")
    print("="*80)

    # Original model results from earlier analysis
    original_results = {
        'gamma_s': 0.173,  # From aerobic model
        'gamma_a': -0.126, # From aerobic model
        'weight_intercept': -1.402,
        'sigma_w': 0.463
    }

    print("\nParameter comparison (Symmetric vs Original Asymmetric Priors):")
    print("Parameter     Symmetric (mean)   Original (mean)   Difference")
    print("-" * 60)

    for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'sigma_w']:
        if param in results and param in original_results:
            sym_mean = results[param]['mean']
            orig_mean = original_results[param]
            diff = sym_mean - orig_mean
            print(f"{param:12} {sym_mean:10.3f}       {orig_mean:10.3f}       {diff:10.3f}")

    # Calculate predictions for comparison
    print("\n" + "="*80)
    print("PREDICTIVE PERFORMANCE")
    print("="*80)

    # Get fitness samples
    fitness_cols_s = [col for col in draws_df.columns if col.startswith('strength_fitness_stored[')]
    fitness_cols_a = [col for col in draws_df.columns if col.startswith('aerobic_fitness_stored[')]

    if fitness_cols_s and fitness_cols_a and 'gamma_s' in draws_df.columns and 'gamma_a' in draws_df.columns:
        # Extract gamma samples
        gamma_s_samples = draws_df['gamma_s'].values
        gamma_a_samples = draws_df['gamma_a'].values
        intercept_samples = draws_df['weight_intercept'].values

        # Calculate predictions for each weight observation
        pred_samples = np.zeros((len(gamma_s_samples), len(df_weight)))

        for i, row in df_weight.iterrows():
            day_idx_val = row['day_idx'] - 1  # Convert to 0-index

            # Get fitness values for this day (approximate - would need proper extraction)
            # For now, just calculate mean prediction
            pass

        print("Fitness states extracted, predictions could be calculated")
    else:
        print("Could not extract all necessary components for prediction calculation")

    # Create simple visualization of gamma distributions
    if 'gamma_s' in draws_df.columns and 'gamma_a' in draws_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Weight Effects with Symmetric Priors (centered at 0)", fontsize=14)

        # Plot gamma_s
        ax = axes[0]
        ax.hist(draws_df['gamma_s'].values, bins=30, density=True, alpha=0.7, color='blue')
        ax.axvline(results['gamma_s']['mean'], color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {results["gamma_s"]["mean"]:.3f}')
        ax.axvspan(results['gamma_s']['ci_lower'], results['gamma_s']['ci_upper'],
                  alpha=0.2, color='red', label='95% CI')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('gamma_s (strength effect)')
        ax.set_ylabel('Density')
        ax.set_title('Strength Weight Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot gamma_a
        ax = axes[1]
        ax.hist(draws_df['gamma_a'].values, bins=30, density=True, alpha=0.7, color='green')
        ax.axvline(results['gamma_a']['mean'], color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {results["gamma_a"]["mean"]:.3f}')
        ax.axvspan(results['gamma_a']['ci_lower'], results['gamma_a']['ci_upper'],
                  alpha=0.2, color='red', label='95% CI')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('gamma_a (aerobic effect)')
        ax.set_ylabel('Density')
        ax.set_title('Aerobic Weight Effect')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir = Path("output/symmetric_model_test")
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / "symmetric_gamma_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nVisualization saved to {output_dir}/symmetric_gamma_distributions.png")

    print("\n" + "="*80)
    print("BAYESIAN INTERPRETATION")
    print("="*80)

    if 'gamma_a' in results:
        gamma_a_mean = results['gamma_a']['mean']
        gamma_a_ci_lower = results['gamma_a']['ci_lower']
        gamma_a_ci_upper = results['gamma_a']['ci_upper']

        # Calculate probability that gamma_a is negative
        if 'gamma_a' in draws_df.columns:
            prob_negative = (draws_df['gamma_a'] < 0).mean()
            prob_positive = (draws_df['gamma_a'] > 0).mean()

            print(f"\nPosterior probability that aerobic exercise decreases weight (gamma_a < 0):")
            print(f"  P(gamma_a < 0 | data) = {prob_negative:.1%}")
            print(f"  P(gamma_a > 0 | data) = {prob_positive:.1%}")

        print(f"\nPosterior distribution of aerobic effect (gamma_a):")
        print(f"  Mean: {gamma_a_mean:.3f}")
        print(f"  95% credible interval: [{gamma_a_ci_lower:.3f}, {gamma_a_ci_upper:.3f}]")

        if gamma_a_ci_lower > 0:
            print(f"\nInterpretation: High posterior probability (>97.5%) that aerobic exercise increases weight")
        elif gamma_a_ci_upper < 0:
            print(f"\nInterpretation: High posterior probability (>97.5%) that aerobic exercise decreases weight")
        else:
            print(f"\nInterpretation: Substantial uncertainty about aerobic exercise effect")
            print(f"  The 95% credible interval includes both positive and negative values")

    print("\nDone!")

    return fit, results


if __name__ == "__main__":
    test_symmetric_model()