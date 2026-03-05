#!/usr/bin/env python3
"""Test symmetric models with different outcome distributions for weight data."""

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


def prepare_data():
    """Prepare data for all models."""
    print("Preparing data for symmetric model comparison...")
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

    # For Gamma and log-normal models, we need positive values
    # Shift weight to be positive (add constant to make all values > 0)
    weight_min = df_weight['weight_std'].min()
    if weight_min <= 0:
        shift = abs(weight_min) + 0.1  # Add small buffer
        df_weight['weight_std_positive'] = df_weight['weight_std'] + shift
        print(f"Weight shifted by {shift:.3f} to ensure positivity for Gamma/log-normal models")
    else:
        df_weight['weight_std_positive'] = df_weight['weight_std']
        print("Weight already positive, no shift needed")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Prepare Stan data for normal model
    stan_data_normal = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    # Prepare Stan data for Gamma and log-normal models (need positive weights)
    stan_data_positive = {
        'D': D,
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std_positive'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"\nData loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")
    print(f"Weight range (standardized): [{df_weight['weight_std'].min():.3f}, {df_weight['weight_std'].max():.3f}]")
    print(f"Weight range (positive): [{df_weight['weight_std_positive'].min():.3f}, {df_weight['weight_std_positive'].max():.3f}]")

    return stan_data_normal, stan_data_positive, weight_mean, weight_std, df_weight, df_daily


def fit_model(model_file, stan_data, model_name):
    """Fit a single model and return results."""
    print(f"\nFitting {model_name}...")
    print("-"*40)

    model = cmdstanpy.CmdStanModel(stan_file=model_file)

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

    # Check diagnostics
    try:
        diagnose = fit.diagnose()
        if "Divergent" in diagnose:
            print(f"WARNING: {model_name} has divergent transitions")
        if "maximum treedepth" in diagnose:
            print(f"WARNING: {model_name} hit maximum treedepth")
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    return fit


def extract_results(fit, model_name, weight_std, shift=0):
    """Extract key results from model fit."""
    draws_df = fit.draws_pd()

    # Key parameters to check
    key_params = ['gamma_s', 'gamma_a', 'weight_intercept']

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

    # Calculate weight effects in original units
    if 'gamma_s' in results and 'gamma_a' in results:
        # Adjust for shift if needed (for Gamma/log-normal models)
        gamma_s_lbs = results['gamma_s']['mean'] * weight_std
        gamma_a_lbs = results['gamma_a']['mean'] * weight_std

        results['gamma_s_lbs'] = gamma_s_lbs
        results['gamma_a_lbs'] = gamma_a_lbs

    # Calculate log likelihood for model comparison
    log_lik_cols = [col for col in draws_df.columns if col.startswith('log_lik_weight[')]
    if log_lik_cols:
        log_lik_vals = draws_df[log_lik_cols].values
        # Sum across observations for each sample
        log_lik_per_sample = np.sum(log_lik_vals, axis=1)
        results['log_lik_mean'] = np.mean(log_lik_per_sample)
        results['log_lik_std'] = np.std(log_lik_per_sample)

    results['model_name'] = model_name
    results['draws_df'] = draws_df

    return results


def compare_models(all_results):
    """Compare results across models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Outcome Distributions")
    print("="*80)

    # Create comparison table
    print("\nWeight Effects Comparison (gamma parameters):")
    print("-"*80)
    print(f"{'Model':25} {'gamma_s':>10} {'gamma_a':>10} {'log-lik':>10}")
    print("-"*80)

    for model_name, results in all_results.items():
        gamma_s = results.get('gamma_s', {}).get('mean', np.nan)
        gamma_a = results.get('gamma_a', {}).get('mean', np.nan)
        log_lik = results.get('log_lik_mean', np.nan)

        print(f"{model_name:25} {gamma_s:10.3f} {gamma_a:10.3f} {log_lik:10.1f}")

    # Compare in original units
    print("\nWeight Effects in Original Units (lbs per fitness unit):")
    print("-"*80)
    print(f"{'Model':25} {'Strength (lbs)':>15} {'Aerobic (lbs)':>15}")
    print("-"*80)

    for model_name, results in all_results.items():
        gamma_s_lbs = results.get('gamma_s_lbs', np.nan)
        gamma_a_lbs = results.get('gamma_a_lbs', np.nan)

        print(f"{model_name:25} {gamma_s_lbs:15.3f} {gamma_a_lbs:15.3f}")

    # Bayesian model comparison using WAIC/LOO if available
    print("\n" + "="*80)
    print("BAYESIAN MODEL COMPARISON")
    print("="*80)

    # Calculate WAIC approximation from log likelihood
    models_with_loglik = [(name, res) for name, res in all_results.items()
                         if 'log_lik_mean' in res]

    if len(models_with_loglik) >= 2:
        print("\nApproximate WAIC comparison (lower is better):")
        print("-"*40)

        # WAIC ≈ -2 * (mean log likelihood - variance of log likelihood)
        waic_values = {}
        for name, res in models_with_loglik:
            mean_ll = res['log_lik_mean']
            var_ll = res['log_lik_std']**2
            waic = -2 * (mean_ll - var_ll)
            waic_values[name] = waic

        # Sort by WAIC
        sorted_waic = sorted(waic_values.items(), key=lambda x: x[1])

        for i, (name, waic) in enumerate(sorted_waic):
            if i == 0:
                print(f"{name:25} WAIC = {waic:8.1f} (best)")
            else:
                diff = waic - sorted_waic[0][1]
                print(f"{name:25} WAIC = {waic:8.1f} (Δ = {diff:5.1f})")


def visualize_comparison(all_results):
    """Create visualization comparing gamma distributions across models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Symmetric Model Comparison: Different Outcome Distributions", fontsize=16)

    # Colors for different models
    colors = {'Normal': 'blue', 'Gamma': 'green', 'Log-Normal': 'red'}

    # Plot gamma_s distributions
    ax = axes[0, 0]
    for model_name, results in all_results.items():
        if 'gamma_s' in results and 'draws_df' in results:
            draws = results['draws_df']['gamma_s'].values
            color = colors.get(model_name.split()[0], 'gray')
            ax.hist(draws, bins=30, density=True, alpha=0.5,
                   label=model_name, color=color)

    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('gamma_s (strength effect)')
    ax.set_ylabel('Density')
    ax.set_title('Strength Weight Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot gamma_a distributions
    ax = axes[0, 1]
    for model_name, results in all_results.items():
        if 'gamma_a' in results and 'draws_df' in results:
            draws = results['draws_df']['gamma_a'].values
            color = colors.get(model_name.split()[0], 'gray')
            ax.hist(draws, bins=30, density=True, alpha=0.5,
                   label=model_name, color=color)

    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_xlabel('gamma_a (aerobic effect)')
    ax.set_ylabel('Density')
    ax.set_title('Aerobic Weight Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot comparison of gamma_s means with credible intervals
    ax = axes[0, 2]
    model_names = []
    gamma_s_means = []
    gamma_s_cis = []

    for model_name, results in all_results.items():
        if 'gamma_s' in results:
            model_names.append(model_name)
            gamma_s_means.append(results['gamma_s']['mean'])
            gamma_s_cis.append([
                results['gamma_s']['ci_lower'],
                results['gamma_s']['ci_upper']
            ])

    if gamma_s_means:
        y_pos = np.arange(len(model_names))
        ax.errorbar(gamma_s_means, y_pos,
                   xerr=[[m - ci[0] for m, ci in zip(gamma_s_means, gamma_s_cis)],
                         [ci[1] - m for m, ci in zip(gamma_s_means, gamma_s_cis)]],
                   fmt='o', capsize=5)
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('gamma_s (strength effect)')
        ax.set_title('Strength Effect Comparison')
        ax.grid(True, alpha=0.3)

    # Plot comparison of gamma_a means with credible intervals
    ax = axes[1, 0]
    model_names = []
    gamma_a_means = []
    gamma_a_cis = []

    for model_name, results in all_results.items():
        if 'gamma_a' in results:
            model_names.append(model_name)
            gamma_a_means.append(results['gamma_a']['mean'])
            gamma_a_cis.append([
                results['gamma_a']['ci_lower'],
                results['gamma_a']['ci_upper']
            ])

    if gamma_a_means:
        y_pos = np.arange(len(model_names))
        ax.errorbar(gamma_a_means, y_pos,
                   xerr=[[m - ci[0] for m, ci in zip(gamma_a_means, gamma_a_cis)],
                         [ci[1] - m for m, ci in zip(gamma_a_means, gamma_a_cis)]],
                   fmt='o', capsize=5)
        ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('gamma_a (aerobic effect)')
        ax.set_title('Aerobic Effect Comparison')
        ax.grid(True, alpha=0.3)

    # Plot log likelihood comparison
    ax = axes[1, 1]
    model_names = []
    log_lik_means = []
    log_lik_stds = []

    for model_name, results in all_results.items():
        if 'log_lik_mean' in results:
            model_names.append(model_name)
            log_lik_means.append(results['log_lik_mean'])
            log_lik_stds.append(results['log_lik_std'])

    if log_lik_means:
        y_pos = np.arange(len(model_names))
        ax.barh(y_pos, log_lik_means, xerr=log_lik_stds,
               alpha=0.7, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Mean Log Likelihood')
        ax.set_title('Model Fit Comparison')
        ax.grid(True, alpha=0.3)

    # Leave last subplot empty or add summary
    axes[1, 2].axis('off')
    summary_text = "Model Comparison Summary:\n\n"
    for model_name, results in all_results.items():
        if 'gamma_s' in results and 'gamma_a' in results:
            gamma_s = results['gamma_s']['mean']
            gamma_a = results['gamma_a']['mean']
            prob_s_positive = (results['draws_df']['gamma_s'] > 0).mean() if 'draws_df' in results else np.nan
            prob_a_positive = (results['draws_df']['gamma_a'] > 0).mean() if 'draws_df' in results else np.nan

            summary_text += f"{model_name}:\n"
            summary_text += f"  γ_s: {gamma_s:.3f} (P>0: {prob_s_positive:.1%})\n"
            summary_text += f"  γ_a: {gamma_a:.3f} (P>0: {prob_a_positive:.1%})\n\n"

    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                   verticalalignment='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_dir = Path("output/symmetric_outcome_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / "outcome_distribution_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_dir}/outcome_distribution_comparison.png")


def main():
    """Main function to compare symmetric models with different outcome distributions."""
    # Prepare data
    stan_data_normal, stan_data_positive, weight_mean, weight_std, df_weight, df_daily = prepare_data()

    # Define models to test
    models = [
        {
            'name': 'Normal',
            'file': 'stan/weight_state_space_training_decay_aerobic_symmetric.stan',
            'data': stan_data_normal,
            'shift': 0
        },
        {
            'name': 'Gamma',
            'file': 'stan/weight_state_space_training_decay_aerobic_symmetric_gamma.stan',
            'data': stan_data_positive,
            'shift': df_weight['weight_std_positive'].iloc[0] - df_weight['weight_std'].iloc[0]
        },
        {
            'name': 'Log-Normal',
            'file': 'stan/weight_state_space_training_decay_aerobic_symmetric_lognormal.stan',
            'data': stan_data_positive,
            'shift': df_weight['weight_std_positive'].iloc[0] - df_weight['weight_std'].iloc[0]
        }
    ]

    # Fit all models
    all_results = {}

    for model_info in models:
        try:
            fit = fit_model(model_info['file'], model_info['data'], model_info['name'])
            results = extract_results(fit, model_info['name'], weight_std, model_info['shift'])
            all_results[model_info['name']] = results
        except Exception as e:
            print(f"ERROR fitting {model_info['name']}: {e}")

    # Compare models
    compare_models(all_results)

    # Create visualizations
    if len(all_results) >= 2:
        visualize_comparison(all_results)

    print("\n" + "="*80)
    print("INTERPRETATION GUIDANCE")
    print("="*80)

    print("\nFor continuous, positively bounded data like strength intensity:")
    print("1. Normal distribution: Assumes symmetric errors, can produce negative predictions")
    print("2. Gamma distribution: Natural for positive data, handles right skew")
    print("3. Log-normal distribution: Multiplicative effects, percentage changes")

    print("\nRecommendation based on data characteristics:")
    print("- If data has right skew: Gamma distribution")
    print("- If effects are multiplicative: Log-normal distribution")
    print("- If data approximately normal: Normal distribution (simplest)")

    print("\nCheck posterior predictive checks to see which distribution fits best.")

    return all_results


if __name__ == "__main__":
    all_results = main()