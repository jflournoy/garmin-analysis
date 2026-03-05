#!/usr/bin/env python3
"""Quick visualization of training decay model results."""

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


def main():
    """Create quick visualizations."""
    print("Creating quick visualizations...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load only strength training intensity
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
        max_hr=185.0,
    )

    # Create date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_intensity['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    print(f"Date range: {min_date} to {max_date} ({D} days)")

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    df_act = df_intensity[['date', 'strength_training']].copy()
    df_daily = pd.merge(df_daily, df_act, on='date', how='left')
    df_daily['strength_intensity'] = df_daily['strength_training'].fillna(0.0)

    # Standardize intensity: (value - min)/std
    min_val = df_daily['strength_intensity'].min()
    std = df_daily['strength_intensity'].std()
    if std > 0:
        df_daily['strength_intensity_std'] = (df_daily['strength_intensity'] - min_val) / std
    else:
        df_daily['strength_intensity_std'] = df_daily['strength_intensity'] - min_val

    print(f"Strength intensity: min={min_val:.2f}, std={std:.2f}, scaled mean={df_daily['strength_intensity_std'].mean():.4f}, max={df_daily['strength_intensity_std'].max():.4f}")

    # Standardize weight: (value - mean)/std
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"Weight: mean={weight_mean:.2f}, std={weight_std:.2f}")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Prepare Stan data for simple model
    stan_data = {
        'D': D,
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
    }

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    # Fit simple model
    model_path = Path("stan/weight_state_space_training_decay_simple.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=200,
        iter_sampling=200,
        adapt_delta=0.8,
        max_treedepth=8,
        show_progress=True,
        seed=12345,
    )

    print("Converting to InferenceData...")
    try:
        idata = az.from_cmdstanpy(
            posterior=fit,
            posterior_predictive='y_weight_rep',
            log_likelihood='log_lik_weight',
            coords={
                'day': np.arange(1, stan_data['D'] + 1),
                'weight_obs': np.arange(stan_data['N_weight']),
            },
            dims={
                'fitness_stored': ['day'],
                'y_weight_rep': ['weight_obs'],
                'log_lik_weight': ['weight_obs'],
            }
        )
    except Exception as e:
        print(f"Warning: Could not convert with arviz: {e}")
        print("Creating minimal InferenceData manually...")
        # Create a minimal InferenceData structure
        import xarray as xr

        # Get posterior samples
        posterior_samples = fit.draws_pd()

        # Extract key parameters
        param_names = ['alpha_d', 'alpha_m', 'beta', 'gamma', 'sigma_w']
        posterior_dict = {}

        for param in param_names:
            if param in posterior_samples.columns:
                # Reshape to (chain, draw, ...)
                chains = fit.chains
                draws_per_chain = fit.num_draws_sampling
                samples = posterior_samples[param].values.reshape((chains, draws_per_chain))
                posterior_dict[param] = xr.DataArray(
                    samples,
                    dims=['chain', 'draw'],
                    coords={'chain': np.arange(1, chains+1), 'draw': np.arange(1, draws_per_chain+1)}
                )

        # Create xarray dataset
        posterior_ds = xr.Dataset(posterior_dict)
        idata = az.InferenceData(posterior=posterior_ds)

    # Create output directory
    output_dir = Path("output/training_decay_quick_viz")
    output_dir.mkdir(exist_ok=True)

    # Extract posterior
    posterior = idata.posterior

    # Plot 1: Intensity and fitness over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("Strength Training: Intensity and Fitness Over Time", fontsize=16)

    # Intensity
    ax = axes[0]
    ax.plot(date_range, df_daily['strength_intensity_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(date_range, 0, df_daily['strength_intensity_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity (standardized)")
    ax.set_title("Workout Intensity")
    ax.grid(True, alpha=0.3)

    # Fitness with CI
    ax = axes[1]
    fitness_samples = posterior['fitness_stored'].values
    fitness_mean = np.mean(fitness_samples, axis=(0, 1))
    fitness_ci_lower = np.percentile(fitness_samples, 5, axis=(0, 1))
    fitness_ci_upper = np.percentile(fitness_samples, 95, axis=(0, 1))

    ax.plot(date_range, fitness_mean, 'b-', linewidth=2, label='Mean fitness')
    ax.fill_between(date_range, fitness_ci_lower, fitness_ci_upper, alpha=0.3, color='blue', label='90% CI')
    ax.set_ylabel("Fitness (standardized)")
    ax.set_title("Fitness State (with 90% Credible Interval)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training days indicator
    ax = axes[2]
    trained_days = df_daily['strength_intensity'] > 0
    ax.plot(date_range, trained_days.astype(float), 'g-', alpha=0.7, linewidth=1, drawstyle='steps-post')
    ax.set_ylabel("Trained (1=yes, 0=no)")
    ax.set_title("Training Days")
    ax.set_xlabel("Date")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "intensity_fitness_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Weight predictions vs observed
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get posterior predictive samples
    y_rep_samples = posterior['y_weight_rep'].values
    y_rep_mean = np.mean(y_rep_samples, axis=(0, 1)) * weight_std + weight_mean
    y_rep_ci_lower = np.percentile(y_rep_samples, 5, axis=(0, 1)) * weight_std + weight_mean
    y_rep_ci_upper = np.percentile(y_rep_samples, 95, axis=(0, 1)) * weight_std + weight_mean

    # Observed weight
    y_obs = df_weight['weight_lbs'].values
    weight_dates = df_weight['timestamp']

    ax.scatter(weight_dates, y_obs, s=40, alpha=0.7, color='blue', label='Observed')
    ax.scatter(weight_dates, y_rep_mean, s=40, alpha=0.7, color='red', label='Predicted (mean)')
    ax.errorbar(weight_dates, y_rep_mean, yerr=[y_rep_mean - y_rep_ci_lower, y_rep_ci_upper - y_rep_mean],
               fmt='none', alpha=0.3, color='red', capsize=3, label='90% CI')

    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (lbs)")
    ax.set_title("Observed vs Predicted Weight (Simple Training Decay Model)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add correlation text
    correlation = np.corrcoef(y_obs, y_rep_mean)[0, 1]
    rmse = np.sqrt(np.mean((y_obs - y_rep_mean) ** 2))
    ax.text(0.02, 0.98, f"Correlation: {correlation:.3f}\nRMSE: {rmse:.2f} lbs",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "weight_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Parameter Posterior Distributions", fontsize=16)

    params = [
        ('alpha_d', 'Decay without training'),
        ('alpha_m', 'Training reduces decay'),
        ('beta', 'Gain per intensity'),
        ('gamma', 'Weight effect'),
        ('sigma_w', 'Measurement noise'),
    ]

    for idx, (param_name, param_title) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        if param_name in posterior:
            samples = posterior[param_name].values.flatten()

            ax.hist(samples, bins=30, density=True, alpha=0.7, color='steelblue')
            ax.axvline(np.mean(samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(samples):.3f}')

            # Add 90% CI
            ci_lower = np.percentile(samples, 5)
            ci_upper = np.percentile(samples, 95)
            ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'90% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

            ax.set_xlabel(param_name)
            ax.set_ylabel("Density")
            ax.set_title(param_title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No {param_name} data", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Weight effect of fitness over time
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate weight effect = fitness * gamma * weight_std
    gamma_samples = posterior['gamma'].values.flatten()
    weight_effect_samples = np.zeros((len(gamma_samples), D))

    for s in range(len(gamma_samples)):
        weight_effect_samples[s, :] = fitness_samples[:, :, :].flatten()[s::D][:D] * gamma_samples[s] * weight_std

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
    ax.text(0.02, 0.98, f"Max effect: {max_effect:.2f} lbs\nAvg effect: {avg_effect:.2f} lbs",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "weight_effect_over_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to {output_dir}/")

    # Print parameter summary
    print("\nParameter Summary (mean ± 90% CI):")
    for param_name, param_title in params:
        if param_name in posterior:
            samples = posterior[param_name].values.flatten()
            mean_val = np.mean(samples)
            ci_lower = np.percentile(samples, 5)
            ci_upper = np.percentile(samples, 95)
            print(f"  {param_name}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}] - {param_title}")

    # Special interpretation for alpha parameters
    if 'alpha_d' in posterior and 'alpha_m' in posterior:
        alpha_d_mean = np.mean(posterior['alpha_d'].values.flatten())
        alpha_m_mean = np.mean(posterior['alpha_m'].values.flatten())
        print(f"\nInterpretation:")
        print(f"  Without training: fitness decays to {alpha_d_mean:.3f} of previous day")
        print(f"  With training: fitness decays to {alpha_d_mean + alpha_m_mean:.3f} of previous day")
        print(f"  Training reduces decay by: {alpha_m_mean:.3f} ({alpha_m_mean/alpha_d_mean*100:.1f}% of base decay)")

    if 'gamma' in posterior:
        gamma_mean = np.mean(posterior['gamma'].values.flatten())
        print(f"  Weight effect: {gamma_mean:.3f} standardized units per fitness unit")
        print(f"  In lbs: {gamma_mean * weight_std:.2f} lbs per fitness unit")


if __name__ == "__main__":
    main()