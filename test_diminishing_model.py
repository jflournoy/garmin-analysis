#!/usr/bin/env python3
"""Test the diminishing returns model."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import arviz as az

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def test_diminishing_model():
    """Test the diminishing returns model."""
    print("Testing diminishing returns model...")

    # Configuration
    data_dir = Path("data")
    output_dir = Path("output/test_diminishing")
    output_dir.mkdir(parents=True, exist_ok=True)

    chains = 2  # Use fewer chains for testing
    iter_warmup = 500
    iter_sampling = 500

    print("Loading data...")

    # Load weight data
    df_weight = load_weight_data(data_dir)
    print(f"  Loaded {len(df_weight)} weight measurements")

    # Load intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['walking', 'cycling', 'strength_training'],
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

    print(f"  Date range: {min_date} to {max_date} ({D} days)")

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge intensity data
    for activity in ['walking', 'cycling', 'strength_training']:
        if activity in df_intensity.columns:
            df_act = df_intensity[['date', activity]].copy()
            df_daily = pd.merge(df_daily, df_act, on='date', how='left')
            df_daily[activity] = df_daily[activity].fillna(0.0)
        else:
            df_daily[activity] = 0.0

    # Combine walking and cycling into aerobic
    df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']
    df_daily['strength_intensity'] = df_daily['strength_training']

    # Standardize intensities (shift so min=0)
    # Use (value - min)/std - this gives reasonable fitness units (0-8)
    for col in ['aerobic_intensity', 'strength_intensity']:
        min_val = df_daily[col].min()
        std = df_daily[col].std()
        if std > 0:
            df_daily[f'{col}_std'] = (df_daily[col] - min_val) / std
        else:
            df_daily[f'{col}_std'] = df_daily[col] - min_val
        print(f"  {col}: min={min_val:.2f}, std={std:.2f}, scaled mean={df_daily[f'{col}_std'].mean():.4f}, max={df_daily[f'{col}_std'].max():.4f}")

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"  Weight: mean={weight_mean:.2f}, std={weight_std:.2f}")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)

    # Drop any outside date range
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Scale time to [0, 1] for GP
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()

    if time_range > 0:
        df_weight['t_scaled'] = (df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
    else:
        df_weight['t_scaled'] = 0.0

    # Hour of day for daily cycle
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Create inducing points
    n_inducing_points = 30
    t_inducing = np.linspace(0, 1, n_inducing_points)

    # Prepare Stan data
    stan_data = {
        'D': D,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),

        'N_weight': len(df_weight),
        't_weight': df_weight['t_scaled'].values.astype(float),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),

        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': 2,  # Fourier harmonics

        'use_sparse': 1,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),
    }

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Aerobic days > 0: {(df_daily['aerobic_intensity'] > 0).sum()}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    # Compile model
    model_path = Path("stan/weight_state_space_diminishing_fixed.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    # Fit model
    print("\nFitting model (this may take a few minutes)...")
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=0.95,
        max_treedepth=10,
        show_progress=True,
        seed=12345,
    )

    # Convert to ArviZ InferenceData
    print("\nConverting to InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive='y_weight_rep',
        log_likelihood='log_lik_weight',
        coords={
            'day': np.arange(1, D + 1),
            'weight_obs': np.arange(len(df_weight)),
        },
        dims={
            'fitness_a_stored': ['day'],
            'fitness_s_stored': ['day'],
            'impulse_a_stored': ['day'],
            'impulse_s_stored': ['day'],
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Save results
    idata_path = output_dir / "inference_data.nc"
    idata.to_netcdf(str(idata_path))
    print(f"  InferenceData saved to: {idata_path}")

    # Parameter summary
    param_names = ['psi_a', 'psi_s', 'alpha_a', 'alpha_s', 'k_a', 'k_s',
                   'gamma_a', 'gamma_s', 'sigma_w', 'alpha_gp', 'rho_gp']

    param_summary = az.summary(idata, var_names=param_names, round_to=4)
    param_summary.to_csv(output_dir / "parameter_summary.csv")
    print(f"  Parameter summary saved")

    print("\nParameter estimates:")
    for param in param_names:
        if param in param_summary.index:
            mean_val = param_summary.loc[param, 'mean']
            hdi_low = param_summary.loc[param, 'hdi_3%']
            hdi_high = param_summary.loc[param, 'hdi_97%']
            print(f"  {param}: {mean_val:.4f} [{hdi_low:.4f}, {hdi_high:.4f}]")

    # Create simple visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Fitness states
    ax = axes[0]
    fitness_a_mean = idata.posterior['fitness_a_stored'].mean(dim=['chain', 'draw']).values
    fitness_s_mean = idata.posterior['fitness_s_stored'].mean(dim=['chain', 'draw']).values

    days = np.arange(1, D + 1)
    ax.plot(days, fitness_a_mean, 'b-', label='Aerobic fitness', linewidth=2)
    ax.plot(days, fitness_s_mean, 'r-', label='Strength fitness', linewidth=2)

    ax.set_xlabel('Day')
    ax.set_ylabel('Fitness (standardized)')
    ax.set_title('Fitness States with Diminishing Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Intensity vs fitness
    ax = axes[1]

    # Plot intensity (scaled for visualization)
    intensity_a = df_daily['aerobic_intensity_std'].values
    intensity_s = df_daily['strength_intensity_std'].values

    ax.plot(days, intensity_a, 'b--', alpha=0.5, label='Aerobic intensity', linewidth=1)
    ax.plot(days, intensity_s, 'r--', alpha=0.5, label='Strength intensity', linewidth=1)
    ax.plot(days, fitness_a_mean, 'b-', label='Aerobic fitness', linewidth=2)
    ax.plot(days, fitness_s_mean, 'r-', label='Strength fitness', linewidth=2)

    ax.set_xlabel('Day')
    ax.set_ylabel('Standardized value')
    ax.set_title('Intensity (dashed) vs Fitness (solid)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Parameter distributions
    ax = axes[2]

    # Extract key parameters
    key_params = ['gamma_a', 'gamma_s', 'k_a', 'k_s', 'alpha_a', 'alpha_s']
    param_means = []
    param_hdi_low = []
    param_hdi_high = []

    for param in key_params:
        if param in param_summary.index:
            param_means.append(param_summary.loc[param, 'mean'])
            param_hdi_low.append(param_summary.loc[param, 'hdi_3%'])
            param_hdi_high.append(param_summary.loc[param, 'hdi_97%'])
        else:
            param_means.append(0)
            param_hdi_low.append(0)
            param_hdi_high.append(0)

    x_pos = np.arange(len(key_params))
    ax.errorbar(x_pos, param_means,
                yerr=[np.array(param_means) - np.array(param_hdi_low),
                      np.array(param_hdi_high) - np.array(param_means)],
                fmt='o', capsize=5, capthick=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(key_params, rotation=45)
    ax.set_ylabel('Parameter value')
    ax.set_title('Key Parameter Estimates with 94% HDI')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "diminishing_model_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved to: {output_dir / 'diminishing_model_results.png'}")

    # Compare with original model
    print("\n" + "=" * 70)
    print("COMPARISON WITH ORIGINAL MODEL")
    print("=" * 70)

    print("\nOriginal model (sensitivity analysis) had:")
    print("  • High uncertainty in beta parameters (impulse→fitness conversion)")
    print("  • Weight effects centered near zero with wide credible intervals")
    print("  • Fitness explained almost no variance")

    print("\nNew model (diminishing returns) features:")
    print("  • Fixed beta = 1.0 (impulse converts 1:1 when fitness=0)")
    print("  • Diminishing returns: gain = impulse * exp(-k * current_fitness)")
    print("  • Only estimate decay, diminishing returns, and weight effects")
    print("  • Physiological priors on weight effects (aerobic negative, strength positive)")

    return 0

if __name__ == "__main__":
    sys.exit(test_diminishing_model())