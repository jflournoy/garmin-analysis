#!/usr/bin/env python3
"""Test the tightened four-fitness model with hierarchical priors."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import arviz as az
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def test_tight_model():
    """Test the tightened four-fitness model."""
    print("\n" + "=" * 70)
    print("TESTING TIGHTENED FOUR-FITNESS MODEL")
    print("=" * 70)

    # Configuration
    output_dir = Path("output/four_fitness_tight")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    chains = 4
    iter_warmup = 500
    iter_sampling = 500
    adapt_delta = 0.95
    max_treedepth = 12
    fourier_harmonics = 2

    print("Loading data...")

    # Load weight data
    df_weight = load_weight_data(data_dir)
    print(f"  Loaded {len(df_weight)} weight measurements")
    print(f"  Date range: {df_weight['timestamp'].min()} to {df_weight['timestamp'].max()}")

    # Load aerobic intensity (walking + cycling)
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['walking', 'cycling'],
        max_hr=185.0,
    )

    if len(df_intensity) == 0:
        raise ValueError("No aerobic intensity data found")

    # Sum walking and cycling
    df_intensity['aerobic_intensity'] = df_intensity.get('walking', 0) + df_intensity.get('cycling', 0)
    df_aerobic = df_intensity[['date', 'aerobic_intensity']].copy()

    print(f"  Loaded {len(df_aerobic)} days of aerobic intensity")
    print(f"  Non-zero days: {(df_aerobic['aerobic_intensity'] > 0).sum()}")

    # Load strength training intensity
    df_strength_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],
        max_hr=185.0,
    )

    if len(df_strength_intensity) > 0 and 'strength_training' in df_strength_intensity.columns:
        df_strength_daily = df_strength_intensity[['date', 'strength_training']].copy()
        df_strength_daily = df_strength_daily.rename(columns={'strength_training': 'strength_intensity'})
        print(f"  Loaded {len(df_strength_daily)} days with strength training intensity")
    else:
        print("  No strength training intensity data found")
        df_strength_daily = pd.DataFrame(columns=['date', 'strength_intensity'])

    # Create full date range
    all_dates = set()
    all_dates.update(df_weight['timestamp'].dt.date)
    all_dates.update(df_aerobic['date'].dt.date)
    if len(df_strength_daily) > 0:
        all_dates.update(df_strength_daily['date'].dt.date)

    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    D = len(date_range)

    print(f"\nDate range for analysis: {min_date} to {max_date} ({D} days)")

    # Create daily dataframe
    df_daily = pd.DataFrame({'date': date_range})

    # Merge aerobic intensity
    df_daily = pd.merge(df_daily, df_aerobic, on='date', how='left')
    df_daily['aerobic_intensity'] = df_daily['aerobic_intensity'].fillna(0.0)

    # Merge strength intensity
    if len(df_strength_daily) > 0:
        df_daily = pd.merge(df_daily, df_strength_daily, on='date', how='left')
    df_daily['strength_intensity'] = df_daily['strength_intensity'].fillna(0.0)

    # Standardize inputs but ensure non-negative intensity
    # Workout intensity should be >= 0 (0 = no workout, >0 = workout)
    # Shift by minimum so min becomes 0, then scale by std
    for col in ['aerobic_intensity', 'strength_intensity']:
        min_val = df_daily[col].min()
        std = df_daily[col].std()
        if std > 0:
            df_daily[f'{col}_std'] = (df_daily[col] - min_val) / std
        else:
            df_daily[f'{col}_std'] = df_daily[col] - min_val
        print(f"  {col}: min={min_val:.2f}, std={std:.2f} (shifted so min=0)")

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    print(f"\nWeight standardization: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}

    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    missing = df_weight['day_idx'].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} weight observations outside date range, dropping")
        df_weight = df_weight[df_weight['day_idx'].notna()]

    # Scale time to [0, 1] for daily component (not GP anymore)
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()

    if time_range > 0:
        df_weight['t_scaled'] = (df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
    else:
        df_weight['t_scaled'] = 0.0

    # Hour of day for daily cycle
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Prepare Stan data (NO GP/inducing points)
    stan_data = {
        'D': D,
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'strength_intensity': df_daily['strength_intensity_std'].values.astype(float),

        'N_weight': len(df_weight),
        't_weight': df_weight['t_scaled'].values.astype(float),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),

        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': fourier_harmonics,

        'N_pred': 0,
        't_pred': np.array([]).astype(float),
        'hour_of_day_pred': np.array([]).astype(float),
    }

    # Save standardization info
    standardization = {
        'weight_mean': float(weight_mean),
        'weight_std': float(weight_std),
        'aerobic_min': float(df_daily['aerobic_intensity'].min()),
        'aerobic_std': float(df_daily['aerobic_intensity'].std()),
        'strength_min': float(df_daily['strength_intensity'].min()),
        'strength_std': float(df_daily['strength_intensity'].std()),
        'min_time': min_time.isoformat(),
        'max_time': max_time.isoformat(),
        'date_range_start': min_date.isoformat(),
        'date_range_end': max_date.isoformat(),
        'n_days': D
    }

    with open(output_dir / "standardization.json", 'w') as f:
        json.dump(standardization, f, indent=2)

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Aerobic days > 0: {(df_daily['aerobic_intensity'] > 0).sum()}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    # Fit model
    print("\n" + "=" * 70)
    print("FITTING TIGHTENED MODEL")
    print("=" * 70)

    print(f"Configuration:")
    print(f"  Chains: {chains}")
    print(f"  Warmup iterations: {iter_warmup}")
    print(f"  Sampling iterations: {iter_sampling}")
    print(f"  Adapt delta: {adapt_delta}")
    print(f"  Max treedepth: {max_treedepth}")
    print(f"  Fourier harmonics: {fourier_harmonics}")
    print(f"  NO GP component - deterministic relationship")

    # Compile model
    model_path = Path("stan/weight_state_space_four_fitness_tight.stan")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    # Fit model
    print("\nFitting model (this may take several minutes)...")
    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        show_progress=True,
        seed=12345,
        save_warmup=False,
    )

    # Convert to ArviZ InferenceData
    print("\nConverting to InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive='y_weight_rep',
        log_likelihood='log_lik_weight',
        coords={
            'day': np.arange(1, D + 1),  # Day indices 1..D
            'weight_obs': np.arange(len(df_weight)),  # Weight observation indices
        },
        dims={
            'fitness_a_short_stored': ['day'],
            'fitness_s_short_stored': ['day'],
            'fitness_a_long_stored': ['day'],
            'fitness_s_long_stored': ['day'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Save full InferenceData
    print("\nSaving full InferenceData...")
    idata_path = output_dir / "inference_data.nc"
    idata.to_netcdf(str(idata_path))
    print(f"  InferenceData saved to: {idata_path}")

    # Save parameter summary
    param_names = [
        # Hyperparameters
        'mu_psi_short', 'sigma_psi_short', 'mu_alpha_short', 'sigma_alpha_short', 'mu_beta_short', 'sigma_beta_short',
        'mu_psi_long', 'sigma_psi_long', 'mu_alpha_long', 'sigma_alpha_long', 'mu_beta_long', 'sigma_beta_long',

        # Individual parameters
        'psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long',
        'alpha_a_short', 'alpha_s_short', 'alpha_a_long', 'alpha_s_long',
        'beta_a_short', 'beta_s_short', 'beta_a_long', 'beta_s_long',
        'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',

        # Noise and daily
        'sigma_w', 'sigma_fourier',

        # Variance proportions
        'prop_variance_a_short', 'prop_variance_s_short',
        'prop_variance_a_long', 'prop_variance_s_long',
        'prop_variance_daily',

        # Half-lives
        'half_life_a_short', 'half_life_s_short',
        'half_life_a_long', 'half_life_s_long'
    ]

    available_params = [p for p in param_names if p in idata.posterior]
    summary = az.summary(idata, var_names=available_params)
    summary.to_csv(output_dir / "parameter_summary.csv")
    print(f"  Parameter summary saved to: {output_dir / 'parameter_summary.csv'}")

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS - TIGHTENED MODEL")
    print("=" * 70)

    # Hyperparameters
    if 'mu_psi_short' in idata.posterior:
        mu_psi_short = idata.posterior['mu_psi_short'].values.mean()
        print(f"  Mean short-term impulse decay: {mu_psi_short:.3f} (should be ~0.375, fast)")

    if 'mu_psi_long' in idata.posterior:
        mu_psi_long = idata.posterior['mu_psi_long'].values.mean()
        print(f"  Mean long-term impulse decay: {mu_psi_long:.3f} (should be ~0.714, slow)")

    if 'mu_alpha_short' in idata.posterior:
        mu_alpha_short = idata.posterior['mu_alpha_short'].values.mean()
        print(f"  Mean short-term fitness decay: {mu_alpha_short:.3f} (should be ~0.5, moderate)")

    if 'mu_alpha_long' in idata.posterior:
        mu_alpha_long = idata.posterior['mu_alpha_long'].values.mean()
        print(f"  Mean long-term fitness decay: {mu_alpha_long:.3f} (should be ~0.8, slow)")

    # Weight effects
    if 'gamma_a_short' in idata.posterior:
        gamma_a_short = idata.posterior['gamma_a_short'].values.mean()
        print(f"  Gamma aerobic short-term: {gamma_a_short:.3f} (should be negative ~-0.3)")

    if 'gamma_s_short' in idata.posterior:
        gamma_s_short = idata.posterior['gamma_s_short'].values.mean()
        print(f"  Gamma strength short-term: {gamma_s_short:.3f} (should be positive ~0.2)")

    if 'gamma_a_long' in idata.posterior:
        gamma_a_long = idata.posterior['gamma_a_long'].values.mean()
        print(f"  Gamma aerobic long-term: {gamma_a_long:.3f} (should be negative ~-0.2)")

    if 'gamma_s_long' in idata.posterior:
        gamma_s_long = idata.posterior['gamma_s_long'].values.mean()
        print(f"  Gamma strength long-term: {gamma_s_long:.3f} (should be positive ~0.3)")

    # Noise
    if 'sigma_w' in idata.posterior:
        sigma_w = idata.posterior['sigma_w'].values.mean()
        print(f"  Observation noise: {sigma_w:.3f} (should be small, ~0.1)")

    # Variance proportions
    if 'prop_variance_a_short' in idata.posterior:
        prop_a_short = idata.posterior['prop_variance_a_short'].values.mean()
        prop_s_short = idata.posterior['prop_variance_s_short'].values.mean()
        prop_a_long = idata.posterior['prop_variance_a_long'].values.mean()
        prop_s_long = idata.posterior['prop_variance_s_long'].values.mean()
        prop_daily = idata.posterior['prop_variance_daily'].values.mean()

        print(f"\n  Variance proportions:")
        print(f"    Aerobic short-term: {prop_a_short:.1%}")
        print(f"    Strength short-term: {prop_s_short:.1%}")
        print(f"    Aerobic long-term: {prop_a_long:.1%}")
        print(f"    Strength long-term: {prop_s_long:.1%}")
        print(f"    Daily cycle: {prop_daily:.1%}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"\nKey improvements in tightened model:")
    print(f"  1. REMOVED GP component (was too flexible)")
    print(f"  2. Added hierarchical priors for short/long-term parameters")
    print(f"  3. Tightened priors on weight effects based on physiology")
    print(f"  4. Reduced observation noise (more deterministic relationship)")

if __name__ == "__main__":
    test_tight_model()