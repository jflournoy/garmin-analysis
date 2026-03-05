#!/usr/bin/env python3
"""Test training-dependent decay models.

This script tests the three new models:
1. Strength-only training decay model
2. Combined strength + aerobic training decay model
3. Compare with existing diminishing returns model
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import arviz as az
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity


def prepare_stan_data_for_training_decay_models(df_weight, df_intensity, include_aerobic=True, include_strength=True):
    """Prepare Stan data for training decay models.

    Args:
        df_weight: DataFrame with weight observations
        df_intensity: DataFrame with intensity by activity type (from load_intensity_by_activity)
        include_aerobic: Whether to include aerobic intensity
        include_strength: Whether to include strength intensity

    Returns:
        Dictionary with Stan data format
    """
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

    # Combine walking and cycling into aerobic if needed
    if include_aerobic:
        df_daily['aerobic_intensity'] = df_daily['walking'] + df_daily['cycling']

    if include_strength:
        df_daily['strength_intensity'] = df_daily['strength_training']

    # Standardize intensities (shift so min=0)
    # Use (value - min)/std - this gives reasonable fitness units (0-8)
    intensity_cols = []
    if include_aerobic:
        intensity_cols.append('aerobic_intensity')
    if include_strength:
        intensity_cols.append('strength_intensity')

    for col in intensity_cols:
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

    # Add intensity vectors based on which model we're using
    if include_aerobic and include_strength:
        # Combined model
        stan_data['aerobic_intensity'] = df_daily['aerobic_intensity_std'].values.astype(float)
        stan_data['strength_intensity'] = df_daily['strength_intensity_std'].values.astype(float)
    elif include_strength and not include_aerobic:
        # Strength-only model
        stan_data['strength_intensity'] = df_daily['strength_intensity_std'].values.astype(float)
    elif include_aerobic and not include_strength:
        # Aerobic-only model (not implemented yet)
        stan_data['aerobic_intensity'] = df_daily['aerobic_intensity_std'].values.astype(float)

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    if include_aerobic:
        print(f"  Aerobic days > 0: {(df_daily['aerobic_intensity'] > 0).sum()}")
    if include_strength:
        print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")

    return stan_data, df_weight, df_daily


def test_strength_only_model():
    """Test the strength-only training decay model."""
    print("Testing strength-only training decay model...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data by activity type
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training'],  # Only strength
        max_hr=185.0,
    )

    # Prepare Stan data
    stan_data, df_weight_used, df_daily = prepare_stan_data_for_training_decay_models(
        df_weight, df_intensity, include_aerobic=False, include_strength=True
    )

    # Compile and fit model
    model_path = Path("stan/weight_state_space_training_decay_strength_only.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("\nFitting model (this may take a few minutes)...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=500,
        adapt_delta=0.99,
        max_treedepth=12,
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
            'day': np.arange(1, stan_data['D'] + 1),
            'weight_obs': np.arange(stan_data['N_weight']),
        },
        dims={
            'fitness_stored': ['day'],
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Analyze results
    print("\nStrength-only model results:")
    # Check for convergence issues
    print("Checking convergence...")
    try:
        print(fit.diagnose())
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    print(f"Divergent transitions: {fit.num_divergent}")

    # Extract key parameters
    posterior = idata.posterior
    print("\nKey parameters (mean ± sd):")

    # Decay parameters
    alpha_d = posterior['alpha_d'].values.mean()
    alpha_d_sd = posterior['alpha_d'].values.std()
    print(f"alpha_d (decay without training): {alpha_d:.3f} ± {alpha_d_sd:.3f}")

    alpha_m = posterior['alpha_m'].values.mean()
    alpha_m_sd = posterior['alpha_m'].values.std()
    print(f"alpha_m (training reduces decay): {alpha_m:.3f} ± {alpha_m_sd:.3f}")

    beta = posterior['beta'].values.mean()
    beta_sd = posterior['beta'].values.std()
    print(f"beta (gain per unit intensity): {beta:.3f} ± {beta_sd:.3f}")

    gamma = posterior['gamma'].values.mean()
    gamma_sd = posterior['gamma'].values.std()
    print(f"gamma (weight effect): {gamma:.3f} ± {gamma_sd:.3f}")

    # Total decay rates
    alpha_total_no_train = alpha_d
    alpha_total_train = alpha_d + alpha_m
    print(f"\nDecay rates:")
    print(f"  Without training: {alpha_total_no_train:.3f}")
    print(f"  With training: {alpha_total_train:.3f}")
    print(f"  Training reduces decay by: {alpha_m:.3f} ({alpha_m/alpha_d*100:.1f}% of base decay)")

    return fit, idata, stan_data


def test_combined_model():
    """Test the combined strength + aerobic training decay model."""
    print("\n" + "="*60)
    print("Testing combined strength + aerobic training decay model...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data by activity type
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],  # All activities
        max_hr=185.0,
    )

    # Prepare Stan data
    stan_data, df_weight_used, df_daily = prepare_stan_data_for_training_decay_models(
        df_weight, df_intensity, include_aerobic=True, include_strength=True
    )

    # Compile and fit model
    model_path = Path("stan/weight_state_space_training_decay.stan")
    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("\nFitting model (this may take a few minutes)...")
    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=500,
        adapt_delta=0.99,
        max_treedepth=12,
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
            'day': np.arange(1, stan_data['D'] + 1),
            'weight_obs': np.arange(stan_data['N_weight']),
        },
        dims={
            'fitness_a_stored': ['day'],
            'fitness_s_stored': ['day'],
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Analyze results
    print("\nCombined model results:")
    # Check for convergence issues
    print("Checking convergence...")
    try:
        print(fit.diagnose())
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    print(f"Divergent transitions: {fit.num_divergent}")

    # Extract key parameters
    posterior = idata.posterior
    print("\nKey parameters (mean ± sd):")

    # Aerobic parameters
    alpha_d_a = posterior['alpha_d_a'].values.mean()
    alpha_d_a_sd = posterior['alpha_d_a'].values.std()
    print(f"alpha_d_a (aerobic decay without training): {alpha_d_a:.3f} ± {alpha_d_a_sd:.3f}")

    alpha_m_a = posterior['alpha_m_a'].values.mean()
    alpha_m_a_sd = posterior['alpha_m_a'].values.std()
    print(f"alpha_m_a (aerobic training reduces decay): {alpha_m_a:.3f} ± {alpha_m_a_sd:.3f}")

    beta_a = posterior['beta_a'].values.mean()
    beta_a_sd = posterior['beta_a'].values.std()
    print(f"beta_a (aerobic gain per unit intensity): {beta_a:.3f} ± {beta_a_sd:.3f}")

    gamma_a = posterior['gamma_a'].values.mean()
    gamma_a_sd = posterior['gamma_a'].values.std()
    print(f"gamma_a (aerobic weight effect): {gamma_a:.3f} ± {gamma_a_sd:.3f}")

    # Strength parameters
    alpha_d_s = posterior['alpha_d_s'].values.mean()
    alpha_d_s_sd = posterior['alpha_d_s'].values.std()
    print(f"alpha_d_s (strength decay without training): {alpha_d_s:.3f} ± {alpha_d_s_sd:.3f}")

    alpha_m_s = posterior['alpha_m_s'].values.mean()
    alpha_m_s_sd = posterior['alpha_m_s'].values.std()
    print(f"alpha_m_s (strength training reduces decay): {alpha_m_s:.3f} ± {alpha_m_s_sd:.3f}")

    beta_s = posterior['beta_s'].values.mean()
    beta_s_sd = posterior['beta_s'].values.std()
    print(f"beta_s (strength gain per unit intensity): {beta_s:.3f} ± {beta_s_sd:.3f}")

    gamma_s = posterior['gamma_s'].values.mean()
    gamma_s_sd = posterior['gamma_s'].values.std()
    print(f"gamma_s (strength weight effect): {gamma_s:.3f} ± {gamma_s_sd:.3f}")

    # Total decay rates
    print(f"\nAerobic decay rates:")
    print(f"  Without training: {alpha_d_a:.3f}")
    print(f"  With training: {alpha_d_a + alpha_m_a:.3f}")
    print(f"  Training reduces decay by: {alpha_m_a:.3f} ({alpha_m_a/alpha_d_a*100:.1f}% of base decay)")

    print(f"\nStrength decay rates:")
    print(f"  Without training: {alpha_d_s:.3f}")
    print(f"  With training: {alpha_d_s + alpha_m_s:.3f}")
    print(f"  Training reduces decay by: {alpha_m_s:.3f} ({alpha_m_s/alpha_d_s*100:.1f}% of base decay)")

    return fit, idata, stan_data


def compare_with_diminishing_returns():
    """Compare training decay model with diminishing returns model."""
    print("\n" + "="*60)
    print("Comparing with diminishing returns model...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data by activity type
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],
        max_hr=185.0,
    )

    # Prepare Stan data (same for both models)
    stan_data, df_weight_used, df_daily = prepare_stan_data_for_training_decay_models(
        df_weight, df_intensity, include_aerobic=True, include_strength=True
    )

    # Fit diminishing returns model
    print("\nFitting diminishing returns model...")
    model_path_dim = Path("stan/weight_state_space_diminishing.stan")
    model_dim = cmdstanpy.CmdStanModel(stan_file=str(model_path_dim))

    fit_dim = model_dim.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=500,
        adapt_delta=0.99,
        max_treedepth=12,
        show_progress=True,
        seed=12345,
    )

    # Convert to ArviZ InferenceData
    idata_dim = az.from_cmdstanpy(
        posterior=fit_dim,
        posterior_predictive='y_weight_rep',
        log_likelihood='log_lik_weight',
        coords={
            'day': np.arange(1, stan_data['D'] + 1),
            'weight_obs': np.arange(stan_data['N_weight']),
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

    # Fit training decay model
    print("\nFitting training decay model...")
    model_path_train = Path("stan/weight_state_space_training_decay.stan")
    model_train = cmdstanpy.CmdStanModel(stan_file=str(model_path_train))

    fit_train = model_train.sample(
        data=stan_data,
        chains=4,
        iter_warmup=500,
        iter_sampling=500,
        adapt_delta=0.99,
        max_treedepth=12,
        show_progress=True,
        seed=12346,
    )

    # Convert to ArviZ InferenceData
    idata_train = az.from_cmdstanpy(
        posterior=fit_train,
        posterior_predictive='y_weight_rep',
        log_likelihood='log_lik_weight',
        coords={
            'day': np.arange(1, stan_data['D'] + 1),
            'weight_obs': np.arange(stan_data['N_weight']),
        },
        dims={
            'fitness_a_stored': ['day'],
            'fitness_s_stored': ['day'],
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    # Compare model fit using WAIC/LOO
    print("\nModel comparison:")

    # Calculate WAIC for both models
    try:
        waic_dim = az.waic(idata_dim, pointwise=True)
        waic_train = az.waic(idata_train, pointwise=True)

        print(f"Diminishing returns model WAIC: {waic_dim.waic:.1f} ± {waic_dim.waic_se:.1f}")
        print(f"Training decay model WAIC: {waic_train.waic:.1f} ± {waic_train.waic_se:.1f}")

        # Compare
        if waic_train.waic < waic_dim.waic:
            diff = waic_dim.waic - waic_train.waic
            print(f"Training decay model is better by {diff:.1f} WAIC points")
        else:
            diff = waic_train.waic - waic_dim.waic
            print(f"Diminishing returns model is better by {diff:.1f} WAIC points")

    except Exception as e:
        print(f"Could not compute WAIC: {e}")

    return fit_dim, idata_dim, fit_train, idata_train


def create_summary_plots(fit_strength, idata_strength, fit_combined, idata_combined):
    """Create summary plots for the models."""
    print("\n" + "="*60)
    print("Creating summary plots...")

    # Create output directory
    output_dir = Path("output/training_decay_summary")
    output_dir.mkdir(exist_ok=True)

    try:
        # Plot 1: Parameter distributions for strength-only model
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Strength-Only Training Decay Model Parameters", fontsize=16)

        # alpha_d
        ax = axes[0, 0]
        az.plot_posterior(idata_strength, var_names=['alpha_d'], ax=ax)
        ax.set_title("Decay without training (alpha_d)")

        # alpha_m
        ax = axes[0, 1]
        az.plot_posterior(idata_strength, var_names=['alpha_m'], ax=ax)
        ax.set_title("Training reduces decay (alpha_m)")

        # beta
        ax = axes[1, 0]
        az.plot_posterior(idata_strength, var_names=['beta'], ax=ax)
        ax.set_title("Gain per unit intensity (beta)")

        # gamma
        ax = axes[1, 1]
        az.plot_posterior(idata_strength, var_names=['gamma'], ax=ax)
        ax.set_title("Weight effect (gamma)")

        plt.tight_layout()
        plt.savefig(output_dir / "strength_only_parameters.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot 2: Compare aerobic vs strength parameters in combined model
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Combined Model: Aerobic vs Strength Parameters", fontsize=16)

        # Decay parameters
        ax = axes[0, 0]
        az.plot_posterior(idata_combined, var_names=['alpha_d_a', 'alpha_d_s'], ax=ax)
        ax.set_title("Decay without training")
        ax.legend(['Aerobic', 'Strength'])

        # Training effect on decay
        ax = axes[0, 1]
        az.plot_posterior(idata_combined, var_names=['alpha_m_a', 'alpha_m_s'], ax=ax)
        ax.set_title("Training reduces decay")
        ax.legend(['Aerobic', 'Strength'])

        # Gain coefficients
        ax = axes[0, 2]
        az.plot_posterior(idata_combined, var_names=['beta_a', 'beta_s'], ax=ax)
        ax.set_title("Gain per unit intensity")
        ax.legend(['Aerobic', 'Strength'])

        # Weight effects
        ax = axes[1, 0]
        az.plot_posterior(idata_combined, var_names=['gamma_a', 'gamma_s'], ax=ax)
        ax.set_title("Weight effects")
        ax.legend(['Aerobic', 'Strength'])

        # Total decay rates (no training)
        ax = axes[1, 1]
        # Calculate total decay rates
        posterior = idata_combined.posterior
        alpha_total_a_no_train = posterior['alpha_d_a'].values
        alpha_total_s_no_train = posterior['alpha_d_s'].values

        # Plot distributions
        ax.hist(alpha_total_a_no_train.flatten(), bins=30, alpha=0.7, density=True, label='Aerobic')
        ax.hist(alpha_total_s_no_train.flatten(), bins=30, alpha=0.7, density=True, label='Strength')
        ax.set_xlabel("Decay rate (no training)")
        ax.set_ylabel("Density")
        ax.set_title("Total decay without training")
        ax.legend()

        # Total decay rates (with training)
        ax = axes[1, 2]
        alpha_total_a_train = posterior['alpha_d_a'].values + posterior['alpha_m_a'].values
        alpha_total_s_train = posterior['alpha_d_s'].values + posterior['alpha_m_s'].values

        ax.hist(alpha_total_a_train.flatten(), bins=30, alpha=0.7, density=True, label='Aerobic')
        ax.hist(alpha_total_s_train.flatten(), bins=30, alpha=0.7, density=True, label='Strength')
        ax.set_xlabel("Decay rate (with training)")
        ax.set_ylabel("Density")
        ax.set_title("Total decay with training")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "combined_model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to {output_dir}/")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    """Main function to run all tests."""
    print("Testing Training-Dependent Decay Models")
    print("="*60)

    try:
        # Test strength-only model
        fit_strength, idata_strength, _ = test_strength_only_model()

        # Test combined model
        fit_combined, idata_combined, _ = test_combined_model()

        # Compare with diminishing returns
        fit_dim, idata_dim, fit_train, idata_train = compare_with_diminishing_returns()

        # Create summary plots
        create_summary_plots(fit_strength, idata_strength, fit_combined, idata_combined)

        print("\n" + "="*60)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())