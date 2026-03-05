#!/usr/bin/env python3
"""Run sensitivity analysis model with weaker priors and no GP alpha constraint.

This script runs the weight_state_space_four_fitness_sensitivity.stan model
which has:
1. No hard ceiling on GP alpha (alpha_gp < 0.5 constraint removed)
2. Somewhat tight priors on SDs (not as tight as improved_v2, but tighter than weak)
3. Fitness effects centered at 0 (weaker physiological assumptions)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import arviz as az
import cmdstanpy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def run_sensitivity_analysis():
    """Run sensitivity analysis model and save results."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS MODEL")
    print("=" * 70)
    print("Model: weight_state_space_four_fitness_sensitivity.stan")
    print("Features:")
    print("  • No hard ceiling on GP alpha (alpha_gp < 0.5 constraint removed)")
    print("  • Somewhat tight priors on SDs (between improved_v2 and weak)")
    print("  • Fitness effects centered at 0 (weaker physiological assumptions)")
    print("=" * 70)

    # Configuration
    output_dir = Path("output/sensitivity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    chains = 4
    iter_warmup = 500
    iter_sampling = 500
    adapt_delta = 0.95
    max_treedepth = 12
    n_inducing_points = 50
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
        'K': fourier_harmonics,

        'use_sparse': 1,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),

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
    print("FITTING SENSITIVITY ANALYSIS MODEL")
    print("=" * 70)

    print(f"Configuration:")
    print(f"  Chains: {chains}")
    print(f"  Warmup iterations: {iter_warmup}")
    print(f"  Sampling iterations: {iter_sampling}")
    print(f"  Adapt delta: {adapt_delta}")
    print(f"  Max treedepth: {max_treedepth}")
    print(f"  Inducing points: {n_inducing_points}")
    print(f"  Fourier harmonics: {fourier_harmonics}")

    # Compile model
    model_path = Path("stan/weight_state_space_four_fitness_sensitivity.stan")
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
        save_warmup=False,  # Don't save warmup to reduce file size
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
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Check for divergent transitions
    if hasattr(fit, 'diagnose'):
        try:
            diagnose = fit.diagnose()
            # diagnose() returns a string, not a dict
            if 'divergent transitions' in diagnose:
                print(f"  ⚠️  Divergent transitions detected!")
                # Extract just the divergent transitions line
                for line in diagnose.split('\n'):
                    if 'divergent transitions' in line:
                        print(f"  {line.strip()}")
                        break
        except Exception as e:
            print(f"  ⚠️  Could not check diagnostics: {e}")

    # Save full InferenceData
    print("\nSaving full InferenceData...")
    idata_path = output_dir / "inference_data.nc"
    idata.to_netcdf(str(idata_path))
    print(f"  InferenceData saved to: {idata_path}")

    # Also save as pickle for easier loading
    pickle_path = output_dir / "inference_data.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'idata': idata,
            'df_weight': df_weight,
            'df_daily': df_daily,
            'stan_data': stan_data,
            'standardization': standardization
        }, f)
    print(f"  Pickle data saved to: {pickle_path}")

    # Save parameter summary
    param_names = [
        'psi_a_short', 'psi_s_short', 'psi_a_long', 'psi_s_long',
        'alpha_a_short', 'alpha_s_short', 'alpha_a_long', 'alpha_s_long',
        'beta_a_short', 'beta_s_short', 'beta_a_long', 'beta_s_long',
        'gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
        'sigma_w', 'alpha_gp', 'rho_gp',
        'prop_variance_a_short', 'prop_variance_s_short',
        'prop_variance_a_long', 'prop_variance_s_long',
        'prop_variance_daily', 'prop_variance_gp',
        'half_life_a_short', 'half_life_s_short',
        'half_life_a_long', 'half_life_s_long'
    ]

    # Extract parameter summary
    param_summary = az.summary(idata, var_names=param_names, round_to=4)
    param_summary.to_csv(output_dir / "parameter_summary.csv")
    print(f"  Parameter summary saved to: {output_dir / 'parameter_summary.csv'}")

    # Save key parameter estimates
    key_params = {
        'gamma_a_short': float(param_summary.loc['gamma_a_short', 'mean']),
        'gamma_s_short': float(param_summary.loc['gamma_s_short', 'mean']),
        'gamma_a_long': float(param_summary.loc['gamma_a_long', 'mean']),
        'gamma_s_long': float(param_summary.loc['gamma_s_long', 'mean']),
        'alpha_gp': float(param_summary.loc['alpha_gp', 'mean']),
        'rho_gp': float(param_summary.loc['rho_gp', 'mean']),
        'sigma_w': float(param_summary.loc['sigma_w', 'mean']),
    }

    with open(output_dir / "key_parameters.json", 'w') as f:
        json.dump(key_params, f, indent=2)

    print(f"\nKey parameter estimates:")
    print(f"  gamma_a_short (aerobic short-term): {key_params['gamma_a_short']:.4f}")
    print(f"  gamma_s_short (strength short-term): {key_params['gamma_s_short']:.4f}")
    print(f"  gamma_a_long (aerobic long-term): {key_params['gamma_a_long']:.4f}")
    print(f"  gamma_s_long (strength long-term): {key_params['gamma_s_long']:.4f}")
    print(f"  alpha_gp (GP std): {key_params['alpha_gp']:.4f}")
    print(f"  rho_gp (GP length scale): {key_params['rho_gp']:.4f}")
    print(f"  sigma_w (measurement noise): {key_params['sigma_w']:.4f}")

    # Generate basic diagnostic plots
    print("\nGenerating diagnostic plots...")
    try:
        # Trace plots for key parameters
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()

        trace_params = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
                       'alpha_gp', 'rho_gp', 'sigma_w', 'psi_a_short', 'psi_s_short']

        for i, param in enumerate(trace_params[:len(axes)]):
            if param in idata.posterior:
                az.plot_trace(idata, var_names=[param], axes=[axes[i], axes[i]], show=False)
                axes[i].set_title(f'{param} trace', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Trace plots saved to: {output_dir / 'trace_plots.png'}")

        # Posterior distributions for weight effects
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        weight_effects = ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long']

        for i, param in enumerate(weight_effects):
            ax = axes[i//2, i%2]
            if param in idata.posterior:
                az.plot_posterior(idata, var_names=[param], ax=ax, show=False)
                ax.set_title(f'{param} posterior', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / "weight_effects_posterior.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Weight effects posterior saved to: {output_dir / 'weight_effects_posterior.png'}")

    except Exception as e:
        print(f"  ⚠️  Warning generating diagnostic plots: {e}")

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"📈 Key files:")
    print(f"   • inference_data.nc - Full InferenceData (NetCDF)")
    print(f"   • inference_data.pkl - Pickle with all data")
    print(f"   • parameter_summary.csv - Parameter estimates")
    print(f"   • key_parameters.json - Key parameter values")
    print(f"   • trace_plots.png - Diagnostic trace plots")
    print(f"   • weight_effects_posterior.png - Posterior distributions")

    print(f"\n🔬 Sensitivity analysis insights:")
    print(f"   • Compare with improved_v2 model in output/four_fitness_full/")
    print(f"   • Check if alpha_gp exceeds 0.5 (constraint removed)")
    print(f"   • See if weight effects maintain physiological signs")
    print(f"   • Assess variance proportions vs improved_v2")

    print(f"\n📈 Next steps:")
    print(f"   • Compare parameter estimates with improved_v2 model")
    print(f"   • Check model diagnostics for convergence")
    print(f"   • Analyze variance proportions")
    print(f"   • Create comparison report")

    return 0


if __name__ == "__main__":
    sys.exit(run_sensitivity_analysis())