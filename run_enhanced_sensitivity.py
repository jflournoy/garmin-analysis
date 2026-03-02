#!/usr/bin/env python3
"""Run enhanced sensitivity analysis model with improved settings and predictions.

This script runs the weight_state_space_four_fitness_sensitivity_pred.stan model
with:
1. adapt_delta = 0.99 (more conservative sampling)
2. max_treedepth = 12 (deeper exploration)
3. 1000 warmup, 1000 sampling iterations
4. Good initial values based on priors
5. Enhanced prediction capabilities
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


def generate_initial_values(stan_data, chains=4):
    """Generate good initial values based on priors."""
    inits = []

    for chain in range(chains):
        init = {}

        # Set random seed for reproducibility
        np.random.seed(12345 + chain)

        # Hyperparameters - sample from priors
        init['mu_psi_short'] = np.random.beta(3, 5)
        init['sigma_psi_short'] = np.random.exponential(1/3)  # mean=0.33

        init['mu_alpha_short'] = np.random.beta(4, 4)
        init['sigma_alpha_short'] = np.random.exponential(1/3)

        init['mu_beta_short'] = np.random.exponential(1/7.5)  # mean=0.133
        init['sigma_beta_short'] = np.random.exponential(1/3)

        init['mu_psi_long'] = np.random.beta(5, 2)
        init['sigma_psi_long'] = np.random.exponential(1/3)

        init['mu_alpha_long'] = np.random.beta(6, 2)
        init['sigma_alpha_long'] = np.random.exponential(1/3)

        init['mu_beta_long'] = np.random.exponential(1/15)  # mean=0.067
        init['sigma_beta_long'] = np.random.exponential(1/3)

        # Individual parameters - centered around 0
        init['psi_a_short_raw'] = np.random.normal(0, 0.1)
        init['psi_s_short_raw'] = np.random.normal(0, 0.1)
        init['alpha_a_short_raw'] = np.random.normal(0, 0.1)
        init['alpha_s_short_raw'] = np.random.normal(0, 0.1)
        init['beta_a_short_raw'] = np.random.normal(0, 0.1)
        init['beta_s_short_raw'] = np.random.normal(0, 0.1)

        init['psi_a_long_raw'] = np.random.normal(0, 0.1)
        init['psi_s_long_raw'] = np.random.normal(0, 0.1)
        init['alpha_a_long_raw'] = np.random.normal(0, 0.1)
        init['alpha_s_long_raw'] = np.random.normal(0, 0.1)
        init['beta_a_long_raw'] = np.random.normal(0, 0.1)
        init['beta_s_long_raw'] = np.random.normal(0, 0.1)

        # Weight effects - centered at 0 with small values
        init['gamma_a_short'] = np.random.normal(0, 0.05)
        init['gamma_s_short'] = np.random.normal(0, 0.05)
        init['gamma_a_long'] = np.random.normal(0, 0.03)
        init['gamma_s_long'] = np.random.normal(0, 0.03)

        # Measurement noise - reasonable value
        init['sigma_w'] = 0.3 + np.random.normal(0, 0.05)

        # GP parameters - reasonable values
        init['alpha_gp'] = 0.4 + np.random.normal(0, 0.1)
        init['alpha_gp'] = max(0.1, min(1.0, init['alpha_gp']))  # ensure within bounds
        init['rho_gp'] = 0.2 + np.random.normal(0, 0.05)
        init['rho_gp'] = max(0.1, min(1.0, init['rho_gp']))  # ensure within bounds

        # Fourier coefficients
        K = stan_data['K']
        init['a_sin_raw'] = np.random.normal(0, 0.1, K).tolist()
        init['a_cos_raw'] = np.random.normal(0, 0.1, K).tolist()
        init['sigma_fourier'] = 0.1 + np.random.normal(0, 0.05)

        # Inducing points
        M = stan_data['M']
        init['eta_inducing_raw'] = np.random.normal(0, 0.1, M).tolist()

        inits.append(init)

    return inits


def create_prediction_grid(df_weight, df_daily, n_points=200):
    """Create a dense grid of prediction points."""
    # Time range for predictions
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()

    # Create dense time grid
    t_pred = np.linspace(0, 1, n_points)

    # Convert back to timestamps for hour calculation
    pred_timestamps = [min_time + pd.Timedelta(seconds=t * time_range) for t in t_pred]

    # Get hour of day for each prediction point
    hour_of_day_pred = np.array([ts.hour + ts.minute / 60.0 for ts in pred_timestamps])

    # Map to day indices
    date_to_idx = {date: i+1 for i, date in enumerate(df_daily['date'].dt.date)}

    # For each prediction time, find the corresponding day
    day_idx_pred = []
    for ts in pred_timestamps:
        date = ts.date()
        if date in date_to_idx:
            day_idx_pred.append(date_to_idx[date])
        else:
            # Find nearest day
            days_diff = [(abs((date - d.date()).days), idx) for d, idx in date_to_idx.items()]
            days_diff.sort()
            day_idx_pred.append(days_diff[0][1])

    return t_pred, hour_of_day_pred, np.array(day_idx_pred, dtype=int), pred_timestamps


def run_enhanced_sensitivity():
    """Run enhanced sensitivity analysis model with predictions."""
    print("\n" + "=" * 70)
    print("ENHANCED SENSITIVITY ANALYSIS WITH PREDICTIONS")
    print("=" * 70)
    print("Model: weight_state_space_four_fitness_sensitivity_pred.stan")
    print("Settings:")
    print("  • adapt_delta = 0.99 (more conservative)")
    print("  • max_treedepth = 12 (deeper exploration)")
    print("  • 1000 warmup, 1000 sampling iterations")
    print("  • Good initial values based on priors")
    print("  • Enhanced prediction capabilities")
    print("=" * 70)

    # Configuration
    output_dir = Path("output/enhanced_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data")
    chains = 4
    iter_warmup = 1000
    iter_sampling = 1000
    adapt_delta = 0.99
    max_treedepth = 12
    n_inducing_points = 50
    fourier_harmonics = 2
    n_pred_points = 200  # Dense prediction grid

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

    # Create prediction grid
    print(f"\nCreating prediction grid with {n_pred_points} points...")
    t_pred, hour_of_day_pred, day_idx_pred, pred_timestamps = create_prediction_grid(
        df_weight, df_daily, n_pred_points
    )
    print(f"  Prediction range: {pred_timestamps[0]} to {pred_timestamps[-1]}")

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

        'N_pred': n_pred_points,
        't_pred': t_pred.astype(float),
        'hour_of_day_pred': hour_of_day_pred.astype(float),
        'day_idx_pred': day_idx_pred,
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
        'n_days': D,
        'prediction_timestamps': [ts.isoformat() for ts in pred_timestamps]
    }

    with open(output_dir / "standardization.json", 'w') as f:
        json.dump(standardization, f, indent=2)

    print(f"\nStan data prepared:")
    print(f"  Days: {D}")
    print(f"  Weight observations: {len(df_weight)}")
    print(f"  Aerobic days > 0: {(df_daily['aerobic_intensity'] > 0).sum()}")
    print(f"  Strength days > 0: {(df_daily['strength_intensity'] > 0).sum()}")
    print(f"  Prediction points: {n_pred_points}")

    # Generate initial values
    print("\nGenerating initial values based on priors...")
    inits = generate_initial_values(stan_data, chains=chains)
    print(f"  Generated {len(inits)} sets of initial values")

    # Fit model
    print("\n" + "=" * 70)
    print("FITTING ENHANCED SENSITIVITY MODEL")
    print("=" * 70)

    print(f"Configuration:")
    print(f"  Chains: {chains}")
    print(f"  Warmup iterations: {iter_warmup}")
    print(f"  Sampling iterations: {iter_sampling}")
    print(f"  Adapt delta: {adapt_delta}")
    print(f"  Max treedepth: {max_treedepth}")
    print(f"  Inducing points: {n_inducing_points}")
    print(f"  Fourier harmonics: {fourier_harmonics}")
    print(f"  Prediction points: {n_pred_points}")

    # Compile model
    model_path = Path("stan/weight_state_space_four_fitness_sensitivity_pred.stan")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"\nCompiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    # Fit model with improved settings
    print("\nFitting model with improved settings (this may take 20-30 minutes)...")
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
        inits=inits,
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
            'pred_point': np.arange(n_pred_points),  # Prediction point indices
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
            'y_pred': ['pred_point'],
            'f_gp_pred': ['pred_point'],
            'f_daily_pred': ['pred_point'],
            'fitness_contrib_a_short': ['pred_point'],
            'fitness_contrib_s_short': ['pred_point'],
            'fitness_contrib_a_long': ['pred_point'],
            'fitness_contrib_s_long': ['pred_point'],
        }
    )

    print(f"\nModel fitting completed!")
    print(f"  Number of samples: {fit.num_draws_sampling}")
    print(f"  Number of chains: {fit.chains}")

    # Check for divergent transitions
    if hasattr(fit, 'diagnose'):
        try:
            diagnose = fit.diagnose()
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
            'standardization': standardization,
            'pred_timestamps': pred_timestamps,
            't_pred': t_pred,
            'hour_of_day_pred': hour_of_day_pred,
            'day_idx_pred': day_idx_pred
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
    key_params = {}
    for param in ['gamma_a_short', 'gamma_s_short', 'gamma_a_long', 'gamma_s_long',
                  'alpha_gp', 'rho_gp', 'sigma_w']:
        if param in param_summary.index:
            key_params[param] = float(param_summary.loc[param, 'mean'])

    with open(output_dir / "key_parameters.json", 'w') as f:
        json.dump(key_params, f, indent=2)

    print(f"\nKey parameter estimates:")
    for param, value in key_params.items():
        print(f"  {param}: {value:.4f}")

    # Generate prediction visualizations
    print("\nGenerating prediction visualizations...")
    try:
        # 1. Time series predictions
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Convert predictions back to original scale
        y_pred_mean = idata.posterior['y_pred'].mean(dim=['chain', 'draw']).values
        y_pred_std = idata.posterior['y_pred'].std(dim=['chain', 'draw']).values

        y_pred_original = y_pred_mean * weight_std + weight_mean
        y_pred_upper = (y_pred_mean + 1.96 * y_pred_std) * weight_std + weight_mean
        y_pred_lower = (y_pred_mean - 1.96 * y_pred_std) * weight_std + weight_mean

        # Plot 1: Full prediction with observations
        ax = axes[0]
        ax.plot(pred_timestamps, y_pred_original, 'b-', alpha=0.7, label='Prediction (mean)')
        ax.fill_between(pred_timestamps, y_pred_lower, y_pred_upper,
                       alpha=0.2, color='blue', label='95% CI')
        ax.scatter(df_weight['timestamp'], df_weight['weight_lbs'],
                  color='red', s=30, alpha=0.7, label='Observations')
        ax.set_xlabel('Date')
        ax.set_ylabel('Weight (lbs)')
        ax.set_title('Enhanced Sensitivity Model: Weight Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Components breakdown
        ax = axes[1]

        # Extract component means
        components = {
            'GP': idata.posterior['f_gp_pred'].mean(dim=['chain', 'draw']).values * weight_std,
            'Daily': idata.posterior['f_daily_pred'].mean(dim=['chain', 'draw']).values * weight_std,
            'Aerobic Short': idata.posterior['fitness_contrib_a_short'].mean(dim=['chain', 'draw']).values * weight_std,
            'Strength Short': idata.posterior['fitness_contrib_s_short'].mean(dim=['chain', 'draw']).values * weight_std,
            'Aerobic Long': idata.posterior['fitness_contrib_a_long'].mean(dim=['chain', 'draw']).values * weight_std,
            'Strength Long': idata.posterior['fitness_contrib_s_long'].mean(dim=['chain', 'draw']).values * weight_std,
        }

        # Stack plot of components
        bottom = np.zeros_like(y_pred_original)
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))

        for (label, values), color in zip(components.items(), colors):
            ax.fill_between(pred_timestamps, bottom, bottom + values,
                          alpha=0.6, label=label, color=color)
            bottom += values

        ax.set_xlabel('Date')
        ax.set_ylabel('Component Contribution (lbs)')
        ax.set_title('Model Components Breakdown')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "predictions_time_series.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Time series predictions saved")

        # 2. Component proportions
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get variance proportions
        var_props = {}
        for param in ['prop_variance_gp', 'prop_variance_daily',
                     'prop_variance_a_short', 'prop_variance_s_short',
                     'prop_variance_a_long', 'prop_variance_s_long']:
            if param in param_summary.index:
                var_props[param.replace('prop_variance_', '').replace('_', ' ').title()] = \
                    float(param_summary.loc[param, 'mean'])

        # Create pie chart
        labels = list(var_props.keys())
        sizes = list(var_props.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
              startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Variance Proportions')
        ax.axis('equal')  # Equal aspect ratio ensures pie is circular

        plt.tight_layout()
        plt.savefig(output_dir / "variance_proportions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Variance proportions saved")

        # 3. Diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Trace plots for key parameters
        trace_params = ['gamma_a_short', 'gamma_s_short', 'alpha_gp', 'sigma_w']
        for i, param in enumerate(trace_params):
            ax = axes[i//2, i%2]
            if param in idata.posterior:
                az.plot_trace(idata, var_names=[param], axes=[ax, ax], show=False)
                ax.set_title(f'{param} trace', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "diagnostic_trace_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Diagnostic trace plots saved")

    except Exception as e:
        print(f"  ⚠️  Warning generating visualizations: {e}")

    print("\n" + "=" * 70)
    print("ENHANCED SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Results saved to: {output_dir}")
    print(f"📈 Key files:")
    print(f"   • inference_data.nc - Full InferenceData (NetCDF)")
    print(f"   • inference_data.pkl - Pickle with all data")
    print(f"   • parameter_summary.csv - Parameter estimates")
    print(f"   • key_parameters.json - Key parameter values")
    print(f"   • predictions_time_series.png - Time series predictions")
    print(f"   • variance_proportions.png - Variance component pie chart")
    print(f"   • diagnostic_trace_plots.png - Diagnostic trace plots")

    print(f"\n🔬 Enhanced model features:")
    print(f"   • Smooth continuous predictions at {n_pred_points} points")
    print(f"   • Component breakdown visualization")
    print(f"   • Conservative sampling (adapt_delta=0.99)")
    print(f"   • Good initial values based on priors")
    print(f"   • Proper fitness state computation for predictions")

    print(f"\n📈 Next steps for reporting:")
    print(f"   • Use this model as the primary for the report")
    print(f"   • Compare with previous sensitivity analysis")
    print(f"   • Create comprehensive analysis report")
    print(f"   • Generate interactive visualizations if needed")

    return 0


if __name__ == "__main__":
    sys.exit(run_enhanced_sensitivity())