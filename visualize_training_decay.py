#!/usr/bin/env python3
"""Visualize training decay model predictions with confidence intervals."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import arviz as az
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_stan_data(df_weight, df_intensity):
    """Prepare Stan data for visualization."""
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
    for col in ['aerobic_intensity', 'strength_intensity']:
        min_val = df_daily[col].min()
        std = df_daily[col].std()
        if std > 0:
            df_daily[f'{col}_std'] = (df_daily[col] - min_val) / std
        else:
            df_daily[f'{col}_std'] = df_daily[col] - min_val

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Scale time to [0, 1] for GP
    min_time = df_weight['timestamp'].min()
    max_time = df_weight['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()
    if time_range > 0:
        df_weight['t_scaled'] = (df_weight['timestamp'] - min_time).dt.total_seconds() / time_range
    else:
        df_weight['t_scaled'] = 0.0

    # Hour of day
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
        'K': 2,
        'use_sparse': 1,
        'M': n_inducing_points,
        't_inducing': t_inducing.astype(float),
    }

    return stan_data, df_weight, df_daily, weight_mean, weight_std


def fit_model(stan_data, model_path):
    """Fit model and return InferenceData."""
    print(f"Compiling model: {model_path}")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=2,
        iter_warmup=300,
        iter_sampling=300,
        adapt_delta=0.9,
        max_treedepth=10,
        show_progress=True,
        seed=12345,
    )

    print("Converting to InferenceData...")
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
            'impulse_a_stored': ['day'],
            'impulse_s_stored': ['day'],
            'f_gp_stored': ['weight_obs'],
            'f_daily_stored': ['weight_obs'],
            'y_weight_rep': ['weight_obs'],
            'log_lik_weight': ['weight_obs'],
        }
    )

    return fit, idata


def create_comprehensive_plots(idata, df_daily, df_weight, weight_mean, weight_std, output_dir):
    """Create comprehensive visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract posterior samples
    posterior = idata.posterior

    # Create date range for x-axis
    dates = pd.date_range(start=df_daily['date'].min(), periods=len(df_daily), freq='D')

    # Plot 1: Intensity over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("Workout Intensity Over Time", fontsize=16)

    # Aerobic intensity
    ax = axes[0]
    ax.plot(dates, df_daily['aerobic_intensity_std'], 'b-', alpha=0.7, linewidth=1)
    ax.fill_between(dates, 0, df_daily['aerobic_intensity_std'], alpha=0.3, color='blue')
    ax.set_ylabel("Aerobic Intensity (standardized)")
    ax.set_title("Aerobic (Walking + Cycling)")
    ax.grid(True, alpha=0.3)

    # Strength intensity
    ax = axes[1]
    ax.plot(dates, df_daily['strength_intensity_std'], 'r-', alpha=0.7, linewidth=1)
    ax.fill_between(dates, 0, df_daily['strength_intensity_std'], alpha=0.3, color='red')
    ax.set_ylabel("Strength Intensity (standardized)")
    ax.set_title("Strength Training")
    ax.grid(True, alpha=0.3)

    # Combined intensity
    ax = axes[2]
    combined_intensity = df_daily['aerobic_intensity_std'] + df_daily['strength_intensity_std']
    ax.plot(dates, combined_intensity, 'g-', alpha=0.7, linewidth=1)
    ax.fill_between(dates, 0, combined_intensity, alpha=0.3, color='green')
    ax.set_ylabel("Total Intensity (standardized)")
    ax.set_title("Combined Aerobic + Strength")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "intensity_over_time.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Fitness states over time (with CIs)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Fitness States Over Time (with 90% Credible Intervals)", fontsize=16)

    # Check which fitness variables we have
    if 'fitness_a_stored' in posterior:
        # Aerobic fitness
        ax = axes[0]
        fitness_a_samples = posterior['fitness_a_stored'].values
        fitness_a_mean = np.mean(fitness_a_samples, axis=(0, 1))
        fitness_a_ci_lower = np.percentile(fitness_a_samples, 5, axis=(0, 1))
        fitness_a_ci_upper = np.percentile(fitness_a_samples, 95, axis=(0, 1))

        ax.plot(dates, fitness_a_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(dates, fitness_a_ci_lower, fitness_a_ci_upper, alpha=0.3, color='blue', label='90% CI')
        ax.set_ylabel("Aerobic Fitness (standardized)")
        ax.set_title("Aerobic Fitness State")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Strength fitness
        ax = axes[1]
        fitness_s_samples = posterior['fitness_s_stored'].values
        fitness_s_mean = np.mean(fitness_s_samples, axis=(0, 1))
        fitness_s_ci_lower = np.percentile(fitness_s_samples, 5, axis=(0, 1))
        fitness_s_ci_upper = np.percentile(fitness_s_samples, 95, axis=(0, 1))

        ax.plot(dates, fitness_s_mean, 'r-', linewidth=2, label='Mean')
        ax.fill_between(dates, fitness_s_ci_lower, fitness_s_ci_upper, alpha=0.3, color='red', label='90% CI')
        ax.set_ylabel("Strength Fitness (standardized)")
        ax.set_title("Strength Fitness State")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "fitness_states.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 3: Impulse states (if available)
    if 'impulse_a_stored' in posterior:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle("Impulse States Over Time (with 90% Credible Intervals)", fontsize=16)

        # Aerobic impulse
        ax = axes[0]
        impulse_a_samples = posterior['impulse_a_stored'].values
        impulse_a_mean = np.mean(impulse_a_samples, axis=(0, 1))
        impulse_a_ci_lower = np.percentile(impulse_a_samples, 5, axis=(0, 1))
        impulse_a_ci_upper = np.percentile(impulse_a_samples, 95, axis=(0, 1))

        ax.plot(dates, impulse_a_mean, 'b-', linewidth=2, label='Mean')
        ax.fill_between(dates, impulse_a_ci_lower, impulse_a_ci_upper, alpha=0.3, color='blue', label='90% CI')
        ax.set_ylabel("Aerobic Impulse (standardized)")
        ax.set_title("Aerobic Impulse State")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Strength impulse
        ax = axes[1]
        impulse_s_samples = posterior['impulse_s_stored'].values
        impulse_s_mean = np.mean(impulse_s_samples, axis=(0, 1))
        impulse_s_ci_lower = np.percentile(impulse_s_samples, 5, axis=(0, 1))
        impulse_s_ci_upper = np.percentile(impulse_s_samples, 95, axis=(0, 1))

        ax.plot(dates, impulse_s_mean, 'r-', linewidth=2, label='Mean')
        ax.fill_between(dates, impulse_s_ci_lower, impulse_s_ci_upper, alpha=0.3, color='red', label='90% CI')
        ax.set_ylabel("Strength Impulse (standardized)")
        ax.set_title("Strength Impulse State")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "impulse_states.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 4: Weight decomposition
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle("Weight Decomposition (with 90% Credible Intervals)", fontsize=16)

    # Get weight observation times
    weight_dates = df_weight['timestamp']

    # GP component
    if 'f_gp_stored' in posterior:
        ax = axes[0]
        f_gp_samples = posterior['f_gp_stored'].values
        f_gp_mean = np.mean(f_gp_samples, axis=(0, 1)) * weight_std
        f_gp_ci_lower = np.percentile(f_gp_samples, 5, axis=(0, 1)) * weight_std
        f_gp_ci_upper = np.percentile(f_gp_samples, 95, axis=(0, 1)) * weight_std

        ax.scatter(weight_dates, f_gp_mean, s=20, alpha=0.6, color='purple')
        ax.errorbar(weight_dates, f_gp_mean, yerr=[f_gp_mean - f_gp_ci_lower, f_gp_ci_upper - f_gp_mean],
                   fmt='none', alpha=0.3, color='purple', capsize=2)
        ax.set_ylabel("Weight due to GP (lbs)")
        ax.set_title("Gaussian Process Component (long-term trends)")
        ax.grid(True, alpha=0.3)

    # Daily cycle component
    if 'f_daily_stored' in posterior:
        ax = axes[1]
        f_daily_samples = posterior['f_daily_stored'].values
        f_daily_mean = np.mean(f_daily_samples, axis=(0, 1)) * weight_std
        f_daily_ci_lower = np.percentile(f_daily_samples, 5, axis=(0, 1)) * weight_std
        f_daily_ci_upper = np.percentile(f_daily_samples, 95, axis=(0, 1)) * weight_std

        ax.scatter(weight_dates, f_daily_mean, s=20, alpha=0.6, color='orange')
        ax.errorbar(weight_dates, f_daily_mean, yerr=[f_daily_mean - f_daily_ci_lower, f_daily_ci_upper - f_daily_mean],
                   fmt='none', alpha=0.3, color='orange', capsize=2)
        ax.set_ylabel("Weight due to daily cycle (lbs)")
        ax.set_title("Daily Cycle Component (time of day effects)")
        ax.grid(True, alpha=0.3)

    # Fitness component
    if 'fitness_a_stored' in posterior and 'fitness_s_stored' in posterior:
        ax = axes[2]
        # Get gamma parameters
        if 'gamma_a' in posterior and 'gamma_s' in posterior:
            gamma_a_samples = posterior['gamma_a'].values.flatten()
            gamma_s_samples = posterior['gamma_s'].values.flatten()

            # Calculate fitness contribution for each weight observation
            fitness_contrib = np.zeros((len(gamma_a_samples), len(df_weight)))

            for i in range(len(df_weight)):
                day_idx = df_weight['day_idx'].iloc[i] - 1  # Convert to 0-index
                # For each sample, calculate fitness contribution
                for s in range(len(gamma_a_samples)):
                    fitness_contrib[s, i] = (
                        gamma_a_samples[s] * posterior['fitness_a_stored'].values[:, :, day_idx].flatten()[s] +
                        gamma_s_samples[s] * posterior['fitness_s_stored'].values[:, :, day_idx].flatten()[s]
                    ) * weight_std

            fitness_mean = np.mean(fitness_contrib, axis=0)
            fitness_ci_lower = np.percentile(fitness_contrib, 5, axis=0)
            fitness_ci_upper = np.percentile(fitness_contrib, 95, axis=0)

            ax.scatter(weight_dates, fitness_mean, s=20, alpha=0.6, color='green')
            ax.errorbar(weight_dates, fitness_mean, yerr=[fitness_mean - fitness_ci_lower, fitness_ci_upper - fitness_mean],
                       fmt='none', alpha=0.3, color='green', capsize=2)
            ax.set_ylabel("Weight due to fitness (lbs)")
            ax.set_title("Fitness Component (exercise effects)")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "weight_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Observed vs predicted weight
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get posterior predictive samples
    if 'y_weight_rep' in posterior:
        y_rep_samples = posterior['y_weight_rep'].values
        y_rep_mean = np.mean(y_rep_samples, axis=(0, 1)) * weight_std + weight_mean
        y_rep_ci_lower = np.percentile(y_rep_samples, 5, axis=(0, 1)) * weight_std + weight_mean
        y_rep_ci_upper = np.percentile(y_rep_samples, 95, axis=(0, 1)) * weight_std + weight_mean

        # Observed weight
        y_obs = df_weight['weight_lbs'].values

        ax.scatter(weight_dates, y_obs, s=40, alpha=0.7, color='blue', label='Observed')
        ax.scatter(weight_dates, y_rep_mean, s=40, alpha=0.7, color='red', label='Predicted (mean)')
        ax.errorbar(weight_dates, y_rep_mean, yerr=[y_rep_mean - y_rep_ci_lower, y_rep_ci_upper - y_rep_mean],
                   fmt='none', alpha=0.3, color='red', capsize=3, label='90% CI')

        ax.set_xlabel("Date")
        ax.set_ylabel("Weight (lbs)")
        ax.set_title("Observed vs Predicted Weight (with 90% Credible Intervals)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation text
        correlation = np.corrcoef(y_obs, y_rep_mean)[0, 1]
        rmse = np.sqrt(np.mean((y_obs - y_rep_mean) ** 2))
        ax.text(0.02, 0.98, f"Correlation: {correlation:.3f}\nRMSE: {rmse:.2f} lbs",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_dir / "observed_vs_predicted.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 6: Parameter distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Parameter Posterior Distributions", fontsize=16)

    params_to_plot = []
    if 'alpha_d_a' in posterior:
        params_to_plot.append(('alpha_d_a', 'Aerobic decay (no training)'))
    if 'alpha_d_s' in posterior:
        params_to_plot.append(('alpha_d_s', 'Strength decay (no training)'))
    if 'alpha_m_a' in posterior:
        params_to_plot.append(('alpha_m_a', 'Aerobic training reduces decay'))
    if 'alpha_m_s' in posterior:
        params_to_plot.append(('alpha_m_s', 'Strength training reduces decay'))
    if 'beta_a' in posterior:
        params_to_plot.append(('beta_a', 'Aerobic gain per intensity'))
    if 'beta_s' in posterior:
        params_to_plot.append(('beta_s', 'Strength gain per intensity'))
    if 'gamma_a' in posterior:
        params_to_plot.append(('gamma_a', 'Aerobic weight effect'))
    if 'gamma_s' in posterior:
        params_to_plot.append(('gamma_s', 'Strength weight effect'))

    for idx, (param_name, param_title) in enumerate(params_to_plot[:6]):  # Plot first 6
        ax = axes[idx // 3, idx % 3]
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

    # Hide unused subplots
    for idx in range(len(params_to_plot), 6):
        axes[idx // 3, idx % 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"All plots saved to {output_dir}/")


def main():
    """Main function to create visualizations."""
    print("Creating comprehensive visualizations for training decay model...")

    # Load data
    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data
    df_intensity = load_intensity_by_activity(
        data_dir=data_dir,
        activity_types=['strength_training', 'walking', 'cycling'],
        max_hr=185.0,
    )

    # Prepare Stan data
    stan_data, df_weight_used, df_daily, weight_mean, weight_std = prepare_stan_data(df_weight, df_intensity)

    # Try to fit diminishing returns model first (has all components)
    model_path = Path("stan/weight_state_space_diminishing.stan")

    try:
        fit, idata = fit_model(stan_data, model_path)
        print(f"Successfully fitted {model_path.name}")

        # Create visualizations
        output_dir = "output/training_decay_visualizations"
        create_comprehensive_plots(idata, df_daily, df_weight_used, weight_mean, weight_std, output_dir)

    except Exception as e:
        print(f"Error fitting {model_path.name}: {e}")
        print("Trying simplified model...")

        # Try simplified model
        model_path = Path("stan/weight_state_space_training_decay_simple.stan")
        try:
            fit, idata = fit_model(stan_data, model_path)
            print(f"Successfully fitted {model_path.name}")

            # Create visualizations (simplified)
            output_dir = "output/training_decay_visualizations_simple"
            create_comprehensive_plots(idata, df_daily, df_weight_used, weight_mean, weight_std, output_dir)

        except Exception as e2:
            print(f"Error fitting simplified model: {e2}")
            print("Could not create visualizations.")


if __name__ == "__main__":
    main()