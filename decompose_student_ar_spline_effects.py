#!/usr/bin/env python3
"""Decompose all effects from Student-t AR Spline model over appropriate time periods."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy
import scipy.stats as stats
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.weight import load_weight_data
from src.data.intensity import load_intensity_by_activity

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def prepare_data_with_hour():
    """Prepare data with hour_of_day for spline model."""
    print("Preparing data with hour_of_day for spline model...")

    data_dir = "data"
    df_weight = load_weight_data(data_dir)

    # Load intensity data
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

    # Standardize weight
    weight_mean = df_weight['weight_lbs'].mean()
    weight_std = df_weight['weight_lbs'].std()
    df_weight['weight_std'] = (df_weight['weight_lbs'] - weight_mean) / weight_std

    # Map weight observations to days
    df_weight['date'] = df_weight['timestamp'].dt.date
    date_to_idx = {date: i+1 for i, date in enumerate(date_range.date)}
    df_weight['day_idx'] = df_weight['date'].map(date_to_idx)
    df_weight = df_weight[df_weight['day_idx'].notna()]

    # Sort by timestamp for AR(1) process
    df_weight = df_weight.sort_values('timestamp').reset_index(drop=True)

    # Extract hour of day from timestamp
    df_weight['hour_of_day'] = df_weight['timestamp'].dt.hour + df_weight['timestamp'].dt.minute / 60.0

    # Extract date and time components
    df_weight['date_only'] = df_weight['timestamp'].dt.date
    df_weight['time_of_day'] = df_weight['timestamp'].dt.time

    print(f"\nData loaded: {len(df_daily)} days, {len(df_weight)} weight observations")
    print(f"Weight: mean={weight_mean:.2f} lbs, std={weight_std:.2f} lbs")

    return df_weight, df_daily, weight_mean, weight_std, date_range


def fit_model():
    """Fit the Student-t AR Spline model."""
    print("\nFitting Student-t AR Spline model...")
    print("-"*40)

    # Prepare data
    df_weight, df_daily, weight_mean, weight_std, date_range = prepare_data_with_hour()

    K = 2  # Number of Fourier harmonics (24h and 12h cycles)
    stan_data = {
        'D': len(date_range),
        'strength_intensity': df_daily['strength_training_std'].values.astype(float),
        'aerobic_intensity': df_daily['aerobic_intensity_std'].values.astype(float),
        'N_weight': len(df_weight),
        'y_weight': df_weight['weight_std'].values.astype(float),
        'day_idx': df_weight['day_idx'].values.astype(int),
        'hour_of_day': df_weight['hour_of_day'].values.astype(float),
        'K': K,
    }

    model = cmdstanpy.CmdStanModel(
        stan_file="stan/weight_state_space_training_decay_aerobic_symmetric_student_ar_spline.stan"
    )

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
            print("WARNING: Model has divergent transitions")
        if "maximum treedepth" in diagnose:
            print("WARNING: Model hit maximum treedepth")
    except Exception as e:
        print(f"Could not run diagnose: {e}")

    return fit, df_weight, df_daily, weight_mean, weight_std, date_range, K


def extract_decomposition(fit, df_weight, df_daily, weight_mean, weight_std, K):
    """Extract all components for decomposition."""
    print("\nExtracting model components for decomposition...")

    draws_df = fit.draws_pd()

    # Extract posterior means
    results = {}
    for param in ['gamma_s', 'gamma_a', 'weight_intercept', 'nu', 'rho',
                  'sigma_epsilon', 'sigma_fourier', 'alpha_d_s', 'alpha_m_s',
                  'alpha_d_a', 'alpha_m_a', 'beta_s', 'beta_a']:
        if param in draws_df.columns:
            results[param] = draws_df[param].mean()

    # Extract Fourier coefficients
    for k in range(1, K+1):
        for coef_type in ['a_sin', 'a_cos']:
            param_name = f'{coef_type}[{k}]'
            if param_name in draws_df.columns:
                results[f'{coef_type}_{k}'] = draws_df[param_name].mean()

    # Extract fitness states (mean across samples)
    strength_fitness_cols = [col for col in draws_df.columns if col.startswith('strength_fitness_stored[')]
    aerobic_fitness_cols = [col for col in draws_df.columns if col.startswith('aerobic_fitness_stored[')]

    if strength_fitness_cols and aerobic_fitness_cols:
        # Sort by day index
        strength_fitness_cols_sorted = sorted(strength_fitness_cols,
                                             key=lambda x: int(x.split('[')[1].split(']')[0]))
        aerobic_fitness_cols_sorted = sorted(aerobic_fitness_cols,
                                           key=lambda x: int(x.split('[')[1].split(']')[0]))

        results['strength_fitness'] = np.array([draws_df[col].mean() for col in strength_fitness_cols_sorted])
        results['aerobic_fitness'] = np.array([draws_df[col].mean() for col in aerobic_fitness_cols_sorted])

    # Extract spline component
    spline_cols = [col for col in draws_df.columns if col.startswith('f_daily_stored[')]
    if spline_cols:
        spline_cols_sorted = sorted(spline_cols,
                                   key=lambda x: int(x.split('[')[1].split(']')[0]))
        results['f_daily'] = np.array([draws_df[col].mean() for col in spline_cols_sorted])

    # Extract AR(1) innovations
    epsilon_cols = [col for col in draws_df.columns if col.startswith('epsilon_stored[')]
    if epsilon_cols:
        epsilon_cols_sorted = sorted(epsilon_cols,
                                    key=lambda x: int(x.split('[')[1].split(']')[0]))
        results['epsilon'] = np.array([draws_df[col].mean() for col in epsilon_cols_sorted])

    # Extract linear predictors
    mu_no_ar_cols = [col for col in draws_df.columns if col.startswith('mu_no_ar_stored[')]
    if mu_no_ar_cols:
        mu_no_ar_cols_sorted = sorted(mu_no_ar_cols,
                                     key=lambda x: int(x.split('[')[1].split(']')[0]))
        results['mu_no_ar'] = np.array([draws_df[col].mean() for col in mu_no_ar_cols_sorted])

    # Extract total mu (mu_no_ar + epsilon)
    mu_total_cols = [col for col in draws_df.columns if col.startswith('mu_total[')]
    if mu_total_cols:
        mu_total_cols_sorted = sorted(mu_total_cols,
                                     key=lambda x: int(x.split('[')[1].split(']')[0]))
        results['mu_total'] = np.array([draws_df[col].mean() for col in mu_total_cols_sorted])

    # Extract residuals
    residual_cols = [col for col in draws_df.columns if col.startswith('residual[')]
    if residual_cols:
        residual_cols_sorted = sorted(residual_cols,
                                     key=lambda x: int(x.split('[')[1].split(']')[0]))
        results['residuals'] = np.array([draws_df[col].mean() for col in residual_cols_sorted])

    # Extract decay rates
    alpha_total_s_cols = [col for col in draws_df.columns if col.startswith('alpha_total_s[')]
    alpha_total_a_cols = [col for col in draws_df.columns if col.startswith('alpha_total_a[')]

    if alpha_total_s_cols and alpha_total_a_cols:
        alpha_total_s_cols_sorted = sorted(alpha_total_s_cols,
                                          key=lambda x: int(x.split('[')[1].split(']')[0]))
        alpha_total_a_cols_sorted = sorted(alpha_total_a_cols,
                                          key=lambda x: int(x.split('[')[1].split(']')[0]))

        results['alpha_total_s'] = np.array([draws_df[col].mean() for col in alpha_total_s_cols_sorted])
        results['alpha_total_a'] = np.array([draws_df[col].mean() for col in alpha_total_a_cols_sorted])

    return results


def create_comprehensive_decomposition_plots(results, df_weight, df_daily, weight_mean, weight_std, date_range, K):
    """Create comprehensive decomposition plots for all model effects."""
    print("\nCreating comprehensive decomposition plots...")

    output_dir = Path("docs/student_ar_spline_decomposition")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. FITNESS STATES OVER TIME (Daily scale)
    print("Creating fitness states plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Fitness States Over Time (Daily Scale)', fontsize=16, fontweight='bold')

    # Plot strength fitness
    ax = axes[0]
    ax.plot(date_range, results['strength_fitness'], 'b-', linewidth=2, label='Strength Fitness')
    ax.fill_between(date_range,
                    results['strength_fitness'] - np.std(results['strength_fitness']),
                    results['strength_fitness'] + np.std(results['strength_fitness']),
                    alpha=0.3, color='blue')
    ax.set_ylabel('Fitness (standardized)')
    ax.set_title('Strength Fitness State')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot aerobic fitness
    ax = axes[1]
    ax.plot(date_range, results['aerobic_fitness'], 'g-', linewidth=2, label='Aerobic Fitness')
    ax.fill_between(date_range,
                    results['aerobic_fitness'] - np.std(results['aerobic_fitness']),
                    results['aerobic_fitness'] + np.std(results['aerobic_fitness']),
                    alpha=0.3, color='green')
    ax.set_ylabel('Fitness (standardized)')
    ax.set_title('Aerobic Fitness State')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot intensity inputs
    ax = axes[2]
    ax.plot(date_range, df_daily['strength_training_std'], 'b-', alpha=0.7, label='Strength Intensity')
    ax.plot(date_range, df_daily['aerobic_intensity_std'], 'g-', alpha=0.7, label='Aerobic Intensity')
    ax.set_xlabel('Date')
    ax.set_ylabel('Intensity (standardized)')
    ax.set_title('Training Intensity Inputs')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'fitness_states_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. WEIGHT EFFECT DECOMPOSITION (Daily scale)
    print("Creating weight effect decomposition plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Weight Effect Decomposition (Daily Scale)', fontsize=16, fontweight='bold')

    # Calculate daily contributions (using fitness states)
    strength_effect = results['gamma_s'] * results['strength_fitness'][df_weight['day_idx'].values - 1]
    aerobic_effect = results['gamma_a'] * results['aerobic_fitness'][df_weight['day_idx'].values - 1]
    intercept_effect = np.full(len(df_weight), results['weight_intercept'])

    # Plot strength effect on weight
    ax = axes[0]
    # Group by date for daily aggregation
    daily_strength = pd.DataFrame({
        'date': df_weight['date_only'],
        'effect': strength_effect
    }).groupby('date').mean()

    ax.plot(daily_strength.index, daily_strength['effect'] * weight_std, 'b-', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Weight Effect (lbs)')
    ax.set_title(f'Strength Effect on Weight (γ_s = {results["gamma_s"]:.3f})')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot aerobic effect on weight
    ax = axes[1]
    daily_aerobic = pd.DataFrame({
        'date': df_weight['date_only'],
        'effect': aerobic_effect
    }).groupby('date').mean()

    ax.plot(daily_aerobic.index, daily_aerobic['effect'] * weight_std, 'g-', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Weight Effect (lbs)')
    ax.set_title(f'Aerobic Effect on Weight (γ_a = {results["gamma_a"]:.3f})')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot total fitness effect
    ax = axes[2]
    total_fitness_effect = strength_effect + aerobic_effect
    daily_total = pd.DataFrame({
        'date': df_weight['date_only'],
        'effect': total_fitness_effect
    }).groupby('date').mean()

    ax.plot(daily_total.index, daily_total['effect'] * weight_std, 'purple', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight Effect (lbs)')
    ax.set_title('Total Fitness Effect on Weight (Strength + Aerobic)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'weight_effect_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. DAILY SPLINE COMPONENT (Intraday scale)
    print("Creating daily spline decomposition plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Daily Spline Component Decomposition (Intraday Scale)', fontsize=16, fontweight='bold')

    # Plot spline vs hour of day
    ax = axes[0, 0]
    ax.scatter(df_weight['hour_of_day'], results['f_daily'] * weight_std, alpha=0.6, s=20)

    # Sort by hour for smoother plot
    hour_sorted_idx = np.argsort(df_weight['hour_of_day'].values)
    ax.plot(df_weight['hour_of_day'].values[hour_sorted_idx],
            results['f_daily'][hour_sorted_idx] * weight_std, 'r-', alpha=0.7, linewidth=2)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight Variation (lbs)')
    ax.set_title('Daily Spline Component')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot spline component by day of week
    ax = axes[0, 1]
    df_weight['day_of_week'] = df_weight['timestamp'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for i, day_name in enumerate(day_names):
        day_mask = df_weight['day_of_week'] == i
        if day_mask.any():
            ax.scatter(df_weight.loc[day_mask, 'hour_of_day'],
                      results['f_daily'][day_mask] * weight_std,
                      alpha=0.6, s=20, label=day_name)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight Variation (lbs)')
    ax.set_title('Spline by Day of Week')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot Fourier basis functions
    ax = axes[1, 0]
    hours = np.linspace(0, 24, 100)
    hour_scaled = hours / 24.0

    total_spline = np.zeros_like(hours)
    for k in range(1, K+1):
        freq = 2.0 * np.pi * k
        sin_component = results[f'a_sin_{k}'] * np.sin(freq * hour_scaled)
        cos_component = results[f'a_cos_{k}'] * np.cos(freq * hour_scaled)
        component = sin_component + cos_component

        ax.plot(hours, component * weight_std, alpha=0.7, linewidth=1.5,
               label=f'Harmonic {k} ({24/k:.0f}h cycle)')
        total_spline += component

    ax.plot(hours, total_spline * weight_std, 'k-', linewidth=2.5, label='Total Spline')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Weight Variation (lbs)')
    ax.set_title('Fourier Basis Functions')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Plot spline magnitude distribution
    ax = axes[1, 1]
    ax.hist(results['f_daily'] * weight_std, bins=30, density=True, alpha=0.7,
           color='orange', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Spline Component (lbs)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Spline Values')
    ax.grid(True, alpha=0.3)

    # Add statistics
    spline_mean = np.mean(results['f_daily'] * weight_std)
    spline_std = np.std(results['f_daily'] * weight_std)
    ax.text(0.05, 0.95, f'Mean: {spline_mean:.3f} lbs\nStd: {spline_std:.3f} lbs',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'daily_spline_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. AR(1) PROCESS DECOMPOSITION (Observation scale)
    print("Creating AR(1) process decomposition plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AR(1) Process Decomposition (Observation Scale)', fontsize=16, fontweight='bold')

    # Plot AR(1) innovations over time
    ax = axes[0, 0]
    ax.plot(df_weight['timestamp'], results['epsilon'] * weight_std, 'b-', alpha=0.7, linewidth=1)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('AR(1) Innovation (lbs)')
    ax.set_title(f'AR(1) Innovations (ρ = {results["rho"]:.3f})')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot autocorrelation function
    ax = axes[0, 1]
    max_lag = min(50, len(results['epsilon']) - 1)  # Need at least 1 observation for correlation
    acf = np.zeros(max_lag)

    for lag in range(max_lag):
        if lag > 0 and lag < len(results['epsilon']):
            x = results['epsilon'][:-lag]
            y = results['epsilon'][lag:]
            if len(x) > 1 and len(y) > 1:  # Need at least 2 points for correlation
                acf[lag] = np.corrcoef(x, y)[0, 1]

    ax.bar(range(max_lag), acf, alpha=0.7, color='green')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=results['rho'], color='red', linestyle='--', alpha=0.7,
              label=f'Estimated ρ = {results["rho"]:.3f}')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function of Innovations')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot innovation distribution
    ax = axes[1, 0]
    ax.hist(results['epsilon'] * weight_std, bins=30, density=True, alpha=0.7,
           color='purple', edgecolor='black')

    # Overlay normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    normal_pdf = stats.norm.pdf(x, 0, np.std(results['epsilon'] * weight_std))
    ax.plot(x, normal_pdf, 'k-', linewidth=2, label='Normal fit')

    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Innovation (lbs)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of AR(1) Innovations')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot innovations vs predicted values
    ax = axes[1, 1]
    ax.scatter(results['mu_no_ar'] * weight_std, results['epsilon'] * weight_std,
              alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Linear Predictor (without AR) (lbs)')
    ax.set_ylabel('AR(1) Innovation (lbs)')
    ax.set_title('Innovations vs Linear Predictor')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ar_process_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. STUDENT-T DISTRIBUTION ANALYSIS (Observation scale)
    print("Creating Student-t distribution analysis plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Student-t Distribution Analysis (Observation Scale)', fontsize=16, fontweight='bold')

    # Plot residuals distribution with Student-t fit
    ax = axes[0, 0]
    residuals_lbs = results['residuals'] * weight_std

    ax.hist(residuals_lbs, bins=30, density=True, alpha=0.7,
           color='teal', edgecolor='black', label='Residuals')

    # Fit Student-t distribution
    df = results['nu']
    scale = np.std(residuals_lbs) * np.sqrt((df - 2) / df) if df > 2 else np.std(residuals_lbs)

    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    t_pdf = stats.t.pdf(x, df, loc=0, scale=scale)
    normal_pdf = stats.norm.pdf(x, 0, np.std(residuals_lbs))

    ax.plot(x, t_pdf, 'r-', linewidth=2, label=f'Student-t (ν={df:.1f})')
    ax.plot(x, normal_pdf, 'k--', linewidth=2, label='Normal')

    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residuals (lbs)')
    ax.set_ylabel('Density')
    ax.set_title(f'Residual Distribution (ν = {df:.1f})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot QQ-plot for Student-t
    ax = axes[0, 1]
    stats.probplot(residuals_lbs, dist=stats.t, sparams=(df,), plot=ax)
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[0].set_alpha(0.6)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linewidth(2)
    ax.set_title(f'QQ-plot vs Student-t (ν={df:.1f})')
    ax.grid(True, alpha=0.3)

    # Plot degrees of freedom posterior
    ax = axes[1, 0]
    # We need to get the full posterior of nu from the fit
    # For now, show the mean with uncertainty
    ax.axvline(df, color='red', linewidth=3, label=f'ν = {df:.1f}')
    ax.axvline(30, color='black', linestyle=':', linewidth=2,
              label='ν=30 (≈normal)', alpha=0.7)
    ax.axvline(4, color='blue', linestyle=':', linewidth=2,
              label='ν=4 (heavy-tailed)', alpha=0.7)

    ax.set_xlabel('Degrees of freedom (ν)')
    ax.set_ylabel('Density')
    ax.set_title('Student-t Degrees of Freedom')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add interpretation
    if df > 30:
        tail_text = "Effectively normal"
    elif df > 10:
        tail_text = "Slightly heavy-tailed"
    elif df > 4:
        tail_text = "Moderately heavy-tailed"
    else:
        tail_text = "Very heavy-tailed"

    ax.text(0.05, 0.95, tail_text, transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot residuals vs fitted values
    ax = axes[1, 1]
    fitted_lbs = results['mu_total'] * weight_std

    ax.scatter(fitted_lbs, residuals_lbs, alpha=0.6, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fitted Values (lbs)')
    ax.set_ylabel('Residuals (lbs)')
    ax.set_title('Residuals vs Fitted Values')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'student_t_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 6. COMPLETE MODEL DECOMPOSITION (All components together)
    print("Creating complete model decomposition plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Complete Model Decomposition (Observation Scale)', fontsize=16, fontweight='bold')

    # Use ALL observations for the complete decomposition plot
    sample_idx = np.arange(len(df_weight))  # All indices

    sample_dates = df_weight['timestamp']
    sample_obs = df_weight['weight_std'] * weight_std

    # Decompose predictions for sample
    sample_strength = strength_effect[sample_idx] * weight_std
    sample_aerobic = aerobic_effect[sample_idx] * weight_std
    sample_intercept = intercept_effect[sample_idx] * weight_std
    sample_spline = results['f_daily'][sample_idx] * weight_std
    sample_epsilon = results['epsilon'][sample_idx] * weight_std
    sample_pred = results['mu_total'][sample_idx] * weight_std

    # Stacked area plot of components
    ax = axes[0]
    components = np.vstack([
        sample_intercept,
        sample_strength,
        sample_aerobic,
        sample_spline,
        sample_epsilon
    ])

    labels = ['Intercept', 'Strength Effect', 'Aerobic Effect', 'Daily Spline', 'AR(1) Innovation']
    colors = ['gray', 'blue', 'green', 'orange', 'purple']

    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, sample_obs, 'o', markersize=4, alpha=0.2, color='gray', label='Observed Weight (all)')
    # Plot stacked components
    ax.stackplot(sample_dates, components, labels=labels, colors=colors, alpha=0.7)
    ax.plot(sample_dates, sample_pred, 'k-', linewidth=2, label='Total Prediction')

    ax.set_ylabel('Weight (lbs)')
    ax.set_title('Model Component Decomposition (Stacked)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Bar plot of component magnitudes
    ax = axes[1]
    component_means = np.array([
        np.abs(sample_intercept).mean(),
        np.abs(sample_strength).mean(),
        np.abs(sample_aerobic).mean(),
        np.abs(sample_spline).mean(),
        np.abs(sample_epsilon).mean()
    ])

    component_stds = np.array([
        np.std(sample_intercept),
        np.std(sample_strength),
        np.std(sample_aerobic),
        np.std(sample_spline),
        np.std(sample_epsilon)
    ])

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, component_means, yerr=component_stds,
                 alpha=0.7, color=colors, edgecolor='black', capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Absolute Magnitude (lbs)')
    ax.set_title('Average Component Magnitudes')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, mean_val in zip(bars, component_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)

    # Component correlations
    ax = axes[2]
    all_components = np.column_stack([
        sample_strength,
        sample_aerobic,
        sample_spline,
        sample_epsilon,
        sample_obs - sample_pred  # residuals
    ])

    component_names = ['Strength', 'Aerobic', 'Spline', 'AR(1)', 'Residuals']
    corr_matrix = np.corrcoef(all_components.T)

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(component_names)))
    ax.set_yticks(np.arange(len(component_names)))
    ax.set_xticklabels(component_names)
    ax.set_yticklabels(component_names)

    # Add correlation values
    for i in range(len(component_names)):
        for j in range(len(component_names)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                   ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                   fontsize=8, fontweight='bold')

    ax.set_title('Component Correlation Matrix')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'complete_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 7. DECAY RATE ANALYSIS (Daily scale)
    print("Creating decay rate analysis plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fitness Decay Rate Analysis (Daily Scale)', fontsize=16, fontweight='bold')

    # Plot decay rates over time
    ax = axes[0, 0]
    trained_days_s = np.where(df_daily['strength_training'] > 0)[0]
    untrained_days_s = np.where(df_daily['strength_training'] == 0)[0]

    ax.scatter(date_range[trained_days_s], results['alpha_total_s'][trained_days_s],
              color='blue', alpha=0.7, s=30, label='Training days')
    ax.scatter(date_range[untrained_days_s], results['alpha_total_s'][untrained_days_s],
              color='lightblue', alpha=0.5, s=20, label='Rest days')

    ax.axhline(y=results['alpha_d_s'], color='red', linestyle='--',
              label=f'Base decay: {results["alpha_d_s"]:.3f}')
    ax.axhline(y=results['alpha_d_s'] + (1 - results['alpha_d_s']) * results['alpha_m_s'],
              color='darkred', linestyle=':',
              label=f'Max with training: {results["alpha_d_s"] + (1 - results["alpha_d_s"]) * results["alpha_m_s"]:.3f}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Decay Rate')
    ax.set_title('Strength Fitness Decay Rates')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax = axes[0, 1]
    trained_days_a = np.where(df_daily['aerobic_intensity'] > 0)[0]
    untrained_days_a = np.where(df_daily['aerobic_intensity'] == 0)[0]

    ax.scatter(date_range[trained_days_a], results['alpha_total_a'][trained_days_a],
              color='green', alpha=0.7, s=30, label='Training days')
    ax.scatter(date_range[untrained_days_a], results['alpha_total_a'][untrained_days_a],
              color='lightgreen', alpha=0.5, s=20, label='Rest days')

    ax.axhline(y=results['alpha_d_a'], color='red', linestyle='--',
              label=f'Base decay: {results["alpha_d_a"]:.3f}')
    ax.axhline(y=results['alpha_d_a'] + (1 - results['alpha_d_a']) * results['alpha_m_a'],
              color='darkred', linestyle=':',
              label=f'Max with training: {results["alpha_d_a"] + (1 - results["alpha_d_a"]) * results["alpha_m_a"]:.3f}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Decay Rate')
    ax.set_title('Aerobic Fitness Decay Rates')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot decay rate distributions
    ax = axes[1, 0]
    ax.hist(results['alpha_total_s'][trained_days_s], bins=20, alpha=0.7,
           color='blue', density=True, label='Training days')
    ax.hist(results['alpha_total_s'][untrained_days_s], bins=20, alpha=0.7,
           color='lightblue', density=True, label='Rest days')
    ax.set_xlabel('Decay Rate')
    ax.set_ylabel('Density')
    ax.set_title('Strength Decay Rate Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.hist(results['alpha_total_a'][trained_days_a], bins=20, alpha=0.7,
           color='green', density=True, label='Training days')
    ax.hist(results['alpha_total_a'][untrained_days_a], bins=20, alpha=0.7,
           color='lightgreen', density=True, label='Rest days')
    ax.set_xlabel('Decay Rate')
    ax.set_ylabel('Density')
    ax.set_title('Aerobic Decay Rate Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'decay_rate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 8. CUMULATIVE PREDICTION BUILD-UP (Observation scale)
    print("Creating cumulative prediction build-up plots...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Cumulative Prediction Build-Up: Adding Each Component', fontsize=16, fontweight='bold')

    # Calculate cumulative predictions
    intercept_only = intercept_effect * weight_std
    intercept_strength = (intercept_effect + strength_effect) * weight_std
    intercept_aerobic = (intercept_effect + aerobic_effect) * weight_std
    intercept_strength_aerobic = (intercept_effect + strength_effect + aerobic_effect) * weight_std
    intercept_daily = (intercept_effect + results['f_daily']) * weight_std
    intercept_strength_aerobic_daily = (intercept_effect + strength_effect + aerobic_effect + results['f_daily']) * weight_std
    full_prediction = results['mu_total'] * weight_std  # Includes AR(1) component
    observed_weight = df_weight['weight_std'] * weight_std

    # Use ALL observations for the cumulative prediction plots
    sample_idx = np.arange(len(df_weight))  # All indices
    sample_dates = df_weight['timestamp']

    # Plot 1: Intercept only
    ax = axes[0, 0]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model prediction in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=2, label='Intercept only')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('1. Intercept Only (Baseline)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Intercept + Strength
    ax = axes[0, 1]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model predictions in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=1, alpha=0.5, label='Intercept')
    ax.plot(sample_dates, intercept_strength[sample_idx], 'blue', linewidth=2, label='+ Strength effect')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('2. Intercept + Strength Effect')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Intercept + Aerobic
    ax = axes[1, 0]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model predictions in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=1, alpha=0.5, label='Intercept')
    ax.plot(sample_dates, intercept_aerobic[sample_idx], 'green', linewidth=2, label='+ Aerobic effect')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('3. Intercept + Aerobic Effect')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Intercept + Strength + Aerobic
    ax = axes[1, 1]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model predictions in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=1, alpha=0.5, label='Intercept')
    ax.plot(sample_dates, intercept_strength[sample_idx], 'blue', linewidth=1, alpha=0.5, label='+ Strength')
    ax.plot(sample_dates, intercept_strength_aerobic[sample_idx], 'purple', linewidth=2, label='+ Aerobic')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('4. Intercept + Strength + Aerobic')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 5: Intercept + Daily Spline
    ax = axes[2, 0]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model predictions in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=1, alpha=0.5, label='Intercept')
    ax.plot(sample_dates, intercept_daily[sample_idx], 'orange', linewidth=2, label='+ Daily spline')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('5. Intercept + Daily Spline')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 6: Intercept + Strength + Aerobic + Daily Spline
    ax = axes[2, 1]
    # Plot all observed points in background with high transparency
    ax.plot(sample_dates, observed_weight[sample_idx], 'o', markersize=4, alpha=0.2, color='gray', label='Observed (all)')
    # Plot model predictions in foreground
    ax.plot(sample_dates, intercept_only[sample_idx], 'gray', linewidth=1, alpha=0.5, label='Intercept')
    ax.plot(sample_dates, intercept_strength_aerobic[sample_idx], 'purple', linewidth=1, alpha=0.5, label='+ Strength+Aerobic')
    ax.plot(sample_dates, intercept_strength_aerobic_daily[sample_idx], 'brown', linewidth=2, label='+ Daily spline')
    ax.plot(sample_dates, full_prediction[sample_idx], 'black', linewidth=2, linestyle='--', label='Full prediction (with AR)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Weight (lbs)')
    ax.set_title('6. Intercept + Strength + Aerobic + Daily Spline\n(Full model without AR)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_prediction_buildup.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 9. COMPONENT CONTRIBUTION TO PREDICTION ERROR (Observation scale)
    print("Creating component contribution to prediction error plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Component Contribution to Prediction Error', fontsize=16, fontweight='bold')

    # Calculate prediction errors for each cumulative model
    error_intercept = observed_weight - intercept_only
    error_intercept_strength = observed_weight - intercept_strength
    error_intercept_aerobic = observed_weight - intercept_aerobic
    error_intercept_strength_aerobic = observed_weight - intercept_strength_aerobic
    error_intercept_daily = observed_weight - intercept_daily
    error_intercept_strength_aerobic_daily = observed_weight - intercept_strength_aerobic_daily
    error_full = observed_weight - full_prediction

    # Plot 1: Error reduction by component
    ax = axes[0, 0]
    error_reductions = [
        np.std(error_intercept),
        np.std(error_intercept_strength),
        np.std(error_intercept_aerobic),
        np.std(error_intercept_strength_aerobic),
        np.std(error_intercept_daily),
        np.std(error_intercept_strength_aerobic_daily),
        np.std(error_full)
    ]

    models = ['Intercept', '+Strength', '+Aerobic', '+Both', '+Daily', '+All (no AR)', 'Full (with AR)']
    colors = ['gray', 'blue', 'green', 'purple', 'orange', 'brown', 'black']

    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, error_reductions, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Prediction Error Std Dev (lbs)')
    ax.set_title('Prediction Error Reduction by Component')
    ax.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, error_val in zip(bars, error_reductions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{error_val:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Cumulative error reduction
    ax = axes[0, 1]
    baseline_error = np.std(error_intercept)
    relative_reductions = [(baseline_error - err) / baseline_error * 100 for err in error_reductions]

    ax.plot(x_pos, relative_reductions, 'o-', color='red', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Error Reduction (%)')
    ax.set_title('Cumulative Error Reduction Relative to Intercept')
    ax.grid(True, alpha=0.3)

    # Add values on points
    for i, reduction in enumerate(relative_reductions):
        ax.text(i, reduction + 1, f'{reduction:.1f}%', ha='center', va='bottom', fontsize=8)

    # Plot 3: Component contribution to total explained variance
    ax = axes[1, 0]
    total_variance = np.var(observed_weight)
    explained_by_intercept = total_variance - np.var(error_intercept)
    explained_by_strength = np.var(error_intercept) - np.var(error_intercept_strength)
    explained_by_aerobic = np.var(error_intercept_strength) - np.var(error_intercept_strength_aerobic)
    explained_by_daily = np.var(error_intercept_strength_aerobic) - np.var(error_intercept_strength_aerobic_daily)
    explained_by_ar = np.var(error_intercept_strength_aerobic_daily) - np.var(error_full)

    contributions = [explained_by_intercept, explained_by_strength, explained_by_aerobic,
                     explained_by_daily, explained_by_ar]
    contribution_labels = ['Intercept', 'Strength', 'Aerobic', 'Daily Spline', 'AR(1)']
    contribution_colors = ['gray', 'blue', 'green', 'orange', 'purple']

    wedges, texts, autotexts = ax.pie(contributions, labels=contribution_labels, colors=contribution_colors,
                                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax.set_title('Variance Explained by Each Component')

    # Plot 4: Prediction error distributions
    ax = axes[1, 1]
    error_distributions = [error_intercept, error_intercept_strength_aerobic,
                          error_intercept_strength_aerobic_daily, error_full]
    error_labels = ['Intercept only', '+Strength+Aerobic', '+Daily Spline', 'Full model']
    error_colors = ['gray', 'purple', 'brown', 'black']

    for err, label, color in zip(error_distributions, error_labels, error_colors):
        ax.hist(err, bins=20, density=True, alpha=0.5, label=label, color=color, edgecolor='black')

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Prediction Error (lbs)')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Error Distributions')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'component_contribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll decomposition plots saved to: {output_dir}/")
    return output_dir


def create_summary_report(results, output_dir, weight_std):
    """Create a summary report of the decomposition."""
    print("\nCreating summary report...")

    report_path = output_dir / 'decomposition_summary.md'
    with open(report_path, 'w') as f:
        f.write("# Student-t AR Spline Model Decomposition Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Overview\n\n")
        f.write("The Student-t AR Spline model decomposes weight variations into:\n")
        f.write("1. **Fitness states** (daily scale): Strength and aerobic fitness accumulation/decay\n")
        f.write("2. **Weight effects** (daily scale): Contributions from fitness to weight\n")
        f.write("3. **Daily spline** (intraday scale): 24h and 12h cycles in weight\n")
        f.write("4. **AR(1) process** (observation scale): Temporal correlation in residuals\n")
        f.write("5. **Student-t distribution** (observation scale): Robustness to outliers\n\n")

        f.write("## Key Parameters\n\n")
        f.write("| Parameter | Value | Description |\n")
        f.write("|-----------|-------|-------------|\n")
        f.write(f"| γ_s (strength effect) | {results['gamma_s']:.3f} | Effect of strength fitness on weight |\n")
        f.write(f"| γ_a (aerobic effect) | {results['gamma_a']:.3f} | Effect of aerobic fitness on weight |\n")
        f.write(f"| Weight intercept | {results['weight_intercept']:.3f} | Baseline weight level |\n")
        f.write(f"| ν (degrees of freedom) | {results['nu']:.1f} | Student-t tail heaviness |\n")
        f.write(f"| ρ (AR(1) correlation) | {results['rho']:.3f} | Temporal autocorrelation |\n")
        f.write(f"| σ_ε (innovation std) | {results['sigma_epsilon']:.3f} | AR(1) innovation scale |\n\n")

        f.write("## Fitness Decay Parameters\n\n")
        f.write("| Parameter | Strength | Aerobic |\n")
        f.write("|-----------|----------|---------|\n")
        f.write(f"| α_d (base decay) | {results['alpha_d_s']:.3f} | {results['alpha_d_a']:.3f} |\n")
        f.write(f"| α_m (training effect) | {results['alpha_m_s']:.3f} | {results['alpha_m_a']:.3f} |\n")
        f.write(f"| β (gain coefficient) | {results['beta_s']:.3f} | {results['beta_a']:.3f} |\n\n")

        f.write("## Component Magnitudes (in lbs)\n\n")
        f.write("| Component | Mean Abs Value | Std Dev |\n")
        f.write("|-----------|----------------|---------|\n")

        # Calculate component magnitudes
        components = {
            'Intercept': np.abs(results['weight_intercept'] * weight_std),
            'Strength Effect': np.abs(results['gamma_s'] * np.mean(results['strength_fitness']) * weight_std),
            'Aerobic Effect': np.abs(results['gamma_a'] * np.mean(results['aerobic_fitness']) * weight_std),
            'Daily Spline': np.abs(np.mean(results['f_daily']) * weight_std),
            'AR(1) Innovation': np.abs(np.mean(results['epsilon']) * weight_std)
        }

        for name, value in components.items():
            f.write(f"| {name} | {value:.3f} | - |\n")

        f.write("\n## Generated Plots\n\n")
        f.write("1. **Fitness States Over Time** (`fitness_states_over_time.png`)\n")
        f.write("   - Strength and aerobic fitness evolution\n")
        f.write("   - Training intensity inputs\n")
        f.write("   - Daily scale visualization\n\n")

        f.write("2. **Weight Effect Decomposition** (`weight_effect_decomposition.png`)\n")
        f.write("   - Strength effect on weight over time\n")
        f.write("   - Aerobic effect on weight over time\n")
        f.write("   - Total fitness effect\n")
        f.write("   - Daily scale visualization\n\n")

        f.write("3. **Daily Spline Component** (`daily_spline_decomposition.png`)\n")
        f.write("   - Spline vs hour of day\n")
        f.write("   - Spline by day of week\n")
        f.write("   - Fourier basis functions\n")
        f.write("   - Intraday scale visualization\n\n")

        f.write("4. **AR(1) Process Decomposition** (`ar_process_decomposition.png`)\n")
        f.write("   - AR(1) innovations over time\n")
        f.write("   - Autocorrelation function\n")
        f.write("   - Innovation distribution\n")
        f.write("   - Observation scale visualization\n\n")

        f.write("5. **Student-t Distribution Analysis** (`student_t_analysis.png`)\n")
        f.write("   - Residual distribution with Student-t fit\n")
        f.write("   - QQ-plot vs Student-t distribution\n")
        f.write("   - Degrees of freedom analysis\n")
        f.write("   - Observation scale visualization\n\n")

        f.write("6. **Complete Model Decomposition** (`complete_decomposition.png`)\n")
        f.write("   - Stacked component plot\n")
        f.write("   - Component magnitude comparison\n")
        f.write("   - Component correlation matrix\n")
        f.write("   - Observation scale visualization\n\n")

        f.write("7. **Decay Rate Analysis** (`decay_rate_analysis.png`)\n")
        f.write("   - Strength decay rates over time\n")
        f.write("   - Aerobic decay rates over time\n")
        f.write("   - Decay rate distributions\n")
        f.write("   - Daily scale visualization\n\n")

        f.write("8. **Cumulative Prediction Build-Up** (`cumulative_prediction_buildup.png`)\n")
        f.write("   - Intercept only (baseline)\n")
        f.write("   - Intercept + Strength effect\n")
        f.write("   - Intercept + Aerobic effect\n")
        f.write("   - Intercept + Strength + Aerobic\n")
        f.write("   - Intercept + Daily spline\n")
        f.write("   - Intercept + Strength + Aerobic + Daily spline\n")
        f.write("   - Shows how each component builds toward full prediction\n")
        f.write("   - Observation scale visualization\n\n")

        f.write("9. **Component Contribution Analysis** (`component_contribution_analysis.png`)\n")
        f.write("   - Prediction error reduction by component\n")
        f.write("   - Cumulative error reduction relative to intercept\n")
        f.write("   - Variance explained by each component\n")
        f.write("   - Prediction error distributions\n")
        f.write("   - Shows quantitative impact of each model piece\n")
        f.write("   - Observation scale visualization\n\n")

    print(f"Summary report saved to: {report_path}")


def main():
    """Main function to run the decomposition analysis."""
    print("="*80)
    print("STUDENT-T AR SPLINE MODEL EFFECT DECOMPOSITION")
    print("="*80)

    # Fit model and extract data
    fit, df_weight, df_daily, weight_mean, weight_std, date_range, K = fit_model()

    # Extract decomposition components
    results = extract_decomposition(fit, df_weight, df_daily, weight_mean, weight_std, K)

    # Create comprehensive plots
    output_dir = create_comprehensive_decomposition_plots(
        results, df_weight, df_daily, weight_mean, weight_std, date_range, K
    )

    # Create summary report
    create_summary_report(results, output_dir, weight_std)

    print("\n" + "="*80)
    print("DECOMPOSITION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print(f"Summary report: {output_dir}/decomposition_summary.md")


if __name__ == "__main__":
    main()